"""Test-only actual PLE input replay, projection shadows and halo input VJPs.

Stop after the first PLE call. Captures/cotangents/gradients stay in RAM; only
comparisons, identities and phase resource measurements are saved.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.checkpoint import checkpoint
from peft import get_peft_model_state_dict
from config import load_config
from model import build
from components.ple import EncodedPaths,prepared_loader
from components.ple_chunks import ChunkedPLE
from diagnostics.hyperconnection_replay import metrics
from runtime.evidence import compare_initial,model_identity,sha,snapshot,snapshot_imports,write
from runtime.loop import prepare
from runtime.numeric_profile import verify_profile
from runtime.resources import Resources


class Captured(Exception):pass


def plain_windows(module,x,mask):
    halo=module.short_conv_state_len;values=[]
    for start in range(0,x.shape[1],module.chunk_tokens):
        left=max(0,start-halo);stop=min(start+module.chunk_tokens,x.shape[1])
        values.append(module._window(x[:,left:stop],module.ple_embedding.prepared_payload[:,left:stop],
            None if mask is None else mask[:,left:stop],discard=start-left))
    return torch.cat(values,dim=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=load_config(args.config,'test');out=Path(config.output);out.mkdir(parents=True,exist_ok=False)
    assert config.optimizations.ple_chunking
    torch.manual_seed(config.seed);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot(out,config,'diagnostic-ple');resources=Resources(out);verify_profile(config,out)
    actual=dict(sample=sha(config.samples[0]),**model_identity(config.model))
    assert all(actual.get(k)==v for k,v in config.expected_sha256.items())
    started=time.monotonic()
    def event(name,**values):
        row=dict(event=name,elapsed_seconds=time.monotonic()-started,**values)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    iterator=iter(prepared_loader(EncodedPaths(config.samples),config.model,event_path=str(out/'ple-events.jsonl')))
    architecture=None;handles=[];saved={};guard=False
    try:
        event('load_start')
        with resources.phase('loading'):architecture=build(config,out)
        model=architecture.model;manager=architecture.expert_stager
        initial={n:p.detach().cpu() for n,p in get_peft_model_state_dict(model).items()}
        with resources.phase('initialization_verification'):identity=compare_initial(initial,config.baseline)
        write(out/'initial-comparison.json',identity);assert identity['passed'];del initial
        current=next(iterator);architecture.activate(current);batch,labels,_=prepare(current['batch'],config,'test')
        def make_shadow(name):
            def shadow(module,inputs,kwargs,output):
                nonlocal guard
                if guard:return
                assert len(inputs)==3 and inputs[2] is None
                x,ids,_=inputs;size=module.chunk_tokens;mask=kwargs.get('conv_mask')
                try:
                    guard=True;module.chunk_tokens=0;native=module(x,ids,None,conv_mask=mask)
                finally:module.chunk_tokens=size;guard=False
                row=dict(module=name,output=metrics(output,native),input_shape=list(x.shape),input_stride=list(x.stride()),
                    input_dtype=str(x.dtype),chunk_tokens=size,halo_tokens=module.short_conv_state_len)
                write(out/'ple-shadow.json',row)
                saved.update(module=module,name=name,x=x.detach().cpu().clone(),ids=ids.detach().cpu().clone(),
                    mask=None if mask is None else mask.detach().cpu().clone())
                event('ple_shadow',module=name,**row['output']);raise Captured()
            return shadow
        for name,module in model.named_modules():
            if isinstance(module,ChunkedPLE):handles.append(module.register_forward_hook(make_shadow(name),with_kwargs=True))
        event('shadow_forward_start')
        try:
            with resources.phase('shadow_forward'),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                architecture.loss(batch,labels)
        except Captured:pass
        for handle in handles:handle.remove()
        handles.clear();assert saved,'No PLE module visited'
        del batch,labels,current
        for layer in list(manager.cache):manager.release(layer)
        gc.collect();torch.cuda.empty_cache()
        module=saved['module'];size=module.chunk_tokens
        assert not any(p.requires_grad for p in module.parameters())
        baseline=None;baseline_trace=None;cotangent=None;results=[]
        try:
            for variant in ['native','native-repeat','chunk-no-checkpoint','chunked','custom-unsplit','native-outer-offload','chunk-outer-offload']:
                x=saved['x'].to('cuda').requires_grad_(True);ids=saved['ids'].to('cuda')
                mask=None if saved['mask'] is None else saved['mask'].to('cuda');traces={};trace_handles=[]
                for name in ['key_proj','value_proj','norm_key','norm_query','norm_conv']:
                    def trace_hook(_module,_inputs,value,name=name):traces.setdefault(name,[]).append(value.detach().cpu().clone())
                    trace_handles.append(getattr(module,name).register_forward_hook(trace_hook))
                module.chunk_tokens=0 if variant.startswith('native') else (x.shape[1]+1 if variant=='custom-unsplit' else size)
                def compute(value):
                    return plain_windows(module,value.bfloat16(),mask) if variant=='chunk-no-checkpoint' else module(value,ids,None,conv_mask=mask)
                with resources.phase('ple-vjp-'+variant):
                    context=torch.autograd.graph.save_on_cpu(pin_memory=True) if variant.endswith('outer-offload') else nullcontext()
                    with context,torch.autocast('cuda',dtype=torch.bfloat16):
                        value=checkpoint(compute,x,use_reentrant=False) if variant.endswith('outer-offload') else compute(x)
                    for handle in trace_handles:handle.remove()
                    trace_handles.clear();output=value.detach().cpu().clone()
                    if cotangent is None:
                        seed=config.seed+20000;generator=torch.Generator(device='cpu').manual_seed(seed)
                        cotangent=(torch.randn(output.shape,generator=generator,dtype=torch.float32)*.001).to(output.dtype)
                        write(out/'cotangent.json',dict(seed=seed,shape=list(cotangent.shape),dtype=str(cotangent.dtype),
                            generator='CPU FP32 randn * .001 then output dtype',raw_values_retained=False,
                            sha256=hashlib.sha256(cotangent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                    dx,=torch.autograd.grad(value,x,cotangent.to('cuda'))
                state=dict(output=output,dx=dx.detach().cpu().clone())
                # Window traces include their left halo; align each retained
                # output position before comparing with the full native trace.
                trace_values={name:torch.cat([v[:,0 if i==0 else min(i*module.chunk_tokens,module.short_conv_state_len):]
                    for i,v in enumerate(values)],dim=1) for name,values in traces.items()}
                if baseline is None:baseline=state;baseline_trace=trace_values
                row=dict(variant=variant,output=metrics(output,baseline['output']),input_stride=list(x.stride()),
                    input_gradient=metrics(state['dx'],baseline['dx']),
                    projections={name:metrics(v,baseline_trace[name]) for name,v in trace_values.items()})
                if variant=='native-repeat':assert row['output']['bitwise_equal'] and row['input_gradient']['bitwise_equal']
                results.append(row);write(out/'ple-vjps.json',results)
                event('ple_vjp',variant=variant,output_l2=row['output']['relative_l2'],input_gradient_l2=row['input_gradient']['relative_l2'])
                del value,dx,x,ids,mask,state,output,traces,trace_values
                gc.collect();torch.cuda.empty_cache()
        finally:module.chunk_tokens=size
        write(out/'result.json',dict(completed=True,mode='diagnostic-ple',focused_module=saved['name'],variants=len(results),
            optimizer_updates=0,raw_gradients_retained=False,note='Actual PLE output/projection shadows and input VJPs only; no full-model qualification.'))
    finally:
        for handle in handles:handle.remove()
        write(out/'resources.json',resources.rows);snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close();write(out/'expert-prefetch.json',architecture.expert_stager.report())
        iterator._shutdown_workers()


if __name__=='__main__':main()
