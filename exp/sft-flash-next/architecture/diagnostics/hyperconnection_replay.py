"""Test-only actual-input hyperconnection shadows and fixed-cotangent VJPs.

No model backward, optimizer, or raw activation/gradient archive. The first
different mixer is replayed with every output cotangent nonzero, including the
direct residual and injection coefficients. Projection traces use actual shapes.
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
from components.hyperconnection_chunks import ChunkedResidual
from runtime.evidence import compare_initial,model_identity,sha,snapshot,snapshot_imports,write
from runtime.loop import prepare
from runtime.numeric_profile import verify_profile
from runtime.resources import Resources


class FirstMismatch(Exception):pass


def ports(value):return value if isinstance(value,tuple) else (value,)


def metrics(candidate,reference):
    a=reference.detach().cpu().contiguous();b=candidate.detach().cpu().contiguous()
    assert a.shape==b.shape
    norm=error=maximum=0.;different=0;finite=True
    # Bounded CPU-double scratch; a real residual input has 166 million values.
    for x,y in zip(a.reshape(-1).split(1048576),b.reshape(-1).split(1048576)):
        xd,yd=x.double(),y.double();delta=xd-yd
        norm+=float(xd.square().sum());error+=float(delta.square().sum())
        maximum=max(maximum,float(delta.abs().max()));different+=int(torch.count_nonzero(x!=y))
        finite=finite and bool(torch.isfinite(y).all())
    return dict(bitwise_equal=a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
        relative_l2=(error/norm)**.5 if norm else None,max_absolute_difference=maximum,
        different_values=different,finite=finite,shape=list(a.shape),reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype))


def without_checkpoint(module,x,size):
    x=x.bfloat16()
    assert module.chunk_tokens==size
    value=module._shape_preserving_mix(x,checkpoint_windows=False)
    return (value[0],x,value[1]) if isinstance(value,tuple) else value


def main(stop_after_first=False):
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=load_config(args.config,'test');out=Path(config.output);out.mkdir(parents=True,exist_ok=False)
    assert config.optimizations.hyperconnection_chunking
    torch.manual_seed(config.seed);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot(out,config,'diagnostic-hyperconnection');resources=Resources(out);verify_profile(config,out)
    actual=dict(sample=sha(config.samples[0]),**model_identity(config.model))
    assert all(actual.get(k)==v for k,v in config.expected_sha256.items())
    started=time.monotonic()
    def event(name,**values):
        row=dict(event=name,elapsed_seconds=time.monotonic()-started,**values)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    iterator=iter(prepared_loader(EncodedPaths(config.samples),config.model,event_path=str(out/'ple-events.jsonl')))
    architecture=None;handles=[];saved={};shadows=[];guard=False
    try:
        event('load_start')
        with resources.phase('loading'):architecture=build(config,out)
        model=architecture.model;manager=architecture.expert_stager
        initial={n:p.detach().cpu() for n,p in get_peft_model_state_dict(model).items()}
        with resources.phase('initialization_verification'):identity=compare_initial(initial,config.baseline)
        write(out/'initial-comparison.json',identity);assert identity['passed'];del initial
        current=next(iterator);architecture.activate(current);batch,labels,_=prepare(current['batch'],config,'test')
        def make_shadow(name):
            def shadow(module,inputs,output):
                nonlocal guard
                if guard:return
                x=inputs[0];size=module.chunk_tokens
                try:
                    guard=True;module.chunk_tokens=0;native=module(x)
                finally:module.chunk_tokens=size;guard=False
                assert len(ports(output))==len(ports(native))
                comparisons=[metrics(a,b) for a,b in zip(ports(output),ports(native))]
                row=dict(module=name,ports=comparisons,input_shape=list(x.shape),input_stride=list(x.stride()),
                    input_dtype=str(x.dtype),chunk_tokens=size)
                shadows.append(row);write(out/'hyperconnection-shadows.json',shadows)
                exact=all(value['bitwise_equal'] for value in comparisons)
                event('hyperconnection_shadow',module=name,all_ports_bitwise=exact)
                # Keep an actual mixer even when every forward is exact, to test VJPs.
                if not saved or not exact:saved.update(module=module,name=name,x=x.detach().cpu().clone())
                if not exact or stop_after_first:raise FirstMismatch()
            return shadow
        for name,module in model.named_modules():
            if isinstance(module,ChunkedResidual):handles.append(module.register_forward_hook(make_shadow(name)))
        event('shadow_forward_start')
        try:
            with resources.phase('shadow_forward'),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=architecture.loss(batch,labels)
        except FirstMismatch:pass
        else:event('all_hyperconnection_outputs_bitwise',loss=float(loss));del loss
        for handle in handles:handle.remove()
        handles.clear();del batch,labels,current
        for layer in list(manager.cache):manager.release(layer)
        gc.collect();torch.cuda.empty_cache()
        assert saved,'No hyperconnection module was visited'
        module=saved['module'];size=module.chunk_tokens
        assert not any(p.requires_grad for p in module.parameters()),'Expected frozen mixer weights'
        cotangents=None;baseline=None;results=[];baseline_trace=None
        try:
            for variant in ['native','native-repeat','chunk-no-checkpoint','chunked','custom-unsplit','native-outer-offload','chunk-outer-offload']:
                x=saved['x'].to('cuda').requires_grad_(True);traces={};trace_handles=[]
                for name in ['hc_norm','input_mix_weight_down','input_mix_weight_up','block_inject_weight']:
                    component=getattr(module,name,None)
                    if component is not None:
                        def trace_hook(_module,_inputs,value,name=name):
                            traces.setdefault(name,[]).append(value.detach().cpu().clone())
                        trace_handles.append(component.register_forward_hook(trace_hook))
                native=variant.startswith('native')
                module.chunk_tokens=0 if native else (x.shape[1]+1 if variant=='custom-unsplit' else size)
                def compute(value):
                    return without_checkpoint(module,value,size) if variant=='chunk-no-checkpoint' else module(value)
                with resources.phase('hyperconnection-vjp-'+variant):
                    context=torch.autograd.graph.save_on_cpu(pin_memory=True) if variant.endswith('outer-offload') else nullcontext()
                    # Only outer variants offload; normal variants use their native saved tensors.
                    with context,torch.autocast('cuda',dtype=torch.bfloat16):
                        value=checkpoint(compute,x,use_reentrant=False) if variant.endswith('outer-offload') else compute(x)
                    # Do not collect replay traces twice during checkpoint backward.
                    for handle in trace_handles:handle.remove()
                    trace_handles.clear()
                    output=tuple(v.detach().cpu().clone() for v in ports(value))
                    if cotangents is None:
                        cotangents=[];metadata=[]
                        for index,v in enumerate(output):
                            seed=config.seed+10000+index;generator=torch.Generator(device='cpu').manual_seed(seed)
                            c=(torch.randn(v.shape,generator=generator,dtype=torch.float32)*.001).to(v.dtype)
                            assert torch.count_nonzero(c)>0
                            cotangents.append(c);metadata.append(dict(port=index,seed=seed,shape=list(c.shape),dtype=str(c.dtype),
                                sha256=hashlib.sha256(c.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
                        write(out/'cotangent.json',dict(generator='CPU FP32 randn * .001 then output dtype',ports=metadata,raw_values_retained=False))
                    dx,=torch.autograd.grad(ports(value),x,tuple(c.to('cuda') for c in cotangents))
                state=dict(output=output,dx=dx.detach().cpu().clone())
                trace_values={name:torch.cat(values,dim=1) for name,values in traces.items()}
                if baseline is None:baseline=state;baseline_trace=trace_values
                row=dict(variant=variant,input_stride=list(x.stride()),
                    outputs=[metrics(a,b) for a,b in zip(state['output'],baseline['output'])],
                    input_gradient=metrics(state['dx'],baseline['dx']),
                    projections={name:metrics(v,baseline_trace[name]) for name,v in trace_values.items()})
                if variant=='native-repeat':assert all(r['bitwise_equal'] for r in row['outputs']) and row['input_gradient']['bitwise_equal']
                results.append(row);write(out/'hyperconnection-vjps.json',results)
                event('hyperconnection_vjp',variant=variant,input_gradient_l2=row['input_gradient']['relative_l2'])
                del value,dx,x,state,output,traces,trace_values
                gc.collect();torch.cuda.empty_cache()
        finally:module.chunk_tokens=size
        mismatch=next((row['module'] for row in shadows if not all(v['bitwise_equal'] for v in row['ports'])),None)
        write(out/'result.json',dict(completed=True,mode='diagnostic-hyperconnection',first_mismatching_module=mismatch,
            focused_module=saved['name'],modules_checked=len(shadows),variants=len(results),
            shadow_scope='first-module' if stop_after_first else 'until-first-mismatch-or-complete',
            optimizer_updates=0,raw_gradients_retained=False,note='Actual-input mixer forward and fixed-cotangent VJPs only; no full-model qualification.'))
    finally:
        for handle in handles:handle.remove()
        write(out/'resources.json',resources.rows);snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close();write(out/'expert-prefetch.json',architecture.expert_stager.report())
        iterator._shutdown_workers()


if __name__=='__main__':main()
