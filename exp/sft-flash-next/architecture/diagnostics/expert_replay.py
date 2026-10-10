"""Test-only same-input expert shadow forward and fixed-cotangent VJPs.

Stop at the first native/chunk output mismatch. Activations and gradients remain
in RAM; only comparisons, seeded cotangent identity and phase counters are saved.
No full-model backward, optimizer, package mutation or training pin is introduced.
If every expert output matches, the diagnostic finishes the forward loss only.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from peft import get_peft_model_state_dict
from config import load_config
from model import build
from components.ple import EncodedPaths,prepared_loader
from components.expert_chunks import expert_chunks
from runtime.evidence import compare,compare_initial,model_identity,sha,snapshot,snapshot_imports,write
from runtime.loop import prepare
from runtime.numeric_profile import verify_profile
from runtime.resources import Resources


class FirstMismatch(Exception):pass


def vjp_comparison(candidate,reference):
    # Reuse the all-tensor metric checks without reporting a fictitious loss.
    report=compare(candidate,1.,dict(loss=1.,gradients=reference))
    for key in ['loss','baseline_loss','loss_bitwise_equal','loss_relative_change']:report.pop(key)
    report['criterion']='Bitwise all adapter gradients with fixed upstream cotangent'
    return report


def metrics(candidate,reference):
    # CPU doubles avoid introducing another GPU kernel into the comparison.
    a=reference.detach().cpu();b=candidate.detach().cpu();x=a.double();y=b.double()
    norm=float(x.square().sum());error=float((x-y).square().sum())
    return dict(bitwise_equal=a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),
        relative_l2=(error/norm)**.5 if norm else None,
        max_absolute_difference=float((x-y).abs().max()),different_values=int(torch.count_nonzero(a!=b)),
        finite=bool(torch.isfinite(b).all()),shape=list(a.shape),dtype=str(a.dtype))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=load_config(args.config,'test');out=Path(config.output);out.mkdir(parents=True,exist_ok=False)
    assert config.optimizations.expert_chunking
    torch.manual_seed(config.seed);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot(out,config,'diagnostic-expert');resources=Resources(out);verify_profile(config,out)
    actual=dict(sample=sha(config.samples[0]),**model_identity(config.model))
    assert all(actual.get(k)==v for k,v in config.expected_sha256.items())
    started=time.monotonic()
    def event(name,**values):
        row=dict(event=name,elapsed_seconds=time.monotonic()-started,**values)
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    iterator=iter(prepared_loader(EncodedPaths(config.samples),config.model,event_path=str(out/'ple-events.jsonl')))
    architecture=None;handles=[];saved={};shadows=[]
    try:
        event('load_start')
        with resources.phase('loading'):architecture=build(config,out)
        model=architecture.model;manager=architecture.expert_stager
        initial={n:p.detach().cpu() for n,p in get_peft_model_state_dict(model).items()}
        with resources.phase('initialization_verification'):identity=compare_initial(initial,config.baseline)
        write(out/'initial-comparison.json',identity);assert identity['passed'];del initial
        current=next(iterator);architecture.activate(current);batch,labels,_=prepare(current['batch'],config,'test')
        def shadow(module,inputs,kwargs,output):
            if kwargs.get('_prefetched'):return  # Native shadow invokes this same module.
            assert len(inputs)==3
            x,indices,weights=inputs
            native=manager.call(module,x,indices,weights)
            comparison=metrics(output,native)
            counts=torch.bincount(indices.flatten(),minlength=module.num_experts).tolist()
            row=dict(layer=module.expert_layer,output=comparison,
                split_experts={str(i):c for i,c in enumerate(counts) if c>module.chunk_tokens},
                input_shape=list(x.shape),input_stride=list(x.stride()),input_dtype=str(x.dtype),
                routing_dtype=str(weights.dtype),chunk_tokens=module.chunk_tokens)
            shadows.append(row);write(out/'expert-shadows.json',shadows)
            event('expert_shadow',layer=module.expert_layer,**comparison)
            if not comparison['bitwise_equal'] or (row['split_experts'] and not saved):
                saved.update(module=module,x=x.detach().cpu().clone(),indices=indices.detach().cpu().clone(),
                    weights=weights.detach().cpu().clone(),counts=counts)
            if not comparison['bitwise_equal']:
                raise FirstMismatch()
        for module in manager.modules:handles.append(module.register_forward_hook(shadow,with_kwargs=True))
        event('shadow_forward_start')
        try:
            with resources.phase('shadow_forward'),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=architecture.loss(batch,labels)
        except FirstMismatch:pass
        else:
            event('all_expert_outputs_bitwise',loss=float(loss));del loss
        for handle in handles:handle.remove()
        handles.clear();del batch,labels,current
        for layer in list(manager.cache):manager.release(layer)
        gc.collect();torch.cuda.empty_cache()
        if not saved:
            write(out/'result.json',dict(completed=True,mode='diagnostic-expert',
                all_expert_outputs_bitwise=True,layers_checked=len(shadows),optimizer_updates=0,
                raw_gradients_retained=False,note='No expert mismatch; investigate shared head/kernel repeatability next.'))
            return
        module=saved['module'];named={n:p for n,p in module.named_parameters() if p.requires_grad}
        cotangent_seed=config.seed+module.expert_layer
        generator=torch.Generator(device='cpu').manual_seed(cotangent_seed)
        cotangent=(torch.randn(saved['x'].shape,generator=generator,dtype=torch.float32)*.001).bfloat16()
        write(out/'cotangent.json',dict(seed=cotangent_seed,generator='CPU torch.Generator',
            operation='CPU FP32 randn * .001 then BF16',shape=list(cotangent.shape),
            raw_values_retained=False))
        baseline=None;results=[];size=module.chunk_tokens
        try:
            for variant in ['native','native-repeat','chunked','custom-unsplit']:
                for layer in list(manager.cache):manager.release(layer)
                x=saved['x'].to('cuda').requires_grad_(True)
                weights=saved['weights'].to('cuda').requires_grad_(True)
                indices=saved['indices'].to('cuda')
                module.chunk_tokens=saved['x'].shape[0]*indices.shape[1] if variant=='custom-unsplit' else size
                with resources.phase(f'expert-vjp-{variant}'):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        value=manager.call(module,x,indices,weights) if variant.startswith('native') else expert_chunks(module,x,indices,weights)
                    output=value.detach().cpu()
                    grads=torch.autograd.grad(value,(x,weights,*named.values()),cotangent.to('cuda'),allow_unused=True)
                dx,dweights,*adapters=grads
                adapter_values={n:g.detach().cpu() if g is not None else torch.zeros_like(p,device='cpu')
                    for (n,p),g in zip(named.items(),adapters)}
                state=dict(output=output,dx=dx.detach().cpu(),dweights=dweights.detach().cpu(),adapters=adapter_values)
                if baseline is None:baseline=state
                adapter_comparison=vjp_comparison(state['adapters'],baseline['adapters'])
                partitions={}
                for partition in ['split','unsplit']:
                    names=[n for n in named if (saved['counts'][int(n.split('.')[0])]>size)==(partition=='split')]
                    if names:
                        partitions[partition]=vjp_comparison({n:state['adapters'][n] for n in names},
                            {n:baseline['adapters'][n] for n in names})
                row=dict(variant=variant,output=metrics(state['output'],baseline['output']),
                    input_gradient=metrics(state['dx'],baseline['dx']),routing_gradient=metrics(state['dweights'],baseline['dweights']),
                    adapter_gradients=adapter_comparison,adapter_partitions=partitions)
                if variant=='native-repeat':
                    assert row['output']['bitwise_equal'] and row['input_gradient']['bitwise_equal']
                    assert row['routing_gradient']['bitwise_equal'] and adapter_comparison['passed']
                results.append(row);write(out/'expert-vjps.json',results)
                event('expert_vjp',variant=variant,adapter_l2=adapter_comparison['global_relative_l2'],
                    output_exact=row['output']['bitwise_equal'],input_gradient_exact=row['input_gradient']['bitwise_equal'],
                    routing_gradient_exact=row['routing_gradient']['bitwise_equal'])
                del grads,dx,dweights,adapters,adapter_values,state,output,value,x,weights,indices
                gc.collect();torch.cuda.empty_cache()
        finally:module.chunk_tokens=size
        first_mismatch=next((r['layer'] for r in shadows if not r['output']['bitwise_equal']),None)
        write(out/'result.json',dict(completed=True,mode='diagnostic-expert',first_mismatching_layer=first_mismatch,
            focused_layer=module.expert_layer,all_expert_outputs_bitwise=first_mismatch is None,
            layers_checked=len(shadows),variants=len(results),optimizer_updates=0,raw_gradients_retained=False,
            note='Same-input expert outputs and fixed random-cotangent VJPs only; no full loss/gradient qualification.'))
    finally:
        for handle in handles:handle.remove()
        write(out/'resources.json',resources.rows);snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close();write(out/'expert-prefetch.json',architecture.expert_stager.report())
        iterator._shutdown_workers()


if __name__=='__main__':main()
