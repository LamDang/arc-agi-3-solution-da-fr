"""Test-only layer46 GDN VJP isolation after a native anchor forward.

Stop full backward at this GDN output; replay only that block with identical
inputs/cotangent. Temporarily force one existing FLA reverse-scan configuration
at a time. No package files, production component classes or raw gradient
archives are changed. Captured activations/cotangent live in RAM only.
"""
import argparse
from contextlib import nullcontext
import gc
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils._pytree import tree_map
from config import load_config
from model import build
from peft import get_peft_model_state_dict
from components.ple import EncodedPaths,prepared_loader
from runtime.evidence import compare,compare_initial,model_identity,reference_tensors,sha,snapshot,snapshot_imports,write
from runtime.loop import prepare
from runtime.resources import Resources


class CapturedCotangent(Exception):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=load_config(args.config,'test');out=Path(config.output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config.seed);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    snapshot(out,config,'diagnostic-gdn');resources=Resources(out)
    actual=dict(sample=sha(config.samples[0]),**model_identity(config.model))
    assert all(actual.get(k)==v for k,v in config.expected_sha256.items())
    iterator=iter(prepared_loader(EncodedPaths(config.samples),config.model,event_path=str(out/'ple-events.jsonl')))
    architecture=None;handles=[];saved={}
    try:
        with resources.phase('loading'):architecture=build(config,out)
        model=architecture.model
        initial={n:p.detach().cpu() for n,p in get_peft_model_state_dict(model).items()}
        identity=compare_initial(initial,config.baseline);write(out/'initial-comparison.json',identity)
        assert identity['passed'];del initial
        current=next(iterator);architecture.activate(current);batch,labels,_=prepare(current['batch'],config,'test')
        block=model.get_base_model().model.language_model.layers[46].linear_attn
        cpu=lambda value:tree_map(lambda t:t.detach().cpu().clone() if isinstance(t,torch.Tensor) else t,value)
        cuda=lambda value:tree_map(lambda t:t.to('cuda') if isinstance(t,torch.Tensor) else t,value)
        def capture(module,args,kwargs,output):
            if saved:return
            assert isinstance(output,torch.Tensor) and output.requires_grad
            saved.update(args=cpu(args),kwargs=cpu(kwargs),output=cpu(output))
            def cotangent(gradient):
                saved['cotangent']=cpu(gradient)
                raise CapturedCotangent()
            handles.append(output.register_hook(cotangent))
        handles.append(block.register_forward_hook(capture,with_kwargs=True))
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            with resources.phase('forward'):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=architecture.loss(batch,labels)
            loss_value=float(loss.detach());print(json.dumps({'loss':loss_value}),flush=True)
            assert loss_value==config.baseline['loss']
            try:
                with resources.phase('partial_backward'):loss.backward()
            except CapturedCotangent:pass
            else:raise RuntimeError('GDN output cotangent was not captured')
        for handle in handles:handle.remove()
        handles.clear();del loss,batch,labels,current
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        for layer in list(architecture.expert_stager.cache):architecture.expert_stager.release(layer)
        named={n:p for n,p in model.named_parameters() if p.requires_grad and '.layers.46.linear_attn.' in n}
        reference={n:g for n,g in reference_tensors(config.baseline['gradients']) if n in named}
        assert len(named)==len(reference)==10
        import fla.ops.utils.cumsum as cumsum
        kernel=cumsum.chunk_local_cumsum_scalar_kernel
        auto=kernel.fn;jit=auto.fn;native_run=auto.run
        configs={c.num_warps:c for c in auto.configs}
        assert set(configs)=={1,2,4,8}
        write(out/'kernel-identity.json',dict(wrapper=type(kernel).__name__,autotuner=type(auto).__name__,
            jit=type(jit).__name__,source=inspect.getfile(cumsum),source_sha256=sha(inspect.getfile(cumsum)),
            configs={w:config.all_kwargs() for w,config in configs.items()},
            process_local_test_only=True,package_files_modified=False))
        results=[];default_gradients=None
        for warp in [None,None,1,2,4,8]:
            invoked=[]
            def forced(*args,**kwargs):
                values=dict(zip(auto.arg_names,args));values.update(kwargs)
                if values.get('REVERSE'):
                    row=dict(warps=warp,B=values.get('B'),H=values.get('H'),BT=values.get('BT'))
                    invoked.append(row)
                    if warp is not None:return jit.run(*args,**{**kwargs,**configs[warp].all_kwargs()})
                    result=native_run(*args,**kwargs)
                    selected=getattr(auto,'best_config',None)
                    row['selected_config']=selected.all_kwargs() if selected is not None else None
                    row['cached_configs']=[dict(key=str(k),config=v.all_kwargs()) for k,v in auto.cache.items()]
                    return result
                return native_run(*args,**kwargs)
            auto.run=forced
            try:
                model.zero_grad(set_to_none=True)
                local_args=cuda(saved['args']);local_kwargs=cuda(saved['kwargs'])
                with resources.phase(f'gdn-vjp-{len(results)}'):
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=block(*local_args,**local_kwargs)
                    assert torch.equal(output.detach().cpu().view(torch.uint8),saved['output'].view(torch.uint8))
                    gradients=torch.autograd.grad(output,tuple(named.values()),cuda(saved['cotangent']))
                values={n:g.detach().cpu() for n,g in zip(named,gradients)}
                report=compare(values,loss_value,dict(loss=loss_value,gradients=reference))
                if default_gradients is None:default_gradients={n:g.clone() for n,g in values.items()}
                within=compare(values,loss_value,dict(loss=loss_value,gradients=default_gradients))
                if len(results)==1 and not within['passed']:
                    raise RuntimeError('Default GDN replay is not deterministic; forced-configuration attribution invalid')
                row=dict(reverse_scan_warps=warp,forward_bitwise_equal=True,reference=report,
                    default_replay_relative_l2=within['global_relative_l2'],default_replay_exact_tensors=within['bitwise_equal_tensors'],
                    reverse_scan_invocations=invoked)
                assert invoked
                results.append(row);write(out/'gdn-replays.json',results)
                print(json.dumps({'variant':len(results)-1,'warps':warp,'reference_l2':report['global_relative_l2'],
                    'reference_exact':report['bitwise_equal_tensors'],'default_l2':within['global_relative_l2']}),flush=True)
                del output,gradients,values,local_args,local_kwargs
            finally:auto.run=native_run
        write(out/'result.json',dict(completed=True,mode='diagnostic-gdn',loss=loss_value,
            variants=len(results),optimizer_updates=0,raw_gradients_retained=False,
            matching_warps=[r['reverse_scan_warps'] for r in results if r['reference']['passed']],
            note='Only layer46 GDN is replayed; native full-model gradient gate still required.'))
    finally:
        for handle in handles:handle.remove()
        write(out/'resources.json',resources.rows);snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close();write(out/'expert-prefetch.json',architecture.expert_stager.report())
        iterator._shutdown_workers()


if __name__=='__main__':main()
