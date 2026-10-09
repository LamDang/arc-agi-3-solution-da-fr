"""Run isolated and cumulative gradient checks on the qualified HF+PEFT model.

No optimizer updates. The initial state and input are supplied as exact files.
Baseline replay uses the unchanged native objective; substitutions live in a
separate module. A flag must pass ALL adapter tensors before promotion. Strict
numerical tolerance and bitwise identity are reported separately.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time
import traceback

import torch
from transformers import AutoModelForImageTextToText, AutoRoundConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from overfit_hf_reference import TARGETS, resident_device_map, sha256
from native_gradient_compare import compare, tensor_digest
from native_optimization_flags import apply_flags, objective

_failure_out = None


def json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def default_cases():
    return [dict(name='reference_repeat'),
            dict(name='selected_logits', loss='selected'),
            dict(name='chunked_loss_128', loss='chunked', loss_block=128),
            dict(name='selective_cpu_offload', offload='cpu'),
            dict(name='disk_offload', offload='disk'),
            dict(name='native_rms_blocks', native_norm_block=1024),
            dict(name='native_gated_blocks', native_gated_block=262144),
            dict(name='native_hyper_blocks', native_hyper_block=1024),
            dict(name='rms_blocks', norm_block=1024),
            dict(name='hyper_blocks', hyper_block=1024),
            dict(name='gated_norm_blocks', gated_norm_block=262144),
            dict(name='ple_windows', ple_block=8192),
            dict(name='gdn_blocks', gdn_block_tokens=8192),
            dict(name='sparse_attention', attention='triton')]


def main():
    global _failure_out
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--sample',required=True)
    p.add_argument('--prompt-tokens',required=True,type=int)
    p.add_argument('--adapter-state',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);p.add_argument('--cases')
    p.add_argument('--rtol',type=float,default=1e-5);p.add_argument('--atol',type=float,default=1e-8)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    _failure_out = out
    start=time.monotonic()
    def event(kind,**data):
        row=dict(event=kind,seconds=time.monotonic()-start,**data)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False,default=json_default)+'\n')
    torch.manual_seed(20261009);torch.set_num_threads(8)
    reference_path=Path(args.reference)
    reference=torch.load(reference_path/'gradients.pt',map_location='cpu',weights_only=True)
    reference_result=json.loads((reference_path/'result.json').read_text())
    reference_provenance=json.loads((reference_path/'provenance.json').read_text())
    assert reference_provenance['sample_sha256']==sha256(args.sample)
    assert reference_provenance['config_sha256']==sha256(Path(args.model)/'config.json')
    assert reference_provenance['arguments']['prompt_tokens']==args.prompt_tokens
    cases=json.loads(Path(args.cases).read_text()) if args.cases else default_cases()
    write(out/'cases.json',cases)
    write(out/'identity.json',dict(arguments=vars(args),sample_sha256=sha256(args.sample),
        adapter_sha256=sha256(args.adapter_state),reference_gradients_sha256=sha256(reference_path/'gradients.pt'),
        source_sha256={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
        optimizer_updates=0,clipping=False,baseline_memory_controls=['HF checkpointing','torch save_on_cpu']))
    batch=torch.load(args.sample,map_location='cpu',weights_only=True)
    event('load_start')
    base,loading=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map=resident_device_map(args.model),local_files_only=True,trust_remote_code=False,
        output_loading_info=True,quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
    write(out/'loading.json',loading)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    assert base.config.text_config.num_experts==256
    for m in base.modules():
        for n,v in m._buffers.items():
            if v is not None and v.device.type!='cuda':m._buffers[n]=v.to('cuda')
    for n,v in base.named_parameters():
        assert str(v.device)==('cpu' if 'ple_embedding.ngram_embedding.weight' in n else 'cuda:0')
    base.requires_grad_(False)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',target_modules=TARGETS,task_type='CAUSAL_LM'))
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    state=torch.load(args.adapter_state,map_location='cpu',weights_only=True)
    assert state.keys()==get_peft_model_state_dict(model).keys()
    set_peft_model_state_dict(model,state)
    for n,v in get_peft_model_state_dict(model).items():torch.testing.assert_close(v.cpu(),state[n],rtol=0,atol=0)
    saved_reference_state=torch.load(reference_path/'initial-adapter.pt',map_location='cpu',weights_only=True)
    assert state.keys()==saved_reference_state.keys()
    for n,v in state.items():torch.testing.assert_close(v,saved_reference_state[n],rtol=0,atol=0)
    parameters={n:v for n,v in model.named_parameters() if v.requires_grad}
    assert parameters.keys()==reference.keys()
    batch={k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
    labels=batch['input_ids'].clone();labels[:,:args.prompt_tokens]=-100
    cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state_all()
    phase={'name':None,'pass':'forward'};routes={};handles=[]
    for index,layer in enumerate(base.model.language_model.layers):
        def route_hook(module,inputs,output,index=index):
            if phase['pass']=='forward':routes[index]=tensor_digest(output[2])
        handles.append(layer.mlp.gate.register_forward_hook(route_hook))
        def progress(module,inputs,output,index=index):
            if index%12==11:event('layer',layer=index,**phase)
        handles.append(layer.register_forward_hook(progress))
    event('load_complete',adapter_tensors=len(parameters))
    reports=[];baseline_routes=None
    def run(flags):
        nonlocal baseline_routes
        name=flags['name'];case=out/name;case.mkdir()
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        routes.clear();phase.update(name=name,**{'pass':'forward'})
        started=time.monotonic();event('case_start',flags=flags)
        storage=None
        try:
            if flags.get('offload'):
                from offload import ActivationOffload
                storage=ActivationOffload(model,disk_dir='/tmp/native-gradient-activations' if flags['offload']=='disk' else None,
                    disk_budget_gib=32 if flags['offload']=='disk' else 0,prefetch=2)
            else:storage=torch.autograd.graph.save_on_cpu(pin_memory=False)
            with apply_flags(base,flags),storage:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,batch,labels,args.prompt_tokens,flags)
                value=loss.item();event('forward_complete',name=name,loss=value)
                phase['pass']='backward';loss.backward();del loss
            torch.cuda.synchronize()
            gradients={n:v.grad.detach().cpu().clone() for n,v in parameters.items() if v.grad is not None}
            metric=compare(gradients,reference,rtol=args.rtol,atol=args.atol)
            torch.save(gradients,case/'gradients.pt');write(case/'comparison.json',metric)
            write(case/'routes.json',routes)
            if name=='reference_repeat':baseline_routes=dict(routes)
            route_match=baseline_routes is not None and routes==baseline_routes
            loss_match=abs(value-reference_result['measured_loss']) <= args.atol+args.rtol*abs(reference_result['measured_loss'])
            result=dict(name=name,flags=flags,completed=True,loss=value,
                passed=metric['within_tolerance'] and route_match and loss_match,
                bitwise_equal=metric['bitwise_equal'],relative_l2=metric['relative_l2'],
                failed_tensors=metric['failed_tensors'],routes_equal=route_match,loss_matches=loss_match,
                elapsed_seconds=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                offload_stats=getattr(storage,'stats',None))
        except Exception as exc:
            result=dict(name=name,flags=flags,completed=False,passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        write(case/'result.json',result);reports.append(result);write(out/'results.json',reports);event('case_complete',**result)
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        return result
    passed_flags={}
    for flags in cases:
        result=run(flags)
        if flags['name']=='reference_repeat' and not result['passed']:
            event('stopped',reason='Native replay is not stable at the declared tolerance; no flag can be certified.')
            return
        if result['passed'] and flags['name']!='reference_repeat':
            candidate={**passed_flags,**{k:v for k,v in flags.items() if k!='name'}}
            combined=run(dict(name='cumulative_'+flags['name'],**candidate))
            if combined['passed']:passed_flags=candidate
    write(out/'accepted-flags.json',passed_flags)
    event('finished',accepted_flags=passed_flags,full_context_validated=False)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if _failure_out is not None:
            failure=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            (_failure_out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            with (_failure_out/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event='failed',**failure))+'\n')
        raise
