"""Capacity-only native HF step, restricted to a gradient-qualified flag recipe.

The full request is never truncated. A short-reference gradient gate does not
prove gradient equality at this longer length. All diagnostic updates are
explicitly disposable, never production or fold-0 validation checkpoints.
"""
import argparse
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import time
import traceback
import torch
from transformers import AutoModelForImageTextToText, AutoRoundConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from overfit_hf_reference import TARGETS,resident_device_map,sha256
from native_optimization_flags import apply_flags,objective
from native_gradient_compare import qualified_flags



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--sample',required=True)
    p.add_argument('--prompt-tokens',type=int,required=True)
    p.add_argument('--adapter-state',required=True);p.add_argument('--qualification',required=True)
    p.add_argument('--out',required=True);p.add_argument('--optimizer-step',action='store_true')
    args=p.parse_args();flags,identity=qualified_flags(args.qualification,args.adapter_state)
    assert identity['arguments']['model']==args.model
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def event(kind,**values):
        row=dict(event=kind,elapsed_seconds=time.monotonic()-start,**values)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        torch.manual_seed(20261009);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
        with open(args.sample,'rb') as stream:compressed=stream.read(2)==b'\x1f\x8b'
        opener=gzip.open if compressed else open
        with opener(args.sample,'rb') as f:batch=torch.load(f,map_location='cpu',weights_only=True)
        assert all(isinstance(v,torch.Tensor) for v in batch.values())
        tokens=batch['input_ids'].shape[1]
        assert batch['input_ids'].shape[0]==1 and 0<args.prompt_tokens<tokens
        (out/'provenance.json').write_text(json.dumps(dict(arguments=vars(args),flags=flags,
            sample_sha256=sha256(args.sample),adapter_sha256=sha256(args.adapter_state),
            qualification_identity_sha256=sha256(Path(args.qualification)/'identity.json'),
            tokens=tokens,supervised_tokens=tokens-args.prompt_tokens,
            full_length_gradient_equivalence_proven=False,diagnostic_only=True),indent=2)+'\n')
        event('load_start',tokens=tokens)
        base,loading=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,
            device_map=resident_device_map(args.model),local_files_only=True,trust_remote_code=False,
            output_loading_info=True,quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
        assert base.config.text_config.num_experts==256
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
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
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        batch={k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
        labels=batch['input_ids'].clone();labels[:,:args.prompt_tokens]=-100
        optimizer=torch.optim.AdamW(parameters.values(),lr=2e-4,weight_decay=0.) if args.optimizer_step else None
        if flags.get('offload'):
            from offload import ActivationOffload
            storage=ActivationOffload(model,disk_dir='/tmp/native-capacity-activations' if flags['offload']=='disk' else None,
                disk_budget_gib=32 if flags['offload']=='disk' else 0,prefetch=2)
        else:storage=torch.autograd.graph.save_on_cpu(pin_memory=False)
        event('load_complete',allocated_gib=torch.cuda.memory_allocated()/2**30)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with apply_flags(base,flags),storage:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,batch,labels,args.prompt_tokens,flags)
            value=loss.item();assert torch.isfinite(loss)
            event('forward_complete',loss=value)
            loss.backward()
        for n,p in parameters.items():
            if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError('Invalid gradient: '+n)
        if optimizer is not None:
            norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        result=dict(completed=True,tokens=tokens,supervised_tokens=tokens-args.prompt_tokens,loss=value,
            step_seconds=time.monotonic()-started,optimizer_updates=int(args.optimizer_step),
            adapter_tensors=len(parameters),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,flags=flags,
            full_length_gradient_equivalence_proven=False,updates_discarded=True,
            offload_stats=getattr(storage,'stats',None))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');event('finished',**result)
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');event('failed',**failure);raise


if __name__=='__main__':main()
