"""Observe the native one-sample training loop; do not change its updates."""
import builtins
from collections import Counter
import json
from pathlib import Path
import time


def install(config):
    import torch
    import peft
    out=Path(config['remote_output']);launch=Path(config['remote_launch'])
    state={'step':None,'model':None,'optimizer':None,'completed_updates':0,'gradient_exports':[],'optimizer_records':[]}
    timing=launch/'timing-events.jsonl';allocator=launch/'allocator-phase-peaks.jsonl'
    def mark(event,**data):
        row=dict(event=event,step=state['step'],monotonic_ns=time.monotonic_ns(),**data)
        with timing.open('a') as f:f.write(json.dumps(row)+'\n')
    def peak(phase,reset=True):
        torch.cuda.synchronize()
        row=dict(phase=phase,step=state['step'],monotonic_ns=time.monotonic_ns(),
                 exact_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 exact_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with allocator.open('a') as f:f.write(json.dumps(row)+'\n')
        if reset:torch.cuda.reset_peak_memory_stats()
    original_factory=peft.get_peft_model
    def factory(*args,**kwargs):
        model=original_factory(*args,**kwargs);state['model']=model
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert len(trainable)==744 and all(p.dtype==torch.bfloat16 for p in trainable.values())
        assert all(torch.count_nonzero(p)==0 for n,p in trainable.items() if 'lora_B' in n)
        state['initial']={n:p.detach().cpu().clone() for n,p in trainable.items()}
        return model
    peft.get_peft_model=factory
    original_print=builtins.print
    def observed_print(*args,**kwargs):
        if len(args)==1 and isinstance(args[0],str) and args[0].startswith('{'):
            try:row=json.loads(args[0])
            except ValueError:row={}
            event=row.get('event')
            if event in ('load_start','load_complete','forward_start','loss','backward_start','update','finished','failed'):
                if event=='forward_start':state['step']=row['step']
                if event=='load_complete':peak('loading')
                elif event=='forward_start':peak('forward_setup')
                elif event=='loss':peak('forward')
                elif event=='backward_start':peak('pre_backward_export')
                elif event=='finished':peak('final_export',False)
                mark(event)
        return original_print(*args,**kwargs)
    builtins.print=observed_print
    original_backward=torch.Tensor.backward
    def backward(tensor,*args,**kwargs):
        torch.cuda.synchronize();mark('backward_call_start')
        result=original_backward(tensor,*args,**kwargs)
        peak('pure_backward');mark('backward_call_end')
        trainable={n:p for n,p in state['model'].named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        gradients={n:p.grad.detach().cpu().clone() for n,p in trainable.items()}
        name=f'gradients-step-{state["step"]:03d}.pt';torch.save(gradients,out/name)
        summary={n:dict(shape=list(g.shape),dtype=str(g.dtype),finite=bool(torch.isfinite(g).all()),
                    nonzero=int(torch.count_nonzero(g)),norm=g.double().norm().item(),max_absolute=g.abs().max().item())
                 for n,g in gradients.items()}
        (out/f'gradient-summary-step-{state["step"]:03d}.json').write_text(json.dumps(summary,indent=2)+'\n')
        import hashlib
        with (out/name).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        state['gradient_exports'].append(dict(step=state['step'],file=name,sha256=digest,tensors=len(gradients),
            all_finite=True,raw_before_clipping=True,a_nonzero=sum(v['nonzero']>0 for n,v in summary.items() if 'lora_A' in n),
            b_nonzero=sum(v['nonzero']>0 for n,v in summary.items() if 'lora_B' in n)))
        peak('gradient_export');mark('gradients_saved')
        return result
    torch.Tensor.backward=backward
    original_clip=torch.nn.utils.clip_grad_norm_
    def clip(*args,**kwargs):
        mark('clip_start');result=original_clip(*args,**kwargs);peak('clip');mark('clip_end');return result
    torch.nn.utils.clip_grad_norm_=clip
    original_step=torch.optim.AdamW.step
    def step(optimizer,*args,**kwargs):
        state['optimizer']=optimizer;mark('optimizer_start')
        result=original_step(optimizer,*args,**kwargs)
        peak('optimizer');mark('optimizer_end');state['completed_updates']+=1
        floating=[v for row in optimizer.state.values() for k,v in row.items() if isinstance(v,torch.Tensor) and v.is_floating_point()]
        finite=all(bool(torch.isfinite(t).all()) for t in floating)
        if not finite:raise RuntimeError('Nonfinite optimizer state')
        state['optimizer_records'].append(dict(update=state['completed_updates'],state_tensors=len(floating),
            state_dtype_counts=dict(Counter(str(t.dtype) for t in floating)),state_bytes=sum(t.numel()*t.element_size() for t in floating),finite=finite))
        peak('optimizer_audit');mark('optimizer_audit_end')
        return result
    torch.optim.AdamW.step=step
    def finalize():
        model=state['model'];optimizer=state['optimizer']
        if model is None:return 0
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        changed=sum(not torch.equal(p.detach().cpu(),state['initial'][n]) for n,p in trainable.items())
        if optimizer is not None:
            snapshot=optimizer.state_dict()
            snapshot['state']={k:{n:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for n,v in row.items()} for k,row in snapshot['state'].items()}
            torch.save(snapshot,out/'optimizer-state.pt')
        audit=dict(initialization='seeded-random-A-zero-B',seed=20261009,adapter_tensors=len(trainable),
            completed_updates=state['completed_updates'],all_gradients_finite=all(x['all_finite'] for x in state['gradient_exports']),
            optimizer_states_finite=all(x['finite'] for x in state['optimizer_records']),
            parameters_changed=changed>0,parameter_tensors_changed=changed,
            all_parameters_bf16=all(p.dtype==torch.bfloat16 for p in trainable.values()),
            all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in trainable.values()),
            gradient_exports=state['gradient_exports'],optimizer_records=state['optimizer_records'],
            gradient_equality_required=False,acceptance=config['learning']['acceptance'],diagnostic_only=True)
        (out/'learning-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        provenance=json.loads((out/'provenance.json').read_text())
        provenance.update(objective='target-only CCE exact with Opt3, Opt4 and BF16 model activations/LoRA',
            adapter_initialization='seeded PEFT random A / zero B, deterministically rounded to BF16',
            model_operators='pinned HF/AutoRound with documented Opt3/Opt4/BF16 activation overrides',
            gradient_equivalence_required=False,qualification='one-sample learning only',learning_config=config['learning'])
        if config.get('opt6'):
            provenance['objective']+='; Opt6 Liger RMSNorm and expert SwiGLU'
            provenance['model_operators']+='; explicit Liger RMSNorm and expert/shared-MLP SwiGLU overrides'
            provenance['opt6']=config['opt6']
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        return state['completed_updates']
    return finalize
