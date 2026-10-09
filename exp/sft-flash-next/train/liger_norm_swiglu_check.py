"""Real installed native normalization/expert operator checks for Opt6."""
import copy
import gc
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from liger_norm_swiglu import rms_forward, gated_rms_forward, fused_swiglu, replace_expert_activation


def relative(a, b):
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30))


def qualify(output_dir):
    import transformers.models.qwen4_exp.modeling_qwen4_exp as native
    import auto_round.modeling.fused_moe.moe_experts_interface as moe
    torch.manual_seed(606)
    rows = []
    def compare(label, reference, candidate, values, dtype):
        copies = [[x.detach().clone().requires_grad_(True) for x in values] for _ in range(2)]
        outputs = [fn(*args) for fn,args in zip((reference,candidate),copies)]
        upstream = torch.randn_like(outputs[0])
        grads = [torch.autograd.grad(y,x,upstream.clone()) for y,x in zip(outputs,copies)]
        errors = [relative(outputs[0],outputs[1])] + [relative(a,b) for a,b in zip(*grads)]
        finite = all(torch.isfinite(x).all().item() for x in outputs+list(grads[0])+list(grads[1]))
        tolerance = .02 if dtype == torch.bfloat16 else .00005
        row = dict(label=label,dtype=str(dtype),output_relative_l2=errors[0],
                   gradient_relative_l2=errors[1:],tolerance_relative_l2=tolerance,
                   finite=finite,passed=finite and max(errors)<=tolerance)
        rows.append(row)
        # Save before mutation by any later operation; operator artifacts are small.
        torch.save(dict(inputs=[x.detach().cpu() for x in values],
            reference_output=outputs[0].detach().cpu(),candidate_output=outputs[1].detach().cpu(),
            reference_gradients=[x.cpu() for x in grads[0]],candidate_gradients=[x.cpu() for x in grads[1]]),
            Path(output_dir)/('opt6-check-'+label+'-'+str(dtype).split('.')[-1]+'.pt'))
        assert row['passed'], row
    for dtype in (torch.float32, torch.bfloat16):
        for group in (None,2560):
            width = 2560 if group is None else 10240
            module = native.Qwen4ExpTextRMSNorm(width,group_size=group,eps=1e-6).cuda().to(dtype)
            with torch.no_grad():module.weight.uniform_(-.15,.15)
            x = torch.randn(2,3,width,device='cuda',dtype=dtype)
            def reference(x,w):
                y=x.float()
                if group:y=y.reshape(*y.shape[:-1],-1,group)
                y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+module.eps)
                if group:y=y.flatten(-2)
                return (y*(1+w.float())).to(x.dtype)
            def candidate(x,w):return rms_forward(SimpleNamespace(group_size=group,weight=w,eps=module.eps),x)
            compare('grouped' if group else 'rms',reference,candidate,[x,module.weight.detach()],dtype)
        x=torch.randn(2,3,128,device='cuda',dtype=dtype);gate=torch.randn_like(x)
        w=torch.empty(128,device='cuda',dtype=dtype).uniform_(.85,1.15)
        def gated_reference(x,g,w):
            z=x.float();z=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)
            return ((w*z.to(x.dtype))*torch.nn.functional.silu(g.float())).to(x.dtype)
        def gated_candidate(x,g,w):
            return gated_rms_forward(SimpleNamespace(weight=w,variance_epsilon=1e-6,activation='silu'),x,g)
        compare('gated',gated_reference,gated_candidate,[x,gate,w],dtype)
        for width in (1024,2048):
            a=torch.randn(7,width,device='cuda',dtype=dtype);b=torch.randn_like(a)
            compare('swiglu'+str(width),lambda a,b:torch.nn.functional.silu(a)*b,fused_swiglu,[a,b],dtype)
    # Execute actual native routing twice with the same tiny expert projections.
    # This catches wrong gate ordering, registry wiring, and top-k weighting changes.
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__();self.num_experts=4;self.act_fn=torch.nn.functional.silu
            self._apply_gate=lambda y:self.act_fn(y.chunk(2,-1)[0])*y.chunk(2,-1)[1]
            for i in range(4):
                self.add_module(str(i),torch.nn.ModuleDict({k:torch.nn.Linear(a,b,bias=False)
                    for k,a,b in [('gate_proj',16,32),('up_proj',16,32),('down_proj',32,16)]}))
                # Native loop uses attributes, which ModuleDict exposes by key.
    source=inspect.getsource(moe.linear_loop_experts_forward)
    namespace={**vars(moe),'_opt6_swiglu':fused_swiglu}
    exec(compile(replace_expert_activation(source),'<opt6-operator-experts>','exec'),namespace)
    ref=Experts().cuda().bfloat16();cand=copy.deepcopy(ref)
    x=torch.randn(9,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    other=x.detach().clone().requires_grad_(True)
    indices=torch.tensor([[i%4,(i+1)%4] for i in range(9)],device='cuda')
    weights=torch.rand(9,2,device='cuda',dtype=torch.bfloat16)
    a=moe.linear_loop_experts_forward(ref,x,indices,weights)
    b=namespace['linear_loop_experts_forward'](cand,other,indices,weights)
    dy=torch.randn_like(a)
    ga=torch.autograd.grad(a,[x,*ref.parameters()],dy.clone())
    gb=torch.autograd.grad(b,[other,*cand.parameters()],dy.clone())
    errors=[relative(a,b),*[relative(u,v) for u,v in zip(ga,gb)]]
    row=dict(label='native_expert_routing',output_relative_l2=errors[0],
             gradient_relative_l2=errors[1:],tolerance_relative_l2=.02,passed=max(errors)<=.02)
    rows.append(row);assert row['passed'],row
    torch.save(dict(reference_output=a.detach().cpu(),candidate_output=b.detach().cpu(),
        reference_gradients=[t.cpu() for t in ga],candidate_gradients=[t.cpu() for t in gb]),
        Path(output_dir)/'opt6-check-native-experts.pt')
    del a,b,ga,gb,ref,cand,x,other
    gc.collect();torch.cuda.empty_cache()
    # Full-anchor final/grouped norm: actual per-phase allocator peaks, native CPU offload.
    benchmarks=[]
    for label,fn in [('native',lambda m,x:native.Qwen4ExpTextRMSNorm.forward(m,x)),('liger',rms_forward)]:
        module=native.Qwen4ExpTextRMSNorm(10240,group_size=2560).cuda().bfloat16()
        module.weight.requires_grad_(False)
        x=torch.randn(1,16249,10240,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn_like(x);storage=torch.autograd.graph.save_on_cpu(pin_memory=False)
        saved=[]
        def pack(tensor):
            saved.append(dict(dtype=str(tensor.dtype),shape=list(tensor.shape),bytes=tensor.numel()*tensor.element_size()))
            return storage.pack_hook(tensor)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.autograd.graph.saved_tensors_hooks(pack,storage.unpack_hook):y=fn(module,x)
        torch.cuda.synchronize();fwd=time.monotonic()-start;fpeak=torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();y.backward(dy)
        torch.cuda.synchronize();bwd=time.monotonic()-start;bpeak=torch.cuda.max_memory_allocated()
        benchmarks.append(dict(implementation=label,forward_seconds=fwd,backward_seconds=bwd,
            forward_peak_allocated_bytes=fpeak,backward_peak_allocated_bytes=bpeak,
            saved_tensor_bytes=sum(t['bytes'] for t in saved),saved_tensors=saved))
        del x,dy,y,module;gc.collect();torch.cuda.empty_cache()
    report=dict(passed=True,cases=rows,benchmarks=benchmarks,
        tolerance_scope='isolated operators only; full-model acceptance is finite one-sample learning',
        no_model_optimizer_updates=True)
    (Path(output_dir)/'opt6-operator-check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
