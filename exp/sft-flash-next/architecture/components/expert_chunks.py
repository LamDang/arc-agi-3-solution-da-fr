"""Opt7: native expert dispatch in token chunks, with bounded backward replay.

CPU Parameters remain canonical. A CPU flat concatenation is an autograd input,
so its returned FP32 gradient flows through cat into every original LoRA master.
No GPU parameter closure survives the layer. Backward stages one layer, replays
one chunk under grad, consumes its VJP immediately, then releases its graph.
"""
import torch


class _ExpertChunks(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx,x,indices,weights,flat,module,size):
        ctx.module=module;ctx.size=size
        ctx.save_for_backward(x,indices,weights,flat)
        manager=module.expert_stager
        state=manager.acquire(module,x.dtype,flat.to(x.device))
        try:
            outputs=[]
            for start in range(0,x.shape[0],size):
                stop=min(start+size,x.shape[0])
                outputs.append(torch.func.functional_call(module,state,
                    (x[start:stop],indices[start:stop],weights[start:stop]),{'_prefetched':True},strict=True))
            return torch.cat(outputs,dim=0)
        finally:manager.release(module.expert_layer)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx,grad_output):
        x,indices,weights,flat=ctx.saved_tensors
        module=ctx.module;manager=module.expert_stager
        gpu_flat=flat.detach().to(x.device).requires_grad_(True)
        phase=manager.phase(-1);phase.__enter__()
        state=manager.acquire(module,x.dtype,gpu_flat,event='chunk_backward')
        dx=torch.empty_like(x);dw=torch.empty_like(weights)
        df=torch.zeros_like(gpu_flat)
        try:
            for start in range(0,x.shape[0],ctx.size):
                stop=min(start+ctx.size,x.shape[0])
                with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t):
                    a=x[start:stop].detach().requires_grad_(True)
                    w=weights[start:stop].detach().requires_grad_(True)
                    # Fresh views for each VJP; no graph shared across chunks.
                    local=dict(state);offset=0
                    for name,p in module.named_parameters():
                        local[name]=gpu_flat[offset:offset+p.numel()].view_as(p).to(p.dtype)
                        offset+=p.numel()
                    out=torch.func.functional_call(module,local,(a,indices[start:stop],w),{'_prefetched':True},strict=True)
                    ga,gw,gf=torch.autograd.grad(out,(a,w,gpu_flat),grad_output[start:stop])
                dx[start:stop]=ga;dw[start:stop]=gw;df.add_(gf)
                del out,local,ga,gw,gf,a,w
            return dx,None,dw,df.to(flat.device),None,None
        finally:
            manager.release(module.expert_layer);phase.__exit__(None,None,None)


def expert_chunks(module,x,indices,weights):
    if x.ndim!=2 or indices.shape!=weights.shape or indices.shape[0]!=x.shape[0]:
        raise ValueError('Expected flattened tokens and matching routing decisions')
    flat=torch.cat([p.reshape(-1) for p in module.parameters()])
    return _ExpertChunks.apply(x,indices,weights,flat,module,module.chunk_tokens)
