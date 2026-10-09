"""Opt7: globally route once; replay bounded native per-expert token slices.

The global native argsort order is preserved before slicing. Only row/slot IDs
are expanded; hidden vectors are gathered for the current expert slice. FP32
sums combine already-rounded BF16 routing products, then cast once as native
sum(dim=1) does. CPU master LoRA objects receive FP32 gradients through cat.
"""
import torch


def layout(module):
    result=[];offset=0
    for name,p in module.named_parameters():
        result.append((name,offset,p.numel(),p.shape,p.dtype,p.requires_grad));offset+=p.numel()
    return result


def slices(order,counts,k,size):
    offset=0
    for expert,count in enumerate(counts):
        for start in range(offset,offset+count,size):
            pairs=order[start:min(start+size,offset+count)]
            yield expert,pairs//k,pairs%k
        offset+=count


class _ExpertChunks(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx,x,indices,weights,flat,module,size):
        order=torch.argsort(indices.reshape(-1))
        counts=torch.bincount(indices.reshape(-1),minlength=module.num_experts).tolist()
        ctx.module=module;ctx.size=size;ctx.counts=counts;ctx.layout=layout(module)
        ctx.save_for_backward(x,indices,weights,flat,order)
        manager=module.expert_stager;state=manager.acquire(module,x.dtype,flat.to(x.device))
        output=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
        try:
            for expert,rows,slots in slices(order,counts,indices.shape[1],size):
                value=torch.func.functional_call(module,state,(x[rows],None,None),
                    {'_prefetched':True,'_expert_index':expert},strict=True)
                weighted=value*weights[rows,slots,None].to(x.dtype)
                output.index_add_(0,rows,weighted.float())
            module.chunk_stats=dict(max_dispatched_tokens=min(max(counts),size),
                expert_slices=sum((c+size-1)//size for c in counts),global_native_sort=True,
                fp32_output_accumulator_bytes=output.numel()*output.element_size())
            return output.to(x.dtype)
        finally:manager.release(module.expert_layer)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx,grad_output):
        x,indices,weights,flat,order=ctx.saved_tensors
        module=ctx.module;manager=module.expert_stager
        gpu_flat=flat.detach().to(x.device)
        phase=manager.phase(-1);phase.__enter__()
        state=manager.acquire(module,x.dtype,gpu_flat,event='chunk_backward')
        dx=torch.zeros(x.shape,device=x.device,dtype=torch.float32)
        dw=torch.zeros_like(weights);df=torch.zeros_like(gpu_flat)
        try:
            for expert,rows,slots in slices(order,ctx.counts,indices.shape[1],ctx.size):
                with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t):
                    a=x[rows].detach().requires_grad_(True)
                    w=weights[rows,slots].detach().requires_grad_(True)
                    local=dict(state);leaves=[];offsets=[]
                    for name,offset,count,shape,dtype,trainable in ctx.layout:
                        if name.startswith(str(expert)+'.') and trainable:
                            p=gpu_flat[offset:offset+count].view(shape).detach().requires_grad_(True)
                            local[name]=p;leaves.append(p);offsets.append((offset,count))
                    value=torch.func.functional_call(module,local,(a,None,None),
                        {'_prefetched':True,'_expert_index':expert},strict=True)
                    out=value*w[:,None].to(x.dtype)
                    ga,gw,*gp=torch.autograd.grad(out,(a,w,*leaves),grad_output[rows])
                dx.index_add_(0,rows,ga.float());dw[rows,slots]=gw
                for (offset,count),gradient in zip(offsets,gp):df[offset:offset+count].add_(gradient.reshape(-1))
                del out,value,local,ga,gw,gp,a,w,leaves
            return dx.to(x.dtype),None,dw,df.to(flat.device),None,None
        finally:
            manager.release(module.expert_layer);phase.__exit__(None,None,None)


def expert_chunks(module,x,indices,weights):
    if x.ndim!=2 or indices.shape!=weights.shape or indices.shape[0]!=x.shape[0]:
        raise ValueError('Expected flattened tokens and matching routing decisions')
    flat=torch.cat([p.reshape(-1) for p in module.parameters()])
    return _ExpertChunks.apply(x,indices,weights,flat,module,module.chunk_tokens)
