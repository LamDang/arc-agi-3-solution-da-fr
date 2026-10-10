# Expert projection arithmetic derived from AutoRound (Apache-2.0), pinned in model.py.
"""Opt7: globally route once; replay bounded native per-expert token slices.

The global native argsort order is preserved before slicing. Only row/slot IDs
are expanded; hidden vectors are gathered for the current expert slice. Native
routing multiplication and sum run in original top-k order in token windows,
using temporary CPU slot-output scratch to avoid a full expanded CUDA buffer.
GPU master LoRA objects receive GPU FP32 gradients through cat.
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


def projection_states(state):
    grouped={}
    for name,value in state.items():
        parts=name.split('.',2)
        if len(parts)==3 and parts[0].isdigit():
            grouped.setdefault(int(parts[0]),{}).setdefault(parts[1],{})[parts[2]]=value
    return grouped


def expert_value(module,states,x,expert,pad_to=0):
    rows=x.shape[0]
    if pad_to and rows<pad_to:x=torch.nn.functional.pad(x,(0,0,0,pad_to-rows))
    child=getattr(module,str(expert))
    def project(name,value):
        return torch.func.functional_call(getattr(child,name),states[name],(value,),strict=True)
    gate=project('gate_proj',x);up=project('up_proj',x)
    gated=module._apply_gate(torch.cat([gate,up],dim=-1)) if hasattr(module,'_apply_gate') else module.act_fn(gate)*up
    return project('down_proj',gated)[:rows].to(x.dtype)


def route_sum_native(slot_values,weights,device,size):
    """Use native multiply/sum in original top-k order, with CUDA window bounds."""
    tokens,_,hidden=slot_values.shape
    output=torch.empty((tokens,hidden),device=device,dtype=slot_values.dtype)
    for start in range(0,tokens,size):
        stop=min(start+size,tokens)
        local=slot_values[start:stop].to(device)
        weighted=local*weights[start:stop,:,None].to(local.dtype)
        output[start:stop]=weighted.sum(dim=1).to(output.dtype)
        del local,weighted
    return output


def unroute_native(slot_grad,device,size):
    """Replay native gather backward in token windows; preserve BF16 rounding.

CPU per-slot cotangents are temporary backward scratch, never an artifact.
No full expanded gradient is allocated on CUDA.
"""
    tokens,k,hidden=slot_grad.shape
    dx=torch.empty((tokens,hidden),device=device,dtype=slot_grad.dtype)
    for start in range(0,tokens,size):
        stop=min(start+size,tokens);n=stop-start
        with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t):
            source=torch.zeros((n,hidden),device=device,dtype=slot_grad.dtype,requires_grad=True)
            rows=torch.arange(n,device=device).unsqueeze(1).expand(-1,k).reshape(-1)
            expanded=source[rows]
            grad=slot_grad[start:stop].to(device).reshape(n*k,hidden)
            local=torch.autograd.grad(expanded,source,grad)[0]
        dx[start:stop]=local
        del source,rows,expanded,grad,local
    return dx


class _ExpertChunks(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx,x,indices,weights,flat,module,size):
        order=torch.argsort(indices.reshape(-1))
        counts=torch.bincount(indices.reshape(-1),minlength=module.num_experts).tolist()
        ctx.module=module;ctx.size=size;ctx.counts=counts;ctx.layout=layout(module)
        ctx.save_for_backward(x,indices,weights,flat,order)
        manager=module.expert_stager;state=manager.acquire(module,x.dtype,flat.to(x.device))
        groups=projection_states(state)
        slot_values=torch.empty((*indices.shape,x.shape[-1]),device='cpu',dtype=x.dtype)
        try:
            for expert,rows,slots in slices(order,counts,indices.shape[1],size):
                value=expert_value(module,groups[expert],x[rows],expert,size if counts[expert]>size else 0)
                slot_values[rows.cpu(),slots.cpu()]=value.cpu()
                del value
            output=route_sum_native(slot_values,weights,x.device,size)
            module.chunk_stats=dict(max_dispatched_tokens=min(max(counts),size),
                max_assigned_tokens=max(counts),experts_split=sum(c>size for c in counts),
                expert_slices=sum((c+size-1)//size for c in counts),global_native_sort=True,
                forward_cpu_slot_value_bytes=slot_values.numel()*slot_values.element_size(),
                native_topk_reduction=True,padded_split_terminal_rows=True)
            return output
        finally:manager.release(module.expert_layer)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx,grad_output):
        x,indices,weights,flat,order=ctx.saved_tensors
        module=ctx.module;manager=module.expert_stager
        gpu_flat=flat.detach().to(x.device)
        phase=manager.phase(-1);phase.__enter__()
        state=manager.acquire(module,x.dtype,gpu_flat,event='chunk_backward')
        groups=projection_states(state)
        layouts={}
        for row in ctx.layout:
            if row[0].split('.')[0].isdigit():layouts.setdefault(int(row[0].split('.')[0]),[]).append(row)
        slot_grad=torch.empty((*indices.shape,x.shape[-1]),device='cpu',dtype=x.dtype)
        dw=torch.zeros_like(weights);df=torch.zeros_like(gpu_flat)
        try:
            for expert,rows,slots in slices(order,ctx.counts,indices.shape[1],ctx.size):
                with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t):
                    a=x[rows].detach().requires_grad_(True)
                    w=weights[rows,slots].detach().requires_grad_(True)
                    local={name:dict(values) for name,values in groups[expert].items()};leaves=[];offsets=[]
                    for name,offset,count,shape,dtype,trainable in layouts[expert]:
                        if name.startswith(str(expert)+'.') and trainable:
                            p=gpu_flat[offset:offset+count].view(shape).detach().requires_grad_(True)
                            _,projection,key=name.split('.',2)
                            local[projection][key]=p;leaves.append(p);offsets.append((offset,count))
                    value=expert_value(module,local,a,expert,ctx.size if ctx.counts[expert]>ctx.size else 0)
                    out=value*w[:,None].to(x.dtype)
                    ga,gw,*gp=torch.autograd.grad(out,(a,w,*leaves),grad_output[rows])
                slot_grad[rows.cpu(),slots.cpu()]=ga.cpu();dw[rows,slots]=gw
                for (offset,count),gradient in zip(offsets,gp):df[offset:offset+count].add_(gradient.reshape(-1))
                del out,value,local,ga,gw,gp,a,w,leaves
            dx=unroute_native(slot_grad,x.device,ctx.size)
            module.chunk_stats['backward_cpu_slot_gradient_bytes']=slot_grad.numel()*slot_grad.element_size()
            module.chunk_stats['native_gather_backward']=True
            return dx,None,dw,df.to(flat.device),None,None
        finally:
            manager.release(module.expert_layer);phase.__exit__(None,None,None)


def expert_chunks(module,x,indices,weights):
    if x.ndim!=2 or indices.shape!=weights.shape or indices.shape[0]!=x.shape[0]:
        raise ValueError('Expected flattened tokens and matching routing decisions')
    flat=torch.cat([p.reshape(-1) for p in module.parameters()])
    return _ExpertChunks.apply(x,indices,weights,flat,module,module.chunk_tokens)
