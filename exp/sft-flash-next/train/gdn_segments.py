"""Exact-state GDN segmentation: carry differentiable recurrent state, never detach."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def segmented_delta(native, q, k, v, *, g, beta, initial_state=None,
                    output_final_state=False, chunk_tokens=8192, checkpoint_chunks=True, **kwargs):
    if chunk_tokens < 1 or kwargs.get('head_first') or kwargs.get('cu_seqlens') is not None:
        raise ValueError('GDN segmentation requires positive chunks and unpadded B,T,H,D inputs')
    if q.shape[1] <= chunk_tokens:
        return native(q,k,v,g=g,beta=beta,initial_state=initial_state,
                      output_final_state=output_final_state,**kwargs)
    # Shared split nodes concatenate input gradients once. Independent slices
    # would construct a full-length zero-filled gradient for every segment.
    parts = [x.split(chunk_tokens,dim=1) for x in (q,k,v,g,beta)]
    state, outputs = initial_state, []
    def block(q,k,v,g,beta,state):
        return native(q,k,v,g=g,beta=beta,initial_state=state,output_final_state=True,**kwargs)
    for values in zip(*parts):
        if checkpoint_chunks and torch.is_grad_enabled():
            output,state = checkpoint(block,*values,state,use_reentrant=False)
        else:
            output,state = block(*values,state)
        outputs.append(output)
    return torch.cat(outputs,dim=1), state if output_final_state else None


def _check(device='cuda'):
    from kernel_checks import REFERENCE_GDN
    if device == 'cuda':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as native
        # The real model repeats Q/K to its 48 value heads before this kernel.
        dtype, n, h, d, chunk = torch.bfloat16, 8193, 48, 128, 4096
    else:
        native = REFERENCE_GDN
        dtype, n, h, d, chunk = torch.float32, 129, 2, 8, 32
    torch.manual_seed(13)
    inputs = [torch.randn(1,n,h,d,device=device,dtype=dtype) for _ in range(3)]
    inputs += [torch.full((1,n,h),-.0001,device=device),
               torch.full((1,n,h),.1,device=device,dtype=dtype),
               torch.randn(1,h,d,d,device=device)]

    def run(segmented):
        q,k,v,g,beta,state = [x.detach().clone().requires_grad_(True) for x in inputs]
        kwargs = dict(g=g,beta=beta,initial_state=state,output_final_state=True,
                      use_qk_l2norm_in_kernel=True)
        if segmented:
            out,last = segmented_delta(native,q,k,v,chunk_tokens=chunk,**kwargs)
        else:
            out,last = native(q,k,v,**kwargs)
        # Last outputs/state depend on early blocks; detached carry must fail.
        loss = out[:,-4:].float().square().mean() + last.float().square().mean()
        gradients = torch.autograd.grad(loss,(q,k,v,g,beta,state))
        return [out,last,*gradients]

    ref,got = run(False),run(True)
    errors = [float((a.detach().float()-b.detach().float()).norm() /
                    a.detach().float().norm().clamp_min(1e-12)) for a,b in zip(ref,got)]
    if not all(torch.isfinite(x).all() for x in got) or max(errors) >= (.03 if device=='cuda' else 1e-4):
        raise RuntimeError(f'GDN state-carry parity failed: {errors}')
    return dict(device=device, tokens=n, heads=h, head_dim=d, chunk_tokens=chunk,
                relative_errors=errors, initial_state_gradient_norm=float(ref[-1].norm()),
                reference='Full native GDN, loss on final outputs and final state')


def check(device='cuda'):
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        return _check(device)
