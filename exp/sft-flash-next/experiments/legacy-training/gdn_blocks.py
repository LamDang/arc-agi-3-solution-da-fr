"""Checkpoint whole GDN blocks while retaining both differentiable causal states."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def blocked_forward(module, hidden_states, *, block_tokens, kernels, cache_params=None,
                    attention_mask=None, **kwargs):
    if block_tokens < 1 or cache_params is not None or attention_mask is not None:
        raise ValueError('Blocked GDN requires positive blocks, no inference cache and no padding mask')
    if kwargs.get('cu_seq_lens_q') is not None or kwargs.get('cu_seqlens') is not None:
        raise ValueError('Blocked GDN does not support packed variable-length sequences')
    kwargs = {k:v for k,v in kwargs.items() if k not in ('cu_seq_lens_q','cu_seqlens')}
    native = getattr(kernels.torch_chunk_gated_delta_rule, '_unsegmented', kernels.torch_chunk_gated_delta_rule)

    def block(x, conv_state, state):
        batch, length, _ = x.shape
        mixed = module.in_proj_qkv(x).transpose(1,2)
        if conv_state is not None:
            mixed = torch.cat((conv_state, mixed), dim=-1)
        # Keep the native projection's channel-contiguous convolution layout.
        # The alternate BF16 CUDA backward layout failed the independent VJP gate.
        mixed = mixed.transpose(1,2).contiguous().transpose(1,2)
        halo = module.conv_kernel_size - 1
        next_conv = mixed[..., -halo:].contiguous() if halo else None
        mixed = kernels.causal_conv1d_fn(mixed, module.conv1d.weight.squeeze(1),
                                        module.conv1d.bias, activation=module.activation, **kwargs)
        mixed = mixed[..., -length:].transpose(1,2)
        q,k,v = mixed.split([module.key_dim,module.key_dim,module.value_dim], dim=-1)
        q = q.reshape(batch,length,-1,module.head_k_dim)
        k = k.reshape(batch,length,-1,module.head_k_dim)
        v = v.reshape(batch,length,-1,module.head_v_dim)
        repeats = module.num_v_heads // module.num_k_heads
        if repeats > 1:
            q,k = q.repeat_interleave(repeats,dim=2), k.repeat_interleave(repeats,dim=2)
        beta = module.in_proj_b(x).sigmoid()
        g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(x).float()+module.dt_bias)
        out,state = native(q,k,v,g=g,beta=beta,initial_state=state,
                           output_final_state=True,use_qk_l2norm_in_kernel=True, **kwargs)
        z = module.in_proj_z(x).reshape(-1,module.head_v_dim)
        out = module.norm(out.reshape(-1,module.head_v_dim),z).reshape(batch,length,-1)
        return module.out_proj(out),next_conv,state

    outputs, conv_state, state = [], None, None
    for x in hidden_states.split(block_tokens,dim=1):
        if torch.is_grad_enabled():
            y,conv_state,state = checkpoint(block,x,conv_state,state,use_reentrant=False)
        else:
            y,conv_state,state = block(x,conv_state,state)
        outputs.append(y)
    return torch.cat(outputs,dim=1)


def check():
    """Independent full native module versus blocked module, at real head widths."""
    import backend
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn
    kernels = backend.rm.mq
    original = kernels.torch_chunk_gated_delta_rule, kernels.causal_conv1d_fn
    try:
        kernels.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        kernels.causal_conv1d_fn = causal_conv1d_fn
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(71)
            config = kernels.Qwen4ExpTextConfig(hidden_size=2560,linear_num_key_heads=16,
                linear_num_value_heads=48,linear_key_head_dim=128,linear_value_head_dim=128,
                linear_conv_kernel_dim=4,num_hidden_layers=1,layer_types=['linear_attention'],
                output_gate_type='sigmoid')
            module = kernels.Qwen4ExpTextGatedDeltaNet(config,0).to(device='cuda',dtype=torch.bfloat16)
            module.requires_grad_(False)
            for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
                setattr(module,name,backend.LoRALinear(getattr(module,name),rank=16,alpha=32))
            with torch.no_grad():
                module.A_log.fill_(-7.)
                module.dt_bias.fill_(-2.)
                for name,p in module.named_parameters():
                    if p.requires_grad:p.normal_(0,.003)
            tokens, block_tokens = 16385, 8192
            x = torch.randn(1,tokens,2560,device='cuda',dtype=torch.bfloat16)
            selected = [0,1,block_tokens,block_tokens+1,tokens-2,tokens-1]
            weights = torch.randn(1,len(selected),2560,device='cuda')
            parameters = [(n,p) for n,p in module.named_parameters() if p.requires_grad]
            def run(blocked):
                local = x.detach().clone().requires_grad_(True)
                out = blocked_forward(module,local,block_tokens=block_tokens,kernels=kernels) if blocked else module(local)
                loss = (out[:,selected].float()*weights).mean()
                grads = torch.autograd.grad(loss,[local,*[p for _,p in parameters]])
                return [out.detach(),*[g.detach() for g in grads]]
            reference, actual = run(False), run(True)
            errors = [float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))
                      for a,r in zip(actual,reference)]
            if not all(torch.isfinite(t).all() for t in actual) or max(errors) > .01:
                raise RuntimeError(f'Whole GDN block output/gradient parity failed: {errors}')
            return dict(tokens=tokens,block_tokens=block_tokens,hidden_size=2560,heads=48,head_dim=128,
                output_relative_error=errors[0],input_gradient_relative_error=errors[1],
                adapter_gradient_relative_errors=dict(zip([n for n,_ in parameters],errors[2:])),
                reference='Independent full native GDN module; loss spans first, boundary and last outputs',
                conv_state_detached=False,recurrent_state_detached=False,tolerance=.01)
    finally:
        kernels.torch_chunk_gated_delta_rule, kernels.causal_conv1d_fn = original
