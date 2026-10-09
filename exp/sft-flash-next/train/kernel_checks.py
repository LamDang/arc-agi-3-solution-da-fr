"""Compare CUDA GDN/convolution forward AND backward with Torch references."""
from __future__ import annotations

import torch

from backend import rm
import backend


def unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


REFERENCE_GDN = unwrap(rm.mq.torch_chunk_gated_delta_rule)
REFERENCE_CONV = unwrap(rm.mq.causal_conv1d_fn)

def compare(actual_fn, reference_fn, inputs, tolerance=.05):
    def run(fn):
        args = [x.detach().clone().requires_grad_(True) for x in inputs]
        output = fn(*args)
        loss = output.float().square().mean()
        grads = torch.autograd.grad(loss, args)
        return [output.detach(), *grads]
    actual, reference = run(actual_fn), run(reference_fn)
    errors = []
    for a, b in zip(actual, reference):
        error = float((a.float()-b.float()).norm() / b.float().norm().clamp_min(1e-8))
        if not torch.isfinite(a).all() or error > tolerance:
            raise RuntimeError(f"Native kernel output/gradient parity failed: relative error={error}")
        errors.append(error)
    return errors


def check(device="cuda", attention_backend="sdpa"):
    if torch.device(device).type == 'cuda':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn as native_conv
    else:
        # CPU exercises the comparison harness only, never qualifies native CUDA.
        chunk_gated_delta_rule = REFERENCE_GDN
        native_conv = REFERENCE_CONV
    torch.manual_seed(42)
    shape = (1, 257, 4, 64)
    inputs = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)]
    inputs += [-torch.rand(shape[:-1], device=device), torch.rand(shape[:-1], device=device, dtype=torch.bfloat16)]
    def call(fn):
        return lambda q, k, v, g, beta: fn(q, k, v, g=g, beta=beta,
            use_qk_l2norm_in_kernel=True, output_final_state=False)[0]
    delta = compare(call(chunk_gated_delta_rule),
                    call(REFERENCE_GDN), inputs)
    conv = rm.mq.causal_conv1d_fn
    inputs = [torch.randn(1, 128, 257, device=device, dtype=torch.bfloat16),
              torch.randn(128, 4, device=device, dtype=torch.bfloat16)]
    convolution = compare(lambda x, w: native_conv(x, w, None, activation="silu"),
                          lambda x, w: REFERENCE_CONV(x, w, None, activation="silu"), inputs)
    rm.mq.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    rm.mq.causal_conv1d_fn = native_conv
    extras = {}
    if torch.device(device).type == 'cuda':
        norm = rm.mq.Qwen4ExpTextRMSNorm(10240, group_size=2560).to(device=device, dtype=torch.bfloat16)
        norm.requires_grad_(False)
        with torch.no_grad():
            norm.weight.normal_(0, .1)
        gated = rm.mq.Qwen4ExpTextRMSNormGated(128,activation='sigmoid').to(device=device, dtype=torch.bfloat16)
        gated.requires_grad_(False)
        extras['gated_rms_norm_relative_errors'] = compare(
            lambda x,g: backend.FrozenGatedRMSNorm.apply(x,g,gated.weight,gated.variance_epsilon,31,gated.activation),
            gated, [torch.randn(2,137,128,device=device,dtype=torch.bfloat16) for _ in range(2)], tolerance=.008)
        extras['rms_norm_relative_errors'] = compare(
            lambda x: backend.FrozenRMSNorm.apply(x, norm.weight, norm.eps, norm.group_size, 31),
            norm, [torch.randn(1, 137, 10240, device=device, dtype=torch.bfloat16)], tolerance=.008)
        if attention_backend == 'triton':
            from sparse_kernels import IndexedAttention
            n = 67
            inputs = [torch.randn(h, n, 256, device=device, dtype=torch.bfloat16) for h in (24, 2, 2)]
            selected = torch.arange(n, device=device).repeat(n, 1)
            selected = torch.where(selected <= torch.arange(n, device=device)[:, None], selected, -1)
            extras['indexed_attention_relative_errors'] = compare(
                lambda q,k,v: IndexedAttention.apply(q,k,v,selected,256**-.5,32),
                lambda q,k,v: backend.attention_block(q.float(),k.float(),v.float(),selected,256**-.5),
                inputs, tolerance=.015)
    return dict(**extras, native_cuda=torch.device(device).type == "cuda", gated_delta_output_gradient_relative_errors=delta,
                convolution_output_gradient_relative_errors=convolution, tolerance=.05)


def check_model_backward(model, loss_block=128, tokens=512, tolerance=.05):
    """Check attention backward with identical full-model forward values/routes.

    Changing BF16 attention arithmetic can change discrete expert choices. A
    separate native-forward diagnostic records that drift; it must not be
    confused with a VJP comparison at the same forward computation.
    """
    import types
    vocab = model.config.text_config.vocab_size
    ids = torch.randint(1000, min(vocab, 100000), (1, tokens),
                        generator=torch.Generator().manual_seed(20261008))
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    ple_errors = []
    for layer in model.model.language_model.layers:
        module = layer.ple
        if module is not None and hasattr(module, 'train_ple_block'):
            device = module.key_proj.weight.device
            test_ids = ids[:,:min(tokens,257)].to(device)
            x = torch.randn(1,test_ids.shape[1],module.hidden_size*module.hc_count,
                            device=device,dtype=module.key_proj.weight.dtype)
            with backend.request_ple_cache(model):
                ple_errors.append(compare(
                    lambda h: backend.FrozenPLE.apply(h,test_ids,module,31),
                    lambda h: type(module).forward(module,h,test_ids,None), [x], tolerance=.03))
    routes = {}
    hooks = []
    for i, layer in enumerate(model.model.language_model.layers):
        def record(module, args, output, i=i):
            if i not in routes:
                routes[i] = output[2].detach().cpu().sort(dim=-1).values
        hooks.append(layer.mlp.gate.register_forward_hook(record))
    attention = [(m, m.train_attention_backend) for m in model.modules() if hasattr(m, 'train_attention_backend')]
    has_triton = any(name == 'triton' for _,name in attention)
    original_op = None
    forward_errors = []
    if has_triton:
        import sparse_kernels
        original_op = sparse_kernels.IndexedAttention
        class ReferenceBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, selected, scale, key_block):
                ctx.scale, ctx.block = scale, 16
                ctx.save_for_backward(q, k, v, selected)
                out = original_op.apply(q, k, v, selected, scale, key_block)
                if len(forward_errors) < len(attention):
                    expected = torch.cat([backend.attention_block(q[:,i:i+16].float(),k.float(),v.float(),
                        selected[i:i+16],scale) for i in range(0,q.shape[1],16)],dim=1).to(q.dtype)
                    error = float((out.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-8))
                    forward_errors.append(error)
                return out
            @staticmethod
            def backward(ctx, grad):
                return backend.SparseAttention.backward(ctx, grad)
    def backward():
        routes.clear()
        model.zero_grad(set_to_none=True)
        with backend.request_ple_cache(model):
            loss = backend.loss_sum(model, {'input_ids': ids}, tokens-128, 'cuda', loss_block)/128
            loss.backward()
        return float(loss.detach()), {n:p.grad.detach().float().cpu().clone() for n,p in parameters}, dict(routes)
    originals = []
    group = getattr(model, 'train_checkpoint_group', 1)
    try:
        if has_triton:
            sparse_kernels.IndexedAttention = ReferenceBackward
        try:
            reference_loss, reference, reference_routes = backward()
        finally:
            if has_triton:
                sparse_kernels.IndexedAttention = original_op
        optimized_loss, optimized, optimized_routes = backward()
        if any(not torch.equal(reference_routes[i], optimized_routes[i]) for i in reference_routes):
            raise RuntimeError('Matched-forward gradient check changed expert routes')
        difference = sum(float((optimized[n]-reference[n]).double().square().sum()) for n,_ in parameters)
        denominator = sum(float(reference[n].double().square().sum()) for n,_ in parameters)
        relative = (difference/max(denominator, 1e-30))**.5
        # Independent forward comparison, explicitly allowed to choose different
        # experts after floating-point changes; never substitute its VJP above.
        for module in model.model.language_model.modules():
            if isinstance(module, (rm.mq.Qwen4ExpTextRMSNorm, rm.mq.Qwen4ExpTextGatedResidual,
                                   rm.mq.Qwen4ExpTextPLELayer, rm.mq.Qwen4ExpTextGatedDeltaNet, rm.mq.Qwen4ExpTextRMSNormGated)):
                originals.append((module, module.forward))
                module.forward = types.MethodType(type(module).forward, module)
        for module, _ in attention:
            module.train_attention_backend = 'sdpa'
        model.train_checkpoint_group = 1
        routes.clear()
        with torch.no_grad(), backend.request_ple_cache(model):
            native_loss = float(backend.loss_sum(model, {'input_ids':ids}, tokens-128, 'cuda', loss_block)/128)
        changed = sum(int((routes[i] != optimized_routes[i]).any(dim=-1).sum()) for i in routes)
        total = sum(len(v) for v in routes.values())
        replaced = sum(int((~(routes[i][...,None] == optimized_routes[i][:,None,:]).any(dim=-1)).sum()) for i in routes)
        choices = sum(v.numel() for v in routes.values())
    finally:
        for module, forward in originals:
            module.forward = forward
        for module, name in attention:
            module.train_attention_backend = name
        model.train_checkpoint_group = group
        model.zero_grad(set_to_none=True)
        for hook in hooks:
            hook.remove()
    if not torch.isfinite(torch.tensor([reference_loss, optimized_loss, relative, native_loss])).all() or relative > tolerance or abs(reference_loss-optimized_loss) > 1e-4 or abs(native_loss-optimized_loss) > .01 or max(forward_errors, default=0.) > .005:
        raise RuntimeError(f'Real-model parity failed: matched-forward gradient error={relative}, native/optimized losses={native_loss}/{optimized_loss}')
    return dict(reference_loss=reference_loss, optimized_loss=optimized_loss,
                windowed_ple_output_gradient_relative_errors=ple_errors,
                adapter_gradient_relative_error=relative, tolerance=tolerance,
                gradient_reference='Identical optimized forward; independent FP32 SDPA autograd backward',
                native_forward_loss=native_loss, native_forward_changed_route_fraction=changed/max(total,1),
                native_forward_replaced_expert_fraction=replaced/max(choices,1),
                actual_attention_forward_relative_errors=forward_errors,
                tokens=tokens, targets=128, optimizer_updates=0)
