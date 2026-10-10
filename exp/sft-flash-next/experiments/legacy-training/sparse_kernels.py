"""Gather-free indexed GQA for the frozen QSA selection, BF16/FP16 CUDA.

Each program handles one query and one KV head, sharing its gathered keys
across all grouped query heads. Backward recomputes probabilities and uses
FP32 atomic accumulation for K/V. No context is detached or approximated.
Floating-point reduction order differs from the reference SDPA implementation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, Selected, Out, LSE,
             QH: tl.constexpr, QL: tl.constexpr, QD: tl.constexpr,
             KH: tl.constexpr, KL: tl.constexpr, KD: tl.constexpr,
             VH: tl.constexpr, VL: tl.constexpr, VD: tl.constexpr,
             N: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
             GROUP: tl.constexpr, SCALE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    token, kv = tl.program_id(0), tl.program_id(1)
    heads = kv * GROUP + tl.arange(0, BM)
    dims = tl.arange(0, D)
    valid_head = tl.arange(0, BM) < GROUP
    q = tl.load(Q + heads[:, None] * QH + token * QL + dims[None, :] * QD,
                mask=valid_head[:, None], other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(S, BN)):
        positions = start * BN + tl.arange(0, BN)
        indices = tl.load(Selected + token * S + positions, mask=positions < S, other=-1)
        valid = (indices >= 0) & (indices < N) & (positions < S)
        k = tl.load(K + kv * KH + indices[:, None] * KL + dims[None, :] * KD,
                    mask=valid[:, None], other=0)
        v = tl.load(V + kv * VH + indices[:, None] * VL + dims[None, :] * VD,
                    mask=valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(valid[None, :], scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        # Empty leading tiles must not replace the running maximum with zero:
        # actual pretrained logits can be very negative, especially the first
        # three queries whose only valid keys live in the final tail slots.
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        correction = tl.where(maximum == -float('inf'), 0., tl.exp(maximum - safe_max))
        probabilities = tl.exp(scores - safe_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        acc = acc * correction[:, None] + tl.dot(probabilities, v.to(tl.float32), input_precision='tf32x3')
        maximum = next_max
    result = acc / tl.maximum(denominator[:, None], 1.e-20)
    tl.store(Out + heads[:, None] * N * D + token * D + dims[None, :],
             result, mask=valid_head[:, None])
    tl.store(LSE + heads * N + token, maximum + tl.log(denominator), mask=valid_head)


@triton.jit
def _backward(Q, K, V, Selected, Out, LSE, Grad, DQ, DK, DV,
              QH: tl.constexpr, QL: tl.constexpr, QD: tl.constexpr,
              KH: tl.constexpr, KL: tl.constexpr, KD: tl.constexpr,
              VH: tl.constexpr, VL: tl.constexpr, VD: tl.constexpr,
              GH: tl.constexpr, GL: tl.constexpr, GD: tl.constexpr,
              N: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
              GROUP: tl.constexpr, SCALE: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    token, kv = tl.program_id(0), tl.program_id(1)
    heads = kv * GROUP + tl.arange(0, BM)
    dims = tl.arange(0, D)
    valid_head = tl.arange(0, BM) < GROUP
    q = tl.load(Q + heads[:, None] * QH + token * QL + dims[None, :] * QD,
                mask=valid_head[:, None], other=0)
    do = tl.load(Grad + heads[:, None] * GH + token * GL + dims[None, :] * GD,
                 mask=valid_head[:, None], other=0)
    out = tl.load(Out + heads[:, None] * N * D + token * D + dims[None, :],
                  mask=valid_head[:, None], other=0)
    delta = tl.sum(do.to(tl.float32) * out.to(tl.float32), axis=1)
    lse = tl.load(LSE + heads * N + token, mask=valid_head, other=0)
    dq = tl.zeros((BM, D), tl.float32)
    for start in range(tl.cdiv(S, BN)):
        positions = start * BN + tl.arange(0, BN)
        indices = tl.load(Selected + token * S + positions, mask=positions < S, other=-1)
        valid = (indices >= 0) & (indices < N) & (positions < S)
        k = tl.load(K + kv * KH + indices[:, None] * KL + dims[None, :] * KD,
                    mask=valid[:, None], other=0)
        v = tl.load(V + kv * VH + indices[:, None] * VL + dims[None, :] * VD,
                    mask=valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        p = tl.where(valid[None, :] & valid_head[:, None], tl.exp(scores - lse[:, None]), 0.)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, k.to(tl.float32), input_precision='tf32x3') * SCALE
        dk = tl.dot(tl.trans(ds), q.to(tl.float32), input_precision='tf32x3') * SCALE
        dv = tl.dot(tl.trans(p), do.to(tl.float32), input_precision='tf32x3')
        offsets = kv * N * D + indices[:, None] * D + dims[None, :]
        tl.atomic_add(DK + offsets, dk, mask=valid[:, None], sem='relaxed')
        tl.atomic_add(DV + offsets, dv, mask=valid[:, None], sem='relaxed')
    tl.store(DQ + heads[:, None] * N * D + token * D + dims[None, :], dq,
             mask=valid_head[:, None])


class IndexedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected, scale, key_block=32):
        if q.dtype not in (torch.bfloat16, torch.float16) or q.device.type != 'cuda':
            raise ValueError('Indexed Triton attention requires BF16/FP16 CUDA')
        h, n, d = q.shape
        if k.shape != v.shape or k.shape[1:] != (n, d) or h % k.shape[0]:
            raise ValueError('Invalid grouped attention shape')
        if selected.ndim != 2 or selected.shape[0] != n or not selected.is_contiguous():
            raise ValueError('Selected keys must be contiguous [tokens, slots]')
        if d not in (32, 64, 128, 256):
            raise ValueError('Unsupported attention head dimension')
        group = h // k.shape[0]
        bm = max(16, triton.next_power_of_2(group))
        out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        lse = torch.empty((h, n), device=q.device, dtype=torch.float32)
        _forward[(n, k.shape[0])](q, k, v, selected, out, lse,
            *q.stride(), *k.stride(), *v.stride(), n, selected.shape[1], d,
            group, scale, bm, key_block, num_warps=8, num_stages=1)
        ctx.save_for_backward(q, k, v, selected, out, lse)
        ctx.scale, ctx.key_block = scale, key_block
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, selected, out, lse = ctx.saved_tensors
        h, n, d = q.shape
        group = h // k.shape[0]
        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        _backward[(n, k.shape[0])](q, k, v, selected, out, lse, grad, dq, dk, dv,
            *q.stride(), *k.stride(), *v.stride(), *grad.stride(), n, selected.shape[1], d,
            group, ctx.scale, max(16, triton.next_power_of_2(group)), ctx.key_block,
            num_warps=8, num_stages=1)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None
