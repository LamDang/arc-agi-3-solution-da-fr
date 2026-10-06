"""Qwen3.8-Flash-Next (Intel W4A16 AutoRound checkpoint) instrumented for REAP.

Built on the transformers `qwen4_exp` model with three replacements:

- `Int4Experts`: the routed experts stay 4-bit (GPTQ packing) on the GPU and
  are dequantized per layer. Each expert's output is computed explicitly,
  which is what REAP needs: per expert j, the router weight g_j(x) and the
  output norm ||f_j(x)|| of every token routed to it.
- `indexed_attention_forward`: the full-attention layers keep their QSA
  indexer semantics (each query attends to the top `indexer_budget` tokens,
  chosen in blocks of `indexer_compress_ratio`, plus the incomplete last
  block), but select per query and gather, instead of looping over queries
  in Python and building dense query x key masks as the reference does.
- `MmapEmbedding`: the 51B-parameter n-gram (PLE) table is read in place from
  the safetensors files through numpy memmaps, in host RAM / page cache.

Everything else is the reference transformers code with the checkpoint's
BF16 weights.
"""
from __future__ import annotations

import json
import re
import struct
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq

N_CATEGORIES = 3
# Speed options, set by the caller (see bench.py). Both keep the math:
#   compile_dequant  fuse the int4 -> BF16 dequantization into one kernel (torch.compile)
#   attention        "einsum": gathered keys in fp32; "sdpa": PyTorch's attention kernel on the gathered keys
OPTIONS = {"compile_dequant": False, "attention": "einsum"}
GPTQ_ZERO = 8  # symmetric 4-bit: stored qzeros nibble 7, GPTQ v1 adds one
_ZEROS_WORD = 0x77777777


class ReapRecorder:
    """Per-layer, per-expert sums, split by token category.

    count      tokens routed to the expert
    gate       sum of router weights g_j(x) (after top-k renormalization)
    norm       sum of ||f_j(x)||
    gate_norm  sum of g_j(x) * ||f_j(x)||   -> REAP score = gate_norm / count
    prob       sum of the full softmax probability (routed or not)
    """

    FIELDS = ("count", "gate", "norm", "gate_norm", "prob")
    # the same sums split by the token's position in the sequence (context length so far):
    # <name>_pos [layer, expert, category, band], bands [0, 32K), [32K, 64K), [64K, 96K), [96K, inf)
    POSITION_FIELDS = ("count", "gate", "gate_norm")
    POSITION_BANDS = (32768, 65536, 98304)

    def __init__(self, n_layers: int, n_experts: int, device):
        self.data = {
            f: torch.zeros(n_layers, n_experts, N_CATEGORIES, dtype=torch.float64, device=device)
            for f in self.FIELDS
        }
        n_bands = len(self.POSITION_BANDS) + 1
        self.data.update({
            f"{f}_pos": torch.zeros(n_layers, n_experts, N_CATEGORIES, n_bands, dtype=torch.float64, device=device)
            for f in self.POSITION_FIELDS
        })
        self.categories: torch.Tensor | None = None  # [tokens] of the current chunk
        self.positions: torch.Tensor | None = None  # [tokens] position of each token in its sequence
        self.enabled = True

    def reset(self):
        for value in self.data.values():
            value.zero_()

    def numpy(self) -> dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self.data.items()}

    def _cats(self, n: int, device) -> torch.Tensor:
        if self.categories is None:
            return torch.zeros(n, dtype=torch.long, device=device)
        if self.categories.shape[0] != n:
            raise ValueError(f"recorder has {self.categories.shape[0]} categories for {n} tokens")
        return self.categories

    def _bands(self, n: int, device) -> torch.Tensor:
        if self.positions is None:
            return torch.zeros(n, dtype=torch.long, device=device)
        if self.positions.shape[0] != n:
            raise ValueError(f"recorder has {self.positions.shape[0]} positions for {n} tokens")
        edges = torch.tensor(self.POSITION_BANDS, device=self.positions.device)
        return torch.bucketize(self.positions, edges, right=True)


# --------------------------------------------------------------------------- experts


def dequantize_gptq(qweight: torch.Tensor, scales: torch.Tensor, dtype, out: torch.Tensor | None = None):
    """[..., K/8, N] int32 + [..., K/group, N] fp16 -> [..., K, N] weights.
    Nibble i of each int32 is input row 8*r + i; w = (q - 8) * scale."""
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = (qweight.unsqueeze(-2) >> shifts.view(8, 1)) & 0xF
    q = q.reshape(*qweight.shape[:-2], qweight.shape[-2] * 8, qweight.shape[-1])
    group = q.shape[-2] // scales.shape[-2]
    w = (q.to(torch.float32) - GPTQ_ZERO) * scales.to(torch.float32).repeat_interleave(group, dim=-2)
    if out is None:
        return w.to(dtype)
    out.copy_(w)
    return out


def _dequantize_into(qweight, scales, out):
    dequantize_gptq(qweight, scales, out.dtype, out=out)


_COMPILED_DEQUANT = None


def _dequant_function(device):
    """The compiled dequantization when enabled and on a GPU, else eager."""
    global _COMPILED_DEQUANT
    if not OPTIONS["compile_dequant"] or torch.device(device).type != "cuda":
        return _dequantize_into, False
    if _COMPILED_DEQUANT is None:
        _COMPILED_DEQUANT = torch.compile(_dequantize_into, fullgraph=True, dynamic=False)
    return _COMPILED_DEQUANT, True


class _Workspace:
    """Dequantized weights of one layer, shared by all layers."""

    buffers: dict = {}

    @classmethod
    def get(cls, name, shape, dtype, device):
        key = (name, tuple(shape), dtype, str(device))
        if key not in cls.buffers:
            cls.buffers[key] = torch.empty(shape, dtype=dtype, device=device)
        return cls.buffers[key]


class Int4Experts(nn.Module):
    """Routed experts in GPTQ int4, forward-compatible with Qwen4ExpTextExperts."""

    def __init__(self, config, layer_idx: int, recorder: ReapRecorder | None, group_size: int = 128):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = E = config.num_experts
        self.hidden_dim = H = config.hidden_size
        self.intermediate_dim = I = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.recorder = recorder
        self.impl = "auto"  # "grouped", "loop" or "auto" (grouped if it works)
        self.dequant_block = 64  # experts dequantized at once (bounds fp32 temporaries)
        # gate and up share one matrix: columns [0, I) gate, [I, 2I) up, like gate_up_proj
        self.register_buffer("qweight_gate_up", torch.empty(E, H // 8, 2 * I, dtype=torch.int32))
        self.register_buffer("scales_gate_up", torch.empty(E, H // group_size, 2 * I, dtype=torch.float16))
        self.register_buffer("qweight_down", torch.empty(E, I // 8, H, dtype=torch.int32))
        self.register_buffer("scales_down", torch.empty(E, I // group_size, H, dtype=torch.float16))
        self.filled = None  # [E, 6] bool, set by the loader
        self.keep = None  # bool [E]: experts left after pruning (None: all)

    # -- loading
    _PARTS = {("gate_proj", "qweight"): 0, ("gate_proj", "scales"): 1, ("up_proj", "qweight"): 2,
              ("up_proj", "scales"): 3, ("down_proj", "qweight"): 4, ("down_proj", "scales"): 5}

    def load_part(self, expert: int, proj: str, kind: str, tensor: torch.Tensor):
        if self.filled is None:
            self.filled = torch.zeros(self.num_experts, 6, dtype=torch.bool)
        if kind == "qzeros":
            if not bool((tensor == _ZEROS_WORD).all()):
                raise ValueError(f"layer {self.layer_idx} expert {expert} {proj}: non-symmetric qzeros")
            return
        I = self.intermediate_dim
        if proj == "down_proj":
            target = self.qweight_down if kind == "qweight" else self.scales_down
            target[expert].copy_(tensor)
        else:
            cols = slice(0, I) if proj == "gate_proj" else slice(I, 2 * I)
            target = self.qweight_gate_up if kind == "qweight" else self.scales_gate_up
            target[expert, :, cols].copy_(tensor)
        self.filled[expert, self._PARTS[(proj, kind)]] = True

    def check_filled(self):
        if self.filled is None or not bool(self.filled.all()):
            raise ValueError(f"layer {self.layer_idx}: experts not fully loaded")

    # -- compute
    def _dequantized(self, dtype):
        E, H, I = self.num_experts, self.hidden_dim, self.intermediate_dim
        device = self.qweight_gate_up.device
        w1 = _Workspace.get("w1", (E, H, 2 * I), dtype, device)
        w2 = _Workspace.get("w2", (E, I, H), dtype, device)
        fn, fused = _dequant_function(device)
        block = E if fused else self.dequant_block  # a fused kernel has no fp32 temporaries to bound
        with torch.profiler.record_function("reap.dequant"):
            for s in range(0, E, block):
                e = min(E, s + block)
                fn(self.qweight_gate_up[s:e], self.scales_gate_up[s:e], w1[s:e])
                fn(self.qweight_down[s:e], self.scales_down[s:e], w2[s:e])
        return w1, w2

    def _expert_outputs_loop(self, x, counts, dtype):
        """Per-expert outputs for tokens sorted by expert: x [N, H] -> [N, H]."""
        y = torch.empty_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            rows = slice(start, start + n)
            w1 = dequantize_gptq(self.qweight_gate_up[e], self.scales_gate_up[e], dtype)
            w2 = dequantize_gptq(self.qweight_down[e], self.scales_down[e], dtype)
            gate, up = (x[rows] @ w1).chunk(2, dim=-1)
            y[rows] = (self.act_fn(gate) * up) @ w2
            start += n
        return y

    def _expert_outputs_grouped(self, x, counts, dtype):
        w1, w2 = self._dequantized(dtype)
        offs = counts.cumsum(0).to(torch.int32)
        h = torch._grouped_mm(x, w1, offs=offs)
        gate, up = h.chunk(2, dim=-1)
        return torch._grouped_mm((self.act_fn(gate) * up).contiguous(), w2, offs=offs)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        T, k = top_k_index.shape
        dtype = hidden_states.dtype
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token = order // k
        counts = torch.bincount(flat, minlength=self.num_experts)
        x = hidden_states[token]
        with torch.profiler.record_function("reap.experts"):
            return self._forward_sorted(hidden_states, x, flat, order, token, counts, top_k_weights, T, dtype)

    def _forward_sorted(self, hidden_states, x, flat, order, token, counts, top_k_weights, T, dtype):
        if self.impl == "auto":
            self.impl = _pick_impl(self, x, counts, dtype)
        if self.impl == "grouped":
            y = self._expert_outputs_grouped(x, counts, dtype)
        else:
            y = self._expert_outputs_loop(x, counts, dtype)
        weights = top_k_weights.reshape(-1)[order].to(torch.float32)
        out = torch.zeros(T, self.hidden_dim, dtype=torch.float32, device=hidden_states.device)
        out.index_add_(0, token, y.to(torch.float32) * weights[:, None])

        rec = self.recorder
        if rec is not None and rec.enabled:
            norms = y.to(torch.float32).norm(dim=-1).to(torch.float64)
            cats = rec._cats(T, hidden_states.device)[token]
            key = flat[order] * N_CATEGORIES + cats
            g = weights.to(torch.float64)
            layer = self.layer_idx
            rec.data["count"][layer].view(-1).index_add_(0, key, torch.ones_like(g))
            rec.data["gate"][layer].view(-1).index_add_(0, key, g)
            rec.data["norm"][layer].view(-1).index_add_(0, key, norms)
            rec.data["gate_norm"][layer].view(-1).index_add_(0, key, g * norms)
            n_bands = len(rec.POSITION_BANDS) + 1
            key_pos = key * n_bands + rec._bands(T, hidden_states.device)[token]
            for name, value in (("count", torch.ones_like(g)), ("gate", g), ("gate_norm", g * norms)):
                rec.data[f"{name}_pos"][layer].view(-1).index_add_(0, key_pos, value)
        return out.to(dtype)


_IMPL_CHOICE: dict[str, str] = {}


def _pick_impl(module: Int4Experts, x, counts, dtype) -> str:
    """Use torch._grouped_mm when it exists, runs on this device and matches the
    per-expert loop on this very input; otherwise loop. Decided once."""
    device_key = str(x.device)
    if device_key in _IMPL_CHOICE:
        return _IMPL_CHOICE[device_key]
    choice = "loop"
    if hasattr(torch, "_grouped_mm") and x.is_cuda:
        try:
            got = module._expert_outputs_grouped(x, counts, dtype)
            ref = module._expert_outputs_loop(x, counts, dtype)
            err = ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()
            choice = "grouped" if err < 2e-2 else "loop"
            print(f"[reap] grouped_mm check: relative error {err:.2e} -> {choice}", flush=True)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(f"[reap] grouped_mm unavailable ({type(exc).__name__}: {exc}); per-expert loop", flush=True)
    _IMPL_CHOICE[device_key] = choice
    return choice


def pruned_routing(gate, flat: torch.Tensor, keep: torch.Tensor):
    """The router of a model whose experts outside `keep` were removed: softmax
    over the kept experts only, then top-k and renormalization as before."""
    router_logits = F.linear(flat, gate.weight)
    probs = torch.softmax(router_logits.masked_fill(~keep, float("-inf")), dtype=torch.float, dim=-1)
    top, selected = torch.topk(probs, gate.top_k, dim=-1)
    if gate.norm_topk_prob:
        top = top / top.sum(dim=-1, keepdim=True)
    return router_logits, top.to(router_logits.dtype), selected


def moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Qwen4ExpTextSparseMoeBlock.forward plus the soft-probability statistic,
    and expert pruning when `self.experts.keep` (bool [E]) is set."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    shared = self.shared_expert(flat)
    keep = self.experts.keep
    if keep is None:
        router_logits, routing_weights, selected = self.gate(flat)
    else:
        router_logits, routing_weights, selected = pruned_routing(self.gate, flat, keep)
    rec = self.experts.recorder
    if rec is not None and rec.enabled:
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        cats = rec._cats(flat.shape[0], flat.device)
        onehot = F.one_hot(cats, N_CATEGORIES).to(torch.float32)  # [T, C]
        rec.data["prob"][self.experts.layer_idx] += (probs.T @ onehot).to(torch.float64)
    out = self.experts(flat, selected, routing_weights)
    out = out + torch.sigmoid(self.shared_expert_gate(flat)) * shared
    return out.reshape(batch_size, sequence_length, hidden_dim)


def set_pruning(model, keep: torch.Tensor | None):
    """keep: bool [layers, experts], or None to restore the full model."""
    for i, layer in enumerate(model.model.language_model.layers):
        experts = layer.mlp.experts
        experts.keep = None if keep is None else keep[i].to(experts.qweight_gate_up.device)


def check_fast_linear_attention(device="cuda", log=print) -> bool:
    """If the flash-linear-attention kernel is in use, compare it with the
    reference PyTorch chunked delta rule on random input; on a mismatch or an
    error, switch the model back to the reference. Returns True if fast."""
    wrapped = mq.torch_chunk_gated_delta_rule
    reference = wrapped
    while hasattr(reference, "__wrapped__"):
        reference = reference.__wrapped__
    try:
        import fla  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log(f"[reap] flash-linear-attention unavailable ({type(exc).__name__}); reference delta rule")
        return False
    torch.manual_seed(0)
    B, T, H, K, V = 1, 1000, 8, 128, 128
    q, k = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, device=device, dtype=torch.float32)
    beta = torch.rand(B, T, H, device=device, dtype=torch.bfloat16)
    state = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.1
    args = dict(g=g, beta=beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    try:
        fast, fast_state = wrapped(q, k, v, **args)
        ref, ref_state = reference(q, k, v, **args)
        err = ((fast.float() - ref.float()).norm() / ref.float().norm()).item()
        state_err = ((fast_state.float() - ref_state.float()).norm() / ref_state.float().norm()).item()
    except Exception as exc:  # noqa: BLE001
        err, state_err = float("inf"), float("inf")
        log(f"[reap] flash-linear-attention failed ({type(exc).__name__}: {exc})")
    ok = err < 2e-2 and state_err < 2e-2
    log(f"[reap] flash-linear-attention check: output error {err:.1e}, state error {state_err:.1e} -> "
        f"{'fast kernel' if ok else 'reference'}")
    if not ok:
        mq.torch_chunk_gated_delta_rule = reference
    return ok


# --------------------------------------------------------------------------- attention


def select_tokens(indexer, hidden_states, full_cos, full_sin, past_key_values, query_block: int = 1024):
    """QSA token selection for every query of the chunk: [L, S] long, -1 = empty.

    Same rule as Qwen4ExpTextQSAIndexer (causal, no padding): keys are pooled
    in blocks of `compress_ratio` tokens; a query at token position p scores
    the (p+1)//c complete blocks it can see with sum_heads relu(q . k_block),
    keeps the top `token_budget // c`, and always adds its incomplete last
    block."""
    batch, L, _ = hidden_states.shape
    if batch != 1:
        raise ValueError("indexed attention replay expects batch size 1")
    c, d = indexer.compress_ratio, indexer.index_head_dim
    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [indexer.index_n_heads * d, indexer.index_kv_heads * d], dim=-1)
    q = indexer.q_layernorm(q.reshape(batch, L, -1, d))
    q = mq.apply_rotary_pos_emb(q, cos=full_cos[:, -L:], sin=full_sin[:, -L:], unsqueeze_dim=2)
    raw = token_k.reshape(batch, L, -1, d).squeeze(2)
    if past_key_values is not None:
        raw = past_key_values.update_indexer(raw, indexer.layer_idx)
    kv = raw.shape[1]
    past = kv - L
    nb = kv // c
    device = hidden_states.device
    positions = past + torch.arange(L, device=device)
    visible_blocks = (positions + 1) // c
    width = min(indexer.block_topk, nb) * c + c - 1
    selected = torch.full((L, width), -1, dtype=torch.long, device=device)

    tail = visible_blocks[:, None] * c + torch.arange(c - 1, device=device)
    tail = torch.where(tail <= positions[:, None], tail, -1)
    if nb == 0:
        selected[:, : c - 1] = tail
        return selected

    pooled = raw[0, : nb * c].view(nb, c, d).float().mean(dim=1).to(raw.dtype)
    pooled = indexer.k_layernorm(pooled)
    starts = torch.arange(nb, device=device) * c
    block_keys = mq.apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=full_cos[0].index_select(0, starts), sin=full_sin[0].index_select(0, starts)
    ).squeeze(1).float()
    k_top = min(indexer.block_topk, nb)
    within = torch.arange(c, device=device)
    for s in range(0, L, query_block):
        e = min(L, s + query_block)
        scores = torch.zeros(e - s, nb, dtype=torch.float32, device=device)
        for h in range(q.shape[2]):
            scores += torch.relu(q[0, s:e, h].float() @ block_keys.T)
        scores /= d ** 0.5
        visible = visible_blocks[s:e, None]
        scores.masked_fill_(torch.arange(nb, device=device)[None, :] >= visible, float("-inf"))
        top = scores.topk(k_top, dim=-1).indices  # [q, k_top]
        valid = top < visible
        tokens = (top[..., None] * c + within).reshape(e - s, -1)
        tokens = torch.where(valid.repeat_interleave(c, dim=-1), tokens, -1)
        selected[s:e, : k_top * c] = tokens
        selected[s:e, k_top * c :] = tail[s:e]
    return selected


def gathered_attention(q, k, v, selected, scaling, query_block: int = 64):
    """q [Hq, L, d], k/v [Hkv, KV, d], selected [L, S] -> [L, Hq, d]."""
    with torch.profiler.record_function("reap.attention"):
        if OPTIONS["attention"] == "sdpa":
            return _gathered_attention_sdpa(q, k, v, selected, scaling)
        return _gathered_attention_einsum(q, k, v, selected, scaling, query_block)


def _gathered_attention_sdpa(q, k, v, selected, scaling, query_block: int = 256):
    """Each query is its own batch item attending to its gathered keys; GQA and
    the fp32 softmax stay inside PyTorch's attention kernel."""
    hq, L, d = q.shape
    out = torch.empty(L, hq, d, dtype=q.dtype, device=q.device)
    for s in range(0, L, query_block):
        e = min(L, s + query_block)
        idx = selected[s:e]
        valid = idx >= 0
        idx0 = idx.clamp_min(0)
        kg = k[:, idx0].transpose(0, 1)  # [q, Hkv, S, d]
        vg = v[:, idx0].transpose(0, 1)
        qs = q[:, s:e].transpose(0, 1).unsqueeze(2)  # [q, Hq, 1, d]
        o = F.scaled_dot_product_attention(qs, kg, vg, attn_mask=valid[:, None, None, :], scale=scaling,
                                           enable_gqa=True)
        out[s:e] = o.squeeze(2)
    return out


def _gathered_attention_einsum(q, k, v, selected, scaling, query_block: int = 64):
    """fp32 math on gathered keys."""
    hq, L, d = q.shape
    hkv = k.shape[0]
    rep = hq // hkv
    out = torch.empty(L, hq, d, dtype=q.dtype, device=q.device)
    for s in range(0, L, query_block):
        e = min(L, s + query_block)
        idx = selected[s:e]
        valid = idx >= 0
        idx0 = idx.clamp_min(0)
        kg = k[:, idx0].float()  # [Hkv, q, S, d]
        vg = v[:, idx0].float()
        qs = q[:, s:e].float().view(hkv, rep, e - s, d)
        scores = torch.einsum("grqd,gqsd->grqs", qs, kg) * scaling
        scores.masked_fill_(~valid[None, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("grqs,gqsd->grqd", probs, vg)
        out[s:e] = o.reshape(hq, e - s, d).transpose(0, 1).to(q.dtype)
    return out


def indexed_attention_forward(self, hidden_states, position_embeddings, attention_mask=None,
                              past_key_values=None, **kwargs):
    """Drop-in for Qwen4ExpTextAttention.forward (batch 1, causal, no padding)."""
    full_cos, full_sin = position_embeddings
    with torch.profiler.record_function("reap.indexer"):
        selected = select_tokens(self.indexer, hidden_states, full_cos, full_sin, past_key_values)
    input_shape = hidden_states.shape[:-1]
    L = input_shape[1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = full_cos[:, -L:], full_sin[:, -L:]
    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attn = gathered_attention(query_states[0], key_states[0], value_states[0], selected, self.scaling)
    attn = attn.reshape(*input_shape, -1) * torch.sigmoid(gate)
    return self.o_proj(attn), None


# --------------------------------------------------------------------------- PLE table


def safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


_NP_DTYPES = {"BF16": np.int16, "F16": np.float16, "F32": np.float32}


class MmapEmbedding(nn.Module):
    """Row-sharded embedding read from memory-mapped safetensors, returned in
    the stored dtype (BF16 is memory-mapped as int16 and reinterpreted)."""

    def __init__(self, shards: list[np.ndarray]):
        super().__init__()
        self.shards = shards
        self.offsets = np.cumsum([0] + [s.shape[0] for s in shards])
        self.embedding_dim = shards[0].shape[1]
        # the PLE code reads .weight.device to decide where to send the ids
        self.register_buffer("weight", torch.empty(0), persistent=False)

    @property
    def num_embeddings(self) -> int:
        return int(self.offsets[-1])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.profiler.record_function("reap.ple_lookup"):
            return self._lookup(ids)

    def _lookup(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy()
        shard = np.searchsorted(self.offsets, flat, side="right") - 1
        out = np.empty((flat.size, self.embedding_dim), dtype=self.shards[0].dtype)
        for s in np.unique(shard):
            rows = shard == s
            out[rows] = self.shards[s][flat[rows] - self.offsets[s]]
        tensor = torch.from_numpy(out)
        if tensor.dtype == torch.int16:
            tensor = tensor.view(torch.bfloat16)
        return tensor.reshape(*ids.shape, self.embedding_dim)


def memmap_tensor(path: Path, header: dict, data_start: int, key: str) -> np.ndarray:
    info = header[key]
    begin, end = info["data_offsets"]
    dtype = _NP_DTYPES[info["dtype"]]
    return np.memmap(path, dtype=dtype, mode="r", offset=data_start + begin, shape=tuple(info["shape"]))


# --------------------------------------------------------------------------- loading

_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales)$"
)
_NGRAM_RE = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")


def _set_tensor(model: nn.Module, name: str, tensor: torch.Tensor, device):
    module_name, _, attr = name.rpartition(".")
    module = model.get_submodule(module_name)
    if attr in module._parameters:
        module._parameters[attr] = nn.Parameter(tensor.to(device), requires_grad=False)
    elif attr in module._buffers:
        module._buffers[attr] = tensor.to(device)
    else:
        raise KeyError(name)


def load_model(model_dir, device="cuda", dtype=torch.bfloat16, record=True, log=print):
    """Build the instrumented model from an Intel-format checkpoint directory.
    Returns (model, recorder)."""
    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    quantization = getattr(config, "quantization_config", None) or {}
    if not isinstance(quantization, dict):
        quantization = quantization.to_dict()
    group_size = int(quantization.get("group_size", 128))
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config._attn_implementation = "sdpa"
    text = config.text_config

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    pending_rows = {}
    try:
        with torch.device("meta"):
            model = mq.Qwen4ExpForConditionalGeneration(config)
            lm = model.model.language_model
            for i, layer in enumerate(lm.layers):
                layer.mlp.experts = Int4Experts(text, i, None, group_size)
                if layer.ple is not None:
                    # never materialize the 51B-row table on the device
                    ple = layer.ple.ple_embedding
                    pending_rows[f"model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding"] = (
                        ple.ngram_embedding.weight.shape[0]
                    )
                    ple.ngram_embedding = nn.Identity()
    finally:
        torch.set_default_dtype(default_dtype)
    model.to_empty(device=device)
    recorder = ReapRecorder(text.num_hidden_layers, text.num_experts, device) if record else None
    for layer in lm.layers:
        layer.mlp.experts.recorder = recorder
        layer.mlp.forward = types.MethodType(moe_block_forward, layer.mlp)
        if hasattr(layer, "self_attn"):  # layer_type "full_attention" in config.json, "indexed_attention" once parsed
            layer.self_attn.forward = types.MethodType(indexed_attention_forward, layer.self_attn)

    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = defaultdict(list)
    for key, file in weight_map.items():
        by_file[file].append(key)
    loaded, ngram = set(), defaultdict(dict)
    from safetensors import safe_open

    for n_file, (file, keys) in enumerate(sorted(by_file.items())):
        path = model_dir / file
        ngram_keys = [k for k in keys if _NGRAM_RE.match(k)]
        if ngram_keys:
            header, start = safetensors_header(path)
            for key in ngram_keys:
                m = _NGRAM_RE.match(key)
                ngram[m[1]][int(m[2])] = memmap_tensor(path, header, start, key)
        rest = [k for k in keys if k not in ngram_keys and not k.startswith("mtp.")]
        if not rest:
            continue
        with safe_open(path, framework="pt", device="cpu") as fh:
            for key in rest:
                tensor = fh.get_tensor(key)
                if m := _EXPERT_RE.match(key):
                    experts = lm.layers[int(m[1])].mlp.experts
                    experts.load_part(int(m[2]), m[3], m[4], tensor.to(device))
                else:
                    _set_tensor(model, key, tensor, device)
                loaded.add(key)
        log(f"[reap] loaded {file} ({n_file + 1}/{len(by_file)})")

    for module_path, shards in ngram.items():
        ordered = [shards[i] for i in range(len(shards))]
        embedding = MmapEmbedding(ordered)
        parent_name, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_name)
        expected = pending_rows.pop(module_path)
        if embedding.num_embeddings != expected:
            raise ValueError(f"{module_path}: {embedding.num_embeddings} rows, model expects {expected}")
        setattr(parent, attr, embedding)
        loaded.update(k for k in weight_map if k.startswith(module_path + ".shard_"))

    # non-persistent buffers computed in __init__ were lost on the meta device
    for module in model.modules():
        if isinstance(module, (mq.Qwen4ExpTextRotaryEmbedding, mq.Qwen4ExpVisionRotaryEmbedding)):
            fresh = type(module)(module.config)
            for name, buffer in fresh.named_buffers(recurse=False):
                module._buffers[name] = buffer.to(device)

    if pending_rows:
        raise ValueError(f"no n-gram shards found for {sorted(pending_rows)}")
    for layer in lm.layers:
        layer.mlp.experts.check_filled()
    missing =[k for k in weight_map if k not in loaded and not k.startswith("mtp.")]
    if missing:
        raise ValueError(f"checkpoint tensors not loaded: {missing[:5]} (+{len(missing) - 5})")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            raise ValueError(f"{name} still on the meta device")
    for module in model.modules():
        for name, value in vars(module).items():
            if torch.is_tensor(value) and value.is_meta:
                raise ValueError(f"{type(module).__name__}.{name} still on the meta device")
    model.eval()
    model.requires_grad_(False)
    return model, recorder
