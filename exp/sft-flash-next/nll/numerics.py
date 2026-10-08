"""Shape-stable BF16 scoring without changing weights or routing semantics.

CUDA GEMM algorithms and atomic expert sums can vary with replay chunk shape.
Use fixed padded row batches and sum experts in their original top-k order.
This is scoring-only: calibration recorders are deliberately rejected.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

ROWS = 256
POLICY = {"version": "fixed-256-projections-experts-topk-sum-v1", "rows": ROWS,
          "bf16_reduced_precision_reduction": False}


def linear(x, weight, bias=None):
    flat = x.reshape(-1, x.shape[-1])
    pieces = []
    for start in range(0, len(flat), ROWS):
        rows = flat[start:start + ROWS]
        n = len(rows)
        if n < ROWS:
            rows = F.pad(rows, (0, 0, 0, ROWS - n))
        pieces.append(F.linear(rows, weight, bias)[:n])
    if not pieces:
        return F.linear(x, weight, bias)
    return torch.cat(pieces).reshape(*x.shape[:-1], weight.shape[0])


def linear_forward(self, x):
    return linear(x, self.weight, self.bias)


def routing(gate, flat, keep=None):
    logits = linear(flat.reshape(-1, gate.hidden_dim), gate.weight)
    masked = logits if keep is None else logits.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(masked, dtype=torch.float, dim=-1)
    top, selected = torch.topk(probs, gate.top_k, dim=-1)
    if gate.norm_topk_prob:
        top = top / top.sum(dim=-1, keepdim=True)
    return logits, top.to(logits.dtype), selected


def router_forward(self, hidden_states):
    return routing(self, hidden_states)


def expert_outputs(self, x, counts, dtype):
    y = torch.empty_like(x)
    start = 0
    for expert, n in enumerate(counts.tolist()):
        if not n:
            continue
        w1 = self._nll_dequantize(self.qweight_gate_up[expert], self.scales_gate_up[expert], dtype)
        w2 = self._nll_dequantize(self.qweight_down[expert], self.scales_down[expert], dtype)
        for offset in range(0, n, ROWS):
            rows = x[start + offset:start + min(offset + ROWS, n)]
            nr = len(rows)
            if nr < ROWS:
                rows = F.pad(rows, (0, 0, 0, ROWS - nr))
            gate, up = (rows @ w1).chunk(2, dim=-1)
            y[start + offset:start + offset + nr] = ((self.act_fn(gate) * up) @ w2)[:nr]
        start += n
    return y


def expert_sum(self, hidden_states, x, flat, order, token, counts, weights, total, dtype):
    y = self._expert_outputs_loop(x, counts, dtype)
    unsorted = torch.empty_like(y)
    unsorted[order] = y
    return (unsorted.reshape(total, weights.shape[1], self.hidden_dim).float()
            * weights.float().unsqueeze(-1)).sum(dim=1).to(dtype)


def configure(model, reap_model):
    for layer in model.model.language_model.layers:
        if layer.mlp.experts.recorder is not None:
            raise ValueError("Stable NLL scoring cannot be used for calibration recording")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(linear_forward, module)
    for layer in model.model.language_model.layers:
        layer.mlp.gate.forward = types.MethodType(router_forward, layer.mlp.gate)
        experts = layer.mlp.experts
        experts._nll_dequantize = reap_model.dequantize_gptq
        experts._expert_outputs_loop = types.MethodType(expert_outputs, experts)
        experts._forward_sorted = types.MethodType(expert_sum, experts)
        experts.impl = "loop"
    reap_model.pruned_routing = routing
