# Expert loop derived from AutoRound 0.15.0 (Apache-2.0), pinned original SHA256:
# 2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c
"""Shared/routed SwiGLU components with explicit expert dispatch."""
import logging
import torch
from torch import nn
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
logger = logging.getLogger(__name__)
LINEAR_LOOP_IMPL = 'linear_loop'


def fused_swiglu(gate, up):
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError('Projection tensor metadata differs')
    return LigerSiLUMulFunction.apply(gate, up)


class LigerMLP(native.Qwen4ExpTextMLP):
    def forward(self, x):
        return self.down_proj(fused_swiglu(self.gate_proj(x), self.up_proj(x)))


class LigerExperts(native.Qwen4ExpTextExperts):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return linear_loop_experts_forward(self, hidden_states, top_k_index, top_k_weights)


def linear_loop_experts_forward(self: nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
    """Forward using individual nn.Linear layers per expert.

    This implementation loops over experts and accesses per-expert containers
    (self._modules["0"], self._modules["1"], ...) each with gate_proj,
    up_proj, and down_proj as nn.Linear layers (or quantized equivalents),
    enabling proper quantization support.

    Expected module structure:
        - Numbered children (0, 1, ..., num_experts-1), each an _ExpertContainer with:
            - gate_proj: nn.Linear (in_features=hidden_dim, out_features=intermediate_dim)
            - up_proj: nn.Linear (in_features=hidden_dim, out_features=intermediate_dim)
            - down_proj: nn.Linear (in_features=intermediate_dim, out_features=hidden_dim)
        - act_fn: activation function
        - num_experts: number of experts
        - _apply_gate: optional custom gating function

    Args:
        self: The experts module
        hidden_states: Input tensor of shape (num_tokens, hidden_dim)
        top_k_index: Selected expert indices of shape (num_tokens, top_k)
        top_k_weights: Expert weights of shape (num_tokens, top_k)

    Returns:
        final_hidden_states: Output tensor of shape (num_tokens, hidden_dim)
    """
    logger.debug(f'Using {LINEAR_LOOP_IMPL} experts forward for {self.__class__.__name__}')
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = (None, None)
        hidden_dim = hidden_states.size(-1)
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.num_experts
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1)
    selected_hidden_states = hidden_states[token_idx]
    sort_order = torch.argsort(expert_ids)
    permuted_hidden_states = selected_hidden_states.index_select(0, sort_order)
    counts = torch.bincount(expert_ids, minlength=num_experts).tolist()
    out_permuted = torch.zeros_like(permuted_hidden_states)
    start = 0
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        end = start + count
        expert_input = permuted_hidden_states[start:end]
        expert = getattr(self, str(expert_idx))
        gate_out = expert.gate_proj(expert_input)
        up_out = expert.up_proj(expert_input)
        gated_out = fused_swiglu(gate_out, up_out)
        expert_out = expert.down_proj(gated_out)
        out_permuted[start:end] = expert_out.to(out_permuted.dtype)
        start = end
    out_per_sample = torch.empty_like(out_permuted)
    out_per_sample.index_copy_(0, sort_order, out_permuted)
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states
