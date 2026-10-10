# Selection arithmetic derived from Hugging Face Transformers Qwen4Exp (Apache-2.0).
# Pinned original source: 0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206
"""Direct dense bias: explicit indexer subclass, native selection arithmetic."""
from __future__ import annotations
import math
import torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

class LazyCausalMask:
    def __init__(self,tokens,device):
        self.shape=(1,1,tokens,tokens)
        self.device=device
        self.dtype=torch.bool
        self.keys=torch.arange(tokens,device=device)

    def __getitem__(self,index):
        batch,head,query=index
        if batch!=0 or head!=0 or not isinstance(query,int):
            raise ValueError('Unexpected native causal-mask indexing')
        return self.keys<=query

    def is_floating_point(self):return False

    def __and__(self,selected_bias):
        # Native selection used only visible causal keys. Its additive bias
        # already contains exactly the final allowed set; no mask merge remains.
        if not isinstance(selected_bias,torch.Tensor) or not selected_bias.is_floating_point():
            raise ValueError('Expected the patched indexer additive bias')
        if tuple(selected_bias.shape)!=self.shape:
            raise ValueError('Unexpected native indexer bias shape')
        return selected_bias

class DirectBiasIndexer(native.Qwen4ExpTextQSAIndexer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        # The cos/sin here are the full positions for the keys, so we need to slice to get only the current positions for the queries
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        # Note that the mask is never None here as we only allow eager and sdpa, and we do not allow sdpa's mask skip
        # It's always 4D with either bool (sdpa) or float (eager) and already gives us the valid indices
        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                # Compute selected tokens
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    # Remap the indices of the blocks to the indices of individual tokens
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        kv_length = attention_mask.shape[-1]
        width = ((kv_length + 1 + 7) // 8) * 8
        bias = torch.full((*selected_token_indices.shape[:-1], width), float('-inf'),
                          device=hidden_states.device, dtype=hidden_states.dtype)
        indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        bias.scatter_(-1, indices, 0.)
        return bias[..., :kv_length].unsqueeze(1)


class DirectBiasTextModel(native.Qwen4ExpTextModel):
    def forward(self, *args, **kwargs):
        if args or kwargs.get('past_key_values') is not None or kwargs.get('use_cache'):
            raise ValueError('Direct bias requires keyword inputs and cache-free training')
        mask = kwargs.get('attention_mask')
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 2 or not bool(mask.bool().all())):
            raise ValueError('Direct bias requires an unpadded sample')
        inputs = kwargs.get('inputs_embeds')
        if inputs is None:
            inputs = kwargs.get('input_ids')
        if inputs is None or inputs.shape[0] != 1:
            raise ValueError('Direct bias requires batch one')
        kwargs['attention_mask'] = {'indexed_attention':LazyCausalMask(inputs.shape[1],inputs.device), 'linear_attention':None}
        return super().forward(**kwargs)
