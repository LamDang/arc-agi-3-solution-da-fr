# Native arithmetic: pinned Transformers Qwen4Exp, Apache-2.0 (see model.py).
"""Opt10: replay whole PLE windows with the native dilated-convolution halo."""
from functools import partial
import math
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from .precision import BF16PLE


class ChunkedPLE(BF16PLE):
    def _window(self,hidden,embeddings,mask,*,discard):
        embeddings=embeddings.to(hidden.device)
        key=self.norm_key(self.key_proj(embeddings)).unflatten(-1,(self.hc_count,self.hidden_size))
        value=self.value_proj(embeddings)
        query=self.norm_query(hidden).unflatten(-1,(self.hc_count,self.hidden_size))
        gate=(key*query).sum(dim=-1,keepdim=True)/math.sqrt(self.hidden_size)
        gate=gate.abs().clamp_min(1e-6).sqrt()*gate.sign()
        gated=torch.sigmoid(gate)*value.unsqueeze(-2)
        normed=self.norm_conv(gated.flatten(-2));gated=gated.flatten(-2)
        if mask is not None:
            gated=native.apply_mask_to_padding_states(gated,mask)
            normed=native.apply_mask_to_padding_states(normed,mask)
        output=gated+self._short_conv(normed,None)
        return output[:,discard:].bfloat16()

    def forward(self,hidden_states,input_ids,past_key_values,conv_mask=None):
        if not self.chunk_tokens:return super().forward(hidden_states,input_ids,past_key_values,conv_mask)
        embedding=self.ple_embedding
        if past_key_values is not None or not torch.equal(input_ids.detach().cpu(),embedding.prepared_input_ids):
            raise ValueError('PLE windows require a prepared full sample and no cache')
        halo=(self.conv1d.kernel_size[0]-1)*self.conv1d.dilation[0]
        if halo!=self.short_conv_state_len:raise ValueError('PLE convolution history mismatch')
        hidden=hidden_states.bfloat16();t=hidden.shape[1];outputs=[]
        for start in range(0,t,self.chunk_tokens):
            left=max(0,start-halo);stop=min(start+self.chunk_tokens,t)
            outputs.append(checkpoint(partial(self._window,discard=start-left),hidden[:,left:stop],
                embedding.prepared_payload[:,left:stop],None if conv_mask is None else conv_mask[:,left:stop],use_reentrant=False))
        self.chunk_stats=dict(halo_tokens=halo,max_window_tokens=min(t,self.chunk_tokens+halo),chunks=len(outputs),full_context_prepared_embeddings=True)
        return torch.cat(outputs,dim=1)
