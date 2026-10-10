# Native arithmetic: pinned Transformers Qwen4Exp, Apache-2.0 (see model.py).
"""Opt9: checkpoint token-local mixing and injection, around contextual blocks."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .precision import BF16Residual,BF16Decoder


class ChunkedResidual(BF16Residual):
    def _mix(self,x):
        result=super().forward(x)
        return (result[0],result[2]) if isinstance(result,tuple) else result

    def _gate_mix(self,logits,normed):
        weights=torch.sigmoid(logits).unflatten(-1,(self.hc_count,self.hidden_size))
        return (weights*normed.unflatten(-1,(self.hc_count,self.hidden_size))).mean(dim=-2)

    def _shape_preserving_mix(self,x,checkpoint_windows=True):
        # Native GEMM row counts are numerically significant in BF16. Keep all
        # three projections full-sequence; bound norm/gate/product internals.
        def apply(function,*args):
            return checkpoint(function,*args,use_reentrant=False) if checkpoint_windows else function(*args)
        normed=torch.cat([apply(self.hc_norm,x[:,s:s+self.chunk_tokens])
            for s in range(0,x.shape[1],self.chunk_tokens)],dim=1)
        lowrank=F.silu(self.input_mix_weight_down(normed)/self.hc_count)
        logits=self.input_mix_weight_up(lowrank)
        mixed=torch.cat([apply(self._gate_mix,logits[:,s:s+self.chunk_tokens],normed[:,s:s+self.chunk_tokens])
            for s in range(0,x.shape[1],self.chunk_tokens)],dim=1).bfloat16()
        if self.block_inject_weight is None:return mixed
        weights=2*torch.sigmoid(self.block_inject_weight(normed)/self.hc_count)
        return mixed,weights

    def forward(self,hyper_input):
        if not self.chunk_tokens:return super().forward(hyper_input)
        x=hyper_input.bfloat16()
        result=checkpoint(self._shape_preserving_mix,x,use_reentrant=False)
        self.chunk_stats=dict(max_tokens=min(x.shape[1],self.chunk_tokens),
            chunks=(x.shape[1]+self.chunk_tokens-1)//self.chunk_tokens,projection_tokens=x.shape[1],
            projection_policy='native full-sequence GEMM; windowed norm and gate/product')
        return (result[0],x,result[1]) if isinstance(result,tuple) else result


def inject(x,residual,weights):
    # The next residual input / final decoder output is BF16 in the reference.
    return (residual+(x.unsqueeze(-2)*weights.unsqueeze(-1)).flatten(-2)).bfloat16()


class ChunkedDecoder(BF16Decoder):
    def injection(self,x,residual,weights):
        return torch.cat([checkpoint(inject,x[:,s:s+self.chunk_tokens],residual[:,s:s+self.chunk_tokens],weights[:,s:s+self.chunk_tokens],use_reentrant=False)
            for s in range(0,x.shape[1],self.chunk_tokens)],dim=1)

    def forward(self,hidden_states,position_embeddings,attention_mask=None,conv_mask=None,past_key_values=None,ple_input_ids=None,**kwargs):
        if not self.chunk_tokens:
            return super().forward(hidden_states,position_embeddings,attention_mask,conv_mask,past_key_values,ple_input_ids,**kwargs)
        hidden_states=hidden_states.bfloat16()
        if self.ple is not None:hidden_states=hidden_states+self.ple(hidden_states,ple_input_ids,past_key_values,conv_mask=conv_mask)
        hidden_states,residual,weights=self.attn_hyper_connection(hidden_states)
        if self.layer_type=='linear_attention':
            hidden_states=self.linear_attn(hidden_states,cache_params=past_key_values,attention_mask=conv_mask,**kwargs)
        else:
            hidden_states,_=self.self_attn(hidden_states,position_embeddings,attention_mask=attention_mask,past_key_values=past_key_values,**kwargs)
        hidden_states=self.injection(hidden_states,residual,weights)
        hidden_states,residual,weights=self.mlp_hyper_connection(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=self.injection(hidden_states,residual,weights)
        self.chunk_stats=dict(max_injection_tokens=min(hidden_states.shape[1],self.chunk_tokens),chunks=(hidden_states.shape[1]+self.chunk_tokens-1)//self.chunk_tokens)
        return hidden_states
