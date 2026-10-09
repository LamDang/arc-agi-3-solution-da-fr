# Native arithmetic: pinned Transformers Qwen4Exp, Apache-2.0 (see model.py).
"""Opt9: checkpoint token-local mixing and injection, around contextual blocks."""
import torch
from torch.utils.checkpoint import checkpoint
from .precision import BF16Residual,BF16Decoder


class ChunkedResidual(BF16Residual):
    def _mix(self,x):
        result=super().forward(x)
        return (result[0],result[2]) if isinstance(result,tuple) else result

    def forward(self,hyper_input):
        if not self.chunk_tokens:return super().forward(hyper_input)
        x=hyper_input.bfloat16();outputs=[];weights=[]
        for start in range(0,x.shape[1],self.chunk_tokens):
            result=checkpoint(self._mix,x[:,start:start+self.chunk_tokens],use_reentrant=False)
            if self.block_inject_weight is None:outputs.append(result)
            else:outputs.append(result[0]);weights.append(result[1])
        self.chunk_stats=dict(max_tokens=min(x.shape[1],self.chunk_tokens),chunks=len(outputs))
        mixed=torch.cat(outputs,dim=1)
        return (mixed,x,torch.cat(weights,dim=1)) if weights else mixed


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
