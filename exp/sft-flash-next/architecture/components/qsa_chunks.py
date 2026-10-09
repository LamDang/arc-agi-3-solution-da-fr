# Native arithmetic: pinned Transformers Qwen4Exp, Apache-2.0 (see model.py).
"""Opt8: full-context K/V; construct selection bias inside each query replay."""
from functools import partial
import math
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from .precision import BF16Attention
from .attention import DirectBiasIndexer


class WindowIndexer(DirectBiasIndexer):
    @torch.no_grad()
    def project(self,hidden,positions):
        b,t,_=hidden.shape
        q,k=torch.split(self.index_qk_proj(hidden),
            [self.index_n_heads*self.index_head_dim,self.index_kv_heads*self.index_head_dim],dim=-1)
        q=self.q_layernorm(q.reshape(b,t,-1,self.index_head_dim))
        return native.apply_rotary_pos_emb(q,cos=positions[0],sin=positions[1],unsqueeze_dim=2),k.reshape(b,t,-1,self.index_head_dim).squeeze(2)

    @torch.no_grad()
    def bias(self,q,raw_keys,positions,start,dtype):
        b,n=q.shape[:2];tokens=raw_keys.shape[1];cos,sin=positions
        selected=torch.full((b,n,self.token_budget+self.compress_ratio-1),-1,dtype=torch.int32,device=q.device)
        for batch in range(b):
            for local in range(n):
                # Exactly the unpadded native causal row, using absolute position.
                visible=torch.arange(start+local+1,device=q.device)
                blocks=visible.numel()//self.compress_ratio
                if blocks:
                    indices=visible[:blocks*self.compress_ratio].view(blocks,self.compress_ratio)
                    groups=raw_keys[batch].index_select(0,indices.flatten()).view(*indices.shape,self.index_head_dim)
                    pooled=self.k_layernorm(groups.float().mean(dim=1).to(raw_keys.dtype))
                    begins=indices[:,0]
                    keys=native.apply_rotary_pos_emb(pooled.unsqueeze(1),cos=cos[batch].index_select(0,begins),sin=sin[batch].index_select(0,begins)).squeeze(1)
                    scores=torch.matmul(q[batch,local].float(),keys.float().transpose(-1,-2)).transpose(-1,-2)
                    scores=torch.relu(scores).sum(dim=-1)/math.sqrt(self.index_head_dim)
                    choice=scores.topk(min(self.block_topk,blocks),dim=0).indices
                    chosen=indices.index_select(0,choice).flatten()
                else:chosen=torch.tensor([],device=q.device)
                chosen=torch.cat([chosen,visible[blocks*self.compress_ratio:]]).to(torch.int32)
                selected[batch,local,:chosen.numel()]=chosen
        width=((tokens+1+7)//8)*8
        bias=torch.full((b,n,width),float('-inf'),device=q.device,dtype=dtype)
        bias.scatter_(-1,torch.where(selected>=0,selected,tokens),0.)
        return bias[...,:tokens].unsqueeze(1)


class ChunkedAttention(BF16Attention):
    def _query(self,hidden,k,v,iq,ik,cos,sin,*,start):
        b,t,_=hidden.shape;shape=(b,t,-1,self.head_dim)
        q,gate=torch.chunk(self.q_proj(hidden).view(b,t,-1,self.head_dim*2),2,dim=-1)
        gate=gate.reshape(b,t,-1)
        q=self.q_norm(q.reshape(shape)).transpose(1,2)
        q=native.apply_rotary_pos_emb(q,cos=cos[:,start:start+t],sin=sin[:,start:start+t])
        bias=self.indexer.bias(iq,ik,(cos,sin),start,hidden.dtype)
        interface=native.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,native.eager_attention_forward)
        out,_=interface(self,q,k,v,bias,dropout=0.,scaling=self.scaling)
        out=out.reshape(b,t,-1).contiguous()*torch.sigmoid(gate)
        return self.o_proj(out).bfloat16()

    def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
        if not self.chunk_tokens:
            return super().forward(hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
        if past_key_values is not None or self.attention_dropout or kwargs.get('output_attentions'):
            raise ValueError('Query chunks require cache-free SDPA training, zero dropout, no attention output')
        hidden=hidden_states.bfloat16();b,t,_=hidden.shape
        cos,sin=position_embeddings
        k=self.k_norm(self.k_proj(hidden).view(b,t,-1,self.head_dim)).transpose(1,2)
        k=native.apply_rotary_pos_emb(k,cos=cos,sin=sin)
        v=self.v_proj(hidden).view(b,t,-1,self.head_dim).transpose(1,2)
        iq,ik=self.indexer.project(hidden,(cos,sin))
        outputs=[]
        for start in range(0,t,self.chunk_tokens):
            stop=min(t,start+self.chunk_tokens)
            outputs.append(checkpoint(partial(self._query,start=start),hidden[:,start:stop],k,v,iq[:,start:stop],ik,cos,sin,use_reentrant=False))
        self.chunk_stats=dict(max_query_tokens=min(t,self.chunk_tokens),key_tokens=t,
            max_bias_payload_bytes=b*min(t,self.chunk_tokens)*(((t+1+7)//8)*8)*hidden.element_size(),chunks=len(outputs))
        return torch.cat(outputs,dim=1),None
