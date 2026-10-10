# Native arithmetic: pinned Transformers Qwen4Exp, Apache-2.0 (see model.py).
"""Opt8: full-context K/V; construct selection bias inside each query replay."""
import math
import torch
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
    def select(self,q,raw_keys,positions,start):
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
        return selected

    @staticmethod
    def make_bias(selected,tokens,dtype):
        b,n=selected.shape[:2]
        width=((tokens+1+7)//8)*8
        bias=torch.full((b,n,width),float('-inf'),device=selected.device,dtype=dtype)
        bias.scatter_(-1,torch.where(selected>=0,selected,tokens),0.)
        return bias[...,:tokens].unsqueeze(1)

    def bias(self,q,raw_keys,positions,start,dtype):
        return self.make_bias(self.select(q,raw_keys,positions,start),raw_keys.shape[1],dtype)


class _QueryChunks(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx,q,k,v,selected,module):
        ctx.module=module;ctx.size=module.chunk_tokens
        ctx.save_for_backward(q,k,v,selected)
        output=torch.empty((q.shape[0],q.shape[2],q.shape[1],q.shape[3]),device=q.device,dtype=q.dtype)
        for start in range(0,q.shape[2],ctx.size):
            stop=min(start+ctx.size,q.shape[2])
            output[:,start:stop]=module.attend(q[:,:,start:stop],k,v,selected[:,start:stop])
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx,grad_output):
        q,k,v,selected=ctx.saved_tensors
        dq=torch.empty_like(q);dk=torch.zeros_like(k,dtype=torch.float32);dv=torch.zeros_like(v,dtype=torch.float32)
        for start in range(0,q.shape[2],ctx.size):
            stop=min(start+ctx.size,q.shape[2])
            with torch.enable_grad(),torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t):
                a=q[:,:,start:stop].detach().requires_grad_(True)
                b=k.detach().requires_grad_(True);c=v.detach().requires_grad_(True)
                output=ctx.module.attend(a,b,c,selected[:,start:stop])
                ga,gb,gc=torch.autograd.grad(output,(a,b,c),grad_output[:,start:stop])
            dq[:,:,start:stop]=ga;dk.add_(gb.float());dv.add_(gc.float())
            del output,a,b,c,ga,gb,gc
        return dq,dk.to(k.dtype),dv.to(v.dtype),None,None


class ChunkedAttention(BF16Attention):
    def attend(self,q,k,v,selected):
        # Construct and discard only these query rows, in forward and replay.
        bias=self.indexer.make_bias(selected,k.shape[2],q.dtype)
        interface=native.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,native.eager_attention_forward)
        out,_=interface(self,q,k,v,bias,dropout=0.,scaling=self.scaling)
        return out

    def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
        if not self.chunk_tokens:
            return super().forward(hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
        if past_key_values is not None or self.attention_dropout or kwargs.get('output_attentions'):
            raise ValueError('Query chunks require cache-free SDPA training, zero dropout, no attention output')
        hidden=hidden_states.bfloat16();b,t,_=hidden.shape;shape=(b,t,-1,self.head_dim)
        cos,sin=position_embeddings
        # Preserve native full-sequence projection GEMM shapes. Opt8 chunks
        # selection bias + attention, not the projection layers.
        q,gate=torch.chunk(self.q_proj(hidden).view(b,t,-1,self.head_dim*2),2,dim=-1)
        gate=gate.reshape(b,t,-1)
        q=self.q_norm(q.reshape(shape)).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden).view(shape)).transpose(1,2)
        v=self.v_proj(hidden).view(shape).transpose(1,2)
        q,k=native.apply_rotary_pos_emb(q,k,cos,sin)
        iq,ik=self.indexer.project(hidden,(cos,sin))
        selected=self.indexer.select(iq,ik,(cos,sin),0)
        del iq,ik
        out=_QueryChunks.apply(q,k,v,selected,self)
        out=out.reshape(b,t,-1).contiguous()*torch.sigmoid(gate)
        self.chunk_stats=dict(max_query_tokens=min(t,self.chunk_tokens),key_tokens=t,
            max_bias_payload_bytes=b*min(t,self.chunk_tokens)*(((t+1+7)//8)*8)*hidden.element_size(),
            chunks=(t+self.chunk_tokens-1)//self.chunk_tokens,kv_gradient_accumulator_dtype='torch.float32',native_projection_shapes=True,
            selected_index_bytes=selected.numel()*selected.element_size(),selection_reused_in_attention_backward=True)
        return self.o_proj(out).bfloat16(),None
