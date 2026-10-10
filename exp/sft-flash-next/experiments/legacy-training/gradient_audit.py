"""Real-model gradient ablations against ordinary-autograd W4A16 reference.

The reference retains the repository's pre-existing quantized loader and QSA
indexer. It uses native HF pointwise/GDN/PLE operations, freshly dequantized
experts with ordinary autograd, BF16 PyTorch SDPA, and ordinary target CE.
The primary reference uses a full masked SDPA call per layer; stock layer
checkpointing is recorded explicitly. Tiled SDPA is a separate diagnostic. It is NOT an untouched stock-Transformers run.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import backend as b
import kernel_checks
from dataset import file_hash, write_json
from monitor import MemoryMonitor, host_memory
from offload import ActivationOffload

ORIGINAL_SPARSE = b.SparseAttention


class ReferenceAttention:
    """Ordinary autograd, not a custom backward; checkpoint only gathered tiles."""
    precision = 'bf16'

    @staticmethod
    def apply(q, k, v, selected, scale, block):
        dtype = q.dtype
        if ReferenceAttention.precision == 'fp32':
            # Cast once: all tile contributions accumulate in FP32 before the
            # single input cast backward, rather than rounding every tile.
            q, k, v = q.float(), k.float(), v.float()
        def local(qs, keys, values, indices):
            idx = indices.clamp_min(0)
            kg, vg = keys[:,idx].transpose(0,1), values[:,idx].transpose(0,1)
            out = F.scaled_dot_product_attention(qs.transpose(0,1).unsqueeze(2),kg,vg,
                attn_mask=(indices>=0)[:,None,None,:],scale=scale,enable_gqa=True)
            return out.squeeze(2).transpose(0,1).to(dtype)
        return torch.cat([checkpoint(local,qs,k,v,idx,use_reentrant=False)
                          for qs,idx in zip(q.split(block,dim=1),selected.split(block,dim=0))],dim=1)


class DenseReferenceAttention:
    """Full masked PyTorch SDPA with HF's repeated-KV GQA convention."""
    @staticmethod
    def apply(q, k, v, selected, scale, block):
        from transformers.integrations.sdpa_attention import repeat_kv
        dtype = q.dtype
        if ReferenceAttention.precision == 'fp32':
            q, k, v = q.float(), k.float(), v.float()
        n = q.shape[1]
        # Efficient SDPA requires mask strides aligned to eight elements.
        # Only padding the storage (not the visible tensor) changes no tokens.
        with torch.no_grad():
            mask = torch.full((n, ((n+7)//8)*8), float('-inf'), device=q.device, dtype=q.dtype)[:, :n]
            mask.scatter_(1, selected.clamp_min(0), 0.)
            # Padding indices were clamped to zero; restore its real membership.
            mask[:, 0] = torch.where((selected == 0).any(-1), 0., float('-inf'))
        repeats = q.shape[0] // k.shape[0]
        keys, values = repeat_kv(k[None], repeats), repeat_kv(v[None], repeats)
        return F.scaled_dot_product_attention(q[None], keys, values,
            attn_mask=mask[None,None], scale=scale)[0].to(dtype)


def reference_attention(self, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
    assert past_key_values is None
    cos,sin = position_embeddings
    with torch.no_grad():
        selected = b.rm.select_tokens(self.indexer,hidden_states,cos,sin,None,
                                      query_block=self.index_block)
    shape = hidden_states.shape[:-1]
    hshape = (*shape,-1,self.head_dim)
    q,gate = self.q_proj(hidden_states).view(*shape,-1,self.head_dim*2).chunk(2,-1)
    q = self.q_norm(q.reshape(hshape)).transpose(1,2)
    k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1,2)
    v = self.v_proj(hidden_states).view(hshape).transpose(1,2)
    q,k = b.rm.mq.apply_rotary_pos_emb(q,k,cos,sin)
    kernel = ReferenceAttention if self.train_reference_gathered else DenseReferenceAttention
    out = kernel.apply(q[0],k[0],v[0],selected,self.scaling,self.query_block)
    out = out.transpose(0,1).reshape(*shape,-1) * gate.reshape(*shape,-1).sigmoid()
    return self.o_proj(out),None


def reference_experts(self, hidden, indices, weights):
    # No inference workspace and no FrozenExperts custom backward. Each expert
    # owns its dequantized matrices until ordinary autograd releases them.
    out = torch.zeros_like(hidden)
    for expert in range(self.num_experts):
        rows,slots = torch.where(indices==expert)
        if not len(rows):continue
        w1,w2 = b.expert_weights(self,expert,hidden.dtype)
        gate,up = F.linear(hidden[rows],w1.T).chunk(2,dim=-1)
        value = F.linear(self.act_fn(gate)*up,w2.T)
        out.index_add_(0,rows,value*weights[rows,slots,None])
    return out


def gradient_metrics(actual, reference):
    rows=[]; totals=dict(error_squared=0.,reference_squared=0.,actual_squared=0.,dot=0.,elements=0,different=0)
    for name,r in reference.items():
        g=actual[name]
        if g.shape!=r.shape or not torch.isfinite(g).all():raise ValueError('Invalid gradient: '+name)
        gd,rd=g.double(),r.double();delta=gd-rd
        e=float(delta.square().sum());rn=float(rd.square().sum());gn=float(gd.square().sum());dot=float((gd*rd).sum())
        different=int(torch.count_nonzero(g!=r));denom=math.sqrt(rn*gn)
        rows.append(dict(name=name,relative_l2=math.sqrt(e/max(rn,1e-300)),max_absolute=float(delta.abs().max()),
            reference_norm=math.sqrt(rn),actual_norm=math.sqrt(gn),cosine=dot/denom if denom else None,
            error_squared=e,reference_squared=rn,bitwise_equal=different==0,different_elements=different,elements=g.numel()))
        for key,value in [('error_squared',e),('reference_squared',rn),('actual_squared',gn),('dot',dot),('different',different),('elements',g.numel())]:totals[key]+=value
    denominator=math.sqrt(totals['reference_squared']*totals['actual_squared'])
    return dict(relative_l2=math.sqrt(totals['error_squared']/max(totals['reference_squared'],1e-300)),
        cosine=totals['dot']/denominator if denominator else None,bitwise_equal=totals['different']==0,
        different_fraction=totals['different']/totals['elements'],
        max_parameter_relative_l2=max(r['relative_l2'] for r in rows),
        zero_reference_parameters=sum(r['reference_norm']==0 for r in rows),per_parameter=rows)


def default_cases():
    cases=[dict(name='native_no_layer_checkpoint',layer_checkpoint=False),dict(name='reference'),
           dict(name='reference_repeat_1'),dict(name='reference_repeat_2')]
    for name,flags in [
        ('loss_chunk_128',dict(loss_block=128)),
        ('expert_recompute_8192',dict(expert_block=8192)),
        ('rms_rows_1024',dict(norm_block=1024)),
        ('rms_adaptive_40mib',dict(norm_block=1024,rms_block_mib=40)),
        ('hyper_block_1024',dict(hyper_block=1024)),
        ('gated_norm_262144',dict(gated_norm_block=262144)),
        ('ple_window_8192',dict(ple_block=8192)),
        ('gdn_segments_8192',dict(gdn_chunk_tokens=8192)),
        ('gdn_whole_blocks_32768',dict(gdn_block_tokens=32768)),
        ('attention_projection_32768',dict(attention='reference_projected',attention_projection_block=32768)),
        ('indexer_queries_256',dict(index_block=256)),
        ('gathered_attention_bf16_32',dict(reference_gathered=True,query_block=32)),
        ('gathered_attention_bf16_16',dict(reference_gathered=True,query_block=16)),
        ('reference_attention_fp32',dict(reference_precision='fp32')),
        ('training_attention_sdpa',dict(attention='sdpa')),
        ('training_attention_triton',dict(attention='triton')),
        ('checkpoint_groups_3',dict(checkpoint_group=3)),
        ('ple_checkpoint',dict(ple_checkpoint=True)),
        ('gdn_checkpoint',dict(gdn_checkpoint=True)),
        ('ple_unique_parallel',dict(ple_unique=True)),
        ('ple_request_cache',dict(ple_request_cache=True)),
        ('cpu_offload',dict(offload='cpu')),
        ('disk_offload_prefetch',dict(offload='disk')),
        ('ple_resident',dict(ple_resident=True)),
    ]:cases.append(dict(name=name,**flags))
    combined=dict(loss_block=128,expert_block=8192,norm_block=1024,rms_block_mib=40,hyper_block=1024,
        gated_norm_block=262144,ple_block=8192,gdn_chunk_tokens=8192,attention='triton',
        attention_projection_block=32768,index_block=256,query_block=16,checkpoint_group=3,
        ple_unique=True,ple_request_cache=True,ple_resident=True,offload='cpu')
    cases += [dict(name='combined',**combined),dict(name='combined_repeat',**combined),
              dict(name='combined_low_memory',**dict(combined,gdn_chunk_tokens=0,gdn_block_tokens=32768)),
              dict(name='reference_after_matrix')]
    return cases


def apply_recipe(model, flags):
    mq=b.rm.mq
    b.SparseAttention=DenseReferenceAttention if flags.get('attention')=='reference_projected' else ORIGINAL_SPARSE
    ReferenceAttention.precision=flags.get('reference_precision','bf16')
    model.train_checkpoint_group=flags.get('checkpoint_group',1)
    if flags.get('layer_checkpoint',True):model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.gradient_checkpointing_disable()
    native_types=(mq.Qwen4ExpTextRMSNorm,mq.Qwen4ExpTextGatedResidual,mq.Qwen4ExpTextPLELayer,
                  mq.Qwen4ExpTextGatedDeltaNet,mq.Qwen4ExpTextRMSNormGated)
    for m in model.model.language_model.modules():
        if isinstance(m,native_types):m.forward=types.MethodType(type(m).forward,m)
    for m in model.model.language_model.modules():
        if isinstance(m,mq.Qwen4ExpTextRMSNorm) and flags.get('norm_block'):
            m.train_block=max(1,int(flags['rms_block_mib']*2**20)//(4*m.weight.numel())) if flags.get('rms_block_mib') else flags['norm_block']
            m.forward=types.MethodType(b.training_rms_norm,m)
        if isinstance(m,mq.Qwen4ExpTextGatedResidual) and flags.get('hyper_block'):
            m.unblocked_forward=m.forward;m.train_block=flags['hyper_block'];m.forward=types.MethodType(b.training_hyper_mix,m)
        if isinstance(m,mq.Qwen4ExpTextRMSNormGated) and flags.get('gated_norm_block'):
            m.train_gated_block=flags['gated_norm_block'];m.forward=types.MethodType(b.training_gated_rms_norm,m)
    for m in model.modules():
        if isinstance(m,b.rm.MmapEmbedding):
            m._lookup=types.MethodType(b.unique_mmap_lookup if flags.get('ple_unique') else type(m)._lookup,m)
            m.read_workers=8 if flags.get('ple_unique') else 1
            if flags.get('ple_resident'):b.materialize_ple(m)
    for layer in model.model.language_model.layers:
        e=layer.mlp.experts;e.train_block=flags.get('expert_block',8192)
        e.forward=types.MethodType(b.training_experts if flags.get('expert_block') else reference_experts,e)
        if hasattr(layer,'self_attn'):
            a=layer.self_attn;a.train_reference_gathered=flags.get('reference_gathered',False);a.index_block=flags.get('index_block',1024);a.query_block=flags.get('query_block',32)
            a.key_block=32;a.train_projection_block=flags.get('attention_projection_block',0)
            a.train_attention_backend='triton' if flags.get('attention')=='triton' else 'sdpa'
            a.forward=types.MethodType(b.training_attention if flags.get('attention') else reference_attention,a)
        if layer.ple is not None:
            p=layer.ple
            if flags.get('ple_block'):
                p.train_ple_block=flags['ple_block'];p.forward=types.MethodType(b.training_ple,p)
            if flags.get('ple_checkpoint'):
                p.uncheckpointed_forward=p.forward;p.forward=types.MethodType(b.checkpointed_ple,p)
        if hasattr(layer,'linear_attn'):
            g=layer.linear_attn
            if flags.get('gdn_chunk_tokens'):
                g.train_gdn_chunk_tokens=flags['gdn_chunk_tokens'];g.unsegmented_forward=g.forward
                g.forward=types.MethodType(b.training_segmented_gdn,g)
            if flags.get('gdn_block_tokens'):
                g.train_gdn_block_tokens=flags['gdn_block_tokens'];g.unblocked_gdn_forward=g.forward
                g.forward=types.MethodType(b.training_blocked_gdn,g)
            if flags.get('gdn_checkpoint'):
                g.uncheckpointed_gdn_forward=g.forward;g.forward=types.MethodType(b.checkpointed_gdn,g)


def compare_routes(current, reference):
    changed=total=replaced=choices=0
    for key,r in reference.items():
        a=current[key]
        changed+=int((a!=r).any(-1).sum());total+=len(r)
        replaced+=int(((a>=0)&~(a[...,None]==r[:,None,:]).any(-1)).sum());choices+=int((r>=0).sum())
    return dict(changed_sets=changed,total_sets=total,changed_set_fraction=changed/max(total,1),
                replaced_choices=replaced,total_choices=choices,replaced_choice_fraction=replaced/max(choices,1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--keep',required=True)
    parser.add_argument('--sample',required=True);parser.add_argument('--sample-metadata',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--cases')
    parser.add_argument('--seed',type=int,default=20261009)
    args=parser.parse_args();out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    metadata=json.loads(Path(args.sample_metadata).read_text());enc=torch.load(args.sample,map_location='cpu',weights_only=True)
    assert file_hash(args.sample)==metadata['sample_sha256']
    prompt=metadata['row']['prompt_tokens'];tokens=metadata['row']['total_tokens'];targets=tokens-prompt
    cases=json.loads(Path(args.cases).read_text()) if args.cases else default_cases()
    write_json(out/'cases.json',cases)
    write_json(out/'identity.json',dict(sample=metadata,seed=args.seed,adapters='All A and B initialized nonzero N(0,0.003); no optimizer updates',
        reference='Existing W4A16 loader/vectorized QSA indexer; native HF pointwise/GDN/PLE; ordinary-autograd dequantized experts, BF16 SDPA and CE',
        attention_memory_control='Full masked native SDPA with repeated KV as in HF; query-gather checkpoints only in explicit gathered diagnostics',
        layer_checkpoint_control='First attempt disables stock checkpointing; canonical reference enables non-reentrant layer checkpointing',
        source_hashes={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},keep_sha256=file_hash(args.keep)))
    write_json(out/'kernel-checks.json',kernel_checks.check(attention_backend='triton'))
    b.install_segmented_gdn();torch.manual_seed(args.seed)
    model,_=b.load_pruned(args.model,args.keep)
    b.configure(model,rank=16,alpha=32,norm_block=0,hyper_block=0,ple_cache_gib=0,attention_backend='sdpa')
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.normal_(0,.003)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    torch.save(b.adapter_state(model),out/'initial-adapter.pt')
    ids,embedded,positions=b.embeddings(model,enc,'cuda');del enc
    cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state_all()
    ctx=dict(routes={},selected={},phase='idle',case=None);handles=[]
    samples=sorted({0,1,8191,8192,16383,16384,32767,32768,prompt-1,prompt,tokens-1}&set(range(tokens)))
    for i,layer in enumerate(model.model.language_model.layers):
        def hook(module,inputs,output,i=i):
            if i not in ctx['routes']:ctx['routes'][i]=output[2].detach().sort(-1).values.to(device='cpu',dtype=torch.int16)
        handles.append(layer.mlp.gate.register_forward_hook(hook))
        def progress(module,inputs,i=i):
            if i in (0,12,24,36,47):print(json.dumps(dict(event='layer',case=ctx['case'],phase=ctx['phase'],layer=i)),flush=True)
        handles.append(layer.register_forward_pre_hook(progress))
    original_select=b.rm.select_tokens
    def record_select(indexer,*values,**kw):
        selected=original_select(indexer,*values,**kw)
        if indexer.layer_idx not in ctx['selected']:
            ctx['selected'][indexer.layer_idx]=selected[samples].detach().sort(-1).values.to(device='cpu',dtype=torch.int32)
        return selected
    b.rm.select_tokens=record_select
    reference=None;reference_routes=None;reference_selected=None;reference_loss=None;reports=[]
    def execute(flags):
        apply_recipe(model,flags);model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        ctx.update(routes={},selected={},phase='forward',case=flags['name'])
        offload=flags.get('offload');disk=offload=='disk'
        storage=ActivationOffload(model,disk_dir='/tmp/gradient-audit-activations' if disk else None,disk_budget_gib=32 if disk else 0,prefetch=2) if offload else nullcontext()
        cache=b.request_ple_cache(model) if flags.get('ple_request_cache') else nullcontext()
        started=time.monotonic()
        with cache,storage:
            hidden=b.language_hidden(model,ids,embedded,positions)
            selected=hidden[0,prompt-1:-1]
            if flags.get('loss_block'):
                loss=b.SelectedCrossEntropy.apply(selected,ids[0,prompt:],model.lm_head.weight,flags['loss_block'])/targets
            else:
                logits=F.linear(selected,model.lm_head.weight).float()
                loss=F.cross_entropy(logits,ids[0,prompt:],reduction='mean');del logits
            torch.cuda.synchronize();forward=time.monotonic()-started
            ctx['phase']='backward';print(json.dumps(dict(event='forward_complete',case=flags['name'],seconds=forward,loss=float(loss.detach()))),flush=True)
            loss.backward();torch.cuda.synchronize();elapsed=time.monotonic()-started
        gradients={}
        for name,p in parameters:
            if p.grad is None:raise ValueError('Missing gradient: '+name)
            gradients[name]=p.grad.detach().float().cpu().clone()
        result=dict(name=flags['name'],flags=flags,completed=True,loss=float(loss.detach()),seconds=elapsed,forward_seconds=forward,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            host=host_memory(),resident_ple_bytes=sum(getattr(m,'ple_resident_bytes',0) for m in model.modules()),
            offload_stats=storage.stats if offload else None)
        return result,gradients
    with MemoryMonitor(out,160):
        for flags in cases:
            print('CASE_START',json.dumps(flags),flush=True)
            try:
                report,gradients=execute(flags)
                if flags['name']=='reference':
                    reference=gradients;reference_routes=dict(ctx['routes']);reference_selected=dict(ctx['selected']);reference_loss=report['loss']
                    torch.save(dict(gradients=reference,routes=reference_routes,selected=reference_selected),out/'reference-gradients.pt')
                    previous=out/'native_no_layer_checkpoint-gradients.pt'
                    if previous.exists():
                        native=torch.load(previous,map_location='cpu',weights_only=True)
                        metrics=gradient_metrics(native['gradients'],reference)
                        write_json(out/'native_no_layer_checkpoint-parameters.json',metrics)
                        reports[0]['gradient']={k:v for k,v in metrics.items() if k!='per_parameter'}
                        reports[0]['routes']=compare_routes(native['routes'],reference_routes)
                        reports[0]['loss_difference']=reports[0]['loss']-reference_loss
                        del native
                if reference is not None:
                    metrics=gradient_metrics(gradients,reference);write_json(out/(flags['name']+'-parameters.json'),metrics)
                    report['gradient']={k:v for k,v in metrics.items() if k!='per_parameter'}
                    report['loss_difference']=report['loss']-reference_loss
                    report['routes']=compare_routes(ctx['routes'],reference_routes)
                    report['sampled_attention_keys']=compare_routes(ctx['selected'],reference_selected)
                if flags['name'] in ('native_no_layer_checkpoint','reference_repeat_1','combined','combined_low_memory'):
                    torch.save(dict(gradients=gradients,routes=ctx['routes'],selected=ctx['selected']),out/(flags['name']+'-gradients.pt'))
                del gradients
            except torch.OutOfMemoryError as e:
                report=dict(name=flags['name'],flags=flags,completed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
                if flags['name']=='reference':
                    write_json(out/'reference-failed.json',report);raise
            reports.append(report);write_json(out/'results.json',reports)
            print('CASE_RESULT',json.dumps(report),flush=True)
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    write_json(out/'complete.json',dict(completed=True,cases=len(reports),finite_cases=sum(r['completed'] for r in reports),optimizer_updates=0,
        note='No automatic exactness claim or broad tolerance pass. Inspect repeatability, routes, losses and per-parameter errors.'))


if __name__=='__main__':main()
