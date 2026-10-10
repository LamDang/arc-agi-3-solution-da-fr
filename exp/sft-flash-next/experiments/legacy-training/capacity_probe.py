"""Disposable full-context QLoRA capacity test; never saves updated adapters.

Uses existing Intel AutoRound weights, physically selects 256 experts in GPU
memory, and runs the autograd-safe reference training backend. This is a capacity
test, not an Unsloth benchmark or a dataset-quality experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback
import types

import torch
import backend
from kernel_checks import compare, unwrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prefetch-weights',action='store_true',help='Four bounded streaming readers warm non-PLE source weights before model load; startup only')
    p.add_argument('--keep', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tokens', type=int, default=130000)
    p.add_argument('--targets', type=int, default=10000)
    p.add_argument('--images', type=int, default=54)
    p.add_argument('--sample', help='Prepared real composite tensors (.pt); full test only')
    p.add_argument('--gdn-checkpoint', action='store_true')
    p.add_argument('--gdn-chunk-tokens',type=int,default=0)
    p.add_argument('--gdn-block-tokens',type=int,default=0)
    p.add_argument('--attention-projection-block',type=int,default=0)
    p.add_argument('--gated-norm-block', type=int, default=0)
    p.add_argument('--gpu-budget-gib', type=float, default=0)
    p.add_argument('--model-gradient-check', action='store_true')
    p.add_argument('--ple-checkpoint', action='store_true')
    p.add_argument('--ple-block', type=int, default=0)
    p.add_argument('--ple-resident', action='store_true')
    p.add_argument('--checkpoint-group', type=int, default=1)
    p.add_argument('--ple-cache-gib', type=float, default=0)
    p.add_argument('--ple-workers', type=int, default=1)
    p.add_argument('--hyper-block', type=int, default=0)
    p.add_argument('--suite', help='JSON list of named full-context benchmark cases')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--norm-block', type=int, default=1024)
    p.add_argument('--rms-block-mib', type=float, default=0)
    p.add_argument('--query-block', type=int, default=64)
    p.add_argument('--expert-block', type=int, default=1024)
    p.add_argument('--index-block', type=int, default=256)
    p.add_argument('--loss-block', type=int, default=128)
    p.add_argument('--attention-backend', choices=['sdpa','triton'], default='sdpa')
    p.add_argument('--key-block', type=int, choices=[32,64], default=32)
    p.add_argument('--disk-dir')
    p.add_argument('--disk-budget-gib', type=float, default=0)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--host-anon-limit-gib', type=float, default=160)
    p.add_argument('--max-seconds', type=int, default=7200)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    state = {'phase': 'startup', 'passed': False}
    stop = threading.Event()

    def memory():
        d = dict(gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                 gpu_reserved_gib=torch.cuda.memory_reserved()/2**30,
                 gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                 gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        status = Path('/proc/self/status').read_text().splitlines()
        for name in ('VmRSS', 'RssAnon', 'RssFile'):
            d[name+'_gib'] = next((float(s.split()[1])/2**20 for s in status if s.startswith(name+':')), 0)
        stat = Path('/sys/fs/cgroup/memory.stat')
        if stat.exists():
            v = dict(line.split() for line in stat.read_text().splitlines())
            d['cgroup_anon_gib'] = int(v['anon'])/2**30
            d['cgroup_file_gib'] = int(v['file'])/2**30
        return d

    def emit(event, **kw):
        value = dict(event=event, case=state.get('case'), seconds=round(time.monotonic()-start, 3), phase=state['phase'], **memory(), **kw)
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
        return value

    def save():
        tmp = out/'result.tmp'; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(out/'result.json')

    def monitor():
        while not stop.wait(5):
            m = memory()
            with (out/'memory.jsonl').open('a') as f:
                f.write(json.dumps(dict(seconds=time.monotonic()-start, phase=state['phase'], **m))+'\n')
            reason = None
            if m.get('cgroup_anon_gib', 0)>a.host_anon_limit_gib: reason='host anonymous-memory guard'
            if time.monotonic()-start>a.max_seconds: reason='time guard'
            if reason:
                state.update(error=reason, stopped_before_os_oom=True, final_memory=m)
                save(); print('PROBE_STOP '+reason, flush=True); os._exit(3)

    threading.Thread(target=monitor, daemon=True).start()
    try:
        torch.manual_seed(20261008)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        state.update(hardware=torch.cuda.get_device_name(), total_vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                     versions={n: md.version(n) for n in ('torch','transformers','flash-linear-attention','safetensors')},
                     cuda=torch.version.cuda, tokens=a.tokens, targets=a.targets,
                     synthetic=True, images=a.images, image_shape=[640,640], adapters_saved=False,
                     backend='custom AutoRound LoRA, per-layer non-reentrant checkpointing, CPU activation offload',
                     convolution='causal-conv1d 1.7.0 native, PR23 verified wheel',
                     recipe=dict(norm_block=a.norm_block,query_block=a.query_block,
                                 expert_block=a.expert_block,index_block=a.index_block,loss_block=a.loss_block,hyper_block=a.hyper_block,ple_cache_gib=a.ple_cache_gib,ple_workers=a.ple_workers,checkpoint_group=a.checkpoint_group,ple_checkpoint=a.ple_checkpoint,gdn_checkpoint=a.gdn_checkpoint,ple_block=a.ple_block,gated_norm_block=a.gated_norm_block,ple_resident=a.ple_resident,gdn_chunk_tokens=a.gdn_chunk_tokens,rms_block_mib=a.rms_block_mib,gdn_block_tokens=a.gdn_block_tokens,attention_projection_block=a.attention_projection_block),
                     keep_sha256=hashlib.sha256(Path(a.keep).read_bytes()).hexdigest())
        state['phase']='kernel_check'
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn as native_conv
        mq = backend.rm.mq
        reference = unwrap(mq.torch_chunk_gated_delta_rule)
        shape=(1,257,4,64)
        inputs=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        inputs += [-torch.rand(shape[:-1],device='cuda'),torch.rand(shape[:-1],device='cuda',dtype=torch.bfloat16)]
        def call(fn):
            return lambda q,k,v,g,beta: fn(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True,output_final_state=False)[0]
        state['kernel_errors']=compare(call(chunk_gated_delta_rule),call(reference),inputs)
        conv_inputs=[torch.randn(1,128,257,device='cuda',dtype=torch.bfloat16),
                     torch.randn(128,4,device='cuda',dtype=torch.bfloat16)]
        reference_conv=unwrap(mq.causal_conv1d_fn)
        state['conv_kernel_errors']=compare(lambda x,w:native_conv(x,w,None,activation='silu'),
                                          lambda x,w:reference_conv(x,w,None,activation='silu'),conv_inputs)
        mq.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
        mq.causal_conv1d_fn=native_conv
        norm = mq.Qwen4ExpTextRMSNorm(10240, group_size=2560).to(device='cuda',dtype=torch.bfloat16)
        norm.requires_grad_(False)
        with torch.no_grad(): norm.weight.normal_(0,.1)
        state['norm_kernel_errors']=compare(
            lambda x:backend.FrozenRMSNorm.apply(x,norm.weight,norm.eps,norm.group_size,31),
            norm,[torch.randn(1,137,10240,device='cuda',dtype=torch.bfloat16)],tolerance=.008)
        del norm
        if a.attention_backend == 'triton':
            from sparse_kernels import IndexedAttention
            n=67
            inputs=[torch.randn(h,n,256,device='cuda',dtype=torch.bfloat16) for h in (24,2,2)]
            selected=torch.arange(n,device='cuda').repeat(n,1)
            selected=torch.where(selected<=torch.arange(n,device='cuda')[:,None],selected,-1)
            state['sparse_kernel_errors']=compare(
                lambda q,k,v:IndexedAttention.apply(q,k,v,selected,256**-.5,a.key_block),
                lambda q,k,v:backend.attention_block(q.float(),k.float(),v.float(),selected,256**-.5),
                inputs,tolerance=.015)
            del selected
        del inputs,conv_inputs
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        norm = mq.Qwen4ExpTextRMSNormGated(128,activation='sigmoid').to(device='cuda',dtype=torch.bfloat16)
        norm.requires_grad_(False)
        state['gated_norm_kernel_errors'] = compare(
            lambda x,g: backend.FrozenGatedRMSNorm.apply(x,g,norm.weight,norm.variance_epsilon,31,norm.activation),
            norm, [torch.randn(2,137,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)], tolerance=.008)
        del norm
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        if a.gdn_chunk_tokens:
            import gdn_segments
            state['gdn_segment_checks'] = gdn_segments.check()
        if a.gdn_block_tokens:
            import gdn_blocks
            state['gdn_block_checks'] = gdn_blocks.check()
        if a.attention_projection_block:
            import attention_projection_checks
            state['attention_projection_checks'] = attention_projection_checks.check()
        emit('kernel_check_passed', errors=state['kernel_errors'],conv_errors=state['conv_kernel_errors'])
        if a.prefetch_weights:
            from concurrent.futures import ThreadPoolExecutor
            root = Path(a.model)
            mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
            names = sorted({f for k,f in mapping.items() if not k.startswith('mtp.') and 'ngram_embedding' not in k})
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
                raise ValueError('Unsafe prefetch shard path')
            state['phase']='prefetch_weights'
            began = time.monotonic()
            def warm(name):
                size = 0
                with (root/name).open('rb') as stream:
                    while data := stream.read(8 << 20):size += len(data)
                return size
            with ThreadPoolExecutor(max_workers=4) as pool:total = sum(pool.map(warm,names))
            state['source_prefetch'] = dict(bytes=total,seconds=time.monotonic()-began,workers=4)
            emit('source_prefetch_complete',prefetch_seconds=state['source_prefetch']['seconds'],
                 bytes=total,workers=4)
        state['phase']='load_512'
        model,_=backend.rm.load_model(a.model,record=False,log=lambda s: print(s,flush=True))
        emit('loaded_512')
        state['phase']='prune_256'
        keep=json.loads(Path(a.keep).read_text())['kept']
        for i,layer in enumerate(model.model.language_model.layers):
            ids=torch.tensor(keep[str(i)],device='cuda')
            assert len(ids)==256 and len(set(ids.tolist()))==256
            e=layer.mlp.experts
            for name in ('qweight_gate_up','scales_gate_up','qweight_down','scales_down'):
                setattr(e,name,getattr(e,name).index_select(0,ids))
            e.num_experts=256; e.keep=None; e.filled=None
            gate=layer.mlp.gate
            gate.weight=torch.nn.Parameter(gate.weight.index_select(0,ids),requires_grad=False)
            gate.num_experts=256
        model.config.text_config.num_experts=256
        gc.collect();torch.cuda.empty_cache()
        targets=backend.configure(model,rank=16,alpha=32,expert_block=a.expert_block,
                                  query_block=a.query_block,index_block=a.index_block,norm_block=a.norm_block,hyper_block=a.hyper_block,ple_cache_gib=a.ple_cache_gib,ple_workers=a.ple_workers,checkpoint_group=a.checkpoint_group,ple_checkpoint=a.ple_checkpoint,gdn_checkpoint=(a.gdn_checkpoint or bool(a.suite and any(c.get('gdn_checkpoint') for c in json.loads(Path(a.suite).read_text())))),ple_block=a.ple_block,gated_norm_block=a.gated_norm_block,ple_resident=a.ple_resident,gdn_chunk_tokens=a.gdn_chunk_tokens,rms_block_mib=a.rms_block_mib,gdn_block_tokens=a.gdn_block_tokens,attention_projection_block=a.attention_projection_block)
        for layer in model.model.language_model.layers:
            if hasattr(layer,'self_attn'):
                layer.self_attn.train_attention_backend=a.attention_backend
                layer.self_attn.key_block=a.key_block
        state['recipe'].update(attention_backend=a.attention_backend,key_block=a.key_block,
                               disk_budget_gib=a.disk_budget_gib,prefetch=a.prefetch)
        parameters=[x for x in model.parameters() if x.requires_grad]
        state['adapter_parameters']=sum(x.numel() for x in parameters)
        state['adapter_modules']=targets
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0,foreach=False)
        emit('configured_256',adapter_parameters=state['adapter_parameters'])
        if a.model_gradient_check:
            import kernel_checks
            state['model_gradient_check'] = kernel_checks.check_model_backward(model, a.loss_block)
            emit('model_gradient_check_passed', **state['model_gradient_check'])
        # A zero-initialized B gives zero A gradients initially; preserve the
        # random A initialization, and discard both smoke and full test updates.
        initial=backend.adapter_state(model)
        def layer_hook(i):
            def hook(module, args):
                emit('layer_start', layer=i)
                # Forward pre-hooks must return None to leave inputs unchanged.
            return hook
        for i,layer in enumerate(model.model.language_model.layers):
            layer.register_forward_pre_hook(layer_hook(i))
        cases = json.loads(Path(a.suite).read_text()) if a.suite else [dict(name='full-'+str(i), tokens=a.tokens, targets=a.targets, sample=a.sample, disk_budget_gib=a.disk_budget_gib) for i in range(a.repeat)]
        cases.insert(0, dict(name='smoke', tokens=512, targets=128, sample=None, disk_budget_gib=0))
        state['cases'] = []
        for trial, case in enumerate(cases):
            state['trial'], state['case'] = trial, case['name']
            label = 'smoke' if trial == 0 else 'full'
            n, t = case['tokens'], case['targets']
            sample_path = case.get('sample')
            disk_budget = case.get('disk_budget_gib', a.disk_budget_gib)
            model.train_checkpoint_group = case.get('checkpoint_group', a.checkpoint_group)
            for layer in model.model.language_model.layers:
                if hasattr(layer,'self_attn'):
                    layer.self_attn.train_projection_block = case.get('attention_projection_block',a.attention_projection_block)
                if hasattr(layer,'linear_attn') and hasattr(layer.linear_attn,'train_gdn_block_tokens'):
                    layer.linear_attn.train_gdn_block_tokens = case.get('gdn_block_tokens',a.gdn_block_tokens)
                if hasattr(layer,'linear_attn') and hasattr(layer.linear_attn,'train_gdn_chunk_tokens'):
                    layer.linear_attn.train_gdn_chunk_tokens = case.get('gdn_chunk_tokens',a.gdn_chunk_tokens)
                layer.mlp.experts.train_block = case.get('expert_block', a.expert_block)
                if hasattr(layer,'linear_attn') and hasattr(layer.linear_attn,'uncheckpointed_gdn_forward'):
                    layer.linear_attn.forward = (types.MethodType(backend.checkpointed_gdn, layer.linear_attn)
                        if case.get('gdn_checkpoint', a.gdn_checkpoint) else layer.linear_attn.uncheckpointed_gdn_forward)
            optimizer.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            budget = case.get('gpu_budget_gib', a.gpu_budget_gib)
            torch.cuda.set_per_process_memory_fraction(budget*2**30/torch.cuda.get_device_properties(0).total_memory if budget else 1.)
            state['phase']=label+'_forward'
            # Random ordinary vocabulary IDs exercise routing and PLE hashes;
            # exclude special/image tokens. Sequence length is exact.
            ids=torch.randint(1000,100000,(1,n),generator=torch.Generator().manual_seed(20261008))
            enc={'input_ids':ids}
            images=a.images if label=='full' and not sample_path else 0
            if images:
                assert 16+images*402 < n-t, 'Images must fit entirely inside prompt'
                for j in range(images):
                    offset=16+j*402
                    ids[0,offset]=model.config.vision_start_token_id
                    ids[0,offset+1:offset+401]=model.config.image_token_id
                    ids[0,offset+401]=model.config.vision_end_token_id
                vc=model.config.vision_config
                enc.update(image_grid_thw=torch.tensor([[1,40,40]]*images),
                           pixel_values=torch.randn(images*1600,vc.in_channels*vc.temporal_patch_size*vc.patch_size**2,dtype=torch.bfloat16),
                           mm_token_type_ids=(ids==model.config.image_token_id).int())
            if label=='full' and sample_path:
                enc=torch.load(sample_path,map_location='cpu',weights_only=True)
                assert enc['input_ids'].shape==(1,n)
                images=len(enc.get('image_grid_thw',[]))
                state.update(synthetic=False,composite_real_samples=True,images=images,
                             sample_sha256=hashlib.sha256(Path(sample_path).read_bytes()).hexdigest())
            sample_started=began=time.monotonic();emit('sample_start',tokens=n,targets=t)
            from offload import ActivationOffload
            with backend.request_ple_cache(model), ActivationOffload(model,disk_dir=a.disk_dir,
                    disk_budget_gib=disk_budget,prefetch=a.prefetch) as offload:
                loss=backend.loss_sum(model,enc,n-t,'cuda',loss_block=a.loss_block)/t
                torch.cuda.synchronize()
                forward_seconds=time.monotonic()-began
                emit('forward_complete',loss=float(loss.detach()),phase_seconds=forward_seconds)
                if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite loss')
                state['phase']=label+'_backward';began=time.monotonic()
                loss.backward();torch.cuda.synchronize()
                backward_seconds=time.monotonic()-began
                emit('backward_complete',phase_seconds=backward_seconds)
            state[label+'_offload']=offload.stats
            for name,param in model.named_parameters():
                if param.requires_grad and (param.grad is None or not bool(torch.isfinite(param.grad).all())):
                    raise RuntimeError('Missing/nonfinite gradient: '+name)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if not float(norm)>0: raise RuntimeError('Zero adapter gradient norm')
            state['phase']=label+'_optimizer';began=time.monotonic()
            optimizer.step();torch.cuda.synchronize()
            state[label]=emit('optimizer_complete',loss=float(loss.detach()),grad_norm=float(norm),phase_seconds=time.monotonic()-began)
            state['cases'].append(dict(**case, forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                step_seconds=time.monotonic()-sample_started, loss=float(loss.detach()), grad_norm=float(norm),
                offload=offload.stats, memory=memory(),
                ple_cache=[dict(hits=m.row_cache.hits, misses=m.row_cache.misses) for m in model.modules() if hasattr(m,'row_cache')]))
            # An incremental report survives a later-case failure without claiming suite success.
            (out/'completed-cases.json').write_text(json.dumps(state['cases'],indent=2))
            del loss,enc,ids
            optimizer.zero_grad(set_to_none=True)
            if True:  # Every capacity update is discarded, including repeated trials.
                backend.load_adapter(model,initial)
                optimizer.state.clear()
        state.update(passed=True,phase='complete',final_memory=memory(),elapsed_seconds=time.monotonic()-start)
        save();emit('probe_passed')
    except BaseException as e:
        state.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),final_memory=memory(),elapsed_seconds=time.monotonic()-start)
        save();emit('probe_failed',error_type=type(e).__name__,error=str(e));traceback.print_exc()
    finally:
        stop.set()


if __name__=='__main__': main()
