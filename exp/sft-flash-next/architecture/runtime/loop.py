"""One execution path for architecture qualification and supervised training."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from peft import get_peft_model_state_dict

from model import build
from .evidence import compare, compare_initial, retain_raw_gradients, sha, snapshot, snapshot_imports, write
from .resources import Resources
from .events import event_row


def prepare(raw, config, mode):
    batch = dict(raw)
    labels = batch.pop('labels', None)
    ids = batch['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 < ids.shape[1] <= config.max_tokens:
        raise ValueError('Require one complete sample within the token limit; never truncate')
    if labels is None:
        if mode != 'test' or config.prompt_tokens is None:
            raise ValueError('Training samples require explicit interleaved labels')
        if not 0 < config.prompt_tokens < ids.shape[1]:
            raise ValueError('Diagnostic prompt length exceeds sample')
        labels = ids.clone();labels[:,:config.prompt_tokens] = -100
    if labels.shape != ids.shape or labels.dtype != torch.int64 or labels[0,0].item() != -100:
        raise ValueError('Invalid next-token labels')
    selected = labels != -100
    if not torch.equal(labels[selected],ids[selected]):
        raise ValueError('Supervised labels must match the complete input sequence')
    count = int((labels[:,1:] != -100).sum())
    if not count:
        raise ValueError('No supervised next-token targets')
    batch = {key:value.to('cuda',dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for key,value in batch.items()}
    return batch, labels.to('cuda'), count


def gradients(model,expected=744,allow_unrouted=False):
    result = {}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None and allow_unrouted and '.mlp.experts.' in name:
            result[name]=torch.zeros_like(parameter,device='cpu')
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Missing/nonfinite raw gradient: '+name)
        result[name] = parameter.grad.detach().cpu().clone()
    if len(result) != expected:
        raise RuntimeError('Unexpected adapter gradient count')
    return result


def save_tensors(state,output,stem):
    if len(state)<=744:
        path=output/(stem+'.pt');torch.save(state,path)
        return dict(format='torch',sha256=sha(path),tensors=len(state))
    # Bounded files can be collected over Jupyter without multi-GB requests.
    directory=output/stem;directory.mkdir();rows=[];items=list(state.items())
    for index,start in enumerate(range(0,len(items),1024)):
        shard=dict(items[start:start+1024]);path=directory/f'{index:04d}.pt'
        torch.save(shard,path);rows.append(dict(path=str(path.relative_to(output)),sha256=sha(path),tensors=len(shard)))
    manifest=dict(format='torch-shards',tensors=len(state),shards=rows)
    write(output/(stem+'-manifest.json'),manifest)
    return manifest


def run(config, mode):
    output = Path(config.output);output.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',config.as_dict())
    started = time.monotonic()
    def event(name, **data):
        row = event_row(name,time.monotonic()-started,data)
        with (output/'events.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    torch.manual_seed(config.seed);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    provenance = snapshot(output,config,mode)
    expected = config.expected_sha256 or {}
    actual = dict(sample=sha(config.samples[0]),model_config=sha(Path(config.model)/'config.json'))
    if config.adapter:actual['adapter'] = sha(config.adapter)
    if any(actual.get(key) != digest for key,digest in expected.items()):
        raise RuntimeError('Run input hash differs from configuration')
    resources = Resources()
    loader = iterator = None
    if config.optimizations.disk_ple:
        from components.ple import EncodedPaths,prepared_loader
        loader = prepared_loader(EncodedPaths(config.samples),config.model,lookahead=2,event_path=str(output/'ple-events.jsonl'))
        iterator = iter(loader)
    try:
        event('load_start')
        with resources.phase('loading'):
            architecture = build(config,output)
        model = architecture.model
        write(output/'components.json',architecture.inventory)
        write(output/'loading.json',architecture.loading)
        if architecture.ple_manifest:write(output/'ple-manifest.json',architecture.ple_manifest)
        initial = {name:value.detach().cpu().clone() for name,value in get_peft_model_state_dict(model).items()}
        if mode == 'test' and config.baseline and config.baseline.get('initial_adapter'):
            with resources.phase('initialization_verification'):
                initial_comparison=compare_initial(initial,config.baseline)
            write(output/'initial-comparison.json',initial_comparison)
            if not initial_comparison['passed']:raise RuntimeError('Reference initialization differs; no forward/backward run')
        else:
            save_tensors(initial,output,'initial-adapter')
        del initial
        event('load_complete',trainable_tensors=config.adapter_tensors)
        if mode == 'test':
            current = next(iterator) if iterator else {'batch':torch.load(config.samples[0],map_location='cpu',weights_only=True)}
            architecture.activate(current)
            batch,labels,targets = prepare(current['batch'],config,mode)
            model.zero_grad(set_to_none=True)
            context = torch.autograd.graph.save_on_cpu(pin_memory=False) if config.save_on_cpu else nullcontext()
            with context:
                event('forward_start')
                with resources.phase('forward'):
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss = architecture.loss(batch,labels)
                value = float(loss.detach())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                event('loss',loss=value,targets=targets,tokens=labels.shape[1])
                event('backward_start')
                with resources.phase('backward'):loss.backward()
                del loss
            with resources.phase('gradient_export'):
                absent=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
                raw = gradients(model,config.adapter_tensors,config.optimizations.lora_routed_experts)
                archive=save_tensors(raw,output,'gradients') if retain_raw_gradients(config,mode) else None
            summary = {name:dict(shape=list(g.shape),dtype=str(g.dtype),norm=float(g.double().norm()),
                nonzero=int(torch.count_nonzero(g)),finite=bool(torch.isfinite(g).all())) for name,g in raw.items()}
            write(output/'gradient-summary.json',summary)
            with resources.phase('gradient_comparison'):
                comparison = compare(raw,value,config.baseline) if config.baseline else None
            if comparison:write(output/'comparison.json',comparison)
            result = dict(mode=mode,loss=value,raw_gradient_tensors=len(raw),targets=targets,tokens=labels.shape[1],
                all_finite=True,optimizer_updates=0,clipping_applied=False,
                iso_verified=comparison['passed'] if comparison else None,
                gradients_sha256=archive.get('sha256') if archive else None,gradient_archive=archive,
                raw_gradients_retained=archive is not None,
                gradient_retention='reference-only; candidates compared in memory and discarded',
                absent_unrouted_gradients=absent,elapsed_seconds=time.monotonic()-started)
            from collections import Counter
            result['parameter_dtypes']=dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad))
            result['gradient_dtypes']=dict(Counter(str(g.dtype) for g in raw.values()))
            result['trainable_parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
            result['trainable_parameter_devices']=dict(Counter(p.device.type for p in model.parameters() if p.requires_grad))
            write(output/'result.json',result);event('finished',**result)
            del raw
            if comparison is not None and config.comparison_mode == 'exact' and not comparison['passed']:
                raise RuntimeError('Refactor equality gate failed; preserved comparison metrics')
            return result
        return train_samples(architecture,config,output,resources,event,iterator)
    finally:
        write(output/'resources.json',resources.rows)
        if 'architecture' in locals() and architecture.expert_stager:
            architecture.expert_stager.close()
            write(output/'expert-prefetch.json',architecture.expert_stager.report())
        snapshot_imports(output)
        if iterator is not None and getattr(iterator,'_shutdown_workers',None):iterator._shutdown_workers()


def train_samples(architecture,config,output,resources,event,iterator):
    """Whole samples, target-token accumulation, one normalization before clipping."""
    model = architecture.model
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=config.learning_rate,weight_decay=config.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    pending = updates = 0
    for epoch in range(config.epochs):
        if epoch and iterator is not None:
            raise ValueError('Disk PLE epoch replay needs a fresh loader; use epochs=1')
        for index,path in enumerate(config.samples):
            current = next(iterator) if iterator else {'batch':torch.load(path,map_location='cpu',weights_only=True)}
            architecture.activate(current)
            batch,labels,count = prepare(current['batch'],config,'train')
            context = torch.autograd.graph.save_on_cpu(pin_memory=False) if config.save_on_cpu else nullcontext()
            with context:
                with resources.phase(f'forward-{epoch}-{index}'):
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss = architecture.loss(batch,labels)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                event('loss',epoch=epoch,sample=index,loss=float(loss.detach()),targets=count)
                with resources.phase(f'backward-{epoch}-{index}'):(loss*count).backward()
            pending += count
            final = epoch+1 == config.epochs and index+1 == len(config.samples)
            if pending >= config.target_tokens_per_update or final:
                for p in parameters:
                    if p.grad is None:continue  # A sample may select no tokens for an expert.
                    p.grad.div_(pending)
                # Training keeps resumable optimizer state, not per-update raw gradients.
                with resources.phase(f'optimizer-{updates}'):
                    norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    optimizer.step()
                updates += 1
                event('update',update=updates,targets=pending,gradient_norm=float(norm))
                torch.save(dict(adapter=get_peft_model_state_dict(model),optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    next_epoch=epoch+(index+1==len(config.samples)),next_sample=(index+1)%len(config.samples),
                    updates=updates,config=config.as_dict()),output/'checkpoint.pt')
                optimizer.zero_grad(set_to_none=True);pending=0
    result = dict(mode='train',optimizer_updates=updates,completed=True,epochs=config.epochs)
    write(output/'result.json',result)
    return result
