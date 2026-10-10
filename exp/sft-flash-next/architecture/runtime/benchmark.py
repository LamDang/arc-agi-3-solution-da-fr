"""Measure unchanged full-context F/B without an unchunked control or gradient clones."""
from contextlib import nullcontext
import math
import torch

from .evidence import write


def gradient_statistics(model,expected,allow_unrouted):
    tensors=nonzero=absent=0;norm_squared=0.
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        tensors+=1
        if p.grad is None:
            if not (allow_unrouted and '.mlp.experts.' in name):
                raise RuntimeError('Missing benchmark gradient: '+name)
            absent+=1;continue
        g=p.grad.detach()
        if not bool(torch.isfinite(g).all()):raise RuntimeError('Nonfinite benchmark gradient: '+name)
        nonzero+=int(bool(torch.count_nonzero(g)))
        norm_squared+=float(g.double().square().sum())
    if tensors!=expected:raise RuntimeError('Unexpected benchmark adapter tensor inventory')
    return dict(adapter_tensors=tensors,nonzero_gradient_tensors=nonzero,
        absent_unrouted_tensors=absent,gradient_norm=math.sqrt(norm_squared),all_finite=True,
        raw_gradients_retained=False)


def run_benchmark(architecture,config,output,resources,event,iterator,prepare):
    current=next(iterator) if iterator else {'batch':torch.load(config.samples[0],map_location='cpu',weights_only=True)}
    architecture.activate(current)
    batch,labels,targets=prepare(current['batch'],config,'benchmark')
    tokens=labels.shape[1]
    if tokens!=config.benchmark_tokens:raise ValueError('Benchmark fixture token count differs from requested length')
    model=architecture.model;repeats=[]
    for index in range(config.benchmark_repeats):
        model.zero_grad(set_to_none=True)
        event('benchmark_repeat_start',repeat=index,tokens=tokens,targets=targets)
        context=torch.autograd.graph.save_on_cpu(pin_memory=False) if config.save_on_cpu else nullcontext()
        with context:
            with resources.phase(f'forward-{index}'):
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=architecture.loss(batch,labels)
            value=float(loss.detach())
            if not math.isfinite(value):raise RuntimeError('Nonfinite benchmark loss')
            with resources.phase(f'backward-{index}'):loss.backward()
            del loss
        with resources.phase(f'gradient_statistics-{index}'):
            stats=gradient_statistics(model,config.adapter_tensors,config.optimizations.lora_routed_experts)
        seconds=sum(r['seconds'] for r in resources.rows if r['phase'] in {f'forward-{index}',f'backward-{index}'})
        row=dict(repeat=index,loss=value,tokens=tokens,targets=targets,target_fraction=targets/(tokens-1),
            forward_backward_seconds=seconds,input_tokens_per_second=tokens/seconds,
            target_tokens_per_second=targets/seconds,**stats)
        repeats.append(row);write(output/'benchmark-repeats.json',repeats)
        event('benchmark_repeat_complete',**row)
    result=dict(mode='benchmark',completed=True,repeats=repeats,optimizer_updates=0,
        clipping_applied=False,raw_gradients_retained=False,unchunked_control_executed=False,
        measurement_scope='Forward/backward only; no optimizer state or update',
        first_pass_cache_state='Fresh process; OS/disk/compiled-kernel caches may already be warm',
        ram_poll_interval_seconds=.5)
    write(output/'result.json',result);event('finished',**result)
    return result
