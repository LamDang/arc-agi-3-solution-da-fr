"""Test-only CCE repeatability and actual split-layer expert VJPs.

One no-grad anchor forward captures the exact selected head states and split
expert inputs in RAM. No full-model backward, update or raw tensor archive.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from peft import get_peft_model_state_dict
from config import load_config
from model import build
from components.head import target_positions
from components.ple import EncodedPaths, prepared_loader
from components.expert_chunks import expert_chunks
from diagnostics.expert_replay import metrics, vjp_comparison
from runtime.evidence import compare_initial, model_identity, sha, snapshot, snapshot_imports, write
from runtime.loop import prepare
from runtime.numeric_profile import verify_profile
from runtime.resources import Resources


class HeadCaptured(Exception):
    pass


def tensor_identity(value):
    data = value.detach().cpu().contiguous()
    digest = hashlib.sha256(memoryview(data.view(torch.uint8).numpy())).hexdigest()
    return dict(shape=list(data.shape), dtype=str(data.dtype), stride=list(data.stride()), sha256=digest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config, 'test')
    assert config.optimizations.expert_chunking and config.optimizations.head == 'cce_exact'
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    snapshot(out, config, 'diagnostic-expert-head')
    resources = Resources(out)
    verify_profile(config, out)
    actual = dict(sample=sha(config.samples[0]), **model_identity(config.model))
    assert all(actual.get(k) == v for k, v in config.expected_sha256.items())
    started = time.monotonic()

    def event(name, **values):
        row = dict(event=name, elapsed_seconds=time.monotonic()-started, **values)
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    iterator = iter(prepared_loader(EncodedPaths(config.samples), config.model,
                                   event_path=str(out/'ple-events.jsonl')))
    architecture = None
    handles = []
    captured = {}
    head = {}
    try:
        event('load_start')
        with resources.phase('loading'):
            architecture = build(config, out)
        model = architecture.model
        manager = architecture.expert_stager
        initial = {n:p.detach().cpu() for n,p in get_peft_model_state_dict(model).items()}
        with resources.phase('initialization_verification'):
            identity = compare_initial(initial, config.baseline)
        write(out/'initial-comparison.json', identity)
        assert identity['passed']
        del initial
        current = next(iterator)
        architecture.activate(current)
        batch, labels, targets_count = prepare(current['batch'], config, 'test')
        positions, targets = target_positions(labels)
        base = model.get_base_model()
        weight = base.lm_head.weight
        assert not weight.requires_grad and weight.dtype == torch.bfloat16

        def capture_expert(module, inputs, kwargs):
            if kwargs.get('_prefetched'):
                return
            x, indices, weights = inputs
            counts = torch.bincount(indices.flatten(), minlength=module.num_experts).tolist()
            if any(c > module.chunk_tokens for c in counts):
                assert module.expert_layer not in captured
                captured[module.expert_layer] = dict(module=module, x=x.detach().cpu().clone(),
                    indices=indices.detach().cpu().clone(), weights=weights.detach().cpu().clone(), counts=counts)
                event('expert_input_captured', layer=module.expert_layer,
                      split_experts={str(i):c for i,c in enumerate(counts) if c > module.chunk_tokens})

        def capture_head(module, inputs, output):
            hidden = output.last_hidden_state
            selected = hidden[0].index_select(0, positions).to(weight.dtype).contiguous()
            assert hidden.dtype == torch.bfloat16 and selected.shape[0] == targets_count
            head.update(selected=selected.detach().cpu(), targets=targets.flatten().detach().cpu(),
                        hidden_dtype=str(hidden.dtype), hidden_shape=list(hidden.shape))
            raise HeadCaptured()

        for module in manager.modules:
            handles.append(module.register_forward_pre_hook(capture_expert, with_kwargs=True))
        handles.append(base.model.register_forward_hook(capture_head))
        event('capture_forward_start')
        try:
            with resources.phase('capture_forward'), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                architecture.loss(batch, labels)
        except HeadCaptured:
            pass
        else:
            raise RuntimeError('Head capture did not execute')
        for handle in handles:
            handle.remove()
        handles.clear()
        del batch, labels, current, positions, targets
        for layer in list(manager.cache):
            manager.release(layer)
        gc.collect()
        torch.cuda.empty_cache()
        write(out/'captured-inputs.json', dict(head_selected=tensor_identity(head['selected']),
            targets=tensor_identity(head['targets']), hidden_dtype=head['hidden_dtype'], hidden_shape=head['hidden_shape'],
            head_weight=dict(shape=list(weight.shape), dtype=str(weight.dtype), requires_grad=weight.requires_grad,
                             identity='Same frozen parameter object throughout all repetitions; pinned model index'),
            experts={str(layer):dict(x=tensor_identity(s['x']), indices=tensor_identity(s['indices']),
                                    weights=tensor_identity(s['weights']), counts=s['counts']) for layer,s in captured.items()},
            raw_tensors_retained=False))

        # Repeat the unchanged production CCE call with fresh identical leaves.
        # PyTorch deterministic mode does not guarantee custom Triton reductions.
        from cut_cross_entropy import linear_cross_entropy
        flags = dict(impl='cce_exact', reduction='mean', filter_eps=None, filter_e_grad=False,
                     filter_c_grad=False, accum_e_fp32=True, accum_c_fp32=True)
        write(out/'cce-settings.json', dict(flags=flags, repeats=12, denominator=targets_count,
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            selected_input=tensor_identity(head['selected']), autocast_enabled=False))
        head_rows = []
        baseline = previous = None
        for repeat in range(12):
            selected = head['selected'].to('cuda').requires_grad_(True)
            targets = head['targets'].to('cuda')
            with resources.phase(f'head-vjp-{repeat}'):
                with torch.autocast('cuda', enabled=False):
                    loss = linear_cross_entropy(selected, weight, targets, **flags)
                gradient, = torch.autograd.grad(loss, selected)
            value = float(loss.detach())
            state = dict(loss=value, bits=struct.pack('<f', value).hex(), gradient=gradient.detach().cpu())
            if baseline is None:
                baseline = state
            if previous is None:
                previous = state
            row = dict(repeat=repeat, loss=value, loss_fp32_bits=state['bits'],
                loss_equal_first=state['bits']==baseline['bits'], loss_equal_previous=state['bits']==previous['bits'],
                hidden_gradient_first=metrics(state['gradient'], baseline['gradient']),
                hidden_gradient_previous=metrics(state['gradient'], previous['gradient']))
            head_rows.append(row)
            write(out/'head-repeats.json', head_rows)
            event('head_repeat', repeat=repeat, loss=value, loss_fp32_bits=state['bits'],
                  gradient_exact=row['hidden_gradient_first']['bitwise_equal'],
                  gradient_l2=row['hidden_gradient_first']['relative_l2'])
            previous = state
            del loss, gradient, selected, targets
        del baseline, previous, state
        gc.collect()
        torch.cuda.empty_cache()

        expert_rows = []
        for layer, saved in sorted(captured.items()):
            module = saved['module']
            size = module.chunk_tokens
            named = {n:p for n,p in module.named_parameters() if p.requires_grad}
            cotangent_seed = config.seed+layer
            generator = torch.Generator(device='cpu').manual_seed(cotangent_seed)
            cotangent = (torch.randn(saved['x'].shape, generator=generator, dtype=torch.float32)*.001).bfloat16()
            baseline = None
            try:
                for variant in ['native', 'native-repeat', 'chunked', 'custom-unsplit']:
                    for cached_layer in list(manager.cache):
                        manager.release(cached_layer)
                    x = saved['x'].to('cuda').requires_grad_(True)
                    weights = saved['weights'].to('cuda').requires_grad_(True)
                    indices = saved['indices'].to('cuda')
                    module.chunk_tokens = x.shape[0]*indices.shape[1] if variant=='custom-unsplit' else size
                    with resources.phase(f'expert-{layer}-vjp-{variant}'):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            value = manager.call(module, x, indices, weights) if variant.startswith('native') else expert_chunks(module, x, indices, weights)
                        output = value.detach().cpu()
                        grads = torch.autograd.grad(value, (x,weights,*named.values()), cotangent.to('cuda'), allow_unused=True)
                    dx, dweights, *adapters = grads
                    adapter_values = {n:g.detach().cpu() if g is not None else torch.zeros_like(p, device='cpu')
                                      for (n,p),g in zip(named.items(),adapters)}
                    state = dict(output=output, dx=dx.detach().cpu(), dweights=dweights.detach().cpu(), adapters=adapter_values)
                    if baseline is None:
                        baseline = state
                    adapter_comparison = vjp_comparison(state['adapters'], baseline['adapters'])
                    partitions = {}
                    for partition in ['split','unsplit']:
                        names = [n for n in named if (saved['counts'][int(n.split('.')[0])]>size)==(partition=='split')]
                        if names:
                            partitions[partition] = vjp_comparison({n:state['adapters'][n] for n in names},
                                                                  {n:baseline['adapters'][n] for n in names})
                    row = dict(layer=layer, variant=variant, cotangent_seed=cotangent_seed,
                        output=metrics(state['output'],baseline['output']),
                        input_gradient=metrics(state['dx'],baseline['dx']), routing_gradient=metrics(state['dweights'],baseline['dweights']),
                        adapter_gradients=adapter_comparison, adapter_partitions=partitions)
                    if variant=='native-repeat':
                        assert row['output']['bitwise_equal'] and row['input_gradient']['bitwise_equal']
                        assert row['routing_gradient']['bitwise_equal'] and adapter_comparison['passed']
                    expert_rows.append(row)
                    write(out/'expert-vjps.json',expert_rows)
                    event('expert_vjp', layer=layer, variant=variant, adapter_l2=adapter_comparison['global_relative_l2'],
                        input_gradient_exact=row['input_gradient']['bitwise_equal'], routing_gradient_exact=row['routing_gradient']['bitwise_equal'])
                    del grads, dx, dweights, adapters, adapter_values, state, output, value, x, weights, indices
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                module.chunk_tokens = size
            del baseline, cotangent
        write(out/'result.json',dict(completed=True,mode='diagnostic-expert-head',head_repeats=len(head_rows),
            head_loss_repeatable=all(r['loss_equal_first'] for r in head_rows),
            head_hidden_gradient_repeatable=all(r['hidden_gradient_first']['bitwise_equal'] for r in head_rows),
            split_layers=sorted(captured), expert_vjps=len(expert_rows), optimizer_updates=0,
            raw_gradients_retained=False, full_model_backward=False,
            note='Same-input CCE repeats and fixed-cotangent expert VJPs; not full-model qualification.'))
    finally:
        for handle in handles:
            handle.remove()
        write(out/'resources.json', resources.rows)
        snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close()
            write(out/'expert-prefetch.json',architecture.expert_stager.report())
        iterator._shutdown_workers()


if __name__=='__main__':
    main()
