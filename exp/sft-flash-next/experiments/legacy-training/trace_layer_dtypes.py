"""Forward-only native layer dtype diagnostic; never computes a loss or updates.

Loads the same HF/AutoRound/PEFT weights as the reference, then executes only
the configured first layers on a short, text-only token prefix. TorchDispatch
records ATen tensor boundaries; module hooks also expose custom kernel outputs.
FP32 accumulation internal to Triton/CUDA kernels is not visible to this trace.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import threading
import time
import traceback
from types import MethodType


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    out = Path(c['remote_output'])
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    for key, value in c['environment'].items():
        os.environ[key] = value
    sys.path = [x for x in sys.path if x != '/usr/local/lib/python3.13/dist-packages']
    sys.path[:0] = ['/tmp/peft-autoround-compat/package',
                   '/tmp/peft-autoround-compat/site-without-torchao', str(Path(__file__).parent)]
    import torch
    import peft
    from transformers import AutoModelForImageTextToText, AutoRoundConfig
    from torch import nn
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    from overfit_hf_reference import TARGETS, resident_device_map
    from ple_preparation import native_class, hash_computer, DiskPLERows, tensor_sha256, PREFIX
    from opt4_ple_capture import DiskTablePlaceholder
    from opt3_mask_capture import patch_model

    def sha(path):
        with Path(path).open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')

    def event(name, **data):
        print(json.dumps(dict(event=name, seconds=time.monotonic()-started, **data)), flush=True)

    for name, path in [('sample', c['sample']), ('adapter', c['adapter']),
                       ('model_config', Path(c['model'])/'config.json')]:
        assert sha(path) == c['expected_sha256'][name], name
    for name, checksum in c['sources_sha256'].items():
        assert sha(Path(__file__).parent/name) == checksum, name
    torch.manual_seed(c['seed'])
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    original_batch = torch.load(c['sample'], map_location='cpu', weights_only=True)
    # Deliberate short-prefix diagnostic; this is not a trajectory training input.
    tokens = original_batch['input_ids'][:, :c['tokens']].contiguous()
    torch.save(tokens, out/'input-ids.pt')
    hasher = hash_computer(c['model'])
    rows = DiskPLERows(c['model'])
    ids = hasher(tokens, None)
    payload, lookup = rows.lookup(ids)
    torch.save(payload, out/'prepared-ple.pt')
    torch.save(ids, out/'ple-row-ids.pt')
    # Same Opt4 frozen-table placeholder and filtered index, without workers:
    # this diagnostic executes a single short forward and has no backward.
    cls = native_class()
    init = cls.__init__
    source = textwrap.dedent(inspect.getsource(init))
    old = 'nn.Embedding(padded_vocab_size, head_dim_per_ngram)'
    assert source.count(old) == 1
    namespace = dict(init.__globals__, DiskTablePlaceholder=DiskTablePlaceholder)
    exec(compile(source.replace(old, 'DiskTablePlaceholder(padded_vocab_size, head_dim_per_ngram)')
         .replace('super().__init__()', 'nn.Module.__init__(self)'), '<dtype-trace-ple-init>', 'exec'), namespace)
    cls.__init__ = namespace['__init__']
    root = Path(c['model'])
    view = out/'model-view'
    view.mkdir()
    index = json.loads((root/'model.safetensors.index.json').read_text())
    omitted = [k for k in index['weight_map'] if k.startswith(PREFIX+'.ngram_embedding.shard_')]
    assert len(omitted) == 128
    index['weight_map'] = {k:v for k,v in index['weight_map'].items() if k not in omitted}
    index['metadata']['total_size'] -= rows.manifest['table_bytes']
    for path in root.iterdir():
        if path.name != 'model.safetensors.index.json':
            (view/path.name).symlink_to(path.resolve())
    (view/'model.safetensors.index.json').write_text(json.dumps(index))
    event('load_start')
    base, loading = AutoModelForImageTextToText.from_pretrained(
        str(view), dtype=torch.bfloat16, device_map=resident_device_map(c['model']),
        local_files_only=True, trust_remote_code=False, output_loading_info=True,
        quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
    write('loading.json', {k:list(v) if isinstance(v,set) else v for k,v in loading.items()})
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    for module in base.modules():
        for name, value in module._buffers.items():
            if value is not None and value.device.type != 'cuda':
                module._buffers[name] = value.to('cuda')
    base.requires_grad_(False)
    model = peft.get_peft_model(base, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=TARGETS, bias='none', task_type='CAUSAL_LM'))
    state = torch.load(c['adapter'], map_location='cpu', weights_only=True)
    assert set(state) == set(peft.get_peft_model_state_dict(model))
    peft.set_peft_model_state_dict(model, state)
    for name, value in peft.get_peft_model_state_dict(model).items():
        assert tensor_sha256(value) == tensor_sha256(state[name]), name
    model.train()
    lm = base.model.language_model
    ple = lm.layers[1].ple.ple_embedding
    def prepared_forward(self, input_ids, past_key_values):
        assert past_key_values is None and torch.equal(input_ids.cpu(), tokens)
        return payload.to(input_ids.device)
    ple.forward = MethodType(prepared_forward, ple)
    # Retain latest Opt3 mask construction; its arithmetic is unchanged.
    patch_model(base, out/'attention-mask.json')
    write('parameters.json', {name:dict(dtype=str(v.dtype), shape=list(v.shape), trainable=v.requires_grad)
        for name,v in base.named_parameters() if name.startswith('model.language_model.embed_tokens')
        or any(name.startswith(f'model.language_model.layers.{i}.') for i in range(c['layers']))})
    load_seconds = time.monotonic()-started
    event('load_complete', allocated_gib=torch.cuda.memory_allocated()/2**30)
    scope = []
    operation_count = 0
    module_rows = []
    layer_rows = []
    handles = []
    fp32_ops = Counter()
    promotions = []
    ops = (out/'operations.jsonl').open('w')

    def tensors(value):
        return [x for x in tree_flatten(value)[0] if isinstance(x,torch.Tensor)]

    def meta(value):
        return [dict(dtype=str(x.dtype), shape=list(x.shape), device=str(x.device),
                     requires_grad=x.requires_grad) for x in tensors(value)]

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal operation_count
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            ins = meta((args,kwargs)); outs = meta(result)
            floating_inputs = {x['dtype'] for x in ins if x['dtype'].startswith(('torch.float','torch.bfloat'))}
            fp32_output = any(x['dtype']=='torch.float32' for x in outs)
            introduces_fp32 = fp32_output and 'torch.float32' not in floating_inputs
            stack = [dict(file=f.filename, line=f.lineno, function=f.name, source=f.line)
                     for f in traceback.extract_stack()[:-1]
                     if ('qwen4_exp' in f.filename or '/peft/' in f.filename or '/auto_round' in f.filename
                         or f.filename.endswith('native_mask_storage.py'))][-8:]
            row = dict(sequence=operation_count, operation=str(func), module=scope[-1] if scope else None,
                       inputs=ins, outputs=outs, introduces_fp32=introduces_fp32,
                       autocast_enabled=torch.is_autocast_enabled('cuda'), source_stack=stack)
            if fp32_output: fp32_ops[str(func)] += 1
            if introduces_fp32: promotions.append(row)
            ops.write(json.dumps(row)+'\n')
            operation_count += 1
            return result

    for i in range(c['layers']):
        layer = lm.layers[i]
        for name, module in layer.named_modules():
            label = f'layers.{i}' + ('.'+name if name else '')
            def before(mod, args, label=label):
                scope.append(label)
                module_rows.append(dict(event='enter', module=label, tensors=meta(args),
                                        autocast_enabled=torch.is_autocast_enabled('cuda')))
            def after(mod, args, result, label=label):
                module_rows.append(dict(event='exit', module=label, tensors=meta(result),
                                        autocast_enabled=torch.is_autocast_enabled('cuda')))
                assert scope.pop() == label
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        original_forward = layer.forward
        def forward(self, *args, _original=original_forward, _i=i, **kwargs):
            torch.cuda.synchronize()
            tick = time.monotonic()
            with Trace():
                result = _original(*args, **kwargs)
            torch.cuda.synchronize()
            torch.save(result.detach().cpu(), out/f'layer-{_i}-output.pt')
            layer_rows.append(dict(layer=_i, layer_type=self.layer_type, input=meta(args), output=meta(result),
                                   traced_seconds=time.monotonic()-tick))
            event('layer_complete', **layer_rows[-1])
            return result
        layer.forward = MethodType(forward, layer)

    class StopAfterLayers(Exception): pass
    def stop_after_layers(module, args, result):
        raise StopAfterLayers()
    handles.append(lm.layers[c['layers']-1].register_forward_hook(stop_after_layers))
    import psutil
    process = psutil.Process()
    ram = dict(process_rss_peak_gib=process.memory_info().rss/2**30,
               whole_host_used_peak_gib=psutil.virtual_memory().used/2**30)
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.wait(0.02):
            ram['process_rss_peak_gib'] = max(ram['process_rss_peak_gib'], process.memory_info().rss/2**30)
            ram['whole_host_used_peak_gib'] = max(ram['whole_host_used_peak_gib'], psutil.virtual_memory().used/2**30)
    sampler = threading.Thread(target=monitor, daemon=True);sampler.start()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    forward_start = time.monotonic()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        embedding = lm.embed_tokens(tokens.cuda())
        write('embedding.json', dict(input_ids=meta(tokens), output=meta(embedding),
                                    weight=meta(lm.embed_tokens.weight)))
        torch.save(embedding.cpu(), out/'embedding.pt')
        try:
            lm(input_ids=tokens.cuda(), use_cache=False)
        except StopAfterLayers:
            pass
        else:
            raise AssertionError('Expected bounded layer stop')
    torch.cuda.synchronize()
    forward_seconds = time.monotonic()-forward_start
    stop_monitor.set();sampler.join()
    assert len(layer_rows) == c['layers'] and scope == []
    ops.close()
    for handle in handles: handle.remove()
    write('modules.json', module_rows)
    write('promotions.json', promotions)
    report = dict(completed=True, diagnostic_only=True, full_forward=False, backward=False,
        optimizer_updates=0, tokens=c['tokens'], layers=layer_rows, operation_count=operation_count,
        introduces_fp32_count=len(promotions), fp32_operations=dict(fp32_ops),
        adapter_storage_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)),
        embedding_dtype=str(embedding.dtype), autocast='torch.bfloat16',
        load_seconds=load_seconds, traced_forward_seconds=forward_seconds,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        process_rss_end_gib=psutil.Process().memory_info().rss/2**30,
        forward_ram=ram, ram_sampling_seconds=0.02, backward_peak=None,
        raw_input_kind='first 32 anchor token IDs, without multimodal image injection',
        trace_limitations=['ATen/module boundaries, not arithmetic inside Triton/CUDA kernels',
                          'TorchDispatch overhead: timings are not training benchmarks',
                          'no backward, no loss, no full sample qualification'],
        ple_lookup=lookup, config=c,
        versions={x:importlib.metadata.version(x) for x in ('torch','transformers','peft','auto-round')},
        gpu=torch.cuda.get_device_name(0))
    write('report.json', report)
    # Freeze the actual Python sources imported by this diagnostic.
    sources = out/'sources';sources.mkdir()
    source_map = {}
    for module in list(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if not file or not file.endswith('.py') or not Path(file).is_file(): continue
        if not any(s in file for s in ('qwen4_exp','/peft/tuners/lora/','/auto_round_extension/triton/',
                                      '/auto_round/modeling/unfused_moe/','native_mask_storage.py',
                                      'opt3_mask_capture.py','ple_preparation.py')): continue
        digest = sha(file)
        name = digest+'-'+Path(file).name
        (sources/name).write_bytes(Path(file).read_bytes())
        source_map[file] = dict(sha256=digest, artifact='sources/'+name)
    write('source-hashes.json',source_map)
    write('file-hashes.json', {str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*'))
          if path.is_file() and 'model-view' not in path.parts and path.name!='file-hashes.json'})
    event('finished', operations=operation_count, introduces_fp32=len(promotions))


if __name__ == '__main__':
    main()
