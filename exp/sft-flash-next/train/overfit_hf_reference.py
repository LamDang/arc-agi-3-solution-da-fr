"""Single-request learning qualification using HF, PEFT and official AutoRound.

The base checkpoint must already contain the desired 256 experts. No model
forward, attention, routing, quantization kernel or loss is replaced here.
Optional memory controls are HF non-reentrant checkpointing and PyTorch
save_on_cpu. A passing overfit is not a gradient-equivalence proof.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoRoundConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

TARGETS = (r'^model\.language_model\.layers\.\d+\.(?:'
           r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
           r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|'
           r'mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))$')


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resident_device_map(model_path):
    """Place all checkpoint branches except PLE on CUDA using HF's CPU fallback.

    No root entry and no CPU map entries: unmatched PLE uses HF's default CPU
    placement, and HF does not install mixed-device inference offload hooks.
    """
    excluded = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding'.split('.')
    keys = json.loads((Path(model_path)/'model.safetensors.index.json').read_text())['weight_map']
    placement = {}
    for name in keys:
        if name.startswith('mtp.'):
            continue
        parts = name.split('.')
        for index, (actual, omitted) in enumerate(zip(parts, excluded)):
            if actual != omitted:
                placement['.'.join(parts[:index+1])] = 0
                break
    assert placement and '' not in placement
    return placement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--target-ratio', type=float, default=.05)
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--save-on-cpu', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'reference.py').write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()

    def event(kind, **data):
        row = {**data, "event": kind, "elapsed_seconds": time.monotonic()-start}
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def save_adapter(name, model):
        state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        torch.save(state, out/(name+'.pt'))

    try:
        torch.manual_seed(20261009)
        torch.set_num_threads(8)
        batch = torch.load(args.sample, map_location='cpu', weights_only=True)
        assert isinstance(batch, dict) and all(isinstance(v, torch.Tensor) for v in batch.values())
        tokens = batch['input_ids'].shape[1]
        assert batch['input_ids'].shape[0] == 1
        assert 0 < args.prompt_tokens < tokens <= args.max_tokens
        cfg = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        assert cfg.text_config.num_experts == 256
        provenance = dict(arguments=vars(args), sample_sha256=sha256(args.sample),
                          config_sha256=sha256(Path(args.model)/'config.json'),
                          tokens=tokens, supervised_tokens=tokens-args.prompt_tokens,
                          versions={x: importlib.metadata.version(x) for x in ('torch','transformers','peft','auto-round')},
                          gpu=torch.cuda.get_device_name(0), objective='native HF prompt-masked causal cross entropy',
                          adapter_initialization='PEFT default: random A, zero B',
                          model_operators='unmodified HF and official AutoRound',
                          gradient_equivalence_proven=False)
        (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        event('load_start', tokens=tokens)
        device_map = resident_device_map(args.model)
        base, loading = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=device_map,
            local_files_only=True, trust_remote_code=False, output_loading_info=True,
            quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
        (out/'loading.json').write_text(json.dumps(loading, default=str, indent=2)+'\n')
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            if loading.get(key):
                raise RuntimeError(f'Non-exact checkpoint loading: {key}={loading[key]}')
        # Nonpersistent rotary/PLE hash buffers need explicit resident placement.
        # No weight or activation arithmetic is changed.
        for module in base.modules():
            for name, value in module._buffers.items():
                if value is not None and value.device.type != 'cuda':
                    module._buffers[name] = value.to('cuda')
        placements = {n:str(v.device) for n,v in base.named_parameters()}
        (out/'placements.json').write_text(json.dumps(placements,indent=2)+'\n')
        for name, device in placements.items():
            expected = 'cpu' if 'ple_embedding.ngram_embedding.weight' in name else 'cuda:0'
            if device != expected:
                raise RuntimeError(f'Incorrect resident placement: {name} -> {device}')
        base.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=TARGETS, bias='none', task_type='CAUSAL_LM'))
        model.train()
        if args.checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trainable = {n:v for n,v in model.named_parameters() if v.requires_grad}
        assert trainable and all('.lora_A.' in n or '.lora_B.' in n for n in trainable)
        assert all(torch.count_nonzero(v).item() == 0 for n,v in trainable.items() if '.lora_B.' in n)
        save_adapter('initial-adapter', model)
        (out/'trainable.json').write_text(json.dumps({n:dict(shape=list(v.shape),dtype=str(v.dtype)) for n,v in trainable.items()},indent=2)+'\n')
        event('load_complete', trainable_parameters=sum(v.numel() for v in trainable.values()),
              device_map=device_map, allocated_gib=torch.cuda.memory_allocated()/2**30)
        # Observational hooks report progress; they never alter inputs or outputs.
        phase = {'step':0, 'name':'forward'}
        for index, layer in enumerate(base.model.language_model.layers):
            def report(module, inputs, output, index=index):
                if index % 4 == 3:
                    event('layer', layer=index, **phase)
            layer.register_forward_hook(report)
        batch = {k:v.to('cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 for k,v in batch.items()}
        labels = batch['input_ids'].clone()
        labels[:,:args.prompt_tokens] = -100
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.learning_rate, weight_decay=0.)
        baseline = None
        best = float('inf')
        # Measurement N is taken after exactly N optimizer updates. The baseline
        # forward is also the first training forward, with zero LoRA contribution.
        for step in range(args.max_steps+1):
            phase.update(step=step,name='forward')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            step_start = time.monotonic()
            context = torch.autograd.graph.save_on_cpu(pin_memory=False) if args.save_on_cpu else nullcontext()
            event('forward_start', step=step)
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result = model(**batch, labels=labels, use_cache=False)
                loss = result.loss
                value = loss.item()
                del result
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if baseline is None:
                    baseline = value
                    if baseline <= 0:
                        raise RuntimeError('Baseline must be positive')
                ratio = value/baseline
                event('loss', step=step, loss=value, baseline_loss=baseline,
                      ratio=ratio, threshold=args.target_ratio*baseline,
                      forward_seconds=time.monotonic()-step_start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                if value < best:
                    best = value
                    save_adapter('best-adapter',model)
                status = dict(passed=ratio <= args.target_ratio, optimizer_updates=step,
                    baseline_loss=baseline, measured_loss=value, best_loss=best,
                    ratio=ratio, target_ratio=args.target_ratio, tokens=tokens,
                    supervised_tokens=tokens-args.prompt_tokens,
                    checkpointing=args.checkpointing, save_on_cpu=args.save_on_cpu,
                    gradient_equivalence_proven=False, elapsed_seconds=time.monotonic()-start)
                (out/'result.json').write_text(json.dumps(status,indent=2)+'\n')
                if status['passed'] or step == args.max_steps:
                    save_adapter('final-adapter',model)
                    event('finished', **status)
                    return
                phase['name'] = 'backward'
                event('backward_start',step=step)
                loss.backward()
                del loss
            # A gradients are legitimately zero before the first B update.
            for name, parameter in trainable.items():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'Missing/nonfinite gradient: {name}')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
            optimizer.step()
            event('update',step=step+1, gradient_norm=norm.item(),
                  seconds=time.monotonic()-step_start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except BaseException as exc:
        failure = dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        event('failed',**failure)
        raise


if __name__ == '__main__':
    main()
