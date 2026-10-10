"""HF + PEFT reference: frozen existing 4-bit base, trainable LoRA only.

No repository model implementation is imported. The paired overfit runner
qualified HF loading, backward and learning on the full 256-expert checkpoint
with a real 16K window. This standalone gradient-export variant has not yet
been run; use the recorded overfit experiment as the measured evidence.
A small PEFT operator-chain test did
propagate gradients through an actual frozen expert using the official
auto_round:tritonv2_zp backend. HF's is_trainable=False metadata alone does not
prove that frozen-weight input gradients are unavailable. Do not substitute
NF4 or another quantization and call it an equivalent reference.

Both sides must use the same 256-expert checkpoint, adapter targets/state,
dtypes, autocast, input and loss. Freeze the base without bulk FP32 upcasting:
PEFT's generic k-bit helper would change base dtypes and double the 95.37 GiB
PLE table. Freezing plus PEFT LoRA is sufficient without checkpointing.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoRoundConfig
from overfit_hf_reference import resident_device_map
from peft import (
    LoraConfig, get_peft_model,
    get_peft_model_state_dict, set_peft_model_state_dict,
)

# Language projections only; exclude routed experts, routers, indexers and vision.
TARGETS = (r'^model\.language_model\.layers\.\d+\.(?:'
           r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
           r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|'
           r'mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--prompt-tokens', required=True, type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--adapter-state', help='Optional matching PEFT state saved with torch.save')
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--expected-experts', type=int, default=256)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--save-on-cpu', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20261009)
    batch = torch.load(args.sample, map_location='cpu', weights_only=True)
    assert isinstance(batch, dict) and all(isinstance(x, torch.Tensor) for x in batch.values())
    assert batch['input_ids'].shape[0] == 1
    assert 0 < args.prompt_tokens < batch['input_ids'].shape[1] <= args.max_tokens

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False, local_files_only=True)
    if config.text_config.num_experts != args.expected_experts:
        raise ValueError(f'Reference needs the matching {args.expected_experts}-expert checkpoint; '
                         f'provided checkpoint has {config.text_config.num_experts}')

    # Honor the checkpoint's existing quantization, without requantizing it.
    base, loading = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=resident_device_map(args.model),
        trust_remote_code=False, local_files_only=True, output_loading_info=True,
        # HF merges only this loading attribute; stored codes/scales stay unchanged.
        quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'),
    )
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(field):
            raise RuntimeError(f'HF checkpoint did not load exactly: {field}={loading[field]}')
    assert base.config.text_config.num_experts == args.expected_experts
    # The omitted PLE table stays on CPU via HF's native default placement.
    # Move any nonpersistent rotary/hash buffers to the decoder device.
    for module in base.modules():
        for name, value in module._buffers.items():
            if value is not None and value.device.type != 'cuda':
                module._buffers[name] = value.to('cuda')
    for name, parameter in base.named_parameters():
        expected = 'cpu' if 'ple_embedding.ngram_embedding.weight' in name else 'cuda:0'
        assert str(parameter.device) == expected, (name, parameter.device)
    quant = base.config.quantization_config
    quant = quant.to_dict() if hasattr(quant, 'to_dict') else quant
    if not (quant.get('bits') == 4 or quant.get('load_in_4bit') or quant.get('_load_in_4bit')):
        raise ValueError('Reference requires the same existing 4-bit checkpoint')
    quantizer = getattr(base, 'hf_quantizer', None)
    hf_trainable_flag = None if quantizer is None else quantizer.is_trainable
    # The flag describes HF integration support; frozen weights still need only
    # an input VJP. Do not reject a measured backward-capable kernel by this flag.
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias='none',
        target_modules=TARGETS, task_type='CAUSAL_LM',
    ))
    model.train()
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all('.lora_A.' in n or '.lora_B.' in n for n in trainable)

    # Audit-only nonzero A and B: zero B would hide A-gradient errors.
    # The optimized side must load these exact bytes, not just reuse the seed.
    if args.adapter_state:
        state = torch.load(args.adapter_state, map_location='cpu', weights_only=True)
        current = get_peft_model_state_dict(model)
        assert set(state) == set(current)
        set_peft_model_state_dict(model, state)
        for name, value in get_peft_model_state_dict(model).items():
            torch.testing.assert_close(value.cpu(), state[name], rtol=0, atol=0)
    else:
        with torch.no_grad():
            for p in trainable.values():
                p.normal_(0, .003)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.save({n: p.detach().cpu().clone() for n, p in get_peft_model_state_dict(model).items()},
               out / 'initial-adapter.pt')
    (out / 'parameter-dtypes.json').write_text(json.dumps(
        {name: str(p.dtype) for name, p in model.named_parameters()}, indent=2) + '\n')

    batch = {name: value.to(device='cuda:0', dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for name, value in batch.items()}
    labels = batch['input_ids'].clone()
    labels[:, :args.prompt_tokens] = -100
    model.zero_grad(set_to_none=True)
    context = torch.autograd.graph.save_on_cpu(pin_memory=False) if args.save_on_cpu else nullcontext()
    with context:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**batch, labels=labels, use_cache=False)
        result.loss.backward()

    gradients = {}
    for name, p in trainable.items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f'Missing or nonfinite adapter gradient: {name}')
        gradients[name] = p.grad.detach().cpu().clone()
    torch.save(gradients, out / 'gradients.pt')
    (out / 'result.json').write_text(json.dumps(dict(
        loss=result.loss.item(), tokens=labels.shape[1], trainable=list(trainable),
        model=args.model, experts=args.expected_experts, rank=16, alpha=32,
        autocast='bfloat16', gradient_checkpointing=args.checkpointing, save_on_cpu=args.save_on_cpu,
        quantization_backend='auto_round:tritonv2_zp', hf_trainable_flag=hf_trainable_flag,
        optimizer_updates=0), indent=2) + '\n')


if __name__ == '__main__':
    main()
