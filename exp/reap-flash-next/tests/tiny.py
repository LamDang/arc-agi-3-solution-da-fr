"""A tiny random Flash-Next and a fake Intel-format checkpoint of it.

Same architecture as the real model (hyper-connections, PLE n-gram table,
gated delta-net, QSA indexer, vision tower), small sizes. The experts are
quantized the way the Intel checkpoint is (GPTQ v1, symmetric int4, packed
along the input dimension) and the reference model holds exactly the
dequantized values, so the instrumented model and the reference compute the
same function.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq

FIXTURES = Path(__file__).parent / "fixtures"
GROUP = 16


def tiny_config():
    raw = json.loads((FIXTURES / "flash_next_config.json").read_text())
    text = raw["text_config"]
    text.update(
        hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_key_head_dim=16, linear_num_value_heads=4, linear_value_head_dim=16,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=16, num_experts_per_tok=4,
        hc_lowrank=16, indexer_n_heads=2, indexer_head_dim=16, indexer_budget=16, indexer_compress_ratio=4,
        ple_embed_dim=64, ngram_vocab_size_base=1000, split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8,
        mtp_num_hidden_layers=0, dtype="float32",
    )
    text["rope_parameters"] = dict(text["rope_parameters"], mrope_section=[2, 1, 1])
    raw["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    raw.pop("transformers_version", None)
    path = Path(tempfile.mkdtemp(prefix="tiny_flash_next_"))
    (path / "config.json").write_text(json.dumps(raw))
    config = AutoConfig.from_pretrained(path)
    config._attn_implementation = "sdpa"
    return config


def quantize(w_in_out: torch.Tensor, group: int = GROUP):
    """[K, N] float -> qweight [K/8, N] int32, scales [K/g, N] fp16, qzeros,
    and the exact dequantized [K, N] matrix."""
    K, N = w_in_out.shape
    groups = w_in_out.reshape(K // group, group, N)
    scales = (groups.abs().amax(dim=1) / 7.0).clamp_min(1e-4).to(torch.float16)
    s = scales.to(torch.float32).repeat_interleave(group, dim=0)
    q = torch.clamp(torch.round(w_in_out / s) + 8, 0, 15).to(torch.int64)
    dq = (q.to(torch.float32) - 8) * s
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    qn = q.numpy().astype(np.uint32).reshape(K // 8, 8, N)
    for i in range(8):
        packed |= qn[:, i, :] << np.uint32(4 * i)
    qweight = torch.from_numpy(packed.view(np.int32))
    qzeros = torch.full((K // group, N // 8), 0x77777777, dtype=torch.int32)
    return qweight, scales, qzeros, dq


def reference_model(seed: int = 0):
    torch.manual_seed(seed)
    config = tiny_config()
    model = mq.Qwen4ExpForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            module = name.rsplit(".", 2)[-2]
            if "norm" in module:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "A_log" in name or "dt_bias" in name:
                p.copy_(0.5 * torch.randn_like(p))
            else:
                p.copy_(0.1 * torch.randn_like(p))
    return model


def write_checkpoint(model, out_dir: Path) -> dict:
    """Intel-format checkpoint of `model`; replaces the model's expert weights
    with their dequantized values. Returns the quantized tensors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = model.config.text_config
    I = text.moe_intermediate_size
    main, ple, quant = {}, {}, {}
    for name, tensor in model.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
            continue
        if name.endswith("ngram_embedding.weight"):
            prefix = name[: -len(".weight")]
            for k, part in enumerate(torch.tensor_split(tensor, text.split_ngram_parts, dim=0)):
                ple[f"{prefix}.shard_{k}.weight"] = part.contiguous()
            continue
        main[name] = tensor.contiguous()
    with torch.no_grad():
        for i, layer in enumerate(model.model.language_model.layers):
            experts = layer.mlp.experts
            for e in range(text.num_experts):
                gate_up = experts.gate_up_proj[e].T  # [H, 2I] (in, out)
                down = experts.down_proj[e].T  # [I, H]
                parts = {"gate_proj": gate_up[:, :I], "up_proj": gate_up[:, I:], "down_proj": down}
                dq = {}
                for proj, w in parts.items():
                    qweight, scales, qzeros, dq[proj] = quantize(w.contiguous())
                    key = f"model.language_model.layers.{i}.mlp.experts.{e}.{proj}"
                    main[f"{key}.qweight"], main[f"{key}.scales"], main[f"{key}.qzeros"] = qweight, scales, qzeros
                    quant[(i, e, proj)] = (qweight, scales)
                experts.gate_up_proj[e].copy_(torch.cat([dq["gate_proj"], dq["up_proj"]], dim=1).T)
                experts.down_proj[e].copy_(dq["down_proj"].T)
    save_file(main, str(out_dir / "model-00001-of-00001.safetensors"))
    save_file(ple, str(out_dir / "model-ple-00001-of-00001.safetensors"))
    weight_map = {k: "model-00001-of-00001.safetensors" for k in main}
    weight_map.update({k: "model-ple-00001-of-00001.safetensors" for k in ple})
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = model.config.to_dict()
    config["quantization_config"] = dict(
        json.loads((FIXTURES / "quantization_config.json").read_text()), group_size=GROUP
    )
    (out_dir / "config.json").write_text(json.dumps(config))
    return quant
