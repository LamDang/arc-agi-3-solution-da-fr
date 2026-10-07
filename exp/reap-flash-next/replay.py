"""Replay one rendered sequence through the instrumented model in chunks.

Chunks share a cache, so each chunk sees the full history, as one long
prefill would. Besides the expert statistics, the replay scores the model on
its own logged generations: next-token NLL and top-1 agreement on the
GENERATED tokens. Those tokens were sampled from this model, so agreement
far below the usual range (roughly 0.7-0.9) means the replay is broken.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq

from render import GENERATED

# Masks are neither needed nor wanted: attention selects tokens itself and
# nothing is padded. Passing the mapping skips building dense causal masks.
NO_MASKS = {"indexed_attention": None, "linear_attention": None}


def truncate(enc: dict, max_tokens: int | None, vision_start_id: int) -> dict:
    """Keep the first `max_tokens` tokens without splitting an image."""
    ids = enc["input_ids"][0]
    if not max_tokens or ids.shape[0] <= max_tokens:
        return enc
    cut = max_tokens
    starts = (ids[:cut] == vision_start_id).nonzero().flatten()
    grid = enc.get("image_grid_thw")
    if grid is not None and len(starts):
        # the image starting at starts[-1] is complete only if all its tokens fit
        n_tokens = int(grid[len(starts) - 1].prod()) // 4 + 2
        if starts[-1] + n_tokens > cut:
            cut = int(starts[-1])
    out = {k: v for k, v in enc.items()}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in enc and enc[key] is not None:
            out[key] = enc[key][:, :cut]
    n_images = int((ids[:cut] == vision_start_id).sum())
    if grid is not None:
        if n_images:
            out["image_grid_thw"] = grid[:n_images]
            out["pixel_values"] = enc["pixel_values"][: int(grid[:n_images].prod(-1).sum())]
        else:
            out.pop("image_grid_thw", None)
            out.pop("pixel_values", None)
    return out


@torch.no_grad()
def replay(model, recorder, enc: dict, categories: torch.Tensor, *, chunk_tokens: int = 8192,
           measure: bool = True, collect_hidden: bool = False, predictions: bool = False,
           device="cuda", lm_block: int = 2048):
    base = model.model
    ids = enc["input_ids"].to(device)
    n = ids.shape[1]
    cats = categories.reshape(-1).to(device)
    embeds = base.get_input_embeddings()(ids)
    grid = enc.get("image_grid_thw")
    if grid is not None and len(grid):
        grid = grid.to(device)
        features = base.get_image_features(enc["pixel_values"].to(device), grid, return_dict=True).pooler_output
        features = torch.cat(features, dim=0).to(embeds.dtype)
        mask = (ids == model.config.image_token_id).unsqueeze(-1)
        embeds = embeds.masked_scatter(mask, features)
        position_ids, _ = base.get_rope_index(
            ids, mm_token_type_ids=enc["mm_token_type_ids"].to(device), image_grid_thw=grid
        )
    else:
        position_ids = torch.arange(n, device=device).view(1, 1, -1).expand(3, 1, -1)
    cache = mq.DynamicCache(config=model.config.text_config)
    lm = base.language_model
    result = {"tokens": n, "scored": 0, "nll": 0.0, "correct": 0}
    hidden, argmax, token_nll = [], [], []
    for s in range(0, n, chunk_tokens):
        e = min(n, s + chunk_tokens)
        if recorder is not None:
            recorder.categories = cats[s:e]
            recorder.positions = torch.arange(s, e, device=device)
        out = lm(
            inputs_embeds=embeds[:, s:e],
            position_ids=position_ids[:, :, s:e],
            past_key_values=cache,
            use_cache=True,
            ple_input_ids=ids[:, s:e],
            attention_mask=NO_MASKS,
        )
        last = out.last_hidden_state
        if collect_hidden:
            hidden.append(last.float().cpu())
        if measure:
            positions = torch.arange(s, min(e, n - 1), device=device)
            keep = cats[positions + 1] == GENERATED
            positions = positions[keep]
            for b in range(0, positions.shape[0], lm_block):
                p = positions[b : b + lm_block]
                logits = model.lm_head(last[0, p - s]).float()
                target = ids[0, p + 1]
                nll = F.cross_entropy(logits, target, reduction="none")
                best = logits.argmax(-1)
                result["nll"] += nll.sum().item()
                result["correct"] += int((best == target).sum())
                result["scored"] += int(p.shape[0])
                if predictions:
                    argmax.append(best.cpu())
                    token_nll.append(nll.cpu())
    if recorder is not None:
        recorder.categories = None
        recorder.positions = None
    if collect_hidden:
        result["hidden"] = torch.cat(hidden, dim=1)
    if predictions:
        result["argmax"] = torch.cat(argmax) if argmax else torch.empty(0, dtype=torch.long)
        result["token_nll"] = torch.cat(token_nll) if token_nll else torch.empty(0)
    return result
