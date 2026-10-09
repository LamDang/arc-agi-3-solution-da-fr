"""All-assistant supervision, with verified processor-prefix boundaries.

The final-reply reference encoder remains untouched. Each assistant prefix is
encoded by that reference; its exact suffix must occur in the complete input.
"""
from __future__ import annotations

import hashlib

from dataset import digest
from data import encode as encode_reply, hf_messages, render


def encode(processor, sample):
    if sample.get("chat_template_kwargs", {}).get("preserve_thinking") is not True:
        raise ValueError("Trajectory encoding requires preserve_thinking=true")
    enc, _ = encode_reply(processor, sample)
    ids = enc["input_ids"][0].tolist()
    positions, target_ids, labels, turns = [], [], [], []
    counts = {}
    for index, message in enumerate(sample["messages"]):
        if message.get("role") != "assistant":
            continue
        prefix = dict(sample, messages=sample["messages"][:index + 1])
        partial, annotation = encode_reply(processor, prefix)
        prefix_ids = partial["input_ids"][0].tolist()
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"Assistant {index}: processor prefix changed inside full trajectory")
        for key in ('image_grid_thw', 'pixel_values'):
            value = partial.get(key)
            full_value = enc.get(key)
            if value is not None and (full_value is None or not full_value[:len(value)].equal(value)):
                raise ValueError(f'Assistant {index}: image processor prefix changed: {key}')
        if positions and annotation["positions"][0] <= positions[-1]:
            raise ValueError("Assistant targets overlap or are not chronological")
        positions.extend(annotation["positions"])
        target_ids.extend(annotation["target_ids"])
        labels.extend(annotation["labels"])
        for name, count in annotation["category_counts"].items():
            counts[name] = counts.get(name, 0) + count
        if message.get("reasoning_content") and not annotation["category_counts"]["thinking"]:
            raise ValueError(f"Assistant {index}: thinking vanished from template")
        turns.append(dict(message_index=index, positions=annotation["positions"],
                          target_tokens=annotation["target_tokens"],
                          category_counts=annotation["category_counts"]))
        del partial
    if not positions or any(p <= 0 for p in positions):
        raise ValueError("Trajectory has no causally predictable assistant targets")
    if [ids[p] for p in positions] != target_ids:
        raise ValueError("Trajectory target IDs do not match input positions")
    return enc, dict(total_tokens=len(ids), target_tokens=len(positions),
        non_target_tokens=len(ids)-len(positions),
        count_definitions={'non_target_tokens': 'all input tokens outside supervised assistant positions; not a contiguous prompt boundary'},
        vision_sha256=vision_hashes(enc), images=len(enc.get("image_grid_thw", [])),
        positions=positions, target_ids=target_ids, labels=labels,
        category_counts=counts, assistant_turns=turns,
        input_sha256=digest(ids), target_sha256=digest(target_ids),
        positions_sha256=digest(positions), objective="all-assistant-token-nll")


def encode_input(processor, sample):
    """Single full processor pass for an already verified immutable annotation."""
    return render(processor, hf_messages(sample['messages']), sample, tokenize=True)


def vision_hashes(enc):
    """Bind exact dtype, shape and tensor bytes for processor vision outputs."""
    import torch
    result = {}
    for key in ('pixel_values', 'image_grid_thw'):
        value = enc.get(key)
        if value is None:
            result[key] = None
            continue
        tensor = value.detach().cpu().contiguous()
        header = digest({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)})
        raw = tensor.view(torch.uint8).numpy().tobytes()
        result[key] = hashlib.sha256(header.encode()+raw).hexdigest()
    return result
