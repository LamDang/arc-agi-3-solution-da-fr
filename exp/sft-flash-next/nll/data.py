"""Final-reply encoding and semantic spans, with strict round-trip checks."""
from __future__ import annotations

import base64
import copy
import io
import itertools
import re

import numpy as np
from PIL import Image

from common import LABELS, digest


def load_processor(path):
    from transformers import AutoImageProcessor, AutoProcessor
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    # Passing backend="pil" to AutoProcessor also forwards it to the video
    # processor, whose backend property is read-only in Transformers 5.18.
    # Set only the image subprocessor; all other pinned settings stay intact.
    processor.image_processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, backend="pil")
    return processor


def hf_messages(messages):
    out = copy.deepcopy(messages)
    for message in out:
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part["type"] == "text":
                    parts.append(part)
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError("Expected inline base64 image")
                    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as im:
                        parts.append({"type": "image", "image": im.convert("RGB")})
                else:
                    raise ValueError(f"Unexpected content type: {part['type']}")
            message["content"] = parts
        for call in message.get("tool_calls", []):
            function = call["function"]
            if function["name"] != "python" or not isinstance(function.get("arguments"), dict):
                raise ValueError("Expected Python tool with a mapping of arguments")
            if set(function["arguments"]) != {"code"} or not isinstance(function["arguments"]["code"], str):
                raise ValueError("Expected code-only Python arguments")
    if not out or out[-1]["role"] != "assistant":
        raise ValueError("The final message must be the teacher assistant reply")
    return out


def render(processor, messages, sample, generation=False, tokenize=False):
    return processor.apply_chat_template(
        messages, tools=sample.get("tools") or None,
        add_generation_prompt=generation, tokenize=tokenize,
        **({"return_dict": True, "return_tensors": "pt", "processor_kwargs": {"add_special_tokens": False}}
           if tokenize else {}),
        **(sample.get("chat_template_kwargs") or {}),
    )


def character_labels(messages, renderer, full, target_start=0):
    """Tag a shadow rendering only; no annotation markers enter model inputs.

    Removing all shadow markers must exactly reconstruct the official render.
    Trying common trim conventions accommodates templates which strip field
    whitespace. Any unverified mapping fails closed.
    """
    prefix = "__SOL_NLL_" + digest(full)[:16] + "_"
    if prefix in full:
        raise ValueError("Annotation marker collision")
    trims = (lambda x: x, lambda x: x.strip("\n"), lambda x: x.strip())
    for trim_reason, trim_content in itertools.product(trims, repeat=2):
        shadow = [*messages[:-1], copy.deepcopy(messages[-1])]
        markers = []

        def marked(value, label):
            # Code whitespace is executable content; never trim it to make
            # an annotation fit. Templates may trim rationale/prose separately.
            value = trim_reason(value) if label == "thinking" else trim_content(value) if label == "assistant_text" else value
            if not value:
                return value
            i = len(markers)
            start, end = f"{prefix}{i}_BEGIN__", f"{prefix}{i}_END__"
            markers.append((start, end, label))
            return start + value + end

        final = shadow[-1]
        if final.get("reasoning_content"):
            final["reasoning_content"] = marked(final["reasoning_content"], "thinking")
        if final.get("content"):
            if not isinstance(final["content"], str):
                raise ValueError("Only text final assistant content is supported")
            final["content"] = marked(final["content"], "assistant_text")
        for call in final.get("tool_calls", []):
            args = call["function"]["arguments"]
            args["code"] = marked(args["code"], "tool_code")
        annotated = renderer(shadow)
        events = {}
        for start, end, label in markers:
            events[start], events[end] = (True, label), (False, label)
        pattern = re.compile("|".join(re.escape(k) for k in events)) if events else None
        clean, spans, cursor, active, offset = [], [], 0, None, 0
        for match in pattern.finditer(annotated) if pattern else []:
            piece = annotated[cursor:match.start()]
            clean.append(piece)
            if active is not None:
                spans.append((offset, offset + len(piece), active))
            offset += len(piece)
            opening, label = events[match.group()]
            if (opening and active is not None) or (not opening and active != label):
                raise ValueError("Invalid annotation nesting")
            active = label if opening else None
            cursor = match.end()
        clean.append(annotated[cursor:])
        if "".join(clean) != full or active is not None or len(spans) != len(markers):
            continue
        labels = np.full(len(full), LABELS.index("turn_format"), dtype=np.uint8)
        skeleton = list(full)
        for start, end, label in spans:
            labels[start:end] = LABELS.index(label)
            skeleton[start:end] = " " * (end - start)
        # Field contents are blanked first: literal tool tags inside code or
        # rationale cannot masquerade as the template's structural wrappers.
        skeleton = "".join(skeleton)
        for match in re.finditer(r"<tool_call>[\s\S]*?</tool_call>", skeleton[target_start:]):
            part = labels[target_start+match.start():target_start+match.end()]
            part[part == LABELS.index("turn_format")] = LABELS.index("tool_format")
        return labels
    raise ValueError("Semantic annotation did not round-trip through the pinned template")


def labels_from_offsets(offsets, char_labels):
    result = []
    for start, end in offsets:
        if not (0 <= start < end <= len(char_labels)):
            result.append(LABELS.index("boundary"))
            continue
        kinds = set(char_labels[start:end].tolist())
        result.append(kinds.pop() if len(kinds) == 1 else LABELS.index("boundary"))
    return np.asarray(result, dtype=np.uint8)


def encode(processor, sample):
    messages = hf_messages(sample["messages"])
    full = render(processor, messages, sample)
    prompt = render(processor, messages[:-1], sample, generation=True)
    if not full.startswith(prompt):
        raise ValueError("Prompt string is not a prefix of full transcript")
    enc = render(processor, messages, sample, tokenize=True)
    prompt_enc = render(processor, messages[:-1], sample, generation=True, tokenize=True)
    ids = enc["input_ids"][0].tolist()
    prompt_ids = prompt_enc["input_ids"][0].tolist()
    p = len(prompt_ids)
    if ids[:p] != prompt_ids or not 0 < p < len(ids):
        raise ValueError("Processor token prefix or nonempty final target check failed")
    for key in ("image_grid_thw", "pixel_values", "mm_token_type_ids"):
        if key == "mm_token_type_ids":
            continue
        a, b = enc.get(key), prompt_enc.get(key)
        if (a is None) != (b is None) or (a is not None and not a.equal(b)):
            raise ValueError(f"Prompt/full image processing differs: {key}")
    chars = character_labels(messages, lambda ms: render(processor, ms, sample), full, len(prompt))
    tokens = processor.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokens["offset_mapping"]
    # Processor expands image placeholders, but the final text-only target
    # must be an identical suffix of the text-tokenizer's full rendering.
    n_target = len(ids) - p
    if tokens["input_ids"][-n_target:] != ids[p:]:
        raise ValueError("Text/processor final-target suffix mismatch")
    target_offsets = offsets[-n_target:]
    if any(start < len(prompt) for start, end in target_offsets if end > start):
        raise ValueError("Target token overlaps prompt characters")
    category = labels_from_offsets(target_offsets, chars)
    target_text = full[len(prompt):]
    relative_offsets = [(max(0, a-len(prompt)), max(0, b-len(prompt))) for a, b in target_offsets]
    metadata = dict(prompt_tokens=p, target_tokens=n_target, total_tokens=len(ids),
                    images=len(enc.get("image_grid_thw", [])),
                    input_sha256=digest(ids), target_sha256=digest(ids[p:]),
                    target_ids=ids[p:], positions=list(range(p, len(ids))),
                    labels=category.tolist(), offsets=relative_offsets, target_text=target_text,
                    category_counts={label: int((category == i).sum()) for i, label in enumerate(LABELS)})
    return enc, metadata
