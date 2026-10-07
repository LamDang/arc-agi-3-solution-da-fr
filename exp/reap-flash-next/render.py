"""Render logged OpenAI-format messages into the token sequence the server saw.

The logs hold what the harness sent (OpenAI chat format), not token ids. The
server applied the model's chat template after conversions this module
repeats: tool-call arguments go from a JSON string to a mapping (the
template iterates them with `|items`), thinking is read from
`reasoning_content` (OpenRouter logs store it as `reasoning`), and tools are
passed as SGLang's `Tool.model_dump()` serializes them. With these, rendered
lengths equal the `prompt_tokens` SGLang logged for the Kaggle runs.
"""
from __future__ import annotations

import base64
import io
import json

import torch
from PIL import Image

# token categories, used to split the expert statistics
CONTEXT, GENERATED, IMAGE = 0, 1, 2
CATEGORY_NAMES = ("context", "generated", "image")


def _image(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("only inline data: images are expected in harness logs")
    payload = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _arguments(raw):
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return raw or {}


def to_hf_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        role = message["role"]
        converted = {"role": role}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif kind == "image_url":
                    parts.append({"type": "image", "image": _image(part["image_url"]["url"])})
                else:
                    raise ValueError(f"unexpected content part {kind!r}")
            converted["content"] = parts
        else:
            converted["content"] = content if content is not None else ""
        if role == "assistant":
            converted["reasoning_content"] = message.get("reasoning_content") or message.get("reasoning") or ""
            if message.get("tool_calls"):
                converted["tool_calls"] = [
                    {
                        "type": "function",
                        "id": call.get("id"),
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": _arguments(call["function"].get("arguments")),
                        },
                    }
                    for call in message["tool_calls"]
                ]
        if role == "tool":
            converted["tool_call_id"] = message.get("tool_call_id")
        out.append(converted)
    return out


def sglang_tools(tools: list[dict]) -> list[dict] | None:
    """Tools as SGLang's serving_chat hands them to the template:
    `Tool.model_dump()`, so `function` gains `strict: false` and the tool keeps
    a `defer_loading: null` (dropped from `function` by its serializer)."""
    out = []
    for tool in tools or []:
        function = tool.get("function", tool)
        out.append({
            "type": tool.get("type", "function"),
            "function": {
                "description": function.get("description"),
                "name": function["name"],
                "parameters": function.get("parameters"),
                "strict": bool(function.get("strict", False)),
            },
            "defer_loading": None,
        })
    return out or None


def render(processor, messages: list[dict], tools: list[dict], *, add_generation_prompt: bool,
           template_kwargs: dict | None = None):
    """Tokenize like the server. Returns the processor's BatchFeature
    (input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)."""
    kwargs = {"preserve_thinking": True}
    kwargs.update(template_kwargs or {})
    return processor.apply_chat_template(
        to_hf_messages(messages),
        tools=sglang_tools(tools),
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )


def token_categories(input_ids: torch.Tensor, tokenizer, image_token_id: int) -> torch.Tensor:
    """Per-token category: GENERATED inside assistant turns (from the token
    after `<|im_start|>assistant\\n` through `<|im_end|>`), IMAGE for image
    placeholders, CONTEXT otherwise."""
    ids = input_ids.reshape(-1)
    start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    header = [start] + assistant + newline
    cats = torch.full_like(ids, CONTEXT)
    values = ids.tolist()
    i, n = 0, len(values)
    while i < n:
        if values[i : i + len(header)] == header:
            j = i + len(header)
            while j < n and values[j] != end:
                j += 1
            cats[i + len(header) : min(j + 1, n)] = GENERATED
            i = j + 1
        else:
            i += 1
    cats[ids == image_token_id] = IMAGE
    return cats.view_as(input_ids)
