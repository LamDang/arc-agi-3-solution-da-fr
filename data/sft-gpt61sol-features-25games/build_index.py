"""Build index.json: one small row per sample in train.jsonl so a loader can
read, filter or length-bucket any sample without scanning the 320 MB file.

    python build_index.py \
        --tokenizer /path/to/Qwen3.8-Flash-Next/tokenizer.json \
        --template  /path/to/Qwen3.8-Flash-Next/chat_template.jinja

Both files come from the model repo (not committed here, ~13 MB tokenizer):

    hf download Qwen/Qwen3.8-Flash-Next tokenizer.json chat_template.jinja --local-dir qwen

Each row has:
- `line`         zero-based line number in train.jsonl
- `offset`, `length`  byte range of that line; read one sample with
                      `f.seek(offset); json.loads(f.read(length))` on the file
                      opened in binary mode (train.jsonl is ASCII)
- `game`, `request_index`  which game and step
- `level`        game level at that request (parsed from the frame opener)
- `context_tokens`  prompt length the student is conditioned on, Qwen text
                    tokens with each board image's `<|image_pad|>` expanded to
                    400 (patch 16, merge 2, 640x640)
- `output_tokens`   the trained target length (the <think> + tool call)
- `images`          board images in the context

Needs `tokenizers` and `jinja2` (both in the ARC3-Inference venv). The template
is rendered the same way as ARC3-Inference/scripts/check_chat_template.py
(standard-library jinja2 set up like transformers' apply_chat_template), kept
self-contained here so the pipeline does not depend on that script.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
_LEVEL = re.compile(r"step\s+\d+,\s*level\s+(\d+)", re.IGNORECASE)
_IMAGE_PAD_EXPANSION = 400 - 1  # <|image_pad|> is one text token; 640x640 = 400 vision tokens


def compile_template(source: str) -> jinja2.Template:
    """A jinja2 environment set up like transformers' apply_chat_template
    (trim_blocks, lstrip_blocks, tojson without ASCII escaping,
    raise_exception), so no model code is needed to render."""
    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
               separators: Any = None, sort_keys: bool = False) -> str:
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


def render(template: jinja2.Template, messages: list[dict[str, Any]],
           tools: list[dict[str, Any]] | None, kwargs: dict[str, Any],
           generation_prompt: bool = False) -> str:
    return template.render(messages=messages, tools=tools or None,
                           add_generation_prompt=generation_prompt, **kwargs)


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def parse_level(messages: list[dict[str, Any]]) -> int | None:
    """The current level, from the last `step N, level M` in the context."""
    level = None
    for message in messages:
        for match in _LEVEL.finditer(message_text(message)):
            level = int(match.group(1))
    return level


def count_images(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages for p in (m.get("content") or [])
               if isinstance(p, dict) and p.get("type") == "image_url")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=HERE / "train.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "index.json")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Qwen3.8-Flash-Next tokenizer.json")
    parser.add_argument("--template", type=Path, required=True,
                        help="Qwen3.8-Flash-Next chat_template.jinja")
    args = parser.parse_args()

    template = compile_template(args.template.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    rows: list[dict[str, Any]] = []
    offset = 0
    with args.data.open("rb") as handle:
        for line_no, raw in enumerate(handle):
            length = len(raw)
            sample = json.loads(raw)
            messages, tools = sample["messages"], sample["tools"]
            kwargs = sample.get("chat_template_kwargs") or {}
            prompt = render(template, messages[:-1], tools, kwargs, generation_prompt=True)
            full = render(template, messages, tools, kwargs)
            if not full.startswith(prompt):
                raise ValueError(f"line {line_no}: prompt is not a prefix of the sample")
            images = count_images(messages[:-1])
            context_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids) \
                + images * _IMAGE_PAD_EXPANSION
            output_tokens = len(tokenizer.encode(full[len(prompt):], add_special_tokens=False).ids)
            rows.append({"line": line_no, "offset": offset, "length": length,
                         "game": sample["game"], "request_index": sample["request_index"],
                         "level": parse_level(messages), "images": images,
                         "context_tokens": context_tokens, "output_tokens": output_tokens})
            offset += length

    args.out.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    ctx = [r["context_tokens"] for r in rows]
    out = [r["output_tokens"] for r in rows]
    print(f"{len(rows)} rows -> {args.out}")
    print(f"context_tokens: min {min(ctx)} median {sorted(ctx)[len(ctx)//2]} max {max(ctx)}")
    print(f"output_tokens:  min {min(out)} median {sorted(out)[len(out)//2]} max {max(out)}")
    print(f"levels: {sorted({r['level'] for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
