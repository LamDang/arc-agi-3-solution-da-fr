"""Exact token counts of a request, with the model's tokenizer and chat template (the "rebuilt" context mode's budget).

A TokenCounter renders a request (the system prompt, the messages and the tool schemas) the way the server does, through
the chat template of tokenizer_config.json (jinja2, as transformers renders it: the assistant `reasoning` field as the
template's reasoning_content, tool calls and tool results as it writes them, the generation prompt added), and counts
the text with tokenizers.Tokenizer; every image part counts at its vision cost (agent.image_part_tokens: one token per
32x32 patch of the PNG's real size, plus two sentinels), in place of the <|image_pad|> run the server would insert. A
list content is flattened to its text parts joined by newlines, as the server does before the template.

Where the files come from (load_counter): a directory with tokenizer.json and tokenizer_config.json, or a Hugging Face
model id fetched with huggingface_hub (cached under ~/.cache/huggingface); `--tokenizer` on run_play, else the
ARC3_TOKENIZER environment variable, else DEFAULT_TOKENIZER. With `--tokenizer-endpoint` (a vLLM server's /tokenize)
the server counts the chat messages itself, exactly, template included, and the local files are not needed. Without a
tokenizer the agent falls back to its calibrated estimate (agent.estimate_request_tokens).

Validated on the v11 sp80 run (engine_re/tools/count_check.py): see the README.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ENV_TOKENIZER = "ARC3_TOKENIZER"


def flatten_content(content: Any) -> tuple[str, int]:
    """A message's content as the template gets it (its text parts joined by newlines) and its image parts."""
    if isinstance(content, str):
        return content, 0
    if not isinstance(content, list):
        return "" if content is None else str(content), 0
    texts, images = [], 0
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            images += 1
        elif part.get("type") == "text":
            texts.append(part.get("text") or "")
    return "\n".join(texts), images


REASONING_BLOCK = "<think>\n{reasoning}\n</think>\n\n"  # how Qwen3's template writes a turn's reasoning before its text


def template_messages(messages: list[dict[str, Any]], reasoning: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The messages as the chat template takes them (content flattened, the tags and the images out) and the image
    parts (for their vision cost). With `reasoning`, an assistant message's `reasoning` field goes to the template as
    reasoning_content (which Qwen3's template writes only after the last user message: see TokenCounter)."""
    out, images = [], []
    for message in messages:
        text, _ = flatten_content(message.get("content"))
        m: dict[str, Any] = {"role": message.get("role", "user"), "content": text}
        if message.get("reasoning") and reasoning:
            m["reasoning_content"] = message["reasoning"]
        if message.get("tool_calls"):
            m["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
                               for c in message["tool_calls"]]
        if message.get("tool_call_id"):
            m["tool_call_id"] = message["tool_call_id"]
        out.append(m)
        if isinstance(message.get("content"), list):
            images += [p for p in message["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    return out, images


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None, separators: Any = None, sort_keys: bool = False) -> str:
    """transformers' tojson filter (plain json.dumps without \\u escapes), not jinja2's html-safe one."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class TokenCounter:
    """Counts a request's tokens with a tokenizer.json and the chat template of its tokenizer_config.json.

    `reasoning`: "all" (the default) counts every assistant message's reasoning as the think block the template writes
    (REASONING_BLOCK), outside the template, since the provider serving the OpenRouter runs keeps every turn's reasoning
    in the prompt (the v11 sp80 run: counted that way the error against the reported prompt_tokens is a few percent;
    as the template writes it, 13-46% under); "template" leaves it to the template, which for Qwen3 keeps only the
    reasoning after the last user message (a server that applies it verbatim, such as vLLM, bills that much less)."""

    def __init__(self, folder: Path, name: str | None = None, reasoning: str = "all"):
        from jinja2 import sandbox
        from tokenizers import Tokenizer

        self.folder = Path(folder)
        self.name = name or str(folder)
        self.reasoning = reasoning
        config = json.loads((self.folder / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = config.get("chat_template")
        if isinstance(template, list):  # several named templates: the default one
            template = next((t.get("template") for t in template if t.get("name") == "default"), template[0].get("template"))
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"{self.folder / 'tokenizer_config.json'} has no chat_template")
        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = self._raise
        self.template = env.from_string(template)
        self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))

    @staticmethod
    def _raise(message: str) -> None:
        raise ValueError(message)

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[str, list[str], list[dict[str, Any]]]:
        """The request's text as the template writes it, the reasoning blocks counted apart (every assistant message's
        with reasoning "all", none otherwise), and its image parts."""
        prepared, images = template_messages(messages, reasoning=self.reasoning != "all")
        text = self.template.render(messages=prepared, tools=tools or None, add_generation_prompt=True)
        blocks = []
        if self.reasoning == "all":
            blocks = [REASONING_BLOCK.format(reasoning=str(m["reasoning"]).strip("\n"))
                      for m in messages if m.get("role") == "assistant" and m.get("reasoning")]
        return text, blocks, images

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, int]:
        """{"tokens", "text_tokens", "reasoning_tokens", "image_tokens"} of the request (text_tokens includes the reasoning)."""
        from engine_re.agent import image_part_tokens

        text, blocks, images = self.render(messages, tools)
        reasoning_tokens = sum(self.count_text(b) for b in blocks)
        text_tokens = self.count_text(text) + reasoning_tokens
        image_tokens = sum(image_part_tokens(p) for p in images)
        return {"tokens": text_tokens + image_tokens, "text_tokens": text_tokens, "reasoning_tokens": reasoning_tokens,
                "image_tokens": image_tokens}

    def describe(self) -> str:
        return f"tokenizer {self.name} ({self.folder})"


class EndpointCounter:
    """Counts with a vLLM server's /tokenize: the chat messages are sent as they are (the server applies its own
    template, so the count is the one it will see), the images at their vision cost."""

    def __init__(self, endpoint: str, model: str | None = None, timeout: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/tokenize"):
            self.endpoint += "/tokenize"
        self.model = model
        self.timeout = timeout

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, int]:
        import requests

        from engine_re.agent import image_part_tokens

        prepared, images = template_messages(messages)
        payload: dict[str, Any] = {"messages": prepared, "add_generation_prompt": True}
        if tools:
            payload["tools"] = tools
        if self.model:
            payload["model"] = self.model
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        text_tokens = int(resp.json()["count"])
        image_tokens = sum(image_part_tokens(p) for p in images)
        return {"tokens": text_tokens + image_tokens, "text_tokens": text_tokens, "image_tokens": image_tokens}

    def describe(self) -> str:
        return f"tokenizer endpoint {self.endpoint}"


def tokenizer_folder(spec: str | None) -> Path | None:
    """The folder with the tokenizer files for `spec` (a directory, or a Hugging Face model id fetched with
    huggingface_hub: the cache first, the hub when it is not there); None when nothing can be found."""
    spec = (spec or os.environ.get(ENV_TOKENIZER) or DEFAULT_TOKENIZER).strip()
    folder = Path(spec).expanduser()
    if folder.is_dir():
        return folder if all((folder / f).exists() for f in TOKENIZER_FILES) else None
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError:
        return None
    paths = []
    for name in TOKENIZER_FILES:
        cached = try_to_load_from_cache(spec, name)
        path = cached if isinstance(cached, str) else None
        if path is None:
            try:
                path = hf_hub_download(spec, name)
            except Exception:  # noqa: BLE001  (no network, no such model: no tokenizer)
                return None
        paths.append(Path(path))
    return paths[0].parent if all(p.parent == paths[0].parent for p in paths) else None


def load_counter(spec: str | None = None, endpoint: str | None = None, model: str | None = None) -> Any:
    """The counter for a run: the server's /tokenize when `endpoint` is given, else the local files of `spec` (see
    tokenizer_folder); None when neither is available (the agent then estimates)."""
    if endpoint:
        return EndpointCounter(endpoint, model=model)
    folder = tokenizer_folder(spec)
    if folder is None:
        return None
    try:
        return TokenCounter(folder, name=spec or os.environ.get(ENV_TOKENIZER) or DEFAULT_TOKENIZER)
    except Exception:  # noqa: BLE001  (the libraries are missing or the files unreadable: no tokenizer)
        return None


__all__ = ["DEFAULT_TOKENIZER", "ENV_TOKENIZER", "EndpointCounter", "TokenCounter", "flatten_content", "load_counter",
           "template_messages", "tokenizer_folder"]
