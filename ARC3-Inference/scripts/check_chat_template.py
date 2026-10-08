"""Check a Qwen chat template before using it to render SFT data.

    uv run --no-sync python scripts/check_chat_template.py TEMPLATE
    uv run --no-sync python scripts/check_chat_template.py TEMPLATE --compare OTHER
    uv run --no-sync python scripts/check_chat_template.py TEMPLATE --kwargs '{"preserve_thinking": false}'
    uv run --no-sync python scripts/check_chat_template.py TEMPLATE --request-log runs/<run>/<game>_p0_requests.jsonl.xz

TEMPLATE is a `chat_template.jinja`, or a `tokenizer_config.json` holding a
`chat_template` key. See experiments/sft-format/README.md for where to get the
Qwen3.8-Flash-Next templates and what the output means.

Without --request-log it renders a small harness-like game (system prompt,
python tool calls, tool results, a second level opened by a user message) and
prints the full text, the tokens each assistant turn trains on, a prefix check
and the known pitfalls. With --request-log it renders every logged request
with the reply that followed it and the chat_template_kwargs the harness sent.

Prefix check: the text the server renders as the prompt for a turn
(add_generation_prompt) must be an exact prefix of the training text up to
that turn. If it is not, the model is trained on a context it never sees at
inference, and loss masks built as "full render minus prompt render" are
wrong.

Rendering uses jinja2 set up like transformers' apply_chat_template
(trim_blocks, lstrip_blocks, tojson without ASCII escaping,
raise_exception), so no model or tokenizer download is needed.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import lzma
import sys
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

# What the harness sends by default (tool_agent._harness_template_kwargs).
HARNESS_KWARGS: dict[str, Any] = {"preserve_thinking": True}

SAMPLE_TOOLS = [{"type": "function", "function": {
    "name": "python",
    "description": "Run Python against the game.",
    "parameters": {"type": "object", "properties": {
        "reasoning": {"type": "string"}, "code": {"type": "string"}},
        "required": ["reasoning", "code"]}}}]


def _call(reasoning: str, code: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": "python", "arguments": {"reasoning": reasoning, "code": code}}}]


SAMPLE_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You play an ARC-AGI-3 game."},
    {"role": "user", "content": "Level 1. Frame:\n0 0 1\n0 2 0"},
    {"role": "assistant", "reasoning_content": "The 2 is the player. Try moving up.",
     "content": "", "tool_calls": _call("test the up action", "act('ACTION1')\nprint(frame())")},
    {"role": "tool", "content": "0 2 1\n0 0 0"},
    {"role": "assistant", "reasoning_content": "It moved up. The goal 1 is to the right.",
     "content": "", "tool_calls": _call("move right onto the goal", "act('ACTION4')")},
    {"role": "tool", "content": "Level complete."},
    # The harness opens each turn with a user message, so a game has several
    # user queries; this is where preserve_thinking matters.
    {"role": "user", "content": "Level 2. Frame:\n2 0\n0 1"},
    {"role": "assistant", "reasoning_content": "Same rules: the goal is down-right.",
     "content": "", "tool_calls": _call("move down", "act('ACTION2')")},
]


def load_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".json":
        return text
    template = json.loads(text)["chat_template"]
    if isinstance(template, list):  # [{"name": "default", "template": ...}, ...]
        template = {t["name"]: t["template"] for t in template}["default"]
    return template


def compile_template(source: str) -> jinja2.Template:
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


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """A logged message in the shape Qwen templates read.

    OpenRouter returns reasoning as `reasoning`, which the template ignores;
    tool-call arguments are a JSON string in the OpenAI format, while the
    template needs a mapping. Raises ValueError on arguments that are not
    valid JSON (a reply cut off inside its tool call)."""
    message = copy.deepcopy(message)
    if message.get("role") == "assistant":
        if not isinstance(message.get("reasoning_content"), str) and isinstance(
                message.get("reasoning"), str):
            message["reasoning_content"] = message["reasoning"]
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments) if arguments.strip() else {}
    return message


def assistant_targets(template: jinja2.Template, messages: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None,
                      kwargs: dict[str, Any]) -> list[tuple[int, str | None]]:
    """(index, trained text) per assistant turn; text is None when the prefix
    check fails for that turn."""
    full = render(template, messages, tools, kwargs)
    targets = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prompt = render(template, messages[:i], tools, kwargs, generation_prompt=True)
        upto = render(template, messages[: i + 1], tools, kwargs)
        ok = upto.startswith(prompt) and full.startswith(upto)
        targets.append((i, upto[len(prompt):] if ok else None))
    return targets


def pitfalls(template: jinja2.Template, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    full = render(template, messages, tools, kwargs)
    reasoning = next(m["reasoning_content"] for m in messages if m.get("reasoning_content"))

    renamed = copy.deepcopy(messages)
    for m in renamed:
        if "reasoning_content" in m:
            m["reasoning"] = m.pop("reasoning_content")
    reasoning_key_dropped = reasoning not in render(template, renamed, tools, kwargs)

    stringified = copy.deepcopy(messages)
    for m in stringified:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    try:
        out = render(template, stringified, tools, kwargs)
        string_arguments = "renders parameters" if "<parameter=" in out else "renders no parameters"
    except Exception as exc:  # noqa: BLE001
        string_arguments = f"raises {type(exc).__name__}: {exc}"

    generation_prompt = render(template, messages[:2], tools, kwargs, generation_prompt=True)
    return {
        "reasoning under `reasoning` dropped": reasoning_key_dropped,
        "arguments as a JSON string": string_arguments,
        "generation prompt ends with": generation_prompt[-40:],
        "render differs without the harness kwargs": render(template, messages, tools, {}) != full,
    }


def check_sample(template: jinja2.Template, kwargs: dict[str, Any],
                 compare: jinja2.Template | None, compare_name: str | None) -> bool:
    full = render(template, SAMPLE_MESSAGES, SAMPLE_TOOLS, kwargs)
    print("=" * 20, "full render", "=" * 20)
    print(full)
    print("=" * 20, "assistant targets (trained text)", "=" * 20)
    ok = True
    for index, target in assistant_targets(template, SAMPLE_MESSAGES, SAMPLE_TOOLS, kwargs):
        if target is None:
            ok = False
            print(f"message {index}: PREFIX BROKEN, the inference prompt differs from the training context")
        else:
            print(f"message {index}: {target!r}")
    print("prefix check:", "OK" if ok else "FAILED")
    print("=" * 20, "pitfalls", "=" * 20)
    for name, value in pitfalls(template, SAMPLE_MESSAGES, SAMPLE_TOOLS, kwargs).items():
        print(f"{name}: {value!r}")
    if compare is not None:
        other = render(compare, SAMPLE_MESSAGES, SAMPLE_TOOLS, kwargs)
        print("=" * 20, f"render diff against {compare_name}", "=" * 20)
        diff = difflib.unified_diff(full.splitlines(keepends=True), other.splitlines(keepends=True),
                                    "template", str(compare_name))
        sys.stdout.writelines(list(diff) or ["(identical)\n"])
    return ok


def read_request_log(path: Path):
    """(request record, reply) pairs from a harness request log."""
    opener = lzma.open if path.name.endswith(".xz") else open
    request = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "request" or "messages" in record and "reply" not in record:
                request = record
            elif record.get("event") == "response" and request is not None:
                yield request, record.get("reply")
                request = None


def check_request_log(template: jinja2.Template, path: Path, override: dict[str, Any] | None) -> bool:
    counts = {"requests": 0, "ok": 0, "prefix broken": 0, "bad arguments": 0,
              "render error": 0, "no reply": 0, "extends previous": 0}
    previous_text = None
    for index, (request, reply) in enumerate(read_request_log(path)):
        counts["requests"] += 1
        if not isinstance(reply, dict):
            counts["no reply"] += 1
            continue
        kwargs = override if override is not None else request.get("chat_template_kwargs") or {}
        tools = request.get("tools")
        try:
            messages = [normalize_message(m) for m in request["messages"]]
            messages.append(normalize_message({"role": "assistant", **reply}))
        except ValueError:
            counts["bad arguments"] += 1
            print(f"request {index}: tool-call arguments are not valid JSON")
            continue
        try:
            prompt = render(template, messages[:-1], tools, kwargs, generation_prompt=True)
            full = render(template, messages, tools, kwargs)
        except Exception as exc:  # noqa: BLE001
            counts["render error"] += 1
            print(f"request {index}: {type(exc).__name__}: {exc}")
            continue
        if full.startswith(prompt):
            counts["ok"] += 1
        else:
            counts["prefix broken"] += 1
            print(f"request {index}: prefix broken")
        # Consecutive requests can be merged into one multi-turn sample only
        # when the next prompt starts with this request's prompt and reply.
        if previous_text is not None and prompt.startswith(previous_text):
            counts["extends previous"] += 1
        previous_text = full
    for name, value in counts.items():
        print(f"{name}: {value}")
    return counts["requests"] > 0 and counts["ok"] == counts["requests"] - counts["no reply"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("template", type=Path, help="chat_template.jinja or tokenizer_config.json")
    parser.add_argument("--compare", type=Path, help="another template to diff the sample render against")
    parser.add_argument("--kwargs", help="chat_template_kwargs as JSON; default: what the harness sends "
                        "(with --request-log: what each request logged)")
    parser.add_argument("--request-log", type=Path, help="a harness *_requests.jsonl[.xz] to check")
    args = parser.parse_args()

    template = compile_template(load_template(args.template))
    override = json.loads(args.kwargs) if args.kwargs else None
    if args.request_log:
        return 0 if check_request_log(template, args.request_log, override) else 1
    compare = compile_template(load_template(args.compare)) if args.compare else None
    ok = check_sample(template, override if override is not None else HARNESS_KWARGS,
                      compare, args.compare)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
