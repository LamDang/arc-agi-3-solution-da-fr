"""Convert a gpt-6.1-sol harness run into SFT data for Qwen3.8-Flash-Next.

The run was played with ARC3_PYTHON_RATIONALE=1: the `python` tool takes
`reasoning` and `description` fields alongside `code`, and the teacher writes
its stated reasoning into them. This script turns each stated reasoning into a
`<think>` block and emits a bare `code`-only tool call, so the data looks like
a run WITHOUT that option, which is the mode the student plays in:

    <think>
    {reasoning}
    Next step : {description}
    </think>

    <tool_call><function=python><parameter=code>...</parameter></function></tool_call>

The reasoning goes in each assistant message's `reasoning_content` (the Qwen
template wraps it in <think>...</think>); the tool call keeps only `code`. The
system prompt and the `python` tool schema are rewritten to their rationale-off
form. See README.md for the format and how to render it.

    python convert.py --run-dir ../../ARC3-Inference/runs/gpt61sol-features-25games \
        --out train.jsonl --meta meta.json

One sample (one JSON line) per model request: the exact context the model was
sent for that step (already trimmed by the harness to fit the context) plus its
reply. The reply is the last message and the only training target; everything
before it is context. Each sample reproduces a real request, so each one fits
the model's context by construction (the run's largest was ~120K tokens); there
is nothing to split.
"""
from __future__ import annotations

import argparse
import json
import lzma
import re
from pathlib import Path
from typing import Any

# The system-prompt line that lists the python tool's fields in rationale mode.
# Matched on its stable opening; the rest of the line (which fields, in which
# wording) varies between harness versions, so replace the whole line.
_PROMPT_TOOL_LINE = re.compile(
    r"^- The only tool is `python`; call it with .*$", re.MULTILINE)
_PROMPT_TOOL_LINE_OFF = "- The only tool is `python`; call it with one ephemeral `code` string."

# The sentence appended to the python tool description in rationale mode. The
# code-only description is everything before it.
_TOOL_DESC_RATIONALE_MARKER = " Before the code,"


def build_think(reasoning: str, description: str) -> str:
    """The reasoning_content for a turn: the stated reasoning, then the
    decision as `Next step : ...`. The Qwen template adds the <think> tags."""
    parts: list[str] = []
    reasoning = (reasoning or "").rstrip()
    description = (description or "").strip()
    if reasoning:
        parts.append(reasoning)
    if description:
        parts.append(f"Next step : {description}")
    return "\n".join(parts)


def rationale_off_system_prompt(content: str) -> str:
    new, n = _PROMPT_TOOL_LINE.subn(_PROMPT_TOOL_LINE_OFF, content)
    if n != 1:
        raise ValueError(f"expected one python-tool prompt line, replaced {n}")
    return new


def rationale_off_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tools as a rationale-off run sends them: python takes only `code`."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") != "python":
            out.append(tool)
            continue
        description = function["description"]
        marker = description.find(_TOOL_DESC_RATIONALE_MARKER)
        if marker != -1:
            description = description[:marker].rstrip()
        properties = function["parameters"]["properties"]
        out.append({"type": "function", "function": {
            "name": "python",
            "description": description,
            "parameters": {"type": "object",
                           "properties": {"code": properties["code"]},
                           "required": ["code"]},
        }})
    return out


def parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        return json.loads(arguments) if arguments.strip() else {}
    return dict(arguments or {})


def rationale_off_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """An assistant turn as a rationale-off student emits it: the stated
    reasoning becomes reasoning_content; the tool call keeps only `code`; the
    teacher's own reasoning/reasoning_details are dropped."""
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    calls = message.get("tool_calls") or []
    new_calls: list[dict[str, Any]] = []
    think = None
    for call in calls:
        function = call["function"]
        args = parse_arguments(function.get("arguments"))
        if think is None:
            think = build_think(args.get("reasoning", ""), args.get("description", ""))
        # arguments as a mapping: the Qwen template loops over `arguments|items`
        # (an OpenAI-style JSON string fails there). code-only, for a student
        # that runs without the reasoning/description fields.
        new_call = {"type": "function",
                    "function": {"name": function["name"],
                                 "arguments": {"code": args.get("code", "")}}}
        if call.get("id"):
            new_call["id"] = call["id"]
        new_calls.append(new_call)
    if new_calls:
        out["tool_calls"] = new_calls
    if think:
        out["reasoning_content"] = think
    return out


def transform_message(message: dict[str, Any]) -> dict[str, Any]:
    if message["role"] == "assistant":
        return rationale_off_assistant(message)
    return message


def read_log(path: Path):
    """(request, reply) pairs in order from a harness request log."""
    opener = lzma.open if path.name.endswith(".xz") else open
    request = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "request":
                request = record
            elif record.get("event") == "response" and request is not None:
                yield request, record.get("reply")
                request = None


def request_sample(request: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    """One training sample reproducing a real request: the context exactly as
    the harness sent it (already trimmed to fit the model's context), rewritten
    to rationale-off form, plus the reply as the final message and sole target."""
    messages = request["messages"]
    out: list[dict[str, Any]] = [
        {"role": "system", "content": rationale_off_system_prompt(messages[0]["content"])}]
    out.extend(transform_message(m) for m in messages[1:])
    out.append(transform_message({"role": "assistant", **reply}))
    return {"messages": out,
            "tools": rationale_off_tools(request.get("tools") or []),
            "chat_template_kwargs": request.get("chat_template_kwargs") or {"preserve_thinking": True}}


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path,
                        default=here / ".." / ".." / "ARC3-Inference" / "runs" / "gpt61sol-features-25games",
                        help="the harness run directory (holds *_requests.jsonl[.xz])")
    parser.add_argument("--out", type=Path, default=here / "train.jsonl")
    parser.add_argument("--meta", type=Path, default=here / "meta.json")
    args = parser.parse_args()

    logs = sorted(args.run_dir.glob("*_requests.jsonl*"))
    if not logs:
        raise SystemExit(f"no request logs in {args.run_dir}")

    games: list[dict[str, Any]] = []
    totals = {"games": 0, "samples": 0, "no_tool_call": 0}
    with args.out.open("w", encoding="utf-8") as out:
        for path in logs:
            game = path.name.split("_")[0]
            stats = {"samples": 0, "no_tool_call": 0}
            for request, reply in read_log(path):
                if not (reply and (reply.get("tool_calls") or (reply.get("content") or "").strip())):
                    continue  # an empty reply is not a training target
                if not reply.get("tool_calls"):
                    stats["no_tool_call"] += 1
                sample = {"game": game, "request_index": stats["samples"],
                          **request_sample(request, reply)}
                out.write(json.dumps(sample, ensure_ascii=True) + "\n")
                stats["samples"] += 1
            totals["games"] += 1
            totals["samples"] += stats["samples"]
            totals["no_tool_call"] += stats["no_tool_call"]
            games.append({"game": game, **stats})
            print(f"{game}: {stats['samples']} samples"
                  + (f" ({stats['no_tool_call']} without a tool call)" if stats["no_tool_call"] else ""))

    meta = {"run": args.run_dir.name, "model": "gpt-6.1-sol",
            "unit": "one sample per model request; the last message is the only training target",
            "format": "rationale-off: reasoning+description -> <think>, code-only tool call",
            "totals": totals, "games": games}
    args.meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\n{totals['games']} games, {totals['samples']} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
