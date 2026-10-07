"""Write review bundles: one readable file per request for error analysis.

    uv run --no-sync python -m think_gen.review --run runs/base-max-dfranzen \
        --manifest experiments/teacher-reasoning/evalset/max5.json --split dev \
        --arm sum=runs/think-eval-b1/sum --arm nosum=runs/think-eval-b1/nosum \
        -o runs/think-eval-b1/review

For each request: `<key>.md` with the newest messages, the call, the summary,
the real thinking and each arm's generated thinking; and `<key>.context.txt`
with the whole context as text (images as `[image]`), to check claims
against the frames and tool outputs. The shared system prompt goes to
`system_prompt.txt` once.
"""
import argparse
import json
from pathlib import Path

from . import logs
from .generate import read_jsonl


def message_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text", "") if p.get("type") == "text" else "[image]" for p in c)
    parts = []
    if m.get("reasoning"):
        parts.append(f"<thinking>\n{m['reasoning']}\n</thinking>")
    if c:
        parts.append(str(c))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            body = args.get("code") if isinstance(args, dict) and "code" in args else json.dumps(args)
        except json.JSONDecodeError:
            body = fn.get("arguments")
        parts.append(f"<tool_call {fn.get('name')}>\n{body}\n</tool_call>")
    return "\n".join(parts)


def context_text(messages: list, skip_system: bool = True) -> str:
    out = []
    for i, m in enumerate(messages):
        if skip_system and m.get("role") == "system":
            out.append(f"### [{i}] system (see system_prompt.txt)")
            continue
        out.append(f"### [{i}] {m.get('role')}\n{message_text(m)}")
    return "\n\n".join(out)


def bundle(rec: logs.Record, item: dict, arms: dict[str, dict], last_n: int = 6) -> str:
    msgs = rec.messages
    tail = msgs[-last_n:]
    lines = [
        f"# {rec.key}",
        f"game position {item['position']} (0 = start, 1 = end), analysis step {rec.analysis_step}."
        f"{rec.request_in_turn}, real thinking {len(rec.real_reasoning)} chars, "
        f"prompt {item.get('prompt_tokens')} tokens, length bin {item['length_bin']} of 0-4",
        f"Full context: {rec.key.replace('#', '_')}.context.txt ({len(msgs)} messages).",
        "",
        f"## Last {len(tail)} messages of the context (the agent's earlier thinking shown in <thinking>)",
        context_text(tail),
        "",
        "## The output (the call that followed)",
        message_text(rec.reply),
        "",
        "## Real thinking (what the agent actually thought before the output)",
        rec.real_reasoning,
    ]
    for name, rows in arms.items():
        r = rows.get(rec.key)
        if not r:
            continue
        lines += ["", f"## Generated thinking: {name}"]
        if r.get("summary"):
            lines += ["Summary given to the generator:", "<summary>", r["summary"], "</summary>", ""]
        else:
            lines += ["(no summary given to the generator)", ""]
        lines.append(r.get("thinking") or "(empty)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--arm", action="append", default=[], help="name=generate output dir")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args(argv)
    items = {i["key"]: i for i in json.loads(args.manifest.read_text())["items"]
             if args.split in ("all", i["split"])}
    arms = {}
    for a in args.arm:
        name, d = a.split("=", 1)
        arms[name] = {r["key"]: r for p in sorted(Path(d).glob("*.jsonl")) for r in read_jsonl(p)}
    args.output.mkdir(parents=True, exist_ok=True)
    n, system = 0, None
    for path in logs.request_logs(args.run):
        for rec in logs.read_log(path):
            if rec.key not in items:
                continue
            system = system or next((m["content"] for m in rec.messages if m.get("role") == "system"), None)
            stem = rec.key.replace("#", "_")
            (args.output / f"{stem}.md").write_text(bundle(rec, items[rec.key], arms))
            (args.output / f"{stem}.context.txt").write_text(context_text(rec.messages))
            n += 1
    if system:
        (args.output / "system_prompt.txt").write_text(system)
    print(f"{n} bundles -> {args.output}")


if __name__ == "__main__":
    main()
