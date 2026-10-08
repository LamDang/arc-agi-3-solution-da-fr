"""Assemble SFT samples from a run's request logs and its generated thinking.

    uv run --no-sync python -m think_gen.assemble --run runs/base-gpt61sol-dfranzen \
        --thinking runs/think-sol-b1 -o runs/think-sol-b1/sft.jsonl

One sample per history stretch, as in exp/reap-flash-next/traces.py: between
history trims each request extends the previous one, so the last request of
a stretch plus its reply holds every turn of the stretch once. Every
assistant turn gets its generated thinking as `reasoning_content`, the key
the deployed SGLang server's Qwen template reads, with `preserve_thinking`
on, so it renders as a `<think>` block on every turn as at deploy time.

Thinking is normalized as the harness normalizes history (blank lines
removed); `--raw-final` keeps the final turn's thinking as generated, which
is what the server's reply looks like before the harness stores it.

Each sample lists every assistant turn's status (`ok`, `rejected`,
`teacher_empty`, `missing`) in `turn_status`, in order, so training can
drop or mask turns whose thinking failed the checks.

Messages stay in the logged OpenAI format (tool-call `arguments` as JSON
strings, images as data URLs, `content: ""` instead of null), without the
harness's private `_arc3_control` keys and without `reasoning_details`.
"""
import argparse
import hashlib
import json
from pathlib import Path

from . import context, logs
from .generate import read_jsonl

TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True}


def _hash(m: dict) -> str:
    return hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()


def stretches(records: list[logs.Record]) -> list[list[int]]:
    """Group records whose request extends the last request of a stretch."""
    hashes = [[_hash(m) for m in r.messages] for r in records]
    out: list[list[int]] = []
    for i, h in enumerate(hashes):
        for s in reversed(out):
            last = hashes[s[-1]]
            if h[:len(last)] == last:
                s.append(i)
                break
        else:
            out.append([i])
    return out


def assistant_turn(m: dict, thinking: str) -> dict:
    out = {k: v for k, v in m.items() if not k.startswith("_") and k not in context.REASONING_KEYS}
    out["role"] = "assistant"
    if out.get("content") is None:
        out["content"] = ""
    out["reasoning_content"] = thinking
    return out


def sample(records: list[logs.Record], idx: list[int], rows: dict[str, dict], raw_final: bool) -> dict:
    last = records[idx[-1]]
    by_ref = {r["ref"]: r for r in rows.values()}
    messages, status = [], []
    for m in [*last.messages, last.reply]:
        if m.get("role") != "assistant":
            messages.append({k: v for k, v in m.items() if not k.startswith("_")})
            continue
        row = by_ref.get(context.message_ref(m))
        text = (row or {}).get("thinking") or ""
        final = m is last.reply
        messages.append(assistant_turn(m, text if (final and raw_final) else context.normalize(text)))
        status.append(row["status"] if row else "missing")
    return {
        "id": f"{last.game}/s{idx[0]}-{idx[-1]}",
        "game": last.game,
        "records": [records[i].index for i in idx],
        "messages": messages,
        "tools": last.tools,
        "chat_template_kwargs": TEMPLATE_KWARGS,
        "turn_status": status,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--thinking", type=Path, required=True, help="think_gen.generate output directory")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--won-only", action="store_true", help="only game runs that ended won")
    ap.add_argument("--raw-final", action="store_true")
    args = ap.parse_args(argv)
    outcomes = logs.game_outcomes(args.run)
    n = 0
    with open(args.output, "w") as f:
        for path in logs.request_logs(args.run, args.games):
            records = logs.read_log(path)
            game = records[0].game
            if args.won_only and (outcomes.get(game) or {}).get("state") != "won":
                print(f"{game}: skipped ({(outcomes.get(game) or {}).get('state')})")
                continue
            rows = {r["key"]: r for r in read_jsonl(args.thinking / f"{game}.jsonl")}
            missing = sum(1 for r in records if r.key not in rows)
            for idx in stretches(records):
                f.write(json.dumps(sample(records, idx, rows, args.raw_final), ensure_ascii=False) + "\n")
                n += 1
            print(f"{game}: {len(stretches(records))} stretches, {len(records)} records, "
                  f"{missing} without generated thinking")
    print(f"{n} samples -> {args.output}")


if __name__ == "__main__":
    main()
