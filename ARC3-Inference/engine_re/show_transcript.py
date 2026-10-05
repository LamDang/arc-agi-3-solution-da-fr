"""Print a condensed view of an agent session.

    uv run --no-sync python -m engine_re.show_transcript runs/engine-re/<name>/<game> [--chars 300] [--from-turn N] [--diffs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _one_line(text: str | None, limit: int) -> str:
    text = (text or "").replace("\n", " | ")
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_dir", type=Path)
    parser.add_argument("--chars", type=int, default=300)
    parser.add_argument("--from-turn", type=int, default=0)
    parser.add_argument("--reasoning", action="store_true", help="Also show the model's reasoning.")
    parser.add_argument("--diffs", action="store_true", help="Also show the diff of every change to engine.py.")
    args = parser.parse_args()

    for line in (args.game_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["turn"] < args.from_turn:
            continue
        if "tool" in record:
            print(f"    <- {record['tool']} ({record['seconds']}s): {_one_line(record['output'], args.chars)}")
            continue
        if "nudge" in record:
            print(f"    [harness nudge after {record['nudge']} turns without a test]")
            continue
        if "auto_test" in record:
            print(f"    [harness auto-test] {_one_line(record['auto_test'], args.chars)}")
            continue
        if "images" in record:
            print(f"    [harness images sent] {', '.join(record['images'])}")
            continue
        if "show_images" in record:
            print(f"    [show_frames()] {', '.join(record['show_images'])}")
            continue
        if "engine_change" in record:
            change = record["engine_change"]
            where = f" (lines {change['lines']})" if change.get("lines") else ""
            print(f"    [engine.py v{change['version']}] {change['op']}: {change['summary']}{where}")
            if args.diffs and change.get("diff"):
                print("\n".join("        " + line for line in change["diff"].splitlines()))
            continue
        if "move" in record:  # the play agent (engine_re.play_agent)
            m = record["move"]
            ok = "unchecked" if m.get("ok") is None else "matches" if m["ok"] else f"DIFFERS: {m.get('verdict')}"
            print(f"    [move #{m['index']} {m['label']}] {ok} ({m.get('state')}, levels {m.get('levels_completed')})")
            continue
        if "batch" in record:
            b = record["batch"]
            print(f"    [batch] sent {b['sent']} of {len(b['moves'])}, matched {b['matched']}, mismatch {b.get('mismatch')}: {_one_line(b.get('note'), args.chars)}")
            continue
        if "plan" in record:
            print(f"    [PLAN] step {record['plan'].get('step')}, level {record['plan'].get('level')}, actions {record['plan'].get('actions')}")
            continue
        if "step_start" in record:
            print(f"    [FIT] step {record['step_start']['step']}: {_one_line(record['step_start'].get('verdict'), args.chars)}")
            continue
        if "advance" in record:
            print(f"    [advance] fixed {record['advance']['fixed']}, next failing step {record['advance']['next']}")
            continue
        if "commit" in record:
            c = record["commit"]
            print(f"    [commit{' (implicit)' if c.get('implicit') else ''}] fixed {c.get('fixed')}, next {c.get('next')}: {_one_line(c.get('message'), args.chars)}")
            continue
        for key in ("plan_nudge", "fit_escape", "out_of_sync", "resync", "refused_batch"):
            if key in record:
                print(f"    [{key}] {json.dumps(record[key])}")
                break
        else:
            key = None
        if key is not None:
            continue
        if "finish_reason" not in record:  # messages, appends, hidden images, replays: shown by their own tools
            continue
        usage = record.get("usage") or {}
        print(
            f"[turn {record['turn']}] prompt={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
            f"finish={record.get('finish_reason')}"
        )
        if args.reasoning and record.get("reasoning"):
            print(f"    thinking: {_one_line(record['reasoning'], args.chars)}")
        if record.get("content"):
            print(f"    says: {_one_line(record['content'], args.chars)}")
        for call in record.get("tool_calls") or []:
            print(f"    -> {call['function']['name']}: {_one_line(call['function']['arguments'], args.chars)}")
    tests = args.game_dir / "tests.jsonl"
    if tests.exists():
        print("\nTests (turn: exact/total, steps passing before the first failure, first failing step):")
        for line in tests.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            if t.get("level") is not None:
                level = f" level={t['level']} only"
            else:
                level = f" from_level={t['from_level']}" if t.get("from_level") else ""
            prefix = t.get("passing_prefix", "?")
            print(f"  turn {t['turn']}{level}: {t['exact']}/{t['total']}, {prefix} before the first failure, step {t['first_fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
