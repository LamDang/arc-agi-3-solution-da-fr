"""Print a condensed view of an agent session.

    uv run --no-sync python -m engine_re.show_transcript runs/engine-re/<name>/<game> [--chars 300] [--from-turn N]
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
        print("\nTests (turn: exact/total, first mismatch):")
        for line in tests.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            level = f" from_level={t['from_level']}" if t.get("from_level") else ""
            print(f"  turn {t['turn']}{level}: {t['exact']}/{t['total']}, first mismatch {t['first_fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
