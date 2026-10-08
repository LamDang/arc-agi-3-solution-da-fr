"""Progress of the same games in two runs, read from their request logs.

    uv run --no-sync python experiments/note-compaction/compare_runs.py \
        runs/note-compaction-3games runs/base-gpt61sol-20games [--games sk48,lf52,bp35]

Per game and run, one row per analyzer turn (the first request of each
analysis step): the game step and level its opener states, the output tokens
and cost spent before it, and whether the turn followed a cut (a logged prompt
dropping by 25K or more, or a note compaction request). Printed: the totals,
then the level reached after equal numbers of turns, game actions and output
tokens, and the same after the first cut.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from inference.utils.run_artifacts import open_log

STATE = re.compile(r"Current state: step (\d+), level (\d+)")


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def load(run: Path, game: str) -> dict:
    paths = sorted(glob.glob(str(run / f"{game}*_requests.jsonl*")))
    turns: list[dict] = []
    out = cost = 0.0
    notes = 0
    prev_prompt = None
    cuts: list[int] = []
    pending = None
    with open_log(Path(paths[0])) as lines:
        for line in lines:
            record = json.loads(line)
            if record["event"] == "request":
                pending = record
                continue
            usage = record.get("usage") or {}
            step = record.get("analysis_step")
            if record.get("kind") == "note_compaction":
                notes += 1
                cuts.append(step)
            elif pending is not None and record.get("request_index_within_turn") == 1:
                found = STATE.search(text_of(pending["messages"][-1]))
                if found:
                    turns.append(dict(turn=step, step=int(found[1]), level=int(found[2]),
                                      out=out, cost=cost))
            prompt = usage.get("prompt_tokens") or 0
            if record.get("kind") != "note_compaction":
                if prev_prompt is not None and prompt < prev_prompt - 25000:
                    cuts.append(step)
                prev_prompt = prompt
            out += usage.get("completion_tokens") or 0
            cost += usage.get("cost") or 0.0
            pending = None
    return dict(turns=turns, out=out, cost=cost, notes=notes, cuts=sorted(set(cuts)))


def level_at(turns: list[dict], key: str, value: float) -> int:
    reached = [t["level"] for t in turns if t[key] <= value]
    return reached[-1] if reached else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", type=Path)
    parser.add_argument("old", type=Path)
    parser.add_argument("--games", default="sk48,lf52,bp35")
    args = parser.parse_args()
    for game in args.games.split(","):
        runs = {"new": load(args.new, game), "old": load(args.old, game)}
        print(f"\n== {game}")
        for name, r in runs.items():
            last = r["turns"][-1] if r["turns"] else dict(turn=0, step=0, level=1)
            print(f"{name}: {last['turn']} turns, {last['step']} actions, level {last['level']} at the last turn, "
                  f"{r['out']:,.0f} output tokens, ${r['cost']:.2f}; notes {r['notes']}, cuts at turn {r['cuts']}")
        new, old = runs["new"]["turns"], runs["old"]["turns"]
        if not new or not old:
            continue
        horizon = min(new[-1]["turn"], old[-1]["turn"])
        first_cut = min([c for r in runs.values() for c in r["cuts"]] or [horizon])
        print(f"common horizon {horizon} turns; first cut at turn {first_cut}")
        for key, label, marks in (
            ("turn", "turns", [first_cut, 60, 90, 120, 150, 180, 210, horizon]),
            ("step", "actions", [200, 300, 400, 500, 600, 700, 800, min(new[-1]["step"], old[-1]["step"])]),
            ("out", "output tokens", [50e3, 100e3, 150e3, 200e3, 250e3, min(runs["new"]["out"], runs["old"]["out"])]),
        ):
            limit = min(new[-1][key], old[-1][key])
            row = [f"{m:,.0f}: {level_at(new, key, m)} vs {level_at(old, key, m)}" for m in sorted(set(marks)) if m <= limit]
            print(f"level at equal {label} (new vs old): " + "; ".join(row))
        # after the first cut: levels gained per 100 turns and actions per level gained
        for name, turns in (("new", new), ("old", old)):
            start = next((t for t in turns if t["turn"] >= first_cut), None)
            if start is None:
                print(f"  {name}: ended at turn {turns[-1]['turn']}, before the first cut")
                continue
            end = next((t for t in turns if t["turn"] >= horizon), turns[-1])
            gained = end["level"] - start["level"]
            span = end["turn"] - start["turn"]
            actions = end["step"] - start["step"]
            spent = end["out"] - start["out"]
            print(f"  {name} turns {start['turn']}-{end['turn']}: +{gained} levels, {actions} actions, "
                  f"{spent:,.0f} output tokens"
                  + (f"; {100 * gained / span:.1f} levels per 100 turns" if span else ""))


if __name__ == "__main__":
    main()
