"""Pick a fixed evaluation set of teacher requests, split into dev and eval.

    uv run --no-sync python -m think_gen.evalset --run runs/base-max-dfranzen \
        -o experiments/teacher-reasoning/evalset/max5.json

From each game run, the requests that have real reasoning and a tool call
are sorted by the length of that reasoning and cut into `--bins` bins of
equal count. From each bin, `--per-bin` requests evenly spaced in game
position are taken, and half of them go to dev, half to eval (alternating
after a seeded shuffle). The manifest lists each request's key, split,
length bin, game position, reasoning length and prompt tokens.
"""
import argparse
import json
import random
from pathlib import Path

from . import logs


def pick(records: list, bins: int, per_bin: int, rng: random.Random) -> list[dict]:
    usable = [r for r in records if r.reply.get("tool_calls") and r.real_reasoning.strip()]
    usable.sort(key=lambda r: (len(r.real_reasoning), r.index))
    out = []
    for b in range(bins):
        chunk = usable[b * len(usable) // bins:(b + 1) * len(usable) // bins]
        chunk.sort(key=lambda r: r.index)
        n = min(per_bin, len(chunk))
        chosen = [chunk[round((i + 0.5) * len(chunk) / n - 0.5)] for i in range(n)]
        splits = ["dev", "eval"] * ((n + 1) // 2)
        rng.shuffle(splits)
        for r, s in zip(chosen, splits[:n]):
            out.append({
                "key": r.key, "game": r.game, "index": r.index, "split": s, "length_bin": b,
                "position": round(r.index / max(1, len(records) - 1), 2),
                "real_chars": len(r.real_reasoning), "reasoning_tokens": r.reasoning_tokens,
                "prompt_tokens": r.usage.get("prompt_tokens"),
            })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    items = []
    for path in logs.request_logs(args.run):
        items += pick(logs.read_log(path), args.bins, args.per_bin, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "run": str(args.run), "bins": args.bins, "per_bin": args.per_bin, "seed": args.seed,
        "items": items}, indent=1))
    for split in ("dev", "eval"):
        print(split, sum(1 for i in items if i["split"] == split))


if __name__ == "__main__":
    main()
