"""Pick a fixed evaluation set of teacher requests, split into dev and eval.

Hidden-reasoning teacher (gpt-6.1-sol), the default:

    uv run --no-sync python -m think_gen.evalset --run runs/gpt61sol-features-25games \
        -o experiments/teacher-reasoning/evalset/sol25.json

The real thinking is not returned, but its billed length is: requests with
a tool call are put in run-wide bins of `reasoning_tokens` (bin 0: none;
bins 1-4: quartiles of the rest). Each game gives `--per-game` requests from
as many different bins, the least filled bins first so the bins end up about
equal, one seeded random request per bin. Whole games go to dev or eval, so
a judge built on dev is measured on games it has not seen.

Visible-reasoning teacher (qwen3.8-max), `--by real-chars`: per game, the
requests are cut into `--bins` bins by real reasoning length, `--per-bin`
requests evenly spaced in game position from each, alternating dev and eval
(made experiments/teacher-reasoning/evalset/max5.json).
"""
import argparse
import json
import random
from pathlib import Path

from . import logs


def pick_by_chars(records: list, bins: int, per_bin: int, rng: random.Random) -> list[dict]:
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


def token_edges(games: dict[str, list]) -> list[int]:
    """Lower edges of bins 2-4: quartiles of the nonzero reasoning tokens."""
    nz = sorted(r.reasoning_tokens for rs in games.values() for r in rs if r.reasoning_tokens > 0)
    return [nz[len(nz) * q // 4] for q in (1, 2, 3)]


def token_bin(tokens: int, edges: list[int]) -> int:
    return 0 if tokens == 0 else 1 + sum(tokens >= e for e in edges)


def pick_by_tokens(games: dict[str, list], per_game: int, dev_share: float,
                   rng: random.Random) -> tuple[list[dict], list[int]]:
    edges = token_edges(games)
    fill = [0] * (len(edges) + 2)
    names = sorted(games)
    dev = set(rng.sample(names, round(len(names) * dev_share)))
    out = []
    # games with the fewest bins available choose first
    def available(g):
        return {token_bin(r.reasoning_tokens, edges) for r in games[g]}
    for g in sorted(names, key=lambda g: (len(available(g)), g)):
        records = games[g]
        by_bin = {}
        for r in records:
            by_bin.setdefault(token_bin(r.reasoning_tokens, edges), []).append(r)
        bins = sorted(by_bin, key=lambda b: (fill[b], rng.random()))[:per_game]
        for b in sorted(bins):
            r = rng.choice(by_bin[b])
            fill[b] += 1
            out.append({
                "key": r.key, "game": r.game, "index": r.index,
                "split": "dev" if g in dev else "eval", "token_bin": b,
                "position": round(r.index / max(1, len(records) - 1), 2),
                "reasoning_tokens": r.reasoning_tokens,
                "prompt_tokens": r.usage.get("prompt_tokens"),
            })
    out.sort(key=lambda i: (i["game"], i["index"]))
    return out, edges


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--by", choices=["tokens", "real-chars"], default="tokens")
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--dev-share", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    if args.by == "real-chars":
        items = []
        for path in logs.request_logs(args.run):
            items += pick_by_chars(logs.read_log(path), args.bins, args.per_bin, rng)
        meta = {"by": "real-chars", "bins": args.bins, "per_bin": args.per_bin}
    else:
        games = {}
        for path in logs.request_logs(args.run):
            rs = [r for r in logs.read_log(path) if r.reply.get("tool_calls")]
            if rs:
                games[rs[0].game] = rs
        items, edges = pick_by_tokens(games, args.per_game, args.dev_share, rng)
        meta = {"by": "tokens", "token_bin_edges": [0, 1, *edges], "per_game": args.per_game,
                "dev_share": args.dev_share}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"run": str(args.run), **meta, "seed": args.seed,
                                       "items": items}, indent=1))
    for split in ("dev", "eval"):
        mine = [i for i in items if i["split"] == split]
        bins = {}
        for i in mine:
            b = i.get("token_bin", i.get("length_bin"))
            bins[b] = bins.get(b, 0) + 1
        print(split, len(mine), "games", len({i["game"] for i in mine}), "per bin", dict(sorted(bins.items())))


if __name__ == "__main__":
    main()
