"""Expert selection and game-level cross-validation from saved REAP statistics.

    python analyze.py OUT_DIR [--keep 448,384,320,288,256,192] [--categories context,generated,image]
        [--criterion gate_norm]

Experts are ranked per layer by a statistic summed over the tokens routed to
them (see expert_scores). The default, gate_norm = sum(g_j * ||f_j||), is
REAP's quantity summed rather than averaged: on held-out games it lost about
40% less next-token NLL than REAP's average at 256 experts. The kept set of a
layer is its top-N experts; every layer keeps the same N.

Cross-validation leaves one game out: experts are chosen from the other
games' statistics, then measured on the held-out game by coverage, the share
of its router weight (sum of g over routed tokens) that lands on kept
experts. Coverage needs no GPU, so it is a quick screen; what pruning costs
is measured by held-out NLL in prune_eval.py, which is what decisions use.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CATEGORIES = ("context", "generated", "image")


def load(out_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    runs = {}
    for path in sorted((out_dir / "stats").glob("*/*.npz")):
        key = f"{path.parent.name}/{path.stem}"
        with np.load(path) as data:
            runs[key] = {k: data[k] for k in data.files}
    return runs


def game_of(run_key: str) -> str:
    return run_key.split("/")[-1].rsplit("_p", 1)[0]


def aggregate(runs: dict, keys, categories) -> dict[str, np.ndarray]:
    """Sum the fields over runs and the chosen category indexes -> [L, E]."""
    total = None
    for key in keys:
        part = {f: v[..., categories].sum(-1) for f, v in runs[key].items() if not f.endswith("_pos")}
        total = part if total is None else {f: total[f] + part[f] for f in total}
    return total


def reap_scores(stats: dict) -> np.ndarray:
    count = stats["count"]
    return np.divide(stats["gate_norm"], count, out=np.zeros_like(count), where=count > 0)


CRITERIA = ("reap", "gate", "count", "prob", "gate_norm")


def expert_scores(stats: dict, criterion: str = "reap") -> np.ndarray:
    """reap: mean g*||f|| over routed tokens. gate: total router weight.
    count: routing frequency. prob: total softmax probability, routed or not.
    gate_norm: total g*||f|| (REAP without the mean, favours frequent experts)."""
    if criterion == "reap":
        return reap_scores(stats)
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}, expected one of {CRITERIA}")
    return stats[criterion]


def keep_mask(scores: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :n]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def coverage(stats: dict, mask: np.ndarray, field: str = "gate") -> np.ndarray:
    """Per-layer share of `field` mass on kept experts."""
    mass = stats[field]
    total = mass.sum(1)
    return np.divide((mass * mask).sum(1), total, out=np.ones_like(total), where=total > 0)


def position_bands(runs: dict, keys, categories, keep: list[int], criterion: str = "gate_norm") -> dict | None:
    """Do long contexts route to other experts? Per position band (see
    reap_model.ReapRecorder.POSITION_BANDS): tokens, and for each N the
    overlap of the top-N experts chosen from that band with those chosen
    from the first band, plus the share of that band's router weight that
    the first band's choice keeps vs its own choice. None without _pos data."""
    keys = [k for k in keys if "count_pos" in runs[k]]
    if not keys:
        return None
    total = {}
    for key in keys:
        for f, v in runs[key].items():
            if f.endswith("_pos"):
                part = v[:, :, categories].sum(2)  # [L, E, band]
                total[f[:-4]] = total.get(f[:-4], 0) + part
    n_bands = total["count"].shape[-1]
    band = [{f: v[..., b] for f, v in total.items()} for b in range(n_bands)]
    out = {"tokens": [float(b["count"][0].sum()) / 10 for b in band], "keep": {}}  # top-10 routing
    for n in keep:
        rows = []
        first = keep_mask(expert_scores(band[0], criterion), n)
        for b in range(n_bands):
            if band[b]["count"].sum() == 0:
                rows.append(None)
                continue
            own = keep_mask(expert_scores(band[b], criterion), n)
            rows.append({"overlap_with_first": float((own & first).sum(1).mean() / n),
                         "coverage_first_choice": float(coverage(band[b], first).mean()),
                         "coverage_own_choice": float(coverage(band[b], own).mean())})
        out["keep"][n] = rows
    return out


def usage_summary(stats: dict) -> dict:
    """How concentrated routing is: experts needed for 50/90/99% of router weight."""
    gate = stats["gate"]
    ranked = -np.sort(-gate, axis=1)
    cumulative = ranked.cumsum(1) / np.maximum(ranked.sum(1, keepdims=True), 1e-12)
    out = {f"experts_for_{int(q * 100)}pct": np.argmax(cumulative >= q, axis=1) + 1 for q in (0.5, 0.9, 0.99)}
    out["never_routed"] = (stats["count"] == 0).sum(1)
    return {k: {"min": int(v.min()), "median": float(np.median(v)), "max": int(v.max())} for k, v in out.items()}


def cross_validate(runs: dict, keep: list[int], categories, criterion: str = "gate_norm") -> dict:
    by_game = defaultdict(list)
    for key in runs:
        by_game[game_of(key)].append(key)
    games = sorted(g for g in by_game if g != "unknown")
    all_keys = list(runs)
    result = {}
    for n in keep:
        rows = []
        for game in games:
            train = [k for k in all_keys if game_of(k) != game]
            held = aggregate(runs, by_game[game], categories)
            mask_out = keep_mask(expert_scores(aggregate(runs, train, categories), criterion), n)
            mask_in = keep_mask(expert_scores(aggregate(runs, all_keys, categories), criterion), n)
            cov_out, cov_in = coverage(held, mask_out), coverage(held, mask_in)
            rows.append({"game": game, "held_out_worst_layer": float(cov_out.min()),
                         "held_out_mean": float(cov_out.mean()), "in_sample_mean": float(cov_in.mean()),
                         "gap_mean": float(cov_in.mean() - cov_out.mean())})
        result[n] = {
            "games": rows,
            "held_out_mean": float(np.mean([r["held_out_mean"] for r in rows])) if rows else None,
            "held_out_worst": float(np.min([r["held_out_worst_layer"] for r in rows])) if rows else None,
            "gap_mean": float(np.mean([r["gap_mean"] for r in rows])) if rows else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--keep", default="448,384,320,288,256,192")
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--criterion", choices=CRITERIA, default="gate_norm")
    args = parser.parse_args()
    out = Path(args.out_dir)
    runs = load(out)
    if not runs:
        raise SystemExit(f"no statistics under {out / 'stats'}")
    categories = [CATEGORIES.index(c) for c in args.categories.split(",")]
    keep = [int(n) for n in args.keep.split(",")]
    everything = aggregate(runs, list(runs), categories)
    report = {
        "runs": list(runs),
        "games": sorted({game_of(k) for k in runs}),
        "categories": args.categories,
        "criterion": args.criterion,
        "usage": usage_summary(everything),
        "in_sample_coverage": {
            n: float(coverage(everything, keep_mask(expert_scores(everything, args.criterion), n)).mean())
            for n in keep},
        "cross_validation": cross_validate(runs, keep, categories, args.criterion),
        "position_bands": position_bands(runs, list(runs), categories, keep, args.criterion),
    }
    (out / "analysis.json").write_text(json.dumps(report, indent=1))
    print(f"{len(runs)} runs, games: {', '.join(report['games'])}")
    print("routing concentration per layer:", json.dumps(report["usage"]))
    print(f"{'keep':>5} {'in-sample':>10} {'held-out':>9} {'worst':>7} {'gap':>7}")
    for n in keep:
        cv = report["cross_validation"][n]
        held = "-" if cv["held_out_mean"] is None else f"{cv['held_out_mean']:.4f}"
        worst = "-" if cv["held_out_worst"] is None else f"{cv['held_out_worst']:.4f}"
        gap = "-" if cv["gap_mean"] is None else f"{cv['gap_mean']:.4f}"
        print(f"{n:>5} {report['in_sample_coverage'][n]:>10.4f} {held:>9} {worst:>7} {gap:>7}")
    bands = report["position_bands"]
    if bands:
        print("\nby position (bands 0-32K, 32-64K, 64-96K, 96K+): tokens", [round(t) for t in bands["tokens"]])
        print(f"{'keep':>5} {'band':>4} {'overlap w/ 0-32K':>16} {'kept weight, 0-32K choice':>25} {'own choice':>10}")
        for n, rows in bands["keep"].items():
            for b, r in enumerate(rows):
                if r:
                    print(f"{n:>5} {b:>4} {r['overlap_with_first']:>16.3f} {r['coverage_first_choice']:>25.4f} "
                          f"{r['coverage_own_choice']:>10.4f}")


if __name__ == "__main__":
    main()
