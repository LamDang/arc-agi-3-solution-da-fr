"""Compare game runs of the harness (benchmark.json files) on the games they share.

    python compare_games.py v3=path/benchmark.json pruned=path/benchmark.json [...] [--games tu93,cd82] [--json out.json]

Per game: every run's score and the mean per label. Overall: the mean over
the shared games of each label's per-game means, and each label's difference
from the first, with a standard error from the pooled pass-to-pass variance
(games are paired, so between-game spread cancels). Also tokens generated
per run and runs won.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, list[dict]]:
    runs = json.loads(Path(path).read_text())["game_runs"]
    by_game = defaultdict(list)
    for r in runs:
        by_game[r["game_id"][:4]].append({
            "score": float(r.get("final_score") or 0.0),
            "levels": r.get("levels_completed"),
            "of": r.get("number_of_levels"),
            "state": r.get("state"),
            "tokens": sum(h.get("generated_tokens", 0) or 0 for h in r.get("history", [])),
            "actions": len(r.get("history", [])),
        })
    return dict(by_game)


def compare(labelled: dict[str, dict], games: list[str] | None = None) -> dict:
    shared = sorted(set.intersection(*(set(g) for g in labelled.values())))
    if games:
        shared = [g for g in shared if g in games]
    labels = list(labelled)
    report = {"games": shared, "per_game": {}, "overall": {}}
    for g in shared:
        report["per_game"][g] = {l: [r["score"] for r in labelled[l][g]] for l in labels}
    means = {l: {g: st.mean(report["per_game"][g][l]) for g in shared} for l in labels}
    for l in labels:
        variances = [st.variance(report["per_game"][g][l]) for g in shared if len(report["per_game"][g][l]) > 1]
        runs = [r for g in shared for r in labelled[l][g]]
        report["overall"][l] = {
            "mean": st.mean(means[l].values()),
            "runs": len(runs),
            "won": sum(r["state"] == "won" for r in runs),
            "pooled_sd": math.sqrt(st.mean(variances)) if variances else None,
            "tokens_per_run": st.mean(r["tokens"] for r in runs),
        }
    ref = labels[0]
    for l in labels[1:]:
        # SE of the difference of means over paired games: sum over games of var/n per arm, divided by G^2
        se2 = 0.0
        for g in shared:
            for arm in (ref, l):
                scores = report["per_game"][g][arm]
                if len(scores) > 1:
                    se2 += st.variance(scores) / len(scores)
        diff = report["overall"][l]["mean"] - report["overall"][ref]["mean"]
        report["overall"][l].update(diff_vs_first=diff, se_diff=math.sqrt(se2) / len(shared))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="LABEL=benchmark.json, the first is the reference")
    parser.add_argument("--games", default="")
    parser.add_argument("--json")
    args = parser.parse_args()
    labelled = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        labelled[label] = load(Path(path))
    report = compare(labelled, [g for g in args.games.split(",") if g] or None)
    labels = list(labelled)
    print(f"{'game':6}" + "".join(f" {l:>24}" for l in labels))
    for g in report["games"]:
        cells = [f"{st.mean(report['per_game'][g][l]):6.1f} ({'/'.join(f'{s:.0f}' for s in report['per_game'][g][l])})"
                 for l in labels]
        print(f"{g:6}" + "".join(f" {c:>24}" for c in cells))
    print()
    for l in labels:
        o = report["overall"][l]
        diff = f"  diff {o['diff_vs_first']:+.1f} +- {o['se_diff']:.1f}" if "diff_vs_first" in o else ""
        print(f"{l:12} mean {o['mean']:6.2f} over {len(report['games'])} games, {o['runs']} runs, {o['won']} won, "
              f"{o['tokens_per_run'] / 1e3:.0f}K tokens/run{diff}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
