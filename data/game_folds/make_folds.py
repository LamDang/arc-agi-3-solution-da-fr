"""Split the 25 public ARC-AGI-3 games into 5 folds stratified by hardness.

    python data/game_folds/make_folds.py                      # rebuild folds from scores.csv
    python data/game_folds/make_folds.py --benchmark PATH     # first rebuild scores.csv from a benchmark.json
    python data/game_folds/make_folds.py --fetch              # first download benchmark.json from Kaggle

Hardness is a game's mean score over the four passes of the dfranzen notebook
v3 run (kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution, 2026-10-03,
mean 46.49). The games are sorted by that mean (hardest first, ties by game
id) and cut into 5 tiers of 5. Each fold takes exactly one game from each
tier. Which game of a tier goes to which fold is chosen to make the folds'
mean scores as equal as possible: a seeded random search over assignments,
then swaps within a tier until no swap helps. The result is deterministic.

Standard library only. Writes scores.csv (with --benchmark/--fetch),
folds.csv and folds.json next to this script. See README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_FOLDS = 5
SEED = 0
RANDOM_TRIALS = 200_000

KAGGLE_OUTPUT = "https://www.kaggle.com/api/v1/kernels/output?userName=dfranzen&kernelSlug=arc-agi-3-milestone-2-solution"
# The run the folds are built from. The Kaggle API serves the latest version's
# output, so a newer version of the notebook would be a different run.
EXPECTED_START = "2026-10-03"
EXPECTED_RUNS = 100

SCORE_FIELDS = ["game", "game_id", "pass", "score", "state", "levels_completed", "number_of_levels", "actions", "generated_tokens"]


def fetch_benchmark(dest: Path) -> Path:
    """Download benchmark.json of the notebook's latest version (public, no credentials)."""
    url, token = None, None
    while url is None:
        page = KAGGLE_OUTPUT + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        with urllib.request.urlopen(page) as response:
            listing = json.load(response)
        url = next((f["url"] for f in listing["files"] if f["fileName"] == "benchmark.json"), None)
        token = listing.get("nextPageToken")
        if url is None and not token:
            sys.exit("benchmark.json not found in the notebook's output")
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())
    return dest


def scores_from_benchmark(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    runs = data["game_runs"]
    if not str(data.get("start_time", "")).startswith(EXPECTED_START) or len(runs) != EXPECTED_RUNS:
        print(f"warning: {path} is not the 2026-10-03 v3 run (start {data.get('start_time')}, "
              f"{len(runs)} runs)", file=sys.stderr)
    rows = []
    for r in runs:
        # the pass index is only in the artifact names: solver_analysis/<game_id>_p<k>.html
        pass_index = int(Path(r["solver_analysis_html"]).stem.rsplit("_p", 1)[1])
        rows.append({
            "game": r["game_id"][:4],
            "game_id": r["game_id"],
            "pass": pass_index,
            "score": round(float(r.get("final_score") or 0.0), 4),
            "state": r["state"],
            "levels_completed": r["levels_completed"],
            "number_of_levels": r["number_of_levels"],
            "actions": len(r.get("history", [])),
            "generated_tokens": sum(h.get("generated_tokens", 0) or 0 for h in r.get("history", [])),
        })
    return sorted(rows, key=lambda row: (row["game"], row["pass"]))


def read_scores(path: Path) -> list[dict]:
    with path.open() as f:
        return [{**row, "score": float(row["score"]), "levels_completed": int(row["levels_completed"]),
                 "number_of_levels": int(row["number_of_levels"])} for row in csv.DictReader(f)]


def per_game(rows: list[dict]) -> list[dict]:
    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game"]].append(row)
    games = []
    for game, runs in by_game.items():
        games.append({
            "game": game,
            "game_id": runs[0]["game_id"],
            "passes": len(runs),
            "mean_score": st.mean(r["score"] for r in runs),
            "pass_scores": [r["score"] for r in runs],
            "won": sum(r["state"] == "won" for r in runs),
            "mean_levels": st.mean(r["levels_completed"] for r in runs),
            "number_of_levels": runs[0]["number_of_levels"],
        })
    return sorted(games, key=lambda g: (g["mean_score"], g["game"]))


def spread(tiers: list[list[dict]], assignment: list[list[int]]) -> tuple[float, float]:
    """(max - min, standard deviation) of the folds' mean scores."""
    totals = [0.0] * N_FOLDS
    for tier, order in zip(tiers, assignment):
        for fold, index in enumerate(order):
            totals[fold] += tier[index]["mean_score"]
    means = [t / len(tiers) for t in totals]
    return max(means) - min(means), st.pstdev(means)


def balance(tiers: list[list[dict]]) -> list[list[int]]:
    """assignment[t][fold] = index within tier t of the game that goes to fold."""
    rng = random.Random(SEED)
    # folds are interchangeable, so the hardest tier is fixed in order
    best = [list(range(N_FOLDS)) for _ in tiers]
    best_cost = spread(tiers, best)
    for _ in range(RANDOM_TRIALS):
        candidate = [best[0]] + [rng.sample(range(N_FOLDS), N_FOLDS) for _ in tiers[1:]]
        cost = spread(tiers, candidate)
        if cost < best_cost:
            best, best_cost = candidate, cost
    improved = True
    while improved:
        improved = False
        for t in range(1, len(tiers)):
            for a in range(N_FOLDS):
                for b in range(a + 1, N_FOLDS):
                    candidate = [list(order) for order in best]
                    candidate[t][a], candidate[t][b] = candidate[t][b], candidate[t][a]
                    cost = spread(tiers, candidate)
                    if cost < best_cost:
                        best, best_cost, improved = candidate, cost, True
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--benchmark", type=Path, help="benchmark.json of the dfranzen v3 run; rewrites scores.csv")
    source.add_argument("--fetch", action="store_true", help="download benchmark.json from Kaggle; rewrites scores.csv")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    scores_csv = args.out / "scores.csv"
    if args.fetch:
        args.benchmark = fetch_benchmark(args.out / "benchmark.json")
    if args.benchmark:
        rows = scores_from_benchmark(args.benchmark)
        with scores_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCORE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    rows = read_scores(scores_csv)

    games = per_game(rows)
    if len(games) % N_FOLDS:
        sys.exit(f"{len(games)} games do not split into {N_FOLDS} tiers of equal size")
    size = len(games) // N_FOLDS
    tiers = [games[i * size:(i + 1) * size] for i in range(N_FOLDS)]
    assignment = balance(tiers)

    table = []
    for t, (tier, order) in enumerate(zip(tiers, assignment), start=1):
        for fold, index in enumerate(order):
            g = tier[index]
            table.append({"game": g["game"], "game_id": g["game_id"], "tier": t, "fold": fold,
                          "mean_score": round(g["mean_score"], 2), "won": f"{g['won']}/{g['passes']}",
                          "mean_levels": round(g["mean_levels"], 2), "number_of_levels": g["number_of_levels"],
                          "pass_scores": " ".join(f"{s:.1f}" for s in g["pass_scores"])})
    table.sort(key=lambda row: (row["fold"], row["tier"]))

    with (args.out / "folds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)

    folds = []
    for fold in range(N_FOLDS):
        members = [row for row in table if row["fold"] == fold]
        folds.append({
            "fold": fold,
            "games": [row["game"] for row in members],
            "game_ids": [row["game_id"] for row in members],
            "mean_score": round(st.mean(row["mean_score"] for row in members), 2),
        })
    summary = {
        "source": "dfranzen/arc-agi-3-milestone-2-solution v3 (Kaggle), run 2026-10-03, 25 games x 4 passes, mean 46.49",
        "hardness": "mean final_score over the 4 passes; tier 1 = hardest (lowest), tier 5 = easiest",
        "seed": SEED,
        "tiers": {str(t): [g["game"] for g in tier] for t, tier in enumerate(tiers, start=1)},
        "folds": folds,
    }
    (args.out / "folds.json").write_text(json.dumps(summary, indent=2) + "\n")

    for t, tier in enumerate(tiers, start=1):
        print(f"tier {t}: " + ", ".join(f"{g['game']} {g['mean_score']:.1f}" for g in tier))
    for f in folds:
        print(f"fold {f['fold']}: {' '.join(f['games'])}  mean {f['mean_score']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
