# 5-fold split of the 25 public games, stratified by hardness

Five folds of five games for cross-validation (for example: fine-tune or tune
on four folds, evaluate on the fifth). Each fold holds one game from each of
five hardness tiers, so every fold has the same mix of easy and hard games.

## The folds

| fold | tier 1 (hardest) | tier 2 | tier 3 | tier 4 | tier 5 (easiest) | fold mean score |
| --- | --- | --- | --- | --- | --- | ---: |
| 0 | sk48 (0.0) | sp80 (13.1) | tn36 (53.5) | cd82 (81.2) | ar25 (85.4) | 46.65 |
| 1 | bp35 (1.2) | cn04 (26.2) | su15 (27.8) | tu93 (83.7) | sb26 (93.2) | 46.42 |
| 2 | lf52 (1.8) | ka59 (18.8) | dc22 (42.5) | r11l (82.1) | ft09 (86.9) | 46.43 |
| 3 | wa30 (3.2) | sc25 (19.6) | m0r0 (46.4) | re86 (62.9) | lp85 (100.0) | 46.43 |
| 4 | g50t (9.2) | s5i5 (23.9) | ls20 (26.9) | vc33 (80.4) | tr87 (92.3) | 46.53 |

Numbers in brackets are the game's mean score in the dfranzen run (below).
All 25 games average 46.49; the folds range from 46.42 to 46.65.

The five games most experiments in `ARC3-Inference/exp/` use (ft09, lp85,
ls20, sp80, vc33) fall in folds 2, 3, 4, 0 and 4.

## Source: the dfranzen notebook v3 run

Hardness comes from version 3 of the competition notebook
[dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution),
run 2026-10-03 14:09-22:20: the Duck harness with the submission settings and
a local quantized Qwen3.8-Flash-Next (SGLang, Intel W4A16), 4 passes on all 25
public games, 100 game runs, 33 won, mean score 46.49. Its `benchmark.json` is
read through the public Kaggle API (`kernels/output`, no credentials). The
same run is `kaggle_v3` in `exp/reap-flash-next/README.md` and is quoted in
`ARC3-Inference/exp/v12-5games.md`.

## Method

1. **Hardness** of a game = its mean `final_score` (the competition score,
   0-100) over the 4 passes. Lower is harder.
2. **Tiers.** Sort the 25 games by hardness (hardest first, ties by game id)
   and cut into 5 tiers of 5 consecutive games:

   | tier | games (mean score) |
   | --- | --- |
   | 1 | sk48 0.0, bp35 1.2, lf52 1.8, wa30 3.2, g50t 9.2 |
   | 2 | sp80 13.1, ka59 18.8, sc25 19.6, s5i5 23.9, cn04 26.2 |
   | 3 | ls20 26.9, su15 27.8, dc22 42.5, m0r0 46.4, tn36 53.5 |
   | 4 | re86 62.9, vc33 80.4, cd82 81.2, r11l 82.1, tu93 83.7 |
   | 5 | ar25 85.4, ft09 86.9, tr87 92.3, sb26 93.2, lp85 100.0 |

3. **Assignment.** Each fold gets one game of each tier. Among those
   assignments, the script picks one that makes the folds' mean scores as
   equal as possible (smallest max-min gap, then smallest standard
   deviation): 200,000 seeded random assignments (seed 0, tier 1 kept in
   order since folds are interchangeable), then swaps of two games within a
   tier until no swap narrows the gap. The output is deterministic.

### Caveats

- Four passes of a stochastic agent are a noisy measure. Several games swing
  between passes (tn36: 100, 10.5, 3.6, 100; m0r0: 14.3, 71.4, 100, 0), and
  the tier boundaries are close in places: cn04 (26.19, tier 2) and ls20
  (26.88, tier 3) differ by less than a point.
- Hardness is for this harness and this model. Another agent can rank the
  games differently (in the repo's OpenRouter runs, sp80 is the game the base
  agent cannot model while the v12 play agent scores 47.6 on it).
- Tier 1 is mostly zeros: sk48 was never past level 0, and bp35, lf52, wa30
  only solved level 1 in every pass.

## Files

| file | contents |
| --- | --- |
| `folds.csv` | one row per game: `game`, `game_id`, `tier` (1-5), `fold` (0-4), `mean_score`, `won` (passes won / passes), `mean_levels`, `number_of_levels`, `pass_scores` |
| `folds.json` | the tiers and, per fold, its games, game ids and mean score |
| `scores.csv` | the input: one row per game run of the dfranzen run (100 rows): `game`, `game_id`, `pass`, `score`, `state`, `levels_completed`, `number_of_levels`, `actions`, `generated_tokens` |
| `make_folds.py` | builds `folds.csv` and `folds.json` from `scores.csv` (standard library only) |

## Reproduce

From the repository root, with any Python 3.9+:

```bash
python data/game_folds/make_folds.py              # from the committed scores.csv
python data/game_folds/make_folds.py --fetch      # re-download benchmark.json from Kaggle, rewrite scores.csv, then split
python data/game_folds/make_folds.py --benchmark path/to/benchmark.json
```

`--fetch` saves `benchmark.json` (5 MB, not committed) next to the script.
The Kaggle API serves the output of the notebook's latest version; the script
warns if that is no longer the 2026-10-03 run. Both `--fetch` and the
committed `scores.csv` give the folds above (checked 2026-10-07).

Reading the folds in Python:

```python
import json
folds = json.load(open("data/game_folds/folds.json"))["folds"]
test_games = folds[k]["games"]                      # e.g. ["sk48", "sp80", "tn36", "cd82", "ar25"] for k = 0
train_games = [g for f in folds if f["fold"] != k for g in f["games"]]
```
