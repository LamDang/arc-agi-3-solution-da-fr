# gpt-6.1-sol on the 25 NVIDIA DreamTeam community games

## Question

The latest official-game run ([gpt61sol-features-25games.md](gpt61sol-features-25games.md))
won all 25 official games with the base agent, dfranzen's settings and the four new harness
settings. How does the same agent do on games it has never been tuned on? The 25 synthetic
games of [NVIDIA DreamTeam](https://github.com/NVIDIA/dream-team), vendored in arcengine format
by [Felix561/arc3-synthetic-games](https://github.com/Felix561/arc3-synthetic-games), are
played through the new `community_games/` submodules (LOCAL_EVAL.md, "Play community games").

## Setup

| | |
| --- | --- |
| runs (DVC) | `runs/gpt61sol-nvidia-25games` (first attempt), `-resume` (first resume), `-resume2` (second resume, scored) |
| launched with | `experiments/nvidia-community/run.sh` (the `dvc_eval.py` call of `experiments/gpt61sol-features/run.sh`, with `ENVIRONMENTS_DIR=environment_files_community`) |
| games | the 25 NVIDIA games, 8 levels each, 1 pass; their metadata's baseline action counts are kept, so scores are comparable in kind to the official games' |
| settings | identical to gpt61sol-features-25games: `xhigh`, `detailed` summaries, reasoning sent back, `ARC3_PYTHON_RATIONALE=1` (description and reasoning fields), note compaction at 120K keeping 10 turns, `ARC3_NO_BUDGET_BURN=1`, repeat hints 3/3/10, 10 model calls at once, 300K output tokens and 240 minutes per game |
| started | 2026-10-10 15:07 UTC; ended 21:34 UTC (stopped by hand, see below) |

**Smoke test.** cc2048 alone with a 30K output cap: 6/8 levels, 55.8, $0.73; both reasoning
fields filled on every call, scoring and packing ran.

## Results

22 of 25 games won, **179/200 levels, mean score 76.7**. Actions per level are from the
attempt that was scored; bold is above the game's baseline. Output and prompt tokens and cost
add up every attempt of the game (see "Restarts").

| game | levels | score | actions per level | baseline | attempts | output tokens | prompt tokens | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| al7306 | 8/8 | 68.5 | **138** **26** 17 **27** **29** **24** **51** 37 | 9 17 17 21 21 23 29 37 | 1 | 247.4K | 8.83M | $5.32 |
| cc2048 | 8/8 | 97.4 | **57** 20 20 19 21 21 20 21 | 16 20 20 19 21 21 20 21 | 1 | 37.6K | 2.73M | $1.02 |
| cg1842 | 8/8 | 95.4 | 15 **29** 20 18 20 22 **23** 27 | 15 20 20 18 20 22 22 27 | 1 | 33.5K | 3.13M | $1.05 |
| cl0426 | 8/8 | 99.0 | **10** 10 6 8 10 10 10 12 | 8 10 6 8 10 10 10 12 | 1 | 24.0K | 1.37M | $0.54 |
| df4821 | 8/8 | 77.2 | 14 20 18 16 **36** 20 **36** **36** | 14 20 18 16 32 20 26 26 | 1 | 5.7K | 0.23M | $0.15 |
| dl4827 | 8/8 | 85.5 | **27** **46** **30** **36** **81** 24 30 23 | 12 29 25 30 32 29 33 29 | 1 | 141.5K | 7.41M | $3.33 |
| fl5273 | 3/8, stopped | 16.7 | 7 20 20 **102** | 9 20 20 17 28 17 21 24 | 3 | 747.8K | 30.23M | $15.43 |
| fw4821 | 8/8 | 93.0 | **10** 8 **72** 6 7 5 **15** 15 | 8 10 9 8 11 11 13 15 | 1 | 270.6K | 11.53M | $5.52 |
| gc4721 | 8/8 | 86.6 | **49** 14 11 7 16 20 **39** 17 | 14 14 16 13 16 20 21 17 | 1 | 38.9K | 2.55M | $1.07 |
| hr2048 | 8/8 | 77.4 | **13** 15 17 **32** **21** **16** **22** 18 | 10 15 18 19 15 15 18 18 | 3 | 624.2K | 29.40M | $13.64 |
| ll4821 | 8/8 | 100.0 | **12** 9 **23** 23 20 17 23 23 | 8 10 22 30 21 17 25 27 | 1 | 25.8K | 1.51M | $0.63 |
| ma4173 | 0/8, frozen | 0.0 | **122** | 8 10 12 13 15 13 15 19 | 3 | 647.4K | 72.92M | $19.08 |
| mb2741 | 8/8 | 96.7 | 8 14 12 15 16 **17** 15 17 | 10 14 12 15 16 15 15 17 | 1 | 20.3K | 1.99M | $0.64 |
| ml2048 | 8/8 | 72.3 | **53** **16** **29** 13 **29** 21 22 **69** | 10 15 15 15 20 25 30 30 | 2 | 331.4K | 17.04M | $7.51 |
| mt4926 | 8/8 | 37.6 | **12** **32** **16** **53** 10 **13** **46** **75** | 10 24 10 6 10 10 12 32 | 2 | 534.3K | 20.42M | $11.31 |
| ne4172 | 8/8 | 54.4 | 15 21 21 **39** **53** **59** **29** **75** | 15 21 21 35 27 27 27 35 | 1 | 56.2K | 2.91M | $1.32 |
| os1842 | 8/8 | 96.8 | **5** 10 **7** 7 8 7 9 10 | 4 10 6 7 8 7 9 10 | 1 | 10.0K | 0.41M | $0.24 |
| ps1842 | 8/8 | 94.8 | 2 6 2 6 13 14 13 **8** | 2 6 2 6 13 14 13 7 | 1 | 6.9K | 0.47M | $0.22 |
| ps7413 | 8/8 | 100.0 | 8 19 32 33 51 48 76 78 | 8 19 32 33 51 48 76 78 | 1 | 11.3K | 2.13M | $0.55 |
| rl2048 | 8/8 | 99.2 | **12** 12 18 14 22 12 14 24 | 10 12 18 14 22 12 14 24 | 1 | 16.1K | 1.03M | $0.44 |
| rs0427 | 8/8 | 91.9 | 8 10 **36** 12 19 25 **25** 33 | 8 10 9 12 23 25 21 37 | 1 | 59.9K | 4.73M | $1.56 |
| sf2048 | 8/8 | 100.0 | **15** 9 11 10 14 15 16 23 | 8 11 13 15 20 21 26 35 | 1 | 22.1K | 2.25M | $0.69 |
| sl4821 | 8/8 | 93.3 | 3 9 5 5 8 5 **11** 9 | 3 9 5 5 8 6 8 9 | 1 | 13.9K | 0.70M | $0.33 |
| ss6041 | 0/8, frozen | 0.0 | **373** | 16 19 13 20 19 17 20 21 | 3 | 613.4K | 36.11M | $13.96 |
| td4826 | 8/8 | 83.7 | **21** 14 **35** 9 **35** 15 15 **24** | 12 14 23 9 23 15 15 23 | 2 | 208.2K | 18.77M | $5.84 |

| | |
| --- | --- |
| games won, levels | 22/25, 179/200 |
| mean score | 76.7 (the 22 won games alone: 86.4) |
| model requests | 4,111 |
| output tokens (reasoning) | 4.75M (3.54M) |
| prompt tokens (cached) | 280.8M (95%) |
| cost, every attempt | **$111.37**, of which ~$62 went to attempts thrown away by the restarts |
| cost of the scored run directory alone (`-resume2.metrics.json`) | $49.20 |

Cost at $2 / $0.10 / $2.50 / $10 per million uncached input, cached input, cache-write and
output tokens, summed from the response records of every run directory, each game counted only
in the directories where it was played.

- **The games the agent understands, it wins.** 18 games were won in the first attempt, most
  for under $1.50 and close to the baseline counts; ps7413 matches the baseline on every level.
- **Low scores are extra actions, not missed levels.** mt4926 (37.6) and ne4172 (54.4) won 8/8
  with 2-8x the baseline on several levels; al7306 spent 138 actions on level 1 against 9.
- **Three games were lost to a harness/adapter design gap**, not to difficulty (next section).

## Frozen games: a failed level is never reported as game over

ma4173, ss6041 and, in the first attempt, ml2048 and td4826 froze early: after one losing
move the board never changed again, and the agent spent the rest of its budget on a dead board.

- The NVIDIA games set `h_t["failed"]` on a loss (td4826: the dolphin moves onto the shark;
  ma4173: all crates dropped; ml2048: all pieces lost) and from then on ignore every action
  except RESET.
- Their adapter (`community_games/arc3-synthetic-games/third_party/nvidia/runtime_support/arc_agi_3/game_creator/arcengine_adapter.py`,
  `FunctionalArcGame._apply`) only checks for a completed level: it never calls `lose()`, so
  the engine stays `NOT_FINISHED`. NVIDIA's own harness re-injects RESET on the agent side (the
  adapter's comment says so); this harness does not.
- The harness auto-RESETs only on GAME_OVER (`inference/framework/solver.py`
  `_is_engine_game_over`), and hides RESET from the agent (`EXPOSE_RESET` off). UNDO is offered
  but these games ignore action 7.
- Evidence: in the scored attempt ma4173's board last changed at action 5 of 122, ss6041's at
  action 4 of 373 (`board_changed` in the runtime state). Three replays (subagents) confirmed
  that rendering, click scaling and action delivery are correct, that the recorded solutions
  in Felix561's `trajectories/nvidia` clear level 1 in our engine, and that a RESET recovers.
  The agents' transcripts say as much: "RESET is not available… please restart level 1".
- With another try, td4826 and ml2048 did not make the fatal first move and won 8/8.
- The run was stopped by hand at 21:34 UTC, with ma4173 and ss6041 frozen (about half their
  300K output tokens left) and fl5273 at 3/8 on its third attempt.

**Proposed fix (not applied):** a patched copy of the adapter kept in this repo, first on
`PYTHONPATH`, whose `_apply` calls `self.lose()` when `h_t.get("failed")`. A loss then becomes
GAME_OVER and the harness's existing auto-RESET restores the level, as on the official games.
Alternatives: `EXPOSE_RESET=on` for these games (it also disables the no-budget-burn line), or
an auto-RESET after N consecutive actions that change nothing.

## Restarts: two VM reclaims

The cloud VM was reclaimed twice, at ~17:20 and ~19:54 UTC, each time about 10 minutes after
the session's background watcher task reached its 2-hour limit and the session went idle.
`RESUME_FROM` keeps finished games and replays every unfinished one from level 1, so:

| directory | started | games played there | result |
| --- | --- | --- | --- |
| `gpt61sol-nvidia-25games` | 15:07 | all 25 | 18 won; 7 killed mid-game |
| `-resume` | 17:44 | fl5273, hr2048, ma4173, ml2048, mt4926, ss6041, td4826 | ml2048, mt4926, td4826 won; 4 killed |
| `-resume2` | 20:20 | fl5273, hr2048, ma4173, ss6041 | hr2048 won; 3 stopped at 21:34 |

About $62 was spent on attempts the restarts threw away. Replaying the recorded actions
reproduces every saved frame exactly (649 of 649 frames on the four games the second reclaim
killed), so a mid-game resume is possible: an unfinished opt-in implementation is on the
branch `claude/mid-game-resume-wip` (not reviewed, not part of this run). Keeping a watcher
task alive for the whole run avoids the reclaims.

## Caveats

- One pass per game, and the hard games had up to three attempts, each a fresh start: the
  scored attempt of fl5273, hr2048, ma4173, ml2048, mt4926, ss6041 and td4826 is not their first.
- The 240-minute per-game clock restarted with each attempt.
- The NVIDIA baselines come from the games' metadata (the source of their "human" counts is
  NVIDIA's), so scores are not directly comparable with the official games' human baselines.
- The first two run directories are stored unpacked: `pack_run.py` refuses a run with games
  still marked playing, which a killed run has.

## Conclusions

- On unseen synthetic games the agent's understanding transfers: 22/25 won, 18 on the first
  try. Its weak spot is efficiency (extra actions), not discovery.
- The NVIDIA adapter's missing game over is the one integration bug that costs levels; fix it
  before the next community-games run.
- Long runs here need a continuous watcher task (or a mid-game resume), otherwise a reclaim
  wipes the progress of every unfinished game.
