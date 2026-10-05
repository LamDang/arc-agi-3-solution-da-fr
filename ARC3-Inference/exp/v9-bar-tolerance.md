# v9: the HUD-bar tolerance (v9, v9c, v9t)

## Hypothesis

Every run that reached lp85's level 2 (v6c, v6cc, v8) spent its remaining
turns on one pixel of the HUD bar, the rounding of a per-level move budget
drawn on a 64-pixel column. If a single differing pixel at the frame's edge
passes with a warning, and the prompt says how to model such a bar, the agent
moves on to the levels after it. Two variants of the same harness ran beside
it: the threshold condenser (v9c), and a 10,000-token thinking budget per
answer (v9t).

## What changed in the harness

From v8 (`8dcb70c`, with the condenser commits of
[v8-condense-per-turn.md](v8-condense-per-turn.md) in between, off unless
`--condense`):

- `34e7779` (**v9**): a recorded step whose final frame differs from the
  engine's by a single pixel within 2 px of the frame's edge passes with a
  warning (`tester.HUD_BORDER`, `TestReport.tolerated`); the level-start
  frame too. The stepwise prompt's "How to work" says that an edge strip
  that changes on every action is a step or time budget, to model it as a
  per-level budget drawn proportionally and rounded, and that one pixel off
  is tolerated.
- `d0bfdac` (**v9c**, with `--condense`): the condenser fires where
  compaction fires (after a turn whose request went over 140K prompt tokens)
  and its view stays the fixed prefix of every request until the next
  firing.
- `5f683d5` (**v9t**, with `--thinking-budget 10000`): OpenRouter's
  `reasoning: {"max_tokens": N}`, at most N thinking tokens per answer.

## Runs

- v9: `runs/engine-re/qwen38flash-v9-lp85` (`qwen38flash-v9-lp85.dvc`), code `34e7779`
- v9c: `runs/engine-re/qwen38flash-v9c-lp85` (`qwen38flash-v9c-lp85.dvc`), code `5f683d5`
- v9t: `runs/engine-re/qwen38flash-v9t-lp85` (`qwen38flash-v9t-lp85.dvc`), code `5f683d5`

Commands (from `ARC3-Inference/`, at the commit above):

v9:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v9-lp85 --model qwen/qwen3.8-flash --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

v9c:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v9c-lp85 --model qwen/qwen3.8-flash --condense \
  --max-turns 100 --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

v9t:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v9t-lp85 --model qwen/qwen3.8-flash \
  --thinking-budget 10000 --max-turns 100 --max-output-tokens 1500000 --max-cost 6 \
  --max-minutes 115
```

## Results

v8 for comparison:

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v8 | turn limit | 100 | 20 | 1 | 71.3 | $0.517 | 10,103,646 | 85% | 336,912 | 297,568 | 5 | 5 |
| v9 | turn limit | 100 | 65 | 65 | 49.3 | $0.359 | 9,701,713 | 92% | 205,589 | 165,193 | 13 | 13 |
| v9c | turn limit | 100 | 65 | 0 | 57.7 | $0.364 | 9,427,616 | 92% | 245,628 | 196,015 | 12 | 12 |
| v9t | turn limit | 100 | 54 | 24 | 45.2 | $0.373 | 10,034,252 | 91% | 192,058 | 122,827 | 12 | 12 |

Turn at which each milestone was first accepted:

| steps passing | v8 | v9 | v9c | v9t |
| --- | --- | --- | --- | --- |
| 9: level 0 solved (step 8) | 31 | 25 | 31 | 18 |
| 18: level 1 solved (step 17) | 68 | 45 | 46 | 39 |
| 20: step 20 (level 2) | 68 | 49 | 46 | 45 |
| 37: level 2 solved (step 36) | - | 63 | 61 | 82 |
| 54: level 3 solved (step 53) | - | 82 | 75 | - |
| 65: 65 steps (level 4 to its last step) | - | 96 | 85 | - |

Thinking per turn:

| run | reasoning tokens per turn: mean | median | max | turns with 10,000 or more | largest prompt |
| --- | --- | --- | --- | --- | --- |
| v9 | 1,652 | 269 | 26,735 | 3 | 142,982 |
| v9c | 1,960 | 297 | 23,683 | 6 | 151,861 |
| v9t | 1,228 | 172 | 10,000 | 1 | 148,755 |

Timeline: `tN: k` means the commit (or, before v7, the passing test) accepted at turn N left steps 0..k-1 passing (the `advance` records of `transcript.jsonl`).

- v9: t9: 8, t25: 9, t33: 10, t37: 17, t45: 18, t49: 25, t54: 36, t63: 37, t71: 53, t82: 54, t86: 55, t92: 61, t96: 65
- v9c: t14: 2, t17: 8, t31: 9, t36: 17, t46: 25, t49: 36, t61: 37, t68: 53, t75: 54, t78: 55, t82: 61, t85: 65
- v9t: t4: 2, t9: 8, t18: 9, t22: 10, t26: 17, t39: 18, t43: 19, t45: 25, t69: 35, t77: 36, t82: 37, t94: 53

## Findings

- **20 to 65 of 120 steps.** v9 passed steps 0-64 at turn 96: levels 0-3 and
  level 4 up to its last step. Step 65, which solves level 4 and shows level
  5's first frame, fails because the engine has no level 5 yet (`IndexError`
  in `make_level`). v6c and v8 had 20.
- **Level 2 took 18 turns** (step 18 passing at turn 45, step 37 at turn 63),
  where v8 spent 32 turns on step 20 alone. The tolerance let the runs past
  step 20 (tests during v9 and v9c tolerated a bar pixel at steps 16, 20-24,
  42, 47, 49, 52 and 56-60), and their best engines then drew the bar
  exactly: the best tests of v9 and v9c tolerate no pixel (v9t's best engine
  still relies on it at steps 42, 47, 49 and 52).
- **Cheaper than v8.** v9 cost $0.359 (v8: $0.517) in 49.3 minutes (v8:
  71.3), with 205,589 output tokens (v8: 336,912) and 92% of its prompt
  tokens cached.
- **The threshold condenser keeps the cache.** v9c fired 4 times in 100
  turns, kept 92% of its prompt tokens cached (as v9) and sent about as many
  prompt tokens as compaction (9,427,616 against 9,701,713; prompt cost
  $0.248 against $0.263). It also reached 65 steps, at turn 85. Its final
  `engine.py` did not load (a syntax error), so its final test is 0;
  `engine_best.py` keeps the 65.
- **The thinking budget bound once.** v9t's reasoning reached the 10,000-token
  budget in 1 of 100 turns (turn 73); its mean per turn was 1,228 against
  1,652 for v9. So v9t is, in effect, a second sample of v9: its best engine
  passes 54 steps (its last commit, at turn 94, left 53 passing), for $0.373.
- **Against the play agent.** The play agent that made the recording solved
  all 8 levels of lp85 with 6,496,936 prompt tokens (6,074,624 cached) and
  172,611 output tokens in 44.6 minutes; the v9 agent used 9,701,713 prompt
  tokens and 205,589 output tokens to reproduce 65 of its 120 steps.
- One sample per configuration: v9 and v9t, the same harness with and
  without a budget that barely bound, reached 65 and 54.

## Engines compared with the golden engine

A read of v9's `workspace/engine.py` and v9c's `engine_best.py` against the
real game (`environment_files/lp85/*/lp85.py`):

| | v9 (`workspace/engine.py`) | v9c (`engine_best.py`) |
| --- | --- | --- |
| steps passing | 0-64; step 65 raises `IndexError` in `make_level(5)` | the same |
| contract tests | 4/5 | 4/5 |
| arrow to loop | each arrow sprite tagged with its loop (as the golden `button_<loop>_<L\|R>` tag) | loops keyed by arrow position, one loop per position |
| win rule | every block framed by four same-colour marks shows that colour: the golden rule on all levels | mark-slot grouping finds phantom slots on levels 5 and 7 (frames 6 px apart in one colour), which would make them unwinnable |

- **Shared and right.** Both model the rotation as one general rule (click an
  arrow, rotate its loop with wrap-around, spend a move, check the win), with
  no per-level branches. Both have the golden bar formula
  `round(64 * used / budget)` with budgets 13, 60, 80, 150 and 80, and the
  eight level dashes exact. Both take the direction from the arrow colour
  (8 backward, 14 forward), which holds in every golden level.
- **v9c's code** is shorter but has 12 unused `SHAPE_*` constants, a stale
  scaffold comment, and ring cells duplicated between its `BLOCKS` maps and
  ring helpers.
- **What both would break on the unseen levels 5-7.** The golden engine fires
  every button under the click, in sprite order, and the later levels stack
  buttons (level 5: 39 R buttons at 9 positions; levels 6-7 stack two or
  three loops per arrow), while both engines rotate one loop per click.
  Neither raises GAME_OVER on the click that exhausts the budget. Both store
  loops as hand-written data, so each new level needs its loop cells in
  golden order and each arrow's loop found by experiment (level 5 has 36
  loops, rings running counter-clockwise, only R buttons, budget 80).
- **Verdict.** v9's engine is the better base. It needs three changes: rotate
  every arrow under the click, raise GAME_OVER when the budget runs out, and
  the data of levels 5-7.
