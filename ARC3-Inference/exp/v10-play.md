# v10: the play-and-model agent (sp80, ls20, ft09)

## Hypothesis

The reverse-engineering agent (v6c-v9) fits an engine to a recording that
another agent played. v10 lets one agent do both on a live game: it fits
`engine.py` to everything played so far, plans its next moves on that
engine, and the harness checks every real move against the engine's
prediction. If the fitted engine is good enough to plan with, the agent
should win in fewer actions than the base play agent, which reasons about
the rules from the screen alone. The three games test three cases
([PLAY_DESIGN.md](../engine_re/PLAY_DESIGN.md), section 7):

- **ft09**, solved by the base agent (100 actions). Does the loop win in as
  few actions, and at what cost in tokens and time?
- **sp80**, the base agent's clear failure (1 of 6 levels). Does a fitted
  engine take the agent past level 1?
- **ls20**, 5 of 7 levels for the base agent in 866 actions, and the game the
  recording-based agent never modelled (17 of 867 steps). Does a partial
  model still cut actions?

## What the harness is

One conversation per game alternates a **plan round** (the current frame;
in python `state_now()` and `simulate(actions)` on the engine; then
`commit_moves(actions, note)`) and a **fit round**, which is the stepwise
harness of v6c-v9 ("fix step k"). `commit_moves` replays the whole game on
`engine.py` first and sends nothing while a step fails. Otherwise each move
is predicted with that engine, sent to the real game and compared by the
tests' rule (final frame, state, levels; one HUD-bar pixel tolerated). The
batch stops at the first difference, which opens a fit round on that step.
It also stops after a solved level. The design is
[PLAY_DESIGN.md](../engine_re/PLAY_DESIGN.md). The implementation is
[engine_re/README.md](../engine_re/README.md), "The play-and-model agent":
`engine_re/play_agent.py`, `live_game.py` and `run_play.py`.

This run used the default configuration. The escape hatch was off
(`fit_turns: null`), so fit rounds had no limit. A plan nudge came after 6
plan turns without a batch (`plan_turns: 6`). Batches held at most 10 moves.
After a game over, the harness played the RESET itself. Context was bounded
by compaction at 140K prompt tokens.

## The run

- `runs/engine-play/qwen38flash-v10` (`runs/engine-play/qwen38flash-v10.dvc`).
- Code `85e34e8`, committed at 22:18:12 UTC. The run started at 22:18:14 UTC on
  2026-10-05 (`config.json` `started`).
- The three games ran in parallel, one sample each. The model was
  qwen/qwen3.8-flash with T 0.7 and top_p 0.95; every request was served by
  Alibaba (`provider` in `transcript.jsonl`).
- The run directory also holds `run.log`, the console output. It was copied
  in from `runs/engine-play/qwen38flash-v10.log` before archiving. It also
  holds the scoring and evaluation outputs added afterwards: `score.json`,
  `evaluation.json`, `evaluation_best.md` and `<game>/evaluation_best.json`.
- The smoke run `runs/engine-play/smoke-ft09` came before the review and is
  not part of the results. It is not archived.

```bash
uv run --no-sync python -m engine_re.run_play --games sp80,ls20,ft09 \
  --out runs/engine-play/qwen38flash-v10 --model qwen/qwen3.8-flash --max-turns 300 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 240 --max-actions 500 --batch-size 10
```

Rerun at `85e34e8` from `ARC3-Inference/`, with `OPENROUTER_API_KEY` set and
the game files in `environment_files/`. To get the run:
`dvc pull runs/engine-play/qwen38flash-v10.dvc`. The anonymous login is
described in [README.md](README.md#conventions).

## Results

Sources:

- Everything comes from each game's `result.json`, except where a row names
  another file.
- **Moves matched** is the sum of `matched` over `batch_log`.
- **Plan/fit** splits come from the `phase` of the assistant records in
  `transcript.jsonl`.
- **Cached** is `cached_tokens / prompt_tokens`.
- **Per level** is `actions_per_level`, and **human** is `baseline_actions`
  (also in `environment_files/<game>/*/metadata.json`).

| | ft09 | sp80 | ls20 |
| --- | --- | --- | --- |
| end | **won** | turn limit (`budget_turns`) | turn limit (`budget_turns`) |
| score (`make score_run`) | 100.0 | 4.01 | 3.57 |
| levels | 6/6 | 2/6 | 1/7 |
| actions | 77 | 239 | 239 |
| per level (human baseline) | 4 (43), 7 (12), 14 (23), 16 (28), 23 (65), 13 (37) | 125 (39), 95 (58), 19 unsolved (25) | 21 (22), 218 unsolved (123) |
| batches sent (refused) | 16 (1) | 46 (3), plus 3 harness RESETs after a game over | 34 (0) |
| moves matched / sent | 66/77 (85.7%) | 218/239 (91.2%) | 223/239 (93.3%) |
| mismatches | 11 | 21 | 16 |
| fit rounds (accepted) | 11 (11) | 22 (21; one replaced after a refused batch) | 16 (16) |
| fit round length: mean, max (turns) | 8.5, 22 | 8.1, 33 | 13.2, 33 |
| turns: plan / fit | 156: 63 / 93 | 300: 122 / 178 (`phase_turns` says 177) | 300: 88 / 212 |
| output tokens: plan / fit | 289,302: 95,754 / 193,548 | 627,305: 246,363 / 380,942 | 675,320: 113,430 / 561,890 |
| reasoning tokens | 229,091 (79%) | 546,947 (87%) | 575,226 (85%) |
| prompt tokens | 17,416,346 | 48,259,100 | 48,003,588 |
| cached | 91.9% | 86.1% | 87.3% |
| largest prompt (`max_prompt_tokens`) | 147,354 | 248,354 | 266,697 |
| cost: total (plan / fit) | $0.605 ($0.254 / $0.351) | $1.967 ($0.707 / $1.260) | $1.904 ($0.549 / $1.355) |
| minutes | 70.0 | 177.3 | 182.5 |
| `commit_engine` calls | 11 | 21 | 18 |
| `final_test.txt` (final `engine.py`) | 78/78 | **0/240** (syntax error) | 240/240 |
| `engine_committed.py` replayed (`evaluate.py`) | 78/78 | 240/240 | 240/240 |

All three games together cost $4.476. They used 1,591,927 output tokens and
113,679,034 prompt tokens. `make score_run` gives a mean score of 35.86
over the three games (`score.json`).

Notes on the table:

- **sp80 turn count.** `phase_turns` counts 299 of sp80's 300 turns. Turn
  111 was a 32,768-token answer with no tool call (`finish_reason: length`),
  and the agent loop goes on to the next request without ending the turn
  (`agent.py`, `if not tool_calls: ... continue`).
- **The `best` field lags.** `result.json`'s `best` records the test that
  last improved the passing prefix: ft09 74 steps (turn 150), sp80 240,
  ls20 230 (turn 298). Steps played after that are not in it. Replayed on
  the whole trace, `engine_best.py` passes 78/78, 240/240 and 240/240
  (`evaluation_best.json`, `replay`).
- **Both unfinished games ended at 239 actions.** This is a coincidence. The
  turn limit bound both. Neither game reached its other caps: they ran 177
  and 183 of 240 minutes, spent under $6, used under 1.5M output tokens and
  played under 500 actions.
- **`run.log` holds 45 lines "HTTP 429; retry"**, plus one ProxyError and
  two ChunkedEncodingError retries. These lines do not name the game.
  `provider_errors` (answers that failed at the provider and were asked
  again) is 2, 2 and 3.
- **The largest prompts.** Notes taken while the run was watched gave
  about 236K (sp80) and 246K (ls20). They predate the last turns:
  `max_prompt_tokens` is 248,354 and 266,697.

### Mismatches by kind

These come from the `verdict` of each mismatched `move` record and the
regions in the fit round's report (`step_start`).

| kind | ft09 | sp80 | ls20 |
| --- | --- | --- | --- |
| first frame of a new level (cannot be predicted) | 4, 11, 25, 41, 64 | 125, 220 | 21 |
| HUD bar only (a per-level budget or rounding) | 13, 50, 73 | 126, 129, 222, 239 | 22, 48, 56, 168 |
| engine predicted "level solved", the game did not | - | 159, 204, 232, 235 | 74, 118, 188 |
| game over the engine did not predict | - | 41, 147 | - |
| a rule the engine did not have | 1, 52, 54 | 1, 13, 35, 102, 135, 137, 175, 180, 197 | 1, 2, 7, 8, 14, 15, 33, 141 |

Three of the level-entry mismatches were also unpredicted solves, where the
engine said the level was not solved: ft09 steps 4 and 41, and sp80 step
125.

Notes taken while the run was watched listed ls20's wrong win predictions
as steps 8, 74 and 118. The verdicts give 74, 118 and 188. At step 8 the engine predicted no
level change, and the game made none. The batch's note called that move a
test of whether covering the icon solves the level.

## Per-game timelines

Each fit round below is given as `sK tA-B (n)`: the round on step K started
at turn A, ended at turn B and took n turns (`fit_rounds`). Levels come
from the `move` records. Cumulative output tokens are summed over the
assistant records. A fit round counts for the level its step was played in,
so the round on a level's solving step counts for that level.

### ft09: won, 6/6 in 77 actions

| level | steps | actions (human, base agent) | turns | solved at turn (min, output tokens so far) | fit rounds (turns) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1-4 | 4 (43, 4) | 0-8 | 8 (3.2, 14,066) | 2 (9) |
| 1 | 5-11 | 7 (12, 7) | 8-17 | 17 (5.9, 26,830) | 1 (8) |
| 2 | 12-25 | 14 (23, 17) | 17-39 | 39 (13.8, 56,394) | 2 (12) |
| 3 | 26-41 | 16 (28, 24) | 39-55 | 55 (23.4, 103,864) | 1 (22) |
| 4 | 42-64 | 23 (65, 27) | 55-137 | 137 (64.6, 276,766) | 4 (36) |
| 5 | 65-77 | 13 (37, 21) | 137-156 | 156 (70.0, 289,302) | 1 (6) |

What each fit round found, from the commit messages (the `commit` records):

- s1 t4-6 (2): a click on a 3x3 block flips its colour (9 and 8), and every
  click shortens the HUD bar.
- s4 t8-15 (7): level 0 is solved when the bracketed panel matches the
  thumbnail in its centre. The win rule and level 1 were added.
- s11 t17-25 (8): one rule for every level: a white thumbnail cell means the
  thumbnail's centre colour, and grey means the other colour.
- s13 t26-31 (5): the bar is a per-level budget.
- s25 t39-46 (7): level 3 has a 3-colour cycle.
- s41 t55-77 (22): level 3 was solved 5 clicks earlier than predicted. The
  rule became "grey = any colour but the centre". This is the real win
  check (`cgj`).
- s50 t86-107 (21): the 9-click diamond that the old rule asked for did not
  solve level 4. The round itself was a 2-pixel bar difference. The bar is
  `64 - round(64*a/B)` with banker's rounding, and level 4's budget is 128.
- s52 t109-113 (4): a probe click on a pattern tile flips the tile and the
  blocks its magenta marks point at.
- s54 t130-134 (4): the marks move only blocks.
- s64 t137-144 (7): level 4 was solved as predicted, and level 5 was drawn.
  Each of its tiles marks its top neighbour.
- s73 t148-154 (6): level 5's budget is 128. Budgets 32/32/96/96/128/128
  reproduce every bar of the game, which are the real `kCv` values.

Plan rounds:

- 16 batches. A round took 3.9 turns on average.
- The longest round was t115-130 (15 turns, with plan nudges at t121 and
  t127). It enumerated level 4's cell flips in python (t124-t126) and sent
  the engine's 9-click solution. The first click mismatched (s54).
- Level 5 was played as the unique solution of a GF(2) system over the 22
  tiles (t148 note): 13 clicks, against a human baseline of 37.
- One batch was refused (t105). It was a probe sent during the s50 fit
  round, refused because the tests failed at step 50.
- There were 3 plan nudges (t83, t121, t127).

Where the actions went:

- Every level took fewer actions than the human baseline (77 against 208).
  Levels 2-5 also took fewer than the base agent (66 against 89).
- Probes: steps 1, 26, 51-53.
- The waste was level 4's 9-click diamond (steps 42-50). It was built on the
  old rule, and the game showed that rule did not decide level 4.

### sp80: turn limit, 2/6 in 239 actions

| level | steps | actions (human, base agent) | turns | solved at turn (min, output tokens so far) | fit rounds (turns) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1-125 | 125 (39, 44) | 0-61 | 61 (27.3, 119,293) | 6 (28) |
| 1 | 126-220 | 95 (58; base agent 67, unsolved) | 61-219 | 219 (141.0, 532,164) | 12 (90) |
| 2 | 221-239 | 19, unsolved (25) | 219-300 | - (95,141 output tokens in the level) | 4 (60) |

- **Level 0 (125 actions).**
  - Fit rounds s1, s13 and s35 modelled the moving bar and the walls.
  - s41 (t27-29) and s102 (t37-43) were game overs on the 30-action budget.
  - From t30 to t37 the agent swept the bar over the play area, testing
    whether the goal was a position, and pressed SPACE at a few spots. This
    took steps 43-102: 60 actions, including a RESET.
  - At t51 it read the animation frames and saw that SPACE releases a
    stream (its "beam").
  - Level 0 was solved at s125. The engine did not predict it: the fit round
    on step 125 (t61-69) learned the win.
- **Level 1 (95 actions).**
  - s147 (t95-104) was the game over after a 5th spill. The engine now ends
    the level on a 5th shot.
  - s159 (t107-115) was a wrong win prediction. Its turn 111 is the
    32,768-token answer with no tool call. At t115 a `commit_moves` was
    refused: the round's edits now failed step 147. That replaced the round
    with one on step 147, accepted at t117.
  - s197 (t140-157, 17): a click switches which bar the player moves.
  - s204 (t161-180, 19): another wrong win prediction. The rule became "the
    win is one shot that lights every socket".
  - A 27-turn plan round (t192-219, nudges at t198, t204, t210 and t216)
    searched bar arrangements on copies of `make_level(1)` (t206-t209). Its
    batch solved the level at s220.
- **Level 2 (19 actions, unsolved).**
  - s232 (t240-262, 22) and s235 (t263-296, 33) were wrong win predictions.
    The agent explained them by "the player's bar must emit the beams" and
    then "every emitter caught".
  - The last fit round (s239, t297-299) was accepted.
  - At t300 an `edit_file` with a `replace_text` whose `oldText` was cut
    mid-word ("...in one shot"; t") left `engine.py` with a syntax error at
    line 586. The tool reported it ("SYNTAX ERROR at line 586"), but the
    turn limit came first. This is the final test's 0/240.
    `engine_committed.py` and `engine_best.py` are the turn-299 commit,
    which passes 240/240.
- **Waste.**
  - 5 RESETs by the agent (steps 11, 72, 160, 181, 205) and 3 game overs
    with their automatic RESETs (41/42, 102/103, 147/148).
  - The 60-action positional sweep (steps 43-102).
  - Four wrong win predictions (159, 204, 232, 235).

### ls20: turn limit, 1/7 in 239 actions

| level | steps | actions (human, base agent) | turns | solved at turn (min, output tokens so far) | fit rounds (turns) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1-21 | 21 (22, 24) | 0-56 | 56 (22.5, 99,349) | 7 (43) |
| 1 | 22-239 | 218, unsolved (123, 98) | 56-300 | - (575,971 output tokens in the level) | 9 (169) |

- **Level 0 (21 actions, under the human baseline).**
  - The first plan round took 16 turns before the first move, with nudges
    at t6 and t12. In it the agent ran BFS over the maze (t7-t11).
  - Seven short fit rounds found the rules:
    - s1: the budget bar.
    - s2: the 5x5 block is the player and moves in 5-px cells.
    - s7: the goal box can be entered.
    - s8: bumping the icon is free and flashes the legend.
    - s14: the plate turns the legend dial 90 degrees.
    - s15: the plate is occluded, not collected.
  - s21 (t56-70) solved the level. The commit gives the rule as "the push
    into the icon solves the level while the legend dial shows the goal
    icon's rotation". That is in effect the real rule: the key's shape,
    colour and rotation must match the goal's (`bejndxqqzf`).
- **Level 1 (218 actions, unsolved).**
  - s22: the level-1 budget is 21.
  - s33 (t84-93): the yellow rings are refills. The agent dropped the
    "dial matches the icon" rule because, by its count, the three presses
    level 1 needs did not fit the budget. It replaced the rule with "one
    press latches the box".
  - s48 (t95-114): presses cost a step unless they align the dial.
  - s56 (t116-142, 26): the third press did align the dial with the goal
    icon. The game charged a step and showed no flash, where the engine
    predicted a free press with a flash. The agent concluded that "the box
    is open while the dial is upright" and played a RESET at step 57.
  - s74 (t145-178, 33) and s118 (t185-214, 29) were pushes with the dial
    upright, refused by the game. Each time the agent adopted a new unlock
    rule: "upright and no ring left", then "a press that lands upright
    latches the box".
  - s141 (t220-230): an empty bar costs a life and restarts the level.
  - s188 (t264-283) was a third refused push.
  - In its reasoning at t291-t294 the agent concluded that "level 1 is
    UNSOLVABLE under this model". It reset at step 229. At t295-t296 it
    wrote a rule close to the real one: the dial must equal the goal's
    rotation, and no ring may be left. The tests failed on it (t296-t297),
    and the agent reverted to the latch engine (t298-t299). The last batch (t300) sent the
    first 10 moves of a 57-action route.
- **The hidden state.** The real game was replayed on the trace and its key
  indices were read (`fwckfzsyc`, `hiaauhahz`, `cklxociuu` against the
  goal's `ldxlnycps`, `yjdexjsoa`, `ehwheiwsk`).
  - The goal needs rotation index 3 (270 degrees).
  - The key reached it after steps 56, 166 and 220. Each time it was lost
    before the block reached the goal: the RESET at 57, the fourth press at
    168, and the death at 223.
  - The three refused pushes (74, 118, 188) were made with rotation 0.
  - So the agent held the winning key three times and walked it away.
- **Where the cue came from.** The free press that flashed the goal box in
  level 0 (step 14) is level 0's tutorial hint. The real game shows it only
  in level 0, when a changer makes the key match a goal, and that press
  costs nothing (`real_engines_brief.md`, ls20 section 7). The agent took
  this hint for the unlock event itself.
- **Waste.** 5 RESETs (steps 57, 75, 119, 189, 229), 2 deaths on an empty
  bar (141, 223), and 3 refused pushes.

## Comparison with the base play agent

The base agent is the archived run `runs/20261004_135539`: the same model,
the main harness, and 500K output tokens per game. Its per-game figures
come from [experiments/engine-code-access/README.md](../experiments/engine-code-access/README.md).
Its per-level actions and wall time come from its `benchmark.json`
(`actions_per_level`, `final_wallclock_seconds`).

| game | | score | levels | actions | per level | output tokens | prompt tokens | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | base | 100 | 6/6 | 100 | 4 7 17 24 27 21 | 101K | 3.7M | $0.13 | 31 |
| ft09 | **v10** | 100 | 6/6 | **77** | 4 7 14 16 23 13 | 289K | 17.4M | $0.605 | 70 |
| sp80 | base | 3.74 | 1/6 | 111 | 44 67 | 501K | 13.0M | $0.59 | 117 |
| sp80 | **v10** | **4.01** | **2/6** | 239 | 125 95 19 | 627K | 48.3M | $1.967 | 177 |
| ls20 | base | **30.97** | **5/7** | 866 | 24 98 151 117 129 347 | 502K | 24.2M | $0.82 | 170 |
| ls20 | **v10** | 3.57 | 1/7 | 239 | 21 218 | 675K | 48.0M | $1.904 | 183 |

The mean score over the three games is 35.86 for v10 and 44.90 for the
base agent.

- **ft09.**
  - v10 won in 23% fewer actions, and in fewer actions on every level from
    2 on.
  - It used 2.9 times the output tokens, 4.7 times the prompt tokens, 4.7
    times the cost and 2.3 times the wall time.
  - The base agent's 101K output tokens are fewer than v10's fit rounds
    alone (194K).
- **sp80.**
  - v10 passed level 1 where the base agent gave up on it, and scored 4.01
    against 3.74.
  - It spent 125 actions on level 0, against 44, and 220 actions on two
    levels.
  - It used more output tokens (627K against 501K) and 3.3 times the cost.
- **ls20.**
  - Level 0 took 21 actions against 24.
  - The base agent solved level 1 in 98 actions. v10 did not solve it in
    218, and finished with 1 level against 5.
- **The binding budget differed.** The base agent stopped at its 500K
  output-token limit on sp80 and ls20 (`state: gave_up`). v10 stopped at
  300 turns with 627K and 675K output tokens.

For reference, the engine-code arm (the base agent reading the obfuscated
game source, same README) won all three games:

- ft09: 75 actions, 52K output tokens, $0.08.
- sp80: 6/6 in 143 actions, 162K output tokens, $0.22.
- ls20: 7/7 in 466 actions, 252K output tokens, $0.46.

## Comparison with the recording-based runs

The recording-based runs fit an engine to the base agent's recording
([RESULTS.md](../engine_re/RESULTS.md), v2 main results and v4):

- **ft09.**
  - v2 reproduced the 101-step recording in 87 turns, 36 minutes and $0.22.
    v4 did it in 22 minutes and $0.14.
  - v10 needed 93 fit turns for its own 78 steps, and it also won the game.
- **sp80.**
  - v2 reproduced the 112-step recording (two levels' worth, level 1
    unsolved) in 94 turns, 63 minutes and $0.37.
  - v10 modelled three levels (240 steps), in 178 fit turns and $1.26 of
    fit-phase cost.
- **ls20.**
  - The recording-based agent never got past step 17 of 867 (v2), with 3%
    held-out.
  - v10's engine reproduces all 240 steps it played, though only levels 0
    and 1.
  - This is not a like-for-like comparison. Levels 0 and 1 have no pushers
    (the real game's `gbvqrjtaqo` sprites, 17-frame animations): no step of
    v10's trace has more than 6 frames. The recording ran on through five
    levels.
- **lp85.** The v9 stepwise harness reached 65 of 120 lp85 steps in 100
  turns, with 13 accepted commits, about 7.7 turns each
  ([v9-bar-tolerance.md](v9-bar-tolerance.md)). v10's fit rounds averaged
  8.5 turns on ft09 and 8.1 on sp80. On ls20 they averaged 13.2, because
  the agent kept re-deriving one rule.

## Held-out accuracy

The command was
`uv run --no-sync python -m engine_re.evaluate runs/engine-play/qwen38flash-v10 --engine best`,
with 8 rollouts of 40 random actions per level reached and seed 0. The
output is `evaluation_best.md` and `<game>/evaluation_best.json`.

`evaluate.py` cannot name `engine_committed.py`, so the committed engines
were evaluated as `best` in a scratch copy holding `trace/` and
`engine_committed.py`. The numbers were identical:

- sp80's and ls20's `engine_committed.py` are byte-identical to their
  `engine_best.py`.
- ft09's two engines differ only in one docstring line.

| game | recorded replay | held-out steps exact | of those that change the screen | per level (exact rate, mean steps to first mismatch, rollouts fully matching) |
| --- | --- | --- | --- | --- |
| ft09 | 78/78 | 454/1920 (24%) | 370/1117 (33%) | L0 5%, 0.2, 0/8; L1 21%, 1.8, 0/8; L2 31%, 5.4, 0/8; L3 32%, 8.1, 0/8; L4 24%, 4.5, 0/8; L5 29%, 3.9, 0/8 |
| sp80 | 240/240 | 860/970 (89%) | 816/919 (89%) | L0 100%, 40.4, 8/8; L1 82%, 26.1, 4/8; L2 84%, 29.0, 5/8 |
| ls20 | 240/240 | 640/640 (100%) | 640/640 (100%) | L0 100%, 40.0, 8/8; L1 100%, 40.0, 8/8 |

For scale, the recording-based v2 engines scored: ft09 85% (L0 7%, L1-L5
100%), sp80 91% (L0 99%, L1 82%) and ls20 3%.

The source scan flags sp80's engine for "subprocess/network". These are
false positives: the pattern matches the word `socket` in the agent's
sprite names (`socket_a`, `tags=("socket",)`).

## Engines compared with the real games

The real games are described in [real_engines_brief.md](real_engines_brief.md)
and their sources are in `environment_files/<game>/*/<game>.py`. The
held-out failures were reproduced one action at a time against the real
game; the scripts were throwaway diagnostics and are not kept.

### ft09

**Right.** Every tile mechanic of all six levels:

- the colour cycles, including level 3's three colours;
- the plain tiles and the pattern tiles (an NTi flips itself and the tiles
  its 6-marks point at);
- the win check, as the real `cgj`;
- the budgets 32/32/96/96/128/128 and the bar with banker's rounding.

Two checks showed this:

- Every grid cell of every tile (9 per tile), clicked from each level's
  start: 0 mismatches in 1,026 clicks, with the HUD row masked out of the
  comparison.
- 15-click random tile sequences on levels 4 and 5: all matched.

**Wrong, two things the agent never played.** Every one of its 77 clicks
landed on a tile: 61 plain tiles (`Hkx`) and 16 pattern tiles (`NTi`), by
the real sprite under each click. None landed on a constraint sprite or on
empty space.

- A click off the tiles costs a bar step in the engine but nothing in the
  real game (`real_engines_brief.md`, ft09 section 3). The HUD bar then
  differs from that click on, which causes most of the held-out misses.
  - The held-out run was repeated with screen row 63 masked out of the
    comparison, as a diagnostic outside `evaluate.py`.
  - The result was 1512/1920 (79%): L0-L3 100%, L4 28%, L5 44%.
- On levels 4-5 the engine treats the constraint sprites (`bsT`) as
  clickable cells. In the real game they are inert. Examples: level 4
  `ACTION6(50,52)` on the `bsT` at (23,26), and level 5 `ACTION6(39,26)` on
  the one at (18,11).
- The recording-based v2 engine got free off-tile clicks right because the
  base agent's recording contained them. Its own blind spot was the level-0
  blink.

### sp80

**Right:**

- the selected-bar movement and walls in screen space, including the 180
  degree rotation of levels 1-2 (its `beam_dir -1` and the bar on row 63);
- clicks select a bar ("swap which bar is the player");
- the stream ("beam") spreads along a bar and falls off both ends;
- a cup fills when the stream enters its notch;
- the 5th spill loses;
- the budgets of levels 0 and 1 (30, 45).

**Wrong:**

- **The win condition.** A real win needs every cup filled **and no floor
  touched** in one spill. The real game was replayed on the trace and its
  cup and floor flags were read (`cevwbinfgl`, `kfdcqkodyy`) after each
  SPACE.
  - Steps 48, 119 and 123 (level 0) filled both cups and touched the floor.
  - Steps 232 and 235 (level 2) filled all three and touched the floor.
  - The agent explained these by "a splash does not fill" (level 0) and
    "the player's bar must emit" / "one emitter" (level 2). Neither rule
    exists.
- **After a failed spill** the real game selects the piece nearest the
  origin. The engine keeps a fitted "control passes to another bar" rule.
  Every first held-out mismatch on levels 1-2 is an ACTION5 that leaves the
  wrong bar selected. This is v2's failure too (`exp/analysis/sp80.md`,
  kind C).
- **Level 2's budget** is 105 in the engine and 100 in the real game. The
  one-pixel HUD tolerance hid it: tests at steps 224, 229, 232, 235, 237 and
  238 pass with a warning.

### ls20

**Right:**

- the 5x5 player moving one 5-px cell;
- walls;
- blocked moves cost a step;
- the budget of 42 and 21 shown on a 42-column bar emptying from the left;
- a refused push into the goal is free and flashes 6 frames;
- the rings refill and are free;
- the plate turns the dial 90 degrees on every entry;
- 3 lives, and an empty bar restarts the level.

All 640 held-out steps match. Random play from levels 0-1 never reaches the
goal with the right key, so it never tests the win rule.

**Wrong:** the unlock rule. The committed engine opens the box when a press
lands the dial upright with no ring left. The real rule is a key equal to
the goal's (rotation 270 in level 1). The engine also keeps an unused
`try_unlock` (dial equal to the goal's rotation and no ring left), left
over from an earlier version.

## What the harness did well, and badly

**Well:**

- **No move was sent on a stale engine.** Refusals: ft09 t105; sp80 t100,
  t115 and t253. In all, 85.7% (ft09), 91.2% (sp80) and 93.3% (ls20) of the
  moves sent were predicted exactly.
- **Rules generalised across levels when the game allowed it.** On ft09 one
  thumbnail rule held from t25. The set rule from t77 is the real check.
  The pattern rule (t113-t134) carried level 5 with no new mechanic.
- **The agent searched on its engine, as the design intended.**
  - ft09: enumerations at t60, t75-t76 and t124-t126, and the GF(2) solve
    of level 5 (t148).
  - sp80: an arrangement search on copies of `make_level(1)` (t206-t209),
    which led to the level-1 solve.
  - ls20: BFS routes (t7-t11, t214 "47 actions verified by search + engine
    simulation", t249-t261).
- **Wins below the human baseline.** ft09 used 77 actions against 208.
  ls20's level 0 took 21 against 22.

**Badly:**

- **Long fit rounds froze the game.** ft09 s41 took 22 turns (level 3's
  win rule and drawing level 4) and s50 took 21 (mostly the bar's
  rounding). ls20 took 26, 33 and 29 turns on s56, s74
  and s118. sp80 took 33 on s235.
  - Fit turns were 60% (ft09), 59% (sp80) and 71% (ls20) of all turns.
  - They used 67%, 61% and 83% of the output tokens.
- **The fitted engine settles on rules that reproduce the record but are
  wrong, and the agent then plays by them.** This led to seven wrong win
  predictions (sp80 4, ls20 3). On ls20 it led to abandoning a correct
  state (t142, the RESET at step 57). The harness checks reproduction, not
  plausibility. A wrong rule that fits every step is accepted as readily as
  the right one.
- **Wasted actions.**
  - sp80: the positional sweep (60 actions), 5 RESETs and 3 game overs.
  - ls20: 5 RESETs and 2 deaths in level 1, which took 218 actions against
    the base agent's 98.
- **Slow first plan rounds.** ls20's first plan round took 16 turns before
  the first move. sp80's level-1 plan round took 27 turns. The plan nudge
  fired at 6, 12, 18 and 24 turns and did not shorten either.
- **The context grew without bound.** Compaction trims old outputs and
  reasoning but keeps every message.
  - After the first turn over 140K (t35 ft09, t39 sp80, t36 ls20) it fired
    on every later turn over 140K: 13, 195 and 182 `compact` records.
  - The prompt still grew to 248,354 (sp80, t298) and 266,697 (ls20, t300).
  - Per 50 turns on ls20, the mean prompt rose from 72,754 (t1-50) to
    240,640 (t251-300), and the cost from $0.148 to $0.376.
  - The cache rate dipped to 67.4% in sp80's turns 101-150 and 78.9% in
    ls20's turns 151-200. These figures come from the per-turn `usage` in
    `transcript.jsonl`.
- **The final engine can be broken.** sp80's `workspace/engine.py` failed
  to load (t300). The run reports the committed engine separately, but
  `final_test.txt` and `summary.md` show 0/240.
- **One answer hit the 32K output cap without a tool call** (sp80 t111,
  32,768 tokens, 5.2% of the game's output).

## Caveats

- **One sample per game at temperature 0.7.** ls20's outcome turned on one
  inference at t142. A second sample could go either way.
- **The turn limit bound both unfinished games**, at 300 turns, 177 and 183
  minutes, and under $2 each. Neither game reached its other caps. More
  turns would have bought more levels only if the rules were right, which
  on ls20 they were not.
- **Prompt growth.** Turn cost and length rose over each long game, so the
  token and cost figures depend on how long the game ran.
- **The base agent is not a controlled comparison.** It used a different
  harness and its own 500K output-token limit, which bound for sp80 and
  ls20. The score compares outcomes, not the two harnesses at equal budget.
- **Held-out play is random.** It rarely reaches a win or a loss rule, so
  ls20's 100% hides a wrong win rule. ft09's misses come mostly from a
  kind of click the agent never had reason to play.
- **Wall time includes rate-limit waits.** `run.log` has 45 HTTP 429
  retries of about 5-6 s each, not attributed to a game.

## Next steps

- **Run the second configuration of the design: `--fit-turns 25`.** In
  this run it would have fired on ls20 s56, s74 and s118 and on sp80 s235.
  Measure whether playing on unchecked to the next RESET or level change
  costs fewer actions than the long rounds did.
- **Scale the turn budget with the levels.** For example, a fixed number
  of turns per level, or a turn budget that grows with each level solved.
  ft09 used 156 turns for 6 levels, while sp80 and ls20 spent 300 on 2 and
  1.
- **Fire the plan nudge earlier.** Try 3 turns instead of 6, including the
  first plan round (ls20: 16 turns before the first move).
- **Make the lenient edit op the default.** v8's lenient edits would have
  avoided sp80's broken final engine, whose exact `replace_text` had a
  truncated `oldText`. Restoring `workspace/engine.py` to the committed
  engine when a run ends broken would also help.
- **When a predicted win fails, ask what differs.** Have the fit message
  ask for the difference between the winning and the failed attempt, from
  the frames: the floor flash in sp80, the dial pose in ls20. Then the fix
  is a stated rule, not a new latch.
- **Bound the context.** The play agent refuses `--condense`
  (`play_agent.py`); v9c's threshold condenser kept 92% cached at a bounded
  prompt.
- **Count turns without a tool call** in `phase_turns`.
- **Run lp85 as the fourth game**, where every v5-v9 measurement was made.

## How the numbers were checked

- **Each game's artifacts are complete:** `result.json`, `final_test.txt`,
  `trace/`, `visible_trace/`, `tests.jsonl`, `transcript.jsonl`, `images/`,
  `engine_versions/`, `engine_best.py`, `engine_committed.py`,
  `workspace/engine.py` and `artifacts/<game_id>_p0_events.jsonl`.
- **`trace/` agrees with `result.json`.** It has 78, 240 and 240 steps.
  Actions, `actions_per_level`, levels and the score recomputed from it with
  TAAF's formula (`live_game.py`) equal `result.json`'s.
- **The batch log agrees with `result.json`.** The sum of `sent` over
  `batch_log` equals `moves_sent`. The batches with a mismatch equal
  `mismatches`. sp80's `batch_log` also lists the 3 harness RESETs.
- **The usage figures agree.** Summing the `usage` of the assistant records
  in `transcript.jsonl` gives exactly `result.json`'s prompt, cached,
  output and reasoning tokens and cost.
- **The events sidecars rebuild the trace.**
  `trace.trace_from_run(<game dir>, game, environment_files)` rebuilt each
  game from its events sidecar with 0 replay mismatches (78, 240 and 240
  verified steps). It gives the same actions, final frames, states and
  levels as `trace/`.
- **`make score_run` agrees.**
  `make score_run SCORE_RUN_DIR=runs/engine-play/qwen38flash-v10` gives
  sp80 4.013470, ls20 3.571429 and ft09 100.000000, equal to `result.json`.
