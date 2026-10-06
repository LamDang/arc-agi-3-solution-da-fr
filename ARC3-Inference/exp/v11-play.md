# v11: the play-and-model agent, replica stepped directly, with support (sp80, ls20)

## Hypothesis

v10 ([v10-play.md](v10-play.md)) won ft09 but stopped at 2/6 on sp80 and 1/7
on ls20. Its model composed routes in text, used `simulate` only to check
them, searched on hand-written copies of the rules, and kept wrong rules
that every step fitted (seven wrong win predictions). v11 lets the model
play moves on its replica directly, measures and shows how much of the
replica the recorded steps ever ran (support), and takes over the base
harness's prompt guidance. If the replica becomes the planning tool, the
agent should solve more levels in fewer actions than v10 on sp80 and ls20,
at the same turn budget.

## What changed

Since v10 (`85e34e8`); [PLAY_DESIGN.md](../engine_re/PLAY_DESIGN.md)
sections 3.2, 3.11 and 3.12 have the detail.

- **No `simulate`.** The model plays moves with `replica.step` on copies of
  `state_now()` and `replica.make_level(n)`. The built-in `engine` is now
  `replica` (the file is still `engine.py`), so that the model does not take
  it for the real game. `commit_moves` takes the Actions as python prints
  them (`Action(4)`, `Action(6, x=12, y=40)`, `Action(0)` for RESET), and the
  messages name moves the same way.
- **Support.** The candidate runner records the engine lines each step runs
  (a `sys.monitoring` line tracer) and the truth value of each `and`/`or`
  operand and `if` test; the passing steps are folded into
  `engine_committed.support.json`. The model sees it as a margin in
  `read_file` and the listings, as per-move lines in the `commit_moves`
  output (weakest line, untested and thin lines, unseparated conditions), in
  the mismatch message and test report, as "Rules with little support" in
  the PLAN message, and through `traced()` and `support()` in python.
- **Prompt rules.** Test important rules in different conditions (plan rule
  4); separate the parts of an `and`/`or` before trusting it (plan rule 5);
  do not over-engineer the replica on one observation (fit rule).
- **Ported from the base harness** (section 3.12): the animation digest
  (transient cells and a timeline); `notes.md`, shown in each PLAN message;
  warnings before a batch (a predicted game over, moves that change
  nothing); the colour legend, action meanings, level transfer, level-start
  paragraph, scene-of-objects rule and "prefer another python call".
- **`--cut-untested`** exists but was off in this run.

## The run

- `runs/engine-play/qwen38flash-v11-a` (`runs/engine-play/qwen38flash-v11-a.dvc`).
  Code `5d2bda7` (the merge of `99b13dc`, `a803063` and `a0dbb9a`, 05:48:34
  UTC on 2026-10-06); `690198b`, committed during the run, only adds the
  transcript page. The run started at 05:49 UTC (`images/turn000_frame0.png`);
  `config.json` `started` and `benchmark.json` `start_time` say 08:08:00
  because the last resume rewrote them.
- Two games in parallel, one sample each; qwen/qwen3.8-flash with T 0.7 and
  top_p 0.95, every request served by Alibaba.
- Container reboots interrupted the run; each game was resumed 6 times
  (`resumes`; `resumed` records at sp80 turns 16, 22, 46, 88, 134, 138 and
  ls20 turns 9, 24, 66, 100, 159, 172).
- The transcript page "v11 Play Transcripts" is built by
  `engine_re/tools/play_transcripts/build.py`, one card per turn.

```bash
uv run --no-sync python -m engine_re.run_play --games sp80,ls20 \
  --out runs/engine-play/qwen38flash-v11-a --model qwen/qwen3.8-flash --max-turns 300 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 240 --max-actions 500 --batch-size 10
```

Rerun at `5d2bda7` from `ARC3-Inference/`, with `OPENROUTER_API_KEY` set and
the game files in `environment_files/`. The same command resumes an
interrupted run. To get the run: `dvc pull runs/engine-play/qwen38flash-v11-a.dvc`.

## Results

Sources are as in v10: each game's `result.json`, and the per-turn `usage`
and `phase` of the assistant records in `transcript.jsonl`. The sums of the
records' `usage` equal `result.json`'s tokens and cost exactly, the sum of
`sent` over `batch_log` equals `moves_sent`, and the score recomputed with
TAAF's formula equals `result.json`'s.

| | sp80 | ls20 |
| --- | --- | --- |
| end | turn limit (`budget_turns`) | turn limit (`budget_turns`) |
| score | **47.6** | **10.7** |
| levels | **4/6** | **2/7** |
| actions | 122 | 231 |
| per level (human baseline) | 11 (39), 19 (58), 21 (25), 71 (148) | 13 (22), 65 (123), 153 unsolved (73) |
| batches sent (refused) | 26 (1) | 36 (2) |
| moves matched / sent | 101/122 (82.8%) | 212/231 (91.8%) |
| mismatches | 21 | 19 |
| fit rounds (accepted) | 21 (20; the last open at the turn limit) | 19 (18; the last open) |
| fit round length: mean, max (turns) | 6.1, 14 | 12.0, 41 |
| turns: plan / fit | 300: 177 / 123 (`phase_turns` says 175 plan) | 300: 81 / 219 |
| plan nudges | 19 | 3 |
| output tokens: plan / fit | 612,788: 341,156 / 271,632 | 468,632: 59,437 / 409,195 |
| reasoning tokens | 505,629 (83%) | 376,296 (80%) |
| prompt tokens | 51,351,421 | 52,641,756 |
| cached | 87.6% | 87.6% |
| largest prompt | 289,211 | 296,762 |
| cost: total (plan / fit) | $1.965 ($1.238 / $0.727) | $1.937 ($0.535 / $1.402) |
| minutes | 165.9 | 135.5 |
| `commit_engine` calls | 19 | 18 |
| `final_test.txt` | 122/123 | 232/232 |

The two games together cost $3.902. They used 1,081,420 output tokens and
104.0M prompt tokens. The mean score is 29.17, against 3.79 for v10 on the
same two games.

- **Every solved level beat the human baseline**, so each scores the
  maximum: sp80's score is 10 of 21 level weights, ls20's 3 of 28.
- **sp80 solved level 3 on turn 299 of 300.** Its last turn opened the fit
  round on step 122 (level 4's first frame, not drawn yet), the one failing
  step in `final_test.txt`.
- **The turn limit bound both games**, as in v10; both stayed under 240
  minutes, $6 and 1.5M output tokens. `run.log` has 9 retries ("analyzer
  endpoint unreachable"), `provider_errors` is 0, and every answer ended
  with a tool call.

### Mismatches by kind

| kind | sp80 | ls20 |
| --- | --- | --- |
| first frame of a new level (cannot be predicted) | 11, 30, 51, 122 | 13, 78 |
| level solved, the replica did not predict it | 11 | 13 |
| replica predicted "level solved", the game did not | 24 | 30, 111, 112, 172 |
| HUD bar only (a per-level budget) | 15, 35, 57, 84 | 14, 58 |
| lost life the replica did not predict | - | 115 |
| a rule the replica did not have | 1, 6, 8, 12, 25, 26, 36, 37, 38, 42, 64, 91 | 1, 6, 7, 12, 25, 54, 86, 114, 210, 231 |

The replica predicted the solves at sp80 steps 30, 51 and 122 and ls20 step
78. On ls20 a predicted solve reads "your replica raised an error
(IndexError)", since `make_level(n + 1)` was not drawn yet; so did the wrong
predictions at steps 30, 111 and 112.

## Per-game timelines

The conventions are v10's. `sK tA-B (n)` is the fit round on step K, from
turn A to turn B, n turns. Levels come from the `move` records, and output
tokens are summed over the assistant records.

### sp80: turn limit, 4/6 in 122 actions

| level | steps | actions (human, base agent) | turns | solved at turn (min, output tokens so far) | fit rounds (turns) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1-11 | 11 (39, 44) | 0-25 | 25 (10.1, 40,852) | 4 (22) |
| 1 | 12-30 | 19 (58; base agent 67, unsolved) | 25-78 | 78 (37.5, 168,632) | 6 (32) |
| 2 | 31-51 | 21 (25; base agent did not reach it) | 78-210 | 210 (125.4, 508,608) | 6 (51) |
| 3 | 52-122 | 71 (148) | 210-299 | 299 (165.7, 612,468) | 4 (17), plus s122 open |

- **Level 0 (11 actions; v10 125, base agent 44).**
  - s1, s6 and s8 (t4-23) found the bar's movement, its ceiling and the HUD
    budget. SPACE at step 4 matched: the replica had it change nothing, and
    the game reverted everything after a 28-frame animation.
  - From the animation digest the model decoded the pour in text: a fluid
    that runs off the bar's unsupported ends, which had fallen on the bins'
    outer arms. The batch at t25 moved the bar so that its ends sat over the
    bins' gaps and pressed SPACE (step 11). The replica still had SPACE as a
    no-op, so the solve was not predicted.
  - s11 (t25-36) wrote the pour model and the win ("every cavity wet and
    nothing lost") and drew level 1, a 180-degree turned board.
- **Level 1 (19 actions; v10 95, base agent 67 unsolved).**
  - s12 (t48-51): the arrow keys are screen directions on the turned board.
  - s24 (t59-63): a wrong win prediction. The model had dropped "every
    cavity filled" because no bar position filled all three; step 24 lost
    nothing, filled one, and the conjunction came back.
  - Two one-move probes found the missing mechanics: the bar destroys a red
    block it moves into (s25, t64-71), and a click on a block swaps roles
    (s26, t72-77).
  - At t75, still in the fit round, a BFS over `replica.step` and
    `replica.pour` found a 3-move route (then SPACE). The batch at t78 solved
    the level as predicted. Its support line said the SPACE rested on lines
    one step had run, with `cav and not lost and cav <= wet` never separated
    (`cav` missing): accurate, since no step had tested it with `cav` false.
- **Level 2 (21 actions; human 25).**
  - Probes at t91 (two clicks, three moves; s35 was a HUD pixel). A
    diagnostic pour (t105) showed the stream eroding the bar's last cell
    into a red 1x1 (s36, t105-115); s37 (t147-158) and s38 (t168-170) pinned
    down how that "tail" grows.
  - The search took most of the level's 132 turns. A BFS over `replica.step`
    on deep copies timed out at 120 s (t117), and so did a search on the
    model's own fast model (t128); each timeout restarted the kernel and lost
    every variable. The model rewrote the rules as a set-based fast model,
    sampled random end layouts scored by its pour (60,000 trials, 69 winning
    layouts, t175), and planned leg-wise paths through role swaps, verified
    with `replica.step`.
  - The first route batch came at t195, after 24 plan turns and 4 nudges
    (t171-194). It stopped at step 42: the bar cannot come within one row of
    the bins, a limit level 1 never probed. The second route (t210, 9
    actions) solved the level as the replica predicted (55 wet cells).
  - The level used 339,976 output tokens, 55% of the game's; plan turns took
    62% of them.
- **Level 3 (71 actions; human 148).**
  - s51 (t210-217) drew a 20x20 board at scale 3 in 7 turns. Thirty plan
    turns followed before the first batch (t218-247, 5 nudges): layouts
    searched with a fast pour model (random seeds, then hill climbing) and
    checked against `replica.pour`. At t227 the two disagreed on the drain
    row (fast: nothing lost; engine: lost).
  - Four routes, each cut by one mismatch and re-planned from where the
    pieces were: t248 stopped at step 57 (the level's budget, HUD only; 6
    actions); t258 at 64 (the two 3x1 bars are one sprite; 7 actions);
    t272-277 at 84 (the budget is 120, not 128) and at 91 (the grey cap is
    the middle cell of that piece; 27 actions).
  - The last, a corrected 31-action route verified leg by leg on the
    replica, went out in four batches (t287, t297-299). Its final SPACE at
    step 122 solved the level as predicted (74 wet cells, every cavity
    filled). One more kernel timeout (t291), 8 plan nudges in the level.
- **Waste.** No RESET and no game over in the whole game. The extra actions
  are probes (steps 1-10, 31-36, 38) and pieces moved for routes that a
  mismatch then cut.

### ls20: turn limit, 2/7 in 231 actions

| level | steps | actions (human, base agent) | turns | solved at turn (min, output tokens so far) | fit rounds (turns) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1-13 | 13 (22, 24) | 0-40 | 40 (14.5, 58,346) | 5 (40) |
| 1 | 14-78 | 65 (123, 98) | 40-126 | 126 (53.8, 205,430) | 6 (75) |
| 2 | 79-231 | 153, unsolved (73, 151) | 126-300 | - (263,202 output tokens in the level) | 7 (101), plus s231 open |

- **Level 0 (13 actions; v10 21, base agent 24).**
  - s1 (t5-16): the 5x5 block moves one 5-px cell; the bar is a 42-column
    budget. The batch at t19 walked onto the white plus on purpose, "to test
    whether the plus is a target"; s6 (t19-27) found that it turns the legend
    90 degrees, "exactly the pattern still shown inside the top room". s7:
    the plus is not consumed.
  - The t36 batch note says "now that the legend pattern equals the room
    pattern"; its last move entered the room, which the replica had as a wall
    (s12, t36-39). The next UP (step 13) put the block inside the room and
    solved the level; the replica had the room walkable by then but
    predicted no solve.
  - s13 (t40-53) wrote the win as "the block wholly inside the room box",
    without the match it had noted, and drew level 1. The base agent's level
    0 (24 actions) found the match by reasoning: blocked at the box, the
    legend flashed, it took the plus, saw the legend change and came back.
- **Level 1 (65 actions; v10 218 unsolved, base agent 98, human 123).**
  - s14 (t60-64): two budget columns per press here. s25: a ring refills the
    bar.
  - Step 30, a press into the room, was predicted to solve on a win line one
    step had run (support 1); the game refused it and nothing changed. The
    verdict read "your replica raised an error (IndexError)", and for about
    10 turns (t72-81) the model drew a "level 2" board from the unchanged
    frame before it saw the frame was byte-identical to step 29's.
  - s30 (t71-92, 21 turns): the room opens only while the legend equals the
    room's pattern, first modelled as a mirror flip (wrong). s54 (t98-109): a
    90-degree rotation, so level 1 needs 3 plus landings. s58: the bar cost
    of a plus landing.
  - Routes came from a hand model of the rules (t85) and BFS over
    `replica.step` (t104-105, timed out; t117); the 43-press route was
    checked on the replica before it was sent (t91). After a RESET (step 31)
    the batch at t126 solved the level at step 78, as the replica predicted.
- **Level 2 (153 actions, unsolved; human 73, base agent 151).**
  - s78 (t126-148, 22 turns) drew the board: the legend is orange here, the
    room's pattern blue. s86 (t149-162): a conveyor lane, read from a
    17-frame animation.
  - Step 111 ended a 33-press route with the legend matching the room in
    shape and rotation, and the game refused the entry. The model read the
    refusal as a new board and invented boards 3 and 4 (s111 t168-181, s112
    t182-195). In that reasoning it noticed the colour and dismissed it:
    "the legend is ORANGE (12) and the room's pattern is BLUE (9)" (t185),
    "maybe the game compares the pattern shapes only" (t188).
  - s114: a press into a wall still costs. Step 115 lost a life on an empty
    bar. s115 (t205-246, 41 turns) re-measured every frame, deleted the
    invented boards, added lives, and explained the refusals by "at least 2
    columns left after the winning press".
  - A 57-press route (t253-260) reached the room matched in shape and
    rotation with 6 columns left, and was refused again (step 172). s172
    (t260-271) blamed the one object never touched, the multi-colour patch:
    "the winning cell requires that the block has covered the patch".
  - RESETs at steps 173 and 193 (the second after drifting out of reach of a
    refill). Covering the patch at step 210 turned the legend from orange to
    blue. s210 (t287-292) first tried a counter-clockwise turn, then read the
    pictures as a recolour, 12 to 9 ("same 3x3 shape"), and modelled it as a
    `patched` flag the win requires, not as a colour the win compares.
  - Step 231, a plus landing, mismatched: the replica redrew the legend
    orange, the game kept it blue. In t298-299 the model wrote that "the
    legend needs to match it in both shape and color"; the turn limit came
    at t300. The real rule is a key equal to the goal in shape, colour and
    rotation ([v10-play.md](v10-play.md), ls20).
- **Waste.** 3 RESETs (steps 31, 173, 193), 1 lost life (115), four wrong win
  predictions (30, 111, 112, 172), and two batches refused because they were
  sent during a fit round before the fix (t154, t193).

## Comparison with the base play agent

The base agent is `runs/20261004_135539`: the same model, the main
text-only harness, and 500K output tokens per game. Its output tokens are
the sum of `generated_tokens` over its `benchmark.json` history. v10-play.md
quotes the solver's notes, 501K and 502K.

| game | | score | levels | actions | per level | output tokens | prompt tokens | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp80 | base | 3.74 | 1/6 | 111 | 44 67 | 494K | 13.0M | $0.59 | 117 |
| sp80 | v10 | 4.01 | 2/6 | 239 | 125 95 19 | 627K | 48.3M | $1.967 | 177 |
| sp80 | **v11** | **47.6** | **4/6** | 122 | 11 19 21 71 | 613K | 51.4M | $1.965 | 166 |
| ls20 | base | **30.97** | **5/7** | 866 | 24 98 151 117 129 347 | 496K | 24.2M | $0.82 | 170 |
| ls20 | v10 | 3.57 | 1/7 | 239 | 21 218 | 675K | 48.0M | $1.904 | 183 |
| ls20 | **v11** | 10.7 | 2/7 | 231 | 13 65 153 | 469K | 52.6M | $1.937 | 136 |

Output tokens per level, base agent against v11:

| | sp80 L0 | sp80 L1 | sp80 L2-L3 | ls20 L0 | ls20 L1 | ls20 L2 |
| --- | --- | --- | --- | --- | --- | --- |
| base agent | 40K (13 calls), 44 actions | 454K, unsolved | not reached | 12K, 24 actions | 47K, 98 actions | 57K, 151 actions |
| v11 | 41K (25 turns, 10 min), 11 actions | 128K, 19 actions | 340K + 104K, 21 + 71 actions | 58K (40 turns, 15 min), 13 actions | 147K, 65 actions | 263K, 153 actions, unsolved |

- **sp80.** Four levels against one, in 122 actions against 111. Level 0
  cost the same output tokens in a quarter of the actions. At similar
  output tokens, v11 used 4 times the prompt tokens and 3.3 times the cost.
- **ls20.** Fewer actions on each level v11 solved, at 3-5 times the output
  tokens. Level 2 is the difference: the base agent solved it in 151 actions
  and 57K tokens, v11 spent 153 actions and 263K tokens on a rule that left
  out colour, and the base agent's 5/7 still scores 30.97 against 10.7.
- **Against v10**, on the same budget (300 turns, about $1.95 a game): sp80
  from 2/6 to 4/6 and from 239 to 122 actions; ls20 from 1/7 to 2/7, with
  level 1 in 65 actions against 218 unsolved.

## Where the tokens went

**Phases.** sp80's plan phase took 55.7% of its output tokens: levels 0-1
were fit-heavy (fit 60% and 68%), levels 2-3 plan-heavy (plan 62% and 71%,
the long searches). ls20's fit phase took 87.3% (84-89% per level); its fit
rounds averaged 12.0 turns, twice sp80's 6.1. The fit share of the cost was
37.0% on sp80 and 72.4% on ls20: cost follows prompt tokens.

**Level changes.** The next board was drawn by hand in the fit round on the
solving step. `recording[k].pieces_after.code()`, the segmentation
generator, was called once in the run (ls20 t10, level 0). sp80 wrote each
board in grid coordinates; ls20 extracted each maze floor with numpy into a
50-row literal and placed the other objects by hand.

| level change | drawing the new board: turns (output tokens) | from that commit to the first batch: turns |
| --- | --- | --- |
| sp80 0 -> 1 | t25-36: 11 (38,225; this round also wrote the pour model) | 12 |
| sp80 1 -> 2 | t78-87: 9 (8,444) | 4 |
| sp80 2 -> 3 | t210-217: 7 (4,490) | 31 |
| ls20 0 -> 1 | t40-53: 13 (26,221) | 7 |
| ls20 1 -> 2 | t126-148: 22 (29,327) | 1 |

**Errors.** Counted per turn: a call that errored (a traceback, an `[E_...]`
code, a kernel timeout) or a refused batch.

| | sp80 | ls20 |
| --- | --- | --- |
| turns with an errored call or a refused batch | 47 (18.2% of output tokens) | 35 (14.2%) |
| the same, or a mismatch | 68 of 300 (24.6%) | 50 of 300 (15.5%) |
| most frequent errors | `E_BAD_OP` 16, TypeError 9, NameError 7, E_NO_MATCH 3, IndexError 3, timeouts 3 | IndexError 7, NameError 5, TypeError 5, KeyError 3, ValueError 3, timeouts 2 |
| `edit_file` called as a tool (run as python by the shim) | 55 | 2 |

Most errors were in the model's own search code; 14 of sp80's 16
`E_BAD_OP`s were `edit_file` tool calls with `oldText`/`newText` and no `op`.

**Context.** Compaction ran from the first prompt over 140K (t35 sp80, t42
ls20) on every later turn, but it trims in place and drops nothing. After
turn 50 the prompt grew by about 760 (sp80) and 840 (ls20) tokens per turn,
to 289,211 and 296,762 at turn 300 (v10: 248,354 and 266,697); the cost per
50 turns went from $0.195 to $0.369 (sp80) and from $0.153 to $0.380 (ls20).
The cached share dipped to 77.7% and 80.8% in turns 151-200 and was 82-93%
otherwise. The cache did not stop at 130K-150K, as notes taken during the
run suggested: after turn 150 the median cached prompt was 207K (sp80) and
217K (ls20).

## How the model used the new parts

**The replica.** Counted over the python calls of each turn:

| | sp80 | ls20 |
| --- | --- | --- |
| turns with a python call | 192 | 242 |
| turns calling `replica.step` or `replica.pour` | 72 | 38 |
| turns calling `replica.make_level` | 28 | 15 |
| turns with a search (`deque`, `heapq`, beam, `random`) | 67 | 39 |
| of those, searching through `replica.step` / `replica.pour` | 16 | 9 |
| kernel timeouts (120 s, all variables lost) | 3 (t117, t128, t291) | 2 (t105, t248) |

The model stepped its replica far more than v10 did with `simulate`,
mostly to verify routes and check layouts. Searching through the replica is
slow (each node deep-copies a State): three of the five timeouts were such
searches (sp80 t117, ls20 t105 and t248). Most searches ran on fast
set-based copies of the rules, which can disagree with the engine (sp80
t227).

**Support.** Every batch's output carried its support lines, but neither
game ever called `traced()` or `support()`, so the model saw a plan's
weakest link only after sending it. The sent moves by the weakest support of
the lines their prediction ran (`support.weakest`), against v10's offline
check over three games:

| weakest line's support | sp80 moves (mismatched) | ls20 | v11 together | v10 offline |
| --- | --- | --- | --- | --- |
| 0 (untested) | 2 (2) | 4 (4) | 6 (6) | 13 (13) |
| 1-2 | 24 (9) | 25 (9) | 49 (18), 37% | 41 (12), 29% |
| 3 or more | 96 (10) | 199 (6) | 295 (16), 5.4% | 488 (23), 4.7% |
| no step code (RESET) | - | 3 (0) | 3 (0) | 13 (0) |

- **The weakest link located the mismatch.** In 17 of sp80's 21 mismatched
  batches and 18 of ls20's 19, the mismatching move was the weakest (or tied
  weakest) of those sent. By their weakest planned move, sp80's batches were
  3 at 0 (3 mismatched), 11 at 1-2 (10) and 12 at 3+ (8); ls20's 4 (4), 15
  (11) and 17 (4).
- **Win lines are always thin.** A level is won once, so every predicted
  solve rested on a line one or two earlier steps had run. The right
  predictions (sp80 steps 30, 51, 122; ls20 78) and the wrong ones (sp80 24;
  ls20 30, 111, 112, 172) all had support 1-2: "thin" on every one, with
  nothing to tell them apart.
- **Unseparated conditions.** 100 of sp80's 122 moves relied on one, mostly
  bounds checks; 17 of those mismatched, against 4 of 22. On ls20, 7 of 68
  against 12 of 160.

**`notes.md`.** sp80 wrote it at t15 and t43 (06:37 UTC, the file's last
change); its level-2 update at t121 failed with `E_BAD_OP` and was not
retried, so every later PLAN message showed level-0/1 notes ("a pour that
loses nothing wins", "2 px eaten per action"). ls20 updated it at t18, t33,
t93, t165 and t251.

**Prediction warnings.** Four batches carried one, all "moves change nothing
in your replica"; none predicted a game over.

## What worked, and what did not

- **Worked: planning on the replica.** Four of the six solved levels came
  from a route the replica verified and whose solve it predicted, and every
  solved level beat the human baseline. One-move probes found sp80's block
  destruction, role swap and erosion; the animation digest gave sp80's pour
  and ls20's conveyor. sp80 had no game over and no RESET (v10: 3 and 5).
- **Did not: a wrong rule that every step fitted**, as in v10: ls20 saw the
  colour twice and dismissed it, and support could not flag the win line.
- **Did not: long plan rounds and slow searches.** sp80 had 19 nudges while
  it rewrote crashing searches, the nudge never changed, and five searches
  hit the 120 s cell limit, which the prompt does not state.

## Caveats

- **One sample per game at temperature 0.7.** sp80's level 3 came on turn
  299; one turn later and the score would have been 28.6.
- **The turn limit bound both games.** More turns would have bought more
  levels on sp80, and on ls20 only once the colour rule was in.
- **Resumes.** The minutes count only the time the agent ran, nothing from
  the gaps between the 6 resumes per game. On each resume the kernel cells
  were replayed to rebuild the model's variables: cells that had failed
  failed again, and three sp80 cells (t117, t127, t128) hit the 20 s replay
  limit, so what they defined was missing after the resume.
- **The base agent is not a controlled comparison**: another harness and a
  500K output-token limit, which bound for both games.
- **No held-out evaluation** (`engine_re.evaluate`) and no `make score_run`
  were run on this run directory; the scores are `result.json`'s.

## Next steps

The detail is in [v11-followups.md](v11-followups.md) (item numbers in
brackets).

- **Seed `make_level(n + 1)` from the segmentation generator at every level
  change** (1), and **bound the context** by dropping old reasoning tails,
  arguments and harness messages instead of stubbing them (2).
- **Escalate the plan nudge** to a required short batch, and **plan from the
  end state** when the goal is a configuration (3, 16).
- **Make replica search cheap** (a cheaper State copy or a built-in move
  search), and state the 120 s cell limit and what a timeout lost (17, 18).
- **Fix the messages**: the predicted-solve verdict (4), a warning when the
  replica raises mid-batch (5), the `edit_file` tool shim (6), grid
  coordinates for unfamiliar pieces (7).
- **Prompt**: rank candidate routes with `traced()` before sending (13); a
  goal comparison may involve colour as well as shape and rotation (14);
  refresh `notes.md` at each new level (15).
- **Win lines need a signal other than support**, since a level is won once:
  ask the fit round of a wrong win prediction what differs between the
  winning and the refused attempt (v10's next step).
- **A second sample of both games**, and a per-level turn budget.
