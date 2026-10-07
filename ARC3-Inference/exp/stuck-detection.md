# Detecting when the base agent is stuck

## Question

gpt-6.1-sol spent 144 minutes and 437 actions on sk48 level 5, and the game lost to the time
limit three levels later ([base-gpt61sol-20games.md](base-gpt61sol-20games.md)). Can the
harness tell, from the game record alone, that an agent is stuck: repeated board positions,
repeated action sequences, resets, effort? Which criteria catch the long stuck stretches, how
early, and how often do they fire on normal play?

## Data

Five runs of the base agent, 40 game runs, 263 level attempts, 15,060 actions:

| run | model | games |
| --- | --- | --- |
| `runs/base-gpt61sol-20games`, `runs/base-gpt61sol-dfranzen` | gpt-6.1-sol | all 25 official games |
| `runs/base-max-default`, `runs/base-max-dfranzen` | qwen3.8-max-0902 | ft09 lp85 ls20 sp80 vc33 |
| `runs/20261004_135539` | qwen3.8-flash | ft09 lp85 ls20 sp80 vc33 |

**Labels.** Five readers each took a set of levels, read the transcripts and marked the
action ranges where the agent was stuck. Stuck means at least about 20 actions or 5 turns
with no progress: repeating failed probes or plans, cycling between hypotheses, burning
actions to force a reset, or the agent saying it is out of ideas. Exploration that keeps
finding things, and a long plan being carried out, are not stuck. The readers did not see
the detectors. They read 50 levels: every level that was slow, long or unsolved, and 18
control levels where an early signal fired. 13 of the 50 have stuck stretches. Stretches
of one level less than 15 actions apart are merged, which gives 17 stuck episodes. The
other 213 levels were not read and count as normal (most were solved under the human
baseline). Labels: `scripts/stuck_detection/labels.json`.

| episode | actions | minutes | what the agent did |
| --- | --- | --- | --- |
| gpt sk48 L5 | 193-511 (319) | 100 | Clicked no-op cells again and again, then `action(['UP','DOWN']*60)` to run out the budget, then went back to the same clicks. Its reachability proof left out cars pushing cars. Unlocked at 512 |
| gpt dc22 L5 | 200-275 (76) | 14 | Drove the crane back and forth by hand with triple clicks; ended when it wrote a BFS |
| gpt bp35 L8 | 336-359 (24) | 15 | Wrongly decided no switch existed, undid the climb, probed elsewhere |
| max ls20 L2 (default) | 280-1546 (2 episodes) | 210 | Flip-flopped on refills, killed itself many times to get a fresh bar; never solved |
| max vc33 L4 (default) | 113-626 (514) | 127 | Rebuilt the same gate states, budget deaths every ~51 actions |
| max sp80 L1 / L2 (default) | 50-160, 250-1101 | 10 / 204 | Re-probed SPACE in many poses; 25 game overs on L2 |
| max ls20 L2 / L4 (dfranzen) | 110-242, 676-1156 (3 episodes) | 18 / 119 | Same knocks rejected 8 times, deliberate budget deaths |
| max sp80 L2 (dfranzen) | 56-670 (615) | 199 | 21 game overs, then "unsolvable, stall with no-op clicks" |
| max lp85 L5 (dfranzen) | 96-150 (55) | 7 | "I give up on modeling; do a live random search" |
| flash ls20 L6 | 741-866 (126) | 31 | Filled the red box 3 times and lost it on each reset |
| flash sp80 L2 | 74-78, 103-111 | 30 / 21 | Thinking stalls: 43 and 38 minutes for 5 and 9 actions, over 400K reasoning characters |

gpt-6.1-sol was stuck once for long (sk48), and twice briefly. Its other long levels (lf52 L7
and L9, wa30 L8, dc22 L6, bp35 L7 and L9) were slow but progressing: 30-60 s turns with 1-4
actions each, ending in a search it wrote. qwen3.8-max was stuck for hours on ls20, sp80 and
vc33. qwen3.8-flash shows the third kind: a stall in thinking with almost no actions.

## Signals

All signals are computed online, from the actions so far on the current level
(`scripts/stuck_detection/`).

- **Board revisit.** The share of the last W actions whose board was already seen on this
  level. Exact matching misses most loops: step counters change the board on every move
  (sk48's budget bar loses a pixel every 3 moves, inside the play area). Rows and columns
  near the border that change on 60% or more of the steps are masked. Two boards then
  count as the same when they differ by at most τ pixels, τ = min(8, 0.3 × the game's
  median pixels changed per action).
- **Action repetition.** The share of the last 30 actions whose 6-gram of actions appeared
  earlier on the level. The same on run-length-collapsed actions (`RIGHT×6` → `RIGHT`), so
  that straight moves do not count.
- **Fails.** Game overs plus voluntary RESETs on the level. Returns to the level's start
  board also count lives lost in ls20, which raises no `game_over`.
- **No-effect actions.** The share of the last 30 actions that did not change the board.
- **Effort.** Actions, model turns, output tokens and minutes on the level: as absolute
  values, or relative to the agent's own median over the levels it already solved in this
  game. The relative form needs no human baseline and no per-model threshold.

## Results

Recall counts the 17 episodes in which the detector is on for at least one action. Delay
is the median number of actions from the episode's start to the first action with the
detector on; 0 means it was already on. A false onset is an alarm that starts outside
every labelled episode (or more than 10 actions before one). False levels counts the
levels without any stuck episode where the detector fired: the real false positives.

| criterion | recall | delay (actions) | false onsets | false levels |
| --- | --- | --- | --- | --- |
| board revisit ≥ 0.5 over 30 | 13/17 | 0 | 23 | 11 |
| board revisit ≥ 0.7 over 30 | 11/17 | 11 | 17 | 5 |
| **board revisit ≥ 0.8 over 60** | 11/17 | 25 | 8 | **1** |
| board revisit ≥ 0.8 over 100 | 11/17 | 59 | 2 | 0 |
| exact board revisit ≥ 0.5 over 30 | 11/17 | 0 | 15 | 4 |
| action 6-gram repeat ≥ 0.6 | 12/17 | 6.5 | 28 | 11 |
| collapsed action 4-gram repeat ≥ 0.5 | 12/17 | 11 | 14 | 6 |
| no-effect ≥ 0.4 over 30 | 1/17 | 105 | 0 | 0 |
| **fails ≥ 2 on the level** | 10/17 | 0 | 1 | **0** |
| back to start ≥ 3 | 11/17 | 0 | 9 | 6 |
| actions ≥ 3× own median | 11/17 | 2 | 18 | 15 |
| turns ≥ 4× own median | 12/17 | 35.5 | 6 | 5 |
| tokens ≥ 5× own median | 12/17 | 3 | 8 | 5 |
| **output tokens ≥ 100K on the level** | 11/17 | 0 | 1 | **0** |
| minutes ≥ 45 on the level | 11/17 | 4 | 7 | 6 |
| **P1: revisit ≥ 0.8 over 60 OR fails ≥ 2 OR tokens ≥ 100K** | **14/17** | 0 | 7 | **1** |
| P1 OR turns ≥ 4× own median | 16/17 | 0 | 13 | 6 |

- **Action sequences give more false alarms than board positions.** A planned route
  repeats moves (re86 L8, ls20 runs), so n-gram repetition fires on normal play. Repeated
  board positions are the better loop signal, but only over a window of 60 actions or
  more: a 30-action window fires on the deliberate budget burns that ls20 players use to
  reset after learning something.
- **Resets are almost free of false alarms.** Two game overs or resets on one level never
  fired on a normal level. A single death followed by a replan (bp35 L2, L3, L6) is
  normal, so the threshold is 2.
- **Effort alone is not enough.** gpt-6.1-sol's hard-but-progressing levels take 3-4 times
  its usual actions and turns and 45+ minutes, so turns, actions and minutes give 5 to 15
  false levels. Only output tokens on the level ≥ 100K separates them here. It is also
  the only signal for a thinking stall (flash sp80 L2).
- **P1 combines the three kinds of stuck: loops, resets, thinking.** It catches 14 of 17
  episodes, with one false level: flash ls20 L3, actions 254-273, the last 20 actions
  before that level was solved. Its other 6 false onsets fire early, in levels that do get
  stuck: max ls20 L2 at action 172 against a labelled start at 280, max ls20 L4 at 644
  against 676.

When P1 fires, and on what:

| episode | P1 on from | via | stuck minutes still ahead |
| --- | --- | --- | --- |
| gpt sk48 L5 | action 247, 36 min into the level | revisit 0.80 over 60 (the UP/DOWN burn) | 78 of 100 |
| max ls20 L2 (default) | 172, 10 min | revisit | 210 |
| max vc33 L4 (default) | 166, 32 min | 2 game overs | 115 of 127 |
| max sp80 L1 / L2 (default) | 51 / 246 | 2 game overs | 10 / 204 |
| max ls20 L2 / L4 (dfranzen) | 156 / 644 | revisit | 8 / 119 |
| max sp80 L2 (dfranzen) | 49, 33 min | 108K tokens | 199 |
| flash ls20 L6 | 652, 27 min | 103K tokens | 31 |
| flash sp80 L2 | 74, 27 min | 140K tokens | 51 |

On sk48, a 30-action window (revisit ≥ 0.7) fires 29 actions earlier, at action 218, 29
minutes into the level. It also fires on 5 normal levels, all ls20.

**Missed:** gpt dc22 L5 (76 actions), maxdf lp85 L5 (55) and gpt bp35 L8 (24). They come
back to old boards, but not often enough to fill 80% of a 60-action window. The count of
positions seen a third time (next section) catches all three.

### A position's third visit

A simpler loop test: crop the 2-pixel border of the board (where most step bars are),
hash it, and count each position's visits on the level. A position seen for the third
time is a repetition. The counts restart at each level, and nothing is masked inside the
board.

| criterion | recall | false onsets | false levels |
| --- | --- | --- | --- |
| 1 position at its 3rd visit | 17/17 | 57 | 48 |
| **3 positions at their 3rd visit** | 15/17 | 13 | **7** |
| 5 positions at their 3rd visit | 10/17 | 7 | 4 |
| 10 of the last 30 actions on a position seen 3+ times | 7/17 | 5 | 2 |
| the same with the counter-tolerant match (τ pixels) | 16/17, 12/17, 11/17 for 3, 5, last 30 | 23, 13, 9 | 15, 8, 3 |
| **P5: 3 positions at their 3rd visit OR fails ≥ 2 OR tokens ≥ 100K** | **17/17** | 14 | **7** |
| P6: the same with 5 positions | 13/17 | 8 | 4 |

- One third visit is not enough. Normal play crosses the same hub or corridor three
  times, or clicks a toggle back and forth: 48 normal levels.
- Three different positions at their third visit is the useful threshold. It catches 15 of
  17 episodes, among them all three that the 60-action window misses (dc22 L5 after 50
  actions, lp85 L5 after 43, bp35 L8 after 18). Its 7 false levels are cd82 L1, max vc33
  L3 and L5, flash vc33 L5 and L7, flash sp80 L1 and maxdf lp85 L6.
- Exact hashing after the crop does better than the counter-tolerant match: tolerance
  merges boards that differ in small, real ways, and adds false levels.
- sk48's budget bar is inside the board (row 53), but its no-op clicks leave the board
  unchanged between ticks, so exact repeats still happen.

When each one fires, relative to the labelled start of the stuck stretch (negative: before
it; "-": never on this level):

| episode | 2nd game over | 3 positions at 3rd visit | P1 | P5 |
| --- | --- | --- | --- | --- |
| gpt sk48 L5 | - | **-21 actions, -6 min** | +54, +22 min | -21, -6 min |
| gpt dc22 L5 | - | +50, +9 min | - | +50, +9 min |
| gpt bp35 L8 | - | +18, +8 min | - | +18, +8 min |
| max ls20 L2 (default) | -92, -7 min | -152, -10 min | -200, -13 min | -152, -10 min |
| max sp80 L1 (default) | +1, +1 min | -33, -8 min | +1, +1 min | -33, -8 min |
| max sp80 L2 (default) | -4, -1 min | -11, -1 min | -4, -1 min | -11, -1 min |
| max vc33 L4 (default) | +53, +12 min | -19, -8 min | +53, +12 min | -19, -8 min |
| maxdf lp85 L5 | - | +43, +5 min | - | +43, +5 min |
| maxdf ls20 L2 | - | +93, +10 min | +46, +9 min | +93, +10 min |
| maxdf ls20 L4 | +156, +30 min | +158, +31 min | -102, -17 min | +72, +16 min |
| maxdf sp80 L2 | 0 | -14, -14 min | -7, -3 min | -14, -14 min |
| flash sp80 L2 | +33, +66 min | +30, +59 min | 0 | 0 |
| flash ls20 L6 | - | - | -89, -20 min | -89, -20 min |

On the 7 episodes with a second game over, the third-visit count fires first on 6. It
fires 72 actions earlier on max vc33 L4, and 34 on max sp80 L1. On sk48 L5 it fires at
action 172, 75 actions before P1 and with about 100 stuck minutes still to come. That is
21 actions before the labelled start: the agent was already re-probing as it explored.

### In the harness: `ARC3_REPEAT_HINT`

`ARC3_REPEAT_HINT=1` turns the third-visit count into a message in the turn prompt. When
3 positions of the current level have each been reached 3 times (2-pixel border left out,
exact match, counted over the whole history of the level), the turn prompt gets a
step-back note: if stuck, list the facts known and tested, look at every visual element,
work backward from what must be true to pass the level, and prioritize the likeliest
strategy; if the repetition is part of the mechanics, ignore the note. After it is shown,
it waits 10 analyzer turns, even while the repetition continues
(`ARC3_REPEAT_HINT_POSITIONS`, `_VISITS`, `_COOLDOWN`). The count takes 0.06 s per turn on
a 1,500-action level.

Replayed over the recorded runs turn by turn (`replay_hint.py`), it would have been shown
110 times in 12 of the 40 games:

- 97 times on the 13 stuck levels. 82 of those fall inside a labelled stuck stretch or less
  than 10 actions before one.
- 13 times on 7 normal levels (cd82 L1, max vc33 L3 and L5, maxdf lp85 L6, flash vc33 L5 and
  L7, flash sp80 L1).
- On gpt-6.1-sol sk48 L5, the first message comes at turn 35 (action 176), 48 actions
  before the UP/DOWN budget burn, then every 10 turns until the level is solved: 11 in
  all. gpt-6.1-sol gets 7 more: 3 on bp35 L8 (from action 355), 3 on dc22 L5 (from 255)
  and 1 on cd82 L1.

## Caveats

- 17 episodes, and the thresholds were chosen on the same data. They show which signal
  separates which kind of stuck, not settled values.
- ls20 makes half the false onsets: burning the budget to reset is a strategy in that
  game.
- The token threshold is absolute. gpt-6.1-sol writes far fewer tokens than the qwen
  models (it reached 100K on one level, sk48 L5), so the same number means a much longer
  stall for it: on sk48 L5 it crosses 100K only at action 492, 19 actions before the unlock. The relative form (≥ 8× own
  median and ≥ 50K) has no false level, but catches only 5 episodes.
- The 213 unread levels count as normal. A stuck stretch in one of them would show up as
  a false alarm, not a miss.
- No agent issued a RESET itself in these runs: RESET was not offered (`EXPOSE_RESET` is
  off in `params.yaml`, as in the final dfranzen notebook). All 76 RESETs are the automatic
  ones after a game over, so "fails" here counts game overs only. To restart a level, the
  agents ran out the step budget on purpose (sk48 L5: about 130 UP/DOWN actions).
  `ARC3_NO_BUDGET_BURN=1` adds a system-prompt line against this; it has not been run yet.
- An alarm already on when an episode starts counts as a delay of 0. The false-onset
  column shows when that alarm started outside the episode.

## Conclusions

- Three cheap criteria, read from the game record and the token count, cover the three
  kinds of stuck: board positions revisited over a long window (loops), repeated deaths
  or resets on one level, and a large token spend on one level (thinking stalls). Together
  they catch 14 of the 17 labelled episodes with one false level out of 250. On sk48 L5
  they fire with 78 of the 100 stuck minutes still to come.
- Action-sequence repetition and plain effort (actions, turns, minutes) fire too often on
  normal play to use alone, mostly on gpt-6.1-sol's long but productive levels.
- Counting positions seen three times (2-pixel border cropped, exact hash) is a better
  loop signal than the 60-action window, and earlier than the second game over. With 3
  such positions on a level, P5 catches all 17 episodes, with 7 false levels out of 250
  instead of 1. Use P1 when false alarms are costly (an intervention that interrupts the
  agent). Use P5 when they are cheap (a note in the prompt, a log line).
- The second game over is a reliable signal, but late. On 6 of the 7 episodes with one, the
  third-visit count fires first.
- Some stuck play is still invisible on the board: flash ls20 L6 has neither repeats nor
  game overs, and only the token count sees it.

## Reproduce

```bash
for r in base-gpt61sol-20games base-gpt61sol-dfranzen base-max-default base-max-dfranzen 20261004_135539; do
  dvc pull runs/$r.dvc && uv run --no-sync python scripts/pack_run.py unpack runs/$r
done
scripts/stuck_detection/run.sh /tmp/stuck   # about 2 minutes; prints the full table
cd /tmp/stuck && python3 <repo>/ARC3-Inference/scripts/stuck_detection/timing.py P1   # per-episode timing
python3 <repo>/ARC3-Inference/scripts/stuck_detection/lead.py   # onsets against the 2nd game over
```

`matrix.py <name substrings>` prints which episodes each detector catches, and its false
onsets.
