# Can qwen3.8-flash rebuild an ARC-AGI-3 game engine from a run?

An agent built on `qwen/qwen3.8-flash` (OpenRouter) was given every action and
observation of a recorded run and asked to write an engine module, with the
same interface as the real games, that reproduces the recording exactly. The
harness is described in [README.md](README.md). Raw results, the generated
engines and gzipped transcripts are in [results/](results/).

## Summary

- **3 of 5 games pass every recorded step.** In the main configuration, the
  ft09, sp80 and lp85 engines reproduce the whole recording: every frame of
  every action, pixel for pixel, plus state, levels completed and available
  actions. ls20 (best 17 of 867 steps) and vc33 (43 of 332) ran out of their
  115-minute budget.
- **Tokens for a passing game:** 140K-280K output tokens (81-93% of them
  reasoning), 6.9M-15.1M prompt tokens (90-95% served from cache),
  $0.22-$0.47 and 36-78 minutes. The whole five-game run used 1.43M output
  tokens and 98.5M prompt tokens (92.2M cached), and cost $3.09.
- **Passing the recording does not make the engine correct.** On new random
  play from each recorded level, the passing engines differ widely:
  - **ft09:** a real re-implementation. Levels 1-5 match 100%. Its only
    failures are two level-0 behaviours the recording never showed.
  - **sp80:** the physics are right, but two hidden rules were replaced by a
    heuristic fitted to five recorded events. 91% held-out.
  - **lp85:** a lookup table from exact recorded click pixels to fitted colour
    permutations, with no model of rings or buttons. 54% held-out, and 24% on
    the steps that change the screen.
- **The harness mattered as much as the model.**
  - With prompting alone, the model analysed the trace for 50-60 minutes per
    game and never ran a test.
  - Three changes got it to finish games: sending its reasoning back each
    turn, testing `engine.py` automatically whenever it changes, and a
    reminder after 30 turns without a test.
  - The model called `run_tests` itself in only 2 of 69 tests.
- **A simpler engine interface (v4) did not fix how the agent works.** With
  `make_level`/`step` on fixed sprite classes, and only each action's final
  frame compared:
  - **Passes:** ft09 passed in 22 minutes for $0.14, the fastest and cheapest
    pass of any configuration, and sp80 passed too.
  - **Failures:** lp85 and ls20 analysed for 91-105 minutes before their first
    engine, and vc33 quit at 46 minutes over a one-pixel timer-bar error. See
    [v4](#v4-the-make_levelstep-interface-resultsv4-simple).
- **Neither did the v5 tools and prompts.** Four trials on lp85 added
  in-python `read`/`edit`/`undo`, `auto_sprites`, image failure reports and a
  worked example of the test-edit loop. In 126 turns the agent ran one test
  and changed `engine.py` zero times. In trial 4 it quoted the instruction to
  test early and put it off. See [v5](#v5-trials-on-lp85-50-turns).

## Setup

**Data.** The source is run `20261004_135539`, archived in DVC: qwen3.8-flash
playing the 5 public games with the Kaggle submission config at 500K output
tokens per game. Its logged actions were replayed through the real engines.
The replay reproduced every logged board exactly and also recovered every
animation frame.

| game | steps | levels completed in the run | frames | game overs |
| --- | --- | --- | --- | --- |
| ls20 | 867 | 5 of 7 | 1689 | 1 |
| ft09 | 101 | 6 of 6 (won) | 106 | 0 |
| vc33 | 332 | 7 of 7 (won) | 1101 | 0 |
| sp80 | 112 | 1 of 6 | 456 | 3 |
| lp85 | 120 | 8 of 8 (won) | 127 | 0 |

**Agent.**
- Model: `qwen/qwen3.8-flash`, served by Alibaba via OpenRouter, with
  temperature 0.7, top_p 0.95, reasoning on and up to 32K tokens per response.
- Tools: a sandboxed Python kernel holding the trace and analysis helpers;
  view, write and edit for `engine.py`; `run_tests`; and `finish`.
- Budget per game: 115 minutes of wall time, 300 turns, 1.5M output tokens and
  $6. Wall time was the limit that bound.
- One session per game, all five in parallel, one sample each.

**Pass criterion.** A fresh instance of the engine replays all recorded actions
from step 0 and returns, at every step, the same frames (count and pixels),
state, levels completed, win levels and available actions.

## Main results (`results/v2-main`)

| game | status | recorded steps exact | turns | minutes | output tokens | reasoning tokens | prompt tokens | cached | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | **passed** | 101/101 | 87 | 36 | 139,871 | 117,946 | 6,888,680 | 95% | $0.22 |
| sp80 | **passed** | 112/112 | 94 | 63 | 262,264 | 244,908 | 8,459,651 | 91% | $0.37 |
| lp85 | **passed** | 120/120 | 151 | 78 | 279,880 | 250,359 | 15,076,725 | 95% | $0.47 |
| vc33 | time limit | 43/332 (levels 0-2) | 266 | 115 | 362,268 | 309,130 | 33,666,393 | 92% | $1.07 |
| ls20 | time limit | 11/867 (best 17) | 276 | 115 | 384,256 | 310,573 | 34,397,345 | 95% | $0.97 |
| **total** | 3/5 | | 874 | 408 | 1,428,539 | 1,232,916 | 98,488,794 | 94% | $3.09 |

Sessions stop at the first full pass, so for the passing games these figures
are the cost of getting there.

**Generalisation** (`results/v2-main/evaluation_best.md`). For each level
the recording reached, both engines replay the recording up to that level's
first step. They then play the same 8 random sequences of 40 actions (keys from
the advertised actions, clicks aimed mostly at objects, occasional RESETs), and
the steps are compared exactly. With the real engine in place of the generated
one, every game scores 100%.

| game | held-out steps exact | of those that change the screen | per level |
| --- | --- | --- | --- |
| ft09 | 85% | 78% | L0 7%; L1-L5 100% |
| sp80 | 91% | 91% | L0 99%; L1 82% |
| lp85 | 54% | 24% | 14%-88% by level |
| vc33 | 42% | 51% | L0-L1 100%, L2 98%; L3+ not modelled |
| ls20 | 3% | 3% | diverges at step 17 |

For vc33 and ls20, levels past the first recorded mismatch start from an
already-diverged state.

## How the generated engines compare with the real ones

Each analysis was checked by running both engines side by side; details and
line references are in [results/analysis/](results/analysis/). The real
mechanics are summarised in
[results/real_engines_brief.md](results/real_engines_brief.md).

- **ft09: correct rules** ([analysis](results/analysis/ft09.md)).
  - **Structure:** a data-driven model of boards, clickable tiles and hint
    cells, with level data exactly matching the real sprites.
  - **Rules recovered correctly:**
    - the hint semantics (0 = same colour, 2 = different, 3 = no tile);
    - the 6-pixel neighbour toggles;
    - the 3-colour cycle on level 3;
    - the budget, including a free winning click and GAME_OVER when the budget
      runs out, which never happened in the recording;
    - the HUD.
  - **What fails:**
    - a 5-frame blink on level 0 when clicking empty space, never shown in the
      recording;
    - three example pictures on level 0 treated as playable boards.
  - **How close it is:** a 20-line fix makes it match 100% of held-out play.
- **sp80: right physics, fitted heuristics** ([analysis](results/analysis/sp80.md)).
  - **What is right:**
    - the spill simulation, frame for frame (all 108 platform placements on
      level 0);
    - the cup and floor rules, the blink timing and the budget;
    - level 1, which the real engine shows rotated 180 degrees.
  - **What is fitted:** two hidden rules the recording did exhibit were missed
    and replaced by a geometric heuristic fitted to five recorded events. The
    missed rules are the 5th failed spill being fatal, and re-selecting the
    piece nearest the origin after a failed spill.
  - **Levels 2-5:** invented.
- **lp85: a lookup table that passes** ([analysis](results/analysis/lp85.md)).
  - **How it works:** each recorded click pixel maps to a colour permutation
    fitted to that click's recorded effect. There is no model of rings or
    buttons, and a click one pixel off a recorded one does nothing.
  - **Where the held-out gap comes from:** about 70-79% from rules it got wrong
    (an exact-pixel hit test, no ring cycle) and 21-30% from buttons the
    recording never clicked.
  - **Why it still passes:** it reproduces all 120 steps because the table was
    built from them.
- **vc33: per-level special cases, levels 0-2 only** ([analysis](results/analysis/vc33.md)).
  - **What is right:** the pumping between containers, the budget, the HUD and
    the win check, which match random play on levels 0-2.
  - **What is missing even there:** the overflow limit and GAME_OVER, which
    random play rarely reaches.
  - **Levels 3-6:** placeholders. Gates and the 43-frame swap animation were
    described in its notes but never coded.
- **ls20: far off** ([analysis](results/analysis/ls20.md); interactive trace viewer: [`results/ls20_trace/ls20_trace.html`](results/ls20_trace/ls20_trace.html), built by `results/ls20_trace/build.py`).
  - **What is right:** the maze layout and plain moves.
  - **What is wrong:**
    - a key rotation that is actually a reflection;
    - a 6-frame flash that ends after 1 frame;
    - goal tiles that cannot be entered;
    - pushers (47 steps of 17 frames each), track-movers and the colour and
      shape changers, which are missing entirely.
  - **Why it was hardest:** it is the longest trace (867 steps), with the most
    mechanics and four frame-count regimes (1, 2, 6, 17).

The guard's source scan flagged nothing in any engine: no frame inspection, no
file or network access. No engine reads the trace. lp85's overfitting is
built into its data tables, not into reads of the recording.

## How the agent worked

- **Long analysis, then one big write.** The first engine came at turn 82
  (ft09), 83 (sp80), 139 (lp85), 98 (ls20) and 151 (vc33), almost always
  written from Python with generated level data. The passing games then
  converged quickly:
  - ft09 went from 0/101 to 101/101 in 5 turns;
  - sp80 from 62/112 to 112/112 in 11;
  - lp85 from 0/120 to 112/120 in 5 turns, then 120/120 in 7 more.
- **It does not test unprompted.** 67 of the 69 tests were the harness's
  automatic tests after `engine.py` changed, and the model ignored most of the
  reminders.
- **Context.** Prompts reached 145K-203K tokens. Caching kept the cost low:
  at most $0.0065 per turn at a 190K-token prompt. Only vc33 kept a
  `notes.md`.

## Harness iterations

The feedback mechanisms were added one at a time, after pilot sessions on the
same five games (all in `results/`):

| configuration | what changed | result |
| --- | --- | --- |
| [pilot A](results/pilot-a-no-feedback) | prompt only; reasoning not sent back; traces from a shorter new run | no test and no engine in 50 min per game (1.0M output tokens, $0.79) |
| [pilot B](results/pilot-b-nudges) | + a reminder after 30 turns without a test (archived traces from here on) | no test in about 59 min per game ($1.02); interrupted once and resumed |
| [v2 main](results/v2-main) | + reasoning sent back each turn, an automatic test when `engine.py` changes, `notes.md` | **3/5 pass**, $3.09 |
| [v3](results/v3-python-quota) | v2 + Python pauses after 30 calls without an engine change | 2/5 pass (ft09, sp80), $3.43 |
| [v4](results/v4-simple) | v2's feedback, but a new interface: `make_level`/`step` on fixed sprite classes, the harness draws and runs the levels, only each action's final frame is compared, 5 contract tests | 2/5 pass (ft09, sp80), $2.77 |

- **Reasoning.** Sending the reasoning back mattered. The provider reads it
  (358 vs 2,159 prompt tokens with a 1.8K-token reasoning block), and without
  it the model loses its own conclusions every turn: its visible content is
  usually empty.
- **The v3 quota backfired.**
  - The model wrote placeholder engines just to unlock Python.
  - On ls20 it kept level data in JSON files that the sandboxed engine cannot
    read.
  - Its engines generalise worse: vc33 matches 34-43% of held-out play on
    levels 0-2, against 98-100% for v2, and ft09 has a latent level-3 bug.
- **A compaction bug, now fixed.** Compaction used to replace old long
  tool-call arguments with an `elided` key. The ls20 agent copied that shape
  and some of its calls failed. The fix keeps each tool's own keys; it came
  after the runs above.

## v4: the make_level/step interface (`results/v4-simple`)

**What changed from v2.**
- **The engine:** `engine.py` is no longer a subclass of the real games'
  base class. It starts with a fixed block: `Sprite`, `Action`, `View` and
  `State` classes with layers, visibility, collision modes, transforms,
  `try_move` and `sprite_at`, plus the drawing rules. Below it the agent
  writes two functions:
  - `make_level(n)`, the state at the start of a level;
  - `step(state, action)`, which applies one action.
- **The harness** draws the state, runs the levels, RESET, WIN and GAME_OVER,
  and turns clicks into grid cells.
- **Pass criterion:** every action's final frame and the game state must match.
  Animation frames are no longer compared. 5 contract tests run before the
  replay.
- **Unchanged:** the model, the recording, the budgets and v2's feedback
  (reasoning sent back, automatic tests, a reminder every 30 turns). The prompt
  was rewritten for the new interface.
- **The interface can express all five games.** Before the run, five reference
  engines written in it by Claude subagents (not committed) reproduced every
  recorded step and 100% of held-out play.

| game | status | recorded steps exact | held-out (v2) | first tested engine: turn, minute (v2) | turns | minutes | output tokens | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | **passed** | 101/101 | 86% (85%) | 62, 21 (82, 36) | 64 | 22 | 87,930 | $0.14 |
| sp80 | **passed** | 112/112 | 74% (91%) | 69, 51 (83, 60) | 114 | 85 | 352,426 | $0.47 |
| vc33 | gave up | 5/332 | 5% (42%) | 79, 29 (151, 70) | 123 | 46 | 183,905 | $0.33 |
| lp85 | time limit | 0/120, crashes on load | 0% (54%) | 167, 105 (139, 75) | 194 | 115 | 470,502 | $0.82 |
| ls20 | time limit | 0/867, crashes on load | 0% (3%) | 182, 91 (98, 41) | 229 | 115 | 459,012 | $1.01 |
| **total** | 2/5 | | | | 724 | 383 | 1,553,775 | $2.77 |

<!-- v4-tokens:start -->
**Tokens (v4).** Every request of every session, as OpenRouter reported it:

| game | requests | prompt tokens | of which cached | output tokens | of which reasoning | largest prompt | cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | 64 | 4,828,376 | 4,631,296 (96%) | 87,930 | 73,051 (83%) | 140,329 | $0.14 |
| sp80 | 114 | 11,125,007 | 10,200,576 (92%) | 352,426 | 316,464 (90%) | 153,193 | $0.47 |
| vc33 | 123 | 11,492,826 | 11,014,656 (96%) | 183,905 | 160,215 (87%) | 142,785 | $0.33 |
| lp85 | 194 | 22,963,208 | 21,251,840 (93%) | 470,502 | 402,185 (85%) | 160,579 | $0.82 |
| ls20 | 229 | 28,844,385 | 26,392,320 (91%) | 459,012 | 401,988 (88%) | 190,449 | $1.01 |
| **total** | 724 | 79,253,802 | 73,490,688 (93%) | 1,553,775 | 1,353,903 (87%) | | $2.77 |

Prompt tokens are the whole conversation re-sent on each request, so they grow
with the turn count; 93% of them were served from the provider's cache.
<!-- v4-tokens:end -->

**What happened.** The transcripts can be read turn by turn in
[`results/v4_transcripts/v4-agent-transcripts.html`](results/v4_transcripts/v4-agent-transcripts.html),
annotated for ls20, lp85 and vc33.
- **ft09 and sp80 passed.**
  - ft09 passed two turns after its first test: two crashes, then 101/101.
  - sp80 went from 60/112 at turn 70 to 112/112 at turn 114.
- **vc33 quit with 69 minutes left, over one pixel.**
  - **What it got right:** levels 0 and 1 played correctly.
  - **What failed:** the timer bar on row 0 was one pixel off on almost every
    step. The real bar is `round(64 × remaining / budget)`, with budgets of 50,
    75 and 200 clicks.
  - **How close it was:** the agent measured the drain rates exactly (1.28,
    0.85 and 0.32 px per click), but tried only `floor`. It concluded the bar
    was a wall clock.
  - **Why it stopped:** at turn 109 it reasoned "even if I implement
    everything, the bar blocks matching. So extra work yields ~0". It then
    called `finish` twice, saying its context was "nearly exhausted (~5k
    left)".
- **lp85 and ls20 analysed until the time limit.**
  - **Python only:** they used nothing but python until turns 164 and 182, and
    ignored 5 and 6 reminders.
  - **One big generator, never finished:** both planned a single script that
    would write every level's data and logic at once.
    - lp85's only test crashed: `make_level` reads a `LEVELS` table that was
      never written.
    - ls20's first engine imports a module that does not exist. Its last one
      runs but matches 0/867, and it had edited the fixed block.
  - **ls20 had the answer:** by turn 177 it had essentially found the
    key-shape rules (a 90-degree turn, and a fixed cycle of 6 shapes). It kept
    doubting them after an arithmetic slip in its reasoning.
- **They think their context is running out.**
  - ls20 says so at turns 151, 160 and 220; vc33 at turns 74, 115 and 123.
  - In fact the harness compacts old tool outputs once a prompt passes 140K
    tokens, so their context never ran out. Their prompts peaked at 143K-190K
    tokens.
  - The belief makes them put off writing ("let me write it all in one go") or
    stop.
- **26 tests in total (v2: 69).** All but one were the harness's automatic
  tests.

**Held-out play.** Each cause below was confirmed by patching a copy of the
candidate engine and replaying `evaluate.py`'s rollouts. Details are in
[results/analysis/v4-ft09.md](results/analysis/v4-ft09.md) and
[results/analysis/v4-sp80.md](results/analysis/v4-sp80.md).
- **ft09 (86%): one cause, the same as in v2.** The three example pictures on
  level 0 are clickable tiles in the candidate.
  - The recording never clicked them; at turn 38 the agent decided, without
    evidence, that they count.
  - Making them unclickable gives 1920/1920.
- **sp80 (74%): three rules differ.** With all three fixed: 644/644, and the
  recording still passes.
  - **Side walls (28% of the gap, new in v4):** invisible walls stop boards
    one column short of the screen edge. The agent misread two blocked "up"
    moves as blocked "left" moves.
  - **Failed-spill counter (2%):** the 5th failed spill ends the game. The
    recording showed this twice; the agent proposed the rule, then dropped it
    because it misread a GAME_OVER as a no-op.
  - **Re-selection (70%):** after a failed spill the agent re-selects the
    highest bar the paint touched. The real game takes the platform nearest the
    origin. The recording cannot tell these rules apart.
- **The same kind of failure as v2:** a rule fitted to the recorded events
  instead of the general one.

**What v4 shows.**
- **The interface is not the obstacle.** It expresses every game, and ft09
  passed faster than ever.
- **The failures come from how the agent works:**
  - no early tests, so no feedback;
  - a one-pixel display error that fails every step, so no reward for fixing
    the rest;
  - a `finish` tool that lets the agent leave.
- **Next:** the v5 harness, tried on lp85 in the [next section](#v5-trials-on-lp85-50-turns).

<!-- v5-trials:start -->
## v5 trials on lp85, 50 turns

**What v5 changes against v4.** The interface, the model and the budgets stay;
how the agent works changes.
- **Tools:** `python`, `run_tests(level, failures)` and `finish`. `finish` runs
  the tests and ends the session only when they all pass.
- **Inside python:** the recording `S`, plus `read` and `edit` (line anchors, as
  in pi's hashline edit), `undo` (every change is a numbered version), `render`,
  `show` (frames as images), `try_step` (the state before and after one
  recorded step) and `auto_sprites` (sprite code that redraws a level's first
  frame exactly). Python cannot write `engine.py` directly.
- **Failure report:** the recording is replayed in order and the report stops
  at the first failing step. It shows both frames as images with the differing
  regions boxed, the agent's sprites in each region, what `step()` printed and
  the `try_step` command that reproduces the step.
- **Prompt:** goal, setup, drawing rules, tests and how to work: reproduce what
  was observed with the simplest general rule, start from `auto_sprites(0)`,
  pass the steps in order, and reuse sprite kinds across levels. Details are in
  [README.md](README.md).

**Four trials.** lp85 only, the game where v4 ran 164 turns without writing to
`engine.py`. Same model and limits except 50 turns, one session each, images
on. Each trial changed the prompt after the one before. Their results and
gzipped transcripts (full reasoning and tool outputs) are in
[results/v5-trials/](results/v5-trials/).


| trial | commit | what it adds | how it ended |
| --- | --- | --- | --- |
| 1 | `fc0f08c` | the v5 harness above | stopped by hand at turn 34 |
| 2 | `7db527b` | a prompt section naming the built-in functions; a kernel that refuses code which redefines one; a first message ending with the first move (`auto_sprites(0)` into `make_level`, then `run_tests`); a retry for answers the provider ends with `finish_reason: error` | stopped by hand at turn 38 |
| 3 | `d10e41b` | a worked example in the system prompt: the first turns of a session on a made-up game (`run_tests`, `auto_sprites(0)`, `edit()` into `make_level`, `try_step` on the first failure, a fix in `step()`, the next failure) | stopped by hand at turn 4, to change the example's order |
| 4 | `a655c2a` | the same example, with every round as `read()`, `run_tests`, `edit()`, `run_tests` | reached the 50-turn limit |

**All tokens.**

| session | status | turns | minutes | requests | prompt tokens | of which cached | output tokens | of which reasoning | largest prompt | tests | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v5 trial 1 | stopped by hand | 34 | 13 | 34 | 1,925,585 | 1,761,536 (91%) | 63,258 | 57,604 (91%) | 114,130 | 1 | $0.076 |
| v5 trial 2 | stopped by hand | 38 | 29 | 38 | 3,049,459 | 2,697,728 (88%) | 136,407 | 129,532 (95%) | 141,188 | 0 | $0.160 |
| v5 trial 3 | stopped by hand | 4 | 2 | 4 | 52,673 | 26,368 (50%) | 8,529 | 7,890 (93%) | 17,953 | 0 | $0.008 |
| v5 trial 4 | turn limit | 50 | 29 | 50 | 4,010,189 | 3,761,664 (94%) | 120,086 | 108,812 (91%) | 140,893 | 0 | $0.154 |

The same sessions at equal turn counts, against v2 and v4 on lp85:

| session | after turn | minutes | prompt tokens | of which cached | output tokens | of which reasoning | cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | 10 | 4 | 225,070 | 81% | 14,275 | 13,255 | $0.016 |
| v2 | 20 | 6 | 760,503 | 92% | 20,767 | 18,425 | $0.030 |
| v2 | 30 | 12 | 1,502,180 | 94% | 42,529 | 38,703 | $0.056 |
| v2 | 40 | 17 | 2,610,819 | 95% | 65,111 | 59,892 | $0.090 |
| v2 | 50 | 25 | 3,604,741 | 93% | 97,799 | 90,403 | $0.135 |
| v4 | 10 | 2 | 192,994 | 80% | 9,200 | 8,387 | $0.013 |
| v4 | 20 | 10 | 846,064 | 89% | 40,915 | 38,139 | $0.045 |
| v4 | 30 | 14 | 1,882,238 | 93% | 59,719 | 55,207 | $0.075 |
| v4 | 40 | 22 | 2,907,624 | 92% | 94,672 | 86,599 | $0.124 |
| v4 | 50 | 31 | 4,139,997 | 93% | 129,388 | 118,186 | $0.165 |
| v5 trial 1 | 10 | 3 | 187,125 | 78% | 10,058 | 8,687 | $0.013 |
| v5 trial 1 | 20 | 7 | 716,606 | 88% | 37,596 | 34,491 | $0.041 |
| v5 trial 1 | 30 | 11 | 1,479,586 | 89% | 54,763 | 50,024 | $0.064 |
| v5 trial 1 | 34 | 13 | 1,925,585 | 91% | 63,258 | 57,604 | $0.076 |
| v5 trial 2 | 10 | 4 | 269,346 | 83% | 17,558 | 15,555 | $0.019 |
| v5 trial 2 | 20 | 15 | 1,060,308 | 89% | 66,398 | 62,739 | $0.064 |
| v5 trial 2 | 30 | 24 | 2,324,375 | 88% | 117,368 | 111,866 | $0.129 |
| v5 trial 2 | 38 | 29 | 3,049,459 | 88% | 136,407 | 129,532 | $0.160 |
| v5 trial 3 | 4 | 2 | 52,673 | 50% | 8,529 | 7,890 | $0.008 |
| v5 trial 4 | 10 | 4 | 294,379 | 82% | 16,773 | 15,123 | $0.020 |
| v5 trial 4 | 20 | 10 | 1,005,662 | 91% | 40,737 | 37,262 | $0.047 |
| v5 trial 4 | 30 | 15 | 2,116,700 | 94% | 60,884 | 55,332 | $0.079 |
| v5 trial 4 | 40 | 21 | 3,051,232 | 93% | 84,446 | 75,449 | $0.116 |
| v5 trial 4 | 50 | 29 | 4,010,189 | 94% | 120,086 | 108,812 | $0.154 |

<details><summary>Every turn of trial 1</summary>

| turn | minute | prompt tokens | cached | output tokens | reasoning |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 9,340 | 0 | 137 | 27 |
| 2 | 0.1 | 10,542 | 0 | 115 | 24 |
| 3 | 0.2 | 10,870 | 10,496 | 105 | 39 |
| 4 | 0.3 | 13,139 | 10,752 | 172 | 52 |
| 5 | 0.9 | 17,428 | 13,056 | 3,314 | 3,176 |
| 6 | 1.3 | 21,312 | 17,408 | 268 | 216 |
| 7 | 1.7 | 22,353 | 21,248 | 2,256 | 2,019 |
| 8 | 1.9 | 24,916 | 22,272 | 805 | 678 |
| 9 | 2.1 | 26,711 | 24,832 | 902 | 746 |
| 10 | 2.5 | 30,514 | 26,624 | 1,984 | 1,710 |
| 11 | 3.4 | 33,886 | 30,464 | 5,258 | 4,982 |
| 12 | 3.7 | 39,751 | 33,792 | 1,445 | 1,299 |
| 13 | 4.1 | 48,999 | 39,680 | 1,902 | 1,568 |
| 14 | 4.4 | 51,689 | 48,896 | 1,729 | 1,679 |
| 15 | 4.5 | 53,468 | 51,456 | 61 | 20 |
| 16 | 4.5 | 53,592 | 53,248 | 194 | 101 |
| 17 | 4.6 | 54,046 | 53,504 | 205 | 40 |
| 18 | 5.4 | 55,301 | 54,016 | 4,955 | 4,690 |
| 19 | 7.1 | 63,869 | 55,296 | 10,673 | 10,435 |
| 20 | 7.4 | 74,880 | 63,744 | 1,116 | 990 |
| 21 | 7.8 | 76,552 | 74,752 | 1,807 | 1,702 |
| 22 | 7.9 | 20,593 | 0 | 174 | 174 |
| 23 | 8.0 | 80,856 | 80,384 | 178 | 23 |
| 24 | 8.1 | 20,834 | 0 | 171 | 171 |
| 25 | 8.9 | 81,588 | 81,152 | 3,987 | 3,841 |
| 26 | 9.2 | 86,450 | 81,408 | 1,423 | 1,077 |
| 27 | 9.8 | 89,271 | 86,272 | 3,295 | 3,059 |
| 28 | 9.9 | 99,438 | 89,088 | 467 | 320 |
| 29 | 10.9 | 100,817 | 99,328 | 5,403 | 5,126 |
| 30 | 11.0 | 106,581 | 100,608 | 262 | 40 |
| 31 | 11.3 | 108,401 | 106,496 | 1,655 | 1,507 |
| 32 | 11.4 | 110,841 | 108,288 | 589 | 280 |
| 33 | 11.7 | 112,627 | 110,592 | 1,245 | 940 |
| 34 | 12.5 | 114,130 | 112,384 | 5,006 | 4,853 |
| **34 turns** | 12.5 | 1,925,585 | 1,761,536 | 63,258 | 57,604 |

</details>

<details><summary>Every turn of trial 2</summary>

| turn | minute | prompt tokens | cached | output tokens | reasoning |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 9,666 | 256 | 129 | 8 |
| 2 | 0.1 | 13,449 | 9,472 | 211 | 111 |
| 3 | 0.7 | 17,700 | 13,312 | 1,964 | 1,711 |
| 4 | 0.7 | 19,745 | 17,664 | 266 | 13 |
| 5 | 0.8 | 20,502 | 19,712 | 190 | 105 |
| 6 | 1.7 | 28,319 | 20,480 | 3,983 | 3,885 |
| 7 | 2.4 | 33,632 | 28,160 | 3,011 | 2,698 |
| 8 | 3.3 | 37,420 | 33,536 | 3,821 | 3,626 |
| 9 | 3.8 | 42,702 | 37,376 | 2,018 | 1,741 |
| 10 | 4.2 | 46,211 | 42,496 | 1,965 | 1,657 |
| 11 | 4.5 | 48,501 | 46,080 | 900 | 710 |
| 12 | 4.8 | 50,248 | 48,384 | 1,472 | 1,363 |
| 13 | 5.4 | 57,628 | 50,176 | 3,015 | 2,916 |
| 14 | 6.6 | 68,455 | 57,600 | 5,721 | 5,492 |
| 15 | 7.7 | 75,664 | 68,352 | 4,624 | 4,491 |
| 16 | 8.5 | 83,752 | 75,520 | 3,523 | 3,438 |
| 17 | 8.8 | 88,566 | 83,712 | 1,192 | 949 |
| 18 | 12.3 | 90,661 | 88,320 | 16,957 | 16,625 |
| 19 | 14.0 | 108,245 | 90,624 | 8,341 | 8,157 |
| 20 | 14.7 | 119,242 | 108,032 | 3,095 | 3,043 |
| 21 | 17.0 | 125,511 | 119,040 | 12,086 | 12,033 |
| 22 | 19.6 | 141,095 | 125,440 | 13,529 | 13,401 |
| 23 | 21.5 | 107,055 | 9,984 | 10,690 | 10,522 |
| 24 | 22.0 | 118,246 | 107,008 | 2,614 | 2,282 |
| 25 | 22.2 | 121,661 | 118,016 | 1,010 | 848 |
| 26 | 22.9 | 122,976 | 121,600 | 3,757 | 3,680 |
| 27 | 23.0 | 126,959 | 122,880 | 779 | 652 |
| 28 | 23.3 | 128,024 | 126,720 | 1,627 | 1,420 |
| 29 | 23.6 | 135,032 | 128,000 | 1,633 | 1,155 |
| 30 | 24.2 | 137,508 | 134,912 | 3,245 | 3,134 |
| 31 | 24.4 | 141,188 | 137,472 | 1,022 | 897 |
| 32 | 24.8 | 73,744 | 17,664 | 1,409 | 1,111 |
| 33 | 24.9 | 75,409 | 73,728 | 453 | 333 |
| 34 | 26.5 | 78,013 | 75,264 | 7,093 | 6,960 |
| 35 | 26.9 | 85,644 | 77,824 | 1,482 | 1,301 |
| 36 | 27.6 | 87,362 | 85,504 | 2,712 | 2,670 |
| 37 | 28.1 | 90,296 | 87,296 | 2,374 | 2,220 |
| 38 | 28.7 | 93,428 | 90,112 | 2,494 | 2,174 |
| **38 turns** | 28.7 | 3,049,459 | 2,697,728 | 136,407 | 129,532 |

</details>

<details><summary>Every turn of trial 4</summary>

| turn | minute | prompt tokens | cached | output tokens | reasoning |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 10,403 | 3,584 | 208 | 57 |
| 2 | 0.2 | 11,116 | 10,240 | 272 | 79 |
| 3 | 0.4 | 15,554 | 11,008 | 979 | 885 |
| 4 | 1.1 | 24,161 | 15,360 | 2,204 | 1,958 |
| 5 | 1.2 | 26,885 | 24,064 | 281 | 160 |
| 6 | 2.1 | 30,210 | 26,880 | 3,799 | 3,708 |
| 7 | 2.5 | 36,866 | 30,208 | 1,513 | 1,124 |
| 8 | 3.3 | 40,037 | 36,864 | 3,580 | 3,483 |
| 9 | 3.9 | 44,239 | 39,936 | 2,857 | 2,788 |
| 10 | 4.2 | 54,908 | 44,032 | 1,080 | 881 |
| 11 | 4.3 | 56,369 | 54,784 | 169 | 89 |
| 12 | 4.6 | 57,410 | 56,320 | 1,117 | 948 |
| 13 | 5.8 | 59,221 | 57,344 | 4,594 | 4,244 |
| 14 | 7.4 | 65,078 | 59,136 | 6,754 | 6,594 |
| 15 | 7.5 | 72,034 | 65,024 | 240 | 121 |
| 16 | 7.6 | 73,144 | 71,936 | 437 | 258 |
| 17 | 8.2 | 74,279 | 72,960 | 3,223 | 2,944 |
| 18 | 8.6 | 81,635 | 74,240 | 1,488 | 1,282 |
| 19 | 9.3 | 83,695 | 81,408 | 2,856 | 2,671 |
| 20 | 10.0 | 88,418 | 83,456 | 3,086 | 2,988 |
| 21 | 10.4 | 92,996 | 88,320 | 1,451 | 1,300 |
| 22 | 11.2 | 96,032 | 92,928 | 3,506 | 3,389 |
| 23 | 12.0 | 101,003 | 96,000 | 2,896 | 2,724 |
| 24 | 12.9 | 107,289 | 100,864 | 4,145 | 3,943 |
| 25 | 13.1 | 112,383 | 107,264 | 935 | 695 |
| 26 | 13.4 | 113,460 | 112,128 | 1,227 | 765 |
| 27 | 14.3 | 117,618 | 113,408 | 3,635 | 3,552 |
| 28 | 14.5 | 121,646 | 117,504 | 429 | 177 |
| 29 | 14.6 | 123,418 | 121,600 | 400 | 248 |
| 30 | 14.9 | 125,193 | 123,392 | 1,523 | 1,277 |
| 31 | 15.1 | 126,911 | 125,184 | 855 | 545 |
| 32 | 15.6 | 130,091 | 126,720 | 1,827 | 1,625 |
| 33 | 17.1 | 133,669 | 130,048 | 6,846 | 6,656 |
| 34 | 18.2 | 140,893 | 133,632 | 4,905 | 4,557 |
| 35 | 18.6 | 59,387 | 10,496 | 1,075 | 989 |
| 36 | 19.9 | 60,843 | 59,136 | 5,203 | 4,591 |
| 37 | 20.1 | 68,077 | 60,672 | 678 | 150 |
| 38 | 20.2 | 69,850 | 67,840 | 659 | 309 |
| 39 | 20.6 | 71,191 | 69,632 | 1,301 | 581 |
| 40 | 20.7 | 73,620 | 71,168 | 213 | 114 |
| 41 | 20.8 | 73,862 | 73,472 | 547 | 165 |
| 42 | 21.3 | 75,231 | 73,728 | 1,923 | 1,418 |
| 43 | 22.1 | 78,766 | 75,008 | 3,057 | 2,898 |
| 44 | 22.9 | 87,550 | 78,592 | 3,618 | 3,374 |
| 45 | 25.0 | 93,881 | 87,296 | 8,463 | 8,368 |
| 46 | 25.3 | 103,261 | 93,696 | 1,317 | 1,090 |
| 47 | 25.4 | 108,387 | 103,168 | 432 | 196 |
| 48 | 26.2 | 109,310 | 108,288 | 3,238 | 3,139 |
| 49 | 26.9 | 112,828 | 109,056 | 2,709 | 2,602 |
| 50 | 29.1 | 115,881 | 112,640 | 10,336 | 10,113 |
| **50 turns** | 29.1 | 4,010,189 | 3,761,664 | 120,086 | 108,812 |

</details>

**What happened.**
- **Trial 1 (34 turns, 13 minutes).**
  - **One test, no engine.** At turn 1 it called `run_tests` and got the first
    image report; the provider took the image without error. It then used only
    python for 33 turns, never wrote to `engine.py`, and never called `read`,
    `edit`, `undo`, `render`, `try_step` or `auto_sprites`.
  - **It hid `show`.** At turn 4 it defined its own `show()`, which replaced the
    built-in. When it later called `show(..., boxes=...)` (turns 14-16) its own
    function failed, and it decided the built-in was broken.
  - **Two empty turns.** Turns 22 and 24 ended with `finish_reason: error` and
    were counted as turns.
  - **Where the analysis went.** Level 0's ring rule at turn 11, then the first
    frames of levels 1-7, then the budget bar over every level.
- **Trial 2 (38 turns, 29 minutes).**
  - **A good start.** Turn 1 was `auto_sprites(0)`: it printed sprite code that
    redraws level 0's first frame exactly. At turn 3 it tried to define `show`,
    the kernel refused to run it, and at turn 4 it renamed its function.
  - **Then the same habit.** It analysed level 0 on turns 2-11, then moved to
    the first frames of every level, level 1's rings, level 2's rings and
    level 3, and in 38 turns never called `edit()` or `run_tests`. The
    `auto_sprites` code from turn 1 was never put into `engine.py`.
  - **Twice the tokens.** At turn 30 it had written 117,368 output
    tokens, against 59,719 for v4 and 54,763 for trial 1 at the same turn. Four
    turns (18, 21, 22 and 23) spent 52,581 reasoning tokens between them, one
    of them 3.5 minutes long. It took 24 minutes for 30 turns, against 14 for
    v4.
  - **A cache miss at turn 23.** The prompt reached 141,095 tokens at turn 22,
    the harness cut old tool outputs, and the next request (107,055 tokens)
    had only 9,984 of them cached.
- **Trial 3 (4 turns).** It began with "Let me start by exploring the
  recording", not with the example's first call, and was printing frame
  differences by turn 4. Stopped there to put `read()` first in the example.
- **Trial 4 (50 turns, 29 minutes).**
  - **No test, no edit, no `auto_sprites`.** All 52 tool calls were python
    analysis; `engine.py` ended as it started and the final replay matched 0 of
    120 steps.
  - **Why `read()` did not start it.** The first message already contains
    `engine.py` with its anchors (276 lines, 216 of them the fixed block), and
    the example's one-line pointer comes after it. At turn 1 the agent noted
    the file was "already shown" and dropped the sequence with that step.
  - **It read the rules and deferred them.** After the harness's reminder at
    turn 30 (30 turns without a test), turn 31's reasoning: "But the
    instructions say to run tests early. Let me do a quick auto_sprites for
    level 0 and run tests to establish the baseline, then continue analysis.
    Let me be efficient: do the analysis of all levels first (a few tool
    calls), then write the engine, then test."
  - **What it worked out.** Most of the game, by hand: the rings of blocks
    that each arrow turns by one place, arrow colour as direction, the bracket
    targets, the time bar (5 pixels a click in level 0, 1 afterwards, refilled
    on a new level) and the level markers. It wrote six frame parsers on the
    way (turns 7, 13, 26, 34, 37 and 39), each a partial `auto_sprites`, and
    repeatedly miscounted columns in printed frames.
  - **Tokens.** 120,086 output tokens, 7% fewer than v4 at 50 turns; the
    longest turn (50) wrote 10,336.

**Conclusion.** The new tools work: the image went through the provider, the
reserved-name rule fired and the agent recovered in one turn, and
`auto_sprites` drew the first frame exactly. No prompt change moved the agent
off its habit of analysing every level before writing: not the rules, not the
first message, not a worked example in two orders, not the harness's reminder.
Turn 31 of trial 4 shows it reads the instructions and chooses to put them
off, so the order has to come from the harness, not the prompt.

Next, not tried yet:
- **The harness plays the opening.** It runs the tests, writes
  `auto_sprites(0)` into `make_level` and tests again; the first message shows
  that edit, the agent's part of `engine.py` (the fixed block folded) and the
  first failing step with its images.
- **A short analysis quota.** The python tool pauses after a few calls (5-8)
  that leave `engine.py` unchanged and resumes on the next edit. The harness
  supports this already (`--python-quota N`); these trials ran without it.
- **Only the levels reached.** Show the agent the levels the tests have
  reached, plus the next one.
<!-- v5-trials:end -->

## Caveats

- **Sample size:** one sample per game at temperature 0.7, with a 115-minute
  budget. Repeats would vary, and more time might finish vc33.
- **Coverage:** the agent can only reproduce what the recording shows. ls20
  level 6 and sp80 levels 2-5 were never reached, and held-out play is scored
  only on recorded levels.
- **Random play misses rules:** random play rarely completes a level or runs
  out of budget, so a wrong win or lose rule can go unnoticed. Examples are
  vc33's overflow limit, sp80's budget check on a spill, and v3 ft09's
  level-3 win check.
- **Harness tuning:** the feedback mechanisms were tuned on pilot runs of these
  same games; there are no held-out games.
- **Sandbox limits:** the sandbox is an audit hook, which keeps an honest model
  honest but is not a security boundary. The source scan found nothing.
- **Pilot A cost:** pilot A's traces came from a new 5-game run at 150K output
  tokens per game, started before the archived run could be pulled. Its cost
  is not measured.

## Reproduce

From `ARC3-Inference/`:

```bash
dvc remote modify --local storage allow_anonymous_login true
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY dvc pull runs/20261004_135539.dvc
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --games ls20,ft09,vc33,sp80,lp85 --out runs/engine-re/<name> --model qwen/qwen3.8-flash \
  --max-turns 300 --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
uv run --no-sync python -m engine_re.evaluate runs/engine-re/<name> --engine best --rollouts 8 --length 40
```

These commands run the current harness (v5; its lp85 trials used `--games lp85 --max-turns 50`, each at the commit given in the [v5 section](#v5-trials-on-lp85-50-turns)). To rerun an earlier configuration exactly, check out its commit first: `41df359` for v4, or `cd76c9d` for v2 and v3, adding `--python-quota 30` for v3. Before running a generated
engine yourself, copy `results/<config>/<game>/engine_best.py` into a game
directory.

## Later experiments (v6-v9)

The stepwise harness (v6), sampling and reasoning effort (v7), frame pieces
and a parsimony prompt (v8), two context condensers and the HUD-bar tolerance
(v9) were tried on lp85 alone, one sample each. Each series, its commits,
commands, results and every run's DVC pointer are in
[exp/README.md](../exp/README.md). Headline: with the v9 tolerance of one
HUD-bar pixel at the frame border, the agent went from 20 to 65 of 120 lp85
steps (v6c and v8: 20; v9: 65, for $0.36).
