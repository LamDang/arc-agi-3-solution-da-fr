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
- **Next (v5, in progress):**
  - **Tools:** only `python`, `run_tests` and `finish`.
    - `finish` runs the tests and ends the session only when all pass.
    - `read`/`edit`/`undo` run inside python, address lines by anchors, and
      cannot change the fixed block.
  - **Failure report:** `run_tests` stops at the first failure and shows both
    frames as images, with the differing regions boxed. It also lists the
    agent's sprites in each region, what `step()` printed, and a one-line
    reproduction (`try_step`).
  - **Generated code:** `auto_sprites` writes sprite code for a level layout,
    and `show` displays frames as images.
  - **Prompt:** goal, setup, drawing rules, tests and tools.

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

These commands now run the v4 configuration. To rerun v2 or v3 exactly, check out commit `cd76c9d` (before the v4 changes) and run the same commands, adding `--python-quota 30` for v3. Before running a generated
engine yourself, copy `results/<config>/<game>/engine_best.py` into a game
directory.
