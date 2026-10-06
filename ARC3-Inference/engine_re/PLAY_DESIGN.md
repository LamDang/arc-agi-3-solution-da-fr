# Design: the play-and-model agent (`engine_re` v10, "play" mode)

An agent that plays a live ARC-AGI-3 game while building `engine.py`, the model
of that game the reverse-engineering agent builds from a recording. It is the
stepwise harness (v6c-v9, [README.md](README.md)) with the recording replaced by
the game being played: the agent fits what it has seen, uses the fitted engine
to choose its next moves, sends them, and the harness checks every real
observation against the engine's prediction. The first prediction that fails
stops the batch and starts a fit round on that step.

This document is the design to review before implementation. The run plan
(section 7) tests it on three of the five public games.

## 1. The loop

```
                 ┌──────────────────────────────────────────────────────┐
                 │  FIT: "fix step k"  (the stepwise harness as it is)   │
                 │  python / edit_file / run_tests ... commit_engine     │
                 └───────────────┬──────────────────────────────────────┘
       commit accepted: steps 0..n-1 pass   │        ▲ the real result of step k
                                            ▼        │ differs from the engine's
                 ┌──────────────────────────────────────────────────────┐
                 │  PLAN: "steps 0..n-1 pass; the game is at level L"    │
                 │  python: state_now(), replica.step on copies of it    │
                 │  commit_moves(actions): the Actions, as printed       │
                 └───────────────┬──────────────────────────────────────┘
                                 │ for each action, in order:
                                 │   predicted = replica.step(...)     (committed engine.py)
                                 │   real      = game.perform(...)     (the arcengine game)
                                 │   same final frame, status, levels, actions?  yes → next action
                                 │                                               no  → stop, FIT on step k
                                 ▼
                  all actions matched → PLAN again (new frame)   |   WIN → done
```

One conversation per game, as in v6c. This is the key idea: fitting and
planning are done by the same agent in the same context, never by two agents
handing an engine over. The reasoning that found a rule is still in context
when the rule is used to plan, and the plan that failed is in context when
its step is fitted. The harness only decides which phase the next message
asks for. Three phases:

- **Opening** (harness only, as today). Step 0 is the RESET that starts the
  game. The harness puts level 0's first frame into `make_level` from
  `recording[0].pieces_after.code()`, tests, and commits for the model. The
  first message is a PLAN message.
- **PLAN.** The model sees every step so far (`recording`), the current frame
  as an image, the level and what the game accepts. In python, `state_now()`
  is engine.py's state after replaying everything played, and the model plays
  moves on copies of it by calling `replica.step` directly (`replica` is the
  kernel's built-in for engine.py). The model ends the phase with
  `commit_moves(actions)`, the actions being the `Action`s it stepped its
  replica with, as python prints them.
- **PLAY** (harness only). The harness predicts the batch with the committed
  engine, then sends the actions to the real game one at a time and compares.
  On the first mismatch it drops the rest of the batch and opens a FIT round
  on that step; when the batch matches, it returns to PLAN; a WIN ends the game.
- **FIT.** Exactly the stepwise harness: "Fix the breaking test: step k", the
  recording up to k, the test report with its picture, engine.py listed. Only
  `commit_engine` with steps 0..k passing moves on, to PLAN.

The nudge the user asked for is the `commit_engine` output in play mode:
"Committed: steps 0-k pass. Plan the next moves: ..." followed, after the
turn's tool messages, by the PLAN user message with the current frame.

## 2. What is reused

| Piece | From | Change |
| --- | --- | --- |
| The real game: load, `perform`, RESET-restarts-the-level, determinism | `engine_re.trace` (`load_game_class`, `new_game`, `perform`) | none; the same code that rebuilt the recordings |
| `Trace`/`Step`: the recording | `engine_re.trace` | grows one step per real action; saved after every batch |
| `engine.py`, the fixed interface, `GameRunner`, rendering, click cells | `engine_re.game_api`, `skeleton` | none |
| Tests: contract + replay, failure report with images, HUD-bar tolerance, `tests.jsonl`, `engine_best.py` | `engine_re.tester`, `candidate_runner` | `ignore` (steps not compared) and `resync` (section 4.6) |
| Kernel: `recording`, `step_to_fix`, pieces, `read_file`/`edit_file`/`undo_edit`, `replay_step`, `show_frames`, reserved names, replay on resume | `engine_re.kernel`, `helpers` | two built-ins added: `state_now`, `click_cell` |
| Agent loop: OpenRouter client, retries, tool dispatch, automatic test after an edit, images after tool messages, compaction/condenser, transcript, resume, budgets | `engine_re.agent.EngineAgent` | subclassed; one tool added |
| Stepwise messages: `episode_message`, `advance_message`, `step_objects` | `engine_re.prompts` | a `play` mode: system prompt, PLAN message, `commit_moves` schema |
| Score | TAAF's formula (`taaf/game.py`) and `metadata.json` `baseline_actions` | reimplemented in 15 lines; the run also writes a `benchmark.json`-shaped record so `make score_run` and the viewer can read it (section 4.8) |

The base play harness (`inference/`) is not reused as code. Its solver class
is hard-wired (`run.py:_make_solver` always builds `HarnessSolver`; the only
hook, `HarnessSolver.analyzer_factory`, is dropped when the solver is
pickled), and its agent loop (`tool_agent.py`, 7,000 lines) is tied to its
one `python` tool, its context trimming and its scheduler. What is borrowed
is its vocabulary (action names `UP`/`DOWN`/`LEFT`/`RIGHT`/`SPACE`/
`MOUSE(row, col)`/`RESET`, the upscaled current frame), two of its behaviours
(an automatic RESET after a game over; a batch stops at a level change) and
its run artifacts, so the two agents can be scored and viewed alike.

## 3. Key implementation points

### 3.1 The live game and the trace (`engine_re/live_game.py`, new, ~120 lines)

```python
class LiveGame:
    def __init__(self, game: str, environments_dir: Path): ...   # ONLY_RESET_LEVELS=true, seed 0
    trace: Trace                      # every step played, step 0 = the opening RESET
    def perform(self, action: Action) -> Step   # sends it, appends and returns the Step
    def replay(self, actions)                   # resume: a fresh game fed the trace's actions
    def save(self, directory)                   # trace.json + frames.npz, as a recording
```

- `Step` already carries what the comparison needs: final frame, `state`
  (NOT_FINISHED / WIN / GAME_OVER), `levels_completed`, `win_levels`,
  `available_actions`, and the animation frames for the model to look at.
- The trace is the single source of truth: the tests, the kernel's
  `recording`, the viewer record and the resume all come from it. Saving it
  after each batch (a few ms) makes a run resumable at batch granularity.
- Actions per level, for the score, are counted from the trace (level
  changes are visible in `levels_completed`); RESET counts as an action, as
  in TAAF.

### 3.2 Tools and built-ins

Tools: `python`, `run_tests`, `commit_engine` (as in stepwise mode) and one new
tool, **`commit_moves(actions, note)`**.

- `actions`: a list of 1 to `--batch-size` (default 10) actions, each the
  fixed block's `Action` as python prints it: `"Action(4)"`, `"Action(6, x=12,
  y=40)"` (screen pixel, x column, y row), `"Action(0)"` for RESET; or the
  printed list as one string, `"[Action(4), Action(0)]"`. The labels stay
  accepted (`"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`, `"SPACE"`, `"RESET"`,
  `"UNDO"`, `{"click": [x, y]}`, the base harness's `MOUSE(row=, col=)`;
  `trace.parse_moves`). Only the game's advertised actions are allowed;
  others are refused before anything is sent.
- **One action object for the replica and the game.** What the model steps
  its replica with in python is what it submits: the kernel's `Action`
  prints as the code that builds it, cell left out (`Action(4)`,
  `Action(6, x=12, y=40)`, `Action(0)`; `trace.action_code`), so `print(moves)`
  is pastable into `actions`, and the messages name moves the same way
  (`#12 Action(4): matches your prediction`, "this game does not accept
  Action(1) (UP)"), so what the model reads it can write back. The cell is
  the harness's to compute (`game_api.click_cell`, before `step()`): a cell
  in the text (`Action(id=6, x=3, y=4, cell=(1, 1))`, the dataclass form) is
  ignored, and `replica.step` fills in a click's cell when it is None, so
  `Action(6, x=3, y=4)` read from a message plays the same click in python as
  in the game. The choice: the printed form is python, not JSON, because it
  is the code that builds the object (as `Sprite` prints), it evaluates back
  in the kernel, and a JSON-looking repr (`{"id": 4}`) would make an object
  read as a dict. The repr is set on the harness side (`game_api.canonical`
  patches the canonical `Action`, the one the kernel preloads), not in the
  FIXED block: the block stays byte-identical, so every engine.py written
  with it, the contract test (`same_interface`) and the reference engines are
  untouched. An engine's own `Action` (the one `GameRunner` hands its
  `step()`) keeps the dataclass repr, `Action(id=4, x=0, y=0, cell=None)`,
  which `commit_moves` accepts as well. RESET is `Action(0)`: `step()` never
  gets it, `commit_moves` sends it.
- `note`: one or two sentences: what the batch is meant to do and what the
  engine predicts (kept in the transcript and `result.json`; it is the
  planning record, as `commit_engine`'s message is the modelling record).
- **It runs the tests first, before anything reaches the real game.** The
  current engine.py is replayed on every step so far (the same full test
  `commit_engine` runs). If any step fails, nothing is sent: the output is
  "Not sent: your engine does not reproduce the game so far" with the test
  report, and the model is back in FIT. This holds whether the model skipped
  the fit round, edited engine.py after committing, or committed an engine
  that passed only by the HUD tolerance and then broke it. If every step
  passes, the engine is recorded as committed (the batch's `note` stands as
  its commit message when `commit_engine` was not called) and the batch is
  sent. The invariant: every prediction comes from an engine that reproduces
  the whole game so far, so a mismatch is new information about the game,
  never a stale engine. The test costs one sandboxed replay, as a commit does.

Built-ins added to the kernel (`helpers.py`), documented in the `# Objects`
reference:

- `replica`: engine.py as it is now (every mode's built-in; it was `engine`
  before, renamed so that the model does not take it for the real game's
  engine, arcengine; the play prompts call engine.py "your replica" and the
  real game "the game"). The file stays `engine.py`.
- `state_now()`: engine.py's `State` after replaying every recorded step
  (the committed engine.py or the current file; the current file, as
  `replay_step` does, so the model can test an edit before committing). Cached
  per engine hash and trace length.
- `click_cell(state, x, y) -> tuple | None`: the grid cell under a screen
  pixel on this state, as the harness computes `action.cell` for a click
  (`replica.step` fills it in when an `Action`'s cell is None).
- No `simulate` helper (it existed in the v10 run and was removed after it):
  the model plays moves by calling `replica.step` on copies of a State and
  `replica.make_level(n)` for a level's first state, with the harness rules
  stated in the prompt (RESET is a fresh `make_level`, `level_solved` starts
  `make_level(n + 1)`). The v10 transcripts showed the model composing routes
  in text and using `simulate` only to verify them, while its searches
  re-implemented the rules by hand; a BFS over moves calling `replica.step` on
  copies is one short function away, which is the point of having a model.
- They are reserved names (`kernel.RESERVED`) and go in the `# Objects` text
  and its test.

### 3.3 Prediction and check (the PLAY phase)

For a batch of actions `a_1..a_m` after `n` recorded steps:

1. One sandboxed run of the committed engine on `recorded_actions + batch`
   (`tester.run_candidate`, the same subprocess the tests use, contract
   checks off): the predicted observation and frame of each step
   `n..n+m-1`. Cost: milliseconds per step. An engine error at predicted step
   `i` is a mismatch at `i` (the action is still sent, see below).
2. For `i = 1..m`: `real = game.perform(a_i)`; the comparison is
   `tester.check_step(real_step, predicted_obs, predicted_frames)`, exactly
   the test's rule: final frame, status, levels completed, win levels,
   available actions; animation frames are not compared; one HUD-bar pixel at
   the frame's edge is tolerated with a warning.
   - match: append the step to the trace; continue.
   - mismatch (or an engine error): append the step (it is real), drop
     `a_{i+1}..a_m`, stop. A tolerated HUD-bar pixel counts as a match, with its
     warning shown (a `--strict` live check is not implemented: the tests
     tolerate it too, so a strict fit round would already pass).
3. Save the trace; `kernel.refocus(n+i-1)` so `recording` grows and
   `step_to_fix` is the last step; write the batch record.

The tool output lists one line per action, each move named as its `Action`
prints: `#12 Action(1): matches your prediction`; `#13 Action(2): differs from
your prediction: the final frame differs`; then how many moves were not sent. Then the harness adds the user message
of the next phase:

- a mismatch: `advance_message`-shaped "Fix step k" (section 3.4);
- all matched: the PLAN message with the new frame;
- WIN: the run ends with status `won`;
- GAME_OVER in the real game (predicted or not): the harness plays RESET
  itself, as the base harness does (`_execute_auto_reset`), so action counts
  stay comparable; the RESET is a real step, checked like any other (the
  engine's `make_level` must redraw the level's start), and the next message
  says the game was lost at step k and restarted. `--no-auto-reset` leaves
  the RESET to the model instead;
- a level change: the batch stops there (as the base harness stops a batch
  at `level_completed`), since the rest of the plan was made for the old
  level. The step that solves the level is where `make_level(L+1)` must draw
  the new level's first frame, so it is usually a FIT round (the model has not
  seen the level) and then a PLAN on the new level.

The model's own mistakes in `actions` (a click outside the screen, an action
the game does not advertise) are refused before sending, with the valid form.

### 3.4 The messages (`prompts.py`, mode `"play"`)

- **System prompt**: the stepwise prompt's Setup, Drawing, Sandbox, Tests
  and Objects sections as they are, with a new Goal ("win the game in as few
  actions as possible; the way to do that is to keep a model that
  reproduces every step so far and to plan with it") and a "How to work"
  that adds the plan phase: look at the current frame and `state_now()`,
  try sequences with `replica.step` on copies, prefer the shortest sequence the replica
  says solves the level, and when the engine has never seen a kind of
  move, send a short batch (1-3 actions) to learn its effect rather than a
  long plan built on a guess. Scoring: each level's score is
  `(baseline / actions)²`, so actions are the cost, not turns; a RESET is an
  action too.
- **PLAN message** (new, `plan_message`): "Steps 0-{n-1} pass with your
  committed engine. The game is at level L ({c} of {w} completed), {status};
  the last action was ...; {budget line}. Current frame: [image, upscaled 8x,
  as the test pictures]. Work out the next moves on your engine, then
  commit_moves(actions, note)." When a level was just solved, it says so and
  that `make_level(L)` drew its first frame correctly (the step matched).
- **FIT message**: `advance_message` as it is, with its first line replaced by
  "Your engine predicted a different result for step k ({action}): {one-line
  diff summary}. Steps 0-{k-1} match." and the test report (which is the
  comparison of engine and game at step k, with the picture: engine's frame
  left, game's right, regions boxed). After the commit, the harness does not
  "replay on" (there is nothing after k yet): it tests 0..k and, on pass,
  returns to PLAN.
- **Budget line**: actions played, turns and minutes used against the caps,
  and the level's human baseline (`metadata.json`) so the model knows what
  "few" means.

### 3.5 Phase rules in the agent (`engine_re/play_agent.py`, new, ~400 lines)

`PlayAgent(EngineAgent)`, `stepwise=True, history=True`, mode `"play"`.

- `self.phase in {"plan", "fit"}`; `self.committed_sha` (the engine that
  passed all steps); `self.live: LiveGame`.
- `_tool_commit_engine`: as in stepwise mode (tests 0..k must pass), then
  `committed_sha = hash`, `phase = "plan"`, and the output carries the nudge;
  `_advance` is replaced: no replay-on, the PLAN message is queued instead.
- `_tool_commit_moves`: runs the full test on the current engine.py; refused
  with the report when any step fails (nothing sent); otherwise records the
  engine as committed and queues the batch (section 3.3). Applied after the turn's
  tool calls like a commit today (`self.pending_batch`), so a turn that
  edits engine.py after `commit_moves` drops the batch with a note, as a
  commit is dropped now. One batch per turn.
- The automatic test after an edit, the nudges, images, compaction and the
  transcript are inherited. `engine_best.py` and `tests.jsonl` keep their
  meaning: the best engine over the game so far.
- Budgets inherited (`turns`, output tokens, cost, minutes) plus
  `--max-actions` (default 500). When a budget runs out the run ends with the
  game where it is (status `turn limit` etc. as today, plus the game's
  outcome).

### 3.6 Escape hatch: a fit round that does not converge

v6c-v9 show the model can spend 30 turns on one step (v8 spent 32 turns on
lp85's step 20). In play mode that would freeze the game. Two mechanisms,
both off by default in the first run so the pure loop is measured first,
then on in the second:

- `--fit-turns N` (suggested 25): after N turns on the same step without an
  accepted commit, the harness tells the model it may play on with the engine
  **out of sync**: `commit_moves` is accepted again (with the last committed
  engine), but predictions are not checked, and the steps from k on are
  marked `unexplained`. The engine is **resynced** at the first later step
  that starts a level fresh: a RESET (level restart) or a level change,
  where `make_level(L)` is the known state. From that step the checks
  resume. The message says that a RESET resyncs at once and costs one action,
  so the model can choose between losing the level's progress and playing
  blind to the end of the level.
- In the tests (`replay_test`), `ignore` (the unexplained steps: compared
  but never failing, excluded from `exact`, `total` and `passing_prefix`; an
  engine error there does not end the replay) and `resync`
  (`{step: {"level": L, "score": s}}`: before that step the runner is set to
  level L with score s, then performs the RESET as usual, or, for a level
  change, shows level L's start without calling `step()`), so the engine's
  replay stays aligned with the game after a divergence. `candidate_runner`
  takes both as arguments; the candidate still sees only actions. As
  implemented, both live in the live trace's meta (with `out_of_sync`, the
  step the engine is out of step since), so the kernel's `state_now` and
  `replay_step` follow them too; an out-of-step batch stops at
  the resync point, after which every step is tested and the loop goes on
  (a FIT round when the resync step itself fails, e.g. `make_level(L+1)` not
  drawing the new level). The hatch is refused when the current engine.py
  fails a step before the fit round's step.
- Unexplained steps are listed in `result.json` and the PLAN message, and the
  model can come back to them later with `replay_step(k)`.

### 3.7 Stopping and outcome

The run ends on WIN, on a budget (turns, tokens, cost, minutes, actions),
when the model stops calling tools (4 idle turns, as today) or on a request
that fails for good. `result.json` adds: `outcome` (`won` / `playing` at the
end), `actions`, `levels_completed`, `win_levels`, `actions_per_level`,
`score` (TAAF's formula with `baseline_actions`), `batches` (per batch: turn,
actions sent, matched, the mismatch step and its one-line diff, the note),
`fit_rounds` (per round: step, turns, commits, accepted), `unexplained`,
`phase_turns` (`plan` / `fit`) and the usual usage.

### 3.8 Artifacts and scoring

Per game directory, as today (`trace/`, `workspace/engine.py`,
`engine_versions/`, `engine_best.py`, `transcript.jsonl`, `tests.jsonl`,
`images/`, `result.json`) plus:

- `artifacts/<game_id>_p0_events.jsonl`, the base harness's event
  sidecar (the sidecar of `<game_id>_p0_viewer_data.json`, as
  `inference/utils/viewer_artifacts.py` names it, so `engine_re.trace.events_path`
  finds it) (`initial` / `action` events with the action label, board, level
  and state, as `inference/utils/viewer_artifacts.py` writes them), which is
  what `engine_re.trace.trace_from_run` reads: a play run can then be fed
  back to the recording-based harness, and `scripts/pack_run.py` can replay
  it.
- `benchmark.json` at the experiment root, built with TAAF's own
  `taaf.game.GameRun` (`actions_per_level`, `base_actions_per_level`,
  `levels_completed`, `state`, a history entry per action with the output
  tokens spent on it, `final_score`), so `make score_run
  SCORE_RUN_DIR=runs/engine-play/<name>` scores it with the same code as the
  base runs (`inference/tools/eval.py` loads `taaf.benchmark.Benchmark` from
  it). The score in `result.json` is the same formula, computed directly.
- `summary.md`: per game, score, levels, actions, batches, mismatches, fit
  rounds and turns per phase, tokens and cost.

### 3.9 Resume

As in stepwise mode (transcript rebuilt, kernel cells replayed), plus the
game: `LiveGame.replay(trace.actions())` rebuilds the real game, which is
deterministic, as `trace.py` relies on. A batch interrupted mid-way is
truncated to the steps the trace holds (saved per step, not per batch).

### 3.10 Context

Unchanged: compaction at 140K prompt tokens, or `--condense`. The PLAN
messages carry one image each; older ones are replaced by the placeholder as
test pictures are today. Each PLAN message lists engine.py as the FIT
messages do only when it changed since the last listing (the `elide` rule
already keeps one listing live).

## 4. Files

| File | Change |
| --- | --- |
| `engine_re/live_game.py` | new: `LiveGame` (section 3.1), action parsing from the model's labels, `events.jsonl` writer |
| `engine_re/play_agent.py` | new: `PlayAgent` (3.3, 3.5, 3.6, 3.7), `score()` |
| `engine_re/run_play.py` | new CLI: `--games`, `--out`, `--model`, budgets, `--max-actions`, `--batch-size`, `--fit-turns`, `--plan-turns`, parallel games, `summary.md`, `benchmark.json` |
| `engine_re/prompts.py` | mode `"play"`: system prompt, `plan_message`, `mismatch_message`, `commit_moves` schema, Objects entries for `state_now`/`click_cell` |
| `engine_re/helpers.py`, `kernel.py` | `state_now`, `click_cell`; `PRELOADED_PLAY`/`RESERVED_PLAY` |
| `engine_re/tester.py`, `candidate_runner.py` | `predict()` (run the engine on recorded + planned actions), `ignore`, `resync` |
| `engine_re/agent.py` | small hooks: a `phase` label on transcript records, `_advance` overridable, the commit output text per mode |
| `tests/test_play.py` | the loop with a scripted client: match → PLAN, mismatch → FIT, refused `commit_moves`, GAME_OVER then RESET, fit cap and resync, resume, score |
| `engine_re/README.md`, `exp/v10-play.md` | the play mode and the run page |

A scripted client (a fake `OpenRouterClient` answering fixed tool calls) with
the reference engines from `runs/engine-re/ports` lets every branch of the
loop be tested without a model: the ft09 port predicts every step, so a
scripted "perfect" agent must win in the baseline's actions, and a port with
one rule removed must produce exactly one FIT round at the first step that
rule affects.

## 5. Decisions taken, to confirm

1. **Standalone harness in `engine_re`, not a solver in `inference/`.** It
   keeps every v6-v9 mechanism and their measurements comparable; the base
   harness is matched on artifacts (score, viewer), not on code.
2. **`commit_moves` tests the engine itself and refuses, with the report,
   when any step so far fails, before the real game is touched.** The pure
   form of the loop: the model cannot skip the fit. The escape hatch (3.6) is
   the only relaxation, and it is explicit in the results (unexplained
   steps).
3. **Planning is in python (`replica.step` on copies), sending is a tool.**
   The thing that "gives the moves and gets final states" is the engine
   itself, called directly in python; `commit_moves` only sends. This lets
   the model search over its engine; a tool call per candidate plan would
   not.
4. **Batch size 10, checked per action.** The batch is predicted at once and
   sent step by step, so the first divergence stops it; later actions of a
   plan are never sent on a wrong premise.
5. **The opening is the harness's** (level 0's frame into `make_level`), as
   in v5e onwards, so the first model turn is a PLAN.
8. **The real game is stepped directly** (`arcengine` game class, as
   `trace.py` does), not through TAAF's `GameAPI`. TAAF adds validation and
   bookkeeping the harness does itself here; stepping directly keeps the
   trace, the tests and the live game on one code path. The TAAF record is
   written at the end for scoring (3.8).
6. **Comparison rule = the tests' rule** (final frame + status + levels +
   actions, HUD-bar pixel tolerated), so "matches" means the same in play
   and in tests.
7. **Fit rounds are open-ended by default**; `--fit-turns` is the second
   configuration, not the first.

Open: whether the PLAN message should also give the base harness's
board-diff image of the last step (cheap, may help). Playing moves on the
engine in python is not capped: searching needs long rollouts; only sending is capped.

## 6. Expected failure modes

- **Fit rounds too long for a live game.** Measured by `phase_turns` and
  `fit_rounds`; mitigated by `--fit-turns` in the second run.
- **Over-fitted engines plan badly.** v2's lp85 engine was a lookup table
  that passes the recording; such an engine predicts nothing new, every new
  click mismatches, and the loop degrades to one fit round per action. The
  parsimony prompt and the stepwise harness reduced this (v9's lp85 engine
  is a general rule); the batch records show it directly (actions matched per
  batch).
- **Exploration is not planned.** A model that only sends what the engine
  predicts never tries an unknown mechanic. The prompt asks for short probing
  batches; whether it does so is read from the notes and batch lengths.
- **Image load.** One image per PLAN message on top of the test pictures;
  the placeholder rule keeps only the latest live.

## 7. The run plan: three games

The five public games and what the base play agent did
(`runs/20261004_135539`, qwen3.8-flash, 500K output tokens per game):

| game | base agent | levels | actions | engine_re result (v2/v4/v9) |
| --- | --- | --- | --- | --- |
| sp80 | gave up on level 1, 3 game overs | 1 of 6 | 111 | v2 passed the recording with fitted heuristics (91% held-out) |
| ls20 | stopped at level 6 on the token limit | 5 of 7 | 866 | never beyond 17 of 867 steps |
| ft09 | won | 6 of 6 | 100 | v2/v4: a correct re-implementation, cheapest pass ($0.14) |
| vc33 | won | 7 of 7 | 331 | v2: levels 0-2 only |
| lp85 | won | 8 of 8 | 119 | v9: 65 of 120 steps, general rules |

Proposed:

| case | game | why |
| --- | --- | --- |
| not managed | **sp80** | the base agent's only clear failure (level 1 of 6, score 3.7); the recording was modelled correctly for level 0 physics, so a fitted engine has a chance to carry the agent past level 1 |
| in the middle | **ls20** | 5 of 7 at 866 actions; the hardest game to model (pushers, 17-frame animations), so it tests the escape hatch and whether a partial model still cuts actions |
| solved | **ft09** | won by the base agent in 100 actions; the engine was re-implemented correctly in 22 minutes in v4. The question here is cost: does the loop win at the same action count for less (or more) than the base agent's 101K output tokens and 31 minutes? |

ft09 is the solved case (decided); lp85, where every v5-v9 measurement was
made, is a fourth run if the first three say the loop works.

Settings, one sample per game, all three in parallel, as the v9 runs but
with the play agent's caps:

```bash
uv run --no-sync python -m engine_re.run_play --games sp80,ls20,ft09 \
  --out runs/engine-play/qwen38flash-v10 --model qwen/qwen3.8-flash \
  --max-turns 300 --max-output-tokens 1500000 --max-cost 6 --max-minutes 240 \
  --max-actions 500 --batch-size 10
```

Then the same with `--fit-turns 25`. Read, per game: score and levels against
the base run; actions per level against the base run and the human baseline;
turns and tokens per phase; batches, actions matched per batch, mismatches
per level; fit rounds and their length; the engine's held-out accuracy with
`evaluate.py` (it works on the saved trace as on a recording). Everything is
archived with `dvc add` as the other runs and written up in `exp/v10-play.md`.
