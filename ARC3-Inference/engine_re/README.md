# Engine reverse-engineering experiment

Can an LLM agent rebuild an ARC-AGI-3 game engine from what a run observed?
Given every action of a recorded run and every frame the real engine returned,
the agent edits a Python module until replaying the recorded actions through
it returns exactly the recorded observations. The real game source is never
visible to it.

Results on the 5 public games with qwen3.8-flash are in [RESULTS.md](RESULTS.md):
3 of 5 engines reproduce their whole recording; one of those three is a
correct re-implementation, the others partly fit the recording.

## How it works

```
harness run (events.jsonl: actions + final boards)
   │  trace.py: replay the logged actions through the real engine,
   │            check every final frame against the logged board
   ▼
trace/  (every frame of every action, state, levels, available actions)
   │
   ▼
agent.py ── python ──────────► kernel.py   persistent, sandboxed Python with the trace
   │                             │ read_file / edit_file / undo_edit ──► engine_files.py ──► workspace/engine.py
   │                             │                        (the only writer; versions in engine_versions/)
   │     ── run_tests ───────► tester.py ──► candidate_runner.py (sandboxed; gets only the actions)
   │     ── commit_engine ───► tester.py (submits engine.py; stepwise: the only way on)
   ▼
result.json, transcript.jsonl, tests.jsonl, engine_best.py, images/
   │
   ▼
evaluate.py: candidate vs real engine on new random action sequences per level
```

- **Trace** (`trace.py`). The games are deterministic, so replaying a run's
  logged actions through the real engine reproduces the run's observations
  exactly; the replay is checked against the board the run logged after every
  action. The replay also recovers what the logs drop: every animation frame.
  RESET restarts the current level, as in the harness (`ONLY_RESET_LEVELS=true`).
- **Interface** (`game_api.py`, `skeleton.py`). engine.py is a plain module: a
  fixed block defining `Sprite`, `Action`, `View` and `State` (plus pixel
  helpers; its comment states the drawing rules in the same words as the
  prompt), which the agent must not edit, and two functions the agent writes:
  `make_level(n) -> State` (grid size, all sprites including border,
  background and HUD, and hidden `vars`) and `step(state, action)`, which
  changes the state in place and sets `state.status` to `"level_solved"` or
  `"game_over"`. The harness renders states (identical to the real games'
  drawing on all 34 level starts of the five games), turns clicks into grid
  cells, and applies the episode rules (RESET restarts the level, level
  changes, WIN, GAME_OVER) in `GameRunner`, so the tester, sandbox and
  evaluation work unchanged. The contract test compares the fixed block's code
  with the original, ignoring comments (and accepts the code of earlier blocks,
  `game_api.EARLIER_INTERFACES`, so engines of earlier runs still pass). A
  `Sprite` prints as the code that builds it, with only the fields that differ
  from their defaults (`Sprite([[8, 8, -1], ...], x=28, y=8, rotation=180,
  tags=("shape_8_3x4_79b9",))`; equal rows as `[[3] * 64 for _ in range(64)]`),
  so a printed sprite list is code, and `eval` of it gives equal fields. The
  tester and `evaluate.py` still score engines written as an `arcengine` game
  class (the earlier runs).
- **Tools** (`agent.py`, `prompts.py`): exactly three.
  - `python(code)`: a persistent kernel. Its namespace holds `np`, the
    fixed-block classes, `recording` (the recorded steps) and seven functions
    (`helpers.py`); nothing else is preloaded. These names are reserved: code
    that binds one (`def`, assignment, parameter, loop variable, import as) is
    refused before it runs (`kernel.reserved_bindings`). They are named so that
    they do not collide with the engine's own names (`step`, `State`,
    `State.status`) or the model's natural variable names:
    - `recording`: a list with one `StepView` (`helpers.py`) per recorded step:
      `.index`, `.action` (the recorded action: `.id`, `.x`, `.y`, `.name`; no
      `.cell`), `.before` (the previous step's last frame; None for step 0),
      `.after` (= `.last`, the frame the tests compare), `.frames`, `.level`
      (the level it is played in), `.outcome` (`NOT_FINISHED`, `WIN` or
      `GAME_OVER`; `trace.json` keeps it as `state`), `.levels_completed`,
      `.win_levels`, `.available_actions`. Frames only: no State. And the
      frames' segmentation (`segment.py`), computed when first read and kept,
      only ever from the steps loaded (in the stepwise harness, steps 0..k; a
      new focus guesses the grids again): `.grid` (the `GridGuess` of its level,
      from that level's loaded frames), `.pieces_before` (None for step 0) and
      `.pieces_after`, the frames as `Pieces`, and `.changes`, what the step
      changed piece by piece (None for step 0). The kernel starts as fast as
      before: nothing is segmented until it is read.
    - `read_file(path="engine.py", offset=None, limit=None)` prints the file as
      `LINE#HASH:content` lines (`hashline.py`). The hash is 2 characters from
      `ZPMQVRWSNKTXJBYH` over the previous, current and next line (trailing
      whitespace and `\r` removed; xxh32 when installed, else crc32), so an
      anchor goes stale when its line or a neighbour changes. Without `offset`
      the FIXED block is folded; long output says where to continue.
    - `edit_file(path="engine.py", edits=[...])` applies `replace` (`pos`, optional
      `end`), `append` / `prepend` (optional `pos`; none = end / start of the
      file) and `replace_text` (`oldText`, `newText`; one exact unique match).
      `lines` is a list or one string. All edits of a call are checked against
      one snapshot and applied bottom-up. Rejected, with nothing applied: edits
      that overlap or touch adjacent lines, stale anchors (`[E_STALE_ANCHOR]`,
      listing the lines that now hold the content when an anchor carries a
      `:content` suffix), a `:content` suffix that does not match its line, and
      any edit of the FIXED block (`[E_FIXED_BLOCK]`). The answer: what
      changed, a syntax check, and fresh anchors around each change (at most
      about 12 lines; the first and last lines of a long insert).
    - `undo_edit(n=1, to=None)` restores the engine.py of `n` changes ago (`to=k`:
      version k; `to="best"`: `engine_best.py`, and it says by which rule).
      Every change, a restore
      included, is a new numbered version in `<game_dir>/engine_versions/`
      (outside the workspace), so undo after undo brings a change back. It
      prints the last 8 versions with what changed and their test result
      (matched by the engine's hash in `tests.jsonl`) and reminds to read again.
    - `render_state(state)`: the frame the tests draw (the same code).
    - `show_frames(*frames, titles=None, boxes=None)`: up to 4 frames or States side
      by side, upscaled, titled, with numbered boxes `(x0, y0, x1, y1)`. The
      PNG goes to the harness with the call's output; with images off it
      prints a hex crop of the boxes, or the frame at half resolution.
    - `before, after = replay_step(i, state=None, action=None)` loads engine.py
      fresh, replays steps 0..i-1 through the harness rules (or starts from
      `state`), applies step i's action (or `action`), and prints what engine.py
      printed during the step, what the step changed in its state (sprites by
      `#index` and name, matched by identity; vars; status) and, for the
      recorded action, the comparison with the recording as `run_tests`
      explains it. It returns copies of the State before and after.
    - `summarize_levels()` prints one row per level the recording plays: its
      first frame (`recording[k].after`), its guessed grid, the steps played in
      it and their actions, animated steps, RESETs and game overs, and the step
      that solved it.

    A frame's pieces (`segment.py`, a pure module over `auto_sprites.py`).
    `pieces(frame, grid=None, known=None)` splits a frame as the sprite code
    does: a border sprite, a background sprite, one sprite per single-colour
    4-connected region on the guessed logical grid (the segmentation of
    `inference/utils/segmentation.py`), screen sprites for the HUD and pixels
    that break the grid's blocks. Each is a `Piece`, a real fixed-interface
    `Sprite` (grid cells for grid pieces, screen pixels for screen ones) with
    `.shape` (the stable name `SHAPE_<colours>_<w>x<h>_<4 hex>` from its
    content), `.colour` (main colour), `.size` (pixels), `.transform`
    (rotation, mirror_ud, mirror_lr, scale, recolour relative to its shape:
    a piece equal to a known shape, or one met earlier in the frame, as it is,
    turned or mirrored, scaled 2-5x, or recoloured one-to-one, tried in that
    order, at most two changes at once, solid rectangles only as they are or
    turned), `.children` (the pieces it encloses, under the innermost
    encloser only, as in segmentation.py) and `.role` (`border`,
    `background`, `object`). Drawn in order they redraw the frame exactly (all
    34 reference level starts, and every lp85 level start; about 10 ms a frame).
    `Pieces` is a list with `.grid` (the `GridGuess`) and `.code()`: the Python
    that `auto_sprites.sprite_code` writes for the frame, usable as `edit_file`
    lines (a constant per shape, one `Sprite` per piece, in `level_<n>_sprites`),
    which in the kernel reuses engine.py's pixel constants (as they are,
    turned, mirrored, scaled or recoloured) and does not define its names
    again; it replaces the former `auto_sprites()` built-in, as
    `recording[e].pieces_after.code()` with e the step that entered the level.
    `print(pieces)` lists them, one line each. The grid is guessed
    conservatively (33 of the 34 level starts of the reference ports right, and
    every one of their 1,529 recorded frames on its own).
    `changes(before, after)` is the object-level diff of two frames' pieces:
    `moved` (the same pixels elsewhere, nearest first), `recoloured` (same place
    and shape, with the colour map), `reshaped` (overlapping, other pixels or
    size, e.g. "1x31 -> 1x26, lost 5 px at the top"), `appeared`,
    `disappeared`; unchanged pieces are left out, and pieces of two different
    grids (a new level) never match. lp85's step 1 comes out as 18 recoloured
    ring tiles (the other 2 keep their colour), the bar at the left reshaped and
    its lost top as a black piece that appeared. `summary(changes)` groups
    similar changes into at most 12 lines.

    engine.py cannot be written from the kernel any other way: the sandbox
    denies opening it for writing and every operation that could replace it
    (rename, replace, remove, link, copy onto it, also on the directories
    containing it). `edit_file` and `undo_edit` send their arguments to the
    harness over the kernel's protocol, and `engine_files.py` applies them.
    While the analysis quota pauses python, calls that use `edit_file(` or
    `undo_edit(` still run.

    The system prompt of every mode has an `# Objects` section
    (`prompts.objects_reference`), a typed reference written once and shared
    by the modes: the recorded steps python holds in that mode and `StepView`
    with the recorded action; `make_level`, `step` and every field and method
    of `Sprite`, `Action` (what `step()` receives, with `.cell`), `View` and
    `State`, with how `State.status` maps to the recorded outcome; a frame's
    pieces (`Pieces`, `Piece`, `Change`, `GridGuess`); and every built-in
    function's signature, return value and meaning. The python tool's
    description only names the preloaded names and points there. A test
    checks the reference against the classes and functions themselves.
  - `run_tests(level=None, failures=1)`: first the contract tests
    (interface unchanged, valid states, a fresh state on every `make_level`
    call, every advertised action accepted, determinism), then the acceptance
    test, the recording replayed in order: from step 0, or with `level=L` only
    level L (the engine starts at `make_level(L)`, its drawing is compared with
    the level's recorded start, then it plays that level's steps). The report
    stops after `failures` failing steps (1 to 10, clamped; default 1): it says
    how many steps pass before the first failure, explains the first failure
    in full, with the picture, and gives one line per further failure (step,
    action, differing regions with their colour changes, the engine's sprites
    there). A failure explained in full has the differing regions of the final
    frame, numbered and boxed in a picture of both frames (below), the colour
    changes, the engine's sprites drawn in each region before and after the
    step (a second, short sandboxed run collects them), state fields and vars,
    the end of what the engine printed during the step (prints are captured per
    step and capped), and the `replay_step` command that reproduces it. A failing
    contract test does not hide the replay. The counts kept (`tests.jsonl`,
    best engine, pass) always come from the whole replay; the text is what
    stops. The automatic test uses `failures=1`.
  - `commit_engine(message)` submits engine.py: it always runs the tests
    (`failures=1`). `message` is required: what changed and the analysis
    behind each rule (which steps or frames show it), and the guesses left;
    `result.json` keeps the last one (`commit_message`, `commit_calls`). In the
    single mode, when everything passes the session ends; otherwise it returns
    the report (with its picture) and the session goes on. (It replaces the
    earlier `finish(summary)`; transcripts with `finish` records still load.)
- **The stepwise harness, v6** (`stepwise.py`, the stepwise mode of `agent.py`,
  `run_experiment --mode stepwise`, the default). The harness leads one
  conversation from one breaking step to the next. It replays the whole
  recording through engine.py and, at the first step k that fails, starts the
  conversation with "Fix the breaking test: step k" (the step's action and
  level, what the recorded step changed piece by piece, summarised in at most
  about 15 lines, or, for a step that enters a level, that
  `step_to_fix.pieces_after.code()` draws its first frame; the test report with
  its picture, engine.py with the FIXED block folded). The message for each
  next step has the same block. The model sees the recording only up to step k: `recording` holds
  steps 0..k (so does `visible_trace/` on disk), `step_to_fix` is step k
  (`recording[-1]`, the same `StepView`) and `summarize_levels()` lists the
  levels reached; with `--only-step`, python shows only `step_to_fix`. Its tests
  replay steps 0..k. Only `commit_engine(message)` moves on: when `run_tests`
  or the automatic test after an edit shows steps 0..k pass, the report only
  adds that a commit is now possible, so the model can keep refining (e.g. make
  a rule more general). A commit whose tests fail returns the report and
  nothing moves on. A commit whose tests pass is recorded (a `commit` record in
  the transcript, an entry in `advances`), and the harness replays on and adds
  a user message to the same conversation: the commit was accepted, how many
  more steps passed without error, and the next step k' that fails, with its
  report. If engine.py changes after the commit in the same turn, the commit is
  dropped and the model is told to commit again. The kernel keeps its
  variables and `recording` grows to step k' (the same list object). There is
  no limit per step: the run ends
  when the recording passes, or when a budget runs out, the model stops calling
  tools, or a request fails for good. `result.json` adds `"mode":
  "stepwise"`, `step` (the step being fixed), `passing_prefix` (of the last
  replay) and `advances` (per accepted commit: the turn, the step fixed, the
  next failing step or null, the message, the engine's sha256 and version);
  transcript records carry `step`. Every step report also says, for a
  click, which grid cell it lands on (the `action.cell` that `step()` gets) and
  which of the engine's sprites are there (the one
  `state.sprite_at(*action.cell)` returns marked). `--mode
  single` runs v5.
- **The opening** (`agent.py`). Before the first turn of a new session the
  harness plays the first round itself: in the kernel,
  `recording[0].pieces_after.code()` (through the private `helpers._level_code`,
  not a built-in, which also prints a summary) makes sprite code for level 0's
  first frame and one `edit_file()` puts it above
  `make_level`, which then returns `level_0_sprites()` (for every level, until
  the model adds more); then it runs the tests. The first message gives the
  recording in one sentence (steps, levels, how it ends, the actions the game
  accepts; `summarize_levels()` has the per-level detail), that summary (not
  the code), the test report with its picture, engine.py as
  `read_file()` shows it (FIXED block folded), and the first task: make the first
  failing step pass (usually step 1, the first action of level 0), then the
  next one, in recorded order. The harness's edit and test are logged
  (`"by": "harness"` in the transcript, `"auto": "opening"` in `tests.jsonl`,
  `result.json` `opening`) but not counted as the model's `engine_changes` or
  `tests_run`. `run_experiment --no-opening` leaves the template as it is and
  asks the model to do that round.
- **Feedback the harness adds** (`agent.py`).
  The model's reasoning is sent back with its turns, as the main harness does
  on OpenRouter; compaction trims old tool outputs and all but the last 10
  turns' reasoning once the prompt passes 140K tokens. When a turn changes
  `engine.py` without testing it, `run_tests()` runs automatically with its
  defaults and its report is appended to the turn's output. Images (on by
  default, `--no-images` for text-only models): after the turn's tool messages
  (which stay strings) one user message carries the turn's pictures: what
  `show_frames()` made, then the latest test's picture (the engine's final frame and
  the original's side by side, upscaled 8x, the differing regions boxed in cyan
  and numbered as in the text). When a newer such message is added, the older
  ones' images are replaced by a placeholder. The PNGs are saved in `images/`
  and the transcript logs their paths. Without images the report shows
  hex-digit crops of the regions instead. After every 30 turns without a test,
  a reminder to edit and test is appended. These came from pilot sessions; see
  [RESULTS.md](RESULTS.md).
- **Passing**. A step matches when its last frame and the state, levels
  completed, win levels and available actions all match; animation frames are
  not compared, so the agent spends nothing on them (it can still look at them
  to understand an action). `--match all` in `tester.py` and `evaluate.py` also
  requires the frame count and every animation frame to match, which is how
  the v2 and v3 runs were scored. The session stops when a full replay matches
  every step (with every contract test passing), whether from `run_tests`, an
  automatic test or `commit_engine` (stepwise: from a commit only), or when a
  budget (turns, output tokens, cost,
  wall time) runs out. Only full replays count towards passing and
  `engine_best.py`, with their counts over the whole recording whatever the
  report shows; one-level tests do not. `engine_best.py` (and
  `undo_edit(to="best")`) is the engine with the most steps passing before the first
  failure, ties broken by the most steps passing in all
  (`engine_files.best_key`), since the agent works through the recording in
  order. The authoritative final test is a full replay with the full report
  (every failing step listed, `final_test.txt`). An interrupted stepwise
  run continues its own conversation, rebuilt from `transcript.jsonl`, which
  logs everything the model is sent: system and user messages as sent (images
  by their saved PNG), assistant turns, tool outputs, the text the harness
  adds to an output (`append`) and the points where old images are hidden
  (`hide_images`) and old turns shortened (`compact`). A note says the run
  resumed and the python kernel restarted, so its variables are gone. A
  transcript from before these records is rebuilt from what it has and
  written back in full, so it is exact from then on. A single-mode session resumes from its
  `engine.py` (shown with anchors, the FIXED block folded) with a fresh
  conversation that carries its last test report and reasoning.
- **Sandbox** (`guard.py`). The kernel and the candidate run in subprocesses
  with an audit hook: reads only under the Python installation, system
  directories, the workspace and (kernel only) the trace; writes only to the
  workspace; no subprocesses or network; no API keys in the environment. The
  candidate process gets only the actions and can read no files after loading
  `engine.py`, so it cannot look up the answers. `evaluate.py` also scans the
  final engine for patterns that could bypass this (frame inspection, `gc`,
  `ctypes`, file access).
- **Correctness beyond the recording** (`evaluate.py`). Passing the recording
  does not prove the rules are right: unplayed actions may behave differently.
  For each level the recording reached, both engines start at `set_level(L)` and
  play the same random action sequences (keys from the advertised actions;
  clicks aimed mostly at object pixels; occasional RESET; RESET after a game
  over), and the steps are compared exactly as in the tests.

## Run it

From `ARC3-Inference/`, with `OPENROUTER_API_KEY` set and the game files in
`environment_files/` (see `LOCAL_EVAL.md`):

```bash
# 1. A harness run to learn from, e.g. the archived one (the bucket allows anonymous reads):
dvc remote modify --local storage allow_anonymous_login true
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY dvc pull runs/20261004_135539.dvc
# or a new one:
make interactive CONFIG_PATH=configs/inference.openrouter.json MODEL=qwen/qwen3.8-flash \
  GAME=ls20,ft09,vc33,sp80,lp85 GAME_TAGS=[] N_PASSES=1 ENVIRONMENTS_DIR=environment_files \
  EXPERIMENTS_DIR=runs MAX_GENERATED_TOKENS_PER_GAME=150000 MAX_RUNTIME_MINUTES=60

# 2. One agent per game, in parallel
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/<run> \
  --games ls20,ft09,vc33,sp80,lp85 --out runs/engine-re/<name> \
  --model qwen/qwen3.8-flash --max-turns 250 --max-cost 5 --max-minutes 150

# 3. Check the engines against the real ones on new action sequences
uv run --no-sync python -m engine_re.evaluate runs/engine-re/<name> --engine best

# Inspect a session
uv run --no-sync python -m engine_re.show_transcript runs/engine-re/<name>/<game>

# Test an engine by hand, as the agent sees it (images saved as PNGs)
uv run --no-sync python -m engine_re.tester ENGINE.py TRACE_DIR --failures 1 [--level L] [--images DIR]
```

OpenRouter requests use the main harness's retry (`inference.agent.tool_agent._post_with_retries`):
rate limits (HTTP 429), gateway errors and failed connections wait and retry without limit
(`ARC3_HTTP_RETRIES`, default -1 here as in `params.yaml`; `ARC3_HTTP_RETRY_BASE_SECONDS` and
`ARC3_HTTP_RETRY_MAX_SECONDS` set the waits). Answers the provider ends with `finish_reason: error`, and
reads that stall, are asked again up to 20 times.

Sampling: `--temperature` (default 0.7), `--top-p` (0.95), `--top-k` (not sent by default) and
`--reasoning-effort` (OpenRouter's `reasoning.effort`, e.g. `low` or `medium`; by default only
`reasoning.enabled` is sent and the provider picks the effort). Qwen's card for its 3.8 Flash models
recommends temperature 1.0, top_p 0.95, top_k 20 with thinking on. The values used are in the run's
`config.json`.

`run_experiment --no-images` gives text-only feedback (test reports and `show_frames()` print hex
digits), for models without image input.

Each game directory holds `trace/`, `workspace/engine.py` (final),
`engine_best.py` (the best full replay, as above),
`engine_versions/` (`vNNNN.py`, one per change of engine.py, and
`versions.jsonl`), `transcript.jsonl` (turns, tool outputs, one record per
edit or undo with its line range and diff, and one per `show_frames()` with its image
paths; `show_transcript --diffs` prints the diffs), `tests.jsonl`,
`result.json` (status, turns, tokens, cost, best and final test, commit calls and message,
engine changes), `images/` (the pictures sent to the model,
`turn<N>_step<S>[_auto|_opening].png` and `turn<N>_show<K>.png`) and
`evaluation_<engine>.json`. In `tests.jsonl` a full replay has `"level": null`
and `total` equal to the trace length; a one-level test has its `level`, and
`from_level` (the level the engine started at) as before; `first_fail` is the
first failing step's index, `passing_prefix` the number of steps passing
before it (the whole scope when none fails; for a full replay the two are
equal while a step fails), `failures` what the report was asked to show, and
`engine_sha` the hash of the engine tested, which `undo_edit` uses to show each
version's result. The experiment directory holds `summary.md` and
`evaluation_<engine>.md`.

Tests: `uv run --no-sync pytest tests/test_engine_re.py`.
