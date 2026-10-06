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
  - `python(code)`: a kernel that is persistent for the whole run (the prompt
    says so plainly: define helpers and data once). Its namespace holds `np`,
    the fixed-block classes, `recording` (the recorded steps), seven functions
    (`helpers.py`) and `replica`; nothing else is preloaded. These names are
    reserved: code that binds one (`def`, assignment, parameter, loop
    variable, import as) is refused before it runs
    (`kernel.reserved_bindings`). They are named so that they do not collide
    with the engine's own names (`step`, `State`, `State.status`) or the
    model's natural variable names:
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
      `LINE#HASH:content` lines (`hashline.py`). The hash is 3 characters from
      `ZPMQVRWSNKTXJBYH` over the previous, current and next line (trailing
      whitespace and `\r` removed; xxh32 when installed, else crc32), so an
      anchor goes stale when its line or a neighbour changes (anchors with 2
      to 4 characters are parsed; a shorter one is the end of the hash). Without
      `offset` the FIXED block is folded; long output says where to continue.
    - `edit_file(path="engine.py", edits=[...])` applies `replace` (`pos`, optional
      `end`), `append` / `prepend` (optional `pos`; none = end / start of the
      file), `replace_text` (`oldText`, `newText`: one exact match, else whole
      lines that match ignoring whitespace, with `newText` used as given and a
      warning) and `replace_def` (`name`, `lines`: the whole top-level `def`,
      `class` or `NAME = ...` found with `ast`, decorators included; `"Game.step"`
      for a method; a name not defined is added at the end of the file, or of
      the class, with a warning). `lines` is a list or one string. All edits of
      a call are checked against one snapshot; the valid ones are applied
      bottom-up and the others reported one by one, after "applied N of M
      edits", each with the lines as they are now (one line of context, fresh
      anchors) and the lines that hold the content it gave: a stale anchor
      (`[E_STALE_ANCHOR]`; one whose `:content` suffix still matches its line is
      accepted), a line that does not exist, a text with no match (up to 6
      lines like its first significant line) or several, an edit of the FIXED
      block (`[E_FIXED_BLOCK]`), and two edits that change the same line
      (`[E_EDIT_CONFLICT]`, both refused; edits on adjacent lines are merged in
      order). Only a malformed request (`[E_BAD_OP]`, a bad anchor string,
      `read_file()` output given as lines) applies nothing. The answer: what
      changed, a syntax check, and fresh anchors around each change (the whole
      region when it has at most 60 lines; else its first and last lines).
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
    - `replica`: engine.py, the model's replica of the game, as it is now
      (`helpers._ReplicaModule`; named so that it is not taken for the real
      game's engine, arcengine): an attribute access loads the file again when
      its content hash changed since the last load, so `replica.step(...)` and
      `replica.make_level(...)` never go stale after an edit; `replica.step`
      fills in a click's `action.cell` when it is None, as the harness does.
      Every `import engine` / `from engine import ...` / `import engine as e`
      is refused by the reserved-name check with a note saying to use the
      built-in. The file stays `engine.py` (and `--engine` stays the flag of
      `evaluate.py` and the tester).

    A tool call named after one of these functions (the model calling
    `read_file` or `edit_file` as if it were a tool) runs through the python
    tool as `name(**args)`, a string argument that parses as JSON (an `edits`
    list given as text) parsed first; the output starts with one line saying
    so, the call counts as a python call and the transcript keeps the name the
    model used (`called_as`). An unknown name still gets the unknown-tool error.

    The kernel answers two more requests (`kernel.py`, `KernelClient`):
    `{"names": true}` lists what the model defined (everything in the
    namespace that is not a preloaded built-in, a module or a dunder) with a
    one-word summary each (`RING: list[20], cols: function, f0: ndarray(64,
    64)`; 40 at most, then "... and N more"); the line "Your python kernel
    keeps: ..." goes into every next-step message and the resume note, never
    into tool outputs. `{"replay": [cells]}` re-runs earlier python cells in
    order in a replay mode (`edit_file`/`undo_edit` do nothing, `show_frames`
    makes no image, output discarded, an exception ends only its cell, 20 s
    per cell by SIGALRM and 120 s in all), which is how a resumed run gets its
    variables back.

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

    The system prompt asks for parsimony, not generality: make the tests pass
    with the most parsimonious model (Occam's razor), the fewest rules and
    assumptions that account for every step observed so far; per-level
    constants in the level data are fine when the steps give no evidence of a
    formula; when a step contradicts a rule, replace it with the simplest rule
    that explains all the steps so far; do not model what has not been
    observed (a new level: only draw its first frame, model its mechanics when
    one of its steps fails); never hard-code recorded frames or anything keyed
    to the step number. It no longer says the engine is later played on
    unrecorded action sequences (`evaluate.py` still does that, as a check
    outside the prompt). A short `# Sandbox` section says what `guard.py`
    enforces.

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
    the level's recorded start, then it plays that level's steps). A final
    frame that differs by a single pixel within 2 px of the frame's edge still
    passes, with a warning (`tester.HUD_BORDER`): that is a HUD bar's rounding,
    and the prompt tells the model to model such bars as a per-level budget
    drawn proportionally and not to chase the pixel. The report
    stops after `failures` failing steps (1 to 10, clamped; default 1): it says
    how many steps pass before the first failure, explains the first failure
    in full, with the picture, and gives one line per further failure (step,
    action, differing regions with their colour changes, the engine's sprites
    there). A failure explained in full has the differing regions of the final
    frame, numbered and boxed in a picture of both frames (below), the colour
    changes, the engine's sprites drawn in each region before and after the
    step (a second, short sandboxed run collects them), state fields and vars,
    the end of what the engine printed during the step (prints are captured per
    step and capped), and, in one line, the `replay_step` command that
    reproduces it ("Reproduce: before, after = replay_step(N)"; `replay_step`
    itself prints one line when the frame equals the recorded one). Every
    report carries a failure `signature` (the first failing step, the exact
    count, the error, the contract results and the first failure's differing
    regions; `tests.jsonl` keeps it). A failing
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
  its picture, engine.py with the FIXED block folded under one fixed header).
  The message for each next step has the same block, engine.py listing
  included, and the line "Your python kernel keeps: ...". Compaction (and the
  rebuild of a resumed conversation) replaces every engine.py listing but the
  latest by a note that `read_file()` shows the current file. The model sees
  the recording only up to step k: `recording` holds
  steps 0..k (so does `visible_trace/` on disk), `step_to_fix` is step k
  (`recording[-1]`, the same `StepView`) and `summarize_levels()` lists the
  levels reached; with `--only-step`, python shows only `step_to_fix`. Its tests
  replay steps 0..k. Only `commit_engine(message)` moves on: when `run_tests`
  or the automatic test after an edit shows steps 0..k pass, the report only
  adds that a commit is now possible, so the model can keep refining. A
  commit whose tests fail returns the report and
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
- **Context: two schemes** (`agent.py`, `condense.py`). Both act at the
  same moments: after a turn whose request went over 140K prompt tokens
  (`compact_prompt_tokens`); in between nothing sent before is rewritten
  (but for the previous images, below), so the prompt's prefix stays the same
  from one request to the next and the provider's prompt cache hits. By
  default (`ModelConfig.context = "compact"`) the conversation is shortened in
  place by age: old tool outputs to 200 characters, all but the last 10
  turns' reasoning to their last 1,200 characters, long old tool-call
  arguments, and every engine.py listing but the latest (`compact` and
  `hide_images` records mark where). With `run_experiment --condense`
  (`context = "condense"`; `config.json` `condense`, `result.json` `context`)
  the agent keeps the full conversation and, when the threshold is crossed,
  condenses all of it once, by iteration (`engine_re/condense.py`): finished
  iterations older than the last three become their failing-step message, the
  net diff of engine.py and the commit; the last three keep every successful
  tool call with its result; the current iteration keeps its real messages,
  with the reasoning and the failed commands stripped from the turns older
  than the last ten (`--condense-keep-turns`, `ModelConfig.condense_keep_turns`)
  and only its latest images live; a safety cap (an estimate at
  `condense_chars_per_token`, 3 by default, against 140K) cuts old results,
  then reduces blocks. That condensed view is the prefix of every request until
  the condenser fires again; the messages added since follow it as they are,
  with only the latest message with images keeping them (the view's own latest
  images stay too, so the view does not change). Each firing condenses the full
  conversation again, never an earlier view, and logs one `condense` record
  (estimated tokens, characters, live images, messages, what the cap cut).
  Nothing is shortened in place, so the transcript rebuilds the full
  conversation, and a resumed run condenses it again at its last `condense`
  record (the conversation and the records up to it), so it sends the prefix
  the uninterrupted run would have sent. The two runs made with the earlier
  per-turn condenser (a `condense` record before every request) resume with
  their last record taken as a firing, not as they ran. Note that the cap is
  not below the trigger: a condensed view near the cap is still over 140K real
  tokens, and the condenser then fires after every turn, as `_compact` does
  in the same case. `engine_re/condense_report.py` compares the schemes on
  finished runs, turn by turn (`--schemes`: compaction, the per-turn condenser
  and the threshold one, with the prompt tokens, the firings and the share of
  each prompt that repeats the previous one).
- **Feedback the harness adds** (`agent.py`).
  The model's reasoning is sent back with its turns, as the main harness does
  on OpenRouter; compaction trims old tool outputs and all but the last 10
  turns' reasoning once the prompt passes 140K tokens. When a turn changes
  `engine.py` without testing it, `run_tests()` runs automatically with its
  defaults and its report is appended to the turn's output; when that report
  fails the same way as the last test (the same `signature`), one line says so
  ("tested automatically: the same result as the last test (step N fails the
  same way: ...)") with the commit hint, and `tests.jsonl` and the transcript
  keep the full report. Images (on by
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
  (`hide_images`), old turns shortened (`compact`) and the condenser fired
  (`condense`). The kernel restarts
  empty, so the harness re-runs every python cell of that conversation (the
  `python` calls, and the built-ins called as tools) in order in the kernel's
  replay mode, and a note says the run resumed, that the kernel re-ran the N
  cells with file edits disabled so the variables and functions are back,
  which cells raised when re-run (by turn: as before, or because engine.py
  changed later), and what the kernel keeps; the transcript logs the replay
  (`replay`: counts, failed turns, seconds). A transcript from before these
  records is rebuilt from what it has and written back in full, so it is
  exact from then on. A single-mode session resumes from its
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

## The play-and-model agent (v10, `play_agent.py`)

The same agent playing a live game instead of fitting a recording: one conversation that alternates a
plan round (the game's current frame and the actions it accepts; in python `state_now()`, the replica's
state after everything played, on copies of which the model plays moves by calling `replica.step`
directly; then `commit_moves(actions, note)` with the very Actions it stepped its replica with, as python
prints them) and the stepwise fit round above. The play prompts call engine.py "your replica" and the real
game "the game". The kernel's `Action` (the fixed block's, `game_api.canonical`) prints as the code that
builds it, its cell left out: `Action(4)`, `Action(6, x=12, y=40)`, `Action(0)` for RESET
(`trace.action_code`). `commit_moves` takes that text as it is, item by item (`["Action(4)", "Action(6,
x=12, y=40)"]`) or as one printed list (`"[Action(4), Action(0)]"`), besides the labels (`"UP"`,
`{"click": [x, y]}`, `MOUSE(row=, col=)`; `trace.parse_moves`); a given cell is ignored, the harness
computes it. The play messages name moves in the same form (`#12 Action(4): matches your prediction`).
The repr is set on the harness side: the FIXED block is unchanged, and an engine's own `Action` keeps the
dataclass repr (`Action(id=4, x=0, y=0, cell=None)`), which `commit_moves` takes too. `commit_moves` runs the full test first
and sends nothing while a step fails (a fit round opens on it); otherwise engine.py becomes the committed
engine (`engine_committed.py`; a `commit_engine` earlier in the same turn is the batch's commit), each move
is predicted with it in one sandboxed run (`tester.predict`), sent to the real game
(`live_game.LiveGame`, the arcengine game stepped directly, every step kept in a growing `Trace`) and
compared by the tests' rule (`tester.check_step`; a one-pixel HUD-bar difference is a match, its warning
shown); the batch stops at the first difference (a fit round opens on that step, with the comparison as
the test report), after a solved level and when the game ends. One batch per turn; a batch is cut to the
actions left in `--max-actions`. After a game over the harness RESETs the level itself (checked like any
move; `--no-auto-reset` leaves it to the model). A commit whose engine passes the fit round's step but
fails a later one gets that step (`advance_message`), as in the stepwise harness; engine.py edited after
a batch in the same turn is tested automatically and the next message says whether it still reproduces
every step. After `--plan-turns` turns (6) of a plan round without `commit_moves` a reminder to send a
short batch is appended to the turn's last output (`plan_nudge`); the test nudge only runs in fit rounds.

The escape hatch (`--fit-turns N`, off by default; PLAY_DESIGN.md 3.6): after N turns in one fit round
without an accepted commit, the model is told it may play on with its replica out of step. `commit_moves`
then sends moves although the tests fail, unchecked; the steps from the failing one on are unexplained
(`ignore`: the tests replay them, an error there does not stop the replay, they never fail and are not
counted in `exact`, `total` or `passing_prefix`), up to the first RESET or level change the game makes, a
resync point (`resync`: `game_api.GameRunner.resync` puts the engine at that level's start, performing
the RESET, or showing the new level without calling `step()`); there every step is tested again and the
loop goes on. Both live in the trace's meta, so the tests (`tester.replay_test`, `candidate_runner
--ignore/--resync`), the kernel (`state_now`, `replay_step`) and a resumed run see them;
`result.json` lists them (`unexplained`, `resync`, `out_of_sync`) and the PLAN message names them.

Support (PLAY_DESIGN.md 3.11, `support.py`): the runner compiles the engine with its conditions wrapped
in a recorder (line numbers kept) and traces, per step, the lines of the model's part it executed and
the conditions it evaluated. A full replay folds its passing steps into a support map (per line, how many
recorded steps ran it: 0 untested, fewer than 3 thin; per `and`/`or`, whether the steps separated its
operands), saved beside the committed engine (`engine_committed.support.json`) and summarised in each
`tests.jsonl` record. Each planned move's path is read against it from the prediction run itself: its
weakest line, the untested lines it runs, the never-separated conditions it relies on, in the
`commit_moves` output, `batch_log`'s `support`, the mismatch message and the fit report; the PLAN message
lists the thin rules on the last batch's path and among the outcome rules; listings and `read_file()`
show the counts in a margin; `traced()`/`support()` read what python ran on the replica.
`--cut-untested` (off) cuts a batch after the first move that runs untested code.
`engine_re/tools/support_check.py` measures the mismatch rate by support on an archived run.

`run_play.py` runs several games in parallel and writes `summary.md`, a TAAF-shaped `benchmark.json`
(`make score_run SCORE_RUN_DIR=<out>` scores it; `final_score` is TAAF's formula, 0 without baselines)
and, per game, `trace/`, the viewer event sidecar `artifacts/<game_id>_p0_events.jsonl` (the base
harness's name; `trace.trace_from_run(<out>/<game>, game, environment_files)` rebuilds and verifies the
trace from it) and `result.json` with the play fields (`PlayResult`: score, actions per level, batches
with their one-line differences, mismatches, fit rounds with their length and outcome, turns per phase,
nudges, unexplained steps). Every transcript record carries its `phase`; `show_transcript` prints the
moves, batches and phase messages. Running the command again resumes interrupted games: the real game
is replayed from `trace/`, the conversation rebuilt from `transcript.jsonl`, and moves played after the
last message the model got (an interruption during a batch) are tested and lead to the next message;
finished games are skipped but still give their `benchmark.json` record. The play mode keeps compaction
(`ModelConfig.context = "compact"`); a PLAN message's engine.py listing is elided like a fit message's.

Guidance ported from the base harness's prompt (PLAY_DESIGN.md 3.11): the play system prompt has the colour
legend and the actions' meanings in # Setup, the animation sentences in # Tests, and plan rules 5-9 (the game
is solvable, levels build on earlier mechanics, no player assumed and no absolute-coordinate goals, prefer
code over reasoning, `notes.md`). An animated step has a digest (`engine_re/animation.py`,
`StepView.animation`): its transient cells (changed and changed back, so in no frame the model can otherwise
reach) and a diff timeline of its frames, printed in two lines by the test report and the step messages, and
named in `commit_moves`' output for a matched move. `commit_moves` warns, without refusing, when the replica
predicts a game over or moves that change nothing (`batch_log[i]["warnings"]`). Each PLAN message shows
`notes.md` (the model's goal model, open questions and plan; 40 lines at most) and, after a solved level, the
base's level-start paragraph with the new board's unfamiliar pieces (shapes the previous level never showed).

```bash
uv run --no-sync python -m engine_re.run_play --games sp80,ls20,ft09 --out runs/engine-play/<name> \
  --model qwen/qwen3.8-flash --max-turns 300 --max-minutes 240 --max-cost 6 --max-actions 500 --batch-size 10
```

Tests: `uv run --no-sync pytest tests/test_play.py` (a scripted model on a two-level key game and a click game).

The first run (sp80, ls20 and ft09, qwen3.8-flash, `runs/engine-play/qwen38flash-v10`) is written up in
[exp/v10-play.md](../exp/v10-play.md): ft09 won in 77 actions, sp80 reached 2 of 6 levels, ls20 1 of 7.

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

Thinking budget: `--thinking-budget N` (`ModelConfig.thinking_budget`, None by default: no limit) sends
OpenRouter's `reasoning: {"max_tokens": N}`, at most N thinking tokens per answer, then the answer as usual.
It cannot be combined with `--reasoning-effort` (OpenRouter takes one or the other; `run_experiment` exits
with an error, `ModelConfig` raises). It is kept in `config.json` (`thinking_budget`) and `result.json`
(`thinking_budget`). When a budget is set, the agent reads OpenRouter's public model listing once at start
and prints a one-line warning when the model's `reasoning` settings do not say `supports_max_tokens: true`
(models that only list efforts): the budget may then be ignored or mapped to an effort level. The run goes
on either way, and a listing that cannot be read is ignored.

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

Run outputs are not kept in git: each run directory, and the per-version
artifacts RESULTS.md cites (`runs/engine-re/results-*`), is archived in DVC
(see [exp/README.md](../exp/README.md#conventions)). The two trace-viewer pages
are built from a pulled run by `tools/ls20_trace/build.py` and
`tools/v4_transcripts/build.py`; their docstrings give the inputs, the output
directory and the commit to run them at.
