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
   │                             │ read / edit / undo ──► engine_files.py ──► workspace/engine.py
   │                             │                        (the only writer; versions in engine_versions/)
   │     ── run_tests ───────► tester.py ──► candidate_runner.py (sandboxed; gets only the actions)
   │     ── finish ──────────► tester.py (the session ends only when every test passes)
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
  with the original, ignoring comments. The tester and `evaluate.py` still
  score engines written as an `arcengine` game class (the earlier runs).
- **Tools** (`agent.py`, `prompts.py`): exactly three.
  - `python(code)`: a persistent kernel. Its namespace holds `np`, the
    fixed-block classes, `S` (the recording's steps) and eight functions
    (`helpers.py`); nothing else is preloaded:
    - `read(path="engine.py", offset=None, limit=None)` prints the file as
      `LINE#HASH:content` lines (`hashline.py`). The hash is 2 characters from
      `ZPMQVRWSNKTXJBYH` over the previous, current and next line (trailing
      whitespace and `\r` removed; xxh32 when installed, else crc32), so an
      anchor goes stale when its line or a neighbour changes. Without `offset`
      the FIXED block is folded; long output says where to continue.
    - `edit(path="engine.py", edits=[...])` applies `replace` (`pos`, optional
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
    - `undo(n=1, to=None)` restores the engine.py of `n` changes ago (`to=k`:
      version k; `to="best"`: `engine_best.py`, and it says by which rule).
      Every change, a restore
      included, is a new numbered version in `<game_dir>/engine_versions/`
      (outside the workspace), so undo after undo brings a change back. It
      prints the last 8 versions with what changed and their test result
      (matched by the engine's hash in `tests.jsonl`) and reminds to read again.
    - `render(state)`: the frame the tests draw (the same code).
    - `show(*frames, titles=None, boxes=None)`: up to 4 frames or States side
      by side, upscaled, titled, with numbered boxes `(x0, y0, x1, y1)`. The
      PNG goes to the harness with the call's output; with images off it
      prints a hex crop of the boxes, or the frame at half resolution.
    - `before, after = try_step(i, state=None, action=None)` loads engine.py
      fresh, replays steps 0..i-1 through the harness rules (or starts from
      `state`), applies step i's action (or `action`), and prints what engine.py
      printed during the step, what the step changed in its state (sprites by
      `#index` and name, matched by identity; vars; status) and, for the
      recorded action, the comparison with the recording as `run_tests`
      explains it. It returns copies of the State before and after.
    - `auto_sprites(level=0, grid=None, frame=None, region=None, merge=False)`
      returns code (a str usable as `edit` lines) for a sprite list that redraws
      a level's recorded start, a given frame, or a region of it exactly
      (`auto_sprites.py`): border and background sprites, one sprite per
      single-colour 4-connected region (`merge=True`: per group of touching
      regions), identical objects sharing a constant named from its content
      (`SHAPE_<colours>_<w>x<h>_<4 hex>`), screen sprites for the HUD. A piece
      that is an existing pixel constant (one of engine.py's module-level
      constants, hex strings or rows of numbers, or one made earlier in the
      call) is drawn from it rather than written again: as it is, turned or
      mirrored (`rotation`, `mirror_ud`, `mirror_lr`), scaled 2-5x, or
      recoloured one-to-one (`shape_pixels(NAME, {old: new})`), tried in that
      order, with at most two of these changes at once and solid one-colour
      rectangles only as they are or turned. It prints a summary such as "14
      pieces: 12 reuse existing kinds (6 as they are, 5 turned, 1
      recoloured; 6 kinds from engine.py), 2 new kinds". Run level after
      level on the reference ports, the later levels draw many pieces from
      the earlier levels' kinds (vc33's levels, each drawn turned by 0 to 270
      degrees, as turned copies), and all 34 level starts are still drawn
      exactly. It guesses the logical grid
      conservatively (33 of the 34 level starts of the reference ports right,
      and every one of their 1,529 recorded frames on its own), runs the code
      and prints whether it renders the frame exactly. A starting point, not
      the real sprites.
    - `summarize_levels()` prints one row per level the recording plays: its
      first frame (`S[k].last`), the steps played in it and their actions,
      animated steps, RESETs and game overs, and the step that solved it.

    engine.py cannot be written from the kernel any other way: the sandbox
    denies opening it for writing and every operation that could replace it
    (rename, replace, remove, link, copy onto it, also on the directories
    containing it). `edit` and `undo` send their arguments to the
    harness over the kernel's protocol, and `engine_files.py` applies them.
    While the analysis quota pauses python, calls that use `edit(` or `undo(`
    still run.
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
    step and capped), and the `try_step` command that reproduces it. A failing
    contract test does not hide the replay. The counts kept (`tests.jsonl`,
    best engine, pass) always come from the whole replay; the text is what
    stops. The automatic test uses `failures=1`.
  - `finish(summary)` always runs the tests (`failures=1`). When
    everything passes the session ends; otherwise it returns the report (with
    its picture) and the session goes on.
- **The stepwise harness, v6** (`stepwise.py`, the stepwise mode of `agent.py`,
  `run_experiment --mode stepwise`, the default). The harness leads one
  conversation from one breaking step to the next. It replays the whole
  recording through engine.py and, at the first step k that fails, starts the
  conversation with "Fix the breaking test: step k" (the step's action and
  level, the test report with its picture, engine.py with the FIXED block
  folded). The model sees the recording only up to step k: `S` holds steps
  0..k (so does `visible_trace/` on disk), `step` is step k (`.before`,
  `.action`, `.after`, `.frames`, `.level`) and `summarize_levels()` lists the
  levels reached; with `--only-step`, python shows only `step`. Its tests
  replay steps 0..k. As soon as they pass (by `finish`, `run_tests` or the
  automatic test after an edit), the harness replays on and adds a user message
  to the same conversation: steps 0..k pass, how many more steps passed without
  error, and the next step k' that fails, with its report. The kernel keeps its
  variables and `S` grows to step k'. There is no limit per step: the run ends
  when the recording passes, or when a budget runs out, the model stops calling
  tools, or a request fails for good. `result.json` adds `"mode":
  "stepwise"`, `step` (the step being fixed), `passing_prefix` (of the last
  replay) and `advances` (per step fixed: the turn, the step, the next failing
  step); transcript records carry `step`. Every step report also says, for a
  click, which grid cell it lands on and which of the engine's sprites are
  there (the one `state.sprite_at(*action.cell)` returns marked). `--mode
  single` runs v5.
- **The opening** (`agent.py`). Before the first turn of a new session the
  harness plays the first round itself: in the kernel, `auto_sprites(0)` makes
  sprite code for level 0's first frame and one `edit()` puts it above
  `make_level`, which then returns `level_0_sprites()` (for every level, until
  the model adds more); then it runs the tests. The first message gives the
  recording in one sentence (steps, levels, how it ends, the actions the game
  accepts; `summarize_levels()` has the per-level detail), what `auto_sprites`
  printed (not its code), the test report with its picture, engine.py as
  `read()` shows it (FIXED block folded), and the first task: make the first
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
  `show()` made, then the latest test's picture (the engine's final frame and
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
  automatic test or `finish`, or when a budget (turns, output tokens, cost,
  wall time) runs out. Only full replays count towards passing and
  `engine_best.py`, with their counts over the whole recording whatever the
  report shows; one-level tests do not. `engine_best.py` (and
  `undo(to="best")`) is the engine with the most steps passing before the first
  failure, ties broken by the most steps passing in all
  (`engine_files.best_key`), since the agent works through the recording in
  order. The authoritative final test is a full replay with the full report
  (every failing step listed, `final_test.txt`). An interrupted session
  resumes from its `engine.py` (shown with anchors, the FIXED block folded)
  with a fresh conversation that carries its last test report and reasoning.
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

`run_experiment --no-images` gives text-only feedback (test reports and `show()` print hex
digits), for models without image input.

Each game directory holds `trace/`, `workspace/engine.py` (final),
`engine_best.py` (the best full replay, as above),
`engine_versions/` (`vNNNN.py`, one per change of engine.py, and
`versions.jsonl`), `transcript.jsonl` (turns, tool outputs, one record per
edit or undo with its line range and diff, and one per `show()` with its image
paths; `show_transcript --diffs` prints the diffs), `tests.jsonl`,
`result.json` (status, turns, tokens, cost, best and final test, finish calls,
engine changes), `images/` (the pictures sent to the model,
`turn<N>_step<S>[_auto|_opening].png` and `turn<N>_show<K>.png`) and
`evaluation_<engine>.json`. In `tests.jsonl` a full replay has `"level": null`
and `total` equal to the trace length; a one-level test has its `level`, and
`from_level` (the level the engine started at) as before; `first_fail` is the
first failing step's index, `passing_prefix` the number of steps passing
before it (the whole scope when none fails; for a full replay the two are
equal while a step fails), `failures` what the report was asked to show, and
`engine_sha` the hash of the engine tested, which `undo` uses to show each
version's result. The experiment directory holds `summary.md` and
`evaluation_<engine>.md`.

Tests: `uv run --no-sync pytest tests/test_engine_re.py`.
