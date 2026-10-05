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
   │     ── view/write/edit ─► workspace/engine.py
   │     ── run_tests ───────► tester.py ──► candidate_runner.py (sandboxed; gets only the actions)
   ▼
result.json, transcript.jsonl, tests.jsonl, engine_best.py
   │
   ▼
evaluate.py: candidate vs real engine on new random action sequences per level
```

- **Trace** (`trace.py`). The games are deterministic, so replaying a run's
  logged actions through the real engine reproduces the run's observations
  exactly; the replay is checked against the board the run logged after every
  action. The replay also recovers what the logs drop: every animation frame.
  RESET restarts the current level, as in the harness (`ONLY_RESET_LEVELS=true`).
- **Interface** (`game_api.py`, `skeleton.py`, `game_notes.md`; `--interface`).
  By default (`simple`) engine.py is a plain module: a fixed block defining
  `Sprite`, `Action` and `State` (plus pixel helpers and the drawing rules),
  which the agent must not edit, and two functions the agent writes:
  `make_level(n) -> State` (grid size, all sprites including border,
  background and HUD, and hidden `vars`) and `step(state, action)`, which
  changes the state in place and sets `state.status` to `"level_solved"` or
  `"game_over"`. The harness renders states (identical to arcengine's camera
  on all 34 level starts of the five games), turns clicks into grid cells,
  and applies arcengine's episode rules (RESET restarts the level, level
  changes, WIN, GAME_OVER) in `GameRunner`, so the tester, sandbox and
  evaluation work unchanged. `--interface arcengine` keeps the earlier
  setup: an `ARCBaseGame` subclass with `api_notes.md` in the prompt.
- **Tools** (`agent.py`, `prompts.py`):
  - `python`: a persistent kernel with the trace loaded as `trace` / `S` and
    analysis helpers (`helpers.py`: summaries, hex region views, frame diffs,
    animation diffs, connected components, logical-grid detection, and
    running the engine in-process with `new_game` / `replay` / `compare`).
    Two helpers work with the tests:
    - `before, after = try_step(i, state=None, action=None, level=None)` loads
      `engine.py` fresh, replays steps 0..i-1 through the harness rules (or
      starts from `state`), applies step i's action (or `action`), and prints
      what `engine.py` printed during the step, what the step changed in the
      engine's state (sprites by `#index`, matched by identity; vars; status)
      and, for the recorded action, the comparison with the recording as
      `run_tests` explains it. It returns copies of the State before and after.
    - `auto_sprites(level=0, grid=None, step=None, frame=None, merge=False)`
      prints code for a sprite list that redraws a level's recorded start
      exactly (`auto_sprites.py`): border and background sprites, one sprite
      per single-colour 4-connected region (`merge=True`: per group of touching
      regions), identical objects sharing a pixel constant and a tag, screen
      sprites for the HUD. It guesses the logical grid conservatively
      (33 of the 34 level starts of the reference ports right, and every one of
      their 1,529 recorded frames on its own), runs the code and checks it
      renders the frame. A starting point, not the real sprites.
  - `view_engine`, `write_engine`, `edit_engine`: read and change `engine.py`.
  - `run_tests(level=None, stop_on_fail=True, details=2)`: for a `simple`
    engine first the contract tests (interface unchanged, valid states, a fresh
    state on every `make_level` call, every advertised action accepted,
    determinism), then the acceptance test: a full replay, or with `level=L`
    only level L (the engine starts at `make_level(L)`, its drawing is compared
    with the level's recorded start, then it plays that level's steps). By
    default the report stops at the first failing test: a failing contract
    test, or "steps a..k-1 match" and step k explained; `stop_on_fail=false`
    reports everything (per-level counts, `details` failing steps explained,
    the list of all failing steps). A failing step is explained with the
    differing regions of the final frame, numbered and boxed in a picture of
    both frames (below), the colour changes, the engine's sprites drawn in each
    region before and after the step (a second, short sandboxed run collects
    them), state fields and vars, the end of what the engine printed during
    the step (prints are captured per step and capped), and the `try_step`
    command that reproduces it. The counts kept (`tests.jsonl`, best engine,
    pass) always come from the whole replay; the text is what stops.
  - `finish(summary)`.
- **Feedback the harness adds** (`agent.py`). The model's reasoning is sent
  back with its turns, as the main harness does on OpenRouter; compaction
  trims old tool outputs and all but the last 10 turns' reasoning once the
  prompt passes 140K tokens. When a turn changes `engine.py` (by any tool,
  including Python) without testing it, `run_tests()` runs automatically with
  its defaults and its report is appended to the turn's output. Images (on by
  default, `--no-images` for text-only models): after the turn's tool messages
  (which stay strings) one user message carries the latest test's picture,
  the engine's final frame and the original's side by side, upscaled 8x, the
  differing regions boxed in cyan and numbered as in the text; older pictures
  are replaced by a placeholder when a newer one arrives. The PNGs are saved
  in `images/` and the transcript logs their paths. Without images the report
  shows hex-digit crops of the regions instead. After every 30 turns without a
  test, a reminder to write and test is appended instead. The prompt asks for a
  `notes.md` of established facts, which outlives compaction and interruptions.
  These came from pilot sessions; see [RESULTS.md](RESULTS.md).
- **Passing** (`--match`). By default (`final`) a step matches when its last
  frame and the state, levels completed, win levels and available actions all
  match; animation frames are not compared, so the agent spends nothing on
  them (it can still look at them to understand an action). `--match all`
  also requires the frame count and every animation frame to match, which is
  how the v2 and v3 runs were scored. The session stops when a full replay
  matches every step (with every contract test passing), when the model calls
  `finish` twice, or when a budget (turns, output tokens, cost, wall time)
  runs out. Only full replays count towards passing and `engine_best.py`, with
  their exact counts over the whole recording whatever `stop_on_fail` shows;
  one-level tests do not. The authoritative final test is a full replay with
  the full report (`final_test.txt`). An interrupted session resumes from its
  `engine.py` with a fresh conversation that carries its notes and last
  reasoning.
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
uv run --no-sync python -m engine_re.tester ENGINE.py TRACE_DIR --stop-on-fail [--level L] [--images DIR]
```

`run_experiment --no-images` gives text-only feedback, for models without image input.

Each game directory holds `trace/`, `workspace/engine.py` (final),
`engine_best.py` (most exactly-matching steps in a full replay),
`transcript.jsonl`, `tests.jsonl`, `result.json` (status, turns, tokens, cost,
best and final test), `images/` (the test pictures sent to the model,
`turn<N>_step<S>[_auto].png`) and `evaluation_<engine>.json`. In
`tests.jsonl` a full replay has `"level": null` and `total` equal to the trace
length; a one-level test has its `level`, and `from_level` (the level the
engine started at) as before. The experiment directory holds `summary.md` and
`evaluation_<engine>.md`.

Tests: `uv run --no-sync pytest tests/test_engine_re.py`.
