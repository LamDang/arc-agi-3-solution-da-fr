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
- **Interface** (`skeleton.py`, `api_notes.md`). The engine is a module with one
  `arcengine.ARCBaseGame` subclass, the same interface as the real games: the
  harness calls `perform_action` and compares what it returns. The skeleton
  has code only for what the agent writes: tagged sprite prototypes, levels
  built from clones of them plus a `data` dict, `on_set_level` and `step` (with
  the library's `try_move_sprite` collision). Comments explain the rest: where
  visible and hidden state live, how entering a level or RESET rebuilds it
  from a pristine copy, screen-space UI, and that only the final frame counts.
  Nothing game-specific beyond the class name and the advertised actions. The
  API notes describe arcengine's fixed main loop, sprites, levels and camera,
  plus the primitives table.
- **Tools** (`agent.py`, `prompts.py`):
  - `python`: a persistent kernel with the trace loaded as `trace` / `S` and
    analysis helpers (`helpers.py`: summaries, hex region views, frame diffs,
    animation diffs, connected components, logical-grid detection, and
    running the engine in-process with `new_game` / `replay` / `compare`).
  - `view_engine`, `write_engine`, `edit_engine`: read and change `engine.py`.
  - `run_tests(from_level=None, details=2)`: full replay, or one level onwards
    (the engine starts at `set_level(L)`), reporting matching steps, the first
    mismatch with side-by-side pixel crops of the final frame, state fields,
    tracebacks, and the list of all mismatching steps.
  - `finish(summary)`.
- **Feedback the harness adds** (`agent.py`). The model's reasoning is sent
  back with its turns, as the main harness does on OpenRouter; compaction
  trims old tool outputs and all but the last 10 turns' reasoning once the
  prompt passes 140K tokens. When a turn changes `engine.py` (by any tool,
  including Python) without testing it, a full replay runs automatically and
  its summary is appended to the turn's output. After every 30 turns without a
  test, a reminder to write and test is appended instead. The prompt asks for a
  `notes.md` of established facts, which outlives compaction and interruptions.
  These came from pilot sessions; see [RESULTS.md](RESULTS.md).
- **Passing** (`--match`). By default (`final`) a step matches when its last
  frame and the state, levels completed, win levels and available actions all
  match; animation frames are not compared, so the agent spends nothing on
  them (it can still look at them to understand an action). `--match all`
  also requires the frame count and every animation frame to match, which is
  how the v2 and v3 runs were scored. The session stops when a full replay matches every step, when the
  model calls `finish` twice, or when a budget (turns, output tokens, cost,
  wall time) runs out. An interrupted session resumes from its `engine.py`
  with a fresh conversation that carries its notes and last reasoning.
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
```

Each game directory holds `trace/`, `workspace/engine.py` (final),
`engine_best.py` (most exactly-matching steps in a full replay),
`transcript.jsonl`, `tests.jsonl`, `result.json` (status, turns, tokens, cost,
best and final test) and `evaluation_<engine>.json`. The experiment directory
holds `summary.md` and `evaluation_<engine>.md`.

Tests: `uv run --no-sync pytest tests/test_engine_re.py`.
