# Local evaluation with OpenRouter

How to run a small subsample of games against an OpenRouter model, score it,
and find what each run records. Commands run from `ARC3-Inference/`.

## Prerequisites

- **Network access** to `openrouter.ai` (model API) and `three.arcprize.org`
  (game files, needed only once per game). In a Claude Code cloud environment,
  add both under the environment's Network access settings.
- **An OpenRouter key** in `OPENROUTER_API_KEY`. In Claude Code cloud sessions,
  the session-start hook sets it from `OR_API_KEY`.

## Setup

In Claude Code cloud sessions, `.claude/hooks/session-start.sh` does all of
this at startup. The cloud environment caches the result, so later sessions
start ready. Elsewhere, run it by hand:

```bash
uv python install 3.12.12   # if uv finds no download, upgrade it: pip install -U uv
uv sync --frozen --extra dev
uv run --no-sync python scripts/fetch_games.py
```

- Use `uv sync --frozen`, not `--locked`. `--locked` re-validates the
  lockfile, which downloads the vLLM wheel from `wheels.vllm.ai`. vLLM is only
  needed to serve a model locally (`make install` / `make server`).
- `scripts/fetch_games.py` downloads the 25 official games into
  `environment_files/` and skips games already there. Pass ids or prefixes to
  download fewer, for example `scripts/fetch_games.py ls20 ft09`.
  This build bundles no game files. Without `ENVIRONMENTS_DIR`, games error
  with "Offline env files (the '__auto__' mode) are not bundled".

## Run a subsample

```bash
make interactive CONFIG_PATH=configs/inference.openrouter.json \
  MODEL=qwen/qwen3.6-27b GAME=ls20,ft09,vc33,sp80,lp85 GAME_TAGS=[] \
  N_PASSES=1 ENVIRONMENTS_DIR=environment_files EXPERIMENTS_DIR=runs \
  MAX_GENERATED_TOKENS_PER_GAME=200000 MAX_RUNTIME_MINUTES=30 \
  ANALYZER_SAVE_REQUEST_LOGS=true
```

- `GAME` takes full ids or short prefixes. List the official ids with
  `uv run --no-sync inference-taaf-run --include-tags official --list-games`.
- `GAME_TAGS=[]` is required. The config's `include_tags: ["official"]`
  otherwise adds all 25 official games to the ones in `GAME`.
- `EXPERIMENTS_DIR=runs` writes to `runs/<timestamp>/` instead of the config
  default `/shared/arc_3_results/<user>`. `make eval` reads `runs/` by default.
- All game runs play at once: the OpenRouter config sets `concurrent_jobs` to
  `0`, meaning no limit. Set `CONCURRENT_JOBS=<n>` to cap it.
- A request that OpenRouter rate-limits (HTTP 429) is retried 3 times with
  backoff. After that, the turn is aborted and rolled back, losing its partial
  tool work. If that happens often, prefix the command with
  `ARC3_HTTP_RETRIES=-1` to retry without limit, or cap `CONCURRENT_JOBS`.
- Game engines and Python tool calls run on local CPUs, so many parallel games
  can slow tool calls on a small machine.
- The OpenRouter config defaults to 5 passes per game and 90 minutes per game.
  The command above sets 1 pass and 30 minutes.
- `MODEL` is the OpenRouter model id, as listed at
  `https://openrouter.ai/api/v1/models`. Without it, the run uses the config's
  `shared.model_name` (`Qwen/Qwen3.6-27B`), which differs in case from the
  catalogue id.
- `ANALYZER_SAVE_REQUEST_LOGS=true` is the only way to record input tokens; see
  [Token spend](#token-spend).

## Resume a run

To replay only the game runs that failed in an earlier run, rerun the same
command with `RESUME_FROM` set to that run's directory:

```bash
make interactive <same settings as the earlier run> RESUME_FROM=runs/<run>
```

- The resumed run gets a new directory. The earlier one is not changed.
- Kept: game runs that won, or that stopped at their own token, time or action
  limit (`gave_up` with the note `tokens=<n>`). Their `benchmark.json` entries
  and artifacts are copied into the new run.
- Replayed from the start: runs that crashed, were cancelled, gave up on
  analyzer errors, or were still `playing` because the process was killed.
  A replayed game starts again at level 1.
- `GAME` and `N_PASSES` must match the earlier run; the run stops with an
  error otherwise. Keep the other settings the same so the results stay
  comparable.
- `resume.json` in the new run lists what was kept and what was replayed.
- `benchmark.json` is saved every 10 minutes, so after a kill a run can be up
  to 10 minutes behind and is replayed even if it had just finished.
- If every run in the earlier directory finished, nothing runs.

## Let the agent read the game code

For experiments, the agent can be given read access to its game's source code:
the game's module and the `arcengine` package it is built on.

```bash
uv run --no-sync python scripts/extract_game_code.py ls20 ft09 vc33 sp80 lp85
ARC3_GAME_CODE_DIR=game_code make interactive <settings>
```

- `scripts/extract_game_code.py` copies each game's module byte for byte from
  `environment_files/`, the file the game loader runs, and the installed
  `arcengine` package, into `game_code/`. `game_code/manifest.json` records
  each file's source and sha256. `game_code/` in git holds the 5 games above.
- With `ARC3_GAME_CODE_DIR` set, the python tool has `game_code_files`,
  `game_code(file=None)` (a file's full text) and
  `read_game_code(start=1, end=None, file=None)` (numbered lines), and the
  system prompt describes them. The agent can read the code but not run it.
- A game whose code is missing from the directory fails instead of playing
  without it. `artifacts/<game>_p<pass>_game_code.json` lists the files the
  agent could read, with their hashes.
- Unset or empty, the default, the agent is unchanged.

## Limits

| Override | Applies to | Notes |
| --- | --- | --- |
| `MAX_GENERATED_TOKENS_PER_GAME` | Each game | Counts output tokens only, including reasoning and rolling-summary generation. Checked between model responses, so a game can go over by one response. |
| `MAX_RUNTIME_MINUTES` | Each game | Wall-clock time. |
| `MAX_ACTIONS` | Each game | Game actions, not model requests. |
| `MAX_EXPERIMENT_RUNTIME_MINUTES` | Whole run | Wall-clock time. When `MAX_RUNTIME_MINUTES` is unset, the runner derives the per-game cap from it. |

No limit covers input tokens, and no token limit covers the whole run. Each
request re-sends the conversation history, up to `shared.context_window`
(65,536 in the OpenRouter config), so input tokens can make up most of the
cost. To bound them, lower `shared.context_window` in a copy of the config, or
use the time and action limits.

## Score a run

```bash
make score_run SCORE_RUN_DIR=runs/<run>
```

This reads `benchmark.json` and prints, for each game, the score, levels
completed out of total, and end state, then the overall score. It writes
`evaluation.json` and `score.json` into the run directory.

The per-game score uses the official ARC-AGI-3 formula. Each completed level
scores `min(115, (baseline_actions / actions_used)² × 100)`, and level *n* has
weight *n*. The game score is the weighted average, capped by the weight share
of the levels that scored.

Other tools:

- `make eval` scores the runs listed in `configs/eval.json`, or every run under
  `runs/` when the list is empty.
- `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...` compares two
  `score.json` files game by game. A 5-game, 1-pass run is too small for it to
  detect much.
- `make view VIEW_RUN_DIR=runs/<run> VIEW_PORT=8011` serves the run viewer at
  `http://127.0.0.1:8011`. It needs no model server.
- `make traces` exports transcripts as JSON.

## Run artifacts

| Path in `runs/<run>/` | Contents |
| --- | --- |
| `run_config.json` | Resolved games, passes, concurrency, limits, model, hardware. |
| `git_info.txt` | Commit and uncommitted diff of the code that ran. |
| `src/` | Copy of the harness and TAAF source that ran. |
| `summary.txt` | Quick summary: mean and median score, total actions, total output tokens, duration, and per-game score, levels, actions and output tokens. |
| `stdout.log` | Run log. |
| `benchmark.json` | Per game: end state (`won`, `gave_up`, `cancelled`, `crashed`), levels completed, score, every action with the output tokens spent on it, and `solver_note` (`tokens=<n>` or the error). |
| `diagnostics.html` | TAAF diagnostics page. |
| `artifacts/*_viewer_data.json`, `artifacts/*_events.jsonl` | Viewer data: boards, actions, rewards, level changes, tokens per step. |
| `transcripts/*.txt`, `solver_analysis/*.html`, `prompts/*.log` | Model reasoning, tool calls and prompts for each game run. |
| `<game>_p<pass>_requests.jsonl` | Only with `ANALYZER_SAVE_REQUEST_LOGS=true`. One file per game run. Two lines per model request: `request` (full messages and tools) and `response` (finish reason, provider, `usage`). These files get large. Older runs can also have a run-level `requests.jsonl` and `prompts/prompt.log`: all of a single-game run's logs, or a multi-game run's logs from whenever only one game was playing. |
| `evaluation.json`, `score.json` | Written by scoring: per-game score, levels completed, total levels, completion rate, trial count; run metadata. |
| `resume.json` | Only in a run started with `RESUME_FROM`: the earlier run, and which game runs were kept or replayed. |
| `artifacts/*_game_code.json` | Only with `ARC3_GAME_CODE_DIR`: the source files the agent could read, with sha256 and line counts. |
| `eval_settings.json` | Only in runs made by `scripts/dvc_eval.py`: the make variables and harness environment of the run. |

## Save and reproduce runs with DVC

DVC keeps the game files and run directories in the S3 bucket
`kaggle-arc-agi-3-dvc` (region `eu-west-3`, set in `.dvc/config`), and small
pointer files in git. Install DVC with `uv tool install 'dvc[s3]==3.67.1'`.
In Claude Code cloud sessions, the session-start hook does this. DVC reads AWS
credentials from the usual places, such as `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, or `~/.aws/credentials`. Run the commands below from
`ARC3-Inference/`.

Without credentials, DVC can use a bucket that allows anonymous access when
`dvc remote modify --local storage allow_anonymous_login true` is set. S3
refuses anonymous multipart uploads, so `dvc push` then fails on large files,
such as request logs over about 100 MB.

### Run an eval through DVC

`params.yaml` holds the run settings: `eval.make` is passed to
`make interactive`, and `eval.env` sets the harness environment. Its values
are those of `runs/20261004_135539`. `dvc.yaml` has two stages:

- `games` downloads the game files into `environment_files/`.
- `eval` runs the games into `runs/dvc-eval/`, scores them, and writes
  `metrics.json`: the score and levels per game, and the API tokens and cost
  from the request logs.

```bash
dvc exp run -n qwen38-500k                     # settings from params.yaml
dvc exp run -n qwen38-200k -S eval.make.MAX_GENERATED_TOKENS_PER_GAME=200000
dvc exp show --only-changed                    # compare experiments
dvc exp push origin qwen38-500k                # git ref to GitHub, data to S3
```

- The `eval` stage runs again only when `params.yaml`, the harness code, the
  configs or the game files changed. When nothing changed, DVC restores the
  earlier outputs from its cache instead of calling the model. Add `--force`
  to draw a new sample with the same settings.
- A rerun reproduces the setup, not the scores: the model samples at
  temperature 0.7.
- Shell variables starting with `ARC3_`, `LOCAL_ANALYZER_` or `MULTIMODAL_`
  that `eval.env` does not list are not passed to the run, so the shell cannot
  change a run without `params.yaml` showing it. API keys are passed.
- `runs/dvc-eval/` is replaced by each run. To keep one in the branch history,
  `dvc exp apply <name>`, commit, then `dvc push`.
- `dvc exp apply` replaces the workspace files with the experiment's,
  including code, and overwrites uncommitted changes. Commit them first.
- A failed run leaves `runs/dvc-eval/` for inspection and caches nothing.

To run the same settings outside DVC, for example two arms of an experiment
at once, call the stage's script with a run directory and overrides:

```bash
uv run --no-sync python scripts/dvc_eval.py --run-dir runs/control \
  --metrics runs/control.metrics.json
uv run --no-sync python scripts/dvc_eval.py --run-dir runs/engine-code \
  --metrics runs/engine-code.metrics.json --env ARC3_GAME_CODE_DIR=game_code
```

`--make KEY=VALUE` and `--env KEY=VALUE` add to or override `eval.make` and
`eval.env`, and can be repeated. The run directory gets `eval_settings.json`
with the settings used.

### Archive a run made with `make interactive`

```bash
dvc add runs/<run>        # writes runs/<run>.dvc and stages it in git
git commit -m "Archive run <run>"
dvc push runs/<run>.dvc
```

The run's `git_info.txt` and `src/` record the code it ran.

### Get a saved run

```bash
dvc pull runs/<run>.dvc                        # an archived run
dvc pull                                       # everything the branch tracks
dvc exp pull origin <name> && dvc exp apply <name>                   # an experiment
```

## Token spend

`benchmark.json` records output tokens per action. Its input-token field
(`uncached_input_tokens`) is always 0, and the score files contain no token
counts. For input tokens and cost, sum the `usage` of the `response` lines in
the request logs:

```bash
uv run --no-sync python - runs/<run> <<'EOF'
import json, pathlib, sys
totals = {}
for path in pathlib.Path(sys.argv[1]).glob("*requests.jsonl"):
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") != "response":
            continue
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
print(json.dumps(totals, indent=2))
EOF
```

The keys are whatever OpenRouter returns in `usage`, such as `prompt_tokens`,
`completion_tokens` and `cost`. Rolling-summary requests are not written to the
request logs, so this total leaves them out when summaries are enabled.

`scripts/token_breakdown.py` breaks the same totals down per game and level,
and output tokens into thinking (`reasoning_tokens`) and tool calls. It also
counts the tool calls that read game code. With `--label`, it labels what the
thinking is about, using Claude Haiku 4.5 through OpenRouter: game mechanics,
planning, tooling or other, and whether it discusses the game's source code.
Labelling costs about $0.30 per 1,000 excerpts of 600 characters, and labels
are cached in the output directory.

```bash
uv run --no-sync python scripts/token_breakdown.py runs/<run> [runs/<run> ...] \
  --out <dir> --label
```
