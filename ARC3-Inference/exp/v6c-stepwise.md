# v6: the stepwise harness (v6, v6h, v6c)

## Hypothesis

In the v5 trials the agent analysed every level before writing or testing an
engine, whatever the prompt said ([RESULTS.md, v5](../engine_re/RESULTS.md#v5-trials-on-lp85-50-turns)).
v6 lets the harness set the order instead: it replays the recording through
`engine.py`, finds the first failing step k, asks the model to fix step k, and
shows the model the recording only up to step k. The model can then only work
on what fails now.

## What changed in the harness

From the v5 harness with the harness-played opening (`e55d603`) and
`--providers` (`9dc9afa`):

- `a46cab7` **v6**: the stepwise harness. A new conversation for each
  breaking step k, opened with "Fix the breaking test: step k"; python shows
  only step k, the tests replay steps 0..k; at most 20 turns per conversation
  (`--episode-turns`) and 2 conversations per step (`--attempts`); only
  `engine.py` carries over.
- `d81ae0d` **v6h**: each conversation sees the recording so far (steps
  0..k, `S`) and `summarize_levels()`; `--only-step` keeps the step-only view.
- `0ba8719`: 20 tries per request, after v6 ended on an HTTP 429.
- `e3efd53` **v6c**: one conversation led from one breaking step to the
  next, with no turn limit per step; when steps 0..k pass the harness replays
  on and adds the next failing step to the same conversation; `result.json`
  gains `advances`.
- `2f3ee95`: rate limits retried without limit (the main harness's retry).

The stepwise harness, as it is now, is described in
[engine_re/README.md](../engine_re/README.md) ("The stepwise harness, v6").
The commits of these runs are inferred from their start times and
`config.json` keys (v6 has `episode_turns`, v6h adds `only_step`, v6c has no
`episode_turns`; v6c started at 12:54, a minute after `2f3ee95`).

## Runs

- v6: `runs/engine-re/qwen38flash-v6-lp85` (`qwen38flash-v6-lp85.dvc`), code `a46cab7`
- v6h: `runs/engine-re/qwen38flash-v6h-lp85` (`qwen38flash-v6h-lp85.dvc`), code `d81ae0d`
- v6c: `runs/engine-re/qwen38flash-v6c-lp85` (`qwen38flash-v6c-lp85.dvc`), code `2f3ee95`
- v6c only-step: `runs/engine-re/qwen38flash-v6c-onlystep-lp85` (`qwen38flash-v6c-onlystep-lp85.dvc`), code `2f3ee95`
- GLM v6: `runs/engine-re/glm53flash-zai-v6-lp85` (`glm53flash-zai-v6-lp85.dvc`), code `a46cab7`
- GLM v6h: `runs/engine-re/glm53flash-zai-v6h-lp85` (`glm53flash-zai-v6h-lp85.dvc`), code `d81ae0d`
- GLM v6c: `runs/engine-re/glm53flash-zai-v6c-lp85` (`glm53flash-zai-v6c-lp85.dvc`), code `2f3ee95`
- GLM v6c only-step: `runs/engine-re/glm53flash-zai-v6c-onlystep-lp85` (`glm53flash-zai-v6c-onlystep-lp85.dvc`), code `2f3ee95`

Commands (from `ARC3-Inference/`, at the commit above):

v6:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v6-lp85 --model qwen/qwen3.8-flash \
  --episode-turns 20 --attempts 2 --max-turns 100 --max-output-tokens 1500000 \
  --max-cost 6 --max-minutes 115
```

v6h:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v6h-lp85 --model qwen/qwen3.8-flash \
  --episode-turns 20 --attempts 2 --max-turns 100 --max-output-tokens 1500000 \
  --max-cost 6 --max-minutes 115
```

v6c:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v6c-lp85 --model qwen/qwen3.8-flash --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

v6c only-step:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v6c-onlystep-lp85 --model qwen/qwen3.8-flash \
  --only-step --max-turns 100 --max-output-tokens 1500000 --max-cost 6 \
  --max-minutes 115
```

GLM v6:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/glm53flash-zai-v6-lp85 --model z-ai/glm-5.3-flash \
  --providers z-ai --episode-turns 20 --attempts 2 --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

GLM v6h:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/glm53flash-zai-v6h-lp85 --model z-ai/glm-5.3-flash \
  --providers z-ai --episode-turns 20 --attempts 2 --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

GLM v6c:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/glm53flash-zai-v6c-lp85 --model z-ai/glm-5.3-flash \
  --providers z-ai --max-turns 100 --max-output-tokens 1500000 --max-cost 6 \
  --max-minutes 115
```

GLM v6c only-step:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/glm53flash-zai-v6c-onlystep-lp85 --model z-ai/glm-5.3-flash \
  --providers z-ai --only-step --max-turns 100 --max-output-tokens 1500000 \
  --max-cost 6 --max-minutes 115
```

## Results

qwen/qwen3.8-flash:

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6 | error (HTTP 429) | 55 | 9 | 9 | 38.9 | $0.120 | 1,889,858 | 82% | 94,779 | 84,007 | - | 0 |
| v6h | error (HTTP 429) | 10 | 1 | 1 | 20.6 | $0.035 | 315,280 | 66% | 34,266 | 31,110 | - | 0 |
| v6c | turn limit | 100 | 20 | 20 | 59.5 | $0.407 | 9,831,010 | 91% | 291,064 | 259,495 | - | 8 |
| v6c only-step | stopped | 17 | 1 | - | 9.1 | $0.036 | 564,646 | 84% | 31,485 | 28,630 | - | 0 |

z-ai/glm-5.3-flash (provider pinned to z-ai):

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM v6 | stopped | 29 | 8 | - | 50.4 | $0.173 | 1,473,044 | 68% | 144,930 | 137,254 | - | 0 |
| GLM v6h | stopped | 12 | 8 | - | 20.2 | $0.072 | 610,679 | 68% | 60,123 | 56,777 | - | 0 |
| GLM v6c | turn limit | 100 | 17 | 17 | 83.6 | $0.615 | 9,530,710 | 81% | 231,168 | 202,250 | - | 6 |
| GLM v6c only-step | stopped | 0 | 1 | - | 0.0 | $0.000 | 0 | - | 0 | 0 | - | 0 |

Timeline: `tN: k` means the commit (or, before v7, the passing test) accepted at turn N left steps 0..k-1 passing (the `advance` records of `transcript.jsonl`). v6 and v6h predate the `advance` records.

- v6c: t13: 2, t16: 8, t34: 9, t43: 10, t49: 16, t63: 17, t90: 18, t94: 20
- GLM v6c: t19: 8, t50: 9, t61: 10, t65: 11, t69: 16, t95: 17

## Findings

- v6 and v6h ended on an OpenRouter HTTP 429 (upstream rate limit) at turns
  55 and 10, with 9 and 1 steps passing; the GLM v6 and v6h runs were stopped
  at turns 29 and 12 with 8. The two qwen endings led to `0ba8719` and
  `2f3ee95`.
- v6c, the one-conversation harness, was the first stepwise run to use its
  100 turns: 20 of 120 steps, $0.407, 59.5 minutes. Level 0 was solved at
  turn 34 and level 1 at turn 90 (step 18 passing); it reached step 20
  (level 2) at turn 94 and ended there.
- GLM v6c reached 17 steps (level 1 up to its last step) at turn 95, for
  $0.615 in 83.6 minutes, with 81% of its prompt tokens cached against 91%
  for qwen.
- The `--only-step` runs were stopped early (17 turns and 0 turns), so they
  say nothing about the step-only view.
- v6c is the baseline of the later series: v7 changes the sampling, v8 the
  tools and prompt, v6cc the context.
