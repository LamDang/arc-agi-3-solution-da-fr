# The per-turn condenser (v8c, v6cc)

## Hypothesis

Compaction shortens the conversation in place by age once a prompt passes
140K tokens: old tool outputs are cut to 200 characters and all but the
last 10 turns' reasoning to their last 1,200 characters, whatever they held. A
condenser that rebuilds the context by iteration (each finished step fix
reduced to its failing-step message, the net diff of `engine.py` and the
commit; the last iterations kept whole) keeps what the model needs and sends
fewer prompt tokens.

## What changed in the harness

- `f561428`: `engine_re/condense.py`, the stepwise conversation condensed by
  iteration (a pure function of the conversation, the transcript records and
  `engine_versions/`).
- `4801b11`: `condense_report.py` compares compaction with the condenser on
  finished runs.
- `dc6fc5a`: the condenser follows the v8 harness messages.
- `6c44be1` (**v8c**): `run_experiment --condense` makes the condenser the
  agent's context scheme. In this version it condensed the full conversation
  before **every** request (one `condense` record per turn).
- `23e0b5f` (**v6cc**): the same per-turn condenser ported to the v6c harness
  (parent `2f3ee95`). This commit is only on the local branch
  `engine-re-v6c-condense`; it is neither on the shared branch of this work
  nor pushed.

`d0bfdac` later replaced the per-turn scheme by a threshold one (v9c, in
[v9-bar-tolerance.md](v9-bar-tolerance.md)); the README section "Context: two
schemes" describes the current condenser.

## Runs

- v8c: `runs/engine-re/qwen38flash-v8c-lp85` (`qwen38flash-v8c-lp85.dvc`), code `6c44be1`
- v6cc: `runs/engine-re/qwen38flash-v6cc-lp85` (`qwen38flash-v6cc-lp85.dvc`), code `23e0b5f`

Commands (from `ARC3-Inference/`, at the commit above):

v8c:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v8c-lp85 --model qwen/qwen3.8-flash --condense \
  --max-turns 100 --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

v6cc:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v6cc-lp85 --model qwen/qwen3.8-flash --condense \
  --max-turns 100 --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

## Results

Each condensed run next to its compaction twin (v6cc with v6c, v8c with v8):

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6c | turn limit | 100 | 20 | 20 | 59.5 | $0.407 | 9,831,010 | 91% | 291,064 | 259,495 | - | 8 |
| v6cc | turn limit | 100 | 20 | 20 | 72.6 | $0.516 | 6,449,678 | 69% | 310,684 | 280,701 | - | 6 |
| v8 | turn limit | 100 | 20 | 1 | 71.3 | $0.517 | 10,103,646 | 85% | 336,912 | 297,568 | 5 | 5 |
| v8c | turn limit | 100 | 16 | 16 | 34.4 | $0.305 | 5,343,655 | 79% | 148,882 | 113,544 | 6 | 6 |

Context and prompt cost (prompt and output cost summed from the `usage` of
each turn in `transcript.jsonl`):

| run | context | condense records | prompt tokens | cached | largest prompt | prompt cost | output cost | prompt cost per 1M prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6c | compaction | 0 | 9,831,010 | 91% | 143,495 | $0.270 | $0.137 | $0.028 |
| v6cc | per-turn condenser | 100 | 6,449,678 | 69% | 140,399 | $0.370 | $0.146 | $0.057 |
| v8 | compaction | 0 | 10,103,646 | 85% | 158,778 | $0.358 | $0.158 | $0.035 |
| v8c | per-turn condenser | 100 | 5,343,655 | 79% | 103,616 | $0.235 | $0.070 | $0.044 |

Turn at which each milestone was first accepted:

| steps passing | v6c | v6cc | v8 | v8c |
| --- | --- | --- | --- | --- |
| 9: level 0 solved (step 8) | 34 | 44 | 31 | 54 |
| 18: level 1 solved (step 17) | 90 | 95 | 68 | - |
| 20: step 20 (level 2) | 94 | 95 | 68 | - |

Timeline: `tN: k` means the commit (or, before v7, the passing test) accepted at turn N left steps 0..k-1 passing (the `advance` records of `transcript.jsonl`).

- v6c: t13: 2, t16: 8, t34: 9, t43: 10, t49: 16, t63: 17, t90: 18, t94: 20
- v6cc: t28: 8, t44: 9, t63: 10, t69: 16, t80: 17, t95: 20
- v8: t8: 8, t31: 9, t36: 16, t50: 17, t68: 20
- v8c: t24: 2, t29: 8, t54: 9, t64: 10, t84: 14, t95: 16

## Findings

- The per-turn condenser cut prompt tokens by 34% (v6cc against v6c) and
  47% (v8c against v8).
- It broke the prompt cache: the condensed view changes at every request, so
  the cached share fell to 69% (v6cc, against 91%) and 79% (v8c, against
  85%), and each prompt token cost more ($0.057 against $0.028 per million
  for v6cc and v6c; $0.044 against $0.035 for v8c and v8).
- It did not make the runs cheaper by itself: v6cc cost $0.516 against
  $0.407, its prompt part up from $0.270 to $0.370. v8c cost $0.305 against
  $0.517, but it also generated 56% fewer output tokens (148,882 against
  336,912) in 34 minutes against 71.
- Progress: v6cc reached 20 steps (turn 95, v6c turn 94); v8c reached 16
  (v8: 20).
- Next: fire the condenser only when compaction would and keep its view as a
  fixed prefix between firings, so the cache hits (`d0bfdac`, v9c).
