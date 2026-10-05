# v7: reasoning effort low and medium

## Hypothesis

Most of qwen/qwen3.8-flash's output is reasoning (89% in v6c), and a few
turns think for 20K-33K tokens. Qwen's card for its 3.8 Flash models
recommends temperature 1.0, top_p 0.95 and top_k 20 with thinking on. A lower
OpenRouter `reasoning.effort` with those settings could cut the thinking per
turn, and so the time and cost of a run, without losing progress.

## What changed in the harness

From v6c (`2f3ee95`):

- `d7041ea`: the kernel built-ins renamed (`step_to_fix`, `recording`,
  `read_file`, `edit_file`, `undo_edit`, `render_state`, `show_frames`,
  `replay_step`) and every object documented in the prompt's `# Objects`
  section.
- `88c0bf6`: `commit_engine(message)` replaces `finish`; only a commit moves
  the stepwise run on.
- `515718e`: `--reasoning-effort`, `--temperature`, `--top-p`, `--top-k`.
- `c67d65d`, `59860f2`, `1b1a304`: an interrupted stepwise run continues its
  own conversation, rebuilt from `transcript.jsonl`.

Both runs started at 14:46 (after `c67d65d`) and were resumed twice
(`result.json` `resumes`: 2); their worktree ended at `1b1a304`, and the
commits after `c67d65d` change only how a conversation is logged and rebuilt. Each run
directory also keeps the transcript of a discarded first start
(`*.discarded-fresh-restart.jsonl`).

## Runs

- v7 low: `runs/engine-re/qwen38flash-v7-low-lp85` (`qwen38flash-v7-low-lp85.dvc`), code `1b1a304`
- v7 medium: `runs/engine-re/qwen38flash-v7-medium-lp85` (`qwen38flash-v7-medium-lp85.dvc`), code `1b1a304`

Commands (from `ARC3-Inference/`, at `1b1a304`):

v7 low:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v7-low-lp85 --model qwen/qwen3.8-flash \
  --reasoning-effort low --temperature 1.0 --top-k 20 --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

v7 medium:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v7-medium-lp85 --model qwen/qwen3.8-flash \
  --reasoning-effort medium --temperature 1.0 --top-k 20 --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

## Results

v6c (temperature 0.7, top_p 0.95, no effort sent) for comparison:

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6c | turn limit | 100 | 20 | 20 | 59.5 | $0.407 | 9,831,010 | 91% | 291,064 | 259,495 | - | 8 |
| v7 low | turn limit | 100 | 16 | 16 | 67.3 | $0.464 | 8,915,562 | 85% | 311,437 | 266,787 | 5 | 4 |
| v7 medium | turn limit | 100 | 16 | 16 | 49.5 | $0.349 | 8,207,474 | 90% | 218,413 | 184,945 | 5 | 5 |

Thinking per turn, from the `usage` of each turn in `transcript.jsonl`:

| run | reasoning tokens per turn: mean | median | max | turns with 10,000 or more | largest prompt |
| --- | --- | --- | --- | --- | --- |
| v6c | 2,595 | 621 | 32,768 | 7 | 143,495 |
| v7 low | 2,668 | 588 | 28,666 | 8 | 207,690 |
| v7 medium | 1,849 | 246 | 30,303 | 5 | 192,230 |

Timeline: `tN: k` means the commit (or, before v7, the passing test) accepted at turn N left steps 0..k-1 passing (the `advance` records of `transcript.jsonl`).

- v6c: t13: 2, t16: 8, t34: 9, t43: 10, t49: 16, t63: 17, t90: 18, t94: 20
- v7 low: t16: 2, t21: 8, t57: 9, t82: 16
- v7 medium: t22: 4, t30: 8, t51: 9, t71: 10, t82: 16

## Findings

- The effort setting did not change the thinking. With effort low the mean
  reasoning per turn (2,668 tokens) was as high as v6c's (2,595) and higher
  than with effort medium (1,849); both runs still had turns of 28K-30K
  reasoning tokens.
- OpenRouter's model listing for qwen/qwen3.8-flash gives `reasoning:
  {"mandatory": false, "default_enabled": true, "supports_max_tokens": true}`
  and no effort levels: the model takes a token budget, not an effort. A
  probe request in the session with effort low still produced about 20K
  reasoning tokens (not archived).
- Both runs reached 16 steps (level 1, not solved), against 20 for v6c, with
  one sample each and the sampling, the effort and the harness all changed at
  once, so the difference is not attributable.
- This led to `--thinking-budget` (`5f683d5`, OpenRouter's
  `reasoning.max_tokens`), tried in v9t ([v9-bar-tolerance.md](v9-bar-tolerance.md)).
