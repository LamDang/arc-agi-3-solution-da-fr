# v8: frame pieces, a lenient edit tool, a parsimony prompt

## Hypothesis

The agent should get through more steps per turn if it gets each step's
frames as objects instead of parsing pixels itself, an edit tool that applies
what it can, the current `engine.py` in every message, and a prompt that asks
for the most parsimonious model of the steps seen so far, rather than rules
general enough for unrecorded play (the earlier prompt said the engine would
later be played on unrecorded action sequences).

## What changed in the harness

From v7 (`1b1a304`), sampling back to the defaults (temperature 0.7,
top_p 0.95, no effort):

- `ffd1de3`: every recorded step carries its frames' pieces (`segment.py`:
  `pieces_before`, `pieces_after`, `changes`, `Pieces.code()`); sprites
  print as the code that builds them.
- `163c945`: a lenient `edit_file` (valid edits applied, the others reported
  with fresh anchors; `replace_def`; 3-character anchors).
- `992aebe`: built-ins called as tools run as python; shorter test feedback
  (one-line reproduce hint, "same result as the last test").
- `18161b6`: `engine.py` listed in every next-step message, older listings
  elided.
- `be88927`: the `engine` built-in always loads the current file; a resumed
  run re-runs its python cells to get its kernel variables back.
- `8dcb70c`: the prompt asks for parsimony, not generality (Occam's razor;
  per-level constants are fine; do not model what has not been observed);
  a `# Sandbox` note.

## Runs

- v8: `runs/engine-re/qwen38flash-v8-lp85` (`qwen38flash-v8-lp85.dvc`), code `8dcb70c`

Command (from `ARC3-Inference/`, at `8dcb70c`):

v8:

```bash
uv run --no-sync python -m engine_re.run_experiment --run-dir runs/20261004_135539 \
  --environments-dir environment_files --games lp85 \
  --out runs/engine-re/qwen38flash-v8-lp85 --model qwen/qwen3.8-flash --max-turns 100 \
  --max-output-tokens 1500000 --max-cost 6 --max-minutes 115
```

## Results

| run | end | turns | best exact | final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | advances |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v6c | turn limit | 100 | 20 | 20 | 59.5 | $0.407 | 9,831,010 | 91% | 291,064 | 259,495 | - | 8 |
| v8 | turn limit | 100 | 20 | 1 | 71.3 | $0.517 | 10,103,646 | 85% | 336,912 | 297,568 | 5 | 5 |

Turn at which each milestone was first accepted:

| steps passing | v6c | v8 |
| --- | --- | --- |
| 9: level 0 solved (step 8) | 34 | 31 |
| 18: level 1 solved (step 17) | 90 | 68 |
| 20: step 20 (level 2) | 94 | 68 |

Timeline: `tN: k` means the commit (or, before v7, the passing test) accepted at turn N left steps 0..k-1 passing (the `advance` records of `transcript.jsonl`).

- v6c: t13: 2, t16: 8, t34: 9, t43: 10, t49: 16, t63: 17, t90: 18, t94: 20
- v8: t8: 8, t31: 9, t36: 16, t50: 17, t68: 20

## Findings

- v8 got further faster: level 0 solved at turn 31 (v6c: 34), level 1 at
  turn 68 (v6c: 90), and 20 steps at turn 68 against turn 94.
- It then spent its last 32 turns (69-100) on step 20, which failed by one
  pixel: row 2 of column 0, expected 14, got 5, the HUD bar. lp85's bar
  blackens `round(64 * used / budget)` pixels of a 64-pixel column, with
  per-level budgets of 13, 60, 80, 150, 80, 80, 80 and 80 moves (the
  reference port in `runs/engine-re/ports/lp85`), so in level 2 a move
  blackens 0 or 1 pixel and per-move costs look irregular.
- Every run that reached level 2 before v9 (v6c, v6cc, v8) spent its
  remaining turns there. With the one-pixel tolerance of v9, the committed
  v8 and v6cc engines pass 25 steps instead of 20 (`34e7779`).
- The final `engine.py` of v8 was broken (1 step passing; a `KeyError` in the
  determinism contract test and at step 1); `engine_best.py` keeps the 20.
- v8 cost more than v6c ($0.517 against $0.407): more output tokens
  (336,912 against 291,064) and a lower cached share (85% against 91%).
