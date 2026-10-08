# Does sol's thinking lead flash back to sol's call?

A quick check on the held-out NLL panel
[`data/sol-nll-fold0-30`](../../../data/sol-nll-fold0-30/README.md): given the
teacher's (gpt-6.1-sol) thinking for a step, can the student qwen3.8-flash
regenerate a **functionally equivalent** `python` tool call? This is the
call-equivalence judge (check 4) of [`think_gen/judge_sol.py`](../../think_gen/judge_sol.py)
run against this dataset, in isolation.

Each of the 30 requests is a full multimodal context with the recorded final
teacher reply. That reply carries sol's converted thinking in `reasoning_content`
and a code-only `python` call. For each one:

1. **Regenerate** — flash (`qwen/qwen3.8-flash`, Alibaba) gets the game context
   up to the decision point plus the teacher's thinking (`REGEN_NOTE`), and is
   forced to emit a `python` call (`judge_sol.regen_code`).
2. **Judge** — gpt-6.1-sol, with the same game context in front, decides whether
   flash's snippet and the teacher's snippet would carry out the same move on the
   game — same effect, ignoring inspection, printing and style
   (`prompts.judge_call_prompt`, effort `high`).

The dataset is already in the code-only deploy format the judge expects
(the system prompt's python line and the `python` tool schema already carry no
rationale), so the only adaptation is per-message cleanup for the APIs: strip
private `_`-prefixed keys, JSON-encode tool-call `arguments` (stored here as
dicts), and move assistant `reasoning_content` into `reasoning` (native history,
as [`think_gen/context.py`](../../think_gen/context.py) documents). The teacher's
own thinking on the final reply is the input; it is not sent as history.

## Result

**21 of 27 judged calls are functionally equivalent — 0.778.** Three of the 30
are unjudged (see below), so the rate is over the 27 that produced a call.

| game | n | call_functionally_same |
| --- | --- | --- |
| `sp80-589a99af` | 5 | 1.00 |
| `ar25-0c556536` | 6 | 0.83 |
| `sk48-d8078629` | 5 | 0.80 |
| `cd82-fb555c5d` | 6 | 0.67 |
| `tn36-ef4dde99` | 5 | 0.60 |

Flash regeneration cost ~$0.14 (the gpt-6.1-sol judge runs on the OpenAI
Responses prompt cache). Teacher thinking is short: 346 / 526 / 668 characters
(min / median / max).

## The six disagreements — two failure modes

Read from the judge's listed differences
([`sol-nll-regen/results.jsonl`](sol-nll-regen/results.jsonl), fields
`differences` and `notes`). None is a "wrong idea"; the gap is about *how far to
act* and *the exact sequence*.

**A. Act-vs-inspect boundary (2).** The thinking states a plan; flash and the
teacher differ on whether *this* turn executes it or only probes.

- `cd82-r26`: the teacher only inspects and computes a plan (no `action()`);
  flash executes the moves.
- `cd82-r36`: the teacher runs `action(plan)`; flash only computes and prints it.

**B. Action sequence / length mismatch (4).** Right operation, different steps.

- `sk48-r82`: both run 20 `UNDO`s; flash then runs 13 extra actions.
- `ar25-r2`: both probe one `LEFT`; the teacher continues four more `LEFT`s,
  flash stops at the probe.
- `tn36-r12`: both click (58,15) and (58,35); flash adds a third click (58,25).
- `tn36-r28`: the teacher plans `UURRRD`, flash plans `UURRDR` (walks into a wall).

## Unjudged (3)

Infrastructure, not quality:

- `sk48-r39`, `tn36-r5`: flash never cleared Alibaba's upstream 429 rate-limit
  (the largest, most image-heavy contexts) within the retry budget.
- `sp80-r8`: flash returned no `python` call.

A rerun of just these three (serial, higher `--retries`) would close the panel to
30/30.

## Takeaway

Sol's short thinking is enough to steer flash to the teacher's intended operation
about four times in five. The residual gap is not reasoning but execution
discipline: when to stop probing and commit, and reproducing a multi-step action
sequence exactly. This bears on the SFT target — the thinking field alone carries
most of the decision, but not the precise action list.

## Reproduce

```bash
# from ARC3-Inference/, with OPENROUTER_API_KEY (flash) and OPENAI_API_KEY (judge)
dvc pull ../data/sol-nll-fold0-30/requests.jsonl.dvc
PYTHONPATH=. uv run --no-sync python \
  experiments/teacher-reasoning/sol-nll-regen/run.py \
  --out experiments/teacher-reasoning/sol-nll-regen --workers 3
```

Outputs in [`sol-nll-regen/`](sol-nll-regen/): `results.jsonl` (per-request
verdict, both snippets and the thinking), `summary.json`, and `calls.jsonl` (the
replayable request/response log, image URIs reduced to sha references).
