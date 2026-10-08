# Generating a teacher's hidden thinking with qwen3.8-flash

GPT-6.1 Sol plays the five public games far better than qwen3.8-flash
([exp/base-gpt61sol-5games.md](../exp/base-gpt61sol-5games.md), on branch
`claude/game-reverse-eng-playing-b4a65f`), but its API returns only an
encrypted reasoning block and, for about a third of the requests, a short
summary. This package writes the missing thinking with qwen3.8-flash, the
student's family, from what the request logs do hold: the agent's context,
the call that followed, and the summary when there is one. It is method 2
("rationalization") of
[distillation-methods.md](../experiments/teacher-reasoning/distillation-methods.md),
with the reconstructor prompt (approach B).

Sol's real thinking is hidden, so the generated thinking is scored and
improved by a judge-and-refine loop on gpt-6.1-sol rather than diffed against a
ground truth. The four checks, the refine pass, the held-out results and the
cost analysis are in
[sol-judge.md](../experiments/teacher-reasoning/sol-judge.md); the whole thing
runs as DVC stages (`dvc repro tg_final`, params under `think_gen:` in
`params.yaml`).

## Pipeline

| step | module | output |
| --- | --- | --- |
| read the request logs into one record per model response | `logs.py` | - |
| generate the thinking, game by game, request by request | `generate.py` | `<out>/<game>.jsonl` |
| judge the thinking (gpt-6.1-sol): coverage, code, fact, call | `judge_sol.py` | `<out>/judge_sol/` (or `--out`) |
| refine from the judge feedback (second pass) | `refine.py` | a new `<out>` |
| pick an evaluation manifest from a run | `evalset.py` | a manifest JSON |
| build SFT samples | `assemble.py` | one JSON line per history stretch |

```bash
# the 2-refine-pass pipeline on an evaluation manifest, as DVC stages
dvc repro tg_final      # params.yaml think_gen.manifest selects the sample

# or a single generate + SFT assemble by hand
uv run --no-sync python -m think_gen.generate --run runs/base-gpt61sol-dfranzen \
  --out runs/think-sol-b1 --summary teacher
uv run --no-sync python -m think_gen.assemble --run runs/base-gpt61sol-dfranzen \
  --thinking runs/think-sol-b1 -o runs/think-sol-b1/sft.jsonl --won-only
```

`dvc pull runs/base-gpt61sol-dfranzen.dvc` fetches the run. Needs
`OPENROUTER_API_KEY` (flash) and `OPENAI_API_KEY` (the gpt-6.1-sol judge).

## How a request is generated

- **The record.** Each model response of the run. Its context is
  the request's `messages` and `tools`, cut where the teacher answered. Its
  target is the reply: the `python` call and any visible text.
- **Order.** Games run in parallel. A game's requests run in order, and the
  earlier turns in each context carry the thinking already generated for
  them. That is what the student sees at deploy time, where SGLang runs
  with `preserve_thinking` and the harness keeps every turn's reasoning in
  history. A rerun resumes after the last record written.
- **History thinking goes in the `reasoning` field.** OpenRouter passes a
  history message's `reasoning` to qwen3.8-flash on Alibaba. Two pieces of
  evidence:
  - A probe with a 1,000-token reasoning block grew the prompt by that much,
    and the model could quote a codeword from it.
  - In `runs/base-max-dfranzen` the prompt grew by at least the previous
    reply's reasoning tokens on all 195 consecutive request pairs.

  As in the harness history, blank lines are removed. `--history inline`
  puts the thinking in the message content between `[thinking]` and
  `[/thinking]` lines instead. The first calibration runs (`b1`, `b2`) used
  inline.
- **The instruction.** A user message after the context (`prompts.RECONSTRUCT`)
  shows the call (the code verbatim) and the summary, and asks for the
  thinking in first person. The thinking should:
  - start from the newest input;
  - use only what is visible;
  - not predict the call's result;
  - not mention the summary or the task;
  - end at that call, describing the code rather than pasting it.

  Flash answers with the thinking as plain text, with its own reasoning off
  (`--reasoning` turns it on) and tools offered with `tool_choice: none`.
- **Summaries.** `--summary teacher` uses the teacher's reasoning summary
  when it has one. A request with no summary gets an instruction to keep
  the thinking short. A request where the teacher spent 0 reasoning tokens
  gets empty thinking, as the teacher had. Length is otherwise left to flash.
- **Checks** (`checks.py`) are advisory, not a gate: a draft is never
  rejected. `check` flags a leak (phrasing that reads as written after the
  fact — reconstructing the *reasoning*, "the agent", a summary, a given
  output, `[thinking]`/`<think>` tags), a too-short draft, or more than half
  the call's code lines (20+ characters) pasted verbatim. The flags are
  recorded on the row and fed to the refine pass as feedback, so the
  judge-and-refine loop fixes them instead of the record being dropped.

## SFT format

`assemble.py` writes the format the deploy-time renderer needs
(`exp/reap-flash-next/render.py`, which matches SGLang's prompt tokens for
the Kaggle runs). Each sample is one history stretch, as in
`exp/reap-flash-next/traces.py`: the last request of the stretch plus its
reply. Between history trims, each request extends the previous one, so
that sample holds every turn of the stretch once.

- Messages are in the logged OpenAI format.
- Assistant turns carry the generated thinking as `reasoning_content`, the
  key the Kaggle notebook sends to SGLang.
- `chat_template_kwargs` is `{"enable_thinking": true, "preserve_thinking": true}`.
- `turn_status` gives each assistant turn's status so training can mask
  turns: `ok`, `teacher_empty` or `missing`.
- The harness's private `_arc3_control` keys and `reasoning_details` are
  dropped. A `content` of null becomes `""`.

The deployed server is SGLang with `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_coder`, `preserve_thinking` and the checkpoint's own
`chat_template.jinja` ([models/README.md](../../models/README.md)). That
template is not in the repo. Check how it renders `reasoning_content` before
training. `tests/test_render.py` in `exp/reap-flash-next/` shows how to
compare rendered lengths with logged prompt tokens.
