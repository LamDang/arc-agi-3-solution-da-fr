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

Calibration comes first: qwen3.8-max returns its real thinking, so on
`runs/base-max-dfranzen` the generated thinking can be compared with what
the teacher actually thought.

## Pipeline

| step | module | output |
| --- | --- | --- |
| read the request logs into one record per model response | `logs.py` | - |
| generate the thinking, game by game, request by request | `generate.py` | `<out>/<game>.jsonl` |
| calibration only: judge generated against real thinking | `judge.py` | `<out>/judge/` |
| read it side by side | `page.py` | an HTML page |
| build SFT samples | `assemble.py` | one JSON line per history stretch |

```bash
# calibration on qwen3.8-max (real thinking available), two arms
uv run --no-sync python -m think_gen.generate --run runs/base-max-dfranzen \
  --out runs/think-calib-b1/sum --games ft09 lp85 vc33 --limit 30 --summary synth
uv run --no-sync python -m think_gen.generate --run runs/base-max-dfranzen \
  --out runs/think-calib-b1/nosum --games ft09 lp85 vc33 --limit 30 --summary none
uv run --no-sync python -m think_gen.judge runs/think-calib-b1/sum runs/think-calib-b1/nosum
uv run --no-sync python -m think_gen.page --run runs/base-max-dfranzen \
  --arm sum=runs/think-calib-b1/sum --arm nosum=runs/think-calib-b1/nosum \
  -o runs/think-calib-b1/page.html

# GPT-6.1 Sol
uv run --no-sync python -m think_gen.generate --run runs/base-gpt61sol-dfranzen \
  --out runs/think-sol-b1 --summary teacher
uv run --no-sync python -m think_gen.assemble --run runs/base-gpt61sol-dfranzen \
  --thinking runs/think-sol-b1 -o runs/think-sol-b1/sft.jsonl --won-only
```

`dvc pull runs/base-gpt61sol-dfranzen.dvc runs/base-max-dfranzen.dvc` fetches
the two runs. Needs `OPENROUTER_API_KEY`.

## How a request is generated

- **The record.** Each model response of the run. Its context is
  the request's `messages` and `tools`, cut where the teacher answered. Its
  target is the reply: the `python` call and any visible text.
- **Order.** Games run in parallel. A game's requests run in order, and the
  earlier turns in each context carry the thinking already generated for
  them. That is what the student sees at deploy time, where SGLang runs
  with `preserve_thinking` and the harness keeps every turn's reasoning in
  history. A rerun resumes after the last record written.
- **History thinking goes in the message text.** OpenRouter drops
  `reasoning` and `reasoning_content` from history messages for
  qwen3.8-flash, and Alibaba strips `<think>` blocks from history content.
  Both were tested on 2026-10-07: with a codeword in the earlier reasoning,
  the prompt token count did not change and the model could not repeat the
  codeword. So the thinking is put in the assistant message's content
  between `[thinking]` and `[/thinking]` lines. As in the harness history,
  blank lines are removed.
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
- **Checks** (`checks.py`). A thinking text is retried, up to 3 attempts,
  when:
  - it mentions a summary, "the agent", reconstructing, a given output, or
    `[thinking]` / `<think>` tags;
  - more than half of the call's code lines (20+ characters) appear in it
    verbatim.

  A text still failing after the last attempt is kept with
  `status: rejected`.

## Calibration

`--summary synth` writes a summary from the teacher's real thinking with
flash, in the style of real GPT-6.1 Sol summaries: four of them, taken from
`--style-run`, are shown as examples. The calibration then generates
thinking from these summaries.

`judge.py` gives the real and the generated thinking to flash, with its
reasoning on. The judge:

- lists the real thinking's key points (at most 10) and marks which ones
  the generated thinking covers;
- lists the generated thinking's contradictions with the real thinking or
  the output;
- says whether the generated thinking leads to the output, and whether it
  leaks.

It reports mean coverage, contradictions per record, the share leading to
the output, the leak rate and the median length ratio (generated over real
characters).

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
  turns: `ok`, `rejected`, `teacher_empty` or `missing`.
- The harness's private `_arc3_control` keys and `reasoning_details` are
  dropped. A `content` of null becomes `""`.

The deployed server is SGLang with `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_coder`, `preserve_thinking` and the checkpoint's own
`chat_template.jinja` ([models/README.md](../../models/README.md)). That
template is not in the repo. Check how it renders `reasoning_content` before
training. `tests/test_render.py` in `exp/reap-flash-next/` shows how to
compare rendered lengths with logged prompt tokens.
