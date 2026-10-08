# SFT data format for Qwen3.8-Flash-Next

How to turn harness games into supervised fine-tuning data for
Qwen3.8-Flash-Next, the student, and how to check the chat template that
renders it. Written 2026-10-08. Where the reasoning comes from (which
teachers return it, and what to do when they do not) is in
[../teacher-reasoning/](../teacher-reasoning/README.md).

The rule: render training text exactly as the server renders the prompt at
inference. The harness serves the model with the `qwen3` reasoning parser,
the `qwen3_coder` tool-call parser and `preserve_thinking: true`
([models/README.md](../../../models/README.md),
`inference/framework/kaggle.py`), and sends `chat_template_kwargs` per
request (`ToolAgent._harness_template_kwargs`). Keep the data structured and
let the model's own `chat_template.jinja` produce the text; do not write the
special tokens by hand.

## Storage format

One JSON line per sample, OpenAI chat messages plus the tools and the
template kwargs the harness sent:

```json
{"tools": [{"type": "function", "function": {"name": "python", "description": "...", "parameters": {}}}],
 "chat_template_kwargs": {"preserve_thinking": true},
 "messages": [
  {"role": "system", "content": "<harness system prompt>"},
  {"role": "user", "content": "<turn opener: frame, diff, state>"},
  {"role": "assistant", "reasoning_content": "Let me look at the grid...", "content": "",
   "tool_calls": [{"type": "function", "function": {"name": "python",
                   "arguments": {"reasoning": "...", "code": "..."}}}]},
  {"role": "tool", "content": "<tool output>"},
  {"role": "assistant", "reasoning_content": "...", "content": "", "tool_calls": ["..."]}
 ]}
```

Rendered by the official template, an assistant turn is:

```
<|im_start|>assistant
<think>
Let me look at the grid...
</think>

<tool_call>
<function=python>
<parameter=reasoning>
...
</parameter>
<parameter=code>
...
</parameter>
</function>
</tool_call><|im_end|>
```

and a tool result is a user turn: `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>`.

Two fields differ from what the harness logs, and both fail on their own
without the fix:

| logged by the harness | the template needs | if not converted |
| --- | --- | --- |
| reasoning under `reasoning` (the OpenRouter key; `ARC3_REASONING_HISTORY_KEY` defaults to it) | `reasoning_content` | the thinking is dropped from the text, silently |
| `tool_calls[].function.arguments` as a JSON string | a mapping (the template loops over `arguments\|items`) | `TypeError: Can only get item pairs from a mapping` |

`normalize_message` in
[scripts/check_chat_template.py](../../scripts/check_chat_template.py) does
both. A reply whose arguments are not valid JSON was cut off inside its tool
call: drop it.

## One sample per request, merged when the history only grows

Build samples from the request logs (`*_requests.jsonl.xz`, written with
`ANALYZER_SAVE_REQUEST_LOGS=true`; see
[LOCAL_EVAL.md](../../LOCAL_EVAL.md)). A `request` line holds `messages`,
`tools` and `chat_template_kwargs`; the `response` line after it holds the
`reply`. A sample is the request's messages exactly as sent, plus the reply
as the trained assistant turn.

A game's requests are not one growing conversation: the harness trims
history, strips old images (`ARC3_HISTORY_IMAGE_KEEP`), compacts notes and
can replace history with a summary. Where a request's rendered prompt starts
with the previous request's prompt and reply, the two can be merged into one
multi-turn sample with loss on both replies; elsewhere start a new sample.
`--request-log` counts how many requests extend the previous one. Per-request
samples also stay under the training context length; the server runs at 139K
tokens and games run longer.

## Loss mask

Train on assistant tokens only. For each assistant turn the trained text is
the render up to and including that turn minus the render of the messages
before it with `add_generation_prompt=True`. With the official template that
is everything after `<|im_start|>assistant\n<think>\n`, up to and including
`<|im_end|>`. The generation prompt already opens `<think>`, so those tokens
belong to the prompt.

This only works if the generation-prompt render is an exact prefix of the
training render (the prefix check below). TRL's `assistant_only_loss` needs
`{% generation %}` tags, which the Qwen template does not have, so build the
mask this way or use a trainer with a Qwen3 agent template.

## The template kwargs change the text

Render each sample with the `chat_template_kwargs` its request logged:

- **`preserve_thinking`.** Off, the template drops the thinking of assistant
  turns before the last user message (tool results do not count as user
  messages). The harness opens every turn with a user message, so off, an
  earlier turn's training context differs from what the model saw when it
  wrote it, and a whole game cannot be one sample. Unset, the official
  template treats it as on. The harness sends `true` unless
  `ARC3_PRESERVE_THINKING` is emptied.
- **`reasoning_effort`.** Unset, the template defaults to `xhigh` and adds
  "Reasoning effort is set to xhigh. Please think carefully..." to the start
  of the system message; `medium` adds nothing; `low` adds a line asking for
  brief thinking. The harness sends it only when
  `ARC3_REASONING_EFFORT_LADDER` steps down after a truncated reply, so most
  requests get the `xhigh` line.
- **`enable_thinking: false`** makes the generation prompt
  `<think>\n\n</think>\n\n`. Not used in games.

## Checking a template

[scripts/check_chat_template.py](../../scripts/check_chat_template.py)
renders with jinja2 set up like transformers' `apply_chat_template`, so it
needs no model or tokenizer download.

**1. Get the templates.** The official one, pinned to the commit checked
here:

```bash
curl -sSLO https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/chat_template.jinja
```

The one actually served in run A is a different file: the
`chat_template.jinja` of dfranzen's checkpoint, passed to SGLang with
`--chat-template` (the source is
`/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1/`
on Kaggle). It was not available here. Diff the two before relying on
either. A `tokenizer_config.json` holding a `chat_template` key works as
input too.

**2. Render a sample game.**

```bash
uv run --no-sync python scripts/check_chat_template.py chat_template.jinja
uv run --no-sync python scripts/check_chat_template.py chat_template.jinja --compare kaggle_chat_template.jinja
uv run --no-sync python scripts/check_chat_template.py chat_template.jinja --kwargs '{"preserve_thinking": false}'
```

It prints the full text, the trained text of each assistant turn, the prefix
check and the pitfalls (whether `reasoning` is dropped, what string
arguments do, how the generation prompt ends, whether the harness kwargs
change the render). The exit code is non-zero when the prefix check fails.

**3. Check real logs.**

```bash
uv run --no-sync python scripts/check_chat_template.py chat_template.jinja \
    --request-log runs/<run>/<game>_p0_requests.jsonl.xz
```

Every logged request is normalized and rendered with its reply and its
logged kwargs. It reports requests whose prefix check fails, replies with
invalid arguments, render errors (an image in a system message raises) and
how many requests extend the previous one.

**4. Check against the server.** The script shows what a template produces,
not what the server feeds the model. Send one logged request to the server
and compare `usage.prompt_tokens` with the token count of the rendered
prompt (vLLM can also return the tokens from `POST /tokenize` with
`messages`). A mismatch means the server uses another template or other
kwargs, or drops a field. One to look at: the harness sends past reasoning
under `reasoning`, while the template reads `reasoning_content`. The
[write-up](../../../WRITEUP.md) reports that removing retained thinking
costs a lot of score, so it reaches the model in the competition setup, but
confirm it on any other server (llama.cpp reads only `reasoning_content`;
see `_reasoning_history_keys` in `inference/agent/tool_agent.py`).

## Results for the official template

`Qwen/Qwen3.8-Flash-Next` at `de4b8e4`, harness kwargs
(`{"preserve_thinking": true}`):

| check | result |
| --- | --- |
| prefix check, sample game (3 assistant turns, 2 user messages) | OK |
| prefix check with `preserve_thinking: false` | fails on the two turns before the second user message |
| generation prompt | `<\|im_start\|>assistant\n<think>\n` |
| reasoning under `reasoning` | dropped |
| arguments as a JSON string | `TypeError` |
| kwargs `{}` vs `{"preserve_thinking": true}` | same text (unset means on) |
| `reasoning_effort` unset | `xhigh` instruction added to the system message |
| `tokenizer_config.json` `chat_template` vs `chat_template.jinja` | same render |
| special tokens | end of turn and EOS `<\|im_end\|>`, padding `<\|endoftext\|>` |

Images in user or tool messages render as
`<|vision_start|><|image_pad|><|vision_end|>`; the processor, not the
tokenizer, expands the pad to the image's tokens, so samples with images
need the model's processor in the training pipeline.

## Not covered

- The served Kaggle template (not reachable from this session).
- A converter from request logs to training JSONL; the normalization, the
  per-request split and the merge rule above are what it needs.
- Training itself. The served checkpoint is Intel's W4A16 AutoRound
  quantization: fine-tune the bf16 model (LoRA or full), then quantize, and
  redo the REAP pruning of [models/](../../../models/README.md) if used.
