# SFT dataset: gpt-6.1-sol teacher, Qwen3.8-Flash-Next format

Supervised fine-tuning data for the student (Qwen3.8-Flash-Next), distilled
from the `gpt61sol-features-25games` run: gpt-6.1-sol playing the 25 public
games, 25/25 at 100 (see `ARC3-Inference/exp/gpt61sol-features-25games` and
the run archived at `ARC3-Inference/runs/gpt61sol-features-25games.dvc`).

- `convert.py` — builds `train.jsonl` from the run's request logs.
- `train.jsonl` — the dataset (DVC-tracked; `dvc pull` to fetch). One JSON
  line per game.
- `meta.json` — per-game turn counts.

## What the conversion does

The teacher played with `ARC3_PYTHON_RATIONALE=1`, so its `python` tool took
`reasoning` and `description` fields next to `code`, and the model wrote its
stated reasoning into them (gpt-6.1-sol does not return its own chain of
thought, so these stated fields are the reasoning signal). The student plays
**without** that option: thinking goes in a `<think>` block and the tool call
carries only `code`. The converter rewrites the teacher's runs into that form:

- Each assistant turn's `reasoning` and `description` become the turn's
  `reasoning_content`, which the Qwen template renders as:

  ```
  <think>
  {reasoning}
  Next step : {description}
  </think>
  ```

- The tool call keeps only `code`: `{"name": "python", "arguments": {"code": ...}}`.
  The teacher's own `reasoning`/`reasoning_details` (its summarized/encrypted
  chain of thought) are dropped.
- The **system prompt** is rewritten to its rationale-off form (the line
  listing the python tool's fields becomes "call it with one ephemeral `code`
  string").
- The **tool schema** is rewritten to code-only (the `reasoning` and
  `description` properties and the matching sentence in the description are
  removed).

## Format

Each line:

```json
{"game": "ft09-0d8bbf25",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]},
              {"role": "assistant", "reasoning_content": "...\nNext step : ...",
               "content": "", "tool_calls": [{"type": "function", "id": "...",
                 "function": {"name": "python", "arguments": {"code": "..."}}}]},
              {"role": "tool", "tool_call_id": "...", "content": "..."}],
 "tools": [{"type": "function", "function": {"name": "python",
            "parameters": {"type": "object", "properties": {"code": {...}}, "required": ["code"]}}}],
 "chat_template_kwargs": {"preserve_thinking": true}}
```

- **One sample per game**, the whole trajectory. Train with the loss on the
  assistant turns only; mask system, user and tool turns.
- `tool_calls[].function.arguments` is a **mapping**, which is what the Qwen
  template's `arguments|items` needs (an OpenAI-style JSON string fails there).
- Board images are kept as `image_url` content parts (base64 data URIs);
  Qwen3.8-Flash-Next is a vision model and the harness feeds it board images.
- Render with the model's own `chat_template.jinja` and
  `preserve_thinking=True`. See
  `ARC3-Inference/experiments/sft-format/README.md` for the template, the loss
  mask, and the checker. Every assistant turn in this dataset passes that
  checker's prefix check.

## Totals

25 games, 1334 assistant turns (1 of them a terminal text turn in sk48 with no
tool call). History is trimmed from the front in long games, so `convert.py`
reconstructs each trajectory across requests instead of reading the last
request alone; for every game the reconstructed turn count equals the number
of model responses.

## Rebuild

```bash
dvc pull ../../ARC3-Inference/runs/gpt61sol-features-25games.dvc
python convert.py            # writes train.jsonl and meta.json
dvc add train.jsonl && dvc push
```

`convert.py` needs only the standard library.
