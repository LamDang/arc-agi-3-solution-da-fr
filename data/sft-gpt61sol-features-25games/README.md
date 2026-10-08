# SFT dataset: gpt-6.1-sol teacher, Qwen3.8-Flash-Next format

Supervised fine-tuning data for the student (Qwen3.8-Flash-Next), distilled
from the `gpt61sol-features-25games` run: gpt-6.1-sol playing the 25 public
games, 25/25 at 100 (see `ARC3-Inference/exp/gpt61sol-features-25games` and
the run archived at `ARC3-Inference/runs/gpt61sol-features-25games.dvc`).

- `convert.py` — builds `train.jsonl` from the run's request logs.
- `train.jsonl` — the dataset (DVC-tracked; `dvc pull` to fetch). One JSON
  line per model request.
- `meta.json` — per-game sample counts.

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
{"game": "ft09-0d8bbf25", "request_index": 3,
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]},
              {"role": "assistant", "reasoning_content": "...\nNext step : ...",
               "content": "", "tool_calls": [{"type": "function", "id": "...",
                 "function": {"name": "python", "arguments": {"code": "..."}}}]},
              {"role": "tool", "tool_call_id": "...", "content": "..."},
              {"role": "user", "content": [...]},
              {"role": "assistant", "reasoning_content": "...", "content": "",
               "tool_calls": [{"function": {"name": "python", "arguments": {"code": "..."}}}]}],
 "tools": [{"type": "function", "function": {"name": "python",
            "parameters": {"type": "object", "properties": {"code": {...}}, "required": ["code"]}}}],
 "chat_template_kwargs": {"preserve_thinking": true}}
```

- **One sample per model request.** The messages are the exact context the
  harness sent for that step, already trimmed to fit the context, plus the
  reply. The **last message is the only training target**; mask everything
  before it (the earlier assistant turns are targets in their own samples).
  This reproduces the real requests, so every sample fits the model's context
  by construction — the run's largest request was ~120K tokens, and none of
  the 1334 samples exceed the 139K window. (The prior assistant turns in the
  context are still rewritten to rationale-off form, so the context matches
  what the student sees.)
- `tool_calls[].function.arguments` is a **mapping**, which is what the Qwen
  template's `arguments|items` needs (an OpenAI-style JSON string fails there).
- Board images are kept as `image_url` content parts (base64 data URIs);
  Qwen3.8-Flash-Next is a vision model and the harness feeds it board images.
  Images repeat across the requests they appeared in, which is why the file is
  large (~320 MB); that is the cost of each sample being a faithful request.
- Render with the model's own `chat_template.jinja` and
  `preserve_thinking=True`. See
  `ARC3-Inference/experiments/sft-format/README.md` for the template, the loss
  mask, and the checker. Every sample's final turn passes that checker's prefix
  check.

## Totals

25 games, 1334 samples (one per request; 1 of them a terminal text turn in
sk48 with no tool call). Rendered size per sample: median ~134K chars, max
~262K; estimated tokens all under the 139K context.

## Reading it (random access and memory)

The file is ~320 MB of JSONL, so do **not** load it all into RAM. A single
sample is small (median ~0.25 MB of JSON; the largest ~1.7 MB), so stream it or
index it:

- **Sequential / streaming:** read line by line; each line is one independent
  sample. Memory stays at one sample at a time.
- **Shuffled / random access:** plain JSONL is not addressable by index. Build
  a one-time byte-offset index (`offsets[i]` = the start of line `i`), then
  `seek(offsets[i]); readline()` for sample `i`. The index is 1334 integers.
- **At scale (many runs):** convert to a memory-mapped format — HuggingFace
  `datasets` (Arrow), WebDataset, or Mosaic MDS — for O(1) indexed reads
  without a custom index. Overkill for one run.

The real training-memory cost is not the file but the decoded images: a sample
with up to 60 board images at 640×640 is ~70 MB of raw pixels once decoded, so
decode per batch in the dataloader, not up front.

## Rebuild

```bash
dvc pull ../../ARC3-Inference/runs/gpt61sol-features-25games.dvc
python convert.py            # writes train.jsonl and meta.json
dvc add train.jsonl && dvc push
```

`convert.py` needs only the standard library.
