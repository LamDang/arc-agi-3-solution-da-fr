# SFT dataset — gpt-6.1-sol teacher, Qwen3.8-Flash-Next format

Supervised fine-tuning data for the student (Qwen3.8-Flash-Next), distilled
from the `gpt61sol-features-25games` run: gpt-6.1-sol playing the 25 public
games, 25/25 at 100. See the experiment notes in
`ARC3-Inference/exp/gpt61sol-features-25games` and the archived run
`ARC3-Inference/runs/gpt61sol-features-25games.dvc`.

**1334 samples, 25 games, levels 1–10.** Each sample reproduces one real model
request (so every sample fits the model's context) rewritten into the
rationale-off student format. All schema and format checks pass — see
[Exploration & validation](#exploration--validation).

The frozen 30-request fold-0 NLL evaluation subset is stored separately in
[`../sol-nll-fold0-30`](../sol-nll-fold0-30/README.md), with a DVC payload,
selected-request index and pinned source/processor provenance.

## Files

This directory is a self-contained DVC pipeline (`dvc.yaml`): **raw run → data
→ report**. The scripts, `dvc.yaml`, `dvc.lock` and `exploration.qmd` live in
git; the generated artifacts (`train.jsonl`, `meta.json`, `index.json`,
`exploration.html`) are DVC outputs, tracked by `dvc.lock` and stored in the
DVC remote. `dvc pull` fetches them; `dvc repro` rebuilds them.

| File | Tracked in | Stage | What it is |
| --- | --- | --- | --- |
| `dvc.yaml`, `dvc.lock` | git | — | The pipeline and its locked hashes. |
| `convert.py` | git | `convert` | Builds `train.jsonl` + `meta.json` from the run's request logs. Standard library only. |
| `build_index.py` | git | `index` | Builds `index.json` from `train.jsonl` (needs the Qwen tokenizer + template, `tokenizers`, `jinja2`). |
| `exploration.qmd` | git | `report` | Quarto EDA + anomaly report source. |
| `train.jsonl` | DVC | `convert` | The dataset, ~320 MB. One JSON line per sample. |
| `meta.json` | DVC | `convert` | Dataset-level and per-game sample counts. |
| `index.json` | DVC (~225 KB) | `index` | One row per sample: byte offset, level, token lengths, image count. Random access + bucketing. |
| `exploration.html` | DVC (~2.5 MB) | `report` | Rendered EDA + anomaly report. |
| `qwen/` | neither (re-fetched) | `fetch_model_files` | Pinned Qwen tokenizer + chat template, for token counts and rendering. |

## Sample format

One JSON line per model request:

```json
{"game": "ft09-0d8bbf25", "request_index": 3,
 "messages": [
   {"role": "system", "content": "...rationale-off system prompt..."},
   {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]},
   {"role": "assistant", "reasoning_content": "...\nNext step : ...", "content": "",
    "tool_calls": [{"type": "function", "id": "...", "function": {"name": "python", "arguments": {"code": "..."}}}]},
   {"role": "tool", "tool_call_id": "...", "content": "...tool output..."},
   {"role": "user", "content": [...]},
   {"role": "assistant", "reasoning_content": "...", "content": "",
    "tool_calls": [{"function": {"name": "python", "arguments": {"code": "..."}}}]}
 ],
 "tools": [{"type": "function", "function": {"name": "python",
            "parameters": {"type": "object", "properties": {"code": {...}}, "required": ["code"]}}}],
 "chat_template_kwargs": {"preserve_thinking": true}}
```

Top-level fields:

| Field | Meaning |
| --- | --- |
| `game` | Game id, e.g. `ft09-0d8bbf25`. |
| `request_index` | 0-based step within the game. |
| `messages` | OpenAI chat messages — see below. |
| `tools` | The code-only `python` tool (rationale-off). |
| `chat_template_kwargs` | Passed to the Qwen chat template at render (`preserve_thinking: true`). |

Rules:

- **One sample = one real request.** The messages are the exact context the
  harness sent for that step — already trimmed by the harness to fit the
  context — plus the reply. Because it reproduces a real request, every sample
  fits the model's window (largest ≈108K tokens; limit 139,264).
- **The last message is the only training target.** Mask everything before it;
  the earlier assistant turns are targets in their own samples. The earlier
  assistant turns are still rewritten to rationale-off form so the context
  matches what the student sees at inference.
- **`tool_calls[].function.arguments` is a mapping** (`{"code": ...}`), which
  is what the Qwen template's `arguments|items` needs — an OpenAI-style JSON
  string fails there.
- **Board images are kept** as `image_url` parts (base64 PNG, 640×640);
  Qwen3.8-Flash-Next is a vision model. Images repeat across the requests they
  appeared in, which is why the file is ~320 MB.
- **Render with the model's own `chat_template.jinja` and
  `preserve_thinking=True`.** Every sample's final turn passes the prefix check
  in `ARC3-Inference/scripts/check_chat_template.py` (see
  `ARC3-Inference/experiments/sft-format/README.md` for the template and loss
  mask).

## How it was built

The teacher played with `ARC3_PYTHON_RATIONALE=1`: its `python` tool took
`reasoning` and `description` fields next to `code`, and the model wrote its
stated reasoning into them (gpt-6.1-sol does not return its own chain of
thought, so those stated fields are the reasoning signal). The student plays
**without** that option — thinking goes in a `<think>` block and the tool call
carries only `code`. `convert.py` rewrites each request into that form:

- Each assistant turn's `reasoning` + `description` become `reasoning_content`,
  which the Qwen template renders as:

  ```
  <think>
  {reasoning}
  Next step : {description}
  </think>
  ```

- The tool call keeps only `code`. The teacher's own `reasoning` /
  `reasoning_details` (its summarized/encrypted chain of thought) are dropped.
- The **system prompt** is rewritten to its rationale-off form (the python-tool
  line becomes "call it with one ephemeral `code` string").
- The **tool schema** is rewritten to code-only (`reasoning`/`description`
  properties and the matching description sentence removed, `strict` dropped).

## Reading the data (random access & memory)

The file is ~320 MB, so **do not load it all into RAM** (~0.5–1 GB as objects).
A single sample is small (median ~0.25 MB JSON, max ~1.7 MB). Two ways to read:

- **Streaming:** iterate lines; each line is one independent sample.
- **Random access via `index.json`:** each row has the byte `offset` and
  `length` of its line — one `seek`, no scan:

  ```python
  import json
  index = json.load(open("index.json"))        # 1334 rows, ~225 KB
  row = index[i]
  with open("train.jsonl", "rb") as f:          # binary; train.jsonl is ASCII
      f.seek(row["offset"])
      sample = json.loads(f.read(row["length"]))
  ```

`index.json` rows also carry `game`, `request_index`, `level`,
`context_tokens`, `output_tokens` and `images`, so you can filter (e.g. by
level) or length-bucket batches straight from the index. Token counts use the
Qwen tokenizer with each 640×640 board image counted as 400 vision tokens.

Measured: context tokens median ≈51K, max ≈108K (all under the 139,264 window);
output tokens median ≈280, max ≈4.3K.

The real training-memory cost is the decoded images, not the file: a sample
with up to 60 images at 640×640 is ~70 MB of raw pixels once decoded, so decode
per batch in the dataloader.

At scale (many runs) convert to a memory-mapped format — HuggingFace `datasets`
(Arrow), WebDataset, or Mosaic MDS — for O(1) indexed reads without a custom
index.

## Exploration & validation

`exploration.qmd` is a Quarto report: dataset statistics (samples per game,
levels, token-length and image distributions, history-trimming over a game) and
a battery of anomaly checks (schema, rationale-off rewrite, tool-call shape,
`<think>` presence, image dimensions, duplicates, offset/context integrity).

It is the `report` stage of the pipeline: `dvc repro report` (or `dvc pull` to
fetch the already-built `exploration.html`).

Current result — **all schema/format checks pass**:

- 1334 samples, 25 games, levels 1–10; one distinct system prompt and one tool
  schema across the whole dataset.
- All 37,677 board images are 640×640 PNG.
- `index.json` offsets match a fresh stream and cover the file exactly;
  0 samples over context.
- The only flags: **1 expected** terminal text turn in `sk48` (the model ended
  the game with no tool call), and a handful of repeated-identical-code probes
  within 3 games (informational, not an error).

## Pipeline

`dvc.yaml` defines the whole chain:

```
convert            raw run (runs/gpt61sol-features-25games) -> train.jsonl, meta.json
fetch_model_files  Qwen tokenizer.json + chat_template.jinja  (pinned commit, not cached)
index              train.jsonl + tokenizer/template          -> index.json
report             train.jsonl + index.json                  -> exploration.html
```

Fetch the built artifacts:

```bash
dvc pull            # train.jsonl, meta.json, index.json, exploration.html
```

Rebuild from the raw run (offsets in `index.json` are byte-exact to
`train.jsonl`, so the pipeline always rebuilds the index after convert):

```bash
dvc pull ../../ARC3-Inference/runs/gpt61sol-features-25games.dvc   # the raw run
dvc repro                                                          # convert -> fetch -> index -> report
dvc push                                                           # upload the outputs
```

Prerequisites for `dvc repro`: the ARC3-Inference venv (referenced directly in
`dvc.yaml` for `pandas`, `tokenizers`, `jinja2`), `quarto` on `PATH` for the
report stage, and internet for `fetch_model_files`. Editing this README or
other docs does not invalidate any stage; each stage depends only on its own
script/source and its data inputs.
