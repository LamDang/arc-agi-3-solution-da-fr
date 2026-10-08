# Generated-thinking fold-0 panel (sol-nll-fold0-30-genthink)

The [`sol-nll-fold0-30`](../sol-nll-fold0-30/README.md) panel with the teacher's
thinking **replaced by the student's generated thinking**. Each sample is the
same request — identical context messages, images, tools and the same
**code-only** Python call — but the final assistant turn's `reasoning_content`
is the thinking produced by the `think_gen` judge-and-refine pipeline
(two refine passes), not the rationale-off conversion of GPT-6.1 Sol's own
`reasoning`/`description`.

So each line is: **request context → generated thinking → `python(code)`** with
no description or rationale fields on the call.

**All 30 panel requests** are included. (The generation gate was removed, so no
record is dropped for leaks or length; the judge-and-refine loop handles quality
instead. `provenance.json` lists any exclusions — currently none.)

## Why

The frozen panel carries Sol's own stated reasoning (short, rationale-style).
This variant carries the fuller, first-person working the student actually
learns from, scored on the same panel: on these records the generated thinking
reached coverage 0.99, code-leads-to-call 1.0, fact-grounded 0.93 and
functional call-equivalence 0.80 (see
[`sol-judge.md`](../../ARC3-Inference/experiments/teacher-reasoning/sol-judge.md)).
It lets the two thinking sources be compared on identical inputs.

## Files and storage

| File | Storage | Contents |
| --- | --- | --- |
| `requests.jsonl` | Private DVC S3 remote | 30 request objects, one per line (~5.8 MiB). Same schema as the source panel; only the final `reasoning_content` differs. |
| `requests.jsonl.dvc` | Git | DVC payload hash and size. |
| `index.json` | Git | Per sample: byte offset/length, sample_id, key, original vs generated reasoning char counts, frozen prompt-token count, line checksum. |
| `provenance.json` | Git | Source dataset + md5, the think_gen run, manifest and pipeline commit, and the excluded records. |
| `build.py` | Git | Rebuilds `requests.jsonl` + `index.json` + `provenance.json` from the source panel and the pipeline's refine-2 output. |

## Fetch and read

```bash
dvc pull data/sol-nll-fold0-30/requests.jsonl.dvc          # source panel
dvc pull data/sol-nll-fold0-30-genthink/requests.jsonl.dvc # this variant
```

```python
import json
from pathlib import Path
root = Path("data/sol-nll-fold0-30-genthink")
samples = [json.loads(l) for l in (root / "requests.jsonl").read_text().splitlines() if l]
assert len(samples) == 30
last = [m for m in samples[0]["messages"] if m["role"] == "assistant"][-1]
assert set(last["tool_calls"][0]["function"]["arguments"]) == {"code"}  # code-only
```

## Provenance and reproducibility

- Source: `data/sol-nll-fold0-30` (frozen fold-0 panel; requests md5 in
  `provenance.json`).
- Thinking: `ARC3-Inference/runs/think-sol-nll30/refine2`, produced by the
  `think_gen` DVC pipeline (`tg_generate → … → tg_final`) on manifest
  `experiments/teacher-reasoning/evalset/sol-nll-fold0-30.json`. The
  `pipeline_commit` in `provenance.json` pins the code.
- Transformation: only the final assistant turn's `reasoning_content` is
  replaced; the context (including earlier assistant turns in their
  rationale-off form), the images, the tool schema and the Python code are
  byte-for-byte the source panel's. Rebuild with `python build.py`.

## NLL evaluation and tool-contract audit

The NLL audit of remote-main revision `debc8a3fcefd68809b45f1d6cd72d6a0f6a15787`
confirmed all 30 Python schemas accept only `code`; all 480 historical calls
and 30 final calls contain only `code`; every system prompt instructs a
code-only Python call. No Sol-only reasoning/description argument fields
needed removal. The generated thinking belongs in `reasoning_content`.

The export sorts JSON keys. Qwen's tool-schema renderer preserves dictionary
insertion order, so rendering the exported objects directly changes the
serialized tool prompt. For a controlled comparison, `prepare_variant.py`
restores the source frozen bundle's dictionary order while preserving every
logical value. It verifies all 30 processor-expanded prompt-token sequences,
images and final Python-code token sequences against that bundle. Only the
final thinking changes. It reuses the original training-only maps and sampling
weights; no resampling, truncation or validation calibration is performed.

```bash
python exp/sft-flash-next/nll/prepare_variant.py \
  --source-bundle /path/to/original/panel30 \
  --dataset data/sol-nll-fold0-30-genthink \
  --revision debc8a3fcefd68809b45f1d6cd72d6a0f6a15787 \
  --out /path/to/new/genthink-panel30
```

The prepared panel has 1,268,524 prompt tokens and 24,358 final-reply tokens:
15,193 thinking, 8,295 tool-code, 720 tool-format and 150 turn-format tokens.
The same 512-first/256-second protocol processes 2,585,764 tokens initially,
up to 6,464,410 if the relative primary-NLL increase exceeds 5%. Reports keep
thinking, tool-code and tool-format losses separate. Full contexts are
preserved, and final replies are scored with teacher forcing.

See [CPU audit](../../exp/sft-flash-next/verification/genthink-preflight.json).


## Completed NLL evaluation

This pipeline is the selected reference after comparing thinking and Python-code
NLL separately against the original Sol panel on all 30 identical prompts.
256 experts pass both +5% category gates. The
[verified result dataset](../sol-nll-fold0-30-genthink-results-20261008/README.md)
contains token losses, exact scoring/analysis code and the one-to-one comparison.
