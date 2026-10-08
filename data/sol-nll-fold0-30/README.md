# Frozen Sol NLL dataset: fold 0, 30 full requests

A **30-request held-out evaluation subset** of
[`sft-gpt61sol-features-25games`](../sft-gpt61sol-features-25games/README.md),
with complete multimodal contexts and the recorded final teacher replies.
Six requests per fold-0 game: two uniformly sampled without replacement from
each within-game context-length tertile, seed **20261008**. This is the exact
panel prepared for the 512/448/384/320/256-expert NLL comparison; no requests
were rerolled, truncated or replaced. These are evaluation data, not training
or calibration examples.

## Selected requests

Indices are the source dataset's `request_index`, not JSONL row numbers.

| Full game ID | Selected request indices (sorted for display) |
| --- | --- |
| `sk48-d8078629` | 39, 42, 82, 85, 99, 100 |
| `sp80-589a99af` | 0, 8, 9, 11, 22, 26 |
| `tn36-ef4dde99` | 5, 9, 11, 12, 24, 28 |
| `cd82-fb555c5d` | 3, 11, 18, 21, 26, 36 |
| `ar25-0c556536` | 0, 2, 10, 11, 18, 19 |

The JSONL preserves the frozen manifest's evaluation order: interleaved games
within each draw/stratum round. `index.json` records this order and each
request's stratum, inclusion probability and population weight.

## Files and storage

| File | Storage | Contents |
| --- | --- | --- |
| `requests.jsonl` | Private DVC S3 remote | 30 complete request objects, one per line, about 5.7 MiB total. Includes all messages, base64 images, tools and template kwargs. |
| `requests.jsonl.dvc` | Git | Immutable DVC payload hash and size. |
| `index.json` | Git | Byte offsets/lengths, selected IDs, source byte ranges, sampling weights, exact token counts and per-row checksums. |
| `provenance.json` | Git | Source revisions/hashes, fold and sampler identity, processor hashes, coverage and dataset checksum. |
| `export.py` | Git | Standard-library export from the verified panel, and standalone dataset verification. |

The payload uses the repository's configured `storage` remote,
`s3://kaggle-arc-agi-3-dvc`. Credentials come from the environment. Git contains
metadata and a DVC pointer; the transcript payload is stored in the existing
private remote. No processor binaries, model weights, calibration statistics
or wheelhouse are duplicated here.

## Fetch, verify and read

From the repository root, with DVC's S3 support and configured AWS access:

```bash
dvc pull data/sol-nll-fold0-30/requests.jsonl.dvc
python data/sol-nll-fold0-30/export.py
```

Read requests in evaluation order:

```python
import json
from pathlib import Path
root = Path("data/sol-nll-fold0-30")
with (root / "requests.jsonl").open() as stream:
    requests = [json.loads(line) for line in stream]
assert len(requests) == 30
```

Random-access a selected request using the corresponding `index.json` row:

```python
rows = json.loads((root / "index.json").read_text())
row = next(r for r in rows if r["sample_id"] == "sk48-d8078629-r39")
with (root / "requests.jsonl").open("rb") as stream:
    stream.seek(row["offset"])
    request = json.loads(stream.read(row["length"]))
```

## Provenance and reproducibility

The teacher is `gpt-6.1-sol`, from the repository's
`gpt61sol-features-25games` run. The source dataset converts logged rationale
and next-step description into the student's thinking field and retains
code-only Python tool calls. This subset performs **no further message,
image, code or target transformation**: only canonical JSON serialization.
Every exported object was compared to the frozen source object for equality.

- Source dataset revision: `37fadfeedfd54d2129db4525e4167596cc7337b6`.
- Source DVC MD5: `train.jsonl` = `b1b7eaed4fe609748d1980751354b9f2`;
  `index.json` = `370ec02490f6567c91bc6ef5140ac172`;
  `meta.json` = `0170a4199d41eb05ad15ca1134afcba2`.
- Split: [`data/game_folds/folds.json`](../game_folds/folds.json), fold 0.
- Sampling implementation: `sample_panel` in
  [`prepare.py`](../../exp/sft-flash-next/nll/prepare.py), at preparation
  commit `d162b9a`, sampler `two-per-game-context-tertile-v1`.
  Eligible rows are sorted by `(context_tokens, request_index, line)` and
  divided into three nearly equal-count strata. Each stratum uses
  `random.Random("sol-nll-v1:20261008:<full-game-id>:<stratum>").sample(..., 2)`.
- Pinned processor: `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Frozen panel manifest SHA256:
  `f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c`.
- This JSONL SHA256:
  `52a14ae74e4dcbb0ccc83038720c6c5b8b8eead20393af655a6f6dfde18610cc`.

`provenance.json` records the fold hash, every processor file hash and the
stratum populations/boundaries. `index.json` includes source row byte ranges,
frozen sample hashes and processor input/target hashes. JSONL offsets refer
to this export; `source_offset` and `source_length` refer to the original
333 MB source JSONL.

To recreate this dataset from the verified panel in a fresh directory:

```bash
python data/sol-nll-fold0-30/export.py \
  --bundle /path/to/verified/panel30 --out /tmp/sol-nll-subset
```

The exporter requires the exact frozen manifest identity and refuses to
overwrite existing outputs. It does not resample. See the
[preparation instructions](../../exp/sft-flash-next/nll/README.md) to rebuild
the full panel, processor annotations and training-only expert maps.

## Scoring and coverage

Condition on the **entire request**. Score only the final assistant reply,
including emitted end-of-turn formatting; earlier assistant messages, images,
tool outputs and the generation prefix are context. Use the pinned processor
and the final-target boundary logic in
[`data.py`](../../exp/sft-flash-next/nll/data.py), rather than treating all
assistant messages as targets. No tool execution or generation is needed.

There are **1,268,524 prompt tokens**, **12,108 target tokens** and **664
images**. Full processor-expanded request lengths range from **5,770 to
84,298 tokens**. The initial 512/256 comparison uses **2,561,264 processed tokens** and
**24,216 scored targets**. Only if 256 primary NLL exceeds 512 by more than 5%
are 448/384/320 evaluated, bringing the maximum to **6,403,160 processed tokens**
and **60,540 scored targets**. The unchanged provenance records maximum
five-candidate capacity; this protocol change does not modify the dataset. These counts depend on the pinned processor,
not on JSONL character counts.

The panel samples **15 of 35 game-level cells**; 20 are explicitly unsampled.
All 30 selected final replies contain tool calls; the source dataset's lone
terminal text-only reply was not sampled. Preserve the probability weights
when estimating population request NLL, and use matched requests for model
comparisons. No additional reserve is included.

The actual panel and packaged bundle passed full offline processor
reprocessing, and clean nested expert maps exclude all five validation games.
See [readiness](../../exp/sft-flash-next/READINESS.md) and the
[verification record](../../exp/sft-flash-next/verification/real-data.json).
No GPU session, training or gameplay has been started.

The complete first-panel evaluation is stored separately in
[the DVC result dataset](../sol-nll-fold0-30-results-20261008/README.md),
including per-token losses and per-game/category reports for 512 and 256.
