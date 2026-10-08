# First Sol NLL panel: complete 512/256 results

Complete results for the [original frozen 30 requests](../sol-nll-fold0-30/README.md),
six per fold-0 game. These use the original Sol reasoning traces; the later
[generated-thinking panel](../sol-nll-fold0-30-genthink/README.md) is a separate run.

All **60 request/model results and 30 paired requests** are checksum-verified.
Scoring uses teacher-forced cross entropy in nats, with the full multimodal
context and losses on final-reply tokens only. Each model scores 12,108 tokens.
The primary metric is the equal-game mean of population-weighted request mean
NLL. Category metrics pool token losses with the same sampling weights.

| Experts | Primary NLL | Change vs 512 |
| --- | ---: | ---: |
| 512 | 0.85856035 | baseline |
| 256 | 0.85335082 | -0.6068% |

The initial pair passed the inclusive +5% gate. No 448/384/320 forwards ran.
The latest two-panel protocol defers further evaluation until the user reviews
both initial pairs. These results establish imitation NLL, not training or
gameplay performance.

## Storage and provenance

- `results.zip.dvc`: Git pointer to `results.zip` in the existing private
  `storage` remote (`s3://kaggle-arc-agi-3-dvc`).
- `provenance.json`: archive SHA256, frozen panel/run identities, model version,
  software versions, scoring runtime hash and completed-result verification.
- `comparison.csv`, `per-game.json`: small scalar summaries tracked in Git.

The archive contains all 60 FP32 per-token NLL arrays and exact scored positions
(`results/<experts>/<sample-id>.npz`), their JSON completion/checksum records,
manifest, run configuration, smoke and CUDA-kernel qualification records,
training-only expert maps, folds, completed heartbeat, selection/coverage reports,
category/per-game slices and an interactive token-loss HTML report. It also
preserves the exact `runtime-code/nll` and `runtime-code/reap` used for scoring;
their combined hash matches the run configuration. `inventory.json` provides a
SHA256 for every payload file except itself.

Failed or unqualified GPU attempts, collector state, process IDs, credentials,
model binaries and generated wheels are excluded. The manifest and HTML contain
teacher text, so the archive stays in the same private DVC remote as the source
transcripts. Source runtime is preserved from the qualified package at commit
`0c2d091`; report regeneration may add metadata without changing recorded losses.

## Fetch and inspect

From the repository root:

```bash
dvc pull data/sol-nll-fold0-30-results-20261008/results.zip.dvc
python -m zipfile -e data/sol-nll-fold0-30-results-20261008/results.zip /tmp/sol-nll-first-results
python /tmp/sol-nll-first-results/runtime-code/nll/report.py --out /tmp/sol-nll-first-results
```

Run report generation in the documented CPU environment (NumPy is required).
It checks the manifest and every result identity, array checksum, token position,
loss-vector length and sum. Open `comparison.html` to inspect token losses, or
read `slices.json` for thinking, Python-code and formatting losses per game.
The archived report is already generated; inspecting it requires no GPU.
