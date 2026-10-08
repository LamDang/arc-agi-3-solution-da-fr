# Generated-thinking panel: verified 512/256 NLL results

Complete evaluation of the [generated-thinking 30-request dataset](../sol-nll-fold0-30-genthink/README.md),
derived from exactly the [original panel](../sol-nll-fold0-30/README.md).
All 30 processor-expanded prompts/images are identical between panels; only
final thinking differs. Python-code target tokens and training-only expert maps
match. System prompts, tool schemas and every previous/final call use Python
with only the `code` field.

All **60 evaluations** and **48,716 per-token losses** are retrieved and
checksum-verified. Scores use teacher forcing, full contexts and final-reply
losses only; no training or gameplay ran. Category NLL uses sampling-weighted
token pooling, matching the original breakdowns.

| Experts | Thinking NLL | Python-code NLL | Overall primary NLL (diagnostic) |
| --- | ---: | ---: | ---: |
| 512 | 1.38660827 | 0.31544506 | 1.04342870 |
| 256 | 1.38012803 | 0.32119906 | 1.03426631 |
| 256 change vs 512 | **-0.4673%** | **+1.8241%** | -0.8781% |

The user selected **thinking-to-thinking and code-to-code** comparisons.
At 512 experts both category losses beat the original Sol panel (1.99065077
thinking, 0.35306809 code). This generated-thinking pipeline becomes the
reference. Both 256 category changes pass the inclusive +5% gate; **256 is
accepted and no 448/384/320 evaluations ran**. This is an NLL decision, not a
claim about training capacity or gameplay performance.

## Payload and provenance

`results.zip.dvc` points to a private ZIP in the existing repository `storage`
remote. Git tracks its pointer, `provenance.json` and scalar reports. The ZIP
contains all FP32 token-loss vectors, scored positions and completion/checksum
records, frozen manifest, run configuration, maps/folds, CPU/GPU qualification,
worker/launch records, reports, semantic/category slices, and the exact scoring
runtime, dependency locks and GPU-disabled notebook. The runtime hash matches
`run.json`. Separate analysis code preserves that immutable scoring identity.

`panel-comparison/` includes all 60 paired request/model rows, per-game/context
comparisons and 16,590 aligned Python-token pairs (source and generated-thinking
NLL plus delta). Thinking traces differ in content and length; they are paired
by request, not aligned token by token. The first panel's token-loss arrays are
in its [separate DVC archive](../sol-nll-fold0-30-results-20261008/README.md).
`inventory.json` checksums every payload file except itself. Model weights,
processor binaries, generated wheels, credentials and collector/PID state are
excluded. Teacher text in the manifest/HTML stays private inside the ZIP.

Prepared manifest SHA256:
`15432affadbfbcba127f7ce253cbf096db17f92955848619686cc2c70a5e37ea`.
Dataset revision: `debc8a3fcefd68809b45f1d6cd72d6a0f6a15787`.
Scoring run identity:
`89771445a3607f3c402bd5b87afce7331d9bfbe7c12647daff8ed03fa64420e8`.
The worker used a user-authorized 120-minute limit and finished in 2,333.4
seconds. Exact per-request timings and GPU peaks are in completion records.

## Fetch and reproduce the analysis

```bash
dvc pull data/sol-nll-fold0-30-results-20261008/results.zip.dvc \
  data/sol-nll-fold0-30-genthink-results-20261008/results.zip.dvc
python -m zipfile -e data/sol-nll-fold0-30-results-20261008/results.zip /tmp/sol-nll-source
python -m zipfile -e data/sol-nll-fold0-30-genthink-results-20261008/results.zip /tmp/sol-nll-genthink
python /tmp/sol-nll-genthink/runtime-code/nll/report.py --out /tmp/sol-nll-genthink
PYTHONPATH=/tmp/sol-nll-genthink/runtime-code/nll \
  python /tmp/sol-nll-genthink/analysis-code/compare_panels.py \
  --source /tmp/sol-nll-source --variant /tmp/sol-nll-genthink \
  --out /tmp/sol-nll-paired --reference-metric categorywise
```

Use the documented CPU environment with NumPy. These commands validate results
and regenerate reports without allocating a GPU. The original reference criterion
in the scoring runtime remains recorded for provenance; `panel-comparison/decision.json`
records the final categorywise user decision.

For a fresh scoring rerun, use the [variant preparation instructions](../sol-nll-fold0-30-genthink/README.md)
and [offline Kaggle packaging/runtime instructions](../../exp/sft-flash-next/nll/README.md).
Run the archived worker with `--initial-pair-only`, its pinned model and
`--reap-dir` pointing at `runtime-code/reap`; start its collector first and
use a fresh output directory. No GPU rerun is launched by this documentation.
