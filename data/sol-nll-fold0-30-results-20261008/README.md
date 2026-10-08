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

## Code to reproduce the run

The maintained implementation is committed under
[`exp/sft-flash-next/nll`](../../exp/sft-flash-next/nll/README.md):
`run.py` scores, `prepare.py` freezes the source requests, `fetch_inputs.py`
fetches pinned inputs, `collect.py` mirrors completed results, and `report.py`
produces token/category/game reports. `build_setup.py` builds the offline Kaggle
package; the [committed notebook](../../exp/sft-flash-next/nll/kaggle/kaggle-nll-30.ipynb)
defaults to GPU disabled and the current manual 512/256 pair.

For the **exact first-run implementation**, the DVC archive also includes:

- `runtime-code/nll/*.py`: the scorer, preparation/fetch, collector, reports,
  protocol, numerical fixes and result verification.
- `runtime-code/reap/*.py`: model loading, teacher-forced replay and rendering.
- `runtime-code/nll/requirements-cpu.lock` and `requirements-kaggle.lock`:
  pinned environments; the CPU lock must not replace torch on a GPU image.
- `runtime-code/kaggle-nll-30.ipynb`: the original GPU-disabled notebook.

The archived scoring Python files are unchanged and match `run.json`'s
`code_sha256`; adding dependency locks and documentation does not change it.
The original implementation uses the historical automatic +5% gate. The
maintained implementation adds `--initial-pair-only` for the latest two-panel
protocol. Use a new output directory when rerunning with changed code.

After extracting the archive as above, rebuild the original bundle on CPU:

```bash
NLL_CODE=/tmp/sol-nll-first-results/runtime-code/nll
python "$NLL_CODE/fetch_inputs.py" --out /tmp/sol-nll-first-inputs
python "$NLL_CODE/prepare.py" \
  --data-dir /tmp/sol-nll-first-inputs/data \
  --processor /tmp/sol-nll-first-inputs/processor \
  --stats-dir /tmp/sol-nll-first-inputs/calib \
  --folds /tmp/sol-nll-first-results/folds.json --out /tmp/sol-nll-first-panel
python "$NLL_CODE/run.py" --bundle /tmp/sol-nll-first-panel \
  --out /tmp/unused-first-preflight --preflight-only
```

Use the CPU environment from the archived lock. The rebuilt manifest must match
`f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c`.
Pinned processor/calibration hashes are validated during preparation. Preparation
preserves the original render-sensitive dictionary order; directly feeding the
sorted-key JSONL export bypasses this guarantee.

For an explicitly authorized GPU rerun, stage the extracted results/code and
prepared bundle on Kaggle, install the qualified dependencies following the
[CUDA kernel instructions](../../exp/sft-flash-next/nll/README.md#qualified-cuda-kernels-and-chunk-stable-scoring-2026-10-08),
and attach the immutable Intel model version recorded in `provenance.json`.
Start the archived collector on an external machine first, with Jupyter
credentials supplied through its environment:

```bash
python /tmp/sol-nll-first-results/runtime-code/nll/collect.py \
  --remote sol-nll-first-rerun --out /durable/sol-nll-first-rerun --interval 30
```

Then, on Kaggle (adjust only the staged directory paths):

```bash
python /kaggle/working/sol-nll-first-results/runtime-code/nll/run.py \
  --bundle /kaggle/working/sol-nll-first-panel \
  --out /kaggle/working/sol-nll-first-rerun \
  --model-dir /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1 \
  --reap-dir /kaggle/working/sol-nll-first-results/runtime-code/reap \
  --session-minutes 120
```

These commands are documentation; archiving them does not launch another run.
Model weights and native wheels are external inputs, not part of the result ZIP.

## NLL by input context length

K = 1,024 processor-expanded prompt tokens, excluding the final scored reply.
Each cell shows **512 → 256 experts**. These are diagnostic slices of the fixed
panel, not new samples. Primary NLL uses game/request weights within each band;
thinking and output NLL use weighted token pooling.

| Context | Requests | Primary NLL | Thinking NLL | Output NLL |
| --- | ---: | ---: | ---: | ---: |
| <16K | 4 | 1.280 → 1.261 | 2.696 → 2.749 | 0.511 → 0.457 |
| 16K–<48K | 14 | 0.823 → 0.818 | 1.957 → 1.969 | 0.348 → 0.351 |
| 48K–<80K | 11 | 0.810 → 0.811 | 1.947 → 1.930 | 0.293 → 0.291 |
| ≥80K | 1 | 0.239 → 0.231 | 1.626 → 1.649 | 0.075 → 0.063 |

The final band contains just one cd82 request and only 10.6% thinking tokens.
Length, game and reply composition vary together; these slices do not establish
that increasing context length causes lower NLL. Exact scalars, game counts,
Python-code-only loss and weighted thinking fractions are in
`by-context-length.json`; the full underlying slices are already in the DVC ZIP.

The previous committed pruning experiment reported **0.417 full-model NLL**,
not 4, on Qwen's own logged generations in ls20/sb26/vc33 (35,514 generated
tokens; prefixes up to 32K). See
[the earlier result](../../exp/reap-flash-next/README.md#pruned-routers-on-held-out-games).
The present panel scores Sol final replies in five different games with full
contexts and different calibration maps and request weights. The underlying
teacher-forced cross-entropy definition is unchanged. A separate earlier value
near 4 has not been identified; no numerical explanation for that value is
claimed without its log and metric definition.
