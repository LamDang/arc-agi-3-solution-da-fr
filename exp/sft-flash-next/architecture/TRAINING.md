# Production trajectory training

## Dataset and objective

The latest export, `data/progressive-sol25-trajectories`, is imported from
`origin/codex/progressive-sol25-trajectories` at `ce2ba09`. Its export code and
DVC lock are included. Obtain the DVC outputs before launching; see its README
for pull/rebuild instructions. The release reported these new outputs as local
cache only. A Git checkout alone is not the dataset, and the older filtered
1,324-request export is not a substitute.

| Split | Games | Trajectories | Distinct turns | Target tokens |
| --- | ---: | ---: | ---: | ---: |
| Train, folds 1–4 | 20 | 47 | 1,066 | 1,122,317 |
| Validation, fold 0 | 5 | 11 | 268 | 277,426 |
| Total | 25 | 58 | 1,334 | 1,399,743 |

Counts above come from the pinned export report; the loader checks actual data,
index and target coverage at launch. The largest sequence is 129,169 tokens.
The hard cap is 130,000 **input plus output** tokens. Trajectories follow real
source compaction boundaries, retaining the source's observed context.

Each trajectory gets one forward/backward. Every turn is eligible, including
the ten excluded by the obsolete input-only rule. Only assistant indices in
`loss_target_message_indices` receive labels. Repeated retained assistant
history is masked to avoid duplicate supervision. Thinking, code, formatting
and end-of-turn tokens are targets; the generation prefix stays context.
One vision pass supplies inputs; text/token prefixes prove each assistant span
and image expansion. Exported counts must match exactly. Nothing is truncated
or silently filtered. Game-disjoint folds, data hashes and unique turn ownership
are verified. A seeded permutation shuffles each epoch. One worker prepares
disk PLE with at most two queued samples; each epoch/validation recreates its
iterator. Current prepared values survive through backward.

## Optimizer and validation

`configs/train-trajectories.json` selects final Opt10: BF16 activation ports,
FP32 GPU LoRA parameters/gradients on all experts, frozen expert buffers in RAM,
two-layer GPU prefetch, disk PLE, target-only CCE exact without filtering, direct
bias and all four 8,192-token chunk components. Production uses random A / zero
B and native FLA autotuning. Diagnostic adapters, initialization and FLA profiles
are rejected. Model component arithmetic is unchanged.

Accumulate summed token-CE gradients until **49,152 targets**; normalize once
by actual targets, then clip global norm to 1.0 and update. Whole trajectories
may overshoot. Flush the short remainder at **every epoch**, then validate.
AdamW: constant LR 1e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0,
`foreach=False`, `fused=False`. Maximum five epochs; no scheduler.

Beta 2 is 0.95 because this dataset is expected to produce only about 18–20
updates per epoch with the current accumulation budget. The shorter decay
lets the squared-gradient estimate respond faster as training changes the
gradients. Adam's bias correction already handles zero initialization; extra
epochs are not required just to initialize its statistics. This setting is
a training choice, not a measured improvement. Evaluate after the first epoch
before deciding whether to continue toward the five-epoch maximum.

Validation uses all fold-0 targets with `eval()`/no gradients. NLL and accuracy
are target-weighted across the split. CCE supplies summed NLL; bounded target
and vocabulary projections compute teacher-forced argmax, preserving first-index
ties without a full T-by-vocabulary matrix. Normal BF16 rounding applies.

## Durable checkpoint protocol

After each update, clear gradients and pause training. Stream 64 MiB payload
shards containing adapters, AdamW states/steps, RNG, epoch/order cursor, epoch
loss counters and identities. State is bound to stable parameter names. Frozen
model/PLE weights and raw gradients are never checkpointed.

The desktop verifies and fsyncs each shard before acknowledgment; only then
does the GPU worker delete its spool copy. A final manifest is acknowledged
after the complete snapshot verifies and is DVC registered. The previous
checkpoint survives until its replacement is durable. Copy-back can wait up to
24 hours, subject to supervisor/kernel lifetime. Restore also streams one
verified shard at a time, so neither direction requires a whole checkpoint on
Kaggle disk or in CPU RAM.

Snapshots live in `exp/sft-flash-next/training-runs/<attempt>/` in this repository.
Each run has a dedicated DVC cache with reflink/hardlink storage. Once a new
snapshot is durable, older update snapshots/pointers are retired and that run's
unused cache objects are collected. The repository's shared cache and remote
are untouched. Logs and `latest.json` also have DVC pointers. No automatic Git
commit or DVC push occurs.

Resume restores the next trajectory, optimizer and RNG; data, folds, processor,
model, config, package versions and source identities must match. A crash during
accumulation replays from the last complete update. A crash before validation
can repeat validation but does not repeat an optimizer update. CPU mock resume
is bitwise exact; native GPU kernels are not forced into test configurations.

Full adapters are ~7.16 GiB and AdamW moments ~14.31 GiB. A new local run requires
**46 GiB free** for safe replacement and headroom; an existing run needs another
23 GiB free at launch. Space is checked again per download. This Mac had only
15 GiB free during implementation. No existing artifacts were deleted.

## Kaggle API utility

Use the prepared pinned GPU runtime and 256-expert model from qualification.
`../train/requirements-gpu.txt` includes TensorBoard 2.20.0; preserve the runtime's
CUDA Torch and its established `/tmp/peft-autoround-compat/` package view. Model
and processor config paths must already exist. This utility does not create a
notebook or rebuild the frozen model export.

Keep the private Jupyter URL in `/tmp/kaggle_probe_url`, never config/Git.
From `architecture/`:

```bash
python train.py --config configs/train-trajectories.json --validate-only
node jupyter.mjs --mode train --config configs/train-trajectories.json \
  --dataset ../../../data/progressive-sol25-trajectories

# Reattach collection after client/network interruption.
node jupyter.mjs --action collect --attempt ATTEMPT \
  --local-run ../training-runs/ATTEMPT --timeout-seconds 86400

# New kernel: restore the last complete local update, reusing its run directory.
node jupyter.mjs --mode train --config configs/train-trajectories.json \
  --dataset ../../../data/progressive-sol25-trajectories \
  --resume ../training-runs/ATTEMPT

tensorboard --logdir ../training-runs/ATTEMPT/telemetry/tensorboard
```

The utility uploads dataset/index/summary, folds and a source snapshot. Optional
overrides: `--folds`, `--local-run`, `--url-file`, `--timeout-seconds`,
`--dvc-python`. DVC's Python is discovered from its executable shebang; supply
the override if needed. Training defaults to a 24-hour supervisor timeout;
Kaggle may stop earlier. Keep the desktop utility running for acknowledgments.
Status checks alone do not service transfers.

Standard DVC commands use the default cache. Select the run's cache when
checking/restoring these local outputs, for example:

```python
from pathlib import Path
from dvc.repo import Repo
run = Path("exp/sft-flash-next/training-runs/ATTEMPT").resolve()
with Repo(".", config={"cache": {"dir": str(run / "dvc-cache")}}) as repo:
    print(repo.status(targets=[str(run / "latest.json")]))
```

## Logging and verification

TensorBoard logs token-weighted train loss, gradient norm before clipping, LR,
actual targets/update, cumulative targets, tail status, throughput, checkpoint
time, validation NLL/accuracy and epoch train NLL. Forward/backward/optimizer
phases record synchronized GPU allocated/reserved peaks, sampled RSS/PSS/host/
cgroup RAM and elapsed time. Attempt logs remain separate; TensorBoard event
files are combined across resumes. No per-tensor gradient archives are written.

Local checks cover an independent token-weighted objective, clipping/flushes,
five epochs, folds, interleaved masks/image expansion, validation reductions,
bounded save/restore, exact CPU dropout/RNG/AdamW resume, corruption rejection,
TensorBoard events, simulated Jupyter transfers and real temporary DVC
publication/retention/shared-cache isolation.

Local verification on 2026-10-10: 60 Python tests passed, eight GPU/optional
checks skipped; five Node transport tests passed; the separate real-DVC test
passed. The Python suite includes a complete mock training → DVC publication
→ crash → restore → continued training comparison.

```bash
python -m pytest -q tests/test_training.py
DVC_PYTHON=/path/to/dvc/python python -m pytest -q tests/test_training_dvc_e2e.py
node --test tests/test_transfer.mjs
PATH_TO_DVC_PYTHON tests/test_dvc_checkpoint.py
```

Live Jupyter execution, real-data processor parity, optimizer-inclusive GPU/RAM
capacity and learning remain unqualified. Existing 120K F/B evidence excludes
AdamW state. The first live run must check optimizer allocation and subsequent
accumulation before committing to five epochs at up to 130K context.
