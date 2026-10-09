# Capacity evidence

The complete evidence is in a checksummed ZIP in the repository's existing
private DVC remote. Keep generated traces and duplicate source snapshots out
of the code diff. From the repository root:

```bash
dvc pull exp/sft-flash-next/train/capacity-results.zip.dvc
python -m zipfile -e exp/sft-flash-next/train/capacity-results.zip exp/sft-flash-next/train
```

The ZIP contains `capacity-results/inventory.json` with a SHA256 for every
member. `../capacity-results.provenance.json` records archive SHA256, MD5 and
byte count. `../archive_capacity.py` builds and independently validates it.
Both successful and failed experiments are retained. No updated adapter is
included. The headline measurements remain directly readable in
[LONG_CONTEXT.md](../LONG_CONTEXT.md) and `../verification.json`.

Results are immutable observations of their adjacent source snapshots. Current
production code is in the parent directory; do not run an archived prototype
as the current recipe. Capacity steps discard their optimizer updates.

- `20261008-rtx96*`: initial OOM attempts, real-composite provenance, storage
  mount/device investigation and physical I/O benchmarks.
- `20261008-overnight/norm-130k`: bounded RMS, interrupted before a full step.
- `fused-130k`: early BF16 Triton attention, OOM at the PLE layer.
- `hyper-130k`: first complete 130K step; prototype attention, not final code.
- `matrix`: prototype memory/speed/disk ablations. Includes device-counter
  evidence that the fast disk-prefetch reads were served from file cache.
- `lowmem`, `gradient-diagnostic`, `precise-gradient`, `precise-gradient32`,
  `masked-tail-gradient`: retained failures and numerical investigations.
- `same-forward`: independent attention backward at identical full-model
  forward values and expert choices.
- `qualified`: corrected FP32/TF32x3 attention and whole-model gradient gate;
  full 90K/130K and disk steps passed, the final 77 GiB allocator-cap case OOMed.
- `ple-window`: explicit convolution-halo PLE recomputation and its independent
  GPU gradient check, with layer-group/disk/allocator-cap comparisons.
- `final`: combined PLE windows and bounded GDN gated normalization. Consult
  `completed-cases.json` and `result.json` for actual outcomes, including any
  last-case failure; do not infer suite success from the directory name.
- `resident`: stopped during slow NFS memory-map-to-RAM table copying, before
  any full sample ran. Retained as a loader-performance failure.
- `resident-stream`: bounded parallel file-to-RAM table copy, adaptive RMS
  blocks, full-context comparisons and differentiable GDN chunk-size sweep.
- `gdn-blocks`: rejected whole-GDN convolution layout; CUDA gradient gate
  failure and diagnostics retained. No full-length success is claimed here.
- `gdn-blocks-layout`: corrected native convolution layout, independent CUDA
  gradient checks, full 90K/130K steps and the final 75 GiB cap failure.
- `attention-blocks`: token-local projection checkpointing around unchanged
  full-sequence indexed attention; standalone gradient gate and capacity cases.
- `pipeline-tests*`: portable runtime regression results, including earlier
  packaging/CPU-autodispatch failures and corrected runs.
- `gdn-segments`: independent state-carry CUDA primitive check, including
  the initial-state gradient and a loss restricted to the final outputs/state.
- `validation`: all 30 held-out requests scored with a fresh untrained base;
  exact source snapshots, per-token losses and its specific recipe retained.
- `processor-comparison.json`: all 30 prepared requests have identical tensors
  under upstream/attached processors; historical full-input hashes differ.
- `runtime-lock.json`: actual GPU package versions and native convolution hash.

`summary.json` is derived by `../summarize_capacity.py` from completed cases,
per-layer events and periodic memory samples. Timings exclude model loading,
CPU encoding/deserialization and durable production checkpoint writes. Warm
repeats reuse the same composite and PLE rows. A successful Blackwell allocation
cap is not an A100 hardware test.
