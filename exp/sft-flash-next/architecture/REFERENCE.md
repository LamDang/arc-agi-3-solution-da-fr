# Current reference

The user-defined reference (2026-10-10) is selected with `--architecture reference`
and `configs/reference.json`:

- Declared model activation ports return BF16. Native FP32 intermediate work,
  reduction statistics, scalar loss and saved statistics retain native precision.
- Every LoRA master parameter and its gradient stays FP32. CUDA autocast performs
  the linear matrix multiplications in BF16.
- LoRA rank16/alpha32 covers the existing attention/GDN/shared projections and
  gate/up/down projections of **all256 routed experts in all48 layers**.
  This gives **74,472 A/B tensors**, including73,728 routed-expert tensors.
- Routed quantized expert weights and their adapter masters live in CPU RAM.
  A bounded copy stream stages current/next expert layers. Frozen state uses
  pinned slabs; adapter copies remain differentiable. Checkpoint recomputation
  reverses prefetch order. Native expert selection and arithmetic are retained.
- The complete frozen PLE table stays on disk. One DataLoader worker prepares
  complete sample lookups with at most two outstanding preparations; the current
  payload is retained through backward/recomputation.
- Native dense attention masks, native vocabulary head and native normalization/
  SwiGLU are retained. CCE/direct-bias/Liger are optional later variations.

`native` preserves the earlier untouched HF/AutoRound baseline. `optimized`
accepts independent flags. The reference preset is fixed; choose `optimized`
with an explicit configuration to vary it.

## Capture versus training initialization

`reference.json` is a **diagnostic capture**, with nonzero A and B. Each tensor
uses CPU FP32 `torch.randn * 0.002`, with its generator seed derived from the
first8 little-endian bytes of SHA256(`run_seed + ':' + full_parameter_name`),
modulo2^63-1. Exact initial values are saved; they are the replay authority.
Production `train.py` rejects this option and uses PEFT random-A/zero-B.
Unselected experts can have zero gradients. Any absent routed-expert gradients
are exported as explicit zeros and their names recorded. The packed differentiable
copy normally gives unused expert leaves explicit zero gradients through cat
backward. Zero tensors are not claimed as positive gradient-equivalence evidence.

## Evidence and execution

```bash
node jupyter.mjs --config configs/reference.json --mode test --timeout-seconds 5400
```

Exactly one anchor forward/backward; no clipping, optimizer update or overfit.
The full16K anchor has16,249 tokens,651 targets and7 images. Forward/backward
record synchronized CUDA allocated/reserved peaks, timings and sampled parent
RSS/treePSS/childRSS/whole-host-used RAM. Initial adapters and all raw gradients
are saved in bounded Torch shards with SHA256 manifests. Large output uses
`/tmp` scratch exposed through the attempt's Jupyter output link; the Kaggle
20GB working volume need not hold multi-GB archives. Collection caches locally
with DVC. **No Git or DVC push.**

Measured16K capture passed with loss0.6244627833366394 and all74,472 raw FP32
gradients finite. Forward/backward GPU allocated peaks48.260/55.636GiB; sampled
tree PSS138.452/138.490GiB. See [reports/reference.md](reports/reference.md) for
full counters, initialization scope, hashes and evidence. No optimizer state
or130K capacity measurement is included.

## Replaying on a replacement server

The frozen model's expert selection is the original
`../train/artifacts/keep-256-fold0.json`, not the REAP smoke selection. Current
configs pin its semantic hash in addition to the model config and sample hashes.

`fla_numeric_profile="reference-v0"` selects FLA's native strict configuration
file for one reverse-scan key: B=1, H=48, BT=64, no variable lengths, reverse,
FP32 input/output; one warp, one CTA and three stages. Forward and other kernel
keys retain native autotuning. `numeric-profile.json` records the actual entry
after backward. No installed package source or callable is replaced.
This profile is **test-only**, including diagnostics and F/B capacity benchmarks.
Real training rejects a non-null `fla_numeric_profile` before CUDA imports.
Both the CLI and Jupyter launcher clear inherited FLA config paths and disable
FLA config-file overrides for training, leaving native Triton autotuning active.

On the replacement server, default autotuning picked two warps. Loss was exact
but full gradients differed by1.738071%. With a fixed input and upstream
cotangent, the isolated layer46 GDN matched all10 saved gradients using one
warp; default/two/four/eight warps reproduced the discrepancy. Full-model replay
`20261010040404197-7f737faf` then matched loss and all74,472 gradients bitwise,
including all74,394 nonzero tensors. Its actual backward configuration was
verified as one warp; see `reports/restoredreference-rerun.json`.
Use `configs/restored-reference-check.json` to check the saved reference without
retaining another raw gradient archive. See `reports/gdn-backward-isolation.json`
and `reports/astra-restored-reference-gradient-review.md`.
