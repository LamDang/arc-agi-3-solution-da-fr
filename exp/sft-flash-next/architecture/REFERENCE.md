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
Unselected experts can have absent gradients; their mathematical zero gradients
are exported explicitly and their names recorded, rather than claimed nonzero.

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

Measurement pending. Neither the old744-adapter result nor a short operator
fixture gives the new full-model loss or full16K memory peaks.
