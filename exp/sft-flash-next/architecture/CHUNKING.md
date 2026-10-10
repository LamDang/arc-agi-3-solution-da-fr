# Opt7–10: bounded activation replay

The reference is unchanged. These are cumulative optimized configurations on top
of Opt3 (CCE exact, filters disabled, direct dense bias). Default chunk size is
8192 tokens. No context truncation, detach at a chunk boundary, optimizer update
or candidate raw gradient archive is permitted.

## Numerical qualification

On every implementation gate failure, request a GPT-6 Astra subagent source and
evidence review before accepting or continuing that implementation. The primary
agent performs fixes and execution. Record findings and their disposition.
The first Opt7 review is in `reports/astra-opt7-review.md`.

User gate, revised2026-10-10: bitwise gradients or **global gradient relative L2 < 2%**. Each capture
first compares all 74,472 FP32 adapter gradients with the saved native-head
reference, then executes one additional F/B on the same model with all chunk
sizes zero. This isolates chunking from Opt3's previously measured 1.7842%
difference against the native-head reference. Both comparisons are reported;
the isolated comparison is subject to the 2% chunking gate. No raw gradients from
either candidate or its control are written to disk. Each control phase is
separately measured, and retains candidate gradients in CPU RAM while it runs.

Opt7 v4 measured1.783424% and is accepted under the user's revised2% gate.
Its original1% failure and process exit remain preserved. The accepted report
records both policies; this acceptance does not establish bitwise equivalence
or prove the discrepancy is solely BF16 rounding. New configs explicitly record
`chunking_gradient_limit=0.02` for the cumulative comparison against Opt3.

## Opt7: routed experts

Routing and the native global argsort occur before expert execution. A custom
autograd operation gathers at most 8192 assigned rows for one expert at a time,
without building a full expanded hidden-state dispatch buffer. It uses unchanged
quantized projections and SiLU. Unweighted expert slot outputs are temporarily placed in CPU scratch. Native
routing multiplication and top-k sum run in bounded GPU token windows in their
original slot order; no reordered scatter sum substitutes for the native reduction.
This is per-expert row chunking: it avoids changing each expert GEMM shape when
that expert already has fewer than 8192 rows. The first input-sequence chunking
implementation is retained as rejected evidence if its paired gate fails. Backward stages the layer again, replays one
chunk, consumes its VJP immediately and releases that graph. It returns input
and routing-weight gradients plus accumulated FP32 adapter gradients through a
CPU concatenation into the original CPU LoRA Parameters. Trainable expert LoRA
adapters are included; only packed base weights are frozen. No staged GPU state
is captured by a deferred Python closure. Current/next frozen expert staging
remains bounded to two layers. Full input/output and routing decisions remain.

## Opt8: QSA queries

Native Q/K/V projections, normalization, rotary embedding and output projection
retain their original full-sequence GEMM shapes. A custom autograd operation
chunks selection-bias construction and SDPA only. Full Q, gate and assembled
attention output remain, in addition to full-context K/V. Backward replays each
query window and accumulates shared K/V gradients in FP32, then casts once to
the original BF16 gradient dtype. Each query uses its absolute position and full-context
keys. The bias is allocated inside replay, including its sentinel column and
alignment padding, with at most chunk_tokens query rows. Frozen indexer q/raw
keys are projected over the full sequence. Selected row IDs are retained,
selection is nondifferentiable as in the native implementation, and backward
rebuilds only each window bias from those IDs. It does not run selection a third
time. The outer decoder checkpoint still repeats selection as in the reference. Selected sets are checked bitwise against native
selection in fixtures. K/V gradients receive contributions from every window.
Current scope: one unpadded sample, SDPA, no cache, zero attention dropout.

## Opt9: hyperconnections

The first implementation changed the projection GEMM row counts and failed the
cumulative gate at285.42%. Actual layer0 shadows identify changed BF16 down and
injection projection outputs, despite bitwise normalization. Native repeat and
custom-unsplit match exactly; checkpointing and CPU offload do not change the
local0.2201% VJP discrepancy. This is direct operator evidence, not proof of the
whole-model amplification mechanism.

The revised candidate preserves all three full-sequence projection GEMM shapes.
Normalization and sigmoid/product/mean internals use8192-token checkpoints;
the entire mixer is also checkpointed so full norm/projection outputs are
recomputed rather than retained across layers. Residual injection has separate
8192-token checkpoints after attention or MLP. Full normalized sequence and
expanded projection outputs still exist as temporary tensors: this is not a
fully row-chunked mixer. Small low-rank and injection-coefficient arithmetic
remains full-sequence. Coefficients retain the native dtype (observed BF16 in
the actual layer0 diagnostic); native FP32 internal statistics are unchanged.
Attention receives the complete mixed sequence. Full residual inputs/outputs
remain allocated. The final model stream mixer uses the same scheme.
**Revised cumulative implementation passed at1.770794% against unchunkedOpt3.**

## Opt10: PLE

Prepared CPU embedding rows come from the complete sample, including original
n-gram history and EOS handling. Window slices transfer only required prepared
rows to GPU inside replay. All projections/norm/gates and the short convolution
are checkpointed together. The left halo is derived from native convolution
kernel and dilation: `(kernel_size - 1) * dilation = 9` for this model. Only
current outputs are retained; autograd slice backward sums overlap input
contributions. Padding/conv masks are sliced with their windows.

## Evidence and limits

`test_chunking.py` covers below/exactly/above chunk boundaries, multiple windows,
QSA full-key selection, all attention adapter gradients, mixing/injection input
gradients, EOS-sensitive PLE preparation and convolution halo gradients. A small
real quantized-expert fixture covers CPU FP32 LoRA and routing gradients through
outer layer checkpointing and inner replay. Full model measurements and the
chunking gate appear in the single `v0.md` table. Small CPU checks are preliminary;
GPU fixtures execute before model loading. No 130K pass is implied by a 16K run.

The accumulation distinction is visible in the pinned Torch2.11 CUDA sources:
[native indexing backward](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Indexing.cu)
and [BF16 sum reduction](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu).
The bounded unroute fixture compares the actual native gather VJP bitwise,
including the model hidden width, rather than assuming a reduction ordering.

## Current qualification status

- Opt7 v4: accepted at1.783424% against same-model unchunkedOpt3, under the
  user's revised2% gate. The original1% failure is immutable evidence.
- Opt8: accepted at1.801111%, cumulative withOpt7.
- Opt9 first row-chunked mixer: rejected at285.419802%. Its actual-input
  diagnostic and Astra review identify changed projection arithmetic.
- Opt9 revised projection-preserving mixer: actual layer0 three-port outputs,
  projection traces and input VJP are bitwise in all seven replay variants,
  including nested checkpointing andCPU offload. Full-model qualification is
  passed at1.770794% against unchunkedOpt3; operator proof preceded this full gate.
- Opt10 and32K/64K/96K/120K benchmarks: pending.

See `v0.md` for every measured phase and `reports/*-rerun.json` for reviewed
full-model comparisons. Rejected/interrupted/disconnected attempts remain
recorded in that table and localDVC; their historical implementation sources
travel with each immutable attempt. Only the original native-reference raw
adapter gradients are retained. No optimizer update or push is part of these
qualification runs.
