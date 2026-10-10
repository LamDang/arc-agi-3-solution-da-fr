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

User gate: bitwise gradients or **global gradient relative L2 < 1%**. Each capture
first compares all 74,472 FP32 adapter gradients with the saved native-head
reference, then executes one additional F/B on the same model with all chunk
sizes zero. This isolates chunking from Opt3's previously measured 1.7842%
difference against the native-head reference. Both comparisons are reported;
the isolated comparison is subject to the 1% chunking gate. No raw gradients from
either candidate or its control are written to disk. Each control phase is
separately measured, and retains candidate gradients in CPU RAM while it runs.

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

Mixing/norm/gate work is checkpointed by token window. Injection is checkpointed
separately after attention or MLP. Attention still receives the complete mixed
sequence. Native FP32 injection coefficients retain their dtype; the injection
result is converted at the same effective BF16 boundary as the reference (before
the next mixing input or at decoder output). The final model stream mixer is
chunked as well. Full residual inputs/outputs remain allocated.

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

## Prepared diagnostics after the connection failure

Opt7 v3 observed loss0.6211259365 still differs from its expected unchunked
control. Its detached backward/control result has not been collected because
the Jupyter connection stopped responding. Do not promote it as numerically
qualified. The next revision adds native-order routed reduction and zero-padding
only terminal slices of split experts to8192 rows. Zero-padding does not introduce
extra routed tokens; discarded padded rows have zero output cotangents. Its
real-size fixture uses hidden2560/intermediate640/rank16 and9705 assigned rows
against an8192-row chunk, with a bitwise forward gate and <1% gradient gate.
**That revision and new fixture are prepared, not GPU-validated yet.** Reconnect
and collect the existing attempt first. Opt8–10 full-model captures are pending.
