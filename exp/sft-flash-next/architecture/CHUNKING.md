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

The measured16K backward GPU allocated peak is19.4697GiB versus21.9467GiB
in the earlierOpt8 capture (2.4770GiB/~11.3% lower); forward remains14.5352GiB.
These are separate captures, not a paired HC-only memory experiment. At130K,
each full normalized stream or expanded projection output is2.47955GiB BF16;
an8192-row tensor of the same width is0.15625GiB. The projection temporaries
remain full length, while norm/gate/product/injection scratch is windowed and
recomputed. Full residual inputs/outputs remain too. No130K peak is implied.

## Opt10: PLE

Prepared CPU embedding rows come from the complete sample, including original
n-gram history and EOS handling. Window slices transfer only required prepared
rows to GPU inside replay. All projections/norm/gates and the short convolution
are checkpointed together. The left halo is derived from native convolution
kernel and dilation: `(kernel_size - 1) * dilation = 9` for this model. Only
current outputs are retained; autograd slice backward sums overlap input
contributions. Padding/conv masks are sliced with their windows.

### PLE storage bottleneck observed during 64K startup

The current table paths under `/tmp/reference-256-hf` are symlinks into the
read-only Kaggle model mount. Inspection found **NFS v3**, with a 524288-byte
maximum read size, rather than a local disk. The original table uses 22
safetensors files containing 128 BF16 tensors. A row is 160 BF16 values (320
bytes). `DiskPLERows.lookup` deduplicates/sorts IDs and gathers rows through
NumPy memory maps, opening each required shard separately.

At64K, 224793 unique rows contain 71933760 bytes (68.6MiB) of useful values.
Preparation took620.0624s. The worker's cumulative startup/preparation counter
reported24675921920 read bytes (22.98GiB), with about23s of CPU time. This
counter also includes imports and sample loading; it is not a lookup-only
counter. These observations support sparse network page faults/read-ahead as
the bottleneck. The local writable working volume is20GiB ext4, about18GiB available
at inspection, so the95.37GiB table cannot be copied there in full.

For a fixed training dataset, the intended correction is a separate preparation
stage: persist each sequence's deduplicated required rows and mapping in compact
contiguous local files, with model/input identities and checksums, then assemble
the CPU payload from those files each epoch. The full original table remains
available for preparing new sequences. This persistent prepared-row cache is
**not implemented yet**; the current DataLoader repeats sparse NFS lookups for
each new sample preparation. The second F/B benchmark repeat already reuses the
current in-memory payload and does not measure another disk preparation.
The first64K F/B took1151.6877s, longer than620.0624s preparation, so lookahead
could hide this cost after startup if the concurrent preparation keeps pace.
Steady-state overlap has not been measured; one-sample benchmarks do not prove it.

Disk inventory during96K startup confirmed17.323GiB available on the writable
working volume (19.518GiB formatted capacity). `/tmp`, `/var/tmp`, `/root` and
`/mnt` fail real write probes with `EROFS`, despite the overlay's reported rw
flag and976GiB free. The8TiB ext4 snapshot mount also reports `emergency_ro`.
A96GiB thin-pool backing device and1GiB metadata device are visible through
sysfs, but their allocation and mapper status cannot be inspected from the
container. Therefore the976GiB `df` value does not establish usable workspace
capacity. The cause of the read-only state is unverified; kernel logs are
inaccessible. The256GiB NVMe partition is exposed through read-only mounts.
`/dev/shm` provides86.5GiB of writable RAM-backed space, charged to host RAM.
No additional writable disk filesystem was found. Full table staging is not
viable on the confirmed working volume; compact prepared rows fit its budget.

### 96K capacity failure

Attempt `20261010075533242-f30e823d` was killed during its first forward, after
PLE preparation completed. There is no 96K loss or backward measurement. The
last incomplete observation recorded 170.005 GiB tree PSS; the post-failure
cgroup peak was 174.49 GiB against a 175 GiB limit, with no swap and oom_kill=1.
No pre-run counter snapshot exists, and max/oom counters are zero: this strongly
supports memory exhaustion but does not establish the exact OOM trigger.

Astra's capacity review identifies 48 full decoder checkpoint inputs and the
final mixer input, each BF16 `[1,T,10240]`. Their combined payload is 89.72 GiB
at 96K and 112.15 GiB at 120K. Inner chunking does not remove these outer
checkpoint boundaries. Do not dispatch 120K unchanged. A reviewed and qualified
capacity remedy is needed; none has been implemented. See
`reports/astra-capacity-review.md` and `reports/benchmark-96000-failure.json`.

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
- Opt10: accepted at1.780845%, cumulative withOpt7–9.
- 32K: twoF/B passes complete and verified; see `reports/benchmark-32000.json`.
- 64K: twoF/B passes complete and verified; see `reports/benchmark-64000.json`.
- 96K: first forward killed before loss/backward; failure evidence verified.
- 120K: not dispatched; blocked by host capacity pending a qualified remedy.

See `v0.md` for every measured phase and `reports/*-rerun.json` for reviewed
full-model comparisons. Rejected/interrupted/disconnected attempts remain
recorded in that table and localDVC; their historical implementation sources
travel with each immutable attempt. Only the original native-reference raw
adapter gradients are retained. No optimizer update or push is part of these
qualification runs.
