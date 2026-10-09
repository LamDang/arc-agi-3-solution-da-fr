# Full-context Flash-Next training: measurements and implementation

## Scope

This work targets the repository's **Qwen3.8-Flash-Next (`qwen4_exp`) Intel
AutoRound W4A16 model, physically selected to 256 experts using the train-only
map**. It trains 33,478,656 rank-16 LoRA parameters in language attention, GDN
and shared-expert projections. Routed experts, routers, indexers, vision, PLE
and the output head remain frozen, with input gradients passing through their
computations. This is not full-weight training or NF4 quantization.

The benchmark convention is **120,000 context + 10,000 target = 130,000 total
tokens**, compared with **80,000 context + the same 10,000 target = 90,000**.
Every reported successful case completes forward, full backward, finite-gradient
checks, clipping and an Adam update. All diagnostic updates are discarded.

## Measured full-context steps

The corrected attention + windowed PLE + bounded gated-normalization run
(`capacity-results/20261008-overnight/final`) completed these cases:

| Context + target | Configuration | Step seconds | Peak GPU allocated / reserved GiB | Sampled host anonymous + shared GiB |
|---|---|---:|---:|---:|
| 120K + 10K | Two-layer groups, warm PLE rows | 125.43 | 80.01 / 85.10 | 66.10 |
| 80K + 10K | Two-layer groups, warm PLE rows | 88.76 | 67.48 / 69.73 | 46.63 |
| 120K + 10K | Three-layer groups | 127.57 | 80.01 / 87.23 | 46.29 |
| 80K + 10K | Three-layer groups | 89.94 | 67.48 / 70.08 | 32.90 |
| 120K + 10K | Two-layer groups, 32 GiB disk budget | 164.50 | 80.01 / 85.10 | 36.37 |

These timings exclude model loading, CPU encoding/deserialization and durable
production checkpoint writes. They include one full forward/backward/Adam
step, with all targets supervised. The same 10K targets make the two-layer
80K case **1.413× faster per sample**, a **29.2% reduction in step time**.
Total-token throughput is about 1,036 versus 1,014 tokens/s; target throughput
is about 79.7 versus 112.7 tokens/s.

The first 120K step took **351.27 s**, versus 125.43 s with its PLE rows warm.
The source model load took **681.88 s** in that run; a separate startup-only
weight prefetch is recorded in `startup-notes.json`. Neither number is hidden
inside the warm training rate. New requests may need different PLE rows.

The final 77 GiB allocator-capped case failed in FLA backward while allocating
`dv` in `prepare_wy_repr_bwd`. Bounded normalization solved the previous
allocation bottleneck; the full-sequence GDN workspace became the next one.
The later resident-PLE / segmented-GDN experiment is reported below.

### Resident PLE and segmented recurrence

With the complete frozen PLE table retained in host RAM, three-layer checkpoint
groups and adaptive RMS tiles, `resident-stream` measured:

| Context + target | Configuration | Step seconds | Peak GPU allocated / reserved GiB | Sampled host anonymous + shared GiB |
|---|---|---:|---:|---:|
| 120K + 10K | Unsegmented GDN, repeated | 121.92 | 80.01 / 87.23 | 141.45 |
| 80K + 10K | Unsegmented GDN, repeated | 86.16 | 67.48 / 70.08 | 128.06 |
| 120K + 10K | GDN segments 8,192, warmed | 123.07 | 76.02 / 78.27 | 141.52 |
| 120K + 10K | Same, **77 GiB allocator cap** | **123.26** | **76.02 / 76.86** | 141.53 |
| 120K + 10K | GDN segments 32,768 | 130.09 | 76.02 / 79.17 | 141.52 |
| 120K + 10K | GDN segments 65,536 | 136.22 | 76.02 / 80.47 | 141.53 |

All cases in that suite passed. The first 8,192-segment step took 139.73 s,
so warm comparisons exclude its new-shape compilation cost. The table copy
and pruning/configuration took 98.32 s with already-warmed source caches;
this is not a cold NFS load benchmark. Resident PLE avoids per-request table
page misses but consumes 95.37 GiB of non-reclaimable host memory.

`gdn-blocks-layout` then moved projections, convolution and gating into the
GDN blocks too. The 8,192-block warmed 120K step took 129.42 s at 76.02 / 79.17
GiB; 80K took 91.26 s at 63.85 / 66.34 GiB. The overall 120K live peak remained
76.02 GiB because a full-sequence attention backward became the limiting layer.
A subsequent **75 GiB allocation cap failed**; its completed earlier cases and
trace are retained. Larger 32,768 and 65,536 whole-GDN blocks did not reduce
that peak. This additional mode is available but is not automatically faster.

### Blocked attention projections: selected recipes

All rows below use 32,768-token attention projection blocks, three-layer
checkpoint groups, resident PLE, 40 MiB RMS tiles, PLE windows of 8,192 and
gated-normalization blocks of 262,144. Indexed attention still sees the full
sequence and its original selected keys.

| Context + target | GDN configuration | Step seconds | Peak GPU allocated / reserved GiB | Sampled host anonymous + shared GiB |
|---|---|---:|---:|---:|
| 120K + 10K | 8,192-token GDN recurrence segments, warm | 124.10 | 74.28 / 79.48 | 141.61 |
| 80K + 10K | Same recipe, repeat | 87.90 | 62.65 / 64.44 | 128.22 |
| 120K + 10K | 32,768-token whole-GDN blocks | 131.76 | 71.42 / 75.04 | 141.61 |
| 80K + 10K | 32,768-token whole-GDN blocks | 92.62 | 60.66 / 63.15 | 128.22 |
| 120K + 10K | GDN recurrence segments, 75 GiB allocator cap | 124.75 | 74.28 / 75.00 | 141.61 |

The balanced launcher recipe measures **124.10 s at 120K versus 87.90 s at 80K**:
80K takes **29.2% less time per sample** (1.412× faster).
Target throughput is 80.6 versus 113.8 tokens/s; total-token throughput
is 1048 versus 1024 tokens/s. These are single-request
forward/backward/Adam benchmark times, excluding encoding, loading and durable
checkpoint writes. Production accumulates four requests per optimizer update.

The lower-memory whole-GDN variant uses **71.42 GiB allocated / 75.04 GiB reserved**
at 120K, taking 131.76 s (6.2% longer than the balanced recipe).
It is available through `SFT_GDN_CHUNK_TOKENS=0 SFT_GDN_BLOCK_TOKENS=32768`.
The fastest measured high-memory alternative remains unsegmented GDN with
unblocked attention projections: 121.92 s / 86.16 s at 120K / 80K, using
80.01 / 67.48 GiB allocated. The selected default gives more GPU headroom for
a small speed cost; the CPU and disk alternatives remain separately qualified.

The **75 GiB allocator-capped 120K run passed**, reserving 75.00 GiB.
This is a Blackwell memory-pressure test, not an A100 compatibility or speed test.

The independent attention projection check used actual 24 Q / 2 KV heads of
dimension 256, nonzero rank-16 adapters, nontrivial RoPE and a cross-block loss.
Output / input-gradient relative errors were 0.00135098 / 0.000476334;
the maximum adapter-gradient relative error was 0.00245172 (each must be below 1%).
The full-model matched-forward VJP check and native convolution/GDN checks
also passed. Source hashes and all per-case losses/gradient norms are archived.


## What “exact chunking” means here

Nothing detaches a prefix, truncates a request, downsamples its images, or
replaces a full-context gradient with independent sequence losses. The native
GDN recurrence and indexed attention still process the complete sequence.
Chunks bound intermediate allocations inside operations whose dependencies
are preserved:

- **Frozen experts:** process routed rows in bounded microbatches and recompute
  local intermediates in backward. Input and routing-score gradients survive.
- **Indexed attention:** a gather-free Triton GQA kernel reads selected K/V
  vectors directly. Online softmax statistics, probability/value products and
  K/V gradient accumulation use FP32; FP32 tensor-core products use `tf32x3`.
  The key tile is 32. Probabilities are recomputed in backward, and the FP32
  pre-cast output is retained for the softmax derivative.
- **RMS normalization and four-stream residual mixing:** bound FP32 temporary
  storage by recomputing token-local operations in blocks.
- **PLE windows:** compute the n-gram embedding IDs on the complete input, then
  process windows with a left halo of `(conv_kernel_size - 1) × dilation`.
  Discard only the halo's *outputs*. Backward recomputes the window and adds
  gradients into overlapping input regions. Thus both n-gram hashing and
  convolution retain their dependencies across window boundaries.
- **GDN gated normalization:** compute normalization and sigmoid gating (also supporting SiLU) in bounded
  row blocks, preserving the native BF16 cast boundaries. Its analytical
  backward returns gradients for both the normalized input and gate.
- **Optional GDN segmentation:** split Q/K/V/g/beta along the token axis,
  run the native delta-rule kernel on each segment, and pass its FP32 final
  recurrent state directly into the next segment. Never detach that state.
  Non-reentrant checkpointing recomputes each segment in backward; derivatives
  flow from later outputs through every earlier state. Shared `split` nodes
  combine input gradients once instead of allocating a full zero-filled slice
  gradient for every segment. This chunks the recurrent operation, not whole
  independent model examples; full-context indexed attention remains intact.
- **Optional whole GDN blocks:** `--gdn-block-tokens` also moves input/output
  projections, causal convolution, gating and normalization inside each token
  block. The last three raw projected tokens are carried as differentiable
  convolution history for the next block (the native kernel is four tokens).
  The FP32 delta-rule state also remains attached. Checkpointing the whole
  block bounds these large projection tensors as well as the recurrence's
  workspace. CPU tests include chunks shorter than the convolution halo and
  losses on late outputs; the CUDA gate compares the entire native block at
  16,385 tokens, 48 value heads and dimension 128, including all LoRA gradients.
- **Optional attention projection blocks:** `--attention-projection-block`
  checkpoints token-local Q/K/V projection, Q/K normalization, RoPE, output
  gating and output projection. The indexer still runs on the full sequence;
  attention still receives the complete Q/K/V and original selected-key lists.
  This is not a local-attention window. Shared splits combine token gradients
  without repeated full-length slice-gradient allocations.
- **Final loss:** materialize vocabulary logits only for supervised final-reply
  positions, in bounded blocks, and sum their cross-entropy gradients.

These methods preserve the first-order objective and dependency graph. They
are numerically equivalent floating-point implementations, **not bitwise
identical** to every native BF16 execution order. The distinction matters in
a mixture-of-experts model, where small forward differences can change discrete
expert selection.

For a recurrent segment `S[j+1] = F(S[j], X[j])`, autograd retains the state
edge. Backward computes both the local loss derivative and
`dL/dS[j+1] × dS[j+1]/dS[j]`. Checkpointing changes when intermediates are
recomputed; detaching `S[j+1]` would delete this cross-segment derivative.
The CUDA primitive check uses a loss only on the last four outputs and final
state, compares against full native GDN, and checks Q/K/V/g/beta **and initial
state** gradients. All eight relative errors measured zero on the tested
BF16 513-token case (128-token segments), and again on **8,193 tokens, 48 heads,
dimension 128 and 4,096-token segments**. The latter initial-state gradient
norm was 4.82e-7, not zero. CPU reference errors were below 5e-7. This is a numerical test at those shapes, not universal bitwise proof.

`--rms-block-mib 40` chooses the row block from tensor width. A wide 10,240-value
row still uses 1,024 rows, while a 256-value Q/K normalization uses 40,960 rows.
That bounds the FP32 working tile while avoiding thousands of tiny Python/GPU
launches for narrow tensors. It does not change the normalization formula.

## Checkpointing and offload work together

Layer checkpointing removes most saved internal activations but retains layer
boundaries. One BF16 four-stream boundary is `tokens × 2560 × 4 × 2` bytes:
2.480 GiB at 130K tokens. Forty-eight boundaries alone require about **119.02
GiB**; at 90K they require **82.40 GiB**. Moving these boundaries off GPU does
not make their host storage disappear.

`--checkpoint-group 2` adds an outer checkpoint around each pair of already
checkpointed layers. Only the outer boundaries persist through the full
forward pass; inner boundaries are reconstructed during backward. This roughly
halved measured host working memory and improved speed here by reducing CPU
transfers, despite additional recomputation. Larger groups trade more temporary
GPU memory and compute for less persistent host storage.

`ActivationOffload` saves exact tensor bytes on CPU and retains resident frozen
weights by reference. Optional disk offload has a bounded asynchronous write
queue, a disk-byte budget, CPU fallback, and a bounded read-prefetch queue.
Prefetch reads upcoming tensors into CPU buffers; host-to-GPU restoration
happens on demand. It does **not** overlap every GPU copy or guarantee that
storage latency disappears. Scratch is cleaned after a request or Python
exception; abrupt termination can leave a request-specific scratch directory.

## Actual storage, not the NVMe label

The server has an RTX PRO 6000 Blackwell Server Edition: 97,887 MiB in
`nvidia-smi`, **94.9705 GiB** usable through PyTorch, compute capability 12.0.
The host cgroup limit is **175 GiB**, without swap.

The 95.37 GiB frozen PLE table resides on the attached model's **NFS** mount.
Unique row reads are sorted by shard and parallelized across eight readers.
A 1 GiB raw-row cache, plus Python index overhead, is retained between requests;
a separate approximately 0.62 GiB lookup cache survives recomputation of a
130K request. The entire table is never copied to GPU or scratch. Optional `--ple-resident`
copies its exact bytes into read-only host arrays once, trading **95.37 GiB of
non-reclaimable host RAM** for predictable row access. Four workers read bounded
16 MiB file ranges directly into the final arrays; this avoids slow NFS memory-map
page faults during the copy. Slice offsets and read-only ownership are CPU-tested. Startup rejects a table
copy that would put existing anonymous/shared memory plus the table above 80%
of the cgroup limit. That check does not prove the later activation peak fits;
the selected training recipe still needs full-data qualification.

`/tmp` is an overlay over a device-mapper snapshot. Its apparent multi-TiB free
space is misleading: the writable thin-pool backing file is **96 GiB**.
`/kaggle/working` is a separate **20 GiB** loop filesystem. Both ultimately
write to the same 256 GiB virtual NVMe device, identified as `nvme_card-pd`
on GCE. The exact persistent-disk tier is unknown; this is not evidence of a
local physical NVMe SSD. Never copy the 95 GiB PLE table into that thin pool.

Underlying-device counters measured about **527 MiB/s** on the writable device,
**1,595 MiB/s** on the origin and **232 MiB/s** on NFS. `/tmp` sequential writes
measured about **463 MiB/s**. Guest loop reads of 17–20 GiB/s came from backing
file cache. Even guest `O_DIRECT` does not bypass that cache when the loop
backing device itself uses buffered I/O. Full storage evidence, including
random IOPS, is in [CAPACITY.md](CAPACITY.md).

The earlier disk prototype moved 29.755 GiB of activations, spent 66.45 s in
writer work and 4.41 s in reader work, with 12 prefetch hits and no synchronous
reads. A monitored backward interval recorded **24.7955 GiB of loop reads and
zero physical backing-device reads**. This demonstrates cache-assisted prefetch,
not cold-NVMe throughput. At 463 MiB/s, writing 32 GiB alone needs roughly
71 seconds of storage service. A lower-RAM host may also incur physical reads.

Host measurements distinguish `RssAnon`, `RssFile` and `RssShmem`. Pinned/shared
buffers can remain allocated after a disk test; Linux cgroup `file` includes
shared memory and is not all reclaimable cache. The report uses sampled
`VmRSS - RssFile`, which retains shared memory. The production guard includes
anonymous, shared and unreclaimable kernel memory. Per-case sample maxima are
not instantaneous allocator peaks, and a same-process prior disk case can
leave approximately 12 GiB of pinned buffers cached.

## Workload construction and processor verification

The 130K composite uses five whole real requests totaling 119,997 tokens,
three real generation-header tokens, and eight whole generated-thinking/code
replies totaling 10,000 target tokens. It has **59 real 640×640 images**,
expanding to 23,600 image tokens.

The 90K composite uses five whole requests totaling 79,985 tokens, 15 real text
tokens, and **the same complete target replies**. It has **34 images**,
expanding to 13,600 image tokens. The filler uses generated text because a
longer pre-target suffix would include an orphan image placeholder.

No original request, image block or whole target reply is truncated. These
concatenations are disposable stress workloads, not coherent teacher
trajectories. The comparison includes a changed image distribution; it is not
a controlled experiment varying only text length. Source IDs and checksums
are retained with the benchmarks.

The attached model's processor files differ from the original upstream export.
Re-preparing all 30 validation requests using the attached processor produced
**identical token IDs, image tensors and annotations for all 30**. The new
manifest is `64789772729c72d8cdd988901d7104acd8ca2b4177aaace7cabc82eeac59274f`.
The production compatibility check remains strict; this result does not
approve arbitrary mismatched processors.

All 30 targets match the archived generated-thinking NLL run, but **none of
the full input-token hashes match**. Consequently, archived NLL is not a
matched baseline for this export. Fresh before/after validation uses the same
prepared requests, processor, expert map, backend and numerical recipe.

## Correctness investigations and retained failures

The initial attention prototype used BF16 probability/value products. It
completed full-length steps, but its timings are **not** the final production
kernel's timings. Later investigation also found an empty-leading-key-tile
softmax bug. The fix preserves a running maximum of negative infinity until
the first valid key; masked output rows return zero. Regression tests cover
very negative logits, initially empty tiles, fully masked rows and duplicate
keys at the model's 24 query / 2 KV heads and head dimension 256.

A naive whole-model gradient comparison initially differed by about 36–40%.
It changed the forward implementation as well as backward, allowing different
expert choices. Fixing the masked-tile bug and increasing attention precision
did not by themselves remove that difference. The appropriate VJP check uses
**identical optimized forward outputs and expert choices**, with an independent
FP32 SDPA autograd attention backward. The corrected pre-window model passed
with **0.88% aggregate adapter-gradient relative error**. Separate forward
checks remain; this is not a claim that routing is bitwise unchanged.

On that 512-token diagnostic, optimized loss was 12.4763403 versus 12.4696703
for the native-pointwise/FP32-SDPA forward. About 44.65% of layer/token expert
*sets* had at least one difference; that number is not the fraction of all
individual experts replaced. Later reports include both quantities and
per-attention forward errors. In the final run, **5.328% of individual expert
choices were replaced**. CPU tests independently compare full hybrid
multimodal gradients, PLE halo gradients and both gated-normalization inputs.
Native GDN/convolution forward and backward are also checked against Torch
references on GPU.

A key tile of 64 for the higher-precision attention backward exceeded
Blackwell's per-block shared-memory limit (114,688 requested vs 101,376 bytes).
The selected tile is 32. The earlier 77 GiB allocator-capped run failed in
GDN normalization. These failures and the earlier layer/PLE OOM traces remain
in `capacity-results/`; successful earlier cases are not erased when a later
suite case fails.

A first whole-GDN block implementation preserved convolution values but used
an unfavorable channel/time memory layout. Its actual-head CUDA check caught
4.85% input-gradient error and up to 5.16% QKV-adapter-gradient error, so it was
rejected before full-length testing. Restoring the native channel-contiguous
convolution layout reduced these to 0.184% and at most 0.282%, respectively.
The gate now requires every compared output/input/adapter relative error to
remain below 1%. CPU-only agreement would not have caught this native-kernel
layout issue. Both the rejected source and corrected measurements are archived.

## Runnable training pipeline and remaining data dependency

[README.md](README.md) describes preparation, qualification, training and
validation. `launch_fit.sh` uses the attached source model and the verified
train-only expert map. `fit` loads the model once, tests actual training-data
extremes with at least **4 GiB above peak reserved GPU memory**, discards qualification updates, evaluates fold 0, trains, and
evaluates again. Each request saves an atomic checkpoint including Adam state,
RNG, cursor and partial accumulated gradients. Only the latest two durable
checkpoint payloads are retained. Resume binds data, model shards, map,
processor assets, code, runtime, hardware and recipe.

Fold 0 is held out by **game**, not by randomly splitting requests. Its games
are sk48, sp80, tn36, cd82 and ar25. The expert map's calibration provenance
contains only the other 20 games and records `validation_exposed: false`.
The 30-request panel contains **zero training requests**. Preparation preserves
full requests and fails on missing generated thinking or an over-limit sample.

The incoming training dataset was not visible in the repository's PRs or
branches during this work. Actual SFT must wait for that artifact; the held-out
panel and capacity composites are not substitutes. The code and diagnostics
are useful completed work, but capacity qualification is not evidence of
training quality or completion of an epoch.

No A100 was available. A Blackwell allocation cap can test memory pressure,
but cannot establish Ampere kernel compatibility, allocator behavior or speed.
The A100 profile requires its own native-kernel and real-data qualification.

## Fresh fold-0 baseline

All **30 requests / 24,358 target tokens** completed in the production evaluator
without training updates. Token-weighted NLL was **1.016282**, thinking NLL
**1.439088**, code NLL **0.339464**, and macro game mean-request NLL **1.033391**.
Per-request token losses, source hashes, the numerical recipe and native/model
gradient checks are in `capacity-results/20261008-overnight/validation/`.

This baseline uses windowed PLE and bounded gated normalization, with the
original RMS row-block sizes and unsegmented GDN. The later speed experiments
change these settings. `fit` therefore computes its own baseline under its
selected recipe before any training; do not silently pair this baseline with
a different numerical recipe. No trained-versus-base improvement is claimed.
