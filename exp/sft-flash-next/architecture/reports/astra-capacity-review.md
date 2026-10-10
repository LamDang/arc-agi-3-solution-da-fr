# Astra review: 96K host-memory capacity failure

Reviewed2026-10-10. Scope: collected local96K artifacts, current source,
CHUNKING.md, and latest train/HANDOFF_NEXT_SESSION.md. Only this report written.
No GPU, Jupyter, network, implementation changes, raw-gradient writes, or pushes.

## Disposition

Do not run120K unchanged. The96K attempt failed during its first forward before
loss/backward, and both source memory scaling and measured32K/64K growth predict
higher host pressure at120K. Preserve the failed attempt and report120K as blocked
by demonstrated capacity pressure until a measured memory remedy is qualified.
This is not permission to truncate sequences, alter precision, change the
reference, or declare the requested benchmark complete. Keep the Jupyter server
available while required work/collection remains.

## Verified failure evidence and OOM attribution

Attempt20261010075533242-f30e823d, executionf5e78c4: monitor.json records
returncode-9, timed_out=false, elapsed1284.5364s. resource-progress.json records
incomplete forward-0 at556.7493s, treePSS182541534208bytes (170.005GiB), parent
RSS181993091072bytes, host-used132275933184bytes. There is no completed96K loss,
backward, CUDA phase peak or second pass. Loading and initialization completed;
all74472 initial tensors matched. The separate PLE preparation had already
completed, so its691s latency does not explain SIGKILL as a supervisor timeout.

The main agent reports post-failure cgroup observations: memory.max187904819200
(175GiB), memory.peak187357286400 (174.490GiB), swap.max0; memory.events/local
has oom_kill1 but max0 and oom0. These observations are consistent with host
memory exhaustion and an OOM kill near the available budget. There is no
before-attempt counter snapshot, and max/oom are zero: do not claim proven
memcg-limit enforcement or uniquely attribute the counter increment to this
process. Host/global OOM is also possible. Preserve the exact counters and their
scope; SIGKILL alone would not prove OOM. The remote post-failure JSON was still
being recollected when this review was written.

## Retained tensors and structural growth

model.py enables non-reentrant decoder checkpointing. runtime/benchmark.py wraps
forward/backward in save_on_cpu(pin_memory=False). Each decoder checkpoint must
retain its full input residual for replay: BF16 shape[1,T,10240], representing
four2560-wide streams. Across48 decoder layers this is48*T*10240*2bytes. The final
hyper_connection_mixer is independently checkpointed and adds a further full
residual input outside the decoder stack. These are structurally required saves
in the current implementation, not a measured exhaustive saved-tensor inventory.
Nested inner checkpoints must not be naively counted as another48 complete
copies: outer recomputation replaces their saved intermediates. Other saved
inputs, vision graph data, head state, Python objects and transient copies add
memory beyond these residuals.

| Tokens | One residual | 48 decoder inputs | Including final mixer input | Expert CPU slot scratch per phase |
| --- | --- | --- | --- | --- |
| 32000 | 0.610GiB | 29.297GiB | 29.907GiB | 1.526GiB |
| 64000 | 1.221GiB | 58.594GiB | 59.814GiB | 3.052GiB |
| 96000 | 1.831GiB | 87.891GiB | 89.722GiB | 4.578GiB |
| 120000 | 2.289GiB | 109.863GiB | 112.152GiB | 5.722GiB |

Expert slot scratch is T*10*2560*2bytes; forward slot outputs and backward slot
cotangents are not retained simultaneously for every layer. It is temporary
per-layer work, not an additional48-fold allocation. Hyperconnection full-M
norm/projection tensors remain GPU temporaries under recomputation; elementwise
chunking does not eliminate full residual checkpoint storage on CPU.

Measured peakPSS121.1751GiB at32K and152.03674GiB at64K grows30.86164GiB per32K,
close to the29.90723GiB added49 residual inputs plus other growth. A two-point
same-profile extrapolation gives approximately182.898GiB at96K and206.045GiB
at120K. These are estimates, not measured peaks or a proof of exact cgroup usage.
The96K last170.005GiB reading is an incomplete-forward lower observation, not a
successful peak. At120K the residual saves alone grow22.430GiB versus96K.

Native16K referencePSS138.49GiB is a different capture path with reference export
and comparison work; it is not a valid baseline to subtract from these lean
benchmark runs. GPU memory is not the observed binding resource:64K measured
25.150/32.957GiB forward/backward allocated, while96K died without a completed
CUDA phase measurement.

## Slabs, copies, mappings and accounting

LayerPrefetch owns approximately29.49GiB of pinned CPU frozen buffer slabs and
approximately7.16GiB CPU FP32 adapter masters. The module buffers are replaced
with views into those slabs, so these two references alias the same storage.
Do not count module buffers and slabs twice. Construction temporarily holds the
old CPU buffer dictionary while copying one layer; it is replaced per layer and
released when __init__ returns. No source-level persistent second all-layer
frozen-buffer copy was found in this component. Prefetch keeps at most two CUDA
layer states and creates temporary differentiable parameter concatenations.

Benchmark initialization uses CPU views where possible and deletes its initial
state dictionary; reference_tensors streams bounded shards. The benchmark avoids
full raw-gradient dictionary clones and the unchunked control. These are not
plausible hidden all-gradient duplicates in the failed forward.

Nevertheless,96K loading already samples treePSS about89.5GiB versus host-used
about43.7GiB. PSS and psutil.virtual_memory().used measure different things: PSS
includes proportional resident file mappings; host-used commonly excludes
reclaimable file cache, and GPU/driver mappings can complicate interpretation.
Neither metric equals cgroup memory.current. This gap alone does not establish
an extra46GiB of anonymous weight duplication. A source mmap, allocator residue,
page cache and driver/pinned accounting require actual mapping evidence.

Minimal read-only accounting needed in a bounded diagnostic: cgroup current,
peak, max, events and memory.stat anon/file/shmem/kernel/unevictable where
available; per-process smaps_rollup and path-grouped smaps (Anonymous,
Private_Dirty/Clean, Pss, Locked); pinned allocator allocated/reserved statistics
if exposed. Distinguish live tensor storage from cached allocator blocks and
file pages before changing ownership. Do not use global drop_caches or unload
required tensors speculatively. Reading a post-exit process cannot recover its
lost mapping inventory; gather metadata during a smaller already-supported
length if this accounting is necessary for the next remedy.

## PLE and disk implications

The96K prepared PLE payload is about0.458GiB, orders of magnitude below checkpoint
storage. lookup closes each NumPy memmap; this limits persistent mapped pages in
the worker but does not guarantee immediate reclamation of NFS page cache charged
to the cgroup. A compact prepared-row cache would address sparse-NFS I/O and may
reduce page-cache pressure; it is not a demonstrated cure for the87.9GiB decoder
checkpoint saves. It is not implemented and must not be credited in benchmark
results. One-sample capacity runs also do not measure multi-sample lookahead RAM.

Only17.323GiB free writable working storage is confirmed. Root/tmp and other
large apparent filesystems fail writes with EROFS; /dev/shm is RAM-backed and
charges the same host budget. Therefore neither the full95.37GiB PLE table nor
all96K/120K activations can simply be moved to confirmed local disk. Do not treat
976GiB df availability or read-only NVMe capacity as writable backing.

## Real training headroom

These are forward/backward capacity tests without optimizer state or updates.
Two FP32 AdamW moments add approximately14.31GiB after the first update, mostly
on CPU for expert adapters. The existing FP32 gradients are already present in
benchmark backward, so do not automatically add another7.16GiB to every reported
backward peak. However training accumulation can carry about7.16GiB gradients
into the NEXT forward, whereas benchmark repeats zero_grad(None) first. Thus
steady training forward may add roughly21.47GiB (moments plus retained grads)
over benchmark forward. Optimizer temporaries and multi-sample PLE lookahead add
further overhead.

At64K, a simplistic152.04+14.31 estimate is166.35GiB, with only about8.65GiB to a
175GiB nominal budget; carrying gradients can reduce that margin to roughly
1.49GiB before additional overhead. This is not proof64K training fits. At32K
there is more observed headroom, but optimizer/update capacity remains untested.
96K training cannot be justified where optimizer-free forward already fails.

There is an independent disk constraint: train_samples serializes adapter state
plus AdamW state in checkpoint.pt. Approximately7.16+14.31=21.47GiB tensor payload
already exceeds confirmed17.323GiB free working storage (and its19.518GiB total),
without serialization metadata or prior artifacts. Training needs an explicit
checkpoint-storage plan in addition to RAM headroom; no new storage destination
or export is authorized by this review.

## Minimal safe next steps

1. Finish collecting96K failure and cgroup/storage metadata, verify hashes/local
   DVC, and record incomplete first-forward failure. Preserve source and flags.
2. Do not dispatch120K unchanged. Explain the capacity blocker and structural
   growth rather than spending another run on a predictable higher-pressure case.
3. Establish cgroup versus mapping/pinned/anonymous accounting at a supported
   length if needed to choose a remedy. Record only storage identities, shapes,
   dtypes and byte counts for saved tensors; never retain activation values.
4. Consider a separately reviewed outer checkpoint grouping/recompute strategy
   that retains fewer full residual boundaries, or genuine available backing
   storage, before changing numerics. Any implementation change requires the
   existing16K2% qualification and renewed capacity measurement. Do not silently
   revise precision, context, optimizer, reference, or accepted flags.
5. Keep production training capacity distinct from F/B-only results, including
   AdamW state, accumulated gradients, update temporaries and checkpoint storage.

No implementation remedy or120K success is claimed. No push or shutdown was
performed or recommended while required work remains.
