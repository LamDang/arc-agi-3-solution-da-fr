# Astra review: GPU LoRA masters and host-RAM correction

Reviewed2026-10-10 against the current uncommitted components/expert_offload.py,
model.py, expert_chunks.py, runtime/loop.py, evidence.py, resources.py and expert
fixture changes. Only this report written. No execution, GPU, network, private
connection access, raw-gradient archive, push or shutdown.

## Recommendation

Proceed to the planned16K qualification against the original saved reference and
same-model unchunked control under the user2% gate. No blocking correctness,
storage-lifetime or stream-ordering defect was found. This is source review,
not numerical or capacity acceptance. GPU placement follows the user's explicit
correction; FP32 master/gradient precision and original numerical reference stay
unchanged. Real training must still use native autotuning, without the test FLA
profile.

## Evidence behind the correction

Reviewed reports/host-memory-diagnostic.json for load-only attempt
20261010083338834-292581ae, execution4b9de56. It completed without forward,
backward, PLE preparation or optimizer updates. Unique live storage included
31661752320bytes frozen pinned expert data (29.488GiB),7549747200bytes CPU expert
adapters (7.03125GiB), and133914624bytes other CUDA adapters (~0.12472GiB).
Total adapter payload is approximately7.156GiB.

The old pinned allocator held51543965709bytes (~48.004GiB) despite29.488GiB
expert payload. Diagnostic malloc_trim reduced PSS by approximately29.11GiB,
with unchanged live model storage and pinned allocator bytes. Thus unused
pageable allocator pages, rather than a necessary second frozen model copy,
were demonstrated. The trim itself was diagnostic only and is not installed in
training.

Cgroup file includes shmem. The observed approximately127.049GiB file and
48.024GiB shmem must not be added as disjoint categories. File page cache also
need not equal unreclaimable process memory. The diagnostic resolves the earlier
PSS/host-used ambiguity more specifically; it does not justify summing all
reported memory counters.

## Ownership and transfer review

- The shared pinned allocation is owned by LayerPrefetch.backing; per-layer slabs
  and module frozen buffers are views into it. These aliases must count once.
  Layer GPU prefetch still copies only the corresponding slab, preserving the
  existing two-layer window. One allocation should avoid48 independent rounding
  gaps; approximately32GiB allocator backing is expected, not measured yet.
- Buffer layout alignment inside each layer is unchanged. Current tensor sizes
  are naturally aligned across layers. If supporting other layouts later, align
  each layer's base offset too, rather than assuming every final size is a
  multiple of the largest dtype alignment.
- Direct view.copy_(source_buffer) uses the default blocking transfer. Therefore
  GPU-to-pinned data is complete before replacing/releasing the old buffer
  reference. The prior all-layer module.to(cpu) and pageable intermediate copies
  are removed. Temporary buffers dictionaries keep their current GPU sources
  alive while copying one layer.
- Frozen H2D copies occur on the staging stream; acquire waits on the ready event
  and records consumer-stream use. Release retains existing stream accounting.
  The backing remains alive for the manager's lifetime; close synchronizes the
  copy stream. No reuse/mutation of pinned backing during a live copy is introduced.
- Canonical FP32 LoRA Parameter objects remain on CUDA. Native functional_call
  receives those parameters directly; chunked execution receives differentiable
  flat CUDA concatenations and returns FP32 gradients to that same device.
  Expert arithmetic is unchanged. Parameters are no longer prefetched on the
  copy stream, removing the need for cross-stream parameter-copy ownership.
  A complete parameter/device assertion exists in model.py; fixtures also assert
  CUDA FP32 gradients.

## Initialization and telemetry

runtime/loop.py retains detached CUDA views for initial comparison rather than
creating a full CPU adapter snapshot. compare_initial transfers one tensor at a
time while reference_tensors streams reference shards. Temporary tensor copies
remain bounded by one current tensor plus a reference shard. The baseline-free
archive path still explicitly clones to CPU; that path was not silently changed.
Candidate raw-gradient retention policy is unchanged.

Resources now samples memory.current, cgroup anon/file/shmem and host pinned
allocated bytes alongside prior metrics. This helps verify the actual gain.
Recorded phase peaks occur at potentially different times: do not add the
individual peak categories, and do not add shmem on top of file. The cgroup path
assumes the observed root-cgroup container layout; another environment would need
its resolved process cgroup. Current code records allocated_bytes.current from
the host allocator; describe that exact counter rather than implying a separate
reserved-byte measurement.

## Expected benefit and shifted capacity limits

Potential host savings relative to the old loaded process comprise roughly:

| Mechanism | Approximate benefit | Status |
| --- | --- | --- |
| Single backing instead of48 rounded slabs | 16GiB | Expected; requires allocator measurement |
| Avoid old pageable frozen-weight copies | 29.11GiB retained arena pages | Old releasable pages measured; avoidance not yet measured |
| Move canonical expert masters to GPU | 7.03125GiB | Source change; new live inventory must verify |

These estimates are not three new benchmark measurements and cannot establish a
120K pass. Fresh-process allocator behavior, page cache, saved tensors and
transient copies can change the realized total. Do not simply subtract them from
a cgroup lifetime peak.

GPU memory rises by roughly7.03GiB resident expert masters and up to another
7.03GiB expert gradients versus the previous CPU-master benchmark, before any
optimizer. AdamW moments for all adapters (~14.31GiB) now belong on GPU with the
masters, following ordinary optimizer placement. Previous CPU-based training
headroom estimates are superseded by this placement correction. Forward/backward
capacity still does not prove full training/update capacity.

The independent checkpoint-disk limit remains: adapters plus two FP32 AdamW
moments require approximately21.47GiB tensor payload, exceeding the confirmed
17.323GiB free working volume and its19.518GiB total. GPU placement does not reduce
that serialization payload. No alternative storage/export is authorized here.

## Qualification evidence to retain

Use the existing component GPU fixtures, all74472 initial equality check, original
reference comparison and paired2% full16K gate. Record actual all-parameter and
gradient device inventories, host unique pinned payload versus allocator bytes,
cgroup phase counters, and GPU peaks. After numerical qualification, capacity
must be remeasured before claiming96K/120K or optimizer success. Preserve the
original96K failure unchanged. No speculative graph/precision changes are needed
as part of this RAM correction.
