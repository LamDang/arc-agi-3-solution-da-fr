# Full-length capacity probe

This disposable test checks a complete forward, backward, and AdamW update on
the existing Intel W4A16 Flash-Next checkpoint, physically reduced from 512 to
256 experts in GPU memory. It never saves updated adapters. It is a custom
PyTorch/Transformers backend test, not an Axolotl or Unsloth benchmark.

## Real input

The input contains exactly **120,000 context tokens and 10,000 target tokens**,
including **59 real 640 × 640 images** (23,600 image tokens). It combines five
whole requests from the verified 30-request generated-thinking panel, three
real generation-header tokens, and eight whole final replies. No original
request or image block is truncated. Source IDs and hashes are recorded in
`capacity-results/20261008-rtx96/sample-provenance.json`.

Real data exercises content-dependent expert routing, sparse attention, PLE
lookups, and image processing. Concatenation changes their joint distribution:
the result is not a coherent teacher trajectory, a representative random draw,
or a quality-evaluation sample. Attention remains causal across the entire
composite, without resets between source requests. Only the final 10,000 tokens
contribute to loss, while gradients propagate through the full context.

| Measure | Original 30-request panel | Capacity composite |
|---|---:|---:|
| Total tokens per request | 5,941–84,870; median 40,340 | 130,000 |
| Final-reply tokens | 324–2,340; median 665 | 10,000 |
| Images per request | 1–54; median 21.5 | 59 |
| Image-token share, token weighted | 20.54% | 18.15% |
| Supervised-token share, token weighted | 1.88% | 7.69% |

The requested 10K target is substantially longer than any original reply.
This is deliberately an extreme capacity test, not an estimate of typical
training throughput. See `panel-distribution.json` for the panel aggregates.

The panel is held out. All capacity-test updates are discarded. The existing
all-game `models/flash-next-reap-256/keep.json` is acceptable only for this
disposable memory test; actual held-out experiments still require the
training-only expert map described in [README.md](README.md).

## Reproduce

Prepare the verified panel with the existing dataset preparation, then run:

```bash
python exp/sft-flash-next/train/build_capacity_sample.py \
  --panel /path/to/prepared-validation --out /path/to/real-capacity.pt.gz
gzip -dk /path/to/real-capacity.pt.gz

python exp/sft-flash-next/train/capacity_probe.py \
  --model /path/to/intel-qwen3.8-flash-next-w4a16-autoround \
  --keep models/flash-next-reap-256/keep.json \
  --sample /path/to/real-capacity.pt --out /path/to/capacity-results
```

The probe starts with a 512-token training smoke test and restores the initial
adapters/clears Adam state before the full test. It uses rank 16 LoRA on language
attention, GDN, and shared-expert projections (33,478,656 trainable parameters).
Routed experts, routers, indexers, vision, PLE, and the output head remain frozen.

Per-layer non-reentrant checkpointing and CPU activation offload are combined
with bounded expert, sparse-attention, and loss recomputation. Local backward
temporaries remain on GPU to avoid redundant CPU transfers; this does not
detach context or truncate backpropagation. CPU gradient-parity tests cover the
custom operations. Native convolution and GDN outputs and gradients are checked
against Torch references on the actual GPU before loading the large model.

## Runtime and provenance

The Kaggle GPU is an RTX PRO 6000 Blackwell Server Edition with 97,887 MiB of
reported VRAM; the host cgroup limit is 175 GiB with no swap. Runtime: Python
3.13.15, torch 2.11.0+cu128, Transformers 5.18.0, FLA 0.5.2. The convolution wheel
comes from [PR #23](https://github.com/LamDang/arc-agi-3-solution-da-fr/pull/23):
`causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl`, SHA-256
`f928f1aa1de1306f26f58a3f2c7a6d9ac2d696090fd1bd26430951d82be9928b`.

`events.jsonl` records phase boundaries and CUDA allocator peaks;
`memory.jsonl` samples process RSS, anonymous RAM, and cgroup file cache every
five seconds. File-backed PLE/model pages are reclaimable and must not be added
to anonymous activation RAM as an irreducible requirement. Sampled host peaks
can miss short spikes; CUDA allocator peaks are recorded by PyTorch.

The four residual streams make one BF16 checkpoint boundary at 130K tokens
approximately `130000 × 4 × 2560 × 2 / 2^30 = 2.480 GiB`. Across 48 layers this
is about **119.0 GiB**, before other host allocations. This fixed-shape term
depends mainly on sequence length; real data matters for the content-dependent
parts of memory and runtime.

The probe stops its own subprocess above 160 GiB cgroup anonymous RAM or after
two hours. It does not stop the notebook or GPU billing. Credentials and raw
panel tensors are not included in the repository results.

### Storage inspection

The actual attached model directory, including PLE, is an **NFS mount**.
`/tmp` is on the root **overlay** filesystem, reporting 8 TiB total and about
1.1 TiB available. `/kaggle/working` is a separate 20 GiB ext4 loop filesystem.

DMI identifies Google Compute Engine. Visible devices are a 256 GiB boot disk
and an 8 TiB disk, both with model **`nvme_card-pd`** and Google vendor ID
`0x1ae0`. No local SSD device is visible. Further inspection exposed `dm-2`
(`snap`) over the 8 TiB `nvme0n2` origin and `dm-1` (`sdg_pool`). That thin pool
has a **96 GiB data loop device** (`loop1`, backing file
`/var/lib/sdg_thin_pool_data`) and a 1 GiB metadata loop device. Thus the visible
8 TiB root size does **not** establish 1.1 TiB of writable scratch capacity.
The exact provisioned persistent-disk tier and any pool-extension policy are
not exposed. Do not infer local SSD from the NVMe interface:
[Google persistent storage can use NVMe](https://docs.cloud.google.com/compute/docs/disks/persistent-disks).

A disposable 512 MiB scratch test measured about 506 MiB/s writes with fsync
and requested `O_DIRECT`. Its exceptionally fast reads (17,555 MiB/s, 6 µs
mean 4 KiB latency) are not proof of cold physical-device performance in this
virtualized/overlay environment. A separate 256-read `O_DIRECT` sample on an
existing NFS PLE shard measured 3.51 ms mean / 9.69 ms p95 4 KiB latency.
These small, different-cache-state tests do not establish a sustained PLE
speedup. They are recorded in the drive-probe logs; both scratch files were
removed. No full PLE copy or storage migration was performed.

A full approximately 95.4 GiB PLE copy under `/tmp` would leave almost no space
in the visible 96 GiB thin write pool even before existing usage. It should
not be attempted based on `df` alone. Partial shard staging or a bounded disk
cache are possible alternatives, subject to benchmark results. Such changes
do not solve GPU backward allocation pressure, and `/tmp` must be treated as
session scratch rather than durable model storage. A subsequent multi-mount
I/O benchmark records backing-device counters to detect lower-layer caching.

### Device-verified I/O benchmark

The completed benchmark is in
[`capacity-results/20261008-rtx96/io-benchmark`](capacity-results/20261008-rtx96/io-benchmark).
Each read test lasts eight seconds, using 4 MiB sequential reads or 4 KiB random
reads. Random concurrency 16 means 16 synchronous worker threads. Write tests
create and fsync a disposable 2 GiB file, then remove it. Existing library files
are read only. These are short application-path measurements, not guaranteed
sustained maxima; provider-side caches and Python overhead can affect results.

| Read path | Sequential MiB/s | Random 4 KiB IOPS, 1 worker | Mean latency, 1 worker | Random 4 KiB IOPS, 16 workers |
|---|---:|---:|---:|---:|
| 256 GiB Google disk, `nvme0n1` | 527 | 2,868 | 0.348 ms | 5,526 |
| 8 TiB Google disk, `nvme0n2` | 1,595 | 2,577 | 0.387 ms | 31,194 |
| Existing NFS PLE mount | 232 | 489 | 2.039 ms | 7,194 |

For the first two rows, read-only library paths produced corresponding physical
device read-byte increments in both global `/proc/diskstats` and cgroup counters.
They therefore exercised the exposed persistent block devices, rather than just
the loop file cache. NFS random reads span all 22 PLE shards (95.37 GiB).

Scratch write tests measured **463 MiB/s under `/tmp`** and **502 MiB/s under
`/kaggle/working`**. Scratch read tests returned 17,336 and 20,773 MiB/s,
respectively, but device counters identify these as reads served below the loop
device from the backing-file cache. Both loop devices have `dio=0`; requesting
`O_DIRECT` on the guest file does not bypass that lower cache. Do not use these
scratch read rates as physical SSD estimates.

The subsequent 1 GiB writeback check increased global `nvme0n1` writes by about
1 GiB after fsync, while `nvme0n2` writes remained zero. RAM `shmem` did not grow
by the file size. This identifies the scratch write path as the 256 GiB
persistent disk through the thin pool, with an intervening file cache—not a
RAM disk or the faster 8 TiB origin. Per-cgroup counters alone missed that
lower-level writeback attribution, which is why the global check was added.

**PLE implication:** the server exposes no writable local-NVMe SSD path, and
the 96 GiB thin pool has insufficient safe headroom for a 95.4 GiB PLE copy plus
existing files. A bounded cache of the PLE rows needed by the dataset is a more
appropriate next candidate than assuming the root filesystem's apparent free
space is all usable. This optimization remains unimplemented and would need
its own capacity and throughput check.

## Measurement status

The default CUDA allocator run **failed during backward at decoder layer 47**.
The full forward pass completed in **477.86 seconds**, with a finite mean loss.
CUDA peaks before failure were **86.67 GiB allocated / 93.29 GiB reserved**.
Process anonymous RAM reached **139.83 GiB**; cgroup anonymous RAM reached
**140.11 GiB**, and measured process RSS peaked near **147.71 GiB**.

The failing allocation was **4.96 GiB**, with only **2.51 GiB** device memory
free and **7.61 GiB** of PyTorch reserved memory unallocated. This suggests
fragmentation might contribute, so the identical sample was retried with
`PYTORCH_ALLOC_CONF=expandable_segments:True`.

The failed run is preserved in `capacity-results/20261008-rtx96-default-allocator`.
The retry is in `capacity-results/20261008-rtx96`. It **also failed during
backward at layer 47**: **89.12 GiB peak allocated / 92.34 GiB peak reserved**,
then a failed **4.96 GiB** allocation. At failure there were 4.49 GiB free and
only 705 MiB reserved but unallocated. This is not explained solely by the
default allocator's fragmentation. Usable CUDA capacity was 94.97 GiB.

The retry's forward pass took 768.37 seconds, including substantial cold NFS
PLE paging. Its loss matched the earlier run (1.12202525). Host anonymous RAM
peaked at 139.84 GiB for the process / 140.12 GiB for the cgroup. Process peak RSS
was 168.45 GiB, including reclaimable file-backed pages. The lower RSS in the
earlier run reflects different cache residency, not a different activation size.

Both runs passed the 512-token forward, backward, finite-gradient, and AdamW
smoke test. Neither reached the 10K-target optimizer update. The full-length
recipe is **not qualified on this 96 GB GPU**. No A100 test was performed, and
this does not establish a lower bound for every possible implementation.
More granular recomputation or fused/chunked pointwise operations are possible
next optimization targets; moving PLE to another drive alone cannot resolve
the observed GPU allocation failure. The probe subprocesses exited and released
their GPU memory; no actual fine-tuning run or adapter export was started.
