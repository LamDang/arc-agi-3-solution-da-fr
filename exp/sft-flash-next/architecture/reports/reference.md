# New all-expert reference — measured16K capture

Attempt `20261009221135477-25bf58fe`, execution commit `3461d6abe9aa2834b0c8006798e6c8895b23750b`.
Kaggle Jupyter API only; process exited0, no timeout. Full anchor16,249 tokens,
651 supervised next-token targets and7 images. One forward/backward, zero
clipping, optimizer updates or overfit. Native head, dense masks, RMSNorm and
SwiGLU; BF16 declared activations; native FP32 statistics/work retained.

**Loss: 0.6244627833366394.**

| Phase | Seconds | GPU allocated GiB | GPU reserved GiB | Parent RSS GiB | Tree PSS GiB | Host-used GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loading | 322.437 | 46.444 | 75.697 | 87.785 | 88.299 | 42.810 |
| forward | 155.384 | 48.260 | 51.320 | 137.939 | 138.452 | 92.596 |
| backward | 222.437 | 55.636 | 58.160 | 137.976 | 138.490 | 92.635 |
| gradient_export | 13.235 | 10.340 | 58.160 | 96.348 | 96.861 | 51.097 |

GPU counters are synchronized CUDA allocator peaks; reserved includes allocator
cache. Loading has a larger reservation than steady forward/backward. CPU RAM
counters are sampled (poll target0.5seconds; smaps sampling itself adds latency).
Tree PSS includes the DataLoader child. Host-used is the separate psutil system
counter, with different cache/mapping accounting; it is not interchangeable with
process RSS/PSS or a cgroup limit. All byte counters are retained in JSON.
These are one forward/backward peaks, **excluding AdamW optimizer state**.
No130K run or capacity claim.

## Reference state and validation

- 74,472 FP32 parameter/gradient tensors:73,728 routed-expert A/B tensors on CPU and744 existing adapter tensors on CUDA. All raw gradients independently checked finite.
- Trainable parameters: 1,920,915,456; FP32 masters 7.155968GiB. FP32 gradients have the same total payload size. CUDA autocast uses BF16 for linear matmuls.
- Nonzero gradient tensors: 74,394; initialization tensors nonzero: 74,472. Zero/unused entries are retained, not used as equivalence proof.
- All256 routed experts ×48 layers have gate/up/down LoRA, rank16/alpha32. Frozen expert buffers occupy29.487305GiB on CPU.
- Full frozen PLE table remains on disk; bounded DataLoader preparation supplies current-sample lookups through recomputation.
- At most two expert layers staged. Full trace verifies forward0→47 and recomputation47→0, with BF16 expert inputs throughout.
- Real quantized expert fixture passes bitwise output, loss, input gradient and12 CPU FP32 adapter gradients with checkpoint recomputation.
- Source, inputs, all146 raw Torch shards and their metadata are independently verified by `verify_reference.py` without importing Torch. Each state contains74,472 FP32 tensors; raw evidence and SHA256 manifest are cached in local DVC.

## Initialization scope

This loss uses the **test-only nonzero A/B fixture** described in
`../REFERENCE.md`; every exact initial value is archived. Training retains PEFT
random-A/zero-B. This new fixture differs from the earlier744-tensor anchor, so
its loss change cannot be attributed solely to precision, prefetch or disk PLE.
This run establishes a measured reference; it does not demonstrate optimizer
learning quality or equivalence to the old reference's different adapters.

The initial synthetic preflight failed before full model loading because PEFT
could not discover its device; that attempt's source/logs remain in local DVC.
The corrected per-config quantized integration passed before this capture.

No Git or DVC push.

## Gradient retention

Only this current reference retains raw full-sample gradients. Later candidates
compare against it in memory, saving loss and all gradient match/difference
metrics without raw gradient archives. Historical gradient removal is documented
in `gradient-retention.json` and `gradient-retention-kaggle.json`; their compact
reports, original hashes, source and input records remain. DVC pointers now
index the intentionally pruned outputs; older raw-only dependencies are retired.
Local cache removal was targeted by gradient identity, with shared non-gradient
objects protected. No remote DVC deletion or pushes occurred.
