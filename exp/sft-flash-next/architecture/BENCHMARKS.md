# Full-context scaling benchmark — pending qualification

Run this matrix after Opt7, Opt8, Opt9 and Opt10 pass their full-anchor gradient
gate: bitwise equality or global relative L2 below 1% against the unchanged
Opt3 control. Keep the native-reference comparison as well. Do not benchmark
an unqualified composition as the accepted optimized architecture.

## Matrix

Use the cumulative `configs/opt10.json` architecture, with 8192-token chunks,
FP32 LoRA masters/gradients on all experts, BF16 activation boundaries, bounded
expert prefetch, disk PLE preparation, target-only CCE exact and direct bias.
Interpret the requested context sizes as decimal token counts:
**32,000; 64,000; 96,000; 120,000**. Run them in ascending order, one detached
GPU process at a time through Kaggle Jupyter HTTP/WebSocket.

Each length needs an immutable encoded benchmark fixture with explicit
interleaved labels and a recorded SHA256. Record actual input tokens, supervised
next-token targets, target fraction, images and provenance. Prefer real
trajectory fixtures; record any diagnostic prefix/repetition as such. Never
silently pad/repeat or present a diagnostic fixture as a complete production
trajectory. Input preparation and source selection remain pending.

## Measurement

For each length use a fresh process, the same model/source/package pins and
exact seeded adapter initialization. Collect a first F/B and one warm repeat
without optimizer updates or clipping. Clear gradients with `set_to_none=True`
between repetitions. Report both repetitions individually; compilation/cache
effects on the first pass must remain visible. If RAM/disk caches persist across
processes, record that state rather than calling the first pass disk-cold.

Record separately for loading/preparation, forward and backward:

- Synchronized elapsed seconds; F/B total and input/target tokens per second.
- CUDA peak allocated and reserved GiB, with counters reset per phase.
- Sampled parent RSS, process-tree PSS including DataLoader workers, child RSS
  and system host-used peak GiB, with polling interval and sample counts.
- Loss and finite-gradient counts; raw candidate gradients are never saved.
- Expert staging bounds, chunk sizes, PLE preparation/transfer timing and cache
  state where instrumented. Mark missing counters explicitly.

The long-context measurement must not execute the automatic unchunked control:
that control may exceed capacity and adds unrelated work/RAM to this benchmark.
Qualification occurs beforehand on the fixed anchor; long-context passes check
finite loss/gradients and capacity, not gradient equivalence to a nonexistent
long-context reference. Avoid cloning the full gradient dictionary merely to
measure capacity. Keep raw gradients only for the saved reference.

These are F/B benchmarks, with no AdamW state or optimizer step. Record that
scope explicitly; they do not measure complete optimizer-inclusive training.
Save configuration, executed source/input hashes, environment, resource records,
loss and supervisor status in local DVC for every attempt, including OOM/timeouts.
Add all results to the existing single `v0.md` table. No Git/DVC push.

## Current prerequisite

The Jupyter connection is unavailable. Reconnect and collect Opt7 attempt
`20261010000620065-39a35a37` before any new GPU job. Opt7–10 are not yet qualified.
The benchmark execution path must omit paired-control and raw-gradient export;
the current qualification `test.py` path is not a capacity benchmark runner.
