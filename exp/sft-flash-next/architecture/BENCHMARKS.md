# Full-context scaling benchmark

Run this matrix after Opt7, Opt8, Opt9 and Opt10 pass their full-anchor gradient
gate: bitwise equality or global relative L2 below 2% against the unchanged
Opt3 control. Keep the native-reference comparison as well. Do not benchmark
an unqualified composition as the accepted optimized architecture. Opt7–10
passed this gate; the user waived an additional numerical gate for the later
GPU LoRA/shared pinned allocation correction.

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
trajectory. Prepared fixtures use `lf52-271a04aa_p0#36` from the completed
progressive-sol25 dataset: full context121,022 tokens/32,283 assistant targets.
The four diagnostic prefixes contain5,816/15,241/23,937/31,261 targets and
16/33/48/59 images. `prepare_benchmarks.py` uses the existing trajectory encoder
and rejects prefix boundaries inside vision blocks. Exact tensors, annotation,
encoder sources and hashes are in local DVC `results/20261010-benchmark-fixtures`.
Production full-trajectory preparation is unchanged.

## Measurement

For each length use a fresh process, the same model/source/package pins and
exact seeded adapter initialization. The 32K/64K and corrected 96K captures
include a first F/B and one warm repeat. The corrected 120K capture uses one
full capacity pass, explicitly without a repeat comparison. No optimizer updates
or clipping. Clear gradients with `set_to_none=True`
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

## Completion shutdown

The user authorizes killing the Kaggle Jupyter server after all requested work
is complete. First collect every required attempt, verify artifact hashes and
local DVC cache, and save the final statistics. Then stop the Jupyter server
through the existing server connection and verify process exit where possible.
Do not shut it down while qualification, benchmarks or collection remain pending.
An unreachable connection alone is not proof of shutdown.

## Completed captures

The replacement server356919557 is connected. The original server's Opt7 attempt
`20261010000620065-39a35a37` cannot be recovered there; its gate remains unknown.
The corrected fold0 export was verified against the original checkpoint index
and84 expert/router byte checks. Native replay20261010040404197-7f737faf passed
bitwise loss and all74,472 gradients with the test-only FLA profile. Real training
rejects that profile and uses native autotuning. Opt7–10 subsequently passed
their full-anchor gates. Historical 32K/64K captures used CPU expert LoRA;
the original 96K attempt was killed during forward. The corrected profile
keeps all FP32 LoRA parameters/gradients on GPU and uses one shared pinned CPU
allocation for frozen experts. Corrected 96K completed and passed evidence/DVC
checks before 120K dispatch; corrected 120K also completed and passed.
Both corrected runs have unchanged OOM counters. All 30 attempt caches were
verified. Reports are `reports/benchmark-96000-gpulora.json` and
`reports/benchmark-120000-gpulora.json`; all phase measurements and historical
profiles are in the single `v0.md` table. No optimizer state was measured.
The dedicated `benchmark.py` path omits paired-control, raw-gradient export and
full gradient dictionary clones. Configuration requires `benchmark_tokens`,
explicit labels, `benchmark_repeats` (default2) and the saved initial-adapter pin.
Dispatch with `node jupyter.mjs --mode benchmark --config CONFIG --timeout-seconds SECONDS`
only after qualification and immutable fixture preparation. The executed configs are
`configs/benchmark-{32000,64000,96000,120000}.json`; each pins its exact fixture
and the fold0 expert selection. Review captures with `reports/verify_benchmark.py`.

## Shutdown observation

After all captures and all 30 local DVC caches passed verification, Jupyter
server PID 12 acknowledged SIGTERM at 11:09:41 UTC. Its API returned HTTP 404
at 11:10:20 UTC. The identified server was the control kernel's parent and its
command hash was rechecked before signalling. An independent process query
after shutdown was unavailable. See `reports/kaggle-shutdown.json`.
