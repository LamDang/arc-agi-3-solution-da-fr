# Sol NLL evaluation: 30 requests, CPU preparation complete

The user-selected core is **30 requests / 60 initial forwards, up to 150 if needed**: two
requests from each context-length tertile in each of the five fold-0 games.
There is no automatic reduction to 10 or 15 requests. Incomplete required stages
resume the same frozen panel. This code does not train or play games.

## Current readiness

The scorer, sampler, annotations, report, atomic results, Jupyter collector
and notebook are implemented. CPU tests use both format fixtures and the
repository's tiny real Flash-Next architecture. They are not measurements of
production CUDA performance or of the real sol validation sample.

**The real-data CPU gate passed on 2026-10-08.** All 30 frozen requests were
processed twice with the pinned processor, with identical full-context token
IDs, final-reply positions and semantic annotations. All five fold-0 games
are excluded from the saved calibration statistics; all 20 training games
contribute to the nested 48-layer expert maps. GPU qualification and the original
512/256 panel completed; the generated-thinking panel is running. See `../READINESS.md`.

The current instructed protocol is **512 first, then 256** on all 30 requests,
using `--initial-pair-only`. Finish both original and generated-thinking pairs,
then wait for the user to decide whether to expand either panel. The relative
+5% NLL gate is reported but does not trigger automatic expansion in this mode.
The historical automatic protocol remains available when neither manual-scope
flag is supplied; see the archived first-run code for that exact implementation.

The panel contains 1,268,524 prompt tokens and 12,108 final-reply tokens.
The initial pair processes **2,561,264 tokens** (24,216 scored targets):
**50.2 scoring minutes at 850 tokens/s**, or **85.4 at 500 tokens/s**.
With one 20–30 minute cold start, estimate **70.2–80.2 minutes** nominally,
or **105.4–115.4 minutes** conservatively; session margins can require resume.
If the gate fails, the maximum remains **6,403,160 processed tokens** across
150 forwards, about **2.8–3.1 hours** nominally or **4.6–5.1 hours** conservatively.
The unchanged frozen manifest records maximum capacity; `--preflight-only`
prints the current `staged_budget`. Production kernel, parity and capacity
qualification passed for the pinned Kaggle image.

The complete private archive was rebuilt with real data and 45 offline
Python 3.13 wheels. Generated data/wheels/archives remain outside Git.
See [readiness](../READINESS.md) and the committed verification records for
identities, coverage and artifact locations. S3 and Hugging Face downloads
succeeded through this session's authorized network path; future sessions
must recheck access. A live Kaggle Jupyter URL and authentication are needed
only when the user starts the GPU session.

## 1. Prepare inputs without allocating a GPU

The existing dataset is pinned to main's PR #19 revision
`37fadfeedfd54d2129db4525e4167596cc7337b6`. The three DVC payload hashes and
the original tokenizer/template hashes are verified by `prepare.py`.

From the repository root, on an internet-enabled CPU machine:

```bash
python -m venv /tmp/sol-nll-venv
/tmp/sol-nll-venv/bin/pip install -r exp/sft-flash-next/nll/requirements-cpu.lock
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/fetch_inputs.py --out /tmp/sol-nll-inputs
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/prepare.py \
  --data-dir /tmp/sol-nll-inputs/data \
  --processor /tmp/sol-nll-inputs/processor \
  --stats-dir /tmp/sol-nll-inputs/calib \
  --folds data/game_folds/folds.json \
  --out /tmp/sol-nll-panel30
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/run.py \
  --bundle /tmp/sol-nll-panel30 --out /tmp/unused-nll-output --preflight-only
```

The fetcher downloads the ~333 MB JSONL, its small index/meta, saved REAP
statistics, per-run provenance and processor files; **no model weights**. Calibration maps are
rebuilt on CPU, excluding every fold-0 game. The preparer creates:

- immutable sampling weights and selected IDs (seed 20261008);
- exactly 30 standalone request JSONs with complete images/history;
- processor-verified target IDs, positions, semantic labels and display spans;
- clean nested expert maps and per-run calibration provenance;
- exact processed-token budget, level coverage and runtime estimates;
- a checksummed completion manifest, written only after all checks pass.

It requires an empty output directory, rejects hash mismatches and never
silently changes the sample, truncates a request or drops a failed input.

## 2. Test and package

```bash
OMP_NUM_THREADS=2 /tmp/sol-nll-venv/bin/python -m pytest -q \
  exp/sft-flash-next/nll/tests \
  exp/reap-flash-next/tests/test_model.py \
  exp/reap-flash-next/tests/test_prune_checkpoint.py
```

Build the wheelhouse for Kaggle's observed Python 3.13 image. It deliberately
does not contain CPU torch or replace the image's torch/torchvision/CUDA stack:

```bash
/tmp/sol-nll-venv/bin/pip download --only-binary=:all: \
  --python-version 313 --implementation cp --abi cp313 \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
  --platform manylinux2014_x86_64 \
  -r exp/sft-flash-next/nll/requirements-kaggle.lock \
  --dest /tmp/sol-nll-wheels-cp313
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/build_setup.py \
  --wheels /tmp/sol-nll-wheels-cp313 --bundle /tmp/sol-nll-panel30 \
  --out /tmp/sol-nll-complete-setup
```

`build_setup.py` produces a SHA256 inventory, `sol-nll-setup.zip` and
`kaggle-nll-30.ipynb`. It neither uploads anything nor starts a Kaggle kernel.
Omitting `--bundle` creates a code/wheel-only artifact explicitly marked as
missing real data, useful for staging but not sufficient for GPU readiness.

## 3. Kaggle interactive Jupyter setup

Import the notebook into a competition-attached interactive session. Attach
the private setup archive and the full immutable model version:

`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1`

Run the setup and CPU preflight cells with GPU disabled. The notebook verifies
archive checksums and exact Python version, installs offline packages and
validates the frozen real-data bundle. Keep the dataset/teacher transcripts
private when staging them. This repository does not assume a public upload.

The runtime is for **RTX PRO 6000 Blackwell 96 GB** with ample host memory,
not a 24/48 GB RTX 6000. Full-resident historical replay exceeded A100 80 GB;
this worker intentionally rejects an unqualified smaller GPU profile.

When the data preflight and collector connection are ready, enable the GPU
in the interactive notebook and set `START_GPU_RUN=True` in the final launch
cell. The worker first checks kernels and repeats a short real request with
two chunk sizes. It then checks the longest selected baseline request, completes all 512 jobs,
then all 256 jobs, and reports the gate without automatically expanding.
Smoke/capacity results that are valid count toward the panel.
A failed check stops before the full sweep. Both GPU-specific correctness and
memory checks must pass before any full-panel comparison.

The session cap is 120 minutes by default. Start/end times include loading;
new work is admitted conservatively with a persistence margin. At the cap,
the worker pauses and keeps the 30-request panel fixed for the next session.
The process lock prevents duplicated evaluation after browser reconnects.
The cap stops the scoring worker; it does not deallocate Kaggle's GPU session.
After verifying the external mirror, stop the interactive session or disable
its GPU to stop spending quota while idle.

## 4. Durable resume through the Jupyter server

Run the collector on a durable external machine. Set `JUPYTER_BASE_URL` and
`JUPYTER_TOKEN` in its environment; do not put them into a notebook, dataset,
shell history command or git. Remote connections require HTTPS.

```bash
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/collect.py \
  --remote sol-nll --out /durable/path/sol-nll --interval 60
```

`--remote` is relative to the Jupyter contents root; use `kaggle/working/sol-nll`
instead if the server exposes filesystem root. The collector uses authenticated
GETs to download completed result pairs, checks their hashes, and PUTs only a
small `mirror-ack.json` acknowledgment into the output directory. It never
executes code or creates a session. The worker pauses if acknowledgment is
missing or more than five minutes old. Start the collector before the worker;
it retries while the output directory is being created.

To resume in a fresh session, archive the **contents** of the verified durable
output directory, attach/upload that archive, and use the notebook restore
cell to extract into an empty `/kaggle/working/sol-nll`. Reuse the same prepared
bundle, code, runtime packages and model version. Reconnect the collector.
Completed request/model pairs are skipped only after identity and checksum
validation. An interrupted request is replayed from its start. No giant model
checkpoint or cache is transferred. Remove a previous `STOP` marker only when
intentionally continuing. A changed scoring environment requires a new run.

The worker's model identity uses the immutable Kaggle model version, config/
index hashes and shard sizes; it does not claim to hash every weight byte.
Scoring on arbitrary mutable local checkpoints is rejected by this entry point.

## 5. Results and interpretation

```bash
/tmp/sol-nll-venv/bin/python exp/sft-flash-next/nll/report.py --out /durable/path/sol-nll
```

The report uses only requests completed by **every required** expert configuration
for paired comparisons: 512/256 initially, all five only if the gate requires a scan. It produces CSV/JSON summaries, granular slices, level
coverage, `selection.json` and an interactive HTML token viewer. Partial
results never automatically select a model. Per-token outputs are exact
full-vocabulary NLL on the teacher's final reply. History is context only.
Labels distinguish logged rationale, Python code, tool format, prose and turn
format; tokens crossing a category boundary remain explicitly marked.

The primary score is the weighted mean request NLL within each game, then
an equal mean across games. Token-weighted NLL/PPL and categories are separate
metrics. The recommendation is 256 if the complete initial pair passes the 5% relative
primary-NLL gate; otherwise it is the smallest scanned configuration within
5% of the 512 baseline, with code/game/leave-one-game-out diagnostics. This
selects an imitation candidate; training capacity is still unverified.

The initial pair has 60 forwards and 2,561,264 processed tokens. Conditional
intermediates add 90 forwards and 3,841,896 tokens only if the gate fails.
Keep all 30 requests fixed across stages and sessions. The protocol is part
of the strict resume identity; old unconditional-sweep runs cannot silently
resume under this code. See [the active protocol](../NLL_EVAL.md) for the
inclusive threshold, selection rule and exact initial/maximum budgets.

The optional extra diagnostic reserve in the design document is not implemented
as an automatic extension. This setup stops after the 512/256 pair passes, or after the triggered five-model
comparison, always using the same 30-request core.

## Qualified CUDA kernels and chunk-stable scoring (2026-10-08)

The observed Kaggle image uses Python 3.13.15, torch 2.11.0+cu128,
CUDA 12.8, Triton 3.6.0 and CXX11 ABI=true on RTX PRO 6000 Blackwell.
The original 42-wheel setup lacked fast kernels; the worker now rejects
missing/unqualified FLA or missing native convolution before loading weights.

Add these wheels to the existing CPU-prepared wheelhouse:

- `flash_linear_attention-0.5.2-py3-none-any.whl`
- `fla_core-0.5.2-py3-none-any.whl`
- `causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl`

The native convolution wheel was built from the PyPI 1.7.0 source distribution
on the exact Kaggle image. A torch-2.10 GitHub wheel is not ABI-compatible
with torch 2.11. To rebuild on the matching image (requires nvcc and ninja):

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=2 python -m pip wheel \
  --no-deps --no-build-isolation /path/to/causal_conv1d-1.7.0 \
  --wheel-dir /path/to/qualified-wheels
```

Copy all 45 wheels into one wheelhouse and use the existing `build_setup.py`
command. The existing notebook installs the three pinned kernels through
`requirements-kaggle.lock`. The image already supplies their torch dependency;
no torch wheel is included and no replacement is requested. Native wheel SHA256:
`f928f1aa1de1306f26f58a3f2c7a6d9ac2d696090fd1bd26430951d82be9928b`.
The [exact qualified convolution wheel](../../../data/sol-nll-causal-conv1d-20261008/README.md)
used by both panels is preserved in private DVC with retrieval instructions
and build provenance; it can be reused on the matching image without rebuilding.
The wheel is specific to this Python/torch/CUDA/ABI combination; rebuild and
requalify when the image changes. Generated wheels and archives stay outside Git.

The production BF16 smoke check exposed shape-dependent GEMM rounding, which
amplified through hard routing across 48 layers. `numerics.py` uses fixed
256-row padded dense and expert projections, plus a fixed top-k expert sum
order. It preserves weights, full contexts, pruning masks and teacher-forced
final-reply targets. Reduced-precision BF16/FP16 reductions are disabled.
The scoring-numerics policy is included in the run identity; results from
previous unqualified attempts cannot be resumed into the corrected run.
The 0.01-nat per-token chunk-parity tolerance is unchanged. The corrected
short-request diagnostic matched exactly at chunk sizes 8192 and 4096.

## Generated-thinking panel

`prepare_variant.py` prepares `data/sol-nll-fold0-30-genthink` from the same
frozen source bundle. It audits code-only system instructions, Python schemas
and every historical/final call, restores render-sensitive dictionary order,
and verifies token-identical prompts/images/Python code. It reuses all source
sampling weights and train-only expert maps. See the
[variant dataset documentation](../../../data/sol-nll-fold0-30-genthink/README.md)
for the pinned revision, preparation command and separate thinking/tool loss
budget. Use a separate bundle and result directory: source and variant runs
have different manifest identities and must never be mixed.


## Current two-panel decision protocol

The source panel completed all 60 evaluations: primary NLL 0.85856035 at
512 experts and 0.85335082 at 256 (-0.6068%). No intermediate counts ran.
The generated-thinking panel runs 512 then 256 only, using
`run.py --initial-pair-only` (also the rebuilt notebook default). After both
panels finish, the user decides whether to expand either panel. A >5% increase
is reported but does not automatically launch further evaluations in this mode.
An explicitly requested expansion uses `--expand-counts 448 384 320` (or a
chosen subset) and reuses verified initial-pair results. Restore `scope.json`
with the other durable run files when resuming a manual run.

The original Kaggle session's two-hour deadline is 2026-10-08 13:53:43 UTC.
It must not be reset implicitly to start the second panel.


## Paired panel comparison and archival

The current user-selected reference criterion compares thinking and Python-code
NLL separately, rather than mixing their changed token proportions. After both
512/256 panels complete:

```bash
python exp/sft-flash-next/nll/compare_panels.py \
  --source /durable/source-panel --variant /durable/genthink-panel \
  --out /durable/panel-comparison --reference-metric categorywise
```

The comparison requires all 120 result records, validates the frozen source/
variant relationship, matching prompts/sampling/maps/code token IDs and scoring
configurations, and checks each result checksum and token position. It records
60 paired request/model rows, separate category means and per-game/context
slices. Python token IDs match exactly and receive aligned per-token loss diffs;
changed thinking traces are compared by request, not by token position.

If both baseline category NLLs improve, the variant becomes the reference.
If either reference-panel category NLL worsens by >5% at 256, use the exact
staged runtime to resume with `--expand-counts 448 384 320`; otherwise no further
GPU counts run. The overall primary score remains a diagnostic.

`archive_results.py` verifies completion and preserves the exact runtime (its
hash must match `run.json`), dependency locks, notebook, all required token-loss
arrays, maps, reports, optional panel comparison and separate analysis code.
It excludes credentials, collector state and model/wheel/processor binaries.

```bash
python exp/sft-flash-next/nll/archive_results.py \
  --root /durable/genthink-panel --bundle /prepared/genthink-bundle \
  --setup /exact/offline-setup --comparison /durable/panel-comparison \
  --out data/sol-nll-fold0-30-genthink-results-20261008
```

Add/push the resulting `results.zip` with DVC and commit the pointer, provenance
and scalar summaries. Verify a download from the DVC remote before shutting
Kaggle down. The staged scoring code remains unchanged during the run; analysis
code is stored separately to preserve strict resume identity.
