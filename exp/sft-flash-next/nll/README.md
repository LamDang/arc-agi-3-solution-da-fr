# Sol NLL evaluation: 30 requests, ready-to-stage code

The user-selected core is **30 requests / 150 model-request forwards**: two
requests from each context-length tertile in each of the five fold-0 games.
There is no automatic reduction to 10 or 15 requests. Incomplete sessions
resume the same frozen panel. This code does not train or play games.

## Current readiness

The scorer, sampler, annotations, report, atomic results, Jupyter collector
and notebook are implemented. CPU tests use both format fixtures and the
repository's tiny real Flash-Next architecture. They are not measurements of
production CUDA performance or of the real sol validation sample.

**The real-data preflight is blocked in the current managed workspace.** Its
enforced network policy rejects the DVC S3 bucket and Hugging Face with proxy
HTTP 403. The dataset, original tokenizer/processor and separated calibration
statistics are not cached here. The setup archive can be built without them,
but is explicitly marked `has_real_panel: false` and cannot launch scoring.
No GPU session has been started or uploaded by this setup.

To finish preparation in this workspace, network access is needed to:

- `kaggle-arc-agi-3-dvc.s3.eu-west-3.amazonaws.com` for the existing DVC inputs;
- `huggingface.co` and any download hosts it redirects the pinned small
  processor files to. Actual redirects should be allowed through the normal
  environment configuration, not bypassed.

The runtime reports AWS credentials configured. No credential values are
stored in source or artifacts. An already authorized machine with these inputs
can run the same CPU preparation commands instead. A live Kaggle Jupyter URL
and its authentication are needed only when the user starts the GPU session.

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
statistics and processor files; **no model weights**. Calibration maps are
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
two chunk sizes. It then checks the longest selected request and completes
the paired panel. Smoke/capacity results that are valid count toward the panel.
A failed check stops before the full sweep. GPU-specific correctness and
memory checks are pending until this first GPU smoke step.

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

The report uses only requests completed by **every** expert configuration for
paired comparisons. It produces CSV/JSON summaries, granular slices, level
coverage, `selection.json` and an interactive HTML token viewer. Partial
results never automatically select a model. Per-token outputs are exact
full-vocabulary NLL on the teacher's final reply. History is context only.
Labels distinguish logged rationale, Python code, tool format, prose and turn
format; tokens crossing a category boundary remain explicitly marked.

The primary score is the weighted mean request NLL within each game, then
an equal mean across games. Token-weighted NLL/PPL and categories are separate
metrics. The recommendation is the smallest expert count within 0.05 nats of
the best primary score, with code/game/leave-one-game-out diagnostics. This
selects an imitation candidate; training capacity is still unverified.

At the previous illustrative 51,280 tokens/request, 150 forwards process
**7.692M tokens**: about **151 minutes at 850 tokens/s**, or **256 minutes at
500 tokens/s**, plus roughly 20–30 minutes for each cold session. A typical
complete run therefore needs around three hours and likely spans two capped
sessions; slower hardware takes longer. Exact estimates are emitted by CPU
preparation. Thirty replies × 280 targets × five models need only **168 KB**
for the FP32 loss vectors. Context processing dominates cost.

The optional extra diagnostic reserve in the design document is not implemented
as an automatic extension. This setup stops after the requested 30-request core.
