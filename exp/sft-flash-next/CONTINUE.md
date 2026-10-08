# Continue the 30-request Sol NLL setup

Repository: `LamDang/arc-agi-3-solution-da-fr`.
Branch: `codex/sol-nll-30-setup`, based on dataset merge main
`37fadfeedfd54d2129db4525e4167596cc7337b6`.

The user's latest implementation instruction was: “15 request is a bit light,
should do 30. prepare the setup, make sure to test it and tell me when you are
ready to start the gpu.” **Do not allocate/start the GPU yet.** Finish CPU
preparation, verify the actual data, then report readiness for GPU startup.

## Read first

1. `exp/sft-flash-next/READINESS.md` — measured test results and remaining gate.
2. `exp/sft-flash-next/nll/README.md` — exact fetch/prepare/test/package commands.
3. `exp/sft-flash-next/NLL_EVAL.md` — scoring, sampling and budget contract.

`PLAN.md` and `RESEARCH.md` retain the later SFT/QLoRA research. Do not start
training, LR/rank sweeps, gameplay, full-fold scoring or an extra sampling
reserve as part of this current task. Selection is NLL-only to conserve quota.

## Implemented and tested

- 30 requests: six per fold-0 game, two uniformly sampled from each within-game
  context-length tertile; seed 20261008. Games: sk48, sp80, tn36, cd82, ar25.
- All five expert counts: 512/448/384/320/256, 150 matched forwards. Complete
  multimodal contexts, no truncation. Only final teacher replies are scored.
- Per-token NLL with rationale/code/tool-format/prose labels and game/level/
  context-length metadata, paired weighted reporting and interactive HTML.
- One full W4A16 model resident on RTX PRO 6000 Blackwell 96 GB; fresh caches
  with router masks for each candidate. This is Qwen3.8-Flash-Next/qwen4_exp,
  not Qwen3-Next-80B. A100 80 GB is not qualified for this full-resident sweep.
- CPU reconstruction of nested train-only expert maps from saved per-game
  REAP stats. Exclude all five validation games. Existing all-game maps are
  validation-exposed; the implemented entry point requires clean maps.
- Atomic/checksummed results, strict resume identities, quota-aware pauses,
  duplicate-worker lock and externally verified Jupyter downloads. Collector
  credentials come from environment variables; never print or commit values.
- Notebook: `exp/sft-flash-next/nll/kaggle/kaggle-nll-30.ipynb`; GPU disabled.
- 35 CPU tests passed; one old real-log integration test skipped because its
  external assets were unavailable. The separate real Sol gate now passes. See committed `verification/pytest.xml`.
  CPU versions: Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
  Transformers 5.18.0. Production CUDA smoke checks have not run.

The replay extension is in `exp/reap-flash-next/replay.py`; new entry points,
tests and pinned requirements are under `exp/sft-flash-next/nll/`.

## CPU preparation completed in the continuation

Read `READINESS.md` and `verification/real-data.json` for measured preparation
results. The actual panel is frozen and passes offline reprocessing:
**6,403,160 processed tokens / 150 forwards**, with 12,108 final-reply tokens
per candidate. No contexts were truncated or requests replaced.

Both S3 and Hugging Face downloads succeeded through the authorized network
path in this session. The fetcher now retains calibration metadata and uses
writable HF download caches because the managed home directory is read-only.
Map building verifies calibration run/category/model provenance; all 20
training games contribute and all five validation games are excluded.
Independent comparison against `analyze.py` and nesting checks pass.

`run.py --preflight-only` now re-encodes every selected request offline and
compares all target annotations, rather than relying only on file checksums.
35 CPU tests passed; the existing old-log integration test remains skipped.
The separate real Sol data gate passed. Production CUDA checks have not run.

The complete private offline package is at
`/workspace/sol-nll-artifacts/complete-setup-final/sol-nll-setup.zip`, with the
panel at `/workspace/sol-nll-artifacts/panel30`. These generated files, inputs
and wheels are outside Git. If absent in a future workspace, rebuild using
the README; the committed verification records retain the frozen identities
and chosen request IDs for comparison. Do not reroll or shrink the panel.

At 850 tokens/s, expect 125.6 scoring minutes, about 2.8–3.1 hours across two
cold capped sessions including overhead. At 500 tokens/s, expect 213.4 scoring
minutes, about 4.6–5.1 hours across three. These remain estimates until the GPU
smoke step measures speed. Session caps pause the worker, not Kaggle billing.

## Next action

**Report readiness and wait for the user's GPU-start instruction.** Do not
start a GPU, Kaggle upload, training or gameplay merely because CPU gates pass.
When authorized, use the qualified RTX PRO 6000 Blackwell 96 GB interactive
session, stage the private archive and immutable model version, and connect
the external Jupyter collector. Live Jupyter URL/token are session inputs.
The notebook is saved with GPU disabled and `START_GPU_RUN=False`.

The first GPU worker performs numerical/chunk-parity and longest-request
capacity checks before completing the paired sweep. Failed checks stop the
run; valid smoke forwards count toward the panel. Preserve the manifest,
code/runtime and model identities for resume. No extra reserve or full-fold
scoring is authorized as part of this 30-request evaluation.

## Small DVC dataset

The selected requests are now retained in
`data/sol-nll-fold0-30/requests.jsonl` (5.7 MiB), pushed to the existing private
S3 DVC remote. Git tracks its DVC pointer, `index.json`, `provenance.json`,
`export.py` and README. A fresh download was checked against both MD5 and
SHA256. Fetch with `dvc pull data/sol-nll-fold0-30/requests.jsonl.dvc`, then
verify with `python data/sol-nll-fold0-30/export.py`. Full objects, contexts,
images and final replies are unchanged; JSON serialization alone differs.
