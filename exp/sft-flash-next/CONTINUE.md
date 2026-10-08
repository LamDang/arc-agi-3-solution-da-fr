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
  external assets were unavailable. See committed `verification/pytest.xml`.
  CPU versions: Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
  Transformers 5.18.0. Production CUDA smoke checks have not run.

The replay extension is in `exp/reap-flash-next/replay.py`; new entry points,
tests and pinned requirements are under `exp/sft-flash-next/nll/`.

## Remaining work and blocker

The previous managed workspace's enforced network proxy returned HTTP 403
for `kaggle-arc-agi-3-dvc.s3.eu-west-3.amazonaws.com` and `huggingface.co`.
AWS credentials were reported configured, but the real dataset, separated
REAP statistics and pinned processor were not cached. Recheck this session's
network/credential readiness; do not assume the old restriction still applies.
Use the authorized network path and handle actual redirects through normal
environment configuration. Never bypass a destination-policy denial.

Once inputs are accessible, run the README's CPU-only sequence:

1. `fetch_inputs.py` — existing DVC payloads and pinned HF processor, no weights.
2. `prepare.py` — freeze actual 30 IDs, full target annotations, clean maps,
   level coverage, exact token/time budget and immutable manifest.
3. `run.py --preflight-only` — validate the real prepared bundle.
4. Fix any real-data issues and run relevant tests. Synthetic/tiny-model tests
   are useful but do not substitute for this real-data gate.
5. Rebuild `build_setup.py --bundle ... --wheels ...` and verify the archive.
6. Report GPU readiness, exact estimated cost and remaining GPU smoke checks.

The 42-wheel Python 3.13 Kaggle wheelhouse and ~60 MB staging archive were
generated locally and are **not in Git**. Rebuild them with README commands;
do not rely on old `/tmp` or `build/` files surviving this session. Keep Kaggle's
CUDA torch/torchvision; never install the CPU torch lock onto the GPU image.

No GPU allocation, Kaggle upload, real-data evaluation or actual model
selection has occurred. The chosen 30 requests/maps and exact budget remain
pending. Illustrative cost is 7.692M processed tokens: about 151 scoring
minutes at 850 tokens/s, plus cold starts. Default worker cap is 120 minutes
per session; resume all 30 requests rather than shrinking the sample. The cap
does not deallocate Kaggle itself, so stop/disable its GPU after verifying the
external mirror to avoid idle quota use.

Do not declare GPU-ready until the actual prepared panel passes. When the user
later starts the Kaggle interactive Jupyter session, the worker performs
short-request numerical/chunk-parity and longest-request capacity checks
before the full sweep. Jupyter URL/token and live connection are session inputs.
