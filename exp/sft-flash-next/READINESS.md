# 30-request NLL setup — readiness

2026-10-08. **Code and offline staging package prepared; real-data preflight
blocked. Do not start the GPU yet.** No GPU allocation or Kaggle upload was
performed in this task.

- Fixed core: 30 requests, six per fold-0 game; two per context-length tertile.
- Candidates: 512 / 448 / 384 / 320 / 256 experts, 150 matched forwards.
- Full multimodal contexts, final-reply-only NLL and per-token semantic labels.
- CPU-built train-only expert maps from saved statistics; no fresh calibration.
- Atomic per-request results, strict resume identities and checksums.
- External Jupyter collector tested over a local authenticated HTTP contents
  server, including remote loss and corrupted-transfer rejection.
- Interactive Kaggle notebook defaults to GPU disabled; scoring refuses
  synthetic/incomplete panels and requires an external-mirror acknowledgment.

## Verification completed

**35 tests passed, one existing real-log integration test skipped.** The
skipped test requires `REAP_TEST_LOG`, `REAP_TEST_PROCESSOR` and
`REAP_TEST_EVAL_LOG`, none of which is available here.

Passed checks include sampling/weights, fold exclusion, nested expert maps,
real Hugging Face multimodal processor API and save/reload with synthetic
tokenizer/template, image expansion, semantic spans and executable whitespace,
final-target mask and next-token shift, chunk boundaries, causal independence
from future tokens, text/image NLL against the official tiny Flash-Next model,
masked versus physically pruned parity, interrupted writes, corruption,
resume identity changes, weighted paired reporting, all-30 completion,
collector acknowledgment freshness, Jupyter download recovery, and packaging.

Python compilation, targeted static analysis, JavaScript syntax, notebook-cell
syntax and archive SHA256 inventory checks passed. The wheelhouse has 42
Python 3.13-compatible package wheels (~59 MB); Kaggle retains its own CUDA
torch/torchvision. CPU tests used Python 3.12.14, torch 2.14.1+cpu, torchvision
0.29.1+cpu and Transformers 5.18.0. Production CUDA checks remain pending.

## Remaining gate

The managed workspace's enforced network policy rejects both the existing
DVC S3 host and Hugging Face with proxy HTTP 403. The real JSONL/index,
separated REAP statistics and pinned processor are not cached locally.
Consequently the real 30 row IDs, exact token/time budget, clean map artifacts
and real-data processor preflight have **not** been generated or validated.

Enable access to `kaggle-arc-agi-3-dvc.s3.eu-west-3.amazonaws.com` and
`huggingface.co` through the environment configuration, or run the CPU
preparation commands on an already authorized machine with those inputs.
AWS credentials are reported configured by the runtime; no credential values
are stored in the setup. Subsequent host redirects, if any, must also use the
normal allowed network path.

Then run `fetch_inputs.py`, `prepare.py`, `run.py --preflight-only`, and rebuild
with `--bundle`. Only after those pass is the setup ready for the GPU smoke
step. The current archive is explicitly a **code/wheels-only** staging bundle.

## Artifacts and instructions

- [Setup and commands](nll/README.md)
- [Kaggle interactive notebook](nll/kaggle/kaggle-nll-30.ipynb)
- Offline archive and wheelhouse are generated locally and excluded from Git;
  rebuild with the commands in `nll/README.md`.
- [Machine-readable readiness](verification/readiness.json)
- [JUnit test results](verification/pytest.xml)

At the previous illustrative 51K context length, 30 requests imply 7.692M
processed tokens across candidates: roughly 151 scoring minutes at 850
tokens/s. With two cold, 120-minute-capped worker sessions, plan around
3.2–3.5 hours total; at 500 tokens/s, around 5.3–5.8 hours across three.
These estimates await actual selected lengths and measured GPU speed. The
worker cap does not deallocate the Kaggle session: stop it after verifying
the external mirror to avoid spending quota while idle.
