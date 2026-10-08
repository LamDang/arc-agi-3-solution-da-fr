# 30-request NLL setup — readiness

2026-10-08. **CPU preparation and real-data preflight complete. Ready for the
user-authorized GPU smoke step; no GPU session has started.** No Kaggle upload,
training, gameplay or model selection was performed.

The frozen panel has six requests per fold-0 game, two per context-length
tertile, seed 20261008. All five candidates (512/448/384/320/256) use the same
full multimodal contexts and final-reply-only targets: 30 requests, 150 forwards.

## Exact panel and budget

- Prompt tokens: **1,268,524**; final-reply tokens: **12,108**.
- Sweep: **6,403,160 processed tokens**, **60,540 scored target tokens**.
- Full request lengths: **5,770–84,298 tokens**; **664 images** across the panel.
- Coverage: **15 of 35 game-level cells sampled**; 20 explicitly not sampled.
- All selected replies are tool calls. The dataset's sole terminal text-only
  reply is not sampled; no extra request or reserve was added.

| Rate assumption | Scoring only | Cold sessions at 120-minute cap | Total with 20–30 min/session overhead |
|---|---:|---:|---:|
| 850 tokens/s | 125.6 min | 2 | 165.6–185.6 min (2.8–3.1 h) |
| 500 tokens/s | 213.4 min | 3 | 273.4–303.4 min (4.6–5.1 h) |

These are planning estimates, not measured production performance. The
manifest's `estimated_minutes` adds a single 30-minute startup; the table
accounts for each cold session. Numerical/chunk-parity and longest-request
capacity checks remain pending on RTX PRO 6000 Blackwell 96 GB. Session
admission margins, cold-start variation and smoke repeats may extend runtime.
The worker's cap does not deallocate Kaggle; stop the session after verifying
the durable mirror to avoid idle quota use.

## Verification completed

**35 CPU tests passed; one existing real-log integration test skipped.** It
requires old SGLang request logs via `REAP_TEST_LOG` / `REAP_TEST_EVAL_LOG`, which
are not the new Sol dataset. This skip does not replace or block the separate
real-data gate, which passed for all 30 selected Sol requests.

- Both previously blocked hosts were reached using the authorized network
  path. All three dataset DVC hashes and pinned tokenizer/template hashes pass.
- Saved calibration statistics and metadata match the pinned DVC tree. Run
  identities, category order and model version are verified before aggregation.
- All **20 training-game runs** contribute; all **five validation-game runs**
  are excluded. The 48-layer maps are nested and independently match the
  repository's `analyze.py` aggregation/ranking exactly.
- Preparation verifies full prompt/image prefix equality, target suffix,
  final-only mask and semantic labels. Offline preflight re-encodes all 30
  requests and matches every stored annotation and manifest field.
- The complete archive has real data, 42 offline Python 3.13 wheels and a
  checksum inventory. Offline dependency resolution, archive SHA256 inventory,
  notebook syntax and disabled-GPU defaults are verified.

CPU: Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
Transformers 5.18.0. Keep Kaggle's own CUDA torch/torchvision stack.

## Frozen identities and artifacts

Panel manifest SHA256:
`f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c`.

Private generated artifacts in this workspace (outside Git):

- Panel: `/workspace/sol-nll-artifacts/panel30`.
- Complete package: `/workspace/sol-nll-artifacts/complete-setup-final/sol-nll-setup.zip`.
- Notebook: `/workspace/sol-nll-artifacts/complete-setup-final/kaggle-nll-30.ipynb`.
- Downloaded inputs: `/tmp/sol-nll-inputs`; wheels: `/tmp/sol-nll-wheels-cp313`.
- CPU environment: `/tmp/sol-nll-venv`.

Generated inputs/archives are not committed and may not survive a new
workspace. Rebuild using [nll/README.md](nll/README.md). Keep teacher transcripts
private when later staging. Archive checksum/size are recorded in
[verification/readiness.json](verification/readiness.json).

Committed records: [real panel, maps and budget](verification/real-data.json),
[JUnit results](verification/pytest.xml),
[offline package verification](verification/package.json), and
[real-data preflight log](verification/preflight.log).

Next step requires the user's GPU-start instruction and live interactive
Jupyter connection. Start the external collector, then authorize the disabled
launch cell. The worker runs production numerical/chunk/capacity checks before
continuing the same frozen panel. GPU correctness, actual throughput/VRAM,
model selection and training capacity remain unmeasured.

## Repository dataset

The exact 30 full requests are also published in the repository's existing
private DVC remote as
[`data/sol-nll-fold0-30`](../../data/sol-nll-fold0-30/README.md).
The 5.7 MiB `requests.jsonl` payload has a Git-tracked DVC pointer, with a
Git-tracked index, source hashes, selection provenance and verification script.
Fetch it using `dvc pull data/sol-nll-fold0-30/requests.jsonl.dvc`.
This is the same frozen panel; exporting it did not resample or modify request
objects. Processor/maps remain in the complete offline package.
