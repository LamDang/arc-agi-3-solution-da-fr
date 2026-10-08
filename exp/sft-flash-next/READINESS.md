# 30-request NLL setup — readiness

2026-10-08. **Real-data preparation passed; the user started the Kaggle
session and GPU qualification is underway.** The corrected short-request
smoke passed with exactly zero per-token difference for an exact repeat and
8192/4096 chunk sizes. The 84,298-token longest request also passed capacity; the staged panel
is now running. No training or gameplay was performed.

The initial GPU attempt lacked fast CUDA kernels. FLA 0.5.2 and a native
causal-conv1d 1.7.0 wheel built against the exact Kaggle torch 2.11.0+cu128
image were installed and numerically qualified. The first accelerated smoke
then exposed BF16 projection-shape rounding amplified through hard routing
(mean absolute per-token chunk delta 0.1702 nats). It stopped automatically.
`nll/numerics.py` fixes dense/expert GEMM row sizes and expert sum order;
weights, contexts, targets and the 0.01-nat tolerance are unchanged. Unqualified
attempt results are archived separately and excluded from the corrected run.

The frozen panel has six requests per fold-0 game, two per context-length
tertile, seed 20261008. All five candidates (512/448/384/320/256) use the same
full multimodal contexts and final-reply-only targets: **60 initial forwards,
150 maximum forwards**. Run all 512 requests, then all 256 requests. Stop if
256 primary NLL is at most **1.05 × the full baseline**; otherwise scan
448, 384 and 320. The relative gate requires all 30 matched requests.

## Exact panel and budget

- Prompt tokens: **1,268,524**; final-reply tokens: **12,108**.
- Initial 512/256 pair: **2,561,264 processed tokens**, **24,216 scored targets**.
- Maximum triggered scan: **6,403,160 processed tokens**, **60,540 scored targets**.
- Full request lengths: **5,770–84,298 tokens**; **664 images** across the panel.
- Coverage: **15 of 35 game-level cells sampled**; 20 explicitly not sampled.
- All selected replies are tool calls. The dataset's sole terminal text-only
  reply is not sampled; no extra request or reserve was added.

| Rate assumption | Scoring only | Cold sessions at 120-minute cap | Total with 20–30 min/session overhead |
|---|---:|---:|---:|
| Initial @850 tokens/s | 50.2 min | 1 | 70.2–80.2 min |
| Initial @500 tokens/s | 85.4 min | 1 | 105.4–115.4 min |
| Maximum @850 tokens/s | 125.6 min | 2 | 165.6–185.6 min (2.8–3.1 h) |
| Maximum @500 tokens/s | 213.4 min | 3 | 273.4–303.4 min (4.6–5.1 h) |

These are planning estimates, not measured production performance. The
manifest's `estimated_minutes` adds a single 30-minute startup; the table
accounts for each cold session. The corrected short-request numerical/chunk-parity check passed; longest-request
capacity passed on RTX PRO 6000 Blackwell 96 GB. Session
admission margins, cold-start variation and smoke repeats may extend runtime.
The worker's cap does not deallocate Kaggle; stop the session after verifying
the durable mirror to avoid idle quota use.

## Verification completed

**54 CPU tests passed; four existing real-log tests skipped** when running
all REAP and NLL tests. They require old SGLang request logs via `REAP_TEST_LOG`,
`REAP_TEST_PROCESSOR` / `REAP_TEST_EVAL_LOG`, which are not the new Sol dataset. This skip does not replace or block the separate
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
- Tests verify stage ordering, complete-pair gating, the inclusive 5% relative
  threshold, conditional scan, early completion and idempotent resume.
- The complete archive has real data, 45 offline Python 3.13 wheels (including exact-image CUDA kernels) and a
  checksum inventory. Offline dependency resolution, archive SHA256 inventory,
  notebook syntax and disabled-GPU defaults are verified.

CPU: Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
Transformers 5.18.0. Keep Kaggle's own CUDA torch/torchvision stack.

## Frozen identities and artifacts

Panel manifest SHA256:
`f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c`.

Use the **qualified** package below. Earlier archives lack either the
staged protocol, the native kernels or chunk-stable scoring.

Private generated artifacts in this workspace (outside Git):

- Panel: `/workspace/sol-nll-artifacts/panel30`.
- Complete package: `/workspace/sol-nll-artifacts/complete-setup-qualified/sol-nll-setup.zip`.
- Notebook: `/workspace/sol-nll-artifacts/complete-setup-qualified/kaggle-nll-30.ipynb`.
- Downloaded inputs: `/tmp/sol-nll-inputs`; wheels: `/tmp/sol-nll-wheels-cp313-qualified`.
- CPU environment: `/tmp/sol-nll-venv`.

Generated inputs/archives are not committed and may not survive a new
workspace. Rebuild using [nll/README.md](nll/README.md). Keep teacher transcripts
private when later staging. Qualified archive SHA256:
`af34abe844dafe9e60e24059469a737bda5897bead4c97d1f13170474cc5f9ad`.
Its 140-file inventory, 45 wheels and disabled-GPU notebook passed validation.
The archive runtime code hash matches the active worker exactly:
`8b85a370ba6f8c77dbcabaa219e89f39c8aec755407bd28272cc8caefa6b04d1`.

Committed records: [real panel, maps and budget](verification/real-data.json),
[JUnit results](verification/pytest.xml),
[offline package verification](verification/package.json), and
[staged packaged preflight log](verification/packaged-preflight.log), and
[current staged budget](verification/staged-budget.json), and
[GPU qualification](verification/gpu-qualification.json).

The externally verified mirror for the corrected active run is
`/workspace/sol-nll-results/staged-20261008-stable`. Run identity:
`970f3fca96f17d9747edec3749f0c4a345f4fc5ca2727d3c125dfacf9392884a`.
The original session deadline remains 13:53:43 UTC. Kernel installation and
numerical repairs did not reset that cap. The longest-context capacity gate passed at 74.15 GiB peak allocated and
75.30 GiB peak reserved. Its cold PLE reads took 432.45 seconds; the 95 GiB
CPU-side table is being warmed in parallel. The complete 30-pair NLL gate
remains pending; no expert count is selected yet.

## Repository dataset

The exact 30 full requests are also published in the repository's existing
private DVC remote as
[`data/sol-nll-fold0-30`](../../data/sol-nll-fold0-30/README.md).
The 5.7 MiB `requests.jsonl` payload has a Git-tracked DVC pointer, with a
Git-tracked index, source hashes, selection provenance and verification script.
Fetch it using `dvc pull data/sol-nll-fold0-30/requests.jsonl.dvc`.
This is the same frozen panel; exporting it did not resample or modify request
objects. Processor/maps remain in the complete offline package.
