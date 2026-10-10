# Current handoff — 2026-10-10

## Repository and layout

- Architecture PR https://github.com/LamDang/arc-agi-3-solution-da-fr/pull/24
  merged at `10e1168`. Production training work: `codex/trajectory-training`.
- Active implementation: `../architecture/`; entry points `train.py`, `test.py`,
  `benchmark.py` share explicit configuration and component assembly.
- This folder retains trajectory preparation/encoding and their tests.
- Retired experiment code/configs/tests: `../experiments/legacy-training/`.
  This is source-only history, not an executable supported runner. Old results,
  artifact pointers and chronological handoffs were removed.
- The consolidated result table is `../architecture/v0.md`. Detailed reports
  and the train-only expert map are indexed by `../evidence-data.dvc`; restore
  original report paths with `../manage_evidence.py restore`.

## Final architecture contract

- BF16 model activation ports; FP32 statistics/internal work where native needs
  them. LoRA parameters and gradients remain FP32 on GPU; matmuls use BF16
  autocast. All routed experts have LoRA: 74,472 adapter tensors total.
- Frozen packed expert buffers remain in CPU RAM, in one shared pinned slab
  (~32 GiB allocation), with at most two frozen layers prefetched to GPU.
- Full frozen PLE table stays on original disk/NFS. A bounded worker prepares
  only required rows and retains the sample payload through recomputation.
- Final cumulative configuration: `configs/opt10.json`: target-only CCE exact
  (gradient filtering disabled), direct dense attention bias, expert chunks,
  QSA query windows with full K/V, hyperconnection windows preserving native
  projection row shapes, and PLE windows with a nine-position convolution halo.
- Default chunk size: 8,192. Full checkpoint residual inputs/outputs remain.
- Liger FLCE/RMSNorm/SwiGLU and BF16 LoRA masters are disabled in the accepted
  stack. Their original trial sources are in the legacy folder.

## Numerical qualification and limits

- Legacy refactor replay matched the scalar loss and all 744 gradients bitwise.
  Reports are retained. Historical raw baseline gradients were retired; its
  standalone verifier now requires an explicitly supplied old baseline.
- Current all-expert reference: attempt `20261009221135477-25bf58fe`, execution
  `3461d6a`; 16,249 tokens / 651 targets / 7 images; loss 0.6244627833366394.
  Its raw reference gradients and exact initial state remain under architecture
  results. That historical capture used CPU routed-expert LoRA masters.
- Opt1 matched all 74,472 reference gradients bitwise. Opt2/Opt3 differences
  were 1.751894% / 1.784223% relative L2 against the native reference.
- Cumulative Opt7/8/9/10 passed the user-selected 2% paired unchunked-control
  gate: 1.783424% / 1.801111% / 1.770794% / 1.780845%.
- Final GPU placement/shared-allocation changes did not receive a new numerical
  qualification: the user explicitly waived that for allocation changes.
- FLA `reference-v0` scan settings are test/capacity-only. Real training rejects
  that profile and clears inherited overrides to use native autotuning.
- Test initialization uses exactly reproducible nonzero A/B, seed 20261009.
  Production must start from clean random A / zero B, without validation exposure.
- Keep raw full-sample gradients only for the current reference; candidates
  retain comparisons/statistics. Qualification/capacity captures do not update
  the optimizer. An implementation gate failure requires Astra review.

## Completed capacity captures

GPU: RTX PRO 6000 Blackwell ~96 GiB; host cgroup limit 175 GiB, no swap.
GPU figures are peak allocated memory; RAM is sampled process-tree PSS.

| Tokens | Profile | GPU GiB | RAM GiB | First F / B seconds |
| --- | --- | ---: | ---: | ---: |
| 32,000 | Historical CPU LoRA | 22.16 | 121.18 | 218 / 343 |
| 64,000 | Historical CPU LoRA | 32.96 | 152.04 | 412 / 739 |
| 96,000 | Corrected GPU LoRA | 57.51 | 130.80 | 583 / 1,281 |
| 120,000 | Corrected GPU LoRA | 65.60 | 153.79 | 772 / 1,795 |

- Original 96K forward failed with strong host-memory exhaustion evidence.
  Shared pinned storage/direct transfers/GPU masters/streamed comparison reduced
  post-load PSS by ~52 GiB. Failure and diagnostic evidence remain with the runs.
- Corrected 96K: `20261010084759860-6c853f57`, two F/B passes; losses
  0.8578786253929138 / 0.8578787446022034. All gradients finite/nonzero.
- 120K: `20261010100622729-a29599a0`, one F/B pass; loss 0.821954071521759.
  All gradients finite, 74,466 nonzero. No repeat-gradient comparison.
- Corrected 96K passed before 120K dispatch. OOM counters did not increase.
- These exclude optimizer state/update, loading and PLE preparation. 32K/64K
  use a different placement profile. No 130K or production learning claim.

## Evidence, server and continuation

- Architecture run archives, reference tensors, benchmark fixtures, source/input
  hashes and reviewers are retained. Old experiment artifacts were removed.
- DVC data is local-only at user direction; AWS login expired. Do not retry an
  upload without renewed direction. A fresh clone cannot fetch the new bundle
  from the remote yet. Git PR updates are authorized; do not merge automatically.
- All 30 architecture attempt caches were verified before the original cache
  cleanup. Subsequent report storage changes do not requalify numerical results.
- Kaggle notebook 356919557 Jupyter was stopped after all captures were verified:
  PID 12 acknowledged SIGTERM; subsequent API HTTP 404. No job is active.
  A new connection is required for further GPU work. Use Jupyter HTTP/WebSocket,
  not computer use. Never print or commit a private connection URL.
- Pinned capture runtime: Torch 2.11.0+cu128, Transformers 5.18.0, PEFT 0.20.0,
  AutoRound 0.15.0, FLA 0.5.2, causal-conv1d 1.7.0. Correct model selection and
  package/source hashes are pinned by configs, model code and saved evidence.
- Production utilities: `architecture/TRAINING.md`, `architecture/training/`
  and `configs/train-trajectories.json`. Latest dataset definition imported from
  `ce2ba09`: 58 compaction trajectories, all 1,334 targets, max 129,169 tokens.
  Fold 0 has 11 trajectories/277,426 targets; training has 47/1,122,317.
  At most five epochs, AdamW 1e-4 with betas (0.9, 0.95), 49,152-target
  accumulation with epoch-tail flush,
  token-weighted validation, TensorBoard, acknowledged sharded DVC recovery.
  Local CPU/transport/DVC utility tests qualify correctness, not GPU learning.
- Remaining: actual dataset outputs/processor parity, live API and optimizer-
  inclusive 130K capacity, then learning/validation. Kernel remains stopped.
  Desktop checkpoint launch needs 46 GiB free; only ~15 GiB was available.
  No production run, remote upload, or artifact deletion was performed.
