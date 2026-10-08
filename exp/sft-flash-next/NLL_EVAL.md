# Staged teacher-forced transcript NLL evaluation

Current protocol, updated 2026-10-08: **512 baseline → 256 → conditional
448/384/320 scan**, on the same frozen 30-request fold-0 panel. This supersedes
the unconditional five-model sweep and the former absolute 0.05-nat selection
rule. No GPU, training, generation or gameplay has started.

## Stage order and gate

1. Score all 30 requests with the full 512-expert model. Numerical/chunk-parity
   and longest-request capacity checks run first; valid baseline smoke forwards
   count toward these 30 jobs.
2. Score all 30 requests with the training-only 256-expert map.
3. Compute the paired **primary NLL** below on all 30 requests.
   If `NLL256 <= 1.05 * NLL512`, stop and recommend 256.
   Only if `NLL256 > 1.05 * NLL512`, score 448, then 384, then 320, each on all
   30 requests. Reuse the completed baseline/256 results without new forwards.
4. After a scan, recommend the smallest evaluated configuration whose primary
   NLL is no more than 5% above the full baseline. If none of the pruned
   candidates meets the limit, retain 512.

Exactly 5% is accepted. A lower NLL also passes. The threshold is a **relative
increase in mean NLL**, not relative perplexity, an absolute 0.05 nats, or a
per-game/per-token threshold. If baseline NLL were exactly zero, only zero
candidate NLL passes. Partial panels never decide the gate. Numerical and
per-game/code diagnostics remain visible, but they do not automatically expand
a passing panel. No reserve, full-fold sweep or automatic fresh calibration.

Example: baseline primary NLL 2.00 permits 256 up to 2.10. A result of 2.11
triggers the three intermediate configurations.

## Teacher forcing and primary score

For final-reply target token t:

    nll[m,i,t] = -log P_m(y_i[t] | full_prompt_i, y_i[:t])
    L[m,i] = mean_t nll[m,i,t]
    game_nll[m,g] = sum_i(w_i * L[m,i]) / sum_i(w_i)
    primary[m] = mean_g game_nll[m,g]

The preceding reply tokens are the recorded teacher tokens. The model never
substitutes generated tokens. The primary score weights requests by their
stratum population, then gives each of the five games equal weight. Both
models use identical requests and targets. Token-weighted NLL/perplexity,
category scores and unweighted metrics are separate diagnostics.

Score **only the final teacher assistant reply**, including its emitted turn
formatting/end token. Earlier assistant messages, user/system text, tool
results and images are unscored context. The supplied generation prefix,
including an opening thinking tag, is masked. Position P is predicted by the
hidden state at P−1, including across replay chunk boundaries.

## Immutable panel and provenance

Six requests per game: sk48, sp80, tn36, cd82, ar25. Two uniformly sampled
without replacement per within-game context-length tertile, seed 20261008.
Sort eligible metadata by `(context_tokens, request_index, line)` before
splitting into three nearly equal-count strata. Population weights are
`stratum_population / 2`. Sampling precedes any model losses and never changes
because of cost or a failed request.

The exact requests are in
[`data/sol-nll-fold0-30`](../../data/sol-nll-fold0-30/README.md), with the
payload in private DVC/S3 storage. The complete prepared panel manifest is
`f75451395a05abaee253543275b901a471b98adbf7112a5b48bac5e6faf0960c`.
The protocol update does not change this panel, its targets or its maps.
Its original `counts`, `jobs`, `processed_tokens` and `estimated_minutes`
represent the **maximum five-model scan capacity**, not mandatory work.
`protocol.py` supplies the active staged policy and budget, bound to run identity.

Dataset revision: `37fadfeedfd54d2129db4525e4167596cc7337b6`.
Processor: `Qwen/Qwen3.8-Flash-Next` at
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. DVC payload hashes, fold hash,
processor/template hashes, selected IDs, inclusion probabilities and exact
lengths are recorded in the dataset provenance and preparation verification.

All 20 training-game REAP runs contribute to CPU-built, nested 48-layer maps
using summed `gate_norm` over context/generated/image categories. All five
fold-0 games are excluded. Run identity, model version and category order
are checked. Maps independently match `analyze.py`; existing validation-exposed
published maps are not used. No new GPU calibration is needed.

## Correct rendering and token labels

Preserve all messages, inline images, code-only tools and template kwargs.
The pinned processor must render the generation prompt as an exact string
and processed-token prefix of the complete transcript. Prompt/full image
processing must be identical; the final text target must match the tokenizer
suffix. Never truncate or silently drop a request.

Structured shadow spans must round-trip through the pinned template without
putting markers into model input. Each final target token receives exactly
one label: logged `thinking`, `tool_code`, `tool_format`, `assistant_text`,
`turn_format`, or explicit `boundary`. Code whitespace is executable content
and must be preserved. Category counts/sums reconstruct the overall score.
The real-data CPU preflight re-encodes all 30 requests and checks every stored
annotation, full input hash, target hash and image backend.

## Execution and durable resume

Use the full W4A16 model resident once on **RTX PRO 6000 Blackwell 96 GB**, with
adequate host RAM. Mask experts between complete replays, retaining top-10
routing and shared experts. Each request/configuration has a **fresh cache**;
never reuse a history encoded under another mask. Masking does not reduce
resident weight memory or certify physically pruned deployment speed.
A100 80 GB is not qualified for this full-resident sweep.

Default replay chunks are 8192 tokens; vocabulary-head blocks are 256 target
positions. Compute exact full-vocabulary FP32 cross-entropy and save FP32 loss
vectors with FP64 summary accumulation. No top-k approximation or retained
context-by-vocabulary logits. Short-request repeat/chunk-parity and the longest
selected request capacity check must pass before full scoring. GPU correctness,
actual throughput/VRAM and training fit remain unmeasured until then.

Completion units are atomic, checksummed `(request, expert count)` results.
An interrupted request restarts from its beginning. Stage loops skip verified
completed results. The gate is recomputed from durable matched results, so
an interrupted baseline/256 stage cannot trigger a scan. The run identity
includes the staged policy, code, packages, model, maps, processor, panel and
kernel/chunk settings. Results from an old protocol cannot silently resume.

The external Jupyter collector downloads/verifies result pairs and acknowledges
the active run. Missing/stale acknowledgment pauses work. It does not start
kernels or allocate GPUs. A worker lock prevents duplicates after reconnects.
The 120-minute session cap includes loading and admits new jobs conservatively.
Resume the same 30 requests across sessions; the cap does not deallocate Kaggle.
Stop or disable the GPU after verifying the durable mirror to avoid idle quota.

## Exact work and estimated time

The panel has 1,268,524 prompt tokens, 12,108 final-target tokens and 664
images. Expanded full requests range from 5,770 to 84,298 tokens.

| Work | Forwards | Processed tokens | Scored targets | Scoring @850 tokens/s | Scoring @500 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial 512 + 256 | 60 | 2,561,264 | 24,216 | 50.2 min | 85.4 min |
| Conditional 448 + 384 + 320 | 90 | 3,841,896 | 36,324 | 75.3 min | 128.1 min |
| Maximum total | 150 | 6,403,160 | 60,540 | 125.6 min | 213.4 min |

With 20–30 minutes per cold session, the initial comparison is nominally
**70.2–80.2 minutes** at 850 tokens/s, or **105.4–115.4 minutes** at 500 tokens/s.
The maximum scan remains approximately 2.8–3.1 hours over two nominal sessions,
or 4.6–5.1 hours over three conservative sessions. Admission/persistence
margins, repeats, startup variation and actual speed can require extra sessions;
these are estimates, not promised completion within one cap.

## Reporting and readiness

Reports record the full baseline and 256 primary NLL, relative delta, gate
threshold/status, required/skipped configurations and matched request count.
A passing 60-job panel is complete; skipped intermediate models have null
scores, not zero losses or unfinished required work. If the gate fails, full
selection requires all 150 jobs. The HTML viewer shows per-token NLL/deltas.
Per-game/code and leave-one-game-out diagnostics do not override the gate.

The 30 requests cover 15 of 35 game-level cells; 20 remain explicitly unsampled.
The sole terminal text-only source reply is not sampled. NLL selects an
imitation candidate; it does not prove gameplay quality or backward/training
capacity. See [READINESS.md](READINESS.md) and [nll/README.md](nll/README.md)
for current tests, private package and disabled notebook launch instructions.
