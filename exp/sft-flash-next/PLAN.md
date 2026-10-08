# Sol → Flash-Next SFT: single-GPU training and recovery design

Status: design, 2026-10-08. No training job has been launched and no training
memory or throughput claim below is a GPU measurement. This design targets
remote `main` **37fadfeedfd54d2129db4525e4167596cc7337b6**, PR #19 (SFT dataset
merge), inspected separately because the working checkout was still at
17d40d8. Implement on that newer revision or a descendant.

## Recommended order: evaluate first, then choose the student

**First compare untrained 512, 448, 384, 320 and 256-expert configurations
using sampled final-reply NLL on fold 0.** The current, budget-limited protocol
is [NLL_EVAL.md](NLL_EVAL.md): 30 shared requests stratified by game and context
length, per-token diagnostics, one resident model with expert masks, optional
small reserve, and resumable results under a 120-minute session cap. There is
no gameplay or SFT pilot prerequisite for this choice. Do not preselect 384.

That protocol supersedes the earlier full-fold/gameplay selection proposal.
[RESEARCH.md](RESEARCH.md) retains framework and hyperparameter evidence for
subsequent training. Training qualification/search below is deferred work,
not part of the current NLL evaluation budget.

Use the other four folds for training, preserve the merged dataset's complete
request context and images, and retain final-reply-only loss. Training runs
through Kaggle's interactive Jupyter server or a Colab notebook. A candidate's
inference fit is followed by a separate worst-case training-memory test.

**Axolotl is the first framework to qualify**, using its exact-model NF4 QLoRA
path and repository extensions for data masks, long-context execution and
recovery. A custom Transformers/PEFT/Accelerate loop and W4A16-LoRA remain
fallbacks. None of these paths is yet qualified on our long multimodal requests.

For later training, researched LR candidates are **5e-5 / 1e-4 / 2e-4**;
provisional center is 1e-4, dense rank 16, alpha 32, dropout 0. Expert-adapter
coverage is a further candidate. These are deferred choices, not a required
multi-trial GPU search before the sampled NLL decision.

Here “Next” means this repository's multimodal **Qwen3.8-Flash-Next /
`qwen4_exp`**, not Qwen3-Next-80B-A3B. “GTX 6000” is interpreted as the
repository's **RTX PRO 6000 Blackwell 96 GB**. A Quadro RTX 6000 24 GB or RTX
6000 Ada 48 GB needs a different capacity plan. Inspect the actual GPU and
memory before downloading or loading weights; Colab must actually allocate
an 80 GB A100, not a 40 GB A100.

## 1. Evidence and immutable inputs

The authoritative inputs on the inspected main revision are:

| Input | Use |
|---|---|
| `data/sft-gpt61sol-features-25games/train.jsonl` | 1,334 request-plus-reply samples, all 25 public games |
| Its `index.json` | Byte offsets, lengths, game, request index, level, approximate token lengths, image count |
| Its `meta.json`, `dvc.yaml`, `dvc.lock` | Source provenance and reproducible data preparation |
| `data/game_folds/folds.json` | Existing game-disjoint fold definition |
| `ARC3-Inference/experiments/sft-format/README.md` | Qwen template, tool-call and loss-mask contract |
| `models/`, `exp/reap-flash-next/` | Pruning recipes, architecture, forward replay and hardware evidence |

The dataset README reports median context ≈51K tokens, maximum ≈108K,
median target ≈280, maximum target ≈4.3K, and 37,677 image occurrences.
The data file is 332,843,217 bytes; DVC MD5 is
`b1b7eaed4fe609748d1980751354b9f2`. Index MD5 is
`370ec02490f6567c91bc6ef5140ac172`. The pinned tokenizer/template revision is
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
These are checked-in provenance/statistics, not a fresh inspection of the
DVC payload in this design session.

Acquire data on an internet-enabled preparation machine, from the newer main:

```bash
cd data/sft-gpt61sol-features-25games
dvc pull train.jsonl meta.json index.json
```

Do not rebuild the dataset on every GPU session. Bundle the three artifacts,
fold definition, tokenizer, template, full vision processor configuration,
training code and platform wheelhouse as immutable inputs. Use the dataset's
existing `dvc repro` only when changing conversion. Compute SHA256 for the
bundle and every required model/processor file, in addition to DVC hashes.

### Split before processing or calibration

Default `validation_fold=0`:

| Partition | Games |
|---|---|
| Validation | sk48, sp80, tn36, cd82, ar25 |
| Training | bp35, cn04, su15, tu93, sb26, lf52, ka59, dc22, r11l, ft09, wa30, sc25, m0r0, re86, lp85, g50t, s5i5, ls20, vc33, tr87 |

Use `folds[*].game_ids` for exact joins with `index[*].game`; do not split
individual requests randomly. Every level, image, note, retry and future
trajectory of a game follows that game's assignment. Assert 25 known game
IDs, 20/5 disjoint games, and complete index coverage. Fail on unknown IDs.
Persist the row IDs and `(game, request_index)` lists in `split.json`.
The number of requests is not necessarily an 80/20 split: games differ in
length. Compute the actual sample and token totals from the index.

**Pruning leakage:** the existing `models/flash-next-reap-*/keep.json` maps
were selected on all 25 games. First recover the saved per-game statistics
and aggregate only training-game runs on CPU, verifying their separation and
provenance, then produce `keep.fold0.<N>.json`. Do not add a new GPU calibration
run to the small NLL budget. NLL_EVAL specifies the explicitly exposed fallback
if the separated statistics are unavailable.
Use training-only data for any new quantization calibration too. Preserve
expert order, slice router rows, renumber experts consistently and keep
top-10 routing. Apply the same map to untrained and trained comparisons.
The old maps can be used for a capacity smoke test, but label their scores
as validation-exposed; they are not a clean held-out model-selection result.
Upstream pretraining and the supplied quantized checkpoint's calibration
remain outside this split's guarantees.

Fold 0 is a validation set once used for LR or checkpoint selection, not a
final untouched test. Report game-level results and later repeat the frozen
recipe across the other folds if quota permits. Twenty-five public games
alone cannot establish performance on unseen private games.

### Stage zero: untrained validation must precede SFT

Use [NLL_EVAL.md](NLL_EVAL.md) as the authoritative stage-zero specification.
Prepare the panel and exact token budget on CPU, then run a paired, final-reply
NLL comparison of all five counts. Preserve complete selected contexts and
images, score every target token, and report thinking/code/format breakdowns
by game, sampled level and request length. Choose using the declared NLL and
memory tradeoff; keep sampling uncertainty and calibration exposure explicit.

The core is 30 requests, two per game/context tertile; quota boundaries
pause and resume this same panel rather than reducing the requested count. A targeted reserve of up to ten requests is optional for
full plus at most two finalists. Full-fold scoring, game episodes and training
pilots are not selection prerequisites. The protocol includes a hard session
cap, exact job identities and externally persisted per-request results.

Inference fit does not establish training fit. If later qualification changes
W4A16 to NF4, retain the W4A16 comparison and baseline the actual selected
training base before optimizer updates.

## 2. Data loading, images and the exact loss mask

Use a map-style PyTorch dataset over the small JSON index. In each worker,
open a separate binary file handle and `seek(offset); read(length)` for each
row. Cross-check the decoded game/request IDs. Avoid loading all JSON/base64
data as Python objects or sharing a mutable seek cursor across workers.
Start with 0 workers; use 1–2 and bounded prefetch only after measuring host
RAM. Microbatch size is one request, so padding is minimal.

Process one sample as follows:

1. Preserve the merged dataset's `messages`, `tools` and template kwargs.
   Earlier assistant turns are context only. The converter already maps the
   teacher's stated `reasoning` and `description` to `reasoning_content`, and
   makes `python` code-only. Do not supervise the unavailable private teacher
   reasoning, encrypted metadata, or its billed reasoning-token count.
2. Render `messages[:-1]` with `add_generation_prompt=True`, and the full
   sample without a generation prompt. Use the pinned template, tools and
   `preserve_thinking=True`. The first string must prefix the second.
   Preserve other logged kwargs, including reasoning effort.
3. Decode base64 PNGs on demand and adapt their representation to the
   processor's accepted image input without changing text or image order.
   Use the real model processor to build `input_ids`, image tensors/grids,
   attention mask and multimodal positions. Keep 640×640 board resolution;
   the index's 400 tokens/image is a bucketing estimate, not the processor.
4. Process prompt and full sample with identical images/settings and verify
   **token-prefix equality after image expansion**, not just string-prefix
   equality. Fail closed if tokenization across the boundary changes it.
   Verify image placeholder counts, grid metadata and positional inputs.
5. Set labels to `-100` everywhere before the final completion, on padding
   and on image tokens. Supervise final thinking text, closing think marker,
   code/tool delimiters, final text when present, and the final end-of-turn
   token. The generation prompt's opening `<think>` remains masked.
   Check the causal shift so the last prompt token predicts the first target.
6. Require a nonempty target and valid structured tool arguments (a mapping,
   not a JSON string). Keep the expected terminal text-only sk48 turn and
   comments-only handover notes. Preserve legitimate failed probes and UNDO
   steps in these successful trajectories; do not filter them just for being
   exploratory. Log any malformed sample and stop preparation until resolved.

TRL's blanket `assistant_only_loss` would supervise history too, and the
official template lacks its generation tags. Use explicit labels with a
custom collator/loss. Do not ask TRL to render or truncate again. Keep
cross-sample packing disabled for the first implementation: multimodal
positions, recurrent states and attention must not flow between examples.

Deduplicate identical PNG storage by content hash if useful. Cache frozen
vision outputs only after proving that all required visual features and
positions are reproduced; cache keys include model/processor hashes, image
bytes, resolution and dtype. Never reuse language-prefix activations across
optimizer updates: adapters change the prefix representation. Do not merge
overlapping request histories unless a later optimization proves token and
loss ownership equivalence across trimming/compaction.

Write `processed-index.json` after processor validation. It records exact
prompt/target lengths, sample hash, image identities and any rejection.
Bucket by these lengths in deterministically shuffled batches; shuffle bucket
order too, so every epoch does not simply run easy/short examples first.
Default is one visit per training row per epoch, without oversampling long
games. Record the materialized epoch order for exact recovery.

## 3. Framework, quantization and model support

Start with **Axolotl**, inspected at commit
`0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a`, over PyTorch/Transformers/PEFT.
Its dedicated Flash-Next vision QLoRA recipe provides a closer starting point
than an entirely custom trainer. Implement the offset loader, exact labels,
objective and durable/partial-accumulation checkpoint contract as focused
extensions. If those or exact long-context execution cannot fit its training
lifecycle cleanly, reuse the qualified model backend in a small Accelerate
loop. SGLang remains the rollout server.

Qualify and lock the complete dependency set per sm_120 and sm_80; the repo's
Transformers 5.18.0 / flash-linear-attention 0.5.2 forward-replay environment
is not automatically the Axolotl training environment. Do not mix untested
versions or use unpinned upgrades on resume. See the cited framework matrix
and source audit in [RESEARCH.md](RESEARCH.md#2-framework-comparison).

The published exact-model Axolotl examples are short-context B300 runs, not
single-80/96-GB long-context evidence. Its current QSA indexer still creates
quadratic score/mask tensors; bounded query processing and parity checks are
needed before a full-transcript capacity claim. ms-swift's exact-model example
uses eight GPUs; Unsloth's reviewed Qwen3.8 training guide concerns a different
27B architecture. Neither replaces exact-workload qualification here.

There are two explicitly different quantized adapter paths:

| Path | Requirements and decision |
|---|---|
| **Canonical NF4 QLoRA (first qualification path)** | Start from the matching unquantized source, prune, then use Axolotl's fused-expert quantization with bitsandbytes NF4 + double quantization and BF16 compute. Audit actual quantized tensors, PLE RAM and loading peaks; ordinary Linear replacement is insufficient. |
| **Repository W4A16 + LoRA (fallback)** | Preserve Intel AutoRound/GPTQ packed experts and BF16 non-quantized components. Qualify an autograd-capable backend from the repo's replay model if direct framework support cannot be established. Label it W4A16-LoRA, not NF4 QLoRA. |

Do not pass the current AutoRound checkpoint through `load_in_4bit=True` and
assume that it becomes NF4. Do not dequantize and requantize it silently.
If canonical NF4 is mandatory, the BF16 source and its proven quantization
backend are prerequisites; select the W4A16 path only with an explicit run
configuration identifying that variant.

`exp/reap-flash-next/reap_model.py` is a useful reference, **not a ready
trainer**. Its shared dequantized-weight workspace can be overwritten by
later layers before backward consumes it. An autograd-safe quantized matmul
must save the packed weight reference and regenerate the correct frozen
weight in backward (including `dX`), or use checkpointed layer-local storage
with proven lifetimes. Freezing experts does not eliminate the gradient
through their inputs. Audit grouped GEMM, indexer attention, linear attention,
convolution and recurrent state updates for valid backward/recomputation.
Keep the model's sparse attention semantics; do not replace the sparse
indexer with generic dense FlashAttention just to make a trainer load.

Keep the enormous frozen PLE n-gram table memory-mapped on host storage as
in the replay model (documented ≈51B parameters), with bounded pinned staging
for selected rows. Budget disk and host page cache as well as VRAM. Do not
cast the entire table to FP32 via a generic k-bit preparation utility.
Remove MTP/drafter execution from the SFT model path and record which tensors
are intentionally unused. Keep the vision tower and merger frozen initially.

### Adapter configuration

- Enumerate actual module/tensor names from the loaded architecture, then
  save the resolved allowlist and trainable parameter count. Do not apply
  `all-linear` blindly or assume every layer has Qwen dense `q_proj/v_proj`.
- Initial targets: existing language attention projections, linear-attention
  input/output projections, and shared-expert projections. Rank **16**, alpha
  **32**, dropout **0**, no bias training. Keep adapters/optimizer state in
  FP32 where supported, with BF16 matmul/autocast; measure the actual footprint.
- Freeze routers throughout the initial trials. Keep routed experts frozen
  in the modules-only control, then compare adding rank-1 or rank-2 adapters
  to retained routed experts. Parameter-targeted adapters require dropout 0
  in the reviewed framework. Rank 8 for dense modules is a capacity fallback;
  rank 32 is a later capacity/quality comparison. Fix alpha/scaling during the
  rank trial. Use the parameter-count estimates and bounded trial design in
  RESEARCH: rank 16 on every expert can add billions of trainable parameters.
- For full vs pruned comparison, retain the same adapter policy, training
  data, seed, objective and processed-token budget. Adapters cannot move
  between different expert maps or quantization bases by merely renaming them.

## 4. Full-context execution and hardware admission

**Sequence length and execution chunk size are different.** Keep all context
in the merged samples. Admit exact prompt + target lengths up to **114,688**
initially, expanding the configured ceiling if the processor audit requires
it and capacity is qualified. Check the model's positional limits as well.
No silent default 2K/4K/8K truncation, dropped images, or target truncation.

First try native autograd with non-reentrant layer activation checkpointing,
CPU activation offload and the model's supported efficient kernels. First
bound QSA query/block-score temporaries; offloading retained activations does
not remove oversized per-operation tensors. Disable
the ordinary inference cache. If full requests do not fit, the production
fallback is **exact chunked recomputation**, initially chunks of 4,096 on
Kaggle and 2,048 on A100; these retain the entire sequence's computation.

An exact chunked backend must preserve and differentiate every cross-chunk
dependency: attention K/V, compressed keys/indexer state, linear-attention
state, convolution history, positions and any PLE/hybrid state. During
reverse recomputation it propagates cotangents from future chunks into the
states and earlier K/V that produced them; it restores RNG for dropout and
does not mutate checkpointed states in place. Top-k choices remain discrete
as in the native model. State offload and restoration must preserve dtype.
Ordinary `past_key_values` with detached states is **truncated backpropagation**,
not this algorithm. It requires a distinct experiment label if explored.

Compute output-head logits only for hidden states predicting supervised
tokens, including the hidden state immediately before the first target.
Chunk that selected-token loss (e.g. 256–512 positions) and accumulate CE
in FP32. A full `[108K, 248320]` logits tensor is an avoidable memory problem;
masking labels after constructing it does not solve it. The frozen head still
needs to pass gradients to selected hidden states.

| GPU / candidate | Existing W4A16 serving weights¹ | First capacity probe | Recommendation |
|---|---:|---|---|
| RTX PRO 6000 96 GB / 512 | ≈69.9 GB | batch 1; 2K execution chunks + offload | Full reference; training feasibility pending |
| RTX PRO 6000 96 GB / 448 | ≈62.5 GB | same | Evaluate before choosing |
| RTX PRO 6000 96 GB / 384 | ≈55.1 GB | batch 1; 4K chunks + offload | Evaluate before choosing |
| A100 80 GB / 320 | ≈47.7 GB | batch 1; 2K chunks + offload | Evaluate before choosing |
| A100 80 GB / 256 | ≈40.4 GB | same | Evaluate before choosing; capacity-oriented candidate |
| A100 80 GB / 384, 448 or 512 | ≈55.1 / 62.5 / 69.9 GB | only after measured headroom | No training fit claim |

¹ Repo serving measurements/estimates, not training allocations and not NF4
estimates; host PLE storage is additional. Treat GB/GiB consistently when
comparing with device memory. Require peak reserved VRAM below 90% of actual
capacity, bounded host RSS/page cache and enough disk for the source model,
pruned rewritten shards, caches and two recovery bundles. The earlier
131K training-memory figures in `exp/reap-flash-next/PLAN.md` are unmeasured
projections, not admission evidence. Pruning leaves top-10 active compute
unchanged, so it mainly saves storage, not proportionate training time.

Qualification: 2K smoke → 8K → 32K → median real request → longest combined
prompt/target → most-image request. Include backward, optimizer state
allocation, checkpoint write, reload and one resumed update. Record peak
allocated/reserved VRAM, RSS, disk, load time and input/target tokens/sec.
Run a sustained 20-update pilot only after these pass. If a backend cannot
support full context, the result is blocked pending backend work; an 8K
subset can serve as a clearly labeled smoke test, not the requested full run.

## 5. Objective, learning rate and training schedule

Default objective: mean final-reply negative log likelihood per request,
then mean across the accumulated requests. This prevents long tool outputs
from receiving weight merely because they contain more tokens. All target
token categories have weight 1 initially. Report token-weighted NLL as well.

| Setting | Initial value |
|---|---|
| Microbatch | 1 complete request |
| Gradient accumulation | 4 requests, effective batch 4 on one GPU |
| Optimizer | AdamW, adapters only; betas (0.9, 0.999), eps 1e-8 |
| Peak LR | **1e-4 provisional**; compare 5e-5 / 1e-4 / 2e-4 |
| Weight decay / clipping | 0 / global trainable grad norm 1.0; decay 0.01 only as a later ablation |
| Scheduler | Linear warmup for max(5, ceil(0.05 × total updates)), capped by total updates; cosine decay to **10% of peak LR** |
| First production budget | **1 epoch**; extension to 2–3 only as a new recorded schedule decision |
| Seed / precision | 42 / BF16 compute; FP32 loss reduction |
| Validation | Reuse the frozen NLL core at baseline and every 25 updates; broader validation only with a separate budget |

For N training requests and accumulation A, one epoch has `ceil(N/A)`
optimizer updates. Normalize the final partial batch by its actual count.
Do not step LR per microbatch, execution chunk, notebook session or restored
checkpoint. Pilot gradients use training rows only, after untrained model
selection. If budget later permits LR trials, compare matching initial weights
and sample order; use 2e-5 only as a stability fallback if the initial range
fails. RESEARCH's sequential search is an optional future experiment, not a
requirement for the current low-budget path. Record each trial's provenance.

Clip after accumulation and any gradient unscaling. Stop on nonfinite loss or
gradients rather than silently skipping many updates. Prefer ordinary AdamW
for the small initial adapter set; add a paged/8-bit optimizer only if its
measured memory reduction justifies an additional compatibility dependency.

Freeze the global schedule before launch. Quota breaks pause it; they do not
restart warmup or redefine an epoch. Extending an exhausted one-epoch cosine
schedule must be a named continuation with an explicit new schedule, not an
invisible change to `num_train_epochs` when restoring state.

### Estimate time from this dataset

From the processed training index calculate `T = sum(prompt + target tokens)`
and `Y = sum(target tokens)`. The long context is reprocessed per request;
do not use the teacher's unique output tokens or an unrelated run's token
count to estimate training cost. Measure training throughput in length/image
buckets and estimate `sum(T_bucket / measured_rate_bucket)` plus validation,
loading, compilation and checkpoint transfer.

For scale only: 50M processed training tokens at 200 tokens/sec take ≈69.4
GPU hours before those extras. This is an illustration, not a prediction.
Report `ceil(total planned hours / measured usable session hours)` sessions
and the account's actual available quota; service limits can change. Do not
reuse SGLang generation throughput as training throughput.

## 6. Validation, rollout monitoring and reporting

Reuse the frozen probability-sampled core in [NLL_EVAL.md](NLL_EVAL.md)
for training trend monitoring with its declared weights. Keep targeted reserve
results separate. Expand to full-fold evaluation only with an explicit later
time budget, preserving the same processor, contexts and target mask. Resume
interrupted validation by row ID and exact evaluated checkpoint.

Report the following to append-only JSONL plus TensorBoard; W&B is optional
on connected Colab and offline-file-only on Kaggle:

| Cadence | Metrics |
|---|---|
| Every microbatch | game/row ID, request loss, prompt and target tokens, images, forward/backward time, peak VRAM, cumulative samples/tokens |
| Every optimizer update | LR, grad norm, mean train loss, tokens/sec, time/update, epoch fraction |
| Every evaluation | token NLL/perplexity, mean request NLL, **macro game NLL**, per-game/level/length-bin loss, reasoning/code/delimiter loss where spans are validated |
| Heartbeat every 60 s | current phase/row, last completed microbatch/update, elapsed time, quota deadline, latest local checkpoint and latest externally verified checkpoint, transfer lag, RSS/disk |

Compute perplexity as `exp(total target CE / target count)`; do not average
per-batch perplexities. Use NLL_EVAL's macro game mean-request NLL for the
sampled selection metric, so sk48 does not dominate through request count.
Save `latest` for recovery and `best` by the declared metric separately. Retain
the small panel's uncertainty; repeated use does not turn it into a test set.

Game performance remains a separate optional downstream question; it is not
required to choose the imitation base. If later budget allows rollouts, run
them in a separate process after releasing the trainer's GPU and compare the
adapter with its own matched untrained base. The controls below apply only
to that separately budgeted experiment.

Keep the harness, game versions, action/output/wall-clock caps, template,
`preserve_thinking`, compaction/hint settings and seeds fixed. The student
uses **`ARC3_PYTHON_RATIONALE=0`** and code-only Python calls. Inspect actual
served prompts/reasoning-history keys to verify they match training. Report
mean competition score, per-game score, wins, levels, actions, output tokens,
wall time, invalid tool calls, execution errors and no-progress loops.
Offline tool-call syntax checks are useful but are not a replacement for
playing games. Run generated code through the existing game sandbox.

Adapters may not be supported for this architecture in the serving backend.
Prove adapter loading and zero-adapter parity before scheduling rollouts.
If a merged export is needed, merge with the matching unquantized base and
expert map, then re-quantize with training-only calibration; do not add a
LoRA delta directly to packed int4 bytes. Evaluate the final exported model
again because quantization changes it. Always retain the original recovery
checkpoint and adapter separately from a deployment export.

Write `report.json` and a small HTML report with loss/LR over global update,
loss versus context length, per-game rollout comparisons, throughput/VRAM,
and session/recovery timelines. Every metric includes run ID, checkpoint ID,
code/data/model/expert-map hashes and hardware profile. Record the small
number of independent held-out games rather than presenting thousands of
correlated requests as independent generalization evidence.

## 7. Recovery protocol: hardware failure and quota

A saved adapter alone is a warm start, **not** a resumable training checkpoint.
Each checkpoint contains:

- adapter weights/config/resolved target names and base/expert-map hashes;
- optimizer and scheduler state, global optimizer step, epoch and next row;
- Python, NumPy, CPU and CUDA RNG states, sampler epoch permutation/cursor;
- total processed/supervised tokens, best metric and early-stop state;
- partial accumulation count and **adapter gradients**, if saved between
  optimizer steps; precision/scaler state when applicable;
- immutable resolved config, dependency lock, split/processed-index hashes,
  processor/template hashes and checkpoint file checksums.

Use explicit microbatch accounting: accumulate the **sum** of per-request
mean-loss gradients, then divide gradients by the actual accumulation count
before clipping/stepping. A partial checkpoint stores the unnormalized sums
and count. Increment the next-row cursor only after backward finishes. On
restore, reload gradients and continue the remaining microbatches without
`zero_grad`; on an optimizer-boundary checkpoint restore with empty gradients.
This avoids losing hours when four long requests span a save interval.
Accelerate's ordinary save alone does not persist all these custom states.

Checkpoint at a completed microbatch when **10 minutes** have elapsed since
the last save, at evaluation boundaries, on a graceful stop request, and
before quota shutdown. The maximum save gap is 10 minutes plus one request's
forward/backward duration, not a hard ten-minute guarantee. Persist heartbeat
and logs more frequently. Never checkpoint halfway through a tensor update.

Write into `checkpoint-<step>-<micro>.partial`, flush/fsync all files, write
the checksum manifest and `COMPLETE.json` last, then atomically rename on
local storage. Retain two complete local recovery checkpoints plus the best
adapter. Copy immutable bundles to durable storage and verify size/checksum
there before updating `latest-complete.json`. Ignore partial/corrupt bundles
on restore and fall back to the preceding complete one. Do not garbage-collect
the previous durable checkpoint until its replacement is verified.

On startup, verify all immutable inputs, restore the latest compatible
complete checkpoint, reopen loaders, restore sampler and RNG, and continue.
Do not blindly trust the highest numbered directory. An OOM retries from the
last completed microbatch with a qualified smaller execution chunk/offload
profile; never silently reduce sequence coverage or skip the offending row.
Keep one writer per run with a run lock; deduplicate log records by checkpoint,
global update and microbatch so replays after a crash are visible.

Resume on another GPU only with the same base, expert map, adapters, data,
global batch, objective and scheduler. Changing 384 → 320 experts starts a
new experiment. Changing GPU/kernel builds may change numerics: record it and
expect stateful continuation, not bitwise-identical trajectories. Revalidate
the new platform's kernels before spending quota on a long continuation.

## 8. Kaggle Jupyter server and Colab operations

### Kaggle: interactive Jupyter is the execution host

The repo documents the RTX PRO 6000 allocation with the ARC competition
attached and **internet disabled**. Recheck current eligibility and quota in
the actual session. Package code, wheels, data, processor, model and the most
recent recovery bundle as Kaggle inputs before starting. Use a separate
training wheelhouse, not the SGLang inference wheelhouse indiscriminately.
Verify sm_120 backward support; the repo used a convolution fallback because
a suitable wheel was missing. Do not assume sm_80 wheels also support sm_120.

The notebook should have seven short cells, delegating logic to versioned
Python modules:

1. Inspect GPU, RAM, disk, offline input hashes and remaining session budget.
2. Install pinned offline training wheels and import-test the backend.
3. Set run ID, fold, model/map, profile, data root and session deadline.
4. Restore and verify the recovery bundle copied from read-only inputs into
   `/kaggle/working/sft/<run-id>`; perform compatibility checks.
5. Run the qualification/pilot or start the resumable worker through the
   **Jupyter kernel/server**. Keep a worker lock and log its PID. A notebook
   monitor can poll its heartbeat; reconnecting must not start a second worker.
6. Show live metrics and provide a graceful-stop control (a stop file/event
   consumed at microbatch boundaries). Leave the GPU computation in the
   Kaggle session; a local client only supervises and retrieves artifacts.
7. Finalize a recovery archive and manifest, verify retrieval, then stop.

A worker subprocess can survive a browser disconnect, but it cannot survive
the Kaggle VM/session being terminated. `/kaggle/working` is ephemeral. Do
not treat a notebook save, `nohup`, an output zip still on that disk, or
internet-dependent S3/W&B uploads as durable recovery in an offline session.

**Required durability mechanism:** an external client connected to the
authorized Kaggle Jupyter/file-download session polls for completed immutable
archives and downloads them to persistent local storage, then mirrors them to
the chosen object store outside Kaggle. Where the platform exposes an
authenticated Jupyter contents/files interface, use it; otherwise use its
supported output-download flow. Do not assume an unauthenticated public
Jupyter endpoint or an undocumented Kaggle API. Keep connection credentials
outside notebooks and logs. The client verifies SHA256 and records a durable
acknowledgement that the notebook can display. Test this transfer path before
admitting an unattended long run.

If automatic retrieval is unavailable, the supported fallback is frequent
manual Output-panel downloads and reattachment as a private Kaggle dataset.
That still resumes across quota windows, but unattended hardware-failure loss
is bounded only by the last external download. State that reduced guarantee
in the report; do not advertise fully automatic fault recovery from local
checkpoints alone. Preparing a resumable run must not depend on receiving
SIGTERM before a quota kill.

At a configured deadline, stop early enough to finish the slowest measured
microbatch plus serialization and transfer, with a further 15-minute margin.
Use the actual session limit, not a permanently hard-coded 12 hours. On the
next eligible session, attach the last externally verified bundle and rerun
the notebook's restore/start cells. Session creation/accelerator allocation
may remain manual; do not attempt to evade platform quota or idle policies.

### Colab A100 80 GB

Use the same worker and input hashes, a separate sm_80 dependency lock and
local `/content/sft/<run-id>` for active I/O. Read secrets through Colab's
secret mechanism. Mirror completed checkpoint bundles to durable object
storage; mounted Drive is an alternative, but copy a completed local archive
and verify it instead of performing live optimizer writes on the mount.
On reconnect, download the latest complete bundle before loading training.
Run the same deadline monitor because allocation/session duration is not
guaranteed. Drive/W&B connectivity failure must surface as growing durable
checkpoint lag. Default action is graceful pause after 30 minutes without a
durable checkpoint, at the next safe boundary, with a configurable override.

## 9. Implementation boundaries and acceptance gates

Proposed files (not implemented by this document):

```text
training/flash_next/
  prepare.py             # verify DVC inputs, split, processor audit, token budgets
  dataset.py             # offset loader, image handling, exact labels
  model.py               # quantized backend, PLE, adapters and architecture checks
  long_context.py        # qualified checkpointing/offload or exact chunked backward
  train.py               # Axolotl entry/extensions; Accelerate-loop fallback
  checkpoint.py          # atomic bundles, gradients, RNG, sampler and restore
  evaluate.py            # all-config pre-SFT comparison, then trained evaluation
  report.py              # JSONL → HTML/TensorBoard summaries
  configs/               # all five untrained candidates; selected training profiles
  locks/                 # proven sm_120 and sm_80 wheel manifests
  notebooks/             # Kaggle Jupyter and Colab launch/monitor notebooks
  collect_checkpoints.py # external authenticated download/verification client
```

Build and accept in this order:

0. **Untrained model selection:** prepare the CPU contract and paired sampled
   NLL comparison in NLL_EVAL, then choose a candidate with explicit uncertainty
   and calibration provenance. No game score or training pilot is required.
   Rebaseline later if framework qualification changes the base quantization.
1. **CPU data contract:** all rows assigned exactly once to train/valid;
   correct IDs/offsets; exact processor prefix/masks; no prompt/history/image
   labels; no empty or truncated targets; code-only tool schema; all images
   retained. Emit split sizes and token budgets. No GPU quota needed.
2. **Model/backward correctness:** zero-adapter parity with the matching
   frozen base; finite nonzero gradients in every selected module; frozen
   weights unchanged. Tiny-model gradient comparison of native and chunked
   paths across attention/recurrent boundaries, including multimodal inputs,
   losses and optimizer updates. Fix workspace aliasing before performance work.
3. **GPU capacity and 20-update pilot:** measured profile for each admitted
   GPU/model combination; full long requests and image extremes, modest loss
   decrease on a fixed train-only mini-set, memory headroom and runtime ETA.
4. **Recovery fault injection:** terminate after a microbatch during partial
   accumulation, during archive write, during transfer and during evaluation.
   Restore using only external artifacts in a fresh session. Compare the next
   row, gradients, optimizer step, LR and subsequent weights with an
   uninterrupted reference within declared numerical tolerance. A browser
   reconnect must not launch a duplicate worker.
5. **Later training experiment:** baseline and periodic sampled NLL, export/
   load validation, quality versus GPU-hours and durable recovery evidence.
   Full-fold checks and rollouts require a separate budget.

The next experiment is the untrained expert-count validation sweep. After
selection, unresolved training work is qualifying Axolotl's model path for our
full contexts (especially bounded QSA and PLE host memory), exact labels and
backward, plus proven external recovery on offline Kaggle. Data conversion
and game-fold definitions already exist. Resolve these gates before long SFT.
