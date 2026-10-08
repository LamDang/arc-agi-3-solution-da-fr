# Research: framework, adapter settings and evaluation before SFT

Researched 2026-10-08. This supplements [PLAN.md](PLAN.md). Published results
below are source evidence; proposed experiments are our choices, not measured
ARC results. No GPU evaluation or fine-tuning was launched in this research.

## Decision

**Evaluate all five untrained configurations first: 512, 448, 384, 320 and
256 experts, using sampled NLL only.** The current low-GPU-budget specification
is [NLL_EVAL.md](NLL_EVAL.md): a 30-request paired core, cheap token-level
breakdowns, optional reserve and a 120-minute session cap. It supersedes the
former full-fold NLL and 25–100 gameplay episode requirements. Choose the
imitation quality/memory tradeoff from this run; no training pilot is required
for base selection. All training-framework work below is a later phase.

**Use Axolotl as the first framework to qualify**, with small repository
extensions for the dataset and recovery contract. Its exact-model support
makes it a better starting point than a wholly custom trainer. Keep a narrow
Transformers/PEFT/Accelerate implementation as the fallback if Axolotl cannot
accommodate the necessary long-context execution and recovery changes.

No reviewed framework demonstrates this exact combination out of the box:
our pruned model, 51K median / 108K maximum multimodal prompts, and one
80–96 GB GPU. Framework selection is provisional until that workload passes.

## 1. Similar work: what transfers and what does not

| Primary source | Relevant evidence | Implication for this project |
|---|---|---|
| [QLoRA, Appendix A/B](https://arxiv.org/html/2305.14314v1) | NF4, double quantization and BF16; final recipes use rank 64, alpha 16, all linear layers, constant LR; 1e-4 for 33B/65B versus 2e-4 for smaller models. Rank ablation finds little effect when adapters cover all layers. | Include 1e-4 and 2e-4 in a real LR comparison. Its dense, short-context experiments do not establish an optimal rank or memory footprint for this MoE. |
| [LoRA Without Regret, Thinking Machines](https://thinkingmachines.ai/blog/lora/) | Controlled SFT experiments include Qwen3 MoE. MLP/MoE coverage beats attention-only adaptation, even with comparable parameter budgets. LR, rank, initialization and scaling interact. | Compare routed-expert coverage before spending memory on a much higher attention rank. Do not turn its empirical LoRA/full-training LR relationship into a universal multiplier. |
| [SWE-smith, Appendix F.1](https://arxiv.org/html/2504.21798v1) | Successful agent trajectories train Qwen using torchtune, full-parameter SFT, LR 5e-5, up to three epochs and 32K context on 2–8 H100s. | Strong precedent for trajectory imitation and matching student tool format. Its 5e-5 is a full-training result, not direct evidence that QLoRA should use that LR. |
| [SWE-Gym, Appendix B.2](https://arxiv.org/html/2412.21139v1) | The 32B MoatlessTools experiment uses Unsloth LoRA, rank 64, LR 5e-4, batch 8, five epochs, 10,240 context on one H100. Other experiments use full training and different settings. | A closer agent-LoRA precedent, but still a dense model with much shorter context. Keep 5e-4 out of the first small-data sweep unless lower rates clearly underfit. |
| [REAP, authors' repository](https://github.com/CerebrasResearch/reap) | Expert saliency combines routing weight and activation magnitude; one-shot compression is evaluated without requiring recovery training. | Establish the actual untrained pruning curve before choosing a recovery target. Results on other coding models cannot predict ARC game loss. |
| [Minitron](https://arxiv.org/html/2408.11796v1) | Structured pruning followed by distillation can recover capability, with substantial continued training. | Useful recovery precedent, not a guarantee that 20 training-game trajectories repair aggressive pruning. Sol transcript SFT does not have teacher logits and is not their logit-distillation experiment. |
| [MAESTRO, Appendix A](https://arxiv.org/html/2607.08601v1) | A recent pruning/recovery preprint uses 2,000 SlimOrca examples, 1,024 tokens, attention LoRA rank/alpha 16/16, no dropout, LR 2e-4, batch 2, one epoch, 10% warmup then linear decay. | Supports testing a modest-rank recovery baseline. Its short text-only setting and different models limit transfer; this is supporting evidence, not the main recipe. |
| [rsLoRA](https://arxiv.org/abs/2312.03732) | Replaces alpha/r scaling with alpha/sqrt(r) to improve higher-rank learning behavior. | Do not change scaling convention during a rank sweep. Treat rsLoRA as a later independent experiment requiring a fresh LR check. |

The evidence does not identify a single best LR/rank. It supports a small,
controlled sweep, explicit adapter coverage, and evaluation on the task itself.
Most comparison studies also have many more independent tasks than our 25 games.

## 2. Framework comparison

| Framework | Exact-model evidence and limits | Decision |
|---|---|---|
| **Axolotl** | Dedicated `qwen4_exp` model support, text and vision QLoRA recipes, fused-expert quantization and PLE offload. Published short-context full-model recipes exceed our GPUs' capacity. | **First choice for qualification.** Reuse upstream loading, quantization, adapters and training lifecycle; add only the missing dataset, long-context and recovery pieces. |
| **ms-swift / Megatron-SWIFT** | Exact Flash-Next support; its demonstrated SFT configuration uses eight GPUs with tensor/expert/pipeline parallelism. | Strong candidate if the hardware changes to a multi-GPU machine. Less suitable as the first implementation for one Kaggle/Colab GPU. |
| **Transformers + PEFT + TRL/Accelerate** | Provides the necessary model, parameter adapters, custom collator/loss and checkpoint extension points. TRL supports VLMs/tools and nontruncated processing. | Fallback when framework integration becomes harder than a small explicit loop. Reuse qualified Axolotl components where practical; do not independently reimplement every optimization. |
| **Unsloth** | Reviewed Qwen3.8 training guide targets dense **27B `qwen3_5`**, not this `qwen4_exp` MoE. Inference/quantized model availability is not training evidence. | Do not select it based on the shared Qwen3.8 name. Reconsider if an exact-model NVIDIA training recipe passes our contract. |
| **LLaMA-Factory** | Reviewed Qwen3.8 template support and issue refer to 27B; they do not demonstrate our quantized Flash-Next workload. | Not ruled out universally, but insufficient exact-model evidence to put ahead of Axolotl. |
| **torchtune** | Used by the cited agent-distillation work, primarily with dense Qwen and full training. | Those experiments support the method, not a ready implementation of this hybrid MoE. |
| **SGLang/vLLM** | Repository has a working inference path. | Use for untrained and trained rollouts, separately from the trainer. |

Sources: [Axolotl exact-model guide](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html),
[SWIFT exact-model best practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html),
[TRL SFT](https://huggingface.co/docs/trl/sft_trainer),
[Unsloth training guide](https://unsloth.ai/docs/models/qwen3.8/train),
[LLaMA-Factory model issue](https://github.com/hiyouga/LlamaFactory/issues/10823),
[SWE-smith training code](https://github.com/SWE-bench/SWE-smith/tree/main/swesmith/train).
“Insufficient evidence” means not established by the sources inspected, not a
claim that support is impossible or absent from every branch.

### Axolotl: actual recipe and source inspection

Inspected Axolotl commit **0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a**,
dated 2026-10-07, rather than relying only on search snippets.

The [vision recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a/examples/qwen3.8-flash-next/vision-qlora.yaml)
uses rank/alpha 16/32, dropout 0, LR 2e-4, effective batch 4, one epoch,
10% warmup, cosine, zero weight decay, non-reentrant checkpointing and
unpacked 2,048-token multimodal samples. It explicitly targets both language
projections and fused expert parameters. This is a reference, not a config
to copy unchanged: our explicit last-reply mask and full contexts must replace
its example data handling.

The model guide reports approximately **110 GiB on one B300 with PLE offload**
for vision QLoRA, and roughly 100 GB of host memory is needed for the PLE
table in the recipe. Those numbers cannot establish fit on 96 GB or 80 GB,
nor can pruning savings be subtracted from them as a reliable memory forecast.
Colab host RAM can rule out that stock offload path even when GPU memory fits.

The [expert-quantization implementation contract](https://docs.axolotl.ai/docs/expert_quantization.html)
requires `quantize_moe_experts: true` for fused expert tensors; ordinary
bitsandbytes Linear replacement misses them. `lora_target_parameters` requires
dropout **0**. Its documented loading path requires an unquantized source for
expert quantization. This does not establish direct support for our Intel
AutoRound checkpoint. Use canonical NF4 from a matching unquantized/pruned
source, or qualify W4A16-LoRA as a separate fallback.

The [model support descriptor](https://github.com/axolotl-ai-cloud/axolotl/blob/0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a/src/axolotl/model_support/qwen4_exp/__init__.py)
and [QSA patch](https://github.com/axolotl-ai-cloud/axolotl/blob/0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a/src/axolotl/monkeypatch/models/qwen4_exp/modeling.py)
show architecture-specific handling. They disable incompatible LoRA kernels,
Liger and varlen SDPA; the indexer is intentionally outside autograd. Use
the named language projections rather than targeting every Linear, which
would include the indexer.

**Source-derived long-context limit:** `qsa_indexer_forward` constructs dense
query/block scores and a token-selection mask. With this model's four index
heads and compression ratio four, at 108,000 tokens the FP32 matmul output is
approximately `4 × L²` bytes = **43.5 GiB**, and one boolean token mask adds
**10.9 GiB**. Other intermediates and model memory are additional. These are
tensor-size calculations, not observed GPU peaks. Vectorizing the indexer
does not remove its quadratic allocation.

Therefore, first profile bounded query-block QSA with identical selections,
and a bounded attention execution path, before claiming that activation
offload makes long requests fit. Prove forward/gradient parity with the
reference. The exact recurrent/chunked-backward requirements in PLAN still
apply if layer checkpointing alone is insufficient. Do not shorten contexts
silently to make a stock recipe run.

[Axolotl activation-offload options](https://docs.axolotl.ai/docs/gradient_checkpointing.html)
can reduce retained activations; they do not by themselves eliminate an
oversized temporary tensor inside one operation. Its
[Cut Cross Entropy integration](https://github.com/axolotl-ai-cloud/axolotl/blob/0ffa4cc101935b6c5a82d3fe6a34cd4da8597e7a/src/axolotl/integrations/cut_cross_entropy/README.md)
lists `qwen4_exp`. Verify final-reply masking and loss/gradient parity with a
small exact CE reference, including any approximation/filtering options.
TRL's documented chunked NLL on nonignored positions is another candidate;
support in its generic API is not a Flash-Next capacity benchmark.

### Why ms-swift is second here

The published exact-model recipe uses rank 8, alpha 32, LR 1e-4, 8K context,
and TP2/EP4/PP2 across **eight GPUs**. Its 77.7 GiB offloaded measurement is
**per GPU in that eight-GPU experiment**, not a single-A100 fit result.
Its infrastructure makes more sense if multi-GPU allocation becomes available.
For current hardware, Axolotl's single-device quantized recipe is the closer
starting point; neither is yet proven at our context lengths.

## 3. Parameters: a budgeted experiment rather than fixed folklore

After baseline model selection and framework qualification, start the pilot
with the following **provisional** recipe:

| Parameter | Pilot default | Comparison / reason |
|---|---|---|
| LR | **1e-4** | Compare **5e-5, 1e-4, 2e-4** from identical initialization/data order. 5e-5 is a conservative candidate, not a settled result. |
| Dense language projection rank | **16** | Rank 8 for capacity fallback; rank 32 only if rank 16 underfits. No blanket rank-64 default. |
| Dense alpha / scaling | **32 / standard alpha/r** | Hold alpha 32, initialization and scaling convention fixed in the rank comparison; verify LR for finalists. Do not also switch to alpha=2r mid-sweep. |
| Adapter coverage | Attention + GDN + shared experts, then a matched trial adding routed-expert rank **1 or 2** | Compare coverage before increasing dense rank. Keep router, vision, PLE, head and embeddings frozen. |
| Dropout | **0** | Compatible with parameter-targeted expert adapters. Test 0.05 only for a modules-only branch if overfitting warrants it. |
| Weight decay | **0** | Minimal initial recipe; test 0.01 later if needed, not simultaneously with every LR/rank change. |
| Batch | Microbatch 1, accumulation 4 | Short batch respects memory and recovery. Compare accumulation 8 only if gradient noise or utilization justifies it. |
| Warmup / schedule | **5% then cosine to 10% of peak** | Five percent is our small-budget choice, not a literature optimum. Constant-after-warmup is a later ablation; published studies disagree on schedule. |
| Clip / optimizer | Grad norm 1.0; AdamW | Preserve simple optimizer state for the small adapter set. With expert adapters, measure 8-bit AdamW as a separate memory option. |
| Epochs | First production run **1** | Inspect half-epoch and final checkpoints; extend deliberately only if held-out performance warrants it. |

### Expert rank is not the same budget as dense rank

For conventional independent per-expert adapters on fused gate/up and down
weights, the config's hidden width 2560 and expert intermediate width 640 give:

`expert_adapter_parameters ≈ 48 × experts × expert_rank × (2560 + 1280 + 640 + 2560)`.

| Experts | Expert rank 1 | Rank 2 | Rank 4 | Rank 16 |
|---:|---:|---:|---:|---:|
| 512 | 173M | 346M | 692M | 2,768M |
| 448 | 151M | 303M | 606M | 2,422M |
| 384 | 130M | 260M | 519M | 2,076M |
| 320 | 108M | 216M | 433M | 1,730M |
| 256 | 87M | 173M | 346M | 1,384M |

These are layout-based estimates excluding dense adapters. Inspect the
framework's resolved tensor shapes/counts because fused implementations can
factor dimensions differently. At 384 experts/rank 2, 16 bytes per parameter
for FP32 adapter + gradient + Adam moments is approximately **3.87 GiB**;
rank 16 is about **30.94 GiB**, before dense adapters or activations. Recovery
archives and transfers grow too. This is why reducing expert rank can matter
far more than changing attention rank from 16 to 8.

[PEFT's parameter-targeting documentation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)
also describes additional expert-adapter runtime overhead. Measure it; do not
assume generic adapter serving support means this model/map/export is supported.

### Optional future search budget and fair comparisons

The following is a research menu for a later training budget, not work needed
for the current NLL-only base selection. The user's limited GPU time takes
priority over running this full search.

Do not take the Cartesian product of five models, three LRs and many ranks.

1. Finish untrained comparisons and pick one primary student plus at most
   one backup. Freeze expert map, quantization and data before training trials.
2. On the primary student, run three LR trials at rank 16 with modules-only
   coverage, each for the same first **1/8 epoch**. Use the same materialized
   sample order, representing all training games and the length distribution.
   Process complete requests, not truncated examples. Record exact token counts.
3. Promote the best two to **1/4 epoch total each** with their original global
   one-epoch schedules. Compare the frozen quick panel and full-fold NLL;
   early stability is useful, but a short-run winner is not automatically the
   best long-run LR.
4. At the leading LR, compare adding rank-1/2 expert adapters over the same
   prefix. If needed, spend the remaining budget on a dense-rank-32 comparison.
   Retest the adjacent LR if changing coverage/rank materially changes updates.
5. Cap exploratory SFT at approximately **one epoch-equivalent** of processed
   training tokens across trials. Three initial trials plus the two promotions
   cost 0.625 epochs; a 1/8-epoch coverage trial leaves 0.25 for a rank/LR check.
   Validation and rollout time are additional and explicitly budgeted. Do not
   continue an unlimited search on five validation games.

Select using validation NLL and stable resource use. A rollout sanity check
is a separately budgeted downstream option. Start the final controlled run from the untouched base unless continuing
an exactly matching promoted run; log reused training work unambiguously.
Selection consumes validation: final generalization requires another held-out
evaluation or fixed-recipe cross-validation, not a test claim on fold 0.

## 4. Current stage zero: sampled teacher NLL only

[NLL_EVAL.md](NLL_EVAL.md) is the detailed design and cost estimate. Its decisions:

- Fold 0 stays game-disjoint: sk48, sp80, tn36, cd82, ar25. Prepare six
  length-stratified random requests per game, shared across all five counts.
  Preserve all 30 selected requests across quota-limited sessions.
- Recover saved per-game REAP statistics and build nested train-only maps on
  CPU when possible. If using existing all-game maps, label that comparison
  validation-exposed instead of claiming a clean estimate.
- Reuse the working W4A16 forward replay, load the full model once, and change
  router masks between fresh-cache replays. Preserve full selected contexts
  and score only the last sol reply, not old Qwen self-trace generations.
- Save token NLL with validated rationale/code/format annotations and request
  game/level/length metadata. No generation, gameplay or backward pass.
- Select from the NLL/memory curve using the predeclared tolerance and small
  sample caveats. A targeted reserve is optional and remains separate from
  the probability-sampled estimate. No automatic full-fold expansion.
- Persist exact model/map/data/scoring identities and per-request results
  outside Kaggle. A later change of quantization needs a matched new baseline.

The previous preference for four gameplay passes per game/configuration is
withdrawn for this budget. Game scores and failure rates need actual rollouts;
this NLL experiment does not claim to measure them. NLL is the direct available
transcript-imitation objective, but does not prove post-SFT learning speed.

## 5. Revised order of work

1. Freeze fold, inputs and clean expert maps.
2. Run the bounded sampled NLL comparison for all expert counts and save it.
3. Select the base from measured NLL and the estimated training-memory envelope;
   actual backward fit remains pending.
4. Qualify Axolotl NF4, exact labels, PLE host memory, bounded QSA, long-context
   backward and external recovery. Rebaseline if the quantization/base changes.
5. Choose a training recipe; run further LR/coverage/rank trials only if budgeted.
6. Run resumable SFT and compare with matched untrained baselines.

This order supersedes the earlier proposal to start by training 384 experts.
