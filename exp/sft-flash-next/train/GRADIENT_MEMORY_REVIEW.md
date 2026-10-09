# Latest Liger FLCE evidence — 2026-10-09

The user-selected target-only frozen-head Liger experiment is logged as
`liger-flce-v2`; see `v2.md`. It removes dense vocabulary saved tensors and v1's
full-context head backward scratch, with synthetic operator GPU peak 1.712 GiB.
However, full-model 16K allocated GPU peak remains **58.523 GiB**, and sampled
backward RSS remains **129.275 GiB**: no additional whole-model peak saving was
measured. Other model operations dominate this anchor's peak; their exact
identity was not profiled in this experiment.

Exact qualification failed: loss +2 FP32 ULPs; all 744 gradients finite/nonzero,
0 byte-identical, **1.610% global relative L2 error**, maximum tensor 4.545%.
Synthetic interleaved gradients differ 5.03–5.06%. The candidate is diagnostic,
with no optimizer update and no production promotion or accepted tolerance.
For 43,806 targets, calculated BF16 chunk size is 512 rows (0.237 GiB), and
saved selected hidden gradients 0.209 GiB. This avoids the former 59.745 GiB
context-shaped head scratch, but is not a measured 130K capacity result.

# Gradient evidence and memory implications (2026-10-09)

## Latest measured target-mask result

`target-mask-v1` supersedes the earlier lack of a complete-model head pass
below. With the pinned **nonzero A/B** fixture, the 16K native loss and every
one of the 744 raw adapter gradients match bitwise. GPU allocated peaks:
forward 49.775 GiB and backward 58.523 GiB, versus 77.802 / 85.100 GiB for v0.
Backward sampled CPU RSS: 129.273 GiB versus 147.794 GiB. Training phase time:
333.606 s versus 336.492 s (one comparison, no demonstrated slowdown).

The mask supports arbitrary interleaved targets. Mean and summed CE with four
separated spans passed exact CUDA loss/hidden-gradient checks using the actual
head and full 16K hidden shape. Native NLL reduction positions are retained
with a tiny T-by-1 buffer; only target rows have logits/log-probabilities.
Two actual FP32 CPU saves have shape [651,248320], 0.602 GiB each. The head
backward retains its native full-context BF16 scratch. Model loading is
unchanged and still dominates overall host RSS. See `v1.md` and
`metrics/target-mask-v1.json` for complete definitions and evidence.

This qualifies the measured anchor and interleaved operator cases. The separate
all-assistant production gate still requires its own real multi-span sample,
native replay and complete-model candidate gradients. This first optimization
alone cannot fit the largest 130K trajectory: the approximately 59.745 GiB
backward scratch plus 39.35 GiB resident model already exceeds this GPU's
capacity, before other allocations. Bounded CE and smaller backward storage
remain separate unqualified next changes.

## Scope of the existing evidence

The real-model reference sample contains 16,249 tokens, 651 final-reply
targets, and seven images. The historical all-744 comparisons use random A,
zero B LoRA initialization. Their 372 zero A gradients are expected but leave
the A-gradient path unexercised. These results are partial diagnostic evidence;
they do not qualify continued training with nonzero A and B or the separate
all-assistant trajectory objective. A diagnostic nonzero A/B fixture must be
marked test-only and must never initialize production training.
Each result also belongs to its recorded source/runtime hashes; a historical
pass does not automatically qualify subsequently changed source files.

The following peaks are PyTorch allocated GPU memory, not nvidia-smi samples.
The 85.10 GiB native replay is a separate historical replay and provides context,
not a paired measurement of each flag's exact memory saving.

| Experiment | Gradient evidence on the 16K sample | Measured GPU peak | CPU evidence |
| --- | --- | ---: | --- |
| Deterministic unchanged native repeats | All 744 gradients bitwise equal, including a fresh-process replay | 85.10 GiB in the later native replay | No paired peak-RSS comparison |
| Native mask storage | All 744 gradients bitwise equal | 84.89 GiB | Peak host RAM not recorded |
| Selective CPU saved-tensor storage | All 744 gradients bitwise equal | 85.13 GiB | 46.99 GiB cumulative copied bytes; 15.65 GiB deduplicated bytes |
| Mask storage + selective CPU storage | All 744 gradients bitwise equal | 84.89 GiB | Same cumulative copy/deduplication counters |
| Revised padded native vocabulary head | Eight bitwise loss/hidden-gradient passes on synthetic mixed-dtype hidden states; no complete-model candidate pass yet | 10.40–10.47 GiB operator-only, versus 46.50 GiB native operator-only | No whole-model host-RAM measurement |

The original selected-logit and chunked-loss candidates failed the all-adapter
gate (about 1.2% relative gradient error). Their 58.52 GiB GPU peaks do not
qualify them. Checkpoint groups, disk offload, bounded normalization, residual
mixing, and the prior custom model backend have no completed applicable
all-744 native-equivalence proof. The custom backend's 130K capacity numbers
cannot be assigned to this native recipe.

Sources: `gradient-results/native-flags-initial/*/result.json`,
`gradient-results/native-head-gradient-audit-initial-v1/reference_repeat/result.json`,
`gradient-results/native-head-mixed-dtype-v1/report.json`, and
`NATIVE_GRADIENT_AUDIT.md`.

## Implications at 130K

These are tensor-size calculations and implementation implications, not measured
130K whole-model peaks. GPU peaks from separate phases or experiments cannot be
added or subtracted as though their lifetimes coincide.

* **Mask storage:** a 130,000-square boolean mask is 15.74 GiB; the corresponding
  BF16 additive bias is 31.48 GiB (FP32 is 62.96 GiB). The patch uses the QSA
  hidden-state dtype, so BF16 must not be assumed without observing that boundary.
  Avoiding redundant boolean masks can prevent
  tens of GiB of transient GPU allocations at this length. The current patch
  still materializes the final dense additive bias and keeps native QSA/SDPA.
  It does not remove quadratic attention-bias storage. No host-RAM saving was
  measured.
* **Selective CPU storage:** exact saved activations move to host RAM while
  frozen resident model tensors are excluded and duplicate saves can share one
  copy. This trades GPU activation storage for CPU storage and transfer time.
  The measured baseline already used stock `save_on_cpu`: the selective version
  replaces it, rather than adding offload to a GPU-only baseline. Its intended
  improvement over stock storage is avoiding unnecessary resident-weight copies
  and duplicate saves; tensors smaller than 1 MiB stay on their original device.
  The 46.99 GiB counter is cumulative across pack calls, including recomputation;
  it is not simultaneous residency or peak RSS and must not be scaled into a
  claimed 130K RAM requirement. The frozen PLE table already occupies about
  95.37 GiB of host RAM. Host peak must be measured with the complete recipe.
* **Mask + CPU storage:** their combination passed at zero B, but its 130K CPU
  and GPU peaks remain unmeasured. A nonzero-adapter comparison is required.
* **Vocabulary head:** the new 58-trajectory dataset has 1,399,743 target tokens
  among 5,712,174 total tokens (24.5046%), versus 4.0064% in the anchor sample.
  The target-only head therefore has a smaller proportional saving on actual
  trajectories. With vocabulary size 248,320, median BF16 logits storage is
  46.36 GiB for all positions and 11.81 GiB for target positions; the median
  paired reduction is 35.07 GiB. FP32 CE workspaces, backward padding, activations
  and offload alter the actual peak. This is an unqualified candidate estimate.

The dataset's maximum target count is 43,806. Twelve of 58 trajectories exceed
32,768 targets, so a fixed 32,768-row backward cannot cover the complete dataset.
A proposed 32,768/65,536 row scheme (46/12 trajectories) reduces the corresponding
BF16 row-buffer size by a median 28.43 GiB relative to native-length rows.
Numerical equivalence of either shape on the multi-span production objective
must be measured; these bucket choices are proposals, not accepted recipes.
For the largest trajectory (129,169 total / 43,806 targets), full/selected BF16
logits are 59.74/20.26 GiB, while a 65,536-row BF16 workspace is 30.31 GiB.

The new-server native anchor capture completed with 86.40 GiB sampled GPU use
and 167.00 GiB sampled process RSS. These are different measurement definitions
and runtime evidence from the historical allocated-memory table. The host is
already close to its 175 GiB limit at 16K. No validated native 130K training
capacity or reliable whole-model memory-saving number exists yet.

These reference and comparison passes perform no optimizer update. Real AdamW
training also materializes moment tensors: for 33,478,656 FP32 adapter values,
the two moments add about 0.249 GiB, in addition to about 0.125 GiB each for
adapter parameters and gradients. Update/clipping temporaries can add more.
Whole-model training capacity must include an actual disposable optimizer step,
not just a successful reference backward.

## Stock reference planning estimate at 130,000 tokens

The stock loss computes vocabulary logits at all positions, including ignored
labels. Full BF16 logits occupy 60.13 GiB; their FP32 cast occupies 120.26 GiB.
If both coexist with the approximately 39.35 GiB resident model, this phase
already needs about 219.74 GiB before other CE workspaces. The measured 16K
native-head peak is 46.50 GiB, including a 1.184 GiB head weight. Extrapolating
its sequence-dependent portion gives:

`39.35 + (46.50 - 1.184) * 130000 / 16249 = 401.90 GiB GPU`.

This is an approximate planning estimate, not measured 130K capacity. It assumes
the same operator allocation pattern; kernel workspace and tensor lifetimes
can change. It is consistent at 16K with the observed whole-model GPU peak.
Host RAM is less predictable: naively scaling everything above the 95.37 GiB
PLE table gives `95.37 + (167.003 - 95.37) * 130000 / 16249 = 668.47 GiB`.
Fixed host overhead and temporary allocations make that extrapolation uncertain;
it is only an illustrative extrapolation. The initially stated 600–700 GiB
range is not supported as a reliable planning envelope and is withdrawn.
Different saved tensors can also scale quadratically, so this is not a bound.
Measuring fixed versus length-dependent RSS is necessary to estimate host RAM.
The stock reference is beyond the current 95 GiB GPU / 175 GiB host limits.

The target-only candidate still computes CE over all selected targets at once.
With median/max target counts 25,539/43,806, an unchanged CE allocation pattern
can itself exceed GPU capacity. Selecting targets alone does not establish fit;
bounded CE workspace is another proposed change that requires qualification.

Priority by expected GPU saving and speed: (1) target-only head plus bounded CE
workspace and padded native backward; (2) redundant native mask storage removal;
(3) blockwise attention avoiding the final dense bias; (4) bounded normalization
and residual intermediates; (5) selective CPU storage; (6) grouped checkpoints;
(7) disk offload. Items 5–7 primarily address activation placement or host RAM
relative to the existing reference and can be reordered when actual peaks are
known. New kernels and row counts remain unqualified until their gradient gates
pass; this ordering is a proposed experiment priority.

The trajectory outputs were found in the main local checkout and all four MD5s
matched PR #26's DVC lockfile at `ce2ba09226aace16586ed53221e218e1422fb3b5`.
A direct DVC pull could not retrieve them because the four output objects had
not been uploaded to the configured remote. Counts above use those verified
local outputs, not an assumed successful remote pull.
