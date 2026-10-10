# Astra review of failed Opt7

User instruction: whenever an implementation fails its numerical gate, obtain
an Astra subagent review before accepting or continuing that implementation.
Review model: GPT-6 Astra; task `astra_opt7_review`; read-only source/evidence review.

The review examined rejected attempt `20261009233337155-c92c57e2`, native
AutoRound expert dispatch and prepared Opt7 v4. It found no demonstrated missing
gradient, stale-weight or staging race in v4. This does not qualify its numerics.

## Findings and actions

1. The component fixture pooled input/router cotangents with adapter gradients;
   their norms could hide adapter-only error. Changed the gate to adapter-only
   global relative L2 below1%; input/router errors are reported separately.
2. Route/unroute checks covered only tiny launch sizes. Added a CUDA test with
   16,249 tokens, top-k10, hidden2560 and8192-token windows, requiring bitwise
   native routed sums and deterministic gather VJPs.
3. Padding split terminal rows to8192 does not prove equivalence to the native
   9705-row projection dispatch. Keep the representative projection diagnostic
   and bitwise forward preflight; isolate base versus LoRA contributions if it
   fails. No claim that BF16 rounding alone explains the full-model drift.
4. Existing routing fixtures used uniform weights and every expert. Added an
   unsplit nonuniform top-k case with uneven expert counts and an unused expert,
   isolating custom dispatch without changing native expert GEMM row counts.
5. Separate squared-output losses mix forward error into backward diagnostics.
   Changed fixture losses to use the same fixed upstream cotangent on both sides.

Historical status at the first review: GPU validation was pending runtime restoration.
The revised fixtures subsequently passed on CUDA in Opt7 v4 attempt
20261010041736293-6ef871bc; see the distinct failure review below. Fixture success
does not qualify full-model gradients.

The replacement Kaggle notebook356919557 has no old temporary exports, package
view, anchor or reference shards. Restore these and validate source/input pins
before resuming full-model qualification. The old disconnected run remains
uncollected; its forward observation cannot establish backward acceptance.


## Opt7 v4 paired gate failure — 2026-10-10

Reviewed frozen attempt `20261010041736293-6ef871bc`, execution `bb7f154`, including
source/components/expert_chunks.py, output/chunking-comparison.json,
output/chunking.json, component fixture JSON and process.log. This review only
updates this report; no GPU execution, source edits or network operations.

### Result

The paired gate fails: candidate loss0.6243785619735718 versus unchanged Opt3
control0.6243783235549927, global gradient relative L2 **0.01783424283598788
(1.7834242836%)**, cosine0.9998411570340994,111/74472 exact tensors,
33/74394 exact nonzero-reference tensors. The absolute scalar loss difference is
2.384185791015625e-7; small scalar error does not establish gradient equivalence.
The user gate remains bitwise gradients or global relative L2 below1%.

Both phases used the verified native FLA strict reverse-scan profile. Immediately
preceding native restoration matched all74472 original reference gradients with
that profile. Thus the previously diagnosed unpinned reverse-scan configuration
is not an adequate explanation for this paired failure. The profile is TEST ONLY:
real training must retain native autotuning, as required by the user and guarded
in f2ec8d6. This review does not authorize applying the test profile to training.

### Updated fixture status

The attempt ran31 tests successfully. CUDA route/unroute checks include the full
16249-token/top-k10/hidden2560 sizes. Fixed-cotangent expert fixtures report:

| Fixture | Adapter relative L2 | Input/router relative L2 |
| --- | --- | --- |
| Small split experts | 0.002394047968155222 | 0 / 0 |
| 9705rows, hidden2560, intermediate640, rank16, split8192 | 0.0030246938185079734 | 0 / 0 |
| Unsplit uneven routing | 0 | 0 / 0 |

The real-size split fixture forward is bitwise identical. These results establish
that the corrected primitive fixtures pass; their activation/weight/cotangent
values do not prove equivalence for the actual model's experts.

### Where the evidence localizes the difference

Unlike restored-native failure, this comparison has no exact top-layer boundary.
Layer47 MLP is already different (relative L2 0.0013548726953749918); its111 exact
tensors account for all exact tensors in the model. Layer47 self attention is
0/8 exact, relative L2 0.003304400567054253. Layer46 MLP is0/1542 exact,
relative L2 0.003973513402773262; layer46 linear attention is0/10 exact,
relative L2 0.005455370294662962. Parameter statistics alone therefore cannot
identify the first local backward divergence: changed forward activations and
upstream cotangents already confound top-layer comparisons.

Only SIX experts actually split in this full anchor:

| Layer | Split experts | Maximum assigned rows |
| --- | --- | --- |
| 15 | 1 | 9235 |
| 44 | 1 | 9694 |
| 45 | 1 | 9173 |
| 46 | 2 | 10061 |
| 47 | 1 | 11948 |

All other43layers execute unchanged per-expert row counts. Layer15 is the first
possible split-induced forward divergence, but that is a hypothesis until actual
layer outputs are compared. Terminal padding to8192 does not reproduce the
native full expert row count, and the fixture's9705-row success does not cover
all listed shapes and real activation distributions.

### Source assessment and focused next diagnostic

The frozen source matches reviewed v4. Global argsort/slot identity, native
routing multiply and top-k reduction, and native gather VJP replay remain
structurally consistent. No demonstrated missing-gradient or staging-lifetime
bug was identified. Split LoRA weight VJPs are computed separately and added in
FP32 at expert_chunks.py:136; this differs from native full-row GEMM reduction
and explains why the fixture permits nonzero adapter error even with exact dx.
That mechanism alone, if outputs and input/router VJPs are exact, would not
explain changed shared-expert/self-attention gradients throughout the model.
Actual forward or input/router VJP differences, or a shared control-kernel
variation, must be isolated before attributing the whole1.78% to adapter sums.

Recommended next diagnostic, before another full qualification:

1. Run one anchor forward with an expert shadow comparison. For each layer,
   evaluate native and v4 experts on the SAME captured hidden input, routing IDs,
   routing weights, and parameters. Compare outputs bitwise and stop at the first
   mismatch. Keep only compact metrics; activation values remain in RAM. Preserve
   staging phase/direction and serialize the shadow calls so they cannot consume
   each other's lookahead state. Start at layer0 rather than assuming layer15.
2. At that first mismatch, isolate per-expert gate/up/down outputs, splitting
   base quantized and LoRA contributions. Record actual row counts, shapes,
   strides, dtype, and kernel choices. Compare split versus unsplit native
   execution with identical values and freeze unrelated autotuner choices.
3. Independently compare native/v4 VJPs on a representative actual split layer
   using one fixed upstream cotangent. Report dx, routing-weight gradients, and
   adapter-only errors separately, distinguishing split from unsplit experts.
   Also replay unsplit v4 on the same layer to isolate custom dispatch from
   changed projection shape. Use actual11948/10061/etc row counts if the first
   mismatch does not already resolve the cause.
4. If every expert shadow forward is exact, compare final head inputs and repeated
   unchanged Opt3 head VJPs on identical values. A paired full pass does not itself
   prove deterministic shared CCE/kernel behavior. Do not change the head or its
   acceptance criterion merely to explain the failure.
5. Only after identifying and correcting a concrete discrepancy, rerun the
   required paired full-model gate. Preserve the original reference and rejected
   v4 evidence; do not promote v4 based on passing component fixtures.

This plan tests actual model values without another blind full forward/backward
candidate attempt. No claim is made that intrinsic BF16 rounding is the sole
cause, nor that a1warp test profile resolves production native-autotuning
qualification.

### Active quantized projection path

The SHA-verified qlinear_tritonv2_zp.py imports QuantLinearFunction from
triton_utils_zp/dequant.py, not the adjacent fused kernels.py. The captured import
manifest confirms dequant.py is active. Its quant_matmul_248 dequantizes W and
uses ordinary Torch `input @ W`, or `grad_output @ W.t()` for the input VJP
(dequant.py:159–188). The dequantizer autotunes by weight numels, which does not
change when splitting rows. Therefore if layer15 is the first forward mismatch,
prioritize BF16 Torch/cuBLAS GEMM shape choices for M9235 versus8192, inspecting
both base projection and LoRA branch products. Do not investigate the unused
fused quantized-matmul M/N/K autotuner as if it executed this capture. Compare
dequantized W bytes once to exclude that independent step, then use identical
W/input/cotangent values for each GEMM comparison. Preserve actual native and
chunked shapes/strides rather than constructing another9705-row synthetic case.

## Verified actual-value shadow diagnostic — 2026-10-10

Independently reviewed collected attempt `20261010044356417-24bc0932`, execution
507af2c, including result.json, all48 expert-shadows.json rows, and all four
expert-vjps.json variants. Every expert forward shadow is bitwise identical on
the candidate's same hidden input, route IDs and route weights. No first forward
mismatch was found. The retained VJP fixture is actual first-split layer15 with
one CPU-seeded fixed random cotangent, not the full model's loss cotangent.

| Layer15 variant | Output/dx/router | All1536 adapter relative L2 | Split6 adapter relative L2 | Unsplit1530 adapters |
| --- | --- | --- | --- | --- |
| Native repeat | bitwise | 0 | 0 | all exact |
| Chunk8192 | bitwise | 0.0008741965513522961 | 0.003074710280653855 | all exact |
| Custom unsplit | bitwise | 0 | 0 | all exact |

This excludes a forward mismatch on the48 observed same-input expert calls and
shows that the tested layer15 input/router VJP does not introduce upstream drift.
Its adapter-only chunk error remains localized to the split expert. It does not
explain the full1.7834% paired difference, does not prove equality for every
possible cotangent, and does not test actual-value VJPs at split layers44–47.
No Opt7 acceptance follows.

The next proposed combined diagnostic is focused: capture actual inputs for the
five split layers during one no-grad forward, then compare fixed-cotangent native
and chunk VJPs on those values. Also capture the actual651x2560 selected head
hidden states and repeat unchanged CCE exact scalar loss and hidden VJP. Give the
CCE repeat priority once inputs are captured: paired full losses differ even
though these expert shadows are exact, and widespread top-layer differences
could originate in the shared head cotangent.

For CCE repeats, use fresh identical leaves with the original selected-hidden
dtype/contiguous layout, target order, frozen head weight, cce_exact implementation,
filter_eps=None, mean denominator and autocast settings. Compare scalar bits and
all hidden-cotangent bits, not just relative tolerance. Torch deterministic mode
alone does not prove deterministic custom Triton execution. If identical-input
CCE repeats drift, the previous paired result cannot be attributed entirely to
Opt7; this still does not grant acceptance. Keep all activations and cotangents
in RAM and save comparison metrics only. FLA profile remains test-only.
