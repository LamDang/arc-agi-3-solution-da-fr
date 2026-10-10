# Astra review: restored native gradient mismatch

Reviewed 2026-10-10. Read-only evidence/source analysis; only this report written.
No GPU runs, network operations, raw candidate gradients, or reference replacement.

## Conclusion and gate

Corrected fold0 replay `20261010033254797-28be4e26` has exactly the saved native
loss, 0.6244627833366394, but fails the unchanged native bitwise gradient gate
against `20261009221135477-25bf58fe`: global relative L2 0.017380714155953398
(1.7380714156%), cosine 0.9998489888439394, 3106/74472 exact tensors and
3028/74394 exact nonzero-reference tensors. Do not relax that gate or promote a
new reference.

The first visible mismatch in reverse layer order is **layer46 linear attention,
only the in_proj_a LoRA A/B tensors**. All layer47 gradients and layer46 MLP
gradients match exactly. This strongly localizes the next investigation to the
GDN decay-gradient branch. Parameter summaries alone cannot prove the earliest
individual operator or exclude differences that disappear when cast to BF16.

## Exact boundary

| Component | Exact tensors | Relative L2 |
| --- | --- | --- |
| Layer47 self attention | 8/8 | 0 |
| Layer47 MLP, including all routed experts | 1542/1542 | 0 |
| Layer46 MLP, including all routed experts | 1542/1542 | 0 |
| Layer46 linear attention | 8/10 | 0.0000075453488670376175 |
| Layer46 in_proj_a LoRA A | 0/1 | 0.0003186249425131773 |
| Layer46 in_proj_a LoRA B | 0/1 | 0.000006692085791528158 |

Layer46 out_proj, in_proj_qkv, in_proj_z, and in_proj_b A/B all match exactly.
Layer45 and earlier then accumulate differences. Aggregate table below reports
percentages; the boundary table above uses raw ratios.

## Identity and source review

Both captures record Torch2.11.0+cu128, Transformers5.18.0, PEFT0.20.0,
AutoRound0.15.0 and Python3.13.15. Sample and config hashes match. The replay
initial-comparison matches all74472 initial tensors. Corrected export index is
`b9dcb19699ef52c057031be4b8b6fed5697b23b641d0ccf0a8a59b1d05d241d6`;
main-agent checks also verified84 expert/router byte comparisons.

There are zero changed hashes at common absolute imported-source paths.
Normalizing `/site-without-torchao/` versus `/package/` prefixes finds only
`tokenizers/trainers/__init__.py` changed among common package paths. In particular,
FLA, causal-conv Python wrappers, native modeling source, and AutoRound sources
match. The artifact records do not establish identity of compiled CUDA extension
binaries, driver, generated Triton code, or selected autotune configurations.

The active local component changes are the disabled chunk branch and call/acquire
refactor in expert_offload.py, plus disabled chunk selection/imports in model.py.
The replay uses architecture label optimized with native head and every chunk
flag false; original label was reference. Source review finds no active chunk
arithmetic change. A frozen-original-source diagnostic can still exclude subtle
execution differences, but the layer46 localization is more informative than a
blind whole-model rollback.

## Leading hypothesis and isolation

Native `modeling_qwen4_exp.py:580` computes
`g = -A_log.float().exp() * softplus(a.float() + dt_bias)`.
The convolution is on the separate qkv branch. Since layer46 qkv adapter gradients
are exact while only a differs, the rebuilt causal convolution extension is a
lower-priority first suspect, despite its incomplete binary provenance.

SHA-verified FLA `ops/gated_delta_rule/chunk.py:217–246` computes dg from
`chunk_bwd_dqkwg` and `prepare_wy_repr_bwd`, adds the two contributions, and calls
`chunk_local_cumsum(..., reverse=True)`. This FP32 decay cotangent flows through
the native softplus to in_proj_a. Q/k/v/beta cotangents can round identically in
BF16 even if intermediate FP32 reductions differ.

A particularly concrete candidate is FLA `ops/utils/cumsum.py:24–30`: its scalar
kernel autotunes num_warps1/2/4/8, with REVERSE in the cache key. Lines68–72 use
`tl.cumsum`, `tl.sum`, and `-cumsum + total + input` for the reverse scan. A fresh
server's selected backward configuration could therefore change dg without
changing forward. This is a hypothesis, not a confirmed diagnosis. The matching
`wy_fast.py:18–22` already restricts Blackwell backward to2warps/4stages, reducing
that particular autotune suspicion.

Minimal next diagnostic:

1. On one unchanged forward, retain layer46 GDN inputs and upstream cotangent in
   RAM only. Record resolved kernel/backend identities and selected Triton configs.
2. Replay the isolated layer46 backward. Compare both in_proj_a gradients against
   the saved reference tensors, and verify other layer46 projections remain exact.
3. Isolate dg before/after reverse chunk_local_cumsum. Replay its four existing
   warp configurations with identical inputs, then propagate each result through
   the native decay transform and a-projection VJP. This tests the narrowest
   backward-only configuration difference without another full capture.
4. If all scan variants agree, inspect the two pre-scan dg contributions and their
   selected configs. Also replay the a-projection with fixed identical da to
   distinguish its GEMM VJP from upstream decay-gradient differences.
5. Confirm any identified correction with the complete unchanged native equality
   gate against the original archive. Do not infer acceptance from an isolated
   adapter match.

An old compiled causal-conv binary hash was not found in reviewed capture
provenance. Its archived Python wrapper hashes match current, which is weaker
than binary identity. Preserve this limitation rather than claiming that all
runtime artifacts have been proven identical.

The local inspected native modeling source SHA256 is
`0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206`;
FLA gated_delta_rule/chunk.py SHA256 is
`fd4e01dc22a8c139c2a6eb61e47ae472a50322e4b4fff006cc5039a4602b310e`.
Both, plus cumsum.py and wy_fast.py, were checked against original imported hashes.

## All layer/component aggregates

Reconstructed from comparison.json using squared reference norms and per-tensor
relative L2; zero-reference exact tensors contribute zero. These are component
norm-weighted errors, not averages of tensor percentages.

| Layer | Component | Exact tensors | Relative L2 percent |
| --- | --- | --- | --- |
| 47 | self_attn | 8/8 | 0.000000000% |
| 47 | mlp | 1542/1542 | 0.000000000% |
| 46 | mlp | 1542/1542 | 0.000000000% |
| 46 | linear_attn | 8/10 | 0.000754535% |
| 45 | mlp | 6/1542 | 0.034684951% |
| 45 | linear_attn | 0/10 | 0.174165174% |
| 44 | mlp | 0/1542 | 0.311885461% |
| 44 | linear_attn | 0/10 | 0.403396074% |
| 43 | self_attn | 0/8 | 0.559780092% |
| 43 | mlp | 0/1542 | 0.498619314% |
| 42 | mlp | 0/1542 | 0.585811688% |
| 42 | linear_attn | 0/10 | 0.648861048% |
| 41 | mlp | 0/1542 | 0.702012191% |
| 41 | linear_attn | 0/10 | 0.695386555% |
| 40 | mlp | 0/1542 | 0.760500237% |
| 40 | linear_attn | 0/10 | 0.617619975% |
| 39 | self_attn | 0/8 | 0.652282193% |
| 39 | mlp | 0/1542 | 0.776072832% |
| 38 | mlp | 0/1542 | 0.877697051% |
| 38 | linear_attn | 0/10 | 0.730269065% |
| 37 | mlp | 0/1542 | 0.982794513% |
| 37 | linear_attn | 0/10 | 0.841096866% |
| 36 | mlp | 0/1542 | 1.014868973% |
| 36 | linear_attn | 0/10 | 0.962607818% |
| 35 | self_attn | 0/8 | 0.840374601% |
| 35 | mlp | 0/1542 | 1.069678465% |
| 34 | mlp | 0/1542 | 1.110059234% |
| 34 | linear_attn | 0/10 | 0.901257225% |
| 33 | mlp | 0/1542 | 1.144369044% |
| 33 | linear_attn | 0/10 | 0.935890942% |
| 32 | mlp | 0/1542 | 1.216166786% |
| 32 | linear_attn | 0/10 | 1.147735111% |
| 31 | self_attn | 0/8 | 1.230641758% |
| 31 | mlp | 0/1542 | 1.388492298% |
| 30 | mlp | 0/1542 | 1.428780579% |
| 30 | linear_attn | 0/10 | 1.239142017% |
| 29 | mlp | 0/1542 | 1.408863092% |
| 29 | linear_attn | 0/10 | 1.215611455% |
| 28 | mlp | 0/1542 | 1.489175489% |
| 28 | linear_attn | 0/10 | 1.184262866% |
| 27 | self_attn | 0/8 | 1.660898912% |
| 27 | mlp | 0/1542 | 1.642696239% |
| 26 | mlp | 0/1542 | 1.722229658% |
| 26 | linear_attn | 0/10 | 1.606007740% |
| 25 | mlp | 0/1542 | 1.910386976% |
| 25 | linear_attn | 0/10 | 1.678655642% |
| 24 | mlp | 0/1542 | 2.017459013% |
| 24 | linear_attn | 0/10 | 1.419442873% |
| 23 | self_attn | 0/8 | 1.265560641% |
| 23 | mlp | 0/1542 | 2.068951170% |
| 22 | mlp | 0/1542 | 1.935640565% |
| 22 | linear_attn | 0/10 | 1.201774502% |
| 21 | mlp | 0/1542 | 2.112304888% |
| 21 | linear_attn | 0/10 | 1.534064941% |
| 20 | mlp | 0/1542 | 2.224449788% |
| 20 | linear_attn | 0/10 | 1.892500982% |
| 19 | self_attn | 0/8 | 2.005662146% |
| 19 | mlp | 0/1542 | 2.235459801% |
| 18 | mlp | 0/1542 | 2.196922502% |
| 18 | linear_attn | 0/10 | 2.066206805% |
| 17 | mlp | 0/1542 | 2.273416291% |
| 17 | linear_attn | 0/10 | 2.216350212% |
| 16 | mlp | 0/1542 | 2.383971590% |
| 16 | linear_attn | 0/10 | 2.211006225% |
| 15 | self_attn | 0/8 | 2.107756307% |
| 15 | mlp | 0/1542 | 2.369436005% |
| 14 | mlp | 0/1542 | 2.296086927% |
| 14 | linear_attn | 0/10 | 2.202330157% |
| 13 | mlp | 0/1542 | 2.328244658% |
| 13 | linear_attn | 0/10 | 2.206357549% |
| 12 | mlp | 0/1542 | 2.296632660% |
| 12 | linear_attn | 0/10 | 2.174580257% |
| 11 | self_attn | 0/8 | 2.272635681% |
| 11 | mlp | 0/1542 | 2.360697529% |
| 10 | mlp | 0/1542 | 2.251250481% |
| 10 | linear_attn | 0/10 | 2.210881972% |
| 9 | mlp | 0/1542 | 2.313043822% |
| 9 | linear_attn | 0/10 | 2.205471311% |
| 8 | mlp | 0/1542 | 2.342045324% |
| 8 | linear_attn | 0/10 | 2.347496029% |
| 7 | self_attn | 0/8 | 2.328989825% |
| 7 | mlp | 0/1542 | 2.263536650% |
| 6 | mlp | 0/1542 | 2.281832457% |
| 6 | linear_attn | 0/10 | 2.319780923% |
| 5 | mlp | 0/1542 | 2.318644387% |
| 5 | linear_attn | 0/10 | 2.309139809% |
| 4 | mlp | 0/1542 | 2.380267149% |
| 4 | linear_attn | 0/10 | 2.357443826% |
| 3 | self_attn | 0/8 | 2.369966197% |
| 3 | mlp | 0/1542 | 2.544640288% |
| 2 | mlp | 0/1542 | 2.372733684% |
| 2 | linear_attn | 0/10 | 2.277041756% |
| 1 | mlp | 0/1542 | 2.355341042% |
| 1 | linear_attn | 0/10 | 2.240188009% |
| 0 | mlp | 0/1542 | 2.256876419% |
| 0 | linear_attn | 0/10 | 2.208112580% |

## Process-local diagnostic implementation notes

To force an existing reverse-scan config without editing installed package files,
inspect the runtime wrapper chain: outer Triton Heuristics, CachedAutotuner, then
JITFunction. A temporary proxy for that one autotuner instance's `run` can delegate
forward calls unchanged and invoke its underlying JITFunction directly for
REVERSE=True, passing all original arguments and all fields from the selected
existing Config (`all_kwargs()` in the matching Triton API). Outer heuristics
must remain active so HAS_SCALE/IS_VARLEN are supplied. Inspect the installed API,
log the actual invoked metadata, and restore the original callable in finally.
This particular scalar scan has no reset/restore hooks to preserve. Do not patch
global triton.autotune or broadly clear caches. Keep other kernels' selected
configurations fixed across the four variants.

Prefer an output Tensor.register_hook to a module full_backward_pre_hook for
capturing the layer46 upstream cotangent. Module backward hooks introduce view
wrappers; a tensor hook avoids that extra intervention. Capture the original
forward hidden input in RAM, attach a hook to the GDN tensor output, clone its
received cotangent, and raise a private sentinel before GDN backward executes.
Guard against checkpoint recomputation overwriting captures or attaching hooks
twice. Remove all handles, clear partial parameter gradients, and restore staging
state in finally. Replay with fresh detached leaves, unchanged parameter values,
the original autocast settings, fixed cotangent, and no outer checkpoint. Verify
identical replay forward outputs. Raw activations/cotangents/candidate gradients
remain in RAM only.

No historical selected-config, cubin, PTX, or Triton-cache artifact was found in
the reviewed architecture/train capture directories. Thus a matching candidate
config would diagnose compatibility with the archived gradients, not prove which
configuration the original server selected without additional evidence.

## Collected isolation result: reverse-scan warp choice is causal

Independent review of local attempt `20261010035123635-8cbdba35` examined
`output/result.json`, `kernel-identity.json`, and every `gdn-replays.json` row.
The diagnostic completed six variants with no optimizer updates or persisted raw
candidate gradients. The native scalar loss remains 0.6244627833366394 and every
isolated forward output is bitwise identical.

| Reverse-scan variant | Exact layer46 GDN gradients vs original | Relative L2 |
| --- | --- | --- |
| Default replay1 | 8/10 | 0.0000075453488670376175 |
| Default replay2 | 8/10 | 0.0000075453488670376175 |
| Existing1warp config | 10/10 | 0 |
| Existing2warp config | 8/10 | 0.0000075453488670376175 |
| Existing4warp config | 8/10 | 0.0000075453488670376175 |
| Existing8warp config | 8/10 | 0.0000075453488670376175 |

All10 reference gradients are nonzero. Both default replays are mutually exact;
2/4/8warp gradients also match the default exactly. Default execution selected
2warps,1CTA,3stages for cache key
`(1, 48, 64, False, True, 'torch.float32', 'torch.float32')`.
The1warp candidate retains1CTA and3stages. The scalar cumsum source SHA256 is
`0405701c46cee331088bfee395b3bf37f8384829dace0aa3a55a509844bcf4cb`.

This intervention isolates reverse-scan launch configuration as a cause of the
observed layer46 in_proj_a mismatch: changing that configuration alone restores
both affected gradients while inputs, upstream cotangent, parameters, and forward
outputs remain fixed. This is stronger than the earlier localization hypothesis.
It does not independently prove the original server selected1warp, since its
historical autotune cache was not captured, and it does not yet prove that this
pin restores every earlier layer or the full-model gradient archive.

### Cache pin review

For the next native qualification, prepopulating the existing autotuner cache
entry with the existing1warp Config avoids changing either package files or
kernel callables. The following checks bound this intervention:

- Guard kernel source SHA, wrapper types, and key schema. Use the observed exact
  tuple and preserve the Config's other fields (1CTA,3stages).
- Leave the REVERSE=False forward entry unchanged. Do not clear or replace other
  autotuner caches.
- Require FLA_CACHE_MODE not ALWAYS. The reviewed CachedAutotuner.run reloads its
  config file in ALWAYS mode even for present keys and could overwrite the pin.
  Other reviewed modes check a prepopulated key before loading a fallback.
- Record the installed entry before execution and confirm it remains1warp after
  real backward. Record the explicit pin in provenance; package SHA alone does
  not describe this runtime choice.
- Different batch/head/chunk/dtype/variable-length keys are outside this isolated
  result. Do not claim a generic pin or numerical qualification for them.
- Run the full unchanged native loss/all74472-gradient equality gate against the
  original archive. Do not promote a new reference or relax equality if another
  residual discrepancy remains.

At this report update, the main agent plans that full native replay. No full-model
acceptance is claimed here.
