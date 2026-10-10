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

GPU validation of these additions is pending runtime restoration. Local suite:
28 discovered,11 passed,17 skipped because this desktop lacks Torch/the CUDA
reference environment. Local checks are not numerical qualification.

The replacement Kaggle notebook356919557 has no old temporary exports, package
view, anchor or reference shards. Restore these and validate source/input pins
before resuming full-model qualification. The old disconnected run remains
uncollected; its forward observation cannot establish backward acceptance.
