# Astra review: Opt9 hyperconnection gate failure

Reviewed 2026-10-10. Read-only source and collected statistics analysis; this report
is the only written file. No GPU runs, private connection access, or implementation
changes. Scope: attempt20261010052326065-031fe094, execution4200f93.

## Gate and observed pattern

Opt9 fails the revised user2% gate: paired global gradient relative L2
2.8541980232629447 (285.419802%), cosine0.2658505449298857. Candidate loss is
0.624601423740387 versus control0.6243785619735718. All74472 initial adapters
match. Of74472 gradient tensors,54 are exact; none of74394 nonzero reference
gradients is exact. Candidate gradient norm0.7528458055069076 versus reference
0.255001348353928 shows a substantial magnitude increase, not just a small
rotation. Opt7 and Opt8's previously accepted approximately1.8% results do not
qualify this change. Original and revised gates must not be conflated.

Layer47 already differs: MLP relative L2 0.1252129085961299 and self attention
0.14533613892769387. Whole-layer errors and candidate/reference norm ratios grow
backward:

| Layer | Relative L2 | Norm ratio |
| --- | --- | --- |
| 47 | 0.129168 | 1.003020 |
| 40 | 0.247583 | 1.007978 |
| 36 | 0.333716 | 0.991621 |
| 32 | 1.320090 | 1.573884 |
| 24 | 1.405724 | 1.664298 |
| 16 | 3.391497 | 3.477791 |
| 8 | 5.190913 | 5.172041 |
| 0 | 4.248797 | 4.179868 |

There is no exact top-layer boundary identifying one backward operator. These
statistics combine forward trajectory differences with local backward changes.
They cannot establish whether a graph bug or shape-dependent arithmetic caused
the amplification.

## Source comparison

Compared components/hyperconnection_chunks.py to precision.py and pinned native
modeling_qwen4_exp.py SHA256
0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206.

- Native GatedResidual computes normalized stream mixing and injection coefficients,
  returning mixed_input, original hyper_input, and coefficients. ChunkedResidual
  calls the same BF16Residual arithmetic per token window, keeps mixed_input and
  coefficients, and supplies the original BF16 x as residual. No missing residual
  branch or coefficient detach was found in lines9–22.
- Native BF16Residual preserves coefficient dtype via activation_role=residual.
  The chunk implementation preserves that third output dtype as well. It does
  not unconditionally BF16-cast coefficients. Observe actual dtypes before proposing
  any coefficient precision change.
- Native attention injection is cast by the next BF16Residual input boundary;
  native final MLP injection is cast by BF16Decoder's output boundary. The explicit
  bfloat16() in inject at line27 is at the same effective value boundary. Moving
  or removing this cast without evidence is not a justified repair.
- Decoder PLE, attention, and MLP ordering matches native. Non-reentrant nested
  checkpoint calls are bound directly to methods/functions and current tensor
  slices; there is no deferred loop closure capturing the final slice. Residual
  checkpoints do not directly use expert staging state. Source inspection alone
  finds no demonstrated checkpoint/offload lifetime error.
- Real projection GEMM row counts change from16249 to8192/8057 in each residual
  mixer, including the final stream mixer. That is a concrete arithmetic change
  even with unchanged formulas. It must be isolated on actual model values before
  attributing the full error to BF16.

## Fixture gap

The existing test_hyperconnection_mixing_and_injection_gradients fixture uses
hidden16, hc_count4, hc_lowrank8, BF16-cast frozen weights, and at most23 tokens.
Its independent random block probes injection and coefficient-gradient paths,
but it does not connect mixed output through the contextual block, does not
exercise actual16249-row GEMMs or real weight dtypes, and does not nest the
complete decoder graph under the outer layer checkpoint/save_on_cpu combination.
A passing small fixture therefore does not exclude the observed full-model
failure. Do not replace it with a broad unrelated suite; add an actual-value
isolation at the failure boundary.

## Smallest justified next action

No source-level defect is proven yet, so there is no evidence-backed arithmetic
fix to recommend before an operator diagnostic. Use one actual model forward to
shadow native and chunked residual modules on identical inputs, comparing mixed
output, residual output, and injection coefficients separately. Include final
stream mixing. Stop at the first mismatch and retain only that input/weights in
RAM; record real shapes, dtypes, strides, and max/relative differences.

On the retained module, compare fixed-cotangent VJPs with BOTH mixed-output and
coefficient-output cotangents nonzero, plus the direct residual cotangent. Use
native/native-repeat, chunked without inner checkpoint, and current chunked
checkpoint versions. Keep casts unchanged. Execute once without saved-tensor
offload and once with save_on_cpu under an outer non-reentrant checkpoint if the
plain VJP passes. This isolates numerical row-shape changes from recomputation
or saved-tensor handling without another full-model backward.

Test injection independently on captured block output/residual/coefficients with
one fixed cotangent, matching native effective output BF16 cast. If mixing and
injection individually pass but composed gradients fail, use a single retained
actual decoder input with unchanged contextual blocks to compare the two
composed graphs. Do not begin with a whole-model rewrite.

If the first mismatch is a mixing projection GEMM, the smallest diagnostic
fallback is to keep those projection/norm calls at native full sequence shape
and chunk only injection. This is a bounded isolation choice, not an accepted
implementation or an assertion that memory goals are met. Conversely, if only
nested checkpoint replay differs, retain arithmetic and isolate/remove that
inner checkpoint before changing projection precision.

The eventual fix still requires cumulative Opt9 paired qualification below2%
against unchanged Opt3. Test-only FLA numerical configuration must remain absent
from real training. No acceptance or production promotion follows from this
review.

## Pre-dispatch review of hyperconnection_replay.py

Read-only review of the prepared diagnostic and launcher whitelist found no
blocking issue for BF16 captured inputs. Native shadow execution disables only
the target module's chunk size and guards recursive hooks, then restores it in
finally. Cotangents independently exercise mixed output, direct residual, and
injection-coefficient output. Frozen mixer weights are asserted. Native-repeat
checks both output ports and input VJP; each variant builds a fresh graph.
Projection hooks are removed before checkpoint recomputation, avoiding duplicate
trace collection. Function closures are consumed synchronously before advancing
the variant. Outer variants combine non-reentrant checkpointing and CPU saved
tensors. The whitelist addition is restricted to the explicit test entrypoint;
no raw candidate tensor writes or full-model backward/optimizer were introduced.

One concrete conditional correction is needed for full dtype fidelity:
without_checkpoint() omits the leading x.bfloat16() in ChunkedResidual.forward.
For FP32 captured input this changes the direct residual port and its cast path,
confounding checkpoint isolation. Add x=x.bfloat16() inside that helper or assert
that the captured input is BF16 before all variants. Other variants use the
actual module boundary. This is not evidence that the Opt9 implementation itself
has that bug; it is a possible discrepancy in the diagnostic variant.

This diagnostic isolates mixers; injection and composed decoder VJPs remain a
follow-up only if mixer results leave the failure unexplained. Passing local
checks would not qualify Opt9 or relax the2% full paired gate. Main agent handles
GPU dispatch; this review ran none.
