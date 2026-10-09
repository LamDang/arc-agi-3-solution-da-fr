# Native HF adapter-gradient verification

The accepted learning reference is committed at `9d12e32`. The new
`--first-pass-only` mode in `overfit_hf_reference.py` records all raw adapter
gradients after the first backward, before clipping or any optimizer update.
It loads the exact initial adapter checkpoint from the learning experiment.
The same 16,249-token diagnostic input and 651 supervised targets are used.
The reference keeps stock HF layer checkpointing and PyTorch `save_on_cpu`;
no optimized model operator is imported by the capture script.

```bash
python overfit_hf_reference.py --model /tmp/reference-256-hf \
  --sample /path/to/overfit-sample-16k/sample.pt --prompt-tokens 15598 \
  --adapter-state /path/to/overfit-hf-v3/initial-adapter.pt \
  --checkpointing --save-on-cpu --first-pass-only --out /path/to/native-first-pass
python native_gradient_audit.py --model /tmp/reference-256-hf \
  --sample /path/to/overfit-sample-16k/sample.pt --prompt-tokens 15598 \
  --adapter-state /path/to/overfit-hf-v3/initial-adapter.pt \
  --reference /path/to/native-first-pass --out /path/to/native-flag-audit
```

`gradients.pt` stores every named tensor, including zero gradients.
`gradient-summary.json` records shapes, dtypes, norms, maxima, finite status and
nonzero counts. Initialization has zero B matrices, so the A gradients are
necessarily zero. A second audit using a fixed nonzero adapter checkpoint is
needed to exercise both matrices before accepting the complete recipe.

The flag runner loads the same model with HF, official AutoRound and PEFT.
It never invokes the old custom model loader or custom LoRA implementation.
Every case uses unchanged adapter bytes, input, labels and RNG state. Model
patches are restored after each case, including failures. No optimizer runs.
The first case replays native forward/backward without a model substitution.
If that replay fails, the runner stops rather than certifying later flags.

Each optimization runs separately, then combines with previously accepted
flags. A combination must independently match the original reference.
The comparison checks exact key sets, shapes and dtypes; nonfinite values fail.
Every tensor must satisfy the declared `rtol=1e-5, atol=1e-8`. Zero reference
gradients must stay exactly zero. This numerical gate is **not a claim of
bitwise identity**; that is reported separately. It also requires matching
loss and unchanged MoE routing hashes. Thresholds are never loosened
implicitly when a case fails. Full gradient tensors and per-parameter error
reports are retained even for a failed comparison.

Initial flags cover target-only native logits, checkpointed loss blocks,
selective CPU activation storage, disk storage with prefetch, RMSNorm,
hyperconnections, gated normalization, PLE windows, GDN blocks and sparse
attention. Any replacement that changes gradients is rejected pending a fix.
The old custom expert kernels and embedding preparation are not silently
enabled as prerequisites. Block boundaries are set below the audit length so
that tests exercise cross-block behavior.

The first capture completed in 535.63 seconds including model load. Its native
loss is exactly **0.6247151494026184**, reproducing the learning baseline.
All **744 tensors / 33,478,656 values** are finite: 372 zero A gradients and
372 nonzero B gradients. The downloaded tensor file matches the server hash:
`4562ea2ec914c1b10d51d001eb01d4e07a4e2eebaf66884b37e8a40ba9bc85e6`.
Per-adapter statistics and capture metadata are checked into
[`gradient-results/native-first-pass`](gradient-results/native-first-pass).
Raw tensors are retained in the shared workspace at
`/workspace/quant-compat-audit/native-gradient-audit/native-first-pass/gradients.pt`
and on Kaggle at `gradient-audit-20261009/native-first-pass/gradients.pt`.

The unchanged native replay **failed** the strict numerical gate before any
optimization ran: relative L2 error **0.01126662097 (1.1267%)**, with **366 of
744 tensors** outside tolerance. Both losses are exactly `0.6247151494026184`.
All zero A gradients agree; only six B gradients are bitwise equal, all in the
last layer. Differences generally grow towards earlier layers. Full per-tensor
reports are in [`gradient-results/native-repeat-default`](gradient-results/native-repeat-default).
The raw `routes_equal: true` field is the replay's self-baseline for subsequent
flags, **not** a comparison against the original capture, which saved no routes.
The runner now reports this unavailable comparison as null.

No optimization was certified or run after the failed native replay. The unchanged-model deterministic two-pass diagnostic **passed bitwise**:
all 744 tensors are identical, relative L2 error is exactly zero, and both
losses are `0.6247151494026184`. It used
`torch.use_deterministic_algorithms(True)` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, without replacing model operators.
Results are in [`gradient-results/native-repeat-deterministic`](gradient-results/native-repeat-deterministic).
The flag sweep uses this deterministic baseline and settings, and first checks
another native replay across processes. Default and deterministic gradients
are not mixed in a comparison.
Neither this implementation nor the prior capacity runs certify full-context
gradient equivalence. A 16K comparison cannot directly prove equality at 130K;
full-context execution is a subsequent capacity and boundary-behavior check.

## Reference reproducibility diagnostics

The original learning run recorded first-step gradient norm `0.8303273916`;
the new saved capture measures `0.8302070836` in FP64. Loss and initial adapters
match, but a norm alone cannot locate or quantify per-element differences.
The unchanged replay must establish reproducibility before accepting flags.

The capture runner also supports `--gradient-repeats 2` to run multiple native
forward/backward passes without any update, retaining every gradient snapshot.
`--deterministic` explicitly requests deterministic PyTorch operators (launch
with `CUBLAS_WORKSPACE_CONFIG=:4096:8`). This is a diagnostic option, not a
promise that third-party CUDA kernels honor PyTorch's determinism setting.
It does not silently replace any native model operator. Deterministic and
default runs cannot be mixed as a baseline/candidate pair.

The first audit attempt stopped after loading because HF loading diagnostics
contained Python sets. No forward or comparison ran. The report serializer
now handles these sets, and uncaught errors produce terminal failure artifacts.
The corrected run is `native-flag-audit-v2`.

## Additional native memory candidates

`native_mask_storage` keeps the HF query-by-query QSA selection arithmetic and
native attention module/SDPA call. For an unpadded cache-free request it supplies
causal rows lazily to the indexer and writes its final additive attention bias
directly. It removes redundant dense boolean mask allocations; it does not
change the selected keys, detach context, or substitute a custom attention
backward. The storage-only source transformation is version-guarded and fails
if the expected native code structure changes. Small native-model loss and
gradients are bitwise identical; real-model qualification remains necessary.

`checkpoint_group=3` wraps groups of unchanged HF layers in an outer standard
non-reentrant checkpoint. Original inner layer checkpoints remain enabled.
This reduces persistent activation boundaries in host RAM, and preserves the
original layer modules and adapter names after each case.

The fresh-process deterministic replay also produced **bitwise-identical 744
adapter gradients**, saved before reporting. Its result writer then failed:
a loop checking unchanged adapter weights shadowed the scalar loss with a
tensor. The loop is now in its own helper, and reporting asserts a scalar
loss. The failed attempt and its saved comparison remain intact.
`--verified-replay` can reuse that measurement only after checking its raw
tensors against the reference, its loss, sample/adapter identity, deterministic
settings, model path and operator-source hashes. Reused measurements are
explicitly labeled with their source; they are not represented as new runs.
The resumed sweep is `native-flag-audit-deterministic-v3`.

### Measured flags in the resumed sweep

These measurements use the 16,249-token real sample (15,598 prompt tokens,
651 supervised tokens), the unchanged initial adapter, and the deterministic
native reference. They are not full-context capacity results.

| Flag | Gradient relative L2 | Failing tensors | Peak GPU allocation | Step time | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Target-only logits | 0.01216609 | 372 / 744 | 58.52 GiB | 331.35 s | Rejected |
| Chunked loss, 128 tokens | 0.01195724 | 372 / 744 | 58.52 GiB | 323.05 s | Rejected |
| Native mask storage | 0 | 0 / 744 | 84.89 GiB | 330.75 s | Bitwise pass |
| Selective CPU offload | 0 | 0 / 744 | 85.13 GiB | 325.94 s | Bitwise pass |
| Native mask storage + selective CPU offload | 0 | 0 / 744 | 84.89 GiB | 329.91 s | Bitwise pass |

Target-only logits retains the complete decoder context, but computes the
vocabulary projection only at positions with supervised next-token labels.
The vocabulary has 248,320 entries: a full BF16 logits tensor occupies about
7.52 GiB for this sample, versus 0.30 GiB for the supervised positions. Native
cross-entropy also creates FP32 tensors, so avoiding ignored logits saves
considerably more than the BF16 tensor alone. The measured memory saving does
not qualify the optimization: all 372 nonzero B gradients fail the strict
gate. The 372 A gradients are exactly zero at initialization. The source of
the numerical discrepancy is still under investigation.

The mask-storage and selective CPU-offload cases, individually and combined,
preserve every adapter gradient bit for bit. All five
cases match the native replay's **MoE routing hashes**; these hashes do not
independently compare QSA-selected attention indices. Before a recipe is
qualified for continued training, it also needs a native-reference comparison
using a saved trained adapter with nonzero A and B matrices. Disk offload and
checkpoint candidates remain unmeasured in this sweep. Compact measurement
reports are saved in [`gradient-results/native-flags-initial`](gradient-results/native-flags-initial).

The sweep was deliberately stopped after saving the combined mask/CPU result,
before the next case started, to prioritize a standalone native LM-head/loss
gradient diagnostic. All six completed reports (including the native replay)
and their raw gradients were preserved. The interrupted sweep has no final
`accepted-flags.json`; no completed qualification artifact was fabricated.

## Isolated diagnosis of the selected-logit gradient mismatch

The standalone native head diagnostic uses the actual frozen BF16 head, the
real sample's labels, and saved **synthetic** hidden states with RMS near one.
It uses the same deterministic settings and library versions as the reference.
Full and selected target logits match bitwise, as do their native CE gradients.
Only the multiplication taking vocabulary-logit gradients back to hidden-state
gradients differs: changing its row count from 16,249 to 651 gives relative L2
error **0.00445865**. With identical input gradients, restoring the original
matrix shape reproduces the native autograd result bit for bit. Parent review
also independently compared the downloaded raw tensors.

Padding is **not** a monotonic rule: 1,024, 4,096 and 8,192 rows match the native
16K head result in this test, while 2,048 rows fail. The same native row count
also matches with targets placed at row zero, so their original row offset is
not necessary for the observed equality. These are measured kernel behaviors,
not guarantees for arbitrary inputs, shapes, hardware or library versions.
An additional 16,384-row test also matches the original 16K/651-target result
bitwise, both at row zero and with the original row alignment modulo 128.

A second synthetic operator test tiles and scales the captured target-gradient
rows to 10,000 targets. Both full 90K and 130K backward shapes produce identical
target hidden gradients. Explicit 16,384-row padding matches both bitwise, with
measured peak allocation **13.42 GiB**, versus **65.97 GiB** for the 130K native
shape. These numbers cover the isolated head test, not the loaded whole model.
They do not establish full-length model gradient parity or training capacity.

Exact diagnostic sources and reports:

- [`native-head-vjp-diagnostic-v1`](gradient-results/native-head-vjp-diagnostic-v1)
- [`native-head-padding-diagnostic-v1`](gradient-results/native-head-padding-diagnostic-v1)
- [`native-head-long-shape-diagnostic-v1`](gradient-results/native-head-long-shape-diagnostic-v1)
- [`native-head-padding16384-diagnostic-v1`](gradient-results/native-head-padding16384-diagnostic-v1)

`loss=selected_native_backward` is a new, **unqualified** candidate in
`native_head_loss.py`. It uses native CE autograd to compute target dLogits,
including upstream loss scaling, and then an ordinary `torch.mm`. Its default
restores the original full row count. `head_backward_rows` requests a separate
explicit padded shape and must be at least the number of target positions.
There is no guessed padding heuristic or altered BF16 precision setting.
It requires a frozen vocabulary head and ignores only the declared prompt.
The new helper is included in the operator-source qualification hashes.
The candidate must pass actual all-adapter comparisons, including a trained
adapter, before promotion.

The implemented custom autograd function also passed an isolated GPU test with
stock `save_on_cpu` and BF16 autocast: native loss and hidden gradients match
bitwise both with the original row count and with 16,384 rows. The latter peaks
at 10.39 GiB in that bounded operator test. The tested source hashes match the
checked-in candidate; the report and exact harness are in
[`native-head-candidate-test-v1`](gradient-results/native-head-candidate-test-v1).
Eighteen local checks passed, covering loss scaling, ignored response labels,
frozen-head enforcement, native decoder gradients, and qualification guards.

The real-model audit `native-head-gradient-audit-initial-v1` completed with
[`native_head_cases.json`](native_head_cases.json): fresh native replay, the
16,384-row loss candidate alone, and that candidate with mask storage plus CPU
offload. It compares all 744 tensors to the original deterministic reference,
with no optimizer updates. The native replay **passed bitwise across all 744
tensors**, independently verified from the downloaded raw gradients. Both
candidate recipes stopped **before backward** at the implementation guard
`Head and hidden states must have matching dtype and width`. No candidate
gradients were produced, and `accepted-flags.json` is empty. The conditional
trained-adapter reference correctly refused to launch. Failure reports are in
[`native-head-gradient-audit-initial-v1`](gradient-results/native-head-gradient-audit-initial-v1).

The candidate had incorrectly assumed that hidden states, head weights and
logits share one dtype. Native autocast can accept FP32 hidden states and BF16
weights, produce BF16 logits, and cast the hidden gradient back to FP32. The
original failure did not log the operands' dtypes, so the actual boundary still
needs observation. A read-only native-head pre-hook now records shapes, dtypes
and autocast settings on the next replay.

The revised helper lets native `F.linear` perform autocast, saves the head
weight in the actual logits dtype for backward, and casts the resulting hidden
gradient back to the original hidden dtype. New CPU tests verify bitwise hidden
gradients for FP32 inputs with either FP32 or BF16 head weights under BF16
autocast. CPU scalar CE can differ by one FP32 ULP when ignored rows are omitted;
this local scalar check does not relax the real-model gradient gate.
The dtype fix **passed an isolated mixed-dtype GPU gate** on the restored
RTX PRO 6000 server. Fresh native full-head loss and input gradients were
computed separately for raw FP32 hidden values and saved BF16-rounded values
cast to FP32, with upstream scales 1 and 0.125. Both native-length and explicit
16,384-row candidate backwards match loss and every FP32 gradient value
bitwise in all eight comparisons. Actual frozen-head tensor and runtime/math
settings match the prior diagnostic. Peak allocation was 46.50 GiB for the
native head and 10.40–10.47 GiB for the candidates. All raw gradients were
downloaded and independently rechecked locally; the exact harness, report and
artifact hashes are in
[`native-head-mixed-dtype-v1`](gradient-results/native-head-mixed-dtype-v1).
This synthetic head test does not qualify the complete model or establish its
actual hidden dtype.

The original server expired with HTTP 404. On the replacement server, 50
restoration files were SHA-verified, the exact Transformers 5.18/FLA 0.5.2/
AutoRound 0.15/causal-conv1d 1.7 dependencies restored, and the successful mixed
head test completed. A byte-preserving 256-expert export was launched, followed
by a sequential watcher for the fresh native replay and two candidate cases.
Subsequent `/api/kernels` and saved-report route checks returned HTTP 503,
then recovered after bounded retries. The export completed (14 rewritten
shards, 24 linked files) and the watcher verified its configuration hash and
native runtime. The queued `native-head-gradient-audit-initial-v2` then stopped
before forward because the restored training backend was missing its unchanged
`exp/reap-flash-next/reap_model.py` import dependency. No gradient comparison
occurred. The last observed GPU state was idle. This environment failure is
separate from numerical qualification; candidate all-adapter gates remain
pending after dependency restoration.

The user-approved all-assistant trajectory objective is a separate,
GPU-unqualified pipeline. It leaves this fixed final-reply reference unchanged.

## Full-context capacity gate

`native_full_context_probe.py` refuses an unaccepted recipe, changed operator
sources or a different adapter checkpoint. It loads the native HF/AutoRound
model with PEFT, applies only the accepted flags, and executes the complete
input without truncation. `--optimizer-step` adds one disposable AdamW update.
Its capacity result explicitly does not claim full-length gradient equivalence.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python native_full_context_probe.py \
  --model /tmp/reference-256-hf --sample /path/to/real-capacity.pt \
  --prompt-tokens 120000 --adapter-state /path/to/initial-adapter.pt \
  --qualification /path/to/completed-gradient-audit \
  --optimizer-step --out /path/to/new-130k-result
```

The corresponding 90K comparison uses the existing 80K-context plus 10K-target
composite and `--prompt-tokens 80000`. These are capacity composites assembled
from real requests, not coherent teacher trajectories or production examples.
