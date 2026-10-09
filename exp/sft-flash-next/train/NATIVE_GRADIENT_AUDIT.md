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
