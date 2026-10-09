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

Status at implementation: capture and comparison jobs started; results pending.
Neither this implementation nor the prior capacity runs certify full-context
gradient equivalence. A 16K comparison cannot directly prove equality at 130K;
full-context execution is a subsequent capacity and boundary-behavior check.
