# Test-only nonzero adapter for gradient qualification

The zero-B reference at `gradient-results/reference-new-server-one-run-v1` is
preserved. Its loss is `0.6243623495101929`, and its 372 A gradients are zero
by construction. A later comparison must exercise both A and B gradients.

The existing trained `overfit-hf-v3/final-adapter.pt` has all 372 A and 372 B
matrices nonzero. Its SHA256 is
`2d5d80d56aa4113bd2e7326a5b60d0b51f14a6782207fa4db67bfa524f9392b8`.
It was trained on the diagnostic fold-0 sample, so this fixture instead keeps
the exact A matrices from the original saved zero-B anchor and fills B with
small, varied CPU normal values. It never reads the sample or runs an optimizer.

The source anchor is
`/kaggle/working/gradient-audit-20261009/overfit-hf-v3/initial-adapter.pt`,
SHA256 `9d93ddb8676d5490d55fa9969d99352189883a4e53f8d189cc2b9d3bc4574a56`.
The algorithm in `make_nonzero_diagnostic_adapter.py` sorts PEFT tensor names,
copies every A tensor bit for bit, then fills each B tensor with
`torch.randn(shape, generator=g, dtype=torch.float32) * 0.003`, where one CPU
generator `g` is seeded once with `20261010` and consumed in sorted B-name order.
Seed `20261010` is an arbitrary fixed diagnostic
seed, not a date or training seed. It saves all 744 FP32 tensors. Its manifest
records the exact algorithm, seed, torch version, source hash, per-tensor
name/shape/dtype/content hash, canonical tensor-state hash, and saved `.pt`
file hash. **Replay the pinned saved `.pt` bytes**: a seed alone is not an
identity across PyTorch versions or serialization contexts. The canonical
tensor-state hash is independent of `torch.save` container metadata.

The pinned fixture is `gradient-results/nonzero-diagnostic-adapter-v2/adapter.pt`,
SHA256 `49e0960ba1f5a2435e47180262a27e652dd797397a5a11ab13425ef2cf041b6f`.
Its canonical named tensor-state SHA256 is
`111c6c8cb56a5d0d3a5f35a97052d268da976a341d4e412abd5705ea6cb28747`.
The exact executed generator is saved alongside it, SHA256
`0a8d8198a3f33650c77ef1aa870241855fa19de98e46e8d608ce7bb18accbf45`.
Two independent fresh CPU processes produced identical saved file bytes,
canonical hashes, and per-name tensor hashes; both manifests are retained.
The `same_process_regeneration_bitwise_equal` manifest flag separately checks
a second construction within each process. All 372 A and 372 B matrices are
nonzero and finite. CPU tests passed (3), including rejection by the real
production initializer validator for both diagnostic provenance and nonzero B.
The remote test process substituted the unavailable dataset module's identical
SHA256 helper; the validator itself was unchanged.

The fixture has `purpose=diagnostic-gradient-qualification-only`,
`production_eligible=false`, and `validation_exposed=true`. Production
initialization validation rejects this provenance and every nonzero-B state.
It must never initialize production training or unbiased validation.

## Exact fixture and native capture commands

Generate the fixture in a new directory; independently regenerate in another
fresh process and compare canonical tensor-state hashes and all named tensor
bytes. Pin one saved file's SHA256 for subsequent replay.

```bash
python make_nonzero_diagnostic_adapter.py \
  --source /kaggle/working/gradient-audit-20261009/overfit-hf-v3/initial-adapter.pt \
  --expected-source-sha256 9d93ddb8676d5490d55fa9969d99352189883a4e53f8d189cc2b9d3bc4574a56 \
  --seed 20261010 --b-std 0.003 \
  --out /kaggle/working/gradient-audit-20261009/nonzero-diagnostic-adapter-v2
```

The single native reference capture launched on this pinned fixture with the
following effective command. The first launcher attempt exited before model
load because its output directory already existed; its evidence is preserved
under `reference-nonzero-ab-v1`. The corrected launch writes monitor metadata
to a separate directory and uses absent output `reference-nonzero-ab-v2`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python overfit_hf_reference.py \
  --model /tmp/reference-256-hf \
  --sample /kaggle/working/gradient-audit-20261009/overfit-sample-16k/sample.pt \
  --prompt-tokens 15598 \
  --adapter-state /kaggle/working/gradient-audit-20261009/nonzero-diagnostic-adapter-v2/adapter.pt \
  --checkpointing --save-on-cpu --deterministic \
  --first-pass-only --gradient-repeats 1 \
  --out /kaggle/working/gradient-audit-20261009/reference-nonzero-ab-v2
```

The corrected run completed one forward/backward with loss
`0.6256952285766602`, 744 finite raw named gradients, and **372 nonzero A plus
372 nonzero B gradient tensors**. Raw `gradients.pt` SHA256 is
`91465990e81dd57bb29368510354674415c1f1d8bbbaba5225b62fdf4a8cced2`.
The saved initial adapter's named tensors exactly match the pinned fixture.
There was no clipping or optimizer update. Monitor return code was 0 with no
timeout, 606.65 seconds elapsed, sampled peak GPU use 88,475 MiB, peak process
RSS 174,955,556,864 bytes, and peak host used 176,949,022,720 bytes.
`gradient-results/reference-nonzero-ab-v2` contains the raw gradients, per-name
summary, validation, events, launch command, memory telemetry, exact executed
reference source, source hashes, and the preserved pre-load launcher failure.
The executed reference source SHA256 is
`f95893baa503ec446d4610878b7d5feb7ee975e9313e282349cac2d8952573a0`,
matching script commit `c7ff17e78776a9888b280531a0c53718efa45c02`.

## Fresh-process native replay

One authorized replay used the same pinned sample, adapter, reference script,
bootstrap, model config, command flags, and environment; only `--out` changed
to `reference-nonzero-ab-repeat-v1`. It completed without clipping or an
optimizer update in 604.64 seconds (monitor return code 0, no timeout). Both
losses have identical FP32 bits `3f202d90` and value `0.6256952285766602`.
All 744 gradient names, shapes, dtypes, and **raw tensor bytes** match exactly,
including signed-zero bits. Both `gradients.pt` files have SHA256
`91465990e81dd57bb29368510354674415c1f1d8bbbaba5225b62fdf4a8cced2`.
The per-tensor comparison is `gradient-results/reference-nonzero-ab-repeat-v1/bitwise-comparison.json`
(SHA256
`09a480825f5add1381359ad27d3beb7b7399b372ba56f463baa36f8fbe8d505f`),
with no mismatches. The replay's sampled peaks were 88,475 MiB GPU, process
RSS 174,823,469,056 bytes, and host used 177,126,789,120 bytes. Its raw
gradients, command, source hashes, exact reference script, events, and telemetry
are saved alongside the comparison.

Any later candidate audit should point to **that new reference** and **the same
pinned adapter file**. No candidate pass has been launched:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python native_gradient_audit.py \
  --model /tmp/reference-256-hf \
  --sample /kaggle/working/gradient-audit-20261009/overfit-sample-16k/sample.pt \
  --prompt-tokens 15598 \
  --adapter-state /kaggle/working/gradient-audit-20261009/nonzero-diagnostic-adapter-v2/adapter.pt \
  --reference /kaggle/working/gradient-audit-20261009/reference-nonzero-ab-v2 \
  --deterministic --cases /path/to/reviewed-cases.json \
  --out /kaggle/working/gradient-audit-20261009/nonzero-ab-candidate-audit-v1
```

The all-nonzero result is an observation of the saved raw gradients, not an
assumption from nonzero adapter weights. Candidate tests remain pending review.

## Artifact storage

Raw adapter and gradient tensors are DVC artifacts. Git contains the generator,
tests, documentation, `.dvc` pointers and ignore entries; it must not contain
the `.pt` tensors or evidence ZIP files. Restore the diagnostic fixture and
first nonzero reference and its reproducibility replay with:

```bash
dvc pull exp/sft-flash-next/train/gradient-results/nonzero-diagnostic-adapter-v2.dvc \
  exp/sft-flash-next/train/gradient-results/reference-nonzero-ab-v2.dvc \
  exp/sft-flash-next/train/gradient-results/reference-nonzero-ab-repeat-v1.dvc
```

The private configured DVC remote is `s3://kaggle-arc-agi-3-dvc`. The capture
directories include source identities, launch settings, raw telemetry and
independent review reports. The private Kaggle connection URL is excluded.
