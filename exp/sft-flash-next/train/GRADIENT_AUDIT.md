# Historical custom-reference gradient audit

The active HF/PEFT first-pass capture and flag comparisons are documented in
[NATIVE_GRADIENT_AUDIT.md](NATIVE_GRADIENT_AUDIT.md). The loader and model substitutions
below are not the accepted reference.

**Status: custom-reference audit stopped; native reference passed its learning qualification.**
The current requested prerequisite is a [single-sample overfit test](OVERFIT_REFERENCE.md):
reduce a fixed real 16K-window loss to at most 5% of its initial value. The
user permits RAM placement and does not require PEFT, though the current
recipe uses HF, official AutoRound and PEFT. A successful overfit demonstrates
learning, not gradient equivalence. No native-HF gradient-equivalence result
is claimed.

The older loader/indexer substitutions below are retained as experimental
diagnostics and are not the accepted independent reference.
[native_hf_reference.py](native_hf_reference.py) describes a one-backward HF
reference; [overfit_hf_reference.py](overfit_hf_reference.py) implements the
new learning qualification. AutoRound is available in an isolated package
view, and a byte-preserving 256-expert checkpoint has been exported from the
512-expert source. An [operator test](QUANTIZATION_COMPATIBILITY.md) demonstrated
input gradients through an actual frozen AutoRound expert. HF's
`is_trainable=False` integration flag alone does not establish absence of
those gradients. The full model subsequently [passed the single-sample overfit criterion](OVERFIT_REFERENCE.md)
after 5 updates. Gradient equivalence remains outstanding.

The 130K capacity runs demonstrate that full forward, backward and an optimizer
step fit. They do **not** establish equivalence to the unoptimized model's
gradients. The earlier 512-token same-forward VJP diagnostic substitutes only
the attention backward; it cannot validate every optimization, or their effect
on expert routing. A native-versus-optimized comparison previously differed by
about 40%. That discrepancy must not be hidden by holding routing fixed.

The experimental `gradient_audit.py` runner is designed to compare the pruned 256-expert model on one intact
real request, first changing one optimization at a time and then enabling the
production combinations. Both LoRA matrices are initialized nonzero, and every
case reuses the same adapters, inputs, positions, targets and RNG state. There
are no optimizer updates and no detached recurrent states or truncated input.

## Reference and limits

The reference uses the existing W4A16 checkpoint loader and the repository's
pre-existing vectorized QSA indexer. HF supplies native RMS normalization,
hyper-connections, PLE and GDN. Frozen expert weights are independently
dequantized and used with ordinary PyTorch autograd. Attention uses PyTorch
SDPA with one full masked call per attention layer and the HF repeated-KV
convention, with ordinary autograd; loss is ordinary target cross entropy.

This is **not untouched stock Transformers**: the quantized checkpoint requires
the loader, and the stock token-by-token indexer is prohibitively slow at this
length. The main reference does not tile attention queries. The first case disables
layer checkpointing and records any OOM. The canonical reference uses stock
non-reentrant layer checkpointing. This shared memory control must be stated
when interpreting the results. Separate gathered-attention diagnostics measure
the effect of query tiling and its BF16 gradient accumulation.

The selected request `sk48-d8078629-r85` contains 33,191 prompt tokens, 365 target
tokens and 22 images: **33,556 total tokens**, slightly over 32K so that the
production 32,768-token projection and whole-GDN blocks cross a boundary. The
run records the prepared-manifest, sample, keep-map and source-file hashes.

For each completed case the audit records loss, elapsed time, GPU memory, all
trainable gradients' relative L2 errors, maximum absolute errors, cosine
similarities, and bitwise equality. Full MoE routes and selected boundary QSA
queries are compared without forcing agreement. Repeated reference runs measure
ordinary numerical variation. BF16 and FP32 SDPA references are explicitly
separate cases; the latter is a precision sensitivity check, not a claim of
bitwise native equivalence.

A finite backward is not a gradient-equivalence pass. There is no automatic
blanket 5% tolerance in this audit. Per-parameter errors and reference-repeat
variation matter alongside the aggregate norm.

## Reproduce

Use the qualified Kaggle environment and the same prepared request bundle and
256-expert keep map as the capacity experiment:

```bash
python prepare_audit_sample.py \
  --bundle /path/to/prepared-data --sample-id sk48-d8078629-r85 \
  --out /path/to/audit-input

PYTORCH_ALLOC_CONF=expandable_segments:True python -u gradient_audit.py \
  --model /path/to/intel-source-512 --keep artifacts/keep-256-fold0.json \
  --sample /path/to/audit-input/sample.pt \
  --sample-metadata /path/to/audit-input/sample.json \
  --out /path/to/fresh-audit-results
```

The output directory must be new. `--cases` can supply an explicit JSON list of
configurations; include a case named `reference` before cases to compare.
`cases.json` and `identity.json` retain the actual experiment definition.
Detailed per-parameter reports accompany `results.json`; selected full gradient
snapshots are retained for reanalysis. Cold PLE reads can dominate the first
pass, so storage warm-up must be recorded separately from GPU timing.
