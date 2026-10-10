# Astra review: reference restoration identity

Reviewed 2026-10-10. Scope: repository and local evidence inspection; no GPU runs,
network calls, or model changes. This report is the only file written by this review.

## Finding

The replacement-server export used the smoke expert-selection map instead of the
documented fold0 map. The maps differ in all 48 layers. This is a confirmed model
identity mismatch and the leading explanation for replacement native loss
0.6362957954406738 versus saved reference loss 0.6244627833366394. The review did
not independently execute either capture, and restoration success still requires
verification and a native replay.

The main agent reported that all 74,472 initial adapter tensors and the model
configuration/native/AutoRound source identities matched. These checks do not
cover the selected frozen experts or router rows. An unchanged 256-expert config
can describe either selection. Reusing the same adapter names also assigns the
same adapters to different underlying source experts after renumbering.

## Map identities and provenance

| Map | SHA256 |
| --- | --- |
| `exp/sft-flash-next/train/artifacts/keep-256-fold0.json` | `c76c2734c30fabcde2a44fa44467f93ae50ade945068a11e4befb5246bd5fca6` |
| `exp/reap-flash-next/kaggle/keep_256_smoke.json` | `6ca9112852eebd4b2176fd5f1ce671ba26f6b34e77e952c7702ed848098eaf52` |

`train/OVERFIT_REFERENCE.md:141–145` explicitly prepares `/tmp/reference-256-hf`
with `--keep-file artifacts/keep-256-fold0.json --keep 256`.
`train/artifacts/provenance.json` pins that map to archive member
`maps/keep-256.json` in `data/sol-nll-fold0-30-results-20261008/results.zip`,
DVC MD5 `4096658b60e9a324a67d205f9e28545c`. Its SHA256 matches the local map.
`train/README.md:293` also identifies this as the verified training-only map.
The smoke map instead cites smoke statistics from `kaggle_v3/sk48_p0` and
`openrouter/ft09_p0`.

`exp/reap-flash-next/prune_checkpoint.py:85–90` renumbers the retained source IDs
in ascending order; lines 115–125 select expert tensor bytes and matching router
rows. For example, layer0 output expert4 maps to original expert10 under fold0,
but original expert9 under smoke. Thus the mismatch affects both packed expert
projections and routing.

## All 48 layer differences

Each map retains 256 experts per layer. “Replaced” counts selected IDs present in
one map but absent from the other; the symmetric difference is twice this count.
These counts do not measure renumbered-position mismatches, which can be larger.

| Layer | Shared selected IDs | Replaced IDs per map |
| --- | --- | --- |
| 0 | 241 | 15 |
| 1 | 248 | 8 |
| 2 | 243 | 13 |
| 3 | 245 | 11 |
| 4 | 241 | 15 |
| 5 | 242 | 14 |
| 6 | 245 | 11 |
| 7 | 240 | 16 |
| 8 | 238 | 18 |
| 9 | 238 | 18 |
| 10 | 243 | 13 |
| 11 | 243 | 13 |
| 12 | 239 | 17 |
| 13 | 239 | 17 |
| 14 | 241 | 15 |
| 15 | 227 | 29 |
| 16 | 235 | 21 |
| 17 | 240 | 16 |
| 18 | 244 | 12 |
| 19 | 239 | 17 |
| 20 | 243 | 13 |
| 21 | 245 | 11 |
| 22 | 246 | 10 |
| 23 | 244 | 12 |
| 24 | 246 | 10 |
| 25 | 242 | 14 |
| 26 | 244 | 12 |
| 27 | 244 | 12 |
| 28 | 245 | 11 |
| 29 | 247 | 9 |
| 30 | 241 | 15 |
| 31 | 246 | 10 |
| 32 | 243 | 13 |
| 33 | 249 | 7 |
| 34 | 245 | 11 |
| 35 | 244 | 12 |
| 36 | 250 | 6 |
| 37 | 248 | 8 |
| 38 | 249 | 7 |
| 39 | 243 | 13 |
| 40 | 241 | 15 |
| 41 | 248 | 8 |
| 42 | 249 | 7 |
| 43 | 247 | 9 |
| 44 | 245 | 11 |
| 45 | 249 | 7 |
| 46 | 248 | 8 |
| 47 | 244 | 12 |

## Minimal identity checks before another qualification

1. Compare the restored export's `keep.json["kept"]` with the fold0 map for every
   layer. Verify the uploaded map's exact SHA256 above.
2. Read safetensors bytes for layer0 output expert4 and router row4; compare them
   with source expert10 and source router row10. Include `qweight`, `qzeros`, and
   `scales` for gate/up/down. This is a small CPU file-read check that immediately
   distinguishes the two selections. For complete export verification, extend
   the byte checks across all selected expert tensors and router rows.
3. If useful, reconstruct the expected export index using only the source index
   and safetensors headers. Reproduce `prune_checkpoint.py:98–155`: sorted shard
   traversal, header data-offset order, retained-key renaming, unchanged-key
   insertion order, metadata, `indent=2`, and trailing newline. Compare the JSON
   SHA256 with the archived value. This does not require rewriting tensor data.
4. After the corrected export passes identity checks, rerun unchanged native
   reference qualification before resuming Opt7. Keep the failed restoration
   evidence separate; it cannot qualify numerical drift for the intended model.

The main agent supplied archived index SHA256
`b9dcb19699ef52c057031be4b8b6fed5697b23b641d0ccf0a8a59b1d05d241d6`
and replacement index SHA256
`54d88a5332d1dfc18b4422e3702352661b84e399fbe25cbece0ab745e6ed1565`.
An index hash mismatch alone is not proof of tensor mismatch: `pruned_from`
metadata contains the source path, and JSON ordering also affects the bytes.
Conversely, matching index hashes do not prove tensor contents. The explicit
map comparison establishes the selection mismatch independently.

## Remediation status and limits

At report creation, the main agent confirmed the wrong-selection diagnosis,
uploaded the fold0 map, and started a separate `/tmp/reference-256-fold0-hf`
export. Tensor-byte verification and native replay remain the main agent's work.
This review does not claim the corrected export or replay has passed. There is
currently no need to attribute this mismatch to the newly built convolution
extension; investigate runtime differences only if the intended frozen model
identity is restored and the native replay still differs.
