Production trajectory preparation is separate from the fixed 16K final-reply
reference. Run `prepare_trajectories.py --input FULL/train.jsonl --generated-dir
FULL/think-progressive-sol25 --processor PINNED/processor --folds data/game_folds/folds.json
--source-run-dir VERIFIED/raw --validation-fold 0 --token-budget 16384 --max-tokens 130000 --out NEW_DIRECTORY`.
The sparse validation panel cannot substitute for the full corpus.

Every game produces one sample containing the complete observed chronological
request stream, reconstructed only through verified history overlap. The required original run logs are re-converted and compared exactly with the
input rows, and original evaluation metadata must confirm game completion. Request indices start at zero
without gaps; changing tools/system/template settings, unexplained assistant
turns, missing generated thoughts, duplicate tool ownership, or unmatched
observations fail preparation. Each assistant reply is replaced with its own
`game_p0#request_index` generated thinking. Both canonical JSONL directories and progressive
`turns/*/*/final.json` records are supported. Progressive `ref` values must match
the owned original tool-call ID. Per-request `training_eligible` export flags
do not remove any turns from a complete trajectory. Tool code and observations are never
invented. All assistant thinking, tool code, tool formatting and reply formatting
are supervised; user/tool/image observations are context only.

`Trajectories(root).get(row)` returns the original processor tensors and an
annotation. `positions` are target INPUT token indices; causal hidden indices
are `position - 1`. `target_ids` equal input IDs at those positions.
`non_target_tokens` counts all masked input positions throughout the trajectory;
it is not a contiguous prompt boundary. Pixel and image-grid tensor hashes bind
exact dtype, shape and bytes, and the loader verifies them on every access. The production loader rejects manifests without verified complete original
logs and game completion. Qualification prefixes use the encoder separately.
The immutable
manifest includes exact positions/counts, source thought coverage, source/input
hashes, game-disjoint fold assignment, and processor annotation hashes. No
trajectory is truncated or split when over capacity. Failed preparation cleans
its staging directory and never publishes a partially prepared dataset.

`update_plan.json` fixes ordered whole-trajectory update groups. Its checksum,
manifest checksum and update index are resume identities. Groups accumulate
until their actual supervised token count reaches the approximate budget;
overshoot and the final short update are recorded explicitly. Sum cross entropy
and gradients, divide accumulated gradients once by the group's actual count,
then clip and update. Log actual counts at each update. Never average per game
or per microbatch. Validation uses the same all-assistant targets, with no fold-0
game entering training.

The first correctness implementation re-encodes every assistant prefix to prove
its position in the complete processor output, including image-prefix equality.
This repeats image processing and should be prepared/cached before expensive
training. Immutable annotations are saved during preparation; the loader uses
one full processor pass and verifies its input/target hashes against the cache.

Actual full trajectories can be much longer than the individually trimmed
request exports: the restored source audit already measures bp35 at 380,908
tokens and lf52 at 406,965 with the pinned processor and canonical generated
thinking. A 130K production preparation fails for these games and writes the
measured lengths in its adjacent audit JSON; it never silently exports a
smaller corpus. Increasing `--max-tokens` alone does not establish model or GPU
capacity.
