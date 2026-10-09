Production objective: one complete, untruncated trajectory per sample, supervising
every assistant turn (thinking, tool calls and final answer). User/tool/image
observations stay in decoder context and have ignored labels. Fold 0 is held out
by game. This mode is independent of the legacy final-reply reference.

`prepare_trajectories.py` reconstructs verified chronological histories and binds
each assistant to its own generated-thinking source. Missing history, generated
thinking, source completeness or an overlength trajectory blocks preparation.
See the data preparation audit for actual sizes and coverage; the sparse local
30-request validation panel is not the full 25-game training corpus.

`trajectory_run.py` accumulates **summed** cross entropy over whole trajectories
until the approximate supervised-token budget is reached. An update divides raw
accumulated gradients by its exact actual supervised-token count, then clips and
steps. It logs the actual count, trajectory count, overshoot and final short
update. It never averages trajectory means or carries stale gradients across
optimizer updates. Version-2 trajectory checkpoints preserve raw accumulated
gradients, pending tokens, pending trajectories, loss sum, optimizer, RNG and
cursor, bound to the immutable update plan. Legacy request-mean checkpoints are
rejected.

The vocabulary head selects input label positions `p` and predicts them from
hidden position `p-1`, including arbitrary disjoint spans. The custom backward
uses native CE autograd and an explicitly recorded matrix row count. Default
native-length rows and reduced row counts are separate numerical candidates.
`--head-backward-rows 16384` is invalid if the trajectory has more than 16384
assistant targets; no automatic fallback is qualified.
Measured coherent examples within 130K have 18,556–27,523 assistant targets,
so 16,384 rows cannot cover them. The production candidate uses 32,768 rows,
with its own native multi-span gradient identity; it is not qualified by the
earlier 16,384-row contiguous-final-target operator test.

This production mode is **implemented but not GPU-qualified**. The preserved
16K contiguous-final-target qualification cannot qualify it. Before production:

1. `trajectory_gradient_audit.py` must capture two unchanged native HF+PEFT
   multi-span backwards and the candidate against a manageable encoder-produced
   sample, using nonzero A/B adapters and the exact quantized checkpoint/runtime.
2. `trajectory_gradient_gate.py` rechecks hashes and all raw 744 gradients for
   bitwise equality, including the scalar loss. It rejects an old final-reply
   audit, changed operator/model identities and zero-B-only evidence.
3. `trajectory_run.py --mode qualify` measures actual full-trajectory context,
   targets and image extremes, one disposable token-normalized optimizer step,
   GPU peak and host RSS. It saves `capacity.json`; diagnostic updates are
   discarded. Training reloads the supplied adapter and requires this matching
   capacity report. A capacity pass does not prove long-length gradient parity.
4. Train with `--mode train --capacity-report .../capacity.json`; resume with
   `--resume`. Use `--mode evaluate` for teacher-forced held-out all-assistant CE
   with `--checkpoint-dir` pointing to the trained v2 checkpoint. Evaluation
   restores that checkpoint and records its file/hash/cursor/update count; it
   never silently evaluates the initial adapter. A forward has no optimizer
   update.

Production starts from clean PEFT random A/zero B. Create it with
`trajectory_initialize.py --model ... --out ...` before loading any dataset.
Pass its `initial-adapter.pt` as `--adapter-state` and `provenance.json` as
`--adapter-provenance` in every production mode. The provenance/hash, zero-update
count and all zero B tensors are validated. The fold-0 overfit nonzero checkpoint
is only gradient-test evidence and is rejected as a production initialization.

CPU validation covers arbitrary spans/native shift and upstream scaling,
unequal trajectory target counts, ignored context counts, normalization before
clipping, partial summed-gradient/token resume and invalid annotations. GPU
multi-span parity, actual trajectory capacity/speed, trained production resume
and fold-0 evaluation remain pending measured evidence.
