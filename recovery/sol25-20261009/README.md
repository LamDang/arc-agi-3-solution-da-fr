# Sol25 emergency recovery checkpoint

**New chat: read [HANDOFF.md](HANDOFF.md) first.** It contains the complete task context, approved decisions, restoration checks, and resume/publication instructions.

This branch backs up 1,004 of 1,334 finalized thinking-generation turns. It is a
recovery copy, not the final training dataset release. Generation was stopped
with no pending requests. The original partial DVC publication failed because
the workspace proxy could not reach S3. No production dataset PR was created.

The data archive contains live per-turn journals, final thinking, SFT exports,
the frozen 939-turn partial snapshot, source game logs, tokenizer/template, and
the local DVC cache. Code is in a small Git bundle whose base commit already
exists in this repository. Every part and reconstructed file has checksums in
manifest.json. Check upload-status.json: do not rely on an incomplete upload.

## Restore in a new environment

Clone this repository's `codex/sol25-recovery-20261009` branch without shallow
history. From the repository directory:

```sh
python recovery/sol25-20261009/restore.py --output ../sol25-restored
git bundle verify ../sol25-restored/sol25-code-delta-20261009.bundle
git fetch ../sol25-restored/sol25-code-delta-20261009.bundle refs/heads/codex/sol25-offline-handoff-20261009:refs/heads/codex/sol25-offline-handoff-20261009
git switch -c codex/progressive-sol25-20261008 debc8a3fcefd68809b45f1d6cd72d6a0f6a15787
git restore --source codex/sol25-offline-handoff-20261009 --staged --worktree -- .
tar -I zstd -xf ../sol25-restored/sol25-handoff-20261009.tar.zst
```

The release branch intentionally retains its original base HEAD with the
checkpoint code staged, preserving the publication watcher's expected HEAD.
Do not reset or replace the pinned partial snapshot. The code bundle restores
checkpoint commit `b4bcaa02a25562c8c8786d0699a072dfc0bb9f87`.

Read `ARC3-Inference/experiments/teacher-reasoning/progressive-sol25.md` before
resuming. Recreate the Python environment from the repository dependency files
if needed. Verify provider connectivity and the run manifest first; the current
manifest is `356aa3ece3068b14619603f424da0794df3ece655d8da13f42b8980cef127747`.

The effective ledger is $67.403473, with zero outstanding reservations. An
audited sidecar clears $229.411247 from 250 definite rejected HTTP attempts;
the original journals remain intact. User-reported account spend for Oct 8 was
about $138 OpenAI and $7 OpenRouter; these account totals have not been reconciled
to the individual run. Do not report the local estimate as confirmed billing.

Continue with 16 workers, two refinement loops, final thinking retained across
turns and compaction, a 120K student input cap, and 15 monitoring samples/game.
Run `ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/run_progressive_sol25.py`
only when connectivity is restored. It resumes under the existing $300 guard.
Retry the pinned partial DVC publication and draft PR, then publish all 1,334
turns when generation completes. No expensive generation needs to be repeated
for the 1,004 finalized turns.
