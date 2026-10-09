# Sol25 generation: complete handoff for a new chat

## Your task and existing authorization

Continue an already approved, partly completed production run in
`LamDang/arc-agi-3-solution-da-fr`. Generate reconstructed thinking for all 25
recorded winning Sol games, finish 1,334 turns, publish an intermediate dataset
with DVC and a draft PR, then update that PR with the complete dataset.
The user already approved the pipeline, production launch, 16 workers, $300
estimated-spend guard, fixes, retrying rejected requests, and publication.
Do not ask the user to repeat these decisions or regenerate completed work.
The user explicitly approved this public GitHub emergency backup after being
told the repository is public. This is not the final dataset release.

The previous chat was stopped to move to another session after the cloud
workspace's outbound proxy broke. Resume generation only after verifying the
restored checkpoint and working connectivity. The user requested an independent
subagent setup review; this was done before launch and after important fixes.
Use another review if restoration requires new substantive code changes.

## Exact saved state

- 1,004 / 1,334 finalized turns; 330 remain. 16 / 25 games are complete.
- 999 eligible training targets; five exceeded 120,000 student input tokens.
  Their final thinking is preserved because later turns can depend on it.
- 4,591 API attempt journals and 4,342 completed stage checkpoints are saved.
- 322 turns were selected for monitoring so far; 321 have a final-judge object.
  The monitor reported 143 sampled cases with issues. Do not describe all audits
  as clean; recompute available/flagged/unavailable counts from the final rows.
- No generation, publication, or sleep-loop process was left running at handoff.
- All 109 backup parts were read back through GitHub and matched the local
  source exactly. Their committed Git blob hashes matched the manifest.
  Reassembly matched both full-file SHA-256 values. The archive passed zstd
  integrity checking, and its attempts/stages/finals matched live file hashes.
- The backup is durable on GitHub. The original DVC publication is incomplete:
  its last attempt failed with 3,569 upload errors and its root object was absent.
  No production dataset PR or release commit has been pushed.

Recovery repository: https://github.com/LamDang/arc-agi-3-solution-da-fr

Recovery branch: `codex/sol25-recovery-20261009`.
The complete verified backup was committed at
`184e16bd28f87470c9badb1ce069b1950384b473`; this handoff is a later addition on
the same branch. Recovery files are in `recovery/sol25-20261009/`.
Read `upload-status.json`, `verification.json`, `manifest.json`, and `README.md`.
The first must say `complete: true` and `verified_parts: 109`.

## Recover without assuming the same VM

First inspect the current workspace and active processes. Do not overwrite an
existing run or reset unrelated user changes. If the live run already exists,
verify its hashes/counts and reuse it. The old workspace was
`/workspace/arc-agi-3-solution-da-fr`, but that path does not establish continuity.

For a fresh directory and working GitHub network access:

```sh
git clone --branch codex/sol25-recovery-20261009 --single-branch https://github.com/LamDang/arc-agi-3-solution-da-fr.git sol25-recovery
cd sol25-recovery
python recovery/sol25-20261009/restore.py --output ../sol25-restored
git bundle verify ../sol25-restored/sol25-code-delta-20261009.bundle
git fetch ../sol25-restored/sol25-code-delta-20261009.bundle refs/heads/codex/sol25-offline-handoff-20261009:refs/heads/codex/sol25-offline-handoff-20261009
git switch -c codex/progressive-sol25-20261008 debc8a3fcefd68809b45f1d6cd72d6a0f6a15787
git restore --source codex/sol25-offline-handoff-20261009 --staged --worktree -- .
tar -I zstd -xf ../sol25-restored/sol25-handoff-20261009.tar.zst
```

Use a full clone, not a shallow clone: the small code bundle requires base
commit `debc8a3fcefd68809b45f1d6cd72d6a0f6a15787`, confirmed on GitHub.
The reconstructed code commit is
`b4bcaa02a25562c8c8786d0699a072dfc0bb9f87`.
Keep a copy of this handoff outside the clone before switching branches, since
the recovery directory belongs to the recovery branch.

The active release branch intentionally stays at the base commit with the
checkpoint code staged. The publication watcher expects that HEAD; committing
or cherry-picking the recovery code onto it prematurely breaks its guard.
The separate checkpoint branch and bundle preserve the code commit safely.

Checksums:

| Restored file | SHA-256 |
| --- | --- |
| sol25-handoff-20261009.tar.zst | 84421cf8d1ca0beea3e7fcc8ac6fedfe887d6e8162d68f14948090549c861cd6 |
| sol25-code-delta-20261009.bundle | e922f5af822c4884a24eb0a9ea1dae286bc05fd60583ae08ac3c4f9076783cb4 |

The archive holds the live run, frozen partial dataset, source logs, tokenizer,
template, and DVC cache. It does not contain credentials or a portable virtualenv.
If GitHub access from the shell is still broken, the GitHub connector may work:
it handled this entire backup despite the workspace proxy failure. It can fetch
the UTF-8 base64 part files by their manifest blob SHA. Do not claim recovery
is complete until reassembly and checksum checks succeed in the new workspace.

## Important paths after extraction

- Live run: `ARC3-Inference/runs/think-progressive-sampled-pilot`.
  Keep using it despite its historical "pilot" name; it is the production run.
- Source: `ARC3-Inference/runs/gpt61sol-features-25games`.
- Frozen partial snapshot: `ARC3-Inference/runs/think-progressive-sol25`.
- DVC pointer: `ARC3-Inference/runs/think-progressive-sol25.dvc`.
- Tokenizer/template: `data/sft-gpt61sol-features-25games/qwen/`.
- Full setup/history doc: `ARC3-Inference/experiments/teacher-reasoning/progressive-sol25.md`.
  It contains historical estimates and updates: prefer the latest handoff and
  actual checkpoints over old status paragraphs.
- Launcher: `ARC3-Inference/scripts/run_progressive_sol25.py`.
- Generator: `ARC3-Inference/think_gen/progressive.py` and `client.py`.
- Monitor: `ARC3-Inference/scripts/monitor_progressive_sol25.py`.
- Publisher: `ARC3-Inference/scripts/manage_progressive_sol25_release.py`.
- Authoritative journals: `turns/<game>/<index>/calls/<stage>/attempt-*.json`,
  stage `complete.json`, and per-turn `final.json` inside the live run.
- `summary.json` is rebuildable and may retain the old inflated cost. Use the
  updated monitor and billing sidecar for the effective cost.

## Agreed generation behavior; preserve it

1. Run games in parallel, currently 16 workers. Turns of one game are sequential.
   Compaction chunks CANNOT run independently: each can retain earlier turns.
2. Later history must include each retained turn's FINAL generated thinking,
   after its refinement loops. This applies across compaction, including turns
   excluded as oversized training targets. Do not substitute initial drafts.
3. Flash creates a draft. A combined Sol judge checks coverage, code consistency,
   and grounding at xhigh. Flash refines if needed; another combined xhigh judge
   checks that refinement; Flash can refine a second time. There are at most two
   refinements; a passing result skips unnecessary remaining refinement calls.
4. Only a deterministic monitoring sample gets regenerated code and a final
   combined xhigh audit including code equivalence. The chosen setting is 15
   evenly spaced turns/game (within the user's 10-20 range), 374 total. Never
   switch back to auditing/regenerating code for every turn. No third refinement.
5. Final audit issues remain recorded. Optional sampled regeneration/audit stages
   may record unavailability after bounded failures and let the game continue.
   Required draft/judge/refinement stages stop safely after exhausted retries.
6. Include all 1,334 source responses, including the one text-only response and
   the 162 with zero teacher reasoning tokens. The generated text is a rationale
   reconstructed from visible evidence, not recovered private model reasoning.
7. Student input cap is 120,000 tokens, including images and retained thinking.
   Count with the saved Qwen tokenizer and template. Do not include the current
   target output in that input limit. Generate every needed turn first, then
   exclude oversized training targets while retaining their thinking/history.
8. Do not move source compaction boundaries, truncate saved thinking, replay games,
   or execute generated Python for the equivalence monitor.
9. SFT exports supervise only `loss_target_message_index`; prior assistant turns
   in each sample are masked. Training must honor that final-message-only mask.

## Models, cache, and retry settings

| Parameter | Current setting |
| --- | --- |
| Draft/refinement/regeneration | qwen/qwen3.8-flash via OpenRouter, Alibaba only, fallbacks disabled |
| Draft/refinement | temperature 0.7, internal reasoning off, 8,192 output tokens |
| Sampled code regeneration | Flash reasoning on, temperature 0.4, forced Python tool call |
| All Sol judges | gpt-6.1-sol through direct OpenAI Responses API, xhigh |
| Sol output cap | 16,000 tokens including reasoning |
| Sol cache | explicit-history-v1, last four stable history boundaries, explicit TTL 30m |
| Concurrency | 16 games; one sequential chain per game |
| Budget | $300 estimated successful spend plus truly uncertain reservations |
| HTTP 429 | retry indefinitely in place with backoff; release reservation while waiting |
| Other errors | launcher max-attempts=12 per stage; inspect before increasing |

Do not replace the explicit Sol cache policy with legacy `prompt_cache_retention`
or a proposed 1h TTL. The cache was specifically fixed and validated on a long
request. Saved explicit-policy Sol cache hits were about 93.3%; the older
implicit policy was about 8.9%. Flash's cumulative cache hit ratio was about
86.1%. Retain stable history and cache keys. Qwen Max availability was tested
successfully earlier, but the user did not request switching this production run.

## Billing correction and connectivity failure

The old local guard showed $296.814721, including $229.411247 in conservative
reservations for 250 explicit failed HTTP attempts: 218 HTTP 503, 24 HTTP 429,
and eight HTTP 400. The user confirmed these rejected requests have no cost.
Those reservations are now zero in the effective ledger. Successful-response
estimated cost is $67.403473: $5.286389 Flash (returned OpenRouter usage cost),
$62.117084 Sol (token pricing estimate). There are no outstanding reservations.
The user's dashboard reported roughly $138 OpenAI and $7 OpenRouter for Oct 8;
those account totals were not reconciled to this run. Do not call $67.40 an
account-verified bill or ask to increase the cap based on the obsolete ledger.

The migration preserved all original journal/final file hashes. Its sidecar is
`http-failure-billing-adjustments.json`; evidence is in
`http-failure-billing-migration.json`. Each adjustment validates the original
attempt path, SHA-256, HTTP status, and reservation. Do not delete or rewrite
the original failed journals. Typed new HTTP rejections are zero charged;
unknown transport outcomes retain a reservation, and successful but invalid
outputs remain billable. Do not broadly clear genuinely uncertain requests.

The current manifest SHA-256 is
`356aa3ece3068b14619603f424da0794df3ece655d8da13f42b8980cef127747`.
`pilot-validated.json` references it. The paid pilot and zero-call resume check
already passed. The billing correction passed independent review and 37 focused
tests. Do not rerun the one-off migration or repeat the paid pilot on recovery.

The observed outage was the Codex workspace's inherited proxy at `proxy:8080`.
Envoy returned HTTP 503 with `cloudflare_https_tunnel` / upstream connection
failure before provider headers. This did not establish provider-side outages.
S3 publication also failed. Read-only OpenAI cost and OpenRouter key-usage probes
failed through this path. GitHub and Sites connector tools still responded;
GitHub successfully stored/read back this backup. Keep configured proxy and
credential handling; do not try to bypass environment network controls.

## Preflight and resume

Inspect source/model/template/code fingerprints and counts before API calls.
Rebuild the virtualenv if absent. The project declares Python 3.12.12; observed
packages include requests 2.32.5, tokenizers 0.22.2, Jinja2 3.1.6, Pillow 12.2.0,
NumPy 2.2.6, and pytest 9.0.2. Inspect pyproject.toml/uv.lock: the full project
also has a local editable tufa-arc-agi-framework path, so a blanket sync can fail
if that checkout is missing. Resolve the actual required dependencies without
altering generation semantics. Keep DVC in its separate tool environment.

Required credentials are `OR_API_KEY` or `OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
and the existing AWS credentials for DVC. GitHub publishing also needs Git/gh
access, or the connector equivalent with the same verification. Never print
credentials. In managed cloud execution, check runtime credential readiness.

Check provider hosts with read-only requests. An unauthenticated OpenAI models
401 establishes HTTP reachability, not key authorization; distinguish those.
Do not burn stage retry slots while the proxy is returning immediate 503s.
Once recovery and connectivity pass, from the repository root:

```sh
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/monitor_progressive_sol25.py --once
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/run_progressive_sol25.py
```

Use a durable/background command session and follow `runner.log` plus
`driver_status.json`. Do not pass `--help` to the launcher: it does not parse
CLI arguments and would start the job. Use the generator's own `--dry-run` for
a no-inference source/settings check. The launcher resumes the existing live
directory and skips its validated pilot. Completed stages/finals are reused.
Old failures may have consumed a required stage's 12 attempt slots; inspect the
actual error before a justified retry-count increase. Do not increase dollar
authorization, force `--retry-uncertain`, or rerun completed stages casually.

Watch every ten minutes as requested. The user's latest chosen method was 20
successive 30-second waits with a printed counter, followed by a health check,
repeated while running. A detached local `codex queue` self-wake attempt failed
because this cloud thread had no local rollout; do not claim it works. A local
watchdog does not guarantee VM survival or wake an inactive chat.
Report genuine progress, retries, cache hit rates, effective cost, audit counts,
remaining sequential game lengths, and a fresh measured ETA. Prior 22-hour or
speedup estimates became obsolete after concurrency changes and the outage.

## Finish DVC and the draft PR, then the complete dataset

The publisher's saved `release-status.json` pins a partial snapshot with 939
turns and 13 complete games. Preserve it across retries; it predates the 1,004
turn recovery archive. It expects release branch
`codex/progressive-sol25-20261008` at base
`debc8a3fcefd68809b45f1d6cd72d6a0f6a15787` until its own publication commit.

- Snapshot JSON SHA-256: c3bee712b63b731415f9aa6da63b8fbd17e44ec3b8cc1f49800468fefb7befdc
- Pointer file SHA-256: 04b104fabb704d2ebfa3d9e3e9775659fb777db56d1bceee7a5e4798ccda3b5b
- DVC remote: s3://kaggle-arc-agi-3-dvc, region eu-west-3, remote name storage.
- DVC tool version: dvc[s3]==3.67.1, launched through uv by the publisher.

Review staged paths before publication. The recovery commit includes the helper
`ARC3-Inference/scripts/recover_progressive_sol25_filtered_regen.py`, while the
publisher's FILES allowlist does not include it. Reconcile that explicitly
(review and include the helper in the allowlist, or keep it unstaged) rather
than allowing the publisher to fail on an unexpected staged file.

Once S3/GitHub connectivity and branch/pin checks pass, run from repository root:

```sh
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/manage_progressive_sol25_release.py --threshold 13 --interval 60
```

This needs Git metadata write permission. The watcher pushes the complete DVC
tree first, commits the pointer and code, pushes the exact commit, then creates
a draft PR against main. It later replaces the snapshot with all 1,334 finalized
turns and updates the PR after generation completes. Never treat local I/O
counters as proof of remote upload. Verify the push result and retrievable root
object. Do not run `dvc repro tg_final`: that is the older experiment pipeline.

Attach any created PR to the new chat if its artifact tool is available. Keep
the emergency GitHub recovery branch until the complete DVC dataset and code
are safely published. Do not merge the emergency backup branch into main.

## First response expected from the new agent

Read this handoff and remote verification records, inspect/recover the workspace,
then briefly report: checkpoint verified or not; current connectivity; current
effective ledger; whether generation and publisher can resume. Continue the
already authorized work after these checks. Ask only for genuinely missing
credentials/access or a new decision, not for choices already recorded here.
