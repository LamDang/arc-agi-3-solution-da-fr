# Progressive thinking generation for the 25 Sol games

Final release (2026-10-09 10:07 UTC): **1,334/1,334 turns finalized** across all
25 games, with 1,324 training samples. See the
[full dataset verification and exploratory analysis](progressive-sol25-exploration.md)
for exact input/output token counts and length distributions.

Historical status (2026-10-09 04:41 UTC): **1,004/1,334 turns finalized** across 25 games;
999 are eligible training targets and five exceed the 120,000-token input limit.
Sixteen games have finished. The resumable driver is stopped while a provider
connectivity failure is investigated; the **$300 guard** remains in force.
The two-turn sampled/unsampled pilot passed, including finalized-history
preservation and a zero-call resume check. HTTP 429 backoff remains unlimited.
The HTTP billing correction passed independent setup review and 37 focused
checks. See `runs/think-progressive-sampled-pilot/driver_status.json` for the
live phase.

## Scope and agreed behavior

Source: `runs/gpt61sol-features-25games`, DVC source hash
`f677e9c8b2f2ade56ace3b21e5072c75.dir`. All 192 archived files were downloaded
and MD5-verified. The source contains 25 winning games, 1,334 model responses,
1,333 tool-call responses and one text-only response. The original requests
consumed 82.65M input tokens. There are 162 zero-reasoning-token responses;
`b4` still reconstructs their stated reasoning, so these are included.

The generated text is a reconstructed rationale, not recovered private Sol
reasoning. Source actions, tool results, images and compaction notes remain
fixed. No games are replayed and no generated Python is executed.

Every turn completes the pipeline below before the next turn starts:

1. Flash writes a draft from the visible history and recorded next output.
2. One Sol `xhigh` call judges coverage, code consistency and facts together.
3. If the judge or advisory checks find an issue, Flash refines the draft.
4. One combined `xhigh` judge checks that refinement. If issues remain, Flash
   performs a second refinement. There are at most two refinements.
5. **Only on the monitoring sample**, Flash regenerates the Python call from
   the final rationale, without the target call. One Sol `xhigh` call checks
   coverage, code consistency, facts **and functional call equivalence together**.
   This is the **final audit**, used for production monitoring. It does not
   trigger a third refinement. Text-only sampled responses skip regeneration
   and equivalence but still receive the other three checks.
6. Commit the final rationale and audit. Only then advance to the next turn.

Each later retained assistant turn carries its **final generated thinking**,
including after compaction. Flash receives it in `reasoning`; the Sol judge
receives it as visible assistant text because the Responses converter does
not serialize a chat message's plain `reasoning` field. Teacher encrypted
reasoning is removed. Both providers see the code-only student history.

The monitoring panel contains **15 evenly spaced responses per full game**
(configurable from 10 to 20), including the first and last response. Games with
fewer than 15 use all responses. This source yields **374 monitored responses**;
all 374 have Python calls. The exact selection is saved in
[`progressive-sol25-monitoring.json`](progressive-sol25-monitoring.json).
The panel is deterministic, independent of pilot
`--limit`, and unchanged on resume/extension. It is chosen before knowing which
requests exceed the student-input limit. Unsampled turns run neither code
regeneration nor final auditing. Their `final_judge` is null, not a fabricated
pass. A refinement-2 result may therefore be unjudged unless sampled;
`last_judge_applies_to_final` distinguishes that from a final verdict.

Final-audit quality failures are reported, not silently considered perfect and
not automatically dropped. HTTP 429s wait and retry indefinitely in place. Other infrastructure errors,
malformed verdicts and truncated/empty responses stop progression after the
bounded retry allowance.

## Scheduling: games parallel, chunks sequential

The source has **58 contiguous context stretches**, longest 43 responses.
Every stretch after the first retains 11–25 earlier responses. These are real
dependencies: **compaction chunks of the same game cannot run independently**.
The earlier estimate claiming 60–75 independent chunks was incorrect.

The current approved run uses **sixteen workers** across games, following the
two-, four-, and eight-worker trials, always one sequential chain per game. The earlier
throughput target of ten workers remains unvalidated:
Alibaba throttled even the one-worker pilot. Any later concurrency increase
should follow checks of errors, cache hits and projected spend. Longest
chain: `sk48`, 147 responses. Compaction changes the cache prefix, but never
permits discarding finalized thinking for retained turns. Each game/chunk gets
a stable Sol prompt-cache key. Per-turn generation, refinement and auditing
run together rather than in dataset-wide stage sweeps.

The running job now selects explicit history caching with TTL `30m`.
**Correction after checking current OpenAI documentation:** GPT-6.1 Sol uses
`prompt_cache_options.ttl`, whose supported/default value is **`30m`**, refreshed
on write/reuse. The runner's `in-memory`/`24h` choices were based on legacy
retention APIs and are not a verified Sol configuration; do not select them
for this model. There is no supported `1h` Sol TTL in the current documentation.
Flash and Sol caches are separate. See the cache diagnosis below.

## Student input limit and output dataset

The agreed limit is **120,000 input tokens**, inclusive. Count the student
prompt with the pinned Qwen tokenizer and chat template, code-only tool schema,
all retained final thinking, images and assistant generation prefix. Do not
include the current target's thinking or call in this input limit.

Pinned model files:
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`,
`tokenizer.json` and `chat_template.jinja`. Match the existing dataset's image
accounting: 640×640 board images, 400 vision tokens each. Unknown image sizes
fail closed. The template uses `enable_thinking=true`, `preserve_thinking=true`.

Generate every required turn, even when its student input exceeds 120,000.
Such a turn is **excluded as a training target only**. Keep its full generated
thinking for subsequent requests and compaction chunks. A later sample can fit
again after compaction. Do not shorten finalized thinking or move source
compaction boundaries to make samples fit.

Export one sample per eligible request under `sft/<game>.jsonl`, with only
`loss_target_message_index` supervised; all preceding turns are masked. The
training loader must honor this final-message loss mask. Do not feed these
files to a loader that trains every assistant message. Every `final.json`
records the exact input count, eligibility and exclusion reason.

## Parameters

| Parameter | Value |
| --- | --- |
| Draft/refinement model | `qwen/qwen3.8-flash` via OpenRouter |
| Provider | Alibaba; fallbacks disabled |
| Prompt | `b4`, no teacher-summary input; separate text-only prompt |
| Length target | Max of ~0.75 × teacher reasoning tokens and 3 × stated-reasoning words |
| Draft/refinement temperature | 0.7 |
| Draft/refinement internal reasoning | Off |
| Flash output cap | 8,192 tokens |
| Judge | `gpt-6.1-sol` via direct Responses API |
| Judge effort | **xhigh for every combined judge, including final equivalence** |
| Sol cache | Explicit-only; last 4 stable history user/tool boundaries, including images; TTL `30m` |
| Sol output cap | 16,000 tokens including reasoning |
| Regeneration | Flash, reasoning on, temperature 0.4, forced Python tool call |
| Refinements | At most two; passed drafts/refinements skip remaining refinement work |
| Final audit + code regeneration/equivalence | 15 evenly spaced requests/game; 374 total |
| Workers | **16**, one sequential chain per game |
| Retry limit | HTTP 429: unlimited backoff in the same attempt; other failures: 12/stage |
| Student input cap | 120,000 tokens |
| Pilot spend guard | $8 estimated spending + unresolved-call reservations |
| Full-run approved guard | $300 estimated spend plus uncertain-call reservations |

`OPENROUTER_API_KEY` or `OR_API_KEY` is accepted; `OPENAI_API_KEY` is required
for Sol. Configured credentials are ready. No credential values are logged.

## Resume guarantees and failure behavior

Entry point: `python -m think_gen.progressive`. The existing experiment DVC
stages remain unchanged; **do not use `dvc repro tg_final` for this run**.

- Each API attempt reserves budget in an atomic, fsynced file **before** sending.
- Its raw response and usage are committed **before** result validation.
  Invalid/truncated responses remain billed in the ledger.
- Definite failed HTTP 4xx/5xx responses without model output have zero charge.
  Historical failed attempts are reconciled through a SHA-verified sidecar;
  their original attempt journals and final thinking remain untouched.
- An HTTP 429 releases its reservation during backoff and atomically reserves
  again before the next send. Unknown network outcomes retain their reservation.
- A stage's `complete.json` is reusable only for the same request hash.
  A response saved just before a crash can be revalidated without another call.
- A turn's `final.json` contains hashes of its retained finalized thinking.
  Resume reloads that history before proceeding. Missing dependencies fail.
- A run-level file lock prevents two processes writing to the same run.
- Source files, generation code, prompts, model parameters and tokenizer/template
  are fingerprinted. Changed semantics reject resume. Games, turn limit,
  concurrency, budget and retry count can change without altering completed work.
- An interrupted in-flight call has uncertain billing. Resume stops by default
  and identifies the attempt. `--retry-uncertain` explicitly permits resending;
  the original worst-case reservation remains charged. Exactly-once remote
  billing cannot be guaranteed across a network/process interruption.
- Stop after exhausted retries or failed budget reservation. Other in-flight
  requests may finish; their reservations already count against the guard.
  Raising `--max-attempts` permits additional attempts after inspecting a failure.
- The cost guard uses a pricing proxy, not an account-enforced dollar limit.
  Reservations include all input as uncached and conservative output allowance.
  Actual Sol account prices must be substituted if they differ.

Checkpoint files live under `turns/<game>/<index>/calls/<stage>/`; these and
`final.json` are authoritative. SFT exports and `summary.json` are rebuildable.
Changing the monitoring policy also changes the semantic fingerprint. The
initial all-audits pilot remains separately archived; do not silently rewrite
its manifest to claim compatibility with the sampled variant.

Raw source data and run checkpoints must be retained to resume or reconstruct
requests. Hashes reference source contexts rather than duplicating their images.

## Validation

Independent subagent review was completed before the first paid request.
Review found and fixed export-schema omissions and weak combined-judge schema
requirements. Existing unsafe legacy behaviors are bypassed: empty generated
history during refinement, missing verdicts considered passed, error results
considered finished, and truncated responses accepted.

Offline checks: 21 tests passed (`tests/test_progressive.py` and
`tests/test_think_gen.py`), including interrupted-stage recovery, invalid-output
billing, budget reservation, uncertain-call handling, final history across a
synthetic compaction boundary, overflow target exclusion with retained history,
zero-call resume of completed turns, stable monitoring selection, and skipping
both regeneration and auditing for unsampled turns. The real tokenizer/template rendered
the first request of all 25 games successfully (5,519–5,539 input tokens).

Initial paid pilot (`runs/think-progressive-pilot`, before the sampling change):

- Two consecutive `ar25` turns finalized, with 5,539 and 7,084 student input
  tokens. First turn needed no refinement; second needed one.
- Both final audits reported complete coverage, grounded facts and equivalent
  calls. Two easy early turns do not establish overall quality.
- A real interruption at turn 2 regeneration resumed without changing any of
  eight completed stage files. Independent review reconstructed and verified
  all 22 then-recorded request-prefix hashes and both finalized-history hashes.
- Re-running the completed two-turn scope made **zero new API calls**, finishing
  in 0.045 seconds. The existing SFT index builder validated both exports and
  their generation-prefix alignment (388 and 449 output tokens).
- Extending to a third turn exhausted nine upstream 429 attempts. No fourth
  turn was attempted. The default retry setting was deliberately raised after
  inspecting the failure, while retaining the same $8 guard.
- Successful-response estimated cost: **$0.078861**. Conservative reservations
  for 15 rate-limited attempts: **$0.184501**. Guarded total: **$0.263362**.
  Reservations are not a claim that rate-limited requests were billed.
- Sol cached 48.7% and Flash 60.5% of input tokens on this tiny, initially cold
  pilot. These do not establish cache rates on long stretches.

This paid pilot does **not** cross a real compaction boundary. Real compaction
starts around request 35 or later; retained-thinking and overflow dependencies
are tested synthetically. The sampled variant has additional offline tests;
its independent review passed before any new paid attempt.

Sampled pilot (`runs/think-progressive-sampled-pilot`): `sk48`, first two turns,
one worker, 15 monitoring samples per full game, $8 guard. After initial 429
failures and the reviewed unlimited-retry update, this pilot **passed at
16:28:05 UTC on 2026-10-08**. Both turns finalized; the first was audited and
the second made neither regeneration nor final-audit calls. One turn needed
one refinement. The next turn's history contained the first turn's final thinking.
The repeat scope made zero new calls and changed no completed stage files.
Guarded pilot cost, including older unresolved reservations: **$0.317260**;
four new stage calls in the successful continuation, 170 seconds elapsed.
The zero-call check took 0.52 seconds. `pilot-validated.json` records the evidence
and manifest hash. The launcher then started the full phase automatically.
The older [`progressive-sol25-pilot.json`](progressive-sol25-pilot.json) preserves
the earlier blocked-pilot evidence; it is not current job status.

## Commands

From the repository root (model files can be fetched using the existing
`data/sft-gpt61sol-features-25games/dvc.yaml` pinned fetch stage):

```bash
PYTHONPATH=ARC3-Inference ARC3-Inference/.venv/bin/python -m think_gen.progressive \
  --run ARC3-Inference/runs/gpt61sol-features-25games \
  --out ARC3-Inference/runs/think-progressive-sampled-pilot \
  --tokenizer data/sft-gpt61sol-features-25games/qwen/tokenizer.json \
  --template data/sft-gpt61sol-features-25games/qwen/chat_template.jinja \
  --games sk48 --limit 2 --workers 1 --monitor-per-game 15 --budget 8
```

The production launcher resumes the validated pilot and full job with the
actual approved settings (sixteen workers, $300 guard, three ordinary attempts):

```bash
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/run_progressive_sol25.py
```

Do not launch a second copy while the current driver or runner is alive.
The pilot output extends to the full source set without repeating completed
turns when its semantic fingerprint is unchanged. A fresh output directory
starts fresh and incurs those calls again. Add `--dry-run` to the module
command to inspect settings/source selection without model calls.

## Evidence and estimate assumptions

The independent held-out 30-request experiment reached coverage 0.988,
fact grounding 0.933 and call equivalence 0.80 after two refinements. It used
separate judges and independent teacher-history contexts. Those metrics are
not established for this new combined-judge progressive variant.

Measured panel refinement rates: 22/30 first refinements and 10/30 second
refinements. Applied to 1,334 responses, expected calls are approximately:

| Stage | Calls |
| --- | ---: |
| Flash draft | 1,334 |
| Sol combined judge 1 | 1,334 |
| Flash refine 1 | 978 |
| Sol combined judge 2 | 978 |
| Flash refine 2 | 445 |
| Flash regeneration, monitoring sample only | 374 |
| Sol combined final audit, monitoring sample only | 374 |
| **Flash total** | **3,131** |
| **Sol total** | **2,686** |

The first estimate assumed a separate equivalence judge on every response.
Combining the judges and sampling the final audit reduces Sol calls to about
2,686 and Flash calls to about 3,131. All remaining Sol calls use xhigh.

Current official OpenAI standard global prices for Sol, dollars per million:
ordinary input / cache read / cache write / output **2 / 0.10 / 2.50 / 10**.
The full-request tier above 272K input is **4 / 0.20 / 5 / 15**. Flash prices
observed in OpenRouter's catalogue: **0.15 / 0.016 / 0.47** for input/cached/output.
Sol pricing is verified against the
[official model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
Retries, uncertain billing and cache misses can increase cost.

### Conditional full-run cost and time

Planning assumptions: approximately 170M Sol input tokens, 3.83M output tokens
including hidden reasoning, and $15 for Flash. Context estimate allows roughly
84M input tokens per complete pass; actual rebuilt student histories and
monitoring selection will differ. Combined-judge output estimates are 1,500
for judge 1 / final audit and 1,300 for judge 2. These are conservative planning
values informed by the separate-judge panel and short combined-judge pilot,
not measurements of all 25 progressive games.

| Sol cache-hit fraction | Estimated total API cost |
| --- | ---: |
| 95% | ~$91 |
| 90% | ~$111 |
| 80% | ~$152 |
| 50% | ~$274 |
| No cache reuse | ~$478 |

These corrected estimates conservatively treat all noncached Sol input as
cache writes at $2.50/M; actual ordinary judge suffixes cost $2/M and can lower
the total. Allow **$100–180 with healthy 80–95% cache reuse**, including a
modest retry and output-length margin. Around a 50% cache rate, allow **$275–330**.
The approved **$300 stop threshold** will stop rather
than guarantee completion if cache reuse or output length is worse. Price
uncertainty and potentially billed failed calls are additional reasons to use
the guard and report actual token usage.

With healthy API service and 10 workers, estimate **5–7 hours**. This assumes
roughly 94 seconds of API service per response on average, about 35 aggregate
worker-hours, and a roughly 4-hour longest sequential game before scheduling
and retry overhead. With only 2 workers, expect roughly **18–24 hours** under
healthy service; with 4 workers, roughly **9–12 hours** if throughput scales.
These are throughput scenarios, **not a reliable ETA while
Alibaba is returning persistent 429s at one worker**. There is currently no
validated completion-time estimate for the throttled endpoint.

Do not reuse the old 1.5–2-hour claim: finalized turns of the longest
147-response game are sequential. The user has approved launch. Start once the bounded pilot can progress reliably
and the OpenRouter allowance is sufficient; no further run approval is required.

## Approved launch attempt

After the user authorized launch, the sampled pilot was resumed at one worker
with `--max-attempts 6` and the same $8 pilot guard. Its three additional draft
attempts all returned upstream Alibaba HTTP 429. No turn was finalized. All
checkpoint files remain available; the full dataset job was not started.

A read-only OpenRouter `/api/v1/key` check returned HTTP 200 and reported:

- Weekly key limit: **$250**.
- Remaining allowance: **$5.338680666**.
- `is_free_tier: true` (reported by the API; not proof of the cause of the 429s).
- The model catalogue still lists Alibaba as the only endpoint.

The remaining allowance is below the approximately $15 Flash estimate,
independently of the upstream rate-limit problem. Increase the remaining
allowance to at least $30 or configure a suitable replacement key, then rerun
the sampled pilot with an increased persisted attempt limit. Verify the
sampled and unsampled paths plus a zero-call restart before expanding to all
25 games, initially at low concurrency and with the approved $300 guard.

## Retry after the spending-limit increase

The user raised the key limit. A fresh read-only check confirms a **$500 weekly
limit and $255.338680666 remaining**, enough for the projected Flash spend.
The pilot was resumed at one worker with three additional allowed attempts;
upstream Alibaba 429s continue. The increased spending limit does not itself
remove provider request/token quotas or capacity limits.

OpenRouter also reports `is_free_tier: true`, and `/api/v1/credits` returned
`total_credits: 0`, `total_usage: 0`. These are diagnostic observations; the
429's exact cause is not established by those fields alone.

The raised-limit retry exhausted its three additional attempts. One further
budget-guarded diagnostic attempt captured the full HTTP error: Alibaba says
`qwen/qwen3.8-flash is temporarily rate-limited upstream`, with `is_byok: false`,
and suggests adding an own provider key through OpenRouter integrations. No
`Retry-After` or rate-limit headers were returned. No new turn was finalized;
the provider-capacity blocker remains, while the spending-allowance blocker is
resolved. This was the prelaunch state; the later unlimited-retry continuation
passed the pilot and started the full job as recorded above.

## Unlimited 429 retries and approved launch driver

At the user's instruction, HTTP 429 is now handled like the harness's
`ARC3_HTTP_RETRIES=-1` mode: retry the **same request indefinitely**, using
exponential backoff (2, 4, 8, …, 60 seconds) plus jitter. A valid server
`Retry-After` may require a longer wait and is respected, including HTTP dates.
A short `Retry-After` never shortens exponential backoff.

429s do not consume the ordinary three-attempt allowance. One conservative
in-flight reservation covers all retries of that attempt; successful usage
replaces it. Previously recorded conservative reservations are preserved.
The checkpoint records `rate_limit_retries`, the latest error and the next
delay. A restart during a known `rate_limited` wait resumes the same slot and
honors the remaining wait; an interrupted request actually in flight is still
treated as uncertain. `new_calls` counts newly allocated attempt slots, not
every HTTP resend; use the saved retry counter for 429 frequency.

The retry change passed independent review and 21 offline tests, including
12 consecutive 429s with a one-attempt allowance, same-request preservation,
same-slot restart during backoff, and Retry-After handling. A reviewed one-time
transport-only manifest migration preserved four completed stage checkpoints
byte-for-byte. Old manifest, old/new code hashes and preserved-checkpoint
hashes are saved in `transport-migration.json`. No prompts, model settings,
sampling or finalized history changed.

The launcher is `scripts/run_progressive_sol25.py`. It finishes the saved
`sk48` two-turn pilot, verifies audited/unaudited paths and finalized history,
checks an exact zero-call restart and preserved stage hashes, then automatically
starts **all 25 games with two workers and the approved $300 guard**. The full
phase restores the ordinary non-429 allowance to three attempts;429 retries
remain unlimited. No additional launch approval is required.

The pilot gets one additional attempt slot (`--max-attempts 11`) because the
old retry policy had already exhausted ten attempts on its second draft.
This does not discard or rewrite the old ledger.

All output remains under `runs/think-progressive-sampled-pilot` so paid
checkpoints are reused. `driver_status.json` names the phase and process;
`runner.log` records progress and 429 waits; `pilot-validated.json` is the
validation gate; `full-summary.json` is written when the full phase stops or
finishes. The launcher uses its own lock to prevent duplicate launchers and
the underlying runner retains its existing output lock.

429s may extend elapsed time without a fixed bound. The $300 guard still
applies to successful requests and unresolved reservations. Budget exhaustion
or non-429 failures stop the job safely; the launcher does not blindly restart
those failures.

## Runtime monitoring

`scripts/monitor_progressive_sol25.py` checks immediately and then every
**10 minutes before the pilot validates**, switching to **30 minutes after
validation**. The manifest hash must match the validation record. The pilot
has passed, so the live watchdog uses 30 minutes. This operational script is
outside the generator fingerprint and does not change requests or checkpoints.

It verifies driver/runner PID identity, counts finalized/eligible/excluded turns
and completed stages, totals estimated charges and conservative reservations,
measures each provider's cache usage, and checks sampled final-audit verdicts.
It reads stage/turn journals rather than the stale pilot `summary.json` while
the full phase runs. Live pending requests are normal; pending attempts after
runner death are flagged for uncertain billing. Persistent 429s remain retries;
30 minutes without a completed stage raises a monitoring alert but does not
stop or resend the job. Completed/stopped phases end the watchdog.

Files in the run directory:

- `monitor_status.json`: latest snapshot, next check time and monitoring PID.
- `monitor_history.jsonl`: durable snapshots at each check.
- `monitor_alerts.jsonl`: new completion, process, stop, progress or quality alerts.
- `monitor.log`: watchdog output and any read errors.

The watchdog has a separate lock. Restarting it performs an immediate check
and preserves history. Read errors retry after one minute. Six focused tests
passed for cadence gating, live pending/429 handling, costs/cache accounting,
stale-summary rejection and PID/output-directory identity. Independent subagent
review passed with no blockers. Its suggestions to schedule checks from the
previous check time, verify the output directory and separately report legacy
429 attempts are implemented.

```bash
# Read-only current snapshot; no model calls or monitor-file writes
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/monitor_progressive_sol25.py --once

# Persistent local watchdog; single instance enforced by its lock
ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/monitor_progressive_sol25.py
```

At 16:42 UTC: 4/1,334 turns finalized, all eligible, 3 sampled final audits
with no reported issues; driver and runner alive. Guarded cost **$0.431576**,
of which **$0.268004** is unreconciled reservations, not confirmed billing.
The early cache fractions were Sol 41.8% and Flash 27.4%; these small, cold
contexts do not establish full-run rates. Upstream 429s still prevent a firm ETA.

**Persistence limitation:** this is a local watchdog. It writes local alerts,
not chat notifications, and cannot survive a VM reclaim by itself. Hosted
automations available in this session permit at most hourly cadence; the
repository skill's `send_later`/`update_trigger` wake-up tools are unavailable.
The watchdog is launched as a tracked shell session, but that is not a
guarantee of VM retention. After a VM restart, inspect pending attempts before
resuming the driver, and relaunch the watchdog. No uncertain call is retried
automatically. A durable 30-minute chat wake-up service has not been configured.

## Sol cache diagnosis (2026-10-08, approximately 17:53 UTC)

The live manifest records `cache_retention: default`; the client omits both
`prompt_cache_retention` and `prompt_cache_options`. Current official
[prompt-caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)
states that Sol uses the newer caching behavior: default/minimum TTL `30m`,
implicit writes at the latest eligible message, and optional explicit
breakpoints. The earlier explanation based on legacy retention flags was wrong
for Sol. No live source or request settings were changed during this diagnosis.

Evidence from 86 completed Sol calls: three cold calls had zero cache hits;
**all other 83 reused exactly 4,080 tokens**. A 60,658-token judge request
reported 4,080 cached and 56,575 cache-write tokens. Its next judge, 51 seconds
later, again read 4,080 and wrote 56,597. Median same-game call intervals were
about 65 seconds (`ar25`) and 46 seconds (`bp35`). TTL expiry therefore does
not explain the repeated small-prefix reuse.

A read-only reconstruction verified unchanged complete game-history prefixes
across 37 consecutive turn pairs in these two games. Cache keys also stayed
constant within their first context stretch. The likely cause is **cache
breakpoint placement**: implicit caching writes through the changing final
judge user message (thinking, target call, and optional regenerated code).
That judge message is replaced on subsequent judge calls and omitted from
the next game's turn history. The reusable game history has no explicit
write boundary, leaving only the initial developer prefix reusable. The
recommended correction is to mark reusable game-history boundaries explicitly,
before the variable judge payload, and verify read/write usage on a small
batch before migrating the live run.

Current documentation also states **cache writes cost 1.25 times uncached
input**, while Sol reads cost 0.05 times uncached input. The running price
reconciler currently omits the write premium. At the configured $2/M input
proxy, 2.802M recorded cache-write tokens implied about **$1.40 additional cost**
at that diagnosis snapshot. The implementation, historical ledger and planning
tables are now corrected as recorded below. Raw usage retains `cache_write_tokens`,
so historical costs can be recomputed without repeating inference. The
reservation calculation uses a higher $4/M input bound, but reconciled totals
still need the missing premium for an accurate aggregate $300 guard.

## Verified cache fix and resumed production (2026-10-08, 18:08 UTC)

Sol now uses `prompt_cache_options: {mode: explicit, ttl: 30m}`. The client
marks the last four stable developer/user/tool-result boundaries after API
conversion, including final image blocks. It excludes the changing judge
message from cache writes. A read-only source check found that the previous
endpoint remains among these four boundaries in **all 1,276 append-only turn
transitions**. Compaction still changes the prefix and starts a new cache.
No thinking, source action, judge rubric or model/effort changed.

A real long-context test used final generated history from `ar25` turn 17,
changed the judge suffix, then appended actual turn 18. All three calls used
the production Sol model, xhigh effort, JSON mode and output cap:

| Request | Input tokens | Cache reads | Cache writes | Read fraction | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm long history | 60,696 | 0 | 59,697 | 0% | $0.158871 |
| Same history, different judge suffix | 60,921 | 59,697 | 0 | **97.99%** | $0.016158 |
| Next real turn | 66,239 | 59,697 | 5,216 | **90.12%** | $0.036632 |

Total validation cost: **$0.211660**, included in the $300 production guard.
The repeated probe scope made **zero new calls**. Evidence is saved in
[`progressive-sol25-cache-validation.json`](progressive-sol25-cache-validation.json)
and the run's `cache-validation.json`; raw probe stages are retained in the
production journal. These are measured cache improvements, not a promise of
the same fraction for every game or compaction boundary.

Production was paused only after SIGSTOP had stopped all worker threads and
the journal contained **zero pending requests**. The reviewed one-time
`scripts/migrate_progressive_sol25_cache.py` verified source/settings and
old-code hashes, archived the old ledger and code, preserved all **282 existing
final/stage checkpoint files byte-for-byte**, and updated the manifest plus
pilot validation marker for the operational change. Semantic request hashes
stay unchanged; new attempts separately record the cache policy and transport
hash. No completed model request was repeated. The two-turn production resume
check made zero calls and confirmed all 282 hashes.

Historical write repricing added **$2.225991** across 109 successful Sol calls.
Raw responses and prior uncertain-call reservations are preserved. After adding
the cache probe, the corrected ledger at migration was **$13.084517**, including
reservations. New reservations cover the full cache-write rate. The audit and
old/new charges are in `cache-migration.json`; original files are archived in
`before-explicit-cache.zip`. The cost tables above now include write pricing.

Production resumed with two workers and the same $300 guard. The local
30-minute watchdog was restarted and now separately reports legacy and
explicit-policy Sol usage so historical low hits do not hide the improvement.
At 18:10:50 UTC it confirmed both processes alive and **57 finalized turns**.
Its next check is 18:40:50 UTC. The monitor also reports cache-write input.

Relevant checks: 33 offline tests passed across generator/client/cache/monitor
suites. Independent subagent review approved the cache fix, long-test evidence
and manifest/ledger migration before restart. The runner now supports graceful
SIGINT/SIGTERM: it journals responses already in flight and stops new calls;
an interrupted unsent reservation has a distinct safe state and reuses its
attempt slot on resume. Hard-interrupted pending calls still require inspection.

Production confirmation after restart: `bp35` turn 28 judge 2 read **88,432 of
89,486 tokens (98.82%)** with zero writes. Turn 29 judge 1 then read the same
88,432 tokens from a **92,398-token request (95.71%)**, writing only 2,995 new
history tokens. The fix therefore works on actual resumed long production
requests as well as the isolated validation sequence.

## Qwen Max availability check

Both `qwen/qwen3.8-max-0902` and `qwen/qwen3.8-max-prime` returned `OK` through
OpenRouter's Alibaba provider, with reasoning enabled: 1.6s and 1.4s respectively,
no 429s. Combined test cost **$0.000846**, separate from production generation.
Both reject `reasoning.enabled: false` with HTTP400: reasoning is mandatory.
This means they require a different reasoning setting from the current Flash
writer; the production writer remains Qwen3.8 Flash. This was a short text
availability probe, not a long multimodal quality or throughput validation.
Exact results/catalog metadata are in
[`progressive-sol25-qwen-max-availability.json`](progressive-sol25-qwen-max-availability.json).

## Four-game concurrency trial (2026-10-08, 18:49 UTC)

The user authorized increasing to **four games concurrently**. The prior
runner handled SIGTERM gracefully, finished/journaled current HTTP responses,
started no new calls after the signal, and exported its partial dataset before
exiting. There were no pending requests or unexpected errors at restart.
All **493 pre-transition finalized/stage checkpoints** stayed byte-identical,
and the manifest did not change: worker count is an operational setting.

The launcher now persists `--workers 4`; the $300 guard, xhigh effort, two-loop
pipeline, finalized-history dependencies and cache configuration stay the same.
At restart, 95 responses were finalized (94 eligible, one oversized target),
with $15.551152 in charges plus reservations. Live calls independently
confirmed four distinct games progressing concurrently: `bp35`, `cd82`,
`cn04` and `dc22`.

The watchdog checks every **10 minutes during the initial four-worker
observation**, returning to 30 minutes after at least 10 minutes of continued
progress with both processes alive. This is a health check, not a claim of
linear speedup. Snapshots expose `worker_trial` with elapsed time, completed
turns and observed turns/hour. The next initial check is **19:01:58 UTC**.
The first 123 seconds completed five more turns, but that short interval
includes reuse of partially finished stages and cannot establish an ETA.

`four-workers-transition.json` records the restart, hashes and baseline.
The monitor's new cadence test passed (eight monitoring tests total).
Measure the actual pace and 429 frequency before treating the conditional
9–12-hour four-worker scenario as a completion forecast.


## Eight-game concurrency trial (2026-10-08, 19:41:49 UTC)

The user authorized increasing from four to eight concurrent games. The
previous runner drained gracefully, preserving all 1,111 existing
finalized/stage checkpoints byte for byte. No requests remained in flight,
and the manifest was unchanged. The new runner resumes from
214 finalized turns; it does not regenerate completed stages.

The launcher now persists `--workers 8`. Eight distinct games were observed
with requests in flight. The $300 guard, model settings, two refinement loops,
15 monitoring samples per game, final-thinking history dependencies, and
explicit Sol history caching with 30-minute TTL retain their approved settings.

Immediately before the ramp, four workers averaged about 138 finalized
turns/hour since their restart; the recent 15-minute window was 112/hour.
Doubling that rate would imply roughly 4–5 hours remaining, but this is an
unvalidated scenario. Increased provider throttling can prevent linear scaling;
use the eight-worker observation window before replacing the ETA.

The watchdog restarted with 10-minute initial health monitoring. Its first
scheduled follow-up is 19:52:02 UTC; after at least 10 minutes with progress
and both processes alive it returns to 30-minute checks. `workers-transition.json`
records the current trial; `eight-workers-transition.json` preserves its ramp
evidence. The generic trial selector also retains compatibility with the
previous four-worker record. Nine monitoring tests passed, including eight-worker
trial precedence, cadence, and rejection of a stale driver identity.


## Sixteen-game concurrency trial (2026-10-08, 20:32:16 UTC)

The user authorized increasing from eight to sixteen concurrent games. The
eight-worker runner drained gracefully and exported the partial dataset. All
2,382 previously completed finalized/stage checkpoints remained
byte-identical; no requests remained in flight and the manifest was unchanged.
The new runner resumed from 451 finalized turns. The saved budget
ledger after drain was $34.960204,
including historical uncertain-call reservations and any safely resumable
unsent/backoff slots; active-request reservations are added as the new run starts.

The launcher now persists `--workers 16`; sixteen distinct games were observed
with requests in flight. Models, caching, refinement loops, monitoring sample
selection, finalized-thinking history, 120K input limit, unlimited 429 retries,
and the $300 guard retain the approved settings.

The eight-worker trial averaged about
285 finalized turns/hour
since its restart, with 270 per hour over the last 30 minutes
before the ramp. Further speedup is unvalidated; short observation windows
and the final serial chain of each game can distort completion estimates.

The watchdog restarted with 10-minute initial monitoring; its first scheduled
follow-up is 20:42:28 UTC. It returns to 30-minute checks after at least ten
minutes with finalized progress and both processes alive.
`workers-transition.json` identifies the current trial and
`sixteen-workers-transition.json` preserves all ramp evidence. Independent
review verified the saved checkpoints, matching manifest and pilot fingerprint,
zero pending calls before restart, and generic sixteen-worker monitor behavior
including initial 600-second cadence and return to 1800 seconds after progress.


## Partial and final DVC publication (user authorized 2026-10-08)

The release branch is `codex/progressive-sol25-20261008`. The user asked for
a first DVC-backed commit and PR once most games finish, followed by a final
commit after the long games finish. The selected majority threshold is
**13 of 25 complete games**. The release watcher polls every minute and does
not interrupt inference. It writes durable publication milestones to
`runs/think-progressive-sampled-pilot/release-status.json` and uses an exclusive
release lock. Failures stop publication with a nonzero exit and a saved error.

Publication targets `runs/think-progressive-sol25`, separate from the live
resumable run. Only fully finalized turn folders and their API journals are
archived; current in-flight turns are excluded. The snapshot verifies source,
inference code, tokenizer and chat-template hashes, all retained final-thinking
dependencies, and complete consecutive prefixes within each game. It then
rebuilds the SFT export from the snapshot, preserving oversized thinking as
masked history. Migration evidence and the verified pre-cache-fix archive
are included. This is a versioned data release, not a snapshot of live
processes or active HTTP requests.

DVC data is pushed before its Git pointer is committed. The publisher stages
an explicit allowlist, records the release commit, pushes that exact SHA,
and checks the remote branch before creating/updating the draft PR. Saved
snapshot and pointer hashes prevent resumed publication from using replaced
outputs; an unexpected branch HEAD causes a stop. The final snapshot replaces
the same DVC output, so the earlier Git commit retains the partial version.
The final stage waits for all 1,334 turns and successful driver completion.
Results are summarized in `progressive-sol25-results.json`, including quality
flags and excluded targets.

A real local snapshot/export preflight succeeded with **703 finalized turns**,
699 eligible targets and all finalized-history dependencies intact. The main
39-test validation passed; two additional mocked publication tests then passed
with the release suite (six release tests total). They cover exact-commit push,
crash-after-commit recovery, and rejection of changed HEAD/snapshot/pointer.
DVC 3.67.1 with S3 support runs in an isolated tool environment, preserving
the inference environment. Read-only cloud status confirmed remote access.

The user also requested a detached ten-minute self-wake test.
`wake_progressive_sol25.py` uses `codex queue --thread <current-thread-id>
--message <prompt>` and records the result in `wakeup-test.json`. The preflight
failed because this cloud chat has no local Codex rollout for its thread ID;
therefore delivery is unverified. A detached retry is scheduled for
**21:05:58 UTC**. The release watcher operates independently of this wake test.

The detached wake-up command did run at **21:05:58 UTC**, but returned exit 1:
`thread/queue/add failed: no rollout found for thread id
01a11bfe-682b-7579-a5cc-cd9957971052`. The app confirms this same ID belongs to
the current chat on host `durable`; the local CLI daemon has no corresponding
rollout. This mechanism did not wake or append to the chat. The detached release
watcher remains active independently, with an agent assigned to attach the first
created PR to this chat.


## Production-monitoring failure recovery (2026-10-08)

The full runner first stopped at **965/1,334 turns** when an optional sampled code
regeneration for `sk48-d8078629_p0#41` received three HTTP 400 content-filter
rejections. Its two refinement loops had already produced final thinking. A
subsequent attempt to run that sample's Sol audit received fifteen gateway HTTP
503 errors. Those monitoring failures do not invalidate the generated thinking
or its use in later history.

The generator now treats only the sampled code-regeneration and final-audit
stages as optional: HTTP 429 still retries indefinitely; other errors use the
configured attempt limit, then write an explicit unavailable marker and allow
the finalized thinking/history chain to continue. Core draft, judge, and
refinement failures remain fatal and resumable. An interrupted optional request
is retained with its conservative reservation and a `monitoring_uncertain`
terminal state; resume repairs a missing completion marker without resending it.
Final rows and run summaries report unavailable monitoring separately from
training eligibility. Oversized turns still remain in subsequent context even
when excluded as training targets.

An independent setup review checked the manifest guard, optional request hashes,
reservation handling, crash recovery, and release validation. **Twenty-three
focused offline generator/release tests passed at the first review, then the
suite was expanded to cover report counts and the crash window (**24 tests pass**).
The run manifest now records the single `think_gen/progressive.py` hash change:
manifest SHA-256 changed from
`1bc1744396315bf7d4c16b8aa5b1ec14b5b43687e7942ba9db03a2e0d0200108` to
`84a780bed991a6a47c9de15ed044ca37e6b701acb72c2713a3e5177c5d54c0ef`. The
migration verified all **4,161 completed stage checkpoints** unchanged and left
source logs, tokenizer, template, model settings, and completed training targets
intact.

The resumed runner finalized the blocked sample (**966/1,334**), then a required
draft request for `s5i5-18d95033_p0` failed three times with HTTP 503 from the
OpenRouter gateway, stopping the run safely. A no-auth GET to the provider's
models endpoint also returned HTTP 503, confirming current provider routing is
unavailable. The ledger is about **$107.51 including uncertain reservations**;
the $300 guard remains in place. The 16-worker runner is stopped and resumable.

The partial DVC transfer continues; local process counters show it reading and
indexing snapshot objects, but remote completion is unverified. No Git commit or
draft PR has been created yet.

## Failed-HTTP billing reconciliation (2026-10-09)

The runner later reached **1,004/1,334** turns (999 eligible, five oversized),
then stopped at the $300 local guard. The old ledger was **$296.814721**, but
**$229.411247** was held as worst-case reservations for 250 explicit failed HTTP
responses: 218 HTTP 503, 24 HTTP 429, and eight HTTP 400. These have no model
response or usage. The confirmed successful-response usage yields an estimated
**$67.403473** ($5.286389 Flash via OpenRouter, $62.117084 Sol via direct OpenAI).
The OpenRouter component uses returned usage cost; Sol is priced from returned
token counts. This is still an estimate, not an account billing statement.

The stopped-run migration wrote `http-failure-billing-adjustments.json`, with
each original attempt's path, SHA-256, HTTP status, original reservation and
zero effective charge. Its record is `http-failure-billing-migration.json`.
The migration preserved the SHA-256 inventory of all **4,591 attempts, 4,342
stage completions, and 1,004 final turns**, and updated the source fingerprint
and pilot validation for the reviewed accounting change. The monitor now reports
**$67.403473** in effective successful-response cost and **zero** outstanding
reservations. Unknown transport failures and invalid successful responses keep
their conservative charge. HTTP 429 releases its reservation while waiting and
re-reserves before sending.

Read-only account spend probes to OpenRouter `/api/v1/key` and OpenAI
`/v1/organization/costs` both returned an Envoy HTTP 503 transport failure on
2026-10-09, so neither account's actual billed spend could be verified. The
partial snapshot remains pinned at 939 turns/13 games for DVC publication;
upload completion, Git commit and draft PR are still pending.

All 4,341 successful calls in the current run ledger finished on **2026-10-08
16:13–22:48 UTC**. The user reports dashboard spend for that day of about **$138
OpenAI** and **$7 OpenRouter**, versus this run's $62.117084 and $5.286389
estimates. Account totals can include other requests and different accounting;
without provider-side request/project attribution the difference cannot be
assigned to this run. The observed 503 is specifically the Codex cloud proxy's
Envoy failing its `cloudflare_https_tunnel` upstream connection before provider
headers, not evidence of a 503 response from either model provider.

## Offline handoff checkpoint (2026-10-09)

Generation is stopped and there are no pending attempts. The run has 1,004
finalized turns; the remaining 330 should resume from those durable checkpoints
once both API hosts are reachable. The corrected effective ledger is
$67.403473 with zero outstanding reservations. The partial publication watcher
also stopped after S3 `PutObject` HTTP 503; its pinned 939-turn snapshot remains
local, but its root `.dir` object is missing remotely. No Git commit or PR was
published at the time of this checkpoint.

The offline archive is
`ARC3-Inference/runs/sol25-handoff-20261009.tar.zst` (54 MiB), SHA-256
`84421cf8d1ca0beea3e7fcc8ac6fedfe887d6e8162d68f14948090549c861cd6`.
It contains the live run and journals, frozen partial snapshot, source logs,
student tokenizer/template, and local DVC cache. `zstd -t` and a full tar index
read succeeded. It is **only on this workspace**; a different VM cannot recover
it without copying the archive or completing the DVC push. Local Git commits
likewise require this worktree or a later Git push to transfer to another VM.

In a new chat with this same workspace, first check the archive checksum and
`driver_status.json`, then probe `https://openrouter.ai/api/v1/models` and
`https://api.openai.com/v1/models`. Once both routes work, run
`ARC3-Inference/.venv/bin/python ARC3-Inference/scripts/run_progressive_sol25.py`.
The launcher uses 16 workers, the $300 estimated-spend guard, 12 attempts per
non-429 stage, and unlimited 429 backoff. It verifies the migrated manifest and
pilot fingerprint and reuses existing turn checkpoints. Recheck the effective
ledger via `ARC3-Inference/scripts/monitor_progressive_sol25.py --once`.
After connectivity returns, rerun
`ARC3-Inference/scripts/manage_progressive_sol25_release.py --threshold 13
--interval 60` from the repository virtualenv with Git write access. It checks
the pinned snapshot hash before retrying DVC, then commits and opens the draft
PR only after the complete DVC object tree is uploaded. The final 1,334-turn
snapshot and PR update follow when generation completes.
