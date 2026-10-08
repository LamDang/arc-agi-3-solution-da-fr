# Judging and refining generated thinking for GPT-6.1 Sol

How we score the thinking that `think_gen` generates for GPT-6.1 Sol, and the
refine loop that uses those scores to improve it. Sol's real reasoning is
hidden, so unlike the qwen3.8-max calibration
([think-gen-calibration.md](think-gen-calibration.md)) there is nothing to diff
against. Written 2026-10-08. Code: [`think_gen/judge_sol.py`](../../think_gen/judge_sol.py),
[`think_gen/refine.py`](../../think_gen/refine.py), prompts in
[`think_gen/prompts.py`](../../think_gen/prompts.py).

## The four checks

Each generated record is judged on four things, each against something that IS
known about the step (the real thinking is not). All four run on **gpt-6.1-sol
through the Responses API with the teacher's own context — the frames, tool
outputs and images — in front of the judge instruction**, so the judge reads
the thinking and the code against the real game state.

1. **Code consistency** — does the thinking's plan match the python code Sol
   ran? *Lenient*: the code may inspect, print, assert or compute more than the
   thinking states; that is expected and is not a disagreement. Only a
   different decision counts (a different action, target or value, or a step the
   thinking commits to that the code skips). → `leads_to_call`, `disagreements`.
2. **Coverage of Sol's own words** — how much of Sol's stated `reasoning`,
   `description` and `summary` the thinking covers, and any contradiction.
   → `covered` (fraction), `contradictions`.
3. **Fact-check** — claims in the thinking the real state does not support
   (wrong coordinates, counts, colours, mechanics). The thinking is exploratory
   working: *a figure it states and then corrects is not an error* — only
   claims the final plan still rests on are judged. → `errors`, `grounded`.
4. **Call equivalence (functional)** — regenerate a python call from the
   thinking with the student model (flash) in the code-only request, then judge
   whether it is *functionally* the same as Sol's code: same effect on the game,
   not the same text or printed output. → `functionally_same`, `differences`.

### Why these settings

- **All four on Sol with the full context.** The first version used
  `gpt-6-luna` on the thinking alone (no context). It was too severe on code
  consistency (it flagged every inspection the code did but the thinking did
  not mention) and could not ground coverage or facts in the real board.
  Giving the judge the context and making code lenient moved
  `code_leads_to_call` from 0.65 to 1.00 and `code_disagreements` from 2.55 to
  ~0 per record.
- **Fact-check ignores self-corrected working.** Early fact-checks flagged
  intermediate values the thinking itself revised a few lines later (e.g. "width
  9 inclusive" corrected to 10). Judging only the standing claims raised
  `fact_grounded` from 0.35 to 0.45 before any refine.
- **Regeneration stays on the student (flash).** The call check is a
  hint-removal test: it only means something if the *student* reproduces Sol's
  call from the thinking. Regenerating with the judge model would measure the
  judge.

## The refine loop

Judging is not just a score; its feedback drives a second pass
([`refine.py`](../../think_gen/refine.py)). For each record the judge flagged
(coverage below 1, a code disagreement, or an ungrounded fact), the student is
given its own first draft plus the feedback — each check's one-line note and
the specific points (missed coverage points, code disagreements, fact errors as
`claim → actual`) — and asked to keep what the draft got right and fix only
what is raised. Records the judge passed are copied through unchanged. Output
rows match `think_gen.generate`, so `judge_sol` and the annotation page read a
refined run the same way. The loop can be repeated: re-feed the feedback to the
records still imperfect after a pass.

## Results on the dev set (dev20, 20 records)

| metric | draft | + fact-fix | refine 1 | refine 2 |
| --- | ---: | ---: | ---: | ---: |
| coverage of Sol's words | 0.88 | 0.88 | **0.99** | 0.96 |
| code leads to call | 1.00 | 1.00 | 1.00 | 1.00 |
| code disagreements / record | 0.05 | 0.05 | **0.00** | 0.00 |
| fact grounded | 0.35 | 0.45 | 0.70 | **0.95** |
| fact errors / record | 1.00 | 0.75 | 0.30 | **0.05** |
| call functionally same | 0.55 | 0.55 | 0.55 | 0.60 |
| records fully perfect | 4/20 | – | 13/20 | **15/20** |

(A record is "perfect" when coverage = 1, no code disagreement, and the facts
are grounded.)

Each refine pass re-feeds the judge feedback to the records still imperfect
after the previous pass (16 after the draft, 7 after refine 1). The gains are
real but show a **trade-off**: refine 1 took coverage to near-complete and
halved fact errors; refine 2, aimed at the 7 remaining (6 with one lingering
fact error, 1 at 0.88 coverage), nearly eliminated fact errors (0.95 grounded,
0.05 errors/record) but **lost a little coverage** (0.99 → 0.96) — correcting a
stubborn fact on a record can drop a minor covered point. Call equivalence stays
flat across passes: it is downstream of the student's code-reproduction, which
the refine (aimed at the thinking's content) does not change. Two passes reach
15/20 perfect; a third would chase single points with more coverage risk than
gain.

## Cost of the full 25-game dataset

Scale, from `runs/gpt61sol-features-25games`: **1,334** model responses,
**1,171** of them need generated thinking (tool call and non-zero reasoning;
the other 162 get empty thinking, as Sol had). Context averages **~62K tokens**
per request — this dominates the cost, since every flash and Sol call re-sends
it.

Decisions that shape the cost (prices below are a **GPT-5-class stand-in**;
`gpt-6.1-sol` pricing is not known here, so substitute it):

- **One combined Sol judge, not three.** Words + code + fact return in a single
  JSON from one call. Run it at **xhigh** (keeps fact-check depth; one effort
  covers all three) and the equivalence at **high**.
- **Two refine passes, then stop.** On dev the loop earns its keep for two
  passes and no more: draft → judge → refine 1 → judge → refine 2 → equivalence
  (see the results above — fact grounding 0.45 → 0.70 → 0.95, 4/20 → 13/20 →
  15/20 perfect; a third pass only fights the coverage metric). So there are
  **two combined-judge rounds** (after the draft, and after refine 1 to drive
  refine 2) and **one final equivalence**. Refine runs only on the records the
  previous round flagged: refine 1 on ~80% (draft failures), refine 2 on ~35%
  (refine-1 failures); the second judge round runs on the ~80% that refine 1
  produced. Refines are flash calls, so these rates barely move the total — the
  cost is in the Sol judge rounds, which share each request's cached context.
- **Run in game-turn order, split per compaction (history stretch).** Within a
  stretch the context is append-only, so processing requests in order lets each
  call hit the previous call's cached prefix; only the new turn (~3–5K tokens)
  is fresh. A compaction invalidates the prefix anyway, so stretches are
  independent: serial within a stretch, **parallel across all ~60–75 stretches**
  of the 25 games. This is also what makes the cache valid (the prefix is stable
  only between trims).
- **Use the 1h prompt-cache tier** so a stretch that runs 60–90 min does not
  lose the prefix to a 5–10 min TTL. The write premium is small; reads stay
  cheap. Run the equivalence call immediately after the combined judge (same
  turn, warm prefix).

Resulting estimate for the full 1,171-request set, **two refine passes**
(stand-in pricing, ±~40% for unknown Sol price, cache rate, per-turn increment
and xhigh reasoning volume):

| | tokens / calls | ~$ |
| --- | --- | ---: |
| Sol input — judge 1 (all, progressive cache) | ~stretch-cold + warm reads | ~$20 |
| Sol input — judge 2 (~80%) + equivalence (all), same cached context | ~130M cached | ~$16 |
| Sol output — judge 1 combined xhigh (1,171 × ~4K) | ~4.7M | ~$47 |
| Sol output — judge 2 combined xhigh (~940 × ~4K) | ~3.7M | ~$37 |
| Sol output — equivalence (1,171 × ~1K) | ~1.2M | ~$12 |
| Flash (draft 1,171 + refine1 ~940 + refine2 ~410 + regen 1,171) | ~3,700 calls | ~$22 |
| **total (combined xhigh)** | | **~$155** |
| **total (combined high)** | | **~$110** |
| wall-clock | | ~1.5–2 h (longest stretch, run wide) |

The second refine pass adds ~$45 over a single pass — almost entirely the
second judge round's **xhigh output**, since its input reuses each request's
already-cached context. Once the context is cached away, output is the floor, so
the combined judge's **xhigh-vs-high** effort is the main remaining knob (~$45
across the two judge rounds).

### Not yet implemented
The combined judge, the stretch-aware serial-within/parallel-across scheduler,
the 1h-cache flag and a `--dry-run` token/cost projector are design decisions
from this analysis, to build before the full 25-game run. The dev-set results
above use the current separate-judge code.

## Reproducibility

The pipeline is DVC stages (`dvc.yaml`, params under `think_gen:` in
`params.yaml`): `tg_generate → tg_judge1 → tg_refine1 → tg_judge2 → tg_refine2
→ tg_final`. Each stage writes a separate, non-nested directory under
`think_gen.out` (`draft/ judge1/ refine1/ judge2/ refine2/ final/`) so DVC
tracks them independently and `dvc repro` reruns only what changed. To run a
different sample, point `think_gen.manifest` at another manifest and
`dvc repro tg_final` (or `dvc exp run`). The source run holds the context and is
itself DVC-tracked, so it is a dependency, not duplicated.

**Request log.** Every Sol and flash call appends one JSON line to the stage's
`requests/<game>.jsonl`, so a run is fully retraceable without storing the
~62K-token context on every line:

- `request` keeps the call's metadata (api, model, effort/reasoning,
  temperature, token caps), the **appended instruction verbatim** (the judge or
  reconstruct prompt — the part that varies), and a **reference** to the context
  prefix: its message count, char length and a sha1. The prefix itself is the
  source record's context (named by `key`, e.g. `bp35-…_p0#32`), which lives in
  the DVC-tracked source run; the sha1 lets a reconstruction be verified.
- `response` keeps the full content, tool calls, `finish_reason`, timing and the
  complete `usage` (prompt/cached/completion/reasoning token counts, and cost
  for flash), including the final give-up on failure.

The results themselves (generated and refined thinking rows, judge verdicts,
per-stage `summary.json`) are the stage outputs, as before.

## The refine prompt across passes

Refine 2 edits refine 1's output as its new draft and is given refine 1's fresh
judge verdicts as feedback. So the **round-1 correction is kept** (it is the
draft), but the round-1 **feedback text is not re-shown** — only the issues
still open on the round-1 output. This is likely why round 2 traded a little
coverage for the fact fixes: re-fixing a fact, the student is not reminded what
it already had right. Carrying the earlier feedback forward (an accumulated
"keep all of this, only these remain") is an untested lever against that
regression.

## Limits
- Prices and the cache discount/TTL for `gpt-6.1-sol` are unknown here; the
  dollar figures are stand-ins.
- dev20 is stratified to be hard (across token bins), so its ~80% first-pass
  fail rate is likely a ceiling; the full set's many short requests should pass
  more often.
- The judge is a single Sol sample per check, with no measure of agreement with
  a person; the annotation page exists to spot-check it by hand.
- Dev split only so far; the eval split is held out for a real measurement.
