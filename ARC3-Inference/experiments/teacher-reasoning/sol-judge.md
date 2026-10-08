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

| metric | draft | + fact-fix | refine 1 |
| --- | ---: | ---: | ---: |
| coverage of Sol's words | 0.88 | 0.88 | **0.99** |
| code leads to call | 1.00 | 1.00 | 1.00 |
| code disagreements / record | 0.05 | 0.05 | **0.00** |
| fact grounded | 0.35 | 0.45 | **0.70** |
| fact errors / record | 1.00 | 0.75 | **0.30** |
| call functionally same | 0.55 | 0.55 | 0.55 |

One refine pass roughly halves the fact errors again and takes coverage to
near-complete. Call equivalence is flat — it is downstream of the student's
code-reproduction, which the refine (aimed at the thinking's content) does not
change. After refine 1, 7 of 20 records are still imperfect (6 with one
lingering fact error, 1 at 0.88 coverage); a second pass re-feeds the feedback
to those.

## Cost of the full 25-game dataset

Scale, from `runs/gpt61sol-features-25games`: **1,334** model responses,
**1,171** of them need generated thinking (tool call and non-zero reasoning;
the other 162 get empty thinking, as Sol had). Context averages **~62K tokens**
per request — this dominates the cost, since every flash and Sol call re-sends
it.

Decisions that shape the cost (prices below are a **GPT-5-class stand-in**;
`gpt-6.1-sol` pricing is not known here, so substitute it):

- **One combined Sol judge, not three.** Words + code + fact return in a single
  JSON from one call, so the ~62K context is read twice per request (combined
  judge + final equivalence) instead of four times. Run the combined judge at
  **xhigh** (keeps fact-check depth; one effort covers all three) and
  equivalence at **high**.
- **Refine only the flagged records.** On dev, ~80% fail the first pass, so the
  refine flash call fires on ~0.8 of records. It is a flash call, so the
  first-pass rate barely moves the total — the two Sol calls run on every record
  regardless.
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

Resulting estimate (stand-in pricing, ±~40% for unknown Sol price, cache rate
and per-turn increment):

| | with progressive stretch caching |
| --- | ---: |
| Sol input (cold stretch-starts + cheap warm reads) | ~$30–40 |
| Sol output (combined xhigh + equivalence) | ~$59 |
| Flash (draft + ~0.8 refine + regen) | ~$21 |
| **total** | **~$110** (combined xhigh) / **~$85** (combined high) |
| wall-clock | ~1–1.5 h (bounded by the longest stretch, run wide) |

Once the context is cached away, **output becomes the floor**, so the combined
judge's xhigh-vs-high effort is the main remaining cost knob (~$25).

### Not yet implemented
The combined judge, the stretch-aware serial-within/parallel-across scheduler,
the 1h-cache flag and a `--dry-run` token/cost projector are design decisions
from this analysis, to build before the full 25-game run. The dev-set results
above use the current separate-judge code.

## Limits
- Prices and the cache discount/TTL for `gpt-6.1-sol` are unknown here; the
  dollar figures are stand-ins.
- dev20 is stratified to be hard (across token bins), so its ~80% first-pass
  fail rate is likely a ceiling; the full set's many short requests should pass
  more often.
- The judge is a single Sol sample per check, with no measure of agreement with
  a person; the annotation page exists to spot-check it by hand.
- Dev split only so far; the eval split is held out for a real measurement.
