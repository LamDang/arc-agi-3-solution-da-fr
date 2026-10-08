# gpt-6.1-sol on the 25 official games with notes, hints and stated reasoning

## Question

gpt-6.1-sol with the base agent and dfranzen's settings won 24 of the 25 official games,
182/183 levels, mean 99.1, for $41.67 ([base-gpt61sol-20games.md](base-gpt61sol-20games.md),
[base-gpt61sol-5games.md](base-gpt61sol-5games.md)). The one loss, sk48, spent 437 actions
and 100 minutes stuck on level 5, burned ~130 actions on purpose to force a reset, and ran out
of time on level 8. Four harness changes had been tested since, each in isolation, on
replayed requests or three games:

| change | setting | evidence before this run |
| --- | --- | --- |
| stated reasoning in every tool call | `ARC3_PYTHON_RATIONALE=1` | 30 replayed requests: fields filled 30/30 (strict schema), coherent with the code ([python-rationale-replay.md](python-rationale-replay.md)) |
| handover note before the history cut | `ARC3_NOTE_COMPACTION_TOKENS=120000` | 3-game replay: 8 correct notes, no re-exploring after them ([note-compaction.md](note-compaction.md)) |
| no deliberate budget burn | `ARC3_NO_BUDGET_BURN=1` | not run ([stuck-detection.md](stuck-detection.md#caveats)) |
| step-back note when positions repeat | `ARC3_REPEAT_HINT=1` | 9 replayed stuck moments: more stock-taking, fewer burns; little change on gpt-6.1-sol ([stuck-detection.md](stuck-detection.md#in-the-harness-arc3_repeat_hint)) |

With all four on, over all 25 games: does the agent still win them, does it get through
sk48-like stuck levels faster, and what do the notes, hints and reasoning fields do over whole
games?

## Setup

| | baseline (`base-gpt61sol-dfranzen` + `base-gpt61sol-20games`) | this run |
| --- | --- | --- |
| run | two runs, 5 + 20 games | `runs/gpt61sol-features-25games` (DVC), one run |
| code | `6bd8eef` / `d59dbbb` | `9b09a8c` (harness as `main` at `1e05a1d`) |
| launched with | `scripts/dvc_eval.py` | `experiments/gpt61sol-features/run.sh` (the same `dvc_eval.py` call, plus reruns) |
| started | 2026-10-07 11:09 and 12:27 UTC | 2026-10-07 20:58 UTC, finished 23:00 |
| games, passes | 25 official, 1 pass | same |
| model calls at once | 5, then 10 | 10 |
| limits per game | 500K output tokens, 240 minutes | **300K** output tokens, 240 minutes |
| reruns | - | each game under 100 played again, up to twice, each attempt in its own directory; none was needed |
| `OPENAI_REASONING_EFFORT` | `xhigh` | `xhigh` |
| `OPENAI_REASONING_SUMMARY` | `auto` | `detailed` |
| `LOCAL_ANALYZER_MAX_OUTPUT`, `ARC3_SEND_REASONING_DETAILS` | `0`, `1` | same |
| `ARC3_PYTHON_RATIONALE` | off | **`1`**: `python(description, reasoning, code)`, strict schema |
| `ARC3_NOTE_COMPACTION_TOKENS`, `_KEEP_TURNS` | off: the trimmer drops the oldest half at ~130K estimated, with no summary | **`120000`**, **`10`** |
| `ARC3_NO_BUDGET_BURN` | off | **`1`** (`EXPOSE_RESET` is off, so the line is active) |
| `ARC3_REPEAT_HINT` | off | **`1`**: 3 positions seen 3 times each, 10 turns between messages |

Everything else is `params.yaml` (dfranzen's) through OpenAI's Responses API, as in the
baseline. A separate reviewer diffed both baselines' `eval_settings.json` against this run's
settings before it got far: they differ only in the rows above.

**Smoke test.** ar25 alone with the note threshold lowered to 20000 won 8/8 in 7.8 minutes for
$0.40. Its one note came at step 12 and cut the history from 41K to 10K tokens; every call had
both reasoning fields; scoring and packing ran.

**Driver.** The reviewer found three bugs in the first version of `run.sh` that would have lost
the reruns (a crashed game makes `make score_run` fail, which stopped the driver; an unreadable
result was taken as "every game at 100"; a rerun game at its 240-minute limit would have been
cancelled by the job's own deadline). They were fixed 10 minutes in and the driver swapped
without stopping the games. No game crashed and no rerun was needed, so none of the fixes was
exercised.

## Results

| game | levels | actions per level (bold: above human) | baseline | output tokens (reasoning) | baseline | prompt tokens | baseline | cost | baseline | notes | hints |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ar25 | 8/8 | 17 11 40 22 28 54 37 47 | 17 13 41 22 28 54 37 47 | 15.1K (6.8K) | 11.4K | 1.05M | 0.82M | $0.43 | $0.32 | 0 | 0 |
| bp35 | 9/9 | 15 **69** 36 23 30 51 67 53 92 | 15 68 59 23 31 59 53 108 89 | 192.5K (128.1K) | 155.1K | 8.56M | 17.76M | $4.48 | $5.65 | 6 | 1 |
| cd82 | 6/6 | 49 6 19 14 13 16 | 25 6 17 14 13 16 | 23.5K (13.1K) | 8.7K | 2.05M | 0.94M | $0.68 | $0.32 | 0 | 2 |
| cn04 | 6/6 | 14 31 24 32 59 43 | 14 30 24 34 48 43 | 20.8K (9.0K) | 17.1K | 1.45M | 2.13M | $0.54 | $0.57 | 0 | 0 |
| dc22 | 6/6 | 26 46 47 87 126 161 | 25 42 45 70 136 170 | 100.4K (63.9K) | 53.4K | 6.22M | 12.25M | $2.59 | $3.10 | 3 | 0 |
| ft09 | 6/6 | 5 7 14 16 21 13 | 4 7 14 16 21 15 | 9.6K (3.4K) | 7.7K | 0.38M | 0.38M | $0.24 | $0.20 | 0 | 0 |
| g50t | 7/7 | 26 31 64 31 50 49 43 | 25 32 64 32 50 52 56 | 19.3K (9.7K) | 25.8K | 1.92M | 4.40M | $0.62 | $1.18 | 0 | 0 |
| ka59 | 7/7 | 28 63 34 46 27 69 143 | 25 39 38 44 20 52 107 | 53.7K (34.4K) | 23.3K | 3.26M | 3.51M | $1.33 | $1.03 | 1 | 0 |
| lf52 | 10/10 | 9 39 49 50 87 102 145 69 112 45 | 9 53 46 52 89 111 168 68 121 61 | 146.1K (92.3K) | 137.0K | 8.75M | 23.86M | $3.78 | $6.43 | 5 | 0 |
| lp85 | 8/8 | 7 9 16 13 10 20 8 14 | 9 11 16 25 10 23 5 11 | 23.0K (10.5K) | 13.9K | 1.46M | 1.85M | $0.57 | $0.53 | 0 | 0 |
| ls20 | 7/7 | 18 97 41 51 **102** 72 67 | 16 79 43 49 52 86 93 | 63.8K (35.2K) | 33.4K | 3.99M | 3.69M | $1.70 | $1.17 | 2 | 0 |
| m0r0 | 6/6 | 15 23 66 11 44 50 | 15 38 61 11 51 53 | 21.1K (9.0K) | 17.7K | 1.58M | 1.61M | $0.59 | $0.53 | 0 | 0 |
| r11l | 6/6 | 3 10 11 13 17 17 | 3 12 11 13 22 17 | 24.6K (15.5K) | 21.9K | 0.97M | 1.72M | $0.52 | $0.56 | 0 | 0 |
| re86 | 8/8 | 20 36 47 42 90 59 105 187 | 20 36 47 46 90 65 109 202 | 49.4K (28.2K) | 35.0K | 4.37M | 3.52M | $1.42 | $0.98 | 1 | 0 |
| s5i5 | 8/8 | 13 26 37 30 28 25 67 36 | 13 26 38 30 28 25 55 36 | 53.9K (33.6K) | 37.1K | 2.89M | 3.47M | $1.34 | $1.20 | 1 | 0 |
| sb26 | 8/8 | 9 15 15 15 17 19 17 **19** | 9 15 15 15 17 19 17 17 | 6.7K (1.9K) | 4.8K | 0.35M | 0.57M | $0.21 | $0.22 | 0 | 0 |
| sc25 | 6/6 | 23 5 17 31 48 39 | 22 5 12 32 39 35 | 23.8K (11.7K) | 7.8K | 1.98M | 1.17M | $0.80 | $0.37 | 1 | 0 |
| sk48 | **8/8 won** | 16 43 66 29 **282** 102 56 40 | 20 34 62 30 437 125 53, 39 on L8 (7/8) | 226.0K (163.1K) | 278.6K | 10.65M | 20.36M | $5.07 | $7.72 | 6 | 5 |
| sp80 | 6/6 | 4 9 20 31 28 35 | 7 9 17 31 25 34 | 36.2K (21.1K) | 23.0K | 1.16M | 1.29M | $0.68 | $0.52 | 0 | 0 |
| su15 | 9/9 | 9 16 18 27 5 11 5 7 13 | 11 13 14 10 17 16 5 35 15 | 100.7K (71.3K) | 53.1K | 4.71M | 3.20M | $2.25 | $1.34 | 2 | 0 |
| tn36 | 7/7 | 18 10 11 16 19 20 44 | 10 11 12 14 19 32 47 | 24.3K (12.4K) | 14.9K | 1.32M | 1.36M | $0.57 | $0.46 | 0 | 0 |
| tr87 | 6/6 | 17 26 26 21 52 24 | 25 32 28 21 31 24 | 27.9K (12.6K) | 17.7K | 1.18M | 1.44M | $0.57 | $0.49 | 0 | 0 |
| tu93 | 9/9 | 18 10 19 33 29 28 14 21 29 | 18 10 24 18 29 28 14 21 30 | 52.0K (25.2K) | 33.8K | 3.73M | 3.19M | $1.35 | $0.92 | 1 | 0 |
| vc33 | 7/7 | 5 9 23 24 67 22 49 | 11 8 23 25 68 20 49 | 26.5K (15.1K) | 21.2K | 1.10M | 1.38M | $0.59 | $0.57 | 0 | 0 |
| wa30 | 9/9 | 27 60 76 54 94 53 38 129 141 | 27 70 83 60 104 51 49 133 70 | 155.0K (114.2K) | 124.8K | 7.57M | 19.36M | $3.57 | $5.29 | 4 | 0 |

| | this run | baseline |
| --- | --- | --- |
| games won, levels | **25/25, 183/183** | 24/25, 182/183 |
| mean score | **100.0** | 99.1 |
| actions | 7,202 | 7,405 |
| levels above the human baseline | 4 (bp35 L2, ls20 L5, sb26 L8, sk48 L5) | 3 (bp35 L2 and L3, sk48 L5) |
| model requests | 1,334 | 1,920 |
| output tokens (reasoning) | 1.50M (0.94M) | 1.18M (0.85M) |
| prompt tokens (cached) | 82.7M (93%) | 135.2M (95%) |
| largest prompt | 120K | 129K |
| cost | **$36.50** | $41.67 |
| wall clock | 2 h 02 | 23 min + 4 h 14 |

Tokens are summed from the `usage` of the response records; cost at $2 / $0.10 / $2.50 / $10
per million uncached input, cached input, cache-write and output tokens. The per-game minutes
are not comparable between the runs: all 25 games start together and share 10 model slots, so
a quick game's clock includes its wait.

- **Every game won.** sk48, the baseline's one loss, solved level 5 in 282 actions against 437
  and won 8/8 at 226K output tokens, under the 300K cap, in 122 minutes. No game reached a
  limit and no rerun was needed.
- **The same play, with more output.** Actions per level stay close to the baseline game by
  game (7,202 against 7,405 in all). Output tokens are 27% higher: the reasoning fields add
  about 130K tokens (524K characters over 1,300 calls), and the hidden reasoning is 10% higher
  (0.94M against 0.85M).
- **Cheaper overall.** Prompt tokens fell 39% and requests 31%. The long games account for it:
  lf52 8.75M prompt tokens against 23.86M, wa30 7.57M against 19.36M, bp35 8.56M against
  17.76M. After a note only the last 10 turns and the note stay: the next request was 39-76K
  tokens (median 50K, 33 notes), where the trimmer kept 59-68K, so every later request re-sends
  less history. The short games, with no note, cost a
  little more (the longer output).

## The four settings

**Reasoning fields.** All 1,300 `python` calls had both `description` and `reasoning`; mean
reasoning 404 characters. Note requests are counted apart.

**Handover notes.** 33 notes in 12 games (sk48 and bp35 6 each, lf52 5, wa30 4, dc22 3), every
one a comments-only `python` call kept in place of the dropped turns. No prompt fell by more
than 30% except right after a note: the trimmer never cut first (largest prompt 120K). A reading of sk48 level 5 found that its three
notes there kept detailed TRIED lists with "do not repeat" lines, and that after each cut it did
not repeat a listed probe (one black-square click was retried twice, argued each time as a new
board state). One note's PLAN said "Consider report of partial completion honestly", and the
next turn ended with "I cleared the first four levels, but couldn't complete level 5"; the
harness carried on and the level was solved later.

**No budget burn.** On sk48 level 5 there were no game overs and no UP/DOWN runs longer than
two moves, where the baseline ran out the budget with ~130 actions to force a reset. The model
backtracked with UNDO instead: 94 of the first 227 actions on the level were UNDO, in four
bursts (actions 231–284, 312–331, 346–369, 376–377). Each burst was a planned rewind to a chosen
position before a new test, not a way to waste moves:

1. 54 UNDOs back to the level's start after a failed setup. On the first one it wrote: "safely
   unwind the unsuccessful setup without spending the budget on a death". It also checked that
   UNDO stops at the level boundary.
2. 20 UNDOs back to the start ("exactly twenty successful moves since the original
   configuration"), to run a 13-move plan found by a search over a bead-motion model it had
   checked against all 87 recorded moves.
3. 13 UNDOs to take that plan back after it failed, then a crossing from below; 5 more to set up
   a mouse test.
4. 2 UNDOs to put the arm right beside the black square for a click from that side.

It counted the moves to rewind and checked the board after each burst. Not all of it was
efficient: a rewind to the start costs as many actions as the moves it undoes, and one burst was
sent as one call per UNDO, which timed out after 13. Every idea tested this way failed, and the
level was solved after action 380 by a different route. UNDO counts as actions like any move; the
level took 282 actions in all.

**Step-back hints.** 8 in 3 games: sk48 level 5 (steps 47, 56, 66, 74, 82), bp35 level 7
(step 45, after two deaths) and cd82 level 1 (steps 9 and 19). Read turn by turn up to step 82:

| game | step | what followed | verdict |
| --- | --- | --- | --- |
| sk48 | 47 | One line ("more empty crossings and clicks are unhelpful"), then more probes of the black square | neutral |
| sk48 | 56 | Ignored; carried on rewinding to the level start (20 UNDOs, then 33 more) | neutral |
| sk48 | 66 | Kept probing; two turns later rebuilt the physics from the move history and searched a new idea | neutral, slightly helpful |
| sk48 | 74 | An audit of past clicks, then 4 more no-op clicks | neutral |
| bp35 | 45 | Ignored; carried on with its checked route and solved level 7 at step 49 | false alarm, no harm |
| cd82 | 9, 19 | A bounded experiment ("not an attempt to exhaust the budget"), then the untested elements; level 1 in 49 actions (human 55) | false alarm, no harm |

None of the hint turns listed the facts, inventoried the elements or reasoned backward as the
message asks. Up to step 82 sk48 had the baseline's blind spot: its model of the level treated
the red cars as fixed and missed pushing loose blue cars into them from below, the move that
solved the level in the baseline; at step 71 it was one row from it. The steps after 82, where
the level was solved, have not been read.

## Transcripts and status page

- [gpt-6.1-sol Feature Run Transcripts](https://claude.ai/artifact/9gisaSUjo4JNEWDD4KbLBX): one
  card per analysis step, with the reasoning fields and the notes in place.
- [gpt-6.1-sol Feature Run](https://claude.ai/artifact/W7VFY5UDpWCCpZCnuWHtTF): the status page
  kept during the run, with the check-in log (`experiments/gpt61sol-features/checkins.json`),
  rebuilt by `experiments/gpt61sol-features/report.py`.

```bash
dvc pull runs/gpt61sol-features-25games.dvc
uv run --no-sync python scripts/pack_run.py unpack runs/gpt61sol-features-25games
uv run --no-sync python scripts/base_transcripts/build.py \
  "runs/gpt61sol-features-25games=All 25 official games" <out.html> \
  --title "gpt-6.1-sol Feature Run Transcripts"
```

## Caveats

- Four changes at once, one pass each. The run shows the bundle wins all 25 and costs less;
  it does not say which change does what. sk48's level 5 swings widely between runs (the
  note-compaction replay won the whole game in 285 actions, before any note), so 282 against
  437 is within run-to-run variance.
- The cost drop comes from the notes' shorter kept history (10 turns), which is a setting of the
  note, not of the note's content; a plain trimmer that kept less would also cut prompt tokens.
- The hint readings cover sk48 up to step 82 and the bp35 and cd82 hints; the later sk48 turns
  were not read.
- The output cap was 300K against the baseline's 500K. No game reached it; sk48 came closest
  at 226K.
- The reasoning summary was `detailed` against `auto`; it changes which responses carry a
  summary, not the tokens or the cost.

## Conclusions

- With the four settings on, gpt-6.1-sol won all 25 official games, 183/183 levels, mean 100.0,
  for $36.50 in 2 hours, against 24/25, mean 99.1, $41.67 without them. sk48, the baseline's loss,
  was won.
- The play itself is about the same: actions per level match the baseline game by game. The
  reasoning fields add 27% output tokens and give a stated intent on every call.
- The handover notes made the long games cheaper (39% fewer prompt tokens overall) and kept the
  record of tried probes across cuts, which was the gap the compaction analysis found.
- The no-budget-burn line stopped deliberate budget deaths. The model backtracked with UNDO
  instead, rewinding to a chosen position before each new test, which is legitimate exploration
  rather than waste. It could be cheaper (fewer rewinds all the way to the start), but nothing
  here calls for a line against UNDO.
- The step-back hint rarely changed what gpt-6.1-sol did and fired twice as a false alarm,
  without harm.
