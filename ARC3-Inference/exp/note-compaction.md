# Note compaction: a handover note before the cut

## Question

[gpt61sol-compaction.md](gpt61sol-compaction.md) found that dropping the oldest half of
gpt-6.1-sol's history at about 115-128K tokens rarely costs anything, because its rules live
in its retained functions. The one visible cost was in sk48, already stuck on level 5: the
list of probes it had tried left with the dropped turns, and it re-tried some. The idea
tested here: at 110K, before cutting, the harness tells the model the context is about to be
cut to its last 20 turns and asks it to write everything it needs to continue as a `python`
call whose code is only comments. The code is kept in context and not run (a comment, not a
string, so nothing would be printed back). The turn then goes on with its normal message.
Where would this fire in the logged games, what do the notes contain, and does a game go
better with them?

## The setting

`ARC3_NOTE_COMPACTION_TOKENS` (0, the default, is off; 120000 with dfranzen's settings) and
`ARC3_NOTE_COMPACTION_KEEP_TURNS` (default 10), described in
[CONFIGURATION.md](../CONFIGURATION.md#note-compaction). At a turn start whose estimated
prompt reaches the threshold, the harness sends the history with `NOTE_COMPACTION_PROMPT`
(`inference/agent/prompts.py`) in place of the turn's opener. If the reply is a `python` call
whose code is a note of at least 300 characters, history becomes the last turns, then the
request, the reply and a stand-in tool result ("Note kept in your context (the code was not
run)..."). The opener is then appended as usual. The request is cached up to the note
request, so it costs about the note's output: about 1.8K tokens, $0.03 at gpt-6.1-sol's
prices.

- The threshold is compared with the harness's own estimate of the prompt, which runs above
  the real count: the trimmer's budget, ~130K by that estimate, cut the logged games at
  111-128K real tokens. At 120000 the note comes before the trimmer, at about 112-120K real
  tokens (the replay below).
- Turns are counted by their openers. Resumptions and nudges are not turns.
- If the last 20 turns would keep the prompt above 3/4 of the threshold, fewer are kept,
  at least one. Without this the note came every turn in a smoke run at 20K with 5 kept
  turns. It did not apply in the replay below.
- The note exchange is never pruned. The next compaction drops it, and the prompt asks the
  model to carry the earlier note forward.
- If the request fails or the note is too short, history is left alone. The trimmer still
  cuts at the budget (dfranzen's settings: ~130K by the estimate, draining by 59K).
- 10 kept turns is the default because 20 turns came to 61-87K real tokens in the replay
  below (two 10x-upscaled images, a ~3,000-character opener, the encrypted reasoning and the
  tool results of each turn: about 3.5K tokens a turn). That left room for only 6-18 turns
  before the next note.

## Where it would fire

Request logs of `runs/base-gpt61sol-20games` and `runs/base-gpt61sol-dfranzen`: 12 games
passed 110K. The trigger is the first turn-start request at or above 110K in each stretch
between two logged compactions (`experiments/note-compaction/replay_notes.py triggers`).

| game | logged compactions at step | 110K reached at step |
| --- | --- | --- |
| bp35 | 38 59 71 95 114 131 152 168 | 32 55 68 92 110 127 149 165 |
| dc22 | 45 71 97 121 | 39 66 93 117 |
| g50t | 46 | 42 |
| ka59 | 44 | 39 |
| lf52 | 43 66 94 121 141 163 188 210 231 | 39 64 90 118 138 162 184 204 229 |
| ls20 | 37 | 34 |
| re86 | none (peak 119K) | 34 |
| s5i5 | 35 | 29 |
| sk48 | 36 56 74 95 106 121 138 161 180 192 | 33 55 72 93 105 118 137 156 179 188 |
| su15 | 35 | 31 |
| tu93 | none (peak 116K) | 30 |
| wa30 | 45 72 99 125 148 167 186 | 40 69 95 122 144 162 183 |

- It fires 1-6 turns (median 4) before the logged cut, 52 times against 43. re86 and tu93 never
  reached the trimmer's budget and would get one note each.
- Only the first trigger of a game is the one the setting would see. Later ones follow the
  logged run, which compacted without a note.
- At the triggers, the last 20 turns are about 40-70K tokens (their share of the prompt's
  characters, at the first three triggers of each game): the note compaction keeps about as
  much as the logged cut (59-68K) plus a ~2K note.

## The notes

At the first trigger of each game, and at the four points of the earlier analysis (sk48 steps
55, 72 and 179, bp35 step 92), the logged history was sent again with the note request in
place of the turn's opener, to gpt-6.1-sol with effort `xhigh`, the logged tools and the
encrypted reasoning sent back (`replay_notes.py replay`). 16 points, two versions of the
request:

- **v0**, the bare request: the context is about to be cut to the last 20 turns, write a
  note with everything you need to continue, as comments only, no action.
- **v1**, `NOTE_COMPACTION_PROMPT`: v0 plus where the kept turns start (game step), what
  survives (retained functions, `history`, `transitions`, signatures only), six headings
  (RULES, TRIED, LEVEL, FUNCTIONS, PLAN, LESSONS), "carry forward an earlier note", and what
  to leave out (what each turn shows anyway, details only in the kept turns).

| 16 notes | v0 | v1 |
| --- | --- | --- |
| a `python` call with comments only | 16 | 16 |
| output tokens, median (range) | 1,715 (1,215-2,142) | 1,920 (1,352-3,952) |
| reasoning tokens | 0 in all 16 | 0 in 14; 182 and 1,674 |
| mentions what was tried or failed (tried, probe, no-op, failed) | 14 | 16 |
| restates a system-prompt rule (segmentation first, no full ASCII, `run_complete`, `history[-1]`) | 16 | 11 |
| squashed words (18+ letters, e.g. `allcrateingoals`) | 52 | 0 |
| says a tool timeout wiped its retained functions | 13 | 11 |
| cost | $4.51 | $4.56 |

The rows from "mentions" to "timeout" count notes matching keywords, not a reading.

The notes are in [notes-v0.md](../experiments/note-compaction/notes-v0.md) and
[notes-v1.md](../experiments/note-compaction/notes-v1.md). The cost is with a cold cache
(each request wrote ~105K tokens to the cache); in a run the prefix is cached.

What the notes hold, read one by one:

- **Both versions are usable handovers.** Every note has the progress (levels done), the
  mechanics, the current level's layout in coordinates, the retained functions with what
  they assume, and the plan in progress, often as an exact action list with checkpoints
  (re86 step 34: the remaining 52 actions of a BFS plan and the next shape's 77).
- **The gap the earlier analysis found is covered.** After the logged cut at sk48 step 56 the
  model burned ~120 actions to force a reset, then clicked (27,38) again. After step 74 it
  clicked (27,32) again. Both v0 and v1 notes at steps 55 and 72 list every MOUSE target
  already tried, (27,38) and (27,32) among them, and the vertical and horizontal probes that
  failed. On the reset, the v0 note at step 55 suggests trying it ("Could automatic budget
  death/reset generate easier layout? Untested"). The v1 note at step 55 says it "would be
  expensive and is not yet justified". At step 72, after the reset happened, v1 records that
  it restored the same layout: "Do not repeat the budget-burning loop".
- **v0 spends its length badly.** All 16 restate rules from the system prompt, which is
  kept. Two squash words together to save space (ls20 and wa30, 24 squashed words each:
  "Lastplannedstage toward13,15"), four more once each. Several describe the latest state at length, which the
  kept turns already show.
- **v1 reads as a reference.** Each fact is marked CONFIRMED with its evidence or
  HYPOTHESIS, corrected beliefs are listed ("Rail length can be ZERO! We incorrectly
  assumed minimum length 1"), and FUNCTIONS says which retained functions are level-specific
  and must not be reused. It still restates some harness rules (11 of 16), more briefly.
- **A side finding.** 13 of the 16 v0 notes warn that a 30-second tool timeout wiped all
  retained functions, though the actions in the call ran. The model spends notes, and
  turns, rebuilding them. That cost is separate from compaction.

## Smoke run

Two short ar25 runs with `ARC3_NOTE_COMPACTION_TOKENS=20000` and
`ARC3_NOTE_COMPACTION_KEEP_TURNS=5`, to check the live path (not results; not archived):

- **Without the cap.** The note came at step 7 (20.8K prompt, 782 output tokens, a
  3,377-character note under the six headings). The next request of the turn was accepted
  with the note exchange in its history, and the model's next call ran the note's PLAN
  (select the L, RIGHT×7, DOWN×7). But 5 kept turns were already about 20K, so a note came at
  every turn after (steps 8 and 9). This is why the cap was added.
- **With the 3/4 cap.** The notes came at steps 7 and 11, each keeping 2 turns (prompt
  ~23K → ~15K). The second note request held the first note, and the second note carried
  it forward ("CORRECTED: Yellow is NOT controllable"). Each request after it held only the
  newest note exchange.

## Replay of three games

`runs/note-compaction-3games` (DVC): sk48, lf52 and bp35, the three games with the most cuts
in the logged run, played again from the start with the logged run's settings (dfranzen's
settings, effort `xhigh`, reasoning summaries `auto`, 240 minutes and 500K output tokens per
game) plus `ARC3_NOTE_COMPACTION_TOKENS=120000` and `ARC3_NOTE_COMPACTION_KEEP_TURNS=20`
(the default was 20 when it started). Code `9513227`. It was stopped
after 42 minutes, at turn 74 of lf52 and bp35, once the comparison was clear enough; it cost
$6.47. `experiments/note-compaction/compare_runs.py` compares it with the logged run
(`runs/base-gpt61sol-20games`) turn by turn.

| | sk48 | lf52 | bp35 |
| --- | --- | --- | --- |
| levels completed, replay | 8/8 at turn 31 (won, 285 actions, $0.79) | 7/10 at turn 74 | 7/9 at turn 74 |
| logged run at the same turn | 4 at turn 31 (7/8 after 201 turns, time limit) | 5 | 6 |
| notes (turn) | none: the prompt peaked at 109K | 37, 49, 58, 67 | 36, 54, 66, 72 |
| trimmer cuts | none | none | none |

From the first note (lf52 turn 37, bp35 turn 36) to turn 74:

| | lf52 replay | lf52 logged | bp35 replay | bp35 logged |
| --- | --- | --- | --- | --- |
| levels gained | +4 | +1 | +3 | +2 |
| game actions | 359 | 156 | 133 | 110 |
| output tokens | 53,282 | 21,323 | 49,615 | 42,345 |

Levels completed at equal effort (replay vs logged):

| | lf52 | bp35 |
| --- | --- | --- |
| at equal turns | 3 vs 4 (turn 37), 6 vs 5 (60), 7 vs 5 (74) | 4 vs 4 (36), 6 vs 5 (60), 7 vs 6 (74) |
| at equal game actions | 6 vs 6 (500), 7 vs 6 (521) | 6 vs 5 (200), 7 vs 6 (265) |
| at equal output tokens | 5 vs 6 (50K) | 6 vs 6 (50K) |

- **The notes.** 8 notes, all usable: 1,688-3,288 output tokens, $0.03-0.05 each, $0.31 in
  all. The prompt was 112-120K real tokens at the note request and 61-87K after. Four were
  read against the turns before them and after. Each was correct: bp35 turn 36's tile map
  equals the last tool output, and its six TRIED actions match the actions logged in the
  dropped turns; lf52 turn 49 explains its last no-op click (the camera had moved 20
  pixels) and gives the corrected coordinates, which the next segmentation confirms. In each
  case the next call carried out the note's PLAN (bp35: RIGHT×4 after its search; lf52: the
  two ferry clicks at (19,40) and (31,40), then the ferry search the note described, which
  solved level 4). No turn after a note re-explored or re-probed.
- **One death after a note.** bp35 died on level 7, two turns after the turn-54 note, on one
  unchecked RIGHT into an unseen spike. It was a planning slip, not a fact missing from the
  note. The logged run also died on level 7.
- **Progress.** Per turn, both games are ahead of the logged run from the first note on
  (lf52 +4 levels against +1, bp35 +3 against +2). Per game action they are level or one
  ahead. Per output token they are level (bp35) or one behind (lf52): the replay did more per
  turn (lf52: 359 actions in 38 turns against 156).
- **Caveats.** One run each, so a level either way is within noise. sk48 won before any
  note, while the logged run spent 437 actions stuck on level 5: run-to-run variance this
  large says as much as any difference here. The replay ran 3 games at a time against the
  logged run's 10 on a 4-CPU machine, so its 30-second python calls timed out less often,
  which by itself lets more get done per turn.

## Conclusions

- At 110-120K the note fires a few turns before the trimmer would cut, in 12 of the 25
  games, for ~$0.03-0.05 each.
- The model writes complete notes in one comment-only `python` call. The bare request (v0)
  already keeps the tried-probe list the earlier analysis found missing. The headings (v1,
  the prompt now used) remove squashed text, cut restated rules, and separate confirmed rules
  from hypotheses and what was tried.
- In play, the notes are correct and the model carries on from them without re-exploring.
  lf52 and bp35 gained more levels per turn after the first note than in the logged run, and
  as many or more per game action, not per output token; one run each, with a lighter
  machine load.
- Settings: `ARC3_NOTE_COMPACTION_TOKENS=120000` with dfranzen's settings, and 10 kept turns
  (the new default), which halves the history kept after a note and doubles the room before
  the next one.
