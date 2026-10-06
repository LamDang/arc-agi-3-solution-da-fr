# v12 forks: the rebuilt context (A) and the v12 messages (B), on 100-turn forks of v11

## Question and method

v11 ([v11-play.md](v11-play.md)) let prompts grow to 289K and 297K tokens and lost about 100 ls20
turns on a win rule without colour. [v11-followups.md](v11-followups.md) proposed changes for both.
Instead of three full games, each change was tested on a **fork** of the v11 run: a copy of the game
directory cut at turn T, resumed for 100 turns with the change in place, and compared with v11's own
turns T+1 to T+100 (the reference window).

- **The fork tool** (`engine_re/tools/fork_run.py`; [engine_re/README.md](../engine_re/README.md),
  play section) keeps what the model had read by turn T + 1, the tests, the trace, engine.py and
  the latest commit; notes.md is rebuilt by replaying the kept cells. `run_play` resumes the fork
  like any run. `result.json` restarts cost and minutes at 0 but keeps the source's tokens.
- **A, context (follow-ups 2, 22, 23).** `--context rebuilt` rebuilds every request from the full
  conversation (`rebuilt_context`): the system prompt, then one user message with every earlier
  commit turn (text, call, output, no reasoning), the current PLAN or FIT message, and the turns
  since it without reasoning, closed by "The context has been compacted. Continue from the context
  above.". The last 10 turns follow as they are. There is no drain threshold. A `rebuilt` record per
  request logs its composition and an estimate of chars / 3.5. Nothing else changed. Two forks were
  started before v11's late breakthroughs: **sp80 at turn 180** (v11 solved level 2 at turn 210;
  window 181-280) and **ls20 at turn 100** (level 1 solved at turn 126; window 101-200).
- **B, prompt and messages (items 4, 5, 13, 14, 16, 18-21, 24-28, 30).** The default compact
  context, with the v12 messages, harness and plan rules of PLAY_DESIGN 3.4, 3.11 and 3.12 (B
  worktree); the texts are listed under B below. Fork: **ls20 at turn 148**, v11's first PLAN
  message of level 2; window 149-248, in which v11 never solved level 2.
- **What each fork inherits.** A's first requests carry v11's last 10 turns in full (sp80: turns
  171-180 and the PLAN message of turn 170; ls20: turns 91-100 and the FIT message of turn 98). B's
  first plan round (149-151) read v11's last PLAN message; its first v12 FIT message came at turn
  151, its first v12 PLAN message at 165. B's **system prompt is v11's**: a resume keeps the system
  message logged in the transcript (`agent.py` says the system prompt only when not resumed), and
  B's is v11's character for character (34,325 characters, no "Every game is solvable", no colour
  rule). The v12 plan rules **did not reach the model in B**; the tool descriptions, built at each
  request, did (the python tool states the 120 s cell limit).

## The runs

| fork | run dir (`runs/engine-play/`, DVC pointer beside it) | code | forked at | `run_play` flags |
| --- | --- | --- | --- | --- |
| A sp80 | `v12a-sp80-t180` | `2518e9e`~ (branch `worktree-agent-a0da47d78c8eef05d`) | turn 180, step 38, plan round | `--context rebuilt --max-turns 280` |
| A ls20 | `v12a-ls20-t100` | `2518e9e`~ (same) | turn 100, step 54, fit round | `--context rebuilt --max-turns 200` |
| B ls20 | `v12b-ls20-t148` | `ea685ec`~ (branch `worktree-agent-a45f46eb4bc4f3e5e`) | turn 148, step 78, plan round | `--max-turns 248` |

Commits inferred from the start times (13:27 and 13:48 UTC, 2026-10-06), on unmerged branches. v11's
model, sampling (T 0.7, top_p 0.95), limits and provider throughout.

```bash
uv run --no-sync python -m engine_re.tools.fork_run --src runs/engine-play/qwen38flash-v11-a/sp80 --turn 180 \
  --out runs/engine-play/v12a-sp80-t180/sp80
uv run --no-sync python -m engine_re.run_play --games sp80 --out runs/engine-play/v12a-sp80-t180 --model "$MODEL" \
  --context rebuilt --max-turns 280 --max-output-tokens 1500000 --max-cost 6 --max-minutes 240 --max-actions 500 --batch-size 10
```

`$MODEL` is `config.json`'s `model`; the ls20 forks change the paths and turns, and B drops `--context`.

## Results

From the transcripts: the records after the `fork` record against v11's for the same turns. Each
fork's sums equal `result.json`'s cost, and its tokens minus the source's first T turns.

| window | levels (start -> end) | solved in the window: turn, step, actions in the level | actions | at the end | turns plan / fit | output tokens | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A sp80** t181-280 | 2 -> 3 | level 2: t185, step 48, **18** | 20 | step 58, level 3: 10 actions | 54 / 46 | 222,268 | $0.675 | 68.6 |
| v11 sp80 t181-280 | 2 -> 3 | level 2: t210, step 51, 21 | 53 | step 91, level 3: 40 actions | 70 / 30 | 194,808 | $0.833 | 55.9 |
| **A ls20** t101-200 | 1 -> 2 | level 1: t126, step 78, 65 | 58 | step 112, level 2: 34 actions | 31 / 69 | 152,444 | $0.556 | 38.1 |
| v11 ls20 t101-200 | 1 -> 2 | level 1: t126, step 78, 65 | 60 | step 114, level 2: 36 actions | 16 / 84 | 186,876 | $0.785 | 48.6 |
| **B ls20** t149-248 | 2 -> 2 | - | 83 | step 161, level 2: 83 actions | 27 / 73 | 133,456 | $0.816 | 44.7 |
| v11 ls20 t149-248 | 2 -> 2 | - | 37 | step 115, level 2: 37 actions | 15 / 85 | 185,396 | $0.870 | 53.4 |

At the end of each window every fork has the score v11 had at the same turn: sp80 3/6 (28.6), ls20
2/7 (10.7). The turn limit ended all three forks.

| after the fork, against v11's same turns | A sp80 | v11 | A ls20 | v11 | B ls20 | v11 |
| --- | --- | --- | --- | --- | --- | --- |
| batches sent (refused) | 4 (0) | 7 (0) | 7 (0) | 10 (2) | 12 (0) | 8 (2) |
| mismatches | 4 | 6 | 4 | 6 | 6 | 5 |
| `commit_engine` calls | 3 | 5 | 4 | 6 | 5 | 5 |
| plan nudges | 6 | 11 | 2 | 0 | 1 | 0 |
| kernel timeouts (120 s) | 4 | 0 | 0 | 1 | 0 | 1 |
| turns with a NameError | 0 | 0 | 3 | 2 | 0 | 1 |
| `read_file` calls (of them as a tool call) | 15 (15) | 0 | 17 (7) | 11 (0) | 5 (0) | 10 (0) |
| RESETs | 0 | 0 | 0 | 0 | 2 | 0 |

## A: the rebuilt context

### sp80, turns 181-280

- **Level 2 in 18 actions at turn 185** (v11: 21 at turn 210). The fork resumed inside v11's
  layout search. At turn 182 the search returned a layout 6 moves away, and turn 184 checked the
  10-action finish on the replica. The batch at turn 185 solved the level as predicted (55 wet
  cells, nothing lost); step 48's mismatch was level 3's first frame. In the same turns v11 found
  another route, sent at turn 195. It stopped at step 42 (the bar cannot come within one row of the
  bins), and v11's second route solved the level at turn 210.
- **Level 3.** The fork drew the board by turn 192 (fit s48, 7 turns, as v11's s51). Its first
  level-3 batch came at turn 204, after 12 plan turns, one nudge and one timeout (v11: turn 248,
  after 30 plan turns and 5 nudges). That batch stopped at step 54, where one click turned both 3x1
  blocks blue. This is the group rule ("a click hands over every same-width block in the row, and
  all of them move in lockstep"). The fork met it at turn 204 and committed it at turn 226; v11 met
  it at turn 258 (step 64) and committed it at turn 265.
- **Then it slowed.** The s54 fit round took 22 turns (204-226). It also fitted the level-3 HUD
  budget, and two `edit_file` calls lacked `op`. A one-move probe at turn 240 found the reverse
  handover (step 55, fit 240-251). The plan round from 251 to 274 had 3 nudges and 3 kernel
  timeouts (257, 269, 270) in end-state enumerations and a layered route planner. After the timeout
  at 269 the model wrote "the kernel restarted, so I must redefine everything", and that rewrite
  timed out at 270. The batch at turn 274 stopped at step 58 after 2 moves, and the window ended
  in that fit round, with 10 level-3 actions (v11: 40; it solved the level at turn 299 with 71).

### ls20, turns 101-200

- **Level 1, identical outcome.** Fit rounds ended at turns 111 and 120 (v11: 109 and 123), the
  batches at 124 and 126 had v11's outcome, and the level fell at turn 126, step 78, in 65 actions.
- **Level 2.** Drawing the board took 34 turns (126-160; v11 22), with 8 `read_file` calls (7 of
  engine.py; v11: 3). The first level-2 batch came at turn 173 (v11: 149). Step 86 was the piston
  strip, v11's conveyor, read from the same 17-frame animation (fit 173-185; v11 149-162). A
  28-press route in three batches (turns 190-194) ended at step 112: the room refused the block
  with the legend matched in shape (v11's first refusal: step 111, turn 168).
- **The colour.** At turn 197 the model wrote that the legend matches the room in shape but is
  orange against the room's blue. At turn 200 one of its three hypotheses was "the room's open
  condition is shape+colour of the legend, and the rainbow changes the legend colour", which is the
  real rule. v11 noticed the colour at turn 185 and dismissed it at turn 188 ("compares the pattern
  shapes only"). It turned to the patch at turns 263-271 and put the colour into the rule only at
  turns 298-299. The fork ended at turn 200 at 2/7, with 112 actions.

### Context sizes

| | A sp80 | v11 sp80 | A ls20 | v11 ls20 | B ls20 | v11 ls20 |
| --- | --- | --- | --- | --- | --- | --- |
| prompt tokens, 100 turns | 6.41M | 23.36M | 5.31M | 17.10M | 21.89M | 21.66M |
| largest prompt (turn) | **83,139** (237) | 273,236 (280) | **68,029** (171) | 228,434 (198) | 266,209 (248) | 256,062 (232) |
| median prompt | 65,245 | 226,508 | 50,730 | 162,579 | 220,954 | 225,767 |
| cached | 45.5% | 88.3% | 43.8% | 81.5% | 86.3% | 85.0% |
| cost: prompt / output | $0.570 / $0.104 | $0.742 / $0.092 | $0.484 / $0.072 | $0.697 / $0.088 | $0.753 / $0.063 | $0.783 / $0.087 |

- The rebuilt context cut the prompt tokens by 73% (sp80) and 69% (ls20), but the cost by only 19%
  and 29%. The cached share halved. The compacted message changes whenever a turn leaves the
  10-turn window, so only the shared prefix is cached: about 29K tokens a request on sp80 and 23K
  on ls20, out of medians of 65K and 51K.
- Follow-up 23's simulation predicted maxima of 73K (sp80) and 68K (ls20); the real ones here were
  83K and 68K, far under the 120K warning.

The largest requests, from their `rebuilt` records (the record logged at turn t describes request
t + 1):

| part of the request | sp80, request 237 | ls20, request 171 |
| --- | --- | --- |
| system prompt | 34,325 chars (15%) | 34,325 (19%) |
| commit turns | 33 turns, 52,106 (22%) | 26 turns, 37,578 (21%) |
| phase message | PLAN of turn 226, 19,119 (8%) | PLAN of turn 160, 20,657 (11%) |
| turns since it, older than the window | 0 | 0 |
| last 10 turns, with reasoning | 126,830 (55%) | 89,029 (49%) |
| total; images | 232,380; 1 | 181,589; 1 |
| estimated / real tokens | 66,394 / 83,139 | 51,882 / 68,029 |

**The estimate is low.** Real prompt tokens were 1.19 times the estimate on sp80 (median; 1.13 to
1.27) and 1.26 times on ls20 (1.18 to 1.35): 2.94 and 2.78 characters per real token, against the
3.5 assumed. Follow-up 33's figures (66,982 against 82,805 and 53,784 against 67,299) are requests
234 and 121, the records of turns 233 and 120. The 120K warning would fire only at about 140K-160K
real tokens.

### What the compaction cost

- **A kernel side effect left the window.** At ls20 turn 155 a search cell set
  `replica.LEVELS[2]["cost"] = cost`. Not a commit turn and before the PLAN message of turn 160,
  it was not sent at all from request 166 on. From turn 166 the kernel's replica said cost 1 and the
  file 2; turns 169-172 went to a "stale replica" hunt, ended by loading engine.py with `importlib`.
- **Listings out of reach.** The phase message's engine.py listing stays, but listings from
  `read_file` and edits more than 10 turns old do not. sp80 called `read_file` 15 times as a tool
  (v11's window: 0). ls20 called it 17 times, 7 as a tool call (v11: 11, all in python), 8 of them
  while drawing level 2. Each tool call costs a turn.
- **Engine names without `replica.`** ls20 had two NameErrors on engine functions (`open_cells` at
  turn 106, `covers` at 139; the model noted "`covers` is in replica" at 140) and one on `run_tests`
  called in python (102). v11's window had none of these; its two were on its own `plan3`.
- **Longer fit rounds where old work was needed**: ls20's level-2 drawing (34 turns against 22).
- **Not lost**: the rules (engine.py), the commit history, the current phase message and the route
  in flight. No batch was sent on a forgotten rule, and no turn was spent on a theory v11 had
  already rejected.

**Verdict: no harm on the breakthrough windows.** sp80 solved level 2 25 turns earlier in 3 fewer
actions, ls20 solved level 1 exactly as v11, on a quarter to a third of the prompt tokens (83K at
most). The later turns, where the model needed what it did more than 10 turns before, are the
stress case and show these costs: 4 turns for the hidden kernel change, a turn per `read_file`,
longer fit rounds. The cache share fell from 82-88% to 44-46%, so cost fell less than tokens.

## B: the v12 messages and harness (ls20, turns 149-248)

### The story

- **149-165, as v11.** The first plan round (149-151) on v11's PLAN message sent v11's batch,
  which stopped at step 86 (the strip). The first v12 FIT message reconciled it: `#20 "block" 5x5:
  yours at (9, 5); the game shows this shape at (34, 5)`. The fit round ran 151-165 (v11: 149-162).
- **Two wrong solve predictions.** After the second plus landing (batch 169) the replica predicted
  that the strip push at step 104 (turn 171) carries the block into the room. The game stopped it
  one cell above. The DOWN at step 105 (turn 175) was refused with the legend matched in shape.
  Both verdicts read "your replica predicts level 2 solved; the game did not". v11's first refusal
  was at step 111, turn 168.
- **First theory: every ring.** At turn 178 the model wrote: "maybe the game compares colours
  too?! ... That can't be (the game is solvable). Hmm, maybe the win in L2 is not the room at all!
  Maybe ... the block must cover the multi-colour object". It settled instead on "all rings
  collected" (turn 184). It RESET at turn 201 and sent a 34-press route through both rings
  (201-205), refused again at steps 140 and 141 (turns 205 and 209).
- **The patch.** In the 31-turn fit round on step 141 (209-240) the model asked "What's left in
  L2 that I haven't touched? The multicolour patch" (turn 212). It then reasoned that the orange
  legend can never match the blue room, so "the patch (which contains orange AND blue) might recolour
  the legend to blue" (turns 214, 218), and "So the patch is needed" (217). It RESET at turn 242 and
  sent a 41-press route through the patch. Press 19 (step 161, turn 245) landed on the patch: the
  legend turned from orange to blue, and the bar was charged where the replica had the press free.
- **End.** At turn 248, in that fit round, the model wrote the rule: "the room needs the legend's
  pattern == the room's pattern AND the legend's colour == the room's colour". The window ended
  there, at 2/7 with 161 actions (83 on level 2, above its human baseline of 73). The last turn's
  unfinished edit left engine.py failing at step 1 (`final_test.txt`: 6/162). The committed engine
  passes steps 0-160.

### The path to the patch, against v11

| | v11 | B |
| --- | --- | --- |
| first refused entry, legend matched in shape | turn 168, step 111 | turn 175, step 105 |
| what followed | boards 3 and 4 invented (168-195), a lost life (step 115), "2 columns left" (205-246) | "every ring" (178-208), a RESET and a 34-press route |
| second refusal | turn 260, step 172 | turn 209, step 141 |
| the patch named as the key | turns 263-271 | turns 212-218, with the colour as the reason |
| the patch covered, legend recoloured | turn 287, step 210 | **turn 245, step 161** |
| colour in the win rule | turns 298-299 | turn 248 |
| level-2 actions by the patch | 132 | 83 |

B reached the patch 42 turns and 49 actions earlier than v11. It did so without the v12 colour and
"solvable" rules. Its "the game is solvable" is v11's plan rule 5, which v11 also cited (turn 269).
The [B rerun](#b-rerun-the-v12-system-prompt-actually-sent) sends them.

### The new texts

| text (follow-up item) | triggered in the window | used by the model |
| --- | --- | --- |
| support comments on `step()` lines (13) | 7 phase-message listings, 15 `read_file`/edit outputs | never cited; `traced()` and `support()` are gone but are still in v11's system prompt, and never called |
| sprite list under the PLAN image (24) | all 11 PLAN messages | cited once (turn 238, a sprite's name); it gives no colours, so the orange legend is not in it (32) |
| sprite-by-sprite reconciliation (25) | 2 of 6 FIT messages (151, 245); the other 4 were predicted solves with no frame to reconcile | restated both (154: the block's jump; 246: "legend shape changed from orange to blue") |
| predicted-solve verdict (4) | 4 fit rounds (steps 104, 105, 140, 141) | never quoted. One turn (177) read "raised an error starting level 3" as a board to draw, corrected at 178. No invented boards (v11: turns 168-195) |
| raise warning before sending (5) | 4 batches (171, 175, 205, 209), each at a predicted solve | never cited |
| budget line after a batch and in FIT messages (27) | 12 batch outputs, 6 FIT messages | the model's level-2 arithmetic used the baseline (also in v11's PLAN message) |
| what the batch changed on the board (28) | 10 of 12 batch outputs (not the two refused single presses) | never cited; turn 245's output already listed the legend's recolour, 12->9 |
| no-op cut (30) | never | - |
| kernel restart replay and note (18, 26) | no timeout in the window (v11: one, turn 248); the 120 s limit is in the python tool's description | - |
| level-start paragraph without the unfamiliar list (19) | no new level in the window | - |
| v12 plan rules 6-9 (14, 16, 20, 21) | **not sent** (v11's system prompt) | - |

## What to change next

- **Forks must carry the harness under test.** Write the current system prompt into the fork at
  resume, and fork at the FIT message that opens a level (or regenerate the last phase message), so
  the first round runs on the new text (35). Done (36); B was rerun with its plan rules: see
  [B rerun](#b-rerun-the-v12-system-prompt-actually-sent).
- **Count the rebuilt request with the model's tokenizer** instead of chars / 3.5 (33): the
  estimate ran 16-26% low. The A branch now does this (`0971b7e`).
- **Keep what the window drops** (34): the kernel's user names with the turn that defined each,
  and the latest engine.py listing when the phase message has none, in the compacted message; and
  a note when a cell assigns into `replica`, whose module then differs from engine.py until the
  next reload. These cost sp80 its 15 `read_file` turns and ls20 its "stale replica" hunt.
- **Keep the cache.** Each turn that leaves the window changes the compacted message, so the
  cached share fell to 44-46%. Moving turns out in blocks (a window of 10 to 20 turns) would keep
  the prefix stable between moves.
- **Colours in the sprite list** (32; `692d361` on the B branch adds each sprite's main colour):
  the one fact ls20 needed, an orange legend against a blue room, was in none of the 11 lists.
- **The predicted-solve verdict should not mention level n + 1** when the game did not solve it
  ("raised an error starting level 3" cost one turn).
- **Long searches remain**: sp80 had 4 timeouts in 100 turns, and A's harness did not yet state the
  120 s limit (18); nudge escalation (3) and a built-in replica search (17) are still open.

## Caveats

- **n = 1 per window, at temperature 0.7, from a provider whose answers are not deterministic.**
  sp80's earlier level-2 solve came from the search the fork continued over turns 181-184, which
  could have gone either way. B's faster path to the patch credits the messages or the sample, not
  the plan rules, which it never saw.
- **Inherited turns.** A's first 10 requests carry v11's reasoning; B's conversation before turn
  149 and its system prompt throughout are v11's, and its first 3 turns ran on v11's PLAN message.
  A's ls20 window began inside a v11 fit round.
- **Windows end mid-level**, so none gives a final score. B's 83 level-2 actions already exceed the
  human baseline (73); v11 used 153 on that level.
- **Run noise.** HTTP 429 retries from the analyzer endpoint: 2 (A sp80), 5 (A ls20), 1 (B), all
  retried to success; one answer failed at the provider and was asked again (A sp80). Every answer
  ended with a tool call. The forks did not resume mid-window; v11's ls20 windows include its
  resumes at turns 159 and 172 (minutes count only the time the agent ran).
- **Code**: `2518e9e` (A) and `ea685ec` (B), inferred from the start times; both unmerged, and the
  A branch has changed since.

## B rerun: the v12 system prompt actually sent

B was run again from the same cut (ls20, turn 148, step 78, `--max-turns 248`, the B commands with
`v12b2-ls20-t148` as the directory) on `aa11286`~ (this branch; inferred from the start time, 15:16
UTC), which adds the fork resume of follow-ups 35 and 36. The `fork_prompts` record shows the
system prompt rebuilt (34,325 -> 35,106 characters, now with plan rules 6 "Every game is solvable", 7
"enumerate the winning end states" and 9's "may involve colour as well as shape and rotation") and
the last PLAN message regenerated (20,010 -> 26,776 characters). Its sprite list now gives colours
(`692d361`): `"room_shape" ... colour 9 (blue)`, `"legend_shape" ... colour 12 (orange)`, `"multi" ...
multi (8 red, 9 blue)`. The first plan round (149-150) therefore ran on v12 text throughout. Same
model, sampling and limits; one analyzer retry, no harness traceback.

| window | levels (start -> end) | actions | at the end | turns plan / fit | output tokens | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **B rerun** t149-248 | 2 -> 2 | 159 | step 237, level 2: 159 actions | 32 / 68 | 100,178 | $0.849 | 36.7 |
| B t149-248 | 2 -> 2 | 83 | step 161, level 2: 83 actions | 27 / 73 | 133,456 | $0.816 | 44.7 |
| v11 t149-248 | 2 -> 2 | 37 | step 115, level 2: 37 actions | 15 / 85 | 185,396 | $0.870 | 53.4 |
| v11 t249-300 | 2 -> 2 | 116 | step 231, level 2: 153 actions | 33 / 19 | 48,479 | $0.393 | 21.8 |

The turn limit ended it at 2/7 (10.7), 237 actions, 159 of them on level 2 (`summary.md`); the
committed engine replays all 238 steps.

| after the fork | B rerun | B | v11 t149-248 | v11 t249-300 |
| --- | --- | --- | --- | --- |
| batches sent (refused) | 22 (1) | 12 (0) | 8 (2) | 13 (0) |
| mismatches (of them predicted solves) | 9 (3) | 6 (4) | 5 | 3 |
| `commit_engine` calls | 8 | 5 | 5 | 2 |
| plan nudges / kernel timeouts / NameError turns | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 1 / 1 | 2 / 0 / 0 |
| `read_file` calls (of them as a tool call) | 7 (0) | 5 (0) | 10 (0) | 3 (0) |
| RESETs | 2 | 2 | 0 | 2 |
| prompt tokens; cached | 23.22M; 86.2% | 21.89M; 86.3% | 21.66M; 85.0% | 14.37M; 92.7% |
| largest prompt (turn); median | 290,228 (248); 228,104 | 266,209 (248); 220,954 | 256,062 (232); 225,767 | 296,762 (300); 280,773 |

The refused batch (turn 227) was sent in a fit round, before step 208 was fixed.

### Where the 159 level-2 actions went

| steps | presses | what |
| --- | --- | --- |
| 79-100 | 22 | v11's route; the conveyor (86-87); press 100 found the bar empty and was **lost**: a pip, block back at the start (the note assumed an empty bar was allowed, as in level 1) |
| 101-133 | 33 | a 33-press route, refused at 133: legend matched in shape, one ring still on the board |
| 134-135 | 2 | a deliberate lost press instead of a RESET; it **rolled the whole level back** (rings, legend), which the replica missed |
| 136-164 | 29 | the legend condition dropped by Occam (200-202); a rings-only route, the single press at 164 refused (both rings taken, legend not matched) |
| 165-208 | 44 | RESET, a 43-press route meeting both conditions, refused at 208 |
| 209-227 | 19 | RESET, a route through the patch: covered at step 226 (turn 236), legend 12 -> 9; step 227 showed the patch stays |
| 228-237 | 10 | the first 10 presses of a 31-press finish |

### The path to the patch, against B and v11

| | v11 | B | B rerun |
| --- | --- | --- | --- |
| first refused entry, legend matched in shape | turn 168, step 111 | turn 175, step 105 | turn 173, step 133 |
| colour raised as part of the match | 185, dismissed 188 | 178, "That can't be (the game is solvable)" | 181: the same words, then "the multi object cycles the legend's colour" (H2) |
| what was tested instead | invented boards, "2 columns left" | "every ring" (178-208) | "all rings" (H1, "more parsimonious", 187), refused at steps 164 and 208 |
| the patch as the key | 263-271 | 212-218 | 181-187 (as the "cheapest single probe"), then 222-232 ("the one object no step has ever touched") |
| patch covered, legend recoloured | turn 287, step 210 | turn 245, step 161 | **turn 236, step 226** |
| colour in the win rule | 298-299 | 248 (reasoning; the edit unfinished) | 240 (one of two readings), engine edit 246, committed 248 |
| level-2 actions at the patch | 132 | 83 | 148 |

The FIT message at step 226 reconciled the colour (`#8 "legend_shape" 6x6 colour 12 (orange): yours
colour 12; the game shows it colour 9 (blue) at the same place`), and the model read it as a recolour
at once (237), where v11 read 12->9 as a rotation. Step 227 showed the patch is not consumed. At
244-246 it rewrote the lock as `v["dial"] != d["room_rot"] or v["leg_col"] != 9` (refused unless the
legend shows the room's pattern in blue) and the rings as refills only; a search at 247 found a
31-press finish (level 2 at 180 actions). Turn 248 committed that engine with its first batch, 10
presses matched (steps 228-237), and the window ended 21 presses short. The model counted "Turns:
244 now -> 4 more turns" at turn 248: fit-round outputs do not repeat the turn count, and the last
one it had read was 243.

### The new texts

| text | used by the model |
| --- | --- |
| rule 6, every game is solvable | turn 181, as intended: a colour lock that can never match "can't be", so something must recolour the legend. Also twice of a dead-end state (154, 210) |
| rule 6, touch what you have not touched | paraphrased at 222-232, after the third refusal; B reached the same words at 212 without it |
| rule 7, end states first | never; the end cell is known on ls20, and BFS over moves sufficed |
| rule 9, colour in a comparison | the H2 hypothesis at 181 and the rule at 240-246; the recolour was never taken for a rotation |
| sprite list with colours (all 22 PLAN messages) | never cited by name; the replica already drew the legend orange (`legend_col=12`) |
| sprite reconciliation (6 of 9 FIT messages; the 3 refused room entries had none) | quoted at 237 ("the legend color shifts from orange to blue") |

**Verdict.** The plan rules changed the reasoning, not yet the play. At the first refusal the
rerun stated the real rule (pattern and colour, the patch recolours the legend) at turn 181, 33
turns before B and over 100 before v11. It then tested the rings first, citing parsimony and a
budget that made the patch-first route need a lost press, and spent 3 refusals, 2 RESETs and 2 lost
presses (148 level-2 actions against B's 83) before covering the patch at turn 236, 9 turns earlier
than B. It ended with the right lock committed and 21 presses to go, already above the human
baseline (159 against 73). What is still missing:

- **Test the untouched object first.** Rule 6 fires only "when stuck"; after one refusal the model
  preferred adding a condition on known objects (Occam, plan rule 5 and fit rule 2) to a probe of the
  one object never touched. Say that a refused goal is the moment to touch the untouched.
- **Turns left in fit rounds**: repeat the turn count in tool outputs, or warn when fewer turns
  remain than the planned batches.
- **Prompt numbering**: `_PLAY_PLAN_RULES` (prompts.py) continues at "5." after rule 6, so the plan
  rules read 1-6, 5-11.
- n = 1 at temperature 0.7: the lost press at step 100 and the order of the tests are samples.
