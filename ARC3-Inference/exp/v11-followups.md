# v11 play run: follow-ups

Noted while monitoring `runs/engine-play/qwen38flash-v11-a` (sp80 and ls20). None of these were
changed during the run. Ordered by the turns they cost.

## Harness

1. **Seed `make_level(n + 1)` at every level change.** The harness runs the segmentation
   generator (`recording[e].pieces_after.code()`) only before turn 1, for level 0. At later level
   changes the prompt tells the model to call it; the model called `.code()` once in the whole run
   (ls20 turn 10, level 0) and redrew every later level by hand: 11 to 22 turns and 4K to 38K
   output tokens per level (ls20 transcribes the maze as a 64-wide ASCII literal, 230 lines for
   level 1). At a level change the harness should insert a `make_level(n + 1)` branch from the
   generated code, check it passes the level's first-frame test, and say so in the PLAN message
   ("level n+1's first frame is drawn from these kinds; rename and restructure as you like").
2. **Compaction has a floor that rises with the turns.** `_compact` trims in place and never
   drops anything: a 1,200-char tail of every old reasoning block (172 turns = ~42K tokens on
   sp80), a 200-char stub of every old tool output (~10K), the shortened arguments of every past
   call (~28K), every PLAN/FIT message with only the engine listing elided (~41K). The prompt sat
   at 160K-290K after the 140K threshold, with the cache covering 130K-150K of it. Drop instead of
   stub: reasoning tails only for the last 30 turns, old arguments to one line, old harness
   messages to their first line (latest PLAN and FIT kept), images older than the last two hidden.
3. **Plan nudges never escalate.** sp80 spent 24 plan turns (4 nudges) before its first level-2
   batch and 26+ before its first level-3 batch, rewriting searches that crashed on its own code.
   After the second nudge, require a batch of 1-3 moves before any further search.
4. **Verdict when the replica predicts a level solve and the game does not.** `fresh()` calls
   `make_level(n + 1)`, which raises `IndexError` when the level is not drawn, and the mismatch
   message says "your replica raised an error". It cost ls20 ~9 turns looking for a "level 2
   board" in an unchanged frame. Say "your replica predicts level n solved; the game did not".
5. **No warning before sending when the replica raises mid-batch.** `_prediction_warnings`
   stops quietly at the first move the replica raises on; add "[harness] Warning: your replica
   raises at move k (...)" when `prediction["error"]` is set.
6. **`edit_file` called as a tool.** With `edits` as a JSON string the shim passes it unparsed
   (`[E_BAD_OP] edits must be a non-empty list`, 16 times on sp80); with `oldText`/`newText` but
   no `op` it fails ("Edit 0 has op None", 9 turns lost on sp80). Parse the string and infer
   `replace_text` in `_dispatch` / `builtin_call_code` (engine_re/agent.py).
7. **"Unfamiliar elements" in screen coordinates** while the engine may use a rotated grid
   (sp80 level 1: the 4x1 at (2,5) is (10,10) in grid terms); the bins were listed as unfamiliar
   on level 3 because of it. Give grid coordinates in the engine's view when there is one.
8. **Piece name vs description** in object diffs: a disappeared piece named
   `SHAPE_9_3x3_710a` described as "colour 12 (orange), 6x6" (stale identity carried over).
9. **`run_play.py` writes "(v10)"** into config.json for every run.
10. **Support lines number moves by batch position** ("move 4") while the Sent lines use step
    numbers ("#30"); the well-supported summary line lacks the action name.
11. **notes.md headings duplicate**: the file is seeded with empty headings and the model appends
    its own below them. Seed with no headings, or merge on write.
12. **The 1 px HUD tolerance** hides a constant 1 px offset in sp80's HUD model (steps 13-15,
    32-34); harmless so far, but it can mask a wrong budget guess.

## Prompt

13. **Nudge `traced()` in plan rule 2.** Neither game ever called `traced()` or `support()`: the
    support text arrives for free only after a batch is sent, so the model sees a plan's weakest
    link only after committing to it. "Rank the candidates your search finds with traced()".
14. **Colour is part of a match.** ls20 noticed the orange legend against the blue room pattern
    (turns 185, 188) and dismissed it ("compares shapes only"); the object diff said "recoloured
    12->9" at step 210 and the model read it as a rotation. The base prompt's colour legend was
    ported; a line that a goal comparison may involve colour as well as shape and rotation was not.
15. **notes.md goes stale**: sp80 never updated it after 06:37 (level 0) although it said it would.
    Refresh it in the FIT message that opens a new level, or prompt an update there.

## Measured on this run

- Turns whose call errored or whose test or batch failed: sp80 61 of 262 (25% of output tokens),
  ls20 48 of 300 (15%). Most are the model's own code errors (TypeError, NameError, IndexError)
  and the `edit_file` shim cases above.
- Of the context kept in full after compaction (last 8 outputs, last 10 reasonings and calls),
  errored material is small: sp80 ~1.4K of ~14.6K tokens, ls20 ~0.3K of ~5K. The cost of errors is
  the turns, not the context.

## Added after the run

16. **Plan from the end state.** On sp80 the win is local to columns (each cavity needs a piece
    end one column inside it with a clear drop; every other drop must land on a lower piece, never
    on the drain; each source's drop must land on the chain), so the winning layouts can be
    enumerated from the rules in milliseconds. The model got there by random placement sampling
    through the pour simulator (60,000 trials, 69 layouts) after four nudges and a dozen crashed
    move searches. A plan rule: when the goal is a configuration, enumerate the winning end states
    from the rules first, then plan the route to the nearest one.
17. **The replica is too slow to search directly.** A BFS over `replica.step` on deep copies of
    the State hit the 120 s kernel timeout twice, so the model rewrote the level's rules as a
    set-based fast model that diverged from the engine twice. A cheaper State copy, or a built-in
    move search over `replica.step`, would remove the rewrite.
18. **The python cell timeout is not in the prompt.** A cell may run 120 s (`kernel.py`, the
    `timeout` of the kernel parent); past that the kernel is restarted and every variable is lost,
    and the model learns this only from the "Timed out after 120s" message after a search is gone
    (sp80 twice, ls20 once). State it in the python tool description: "a cell has 120 s; a longer
    one is killed and the kernel restarts without your variables; bound searches by time
    (`time.time()`) and keep the best result in a variable you print".
