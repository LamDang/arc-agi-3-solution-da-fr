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
   **Decided (v12):** not the seeding: the new level's objects and pixels can differ too much from
   the generator's cut. Instead a nudge in the level-start message naming where the shapes already
   are ("the board's pieces match these engine.py constants: BAR, BIN (turned), ...; combine them
   into the new sprites, and draw what is left from recording[-1].pieces_after.code()").
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
   **Dropped (user's call).**
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
   Still relevant on the v12 forks: the A forks (v11 prompt) called edit_file as a tool 16 and 7
   times, 4 and 2 of them failing this way; the B fork (v12 prompt) never called it as a tool.
   **Decided (v12):** fix the shim (cheap).
   **Done (v12):** `agent.normalise_edits` (in `builtin_call_code`, so a replay runs the same code): a
   JSON or Python-literal string is parsed, one dict is wrapped, oldText/newText (old_text/new_text,
   old/new) without an op is a replace_text; each change is a "[harness] ..." line in the tool output.
   On sp80's 16 failures: 14 lacked the op, 2 were Python literals.
7. **"Unfamiliar elements" in screen coordinates** while the engine may use a rotated grid
   (sp80 level 1: the 4x1 at (2,5) is (10,10) in grid terms); the bins were listed as unfamiliar
   on level 3 because of it. Superseded by 19: the list goes.
8. **Piece name vs description** in object diffs: a disappeared piece named
   `SHAPE_9_3x3_710a` described as "colour 12 (orange), 6x6" (stale identity carried over).
9. **`run_play.py` writes "(v10)"** into config.json for every run.
   **Done (v12):** `engine_re.PLAY_VERSION` and the git short sha: "engine_re.play_agent (v12, git 1a2b3c4)".
10. **Support lines number moves by batch position** ("move 4") while the Sent lines use step
    numbers ("#30"); the well-supported summary line lacks the action name.
11. **notes.md headings duplicate**: the file is seeded with empty headings and the model appends
    its own below them. Seed with no headings, or merge on write.
12. **The 1 px HUD tolerance** hides a constant 1 px offset in sp80's HUD model (steps 13-15,
    32-34); harmless so far, but it can mask a wrong budget guess.

## Prompt

13. **Support per line in the code the model reads, instead of `traced()`.** Neither game ever
    called `traced()` or `support()`. **Decided (v12):** remove both built-ins and their prompt text;
    instead, when read_file or edit_file shows engine.py, every line of step() (and the functions it
    calls) carries a trailing comment with its support: `# support (n): 31, 23, 15, 2 and 27 other`,
    the number of passing steps that ran the line, then the last five of them newest first. The
    support map keeps the first and last 4 steps per line (`STEPS_KEPT`, support.py:62): keep 5.
    The margin counts stay.
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
    **Dropped (user's call).**
18. **The python cell timeout is not in the prompt.** A cell may run 120 s (`kernel.py`, the
    `timeout` of the kernel parent); past that the kernel is restarted and every variable is lost,
    and the model learns this only from the "Timed out after 120s" message after a search is gone
    (sp80 twice, ls20 once). State it in the python tool description: "a cell has 120 s; a longer
    one is killed and the kernel restarts without your variables; bound searches by time
    (`time.time()`) and keep the best result in a variable you print".
    The timeout message itself ("Timed out after 120s. The kernel was restarted and all variables
    were lost.") is explicit but does not say what was lost: the model found out through
    NameErrors on its helpers two and three turns later (sp80 turns 135, 163). The kernel knows
    the names it held (its `names` request): list them in the message ("lost: pour, path_to,
    STATIC, ... 14 names; the harness built-ins and `recording` are back; re-run the cells that
    defined them") and say which cell timed out and roughly where it was (the search's own
    progress prints are lost with it).
19. **Drop the "Unfamiliar elements to test first" list.** It is noise: the segmentation is one
    piece per colour, so ls20's one multi-colour 3x3 patch arrived as five single-colour fragments
    plus "and 4 more", mixed with the new maze floor and two grey bars, in screen coordinates
    (7). The model drew the lot as inert sprites. Remove the list; the level-start message and
    the image are enough.
20. **Rule 6 wording.** "New levels often introduce additional mechanics, sometimes through
    unfamiliar board elements" becomes "... sometimes through an unfamiliar board element or a
    visual change" (ls20 level 2: the legend's colour; sp80 level 0 in v10: the floor blink).
21. **Guaranteed solvable, so being stuck means a missing or wrong rule.** Add to the plan
    rules: every game is solvable. When you are stuck (no plan, or a plan far above the human
    baseline, or the same refusal twice), you are probably missing a rule or one of the written
    rules is wrong; explore more of the game's mechanics (touch what you have not touched, repeat
    a refused move in a changed condition) rather than searching harder on the rules you have.
    ls20 level 2 was refused three times with the dial matched and spent 100 actions on a
    fake-board theory before touching the one object it had never touched.
22. **Adopt the base agent's deployment limits.** The Kaggle submission config (`params.yaml`,
    the write-up's settings table) bounds the base agent by: context window 131,072 tokens with a
    drain to 59,392 once the limit is reached (hysteresis: trim a large block, then let it grow),
    12,288 output tokens per response (a longer thought continues in the next request), 3,072
    tokens of tool output per call, 500K output tokens and 240 minutes per game. The play harness
    has: a 140K compaction threshold with no cap (prompts reached 289K and 297K), 32,768 output
    tokens per response with unbounded thinking (largest turn 18.6K), 8,000 characters of tool
    output (about 2-3K tokens), 300 turns, 1.5M output tokens and $6 per game. Match them: cap the
    prompt at 128K and drain to 58K by dropping (item 2 is the mechanism), cap a response at 12K,
    tool output at 3K tokens, and 500K output tokens per game. Under the 500K cap sp80 would have
    ended around turn 210, at the level-2 solve (3/6, 28.6 instead of 47.6): the level-3 win cost
    the 113K tokens beyond the budget. ls20 (469K) fits.
23. **What must survive a 57K drain.** With the base agent's policy (item 22) about ten turns of
    conversation remain. engine.py, the support map, the test status and the current frame are
    re-shown every turn, so the rules survive. What is only in old turns, and would be lost:
    (a) rejected hypotheses and dead ends (ls20's fake-board theory, "compares shapes only"),
    so the model can walk into them again; (b) the route in flight when it spans several batches
    (ls20's 63-press route went out over 7 batches, the remainder lived in the reasoning of the
    turn that computed it); (c) unexplained observations and open questions (the floor blink,
    the orange legend) that were noticed but not modelled; (d) what was tried on this board:
    positions visited, moves refused, probes already made; (e) the model's own helper functions
    (the kernel keeps them, the PLAN message lists their names but not what they do).
    Make these eviction-safe outside the conversation: a harness-written per-level log in
    notes.md or the PLAN message (each batch's note and outcome, each refused move, each mismatch
    and the rule it changed, each probe), a `plan` kernel variable convention (the PLAN message
    shows the moves left in it), notes.md's "Open questions" refreshed by the harness at each
    level change (item 15), and the kernel's user-defined functions listed with their first
    docstring line. The base agent's analogue is its retained python functions plus an opener
    that restates the state every turn.
    The cheapest version already exists: the commit messages and the batch notes. Measured on
    this run, sp80 wrote 21 commit messages (~3.2K tokens) and 26 batch notes (~2.7K), ls20 19
    (~4.8K) and 36 (~2.7K): 6K to 7.5K tokens for a 300-turn game, about 20 tokens a turn, 10% of
    a 57K window if all of them are kept. They cover most of (a)-(d): each commit message says
    which step changed which rule and which earlier rule it replaced and why ("my halo rule was
    wrong", "killed my 'aligning press is free' rule", the fake-board rule and its correction);
    each batch note says what was tried, what the replica predicted and, with the harness's
    sent/matched counts, what came of it. Not covered: the remaining moves of a route in flight
    (the note names the route, not its tail), observations that never became a fix, and the
    helpers' signatures. So: keep every commit message and every batch note with its outcome in
    the PLAN message (current level in full, earlier levels' commit messages only), plus the
    `plan` variable and the function listing for the rest.
    Budget check against a 57K window, measured on this run: the system prompt is ~8.6K tokens,
    the engine listing with its margin ~5-7K, the PLAN message body ~1.2K, a turn of conversation
    ~1.5K (mean 1.7K / 1.5K on sp80 / ls20 over the last 30 turns, reasoning included). The
    largest per-level log was ls20 level 2: 9 commit messages (~2.3K) and 21 batch notes (~1.7K);
    all commit messages of a game at its end are 3.2K / 4.8K. So "all commit messages + the
    current level's batch notes" is 4-7K at worst, and the window still holds 20-25 turns of
    conversation. The cheaper variant, the current level's batches since the latest commit only,
    is under 1K and loses little: a commit restates what the batches before it established.
    **Decided (v12), revised:** no drain threshold; the context is rebuilt at every turn from the
    transcript as: the system prompt; every commit_engine and commit_moves turn of the game (the
    call and its output, without the reasoning); the current PLAN or FIT message at its place; every
    turn since that message, the last 10 with their full reasoning, the earlier ones without.
    Simulated on the v11 transcripts (chars/token calibrated on the real prompt at turn 30, 1K per
    image): sp80 max 73K at turn 185, median 34K, 30K at turn 300 (v11 real: max 289K, median 157K);
    ls20 max 68K at turn 232, median 33K, 36K at turn 300 (v11: 297K / 163K). At the max turn the
    last 10 turns' reasoning is the bulk (44K / 30K, p90 reasoning 4.5K / 2.8K a turn), the commit
    turns 12K / 16K, the phase message 6K / 3K. All commit turns at the end of the game: sp80 46
    turns, 19K tokens (11K outputs, 8K arguments); ls20 56 turns, 24K (13K / 11K). The gap since the
    last phase message was at most 32 / 41 turns, median 5. Never above 73K, so it fits the 128K cap
    with room, and most turns are under 40K.
    **Form:** one user message holds the rebuilt context (the commit turns, the current PLAN or FIT
    message at its place, the older turns since it) and ends with "The context has been compacted.
    Continue from the context above."; the last 10 turns follow it as real turns with their full
    reasoning.
24. **Add a sprite list to the PLAN message (the image stays).** The PLAN message attaches the current
    frame as a 512x512 image (`_frame_part`, play_agent.py); the FIT message's test report carries
    its own picture. Measured on this run: a turn that carries an image grows the prompt by
    ~1.1-1.4K tokens more than a plain turn (sp80 1,792 vs 404, ls20 1,564 vs 503; the PLAN text is
    part of that), and the model's reading of it was poor: "the bar is at row 3 (lower)... let me look
    at the image once more", "the image shows a red block at the left (cols 6-8, row 2) — actually ..."
    (sp80 turns 17, 64); it read positions reliably only in python (`state_now()` 71 / 34 calls,
    `pieces_after` 13 / 5, `show_frames` never). Both lists the harness could print already exist:
    (a) the replica's own sprites, `state_now().sprites` in the engine's names and coordinates
    (sp80's end: 15 lines, ~200 tokens; ls20's end: 24 lines, ~300 tokens): name, tags, w x h,
    (x, y), layer, rotation / mirror / scale when set, hidden or inert, screen, plus `vars`;
    (b) the segmentation's pieces of the real frame, `str(recording[-1].pieces_after)` (~340 / ~540
    tokens): stable shape names, colour, size, rotation against the shape, enclosures, in the
    segmentation's grid, which is the screen's, not the engine's (sp80's levels 1-3 use a rotated
    View, so its bins sit at y=1 on the screen and y=17 in the engine).
    In the PLAN message the replica is in step, so (a) draws the current frame exactly and is in the
    coordinates the model plans in: print (a) with the image, `vars` on its first line. Out of step
    (the replica plays blind) and at a level start before make_level(n+1) is drawn, print (b)
    instead, since (a) may not match the game. The FIT message keeps its comparison picture and
    gets the reconciliation list (25).
    **Decided (v12):** keep the image (it costs about 1K tokens a PLAN turn and the old ones are
    hidden at the next phase message) and add the list under it. Image geometry, for reference:
    a frame is 64x64 upscaled x8 with nearest neighbour (diff_report.UPSCALE), one panel of
    512x512 under a 30 px title with 12 px padding, so the PLAN image is 536x554 (~6 KB PNG) and
    the FIT comparison, two panels (yours, original) with the differing regions boxed and numbered
    on both, 1060x554 (~9 KB). Sent as a base64 data URL with no detail parameter; the provider
    tokenises it at its own patch size.
25. **FIT message: a sprite-by-sprite reconciliation, not only regions.** The test report describes
    a mismatch by differing pixel regions: for each region, the colours expected and got and the
    replica's sprites drawing there (including the background), plus `state_changes` on the replica's
    side and the segmentation's object diff on the game's side. The model has to put the two sides
    together itself. sp80 step 6: "[2] rows 8-15, cols 20-39: 160 px differ, 12->9 x80, 9->12 x80;
    yours here: #1 background (80 px), #4 player x=5 y=2 (80 px); this step: y 3->2" means "the
    game's bar did not move", which is nowhere said; ls20 step 1 lists three of the replica's
    sprites over one 50 px region with four colour swaps, and the model's reading of such reports
    cost most of the fit turns (sp80 turns 17 and 64 re-read the image for the bar's row). Add, after
    the regions, one line per replica sprite the game disagrees with, and one per piece of the real
    frame no sprite of the replica draws:
      #4 "player" 5x1 blue: yours at (5, 2); the game shows this shape at (5, 3) (it did not move)
      #8 "legend_shape" 6x6: yours colour 12; the game shows it colour 9 at the same place
      #20 "multi" 3x3: yours visible at (30, 46); the game shows nothing there
      the game shows a 3x3 piece (blue, 5 cells) at (35, 16) that none of your sprites draws
    The pieces are the real frame's segmentation (`step_to_fix.pieces_after`), matched to the
    replica's sprites by rendered shape up to translation, rotation and recolour (the matcher
    `pieces_after.code()` uses to recognise engine.py's kinds), in the engine's grid coordinates
    (`cells_text` converts through the View). Sprites that match are not listed; the background,
    border and screen pieces are reported only as "the game shows nothing / something here", never
    as "yours here". With it the region text can shrink to its first line (where, how many px,
    which colours), and the comparison image stays.

## Base agent tooling we do not have (checked against inference/agent and framework/solver)

Equal or better in the play harness: the recording with frames and segmentation (base: `history`,
`.segmentation`, `frame_diff()` with Hungarian matching; ours has stable shape names and transforms),
the animation digest and timeline, numpy and a persistent kernel (base: a fresh 30 s subprocess per
call, no numpy, no classes), the automatic RESET after a game over, the per-move trace of a batch
(base: action echo and per-action trace), tool-call recovery and the degenerate-reply guard (no
malformed call and no length finish in 600 turns of v11). What the base has and we lack:

26. **Helpers that survive a kernel restart.** The base re-executes the source of every retained
    function at the start of every snippet, so a definition is never lost; ours keeps the whole
    namespace but loses it at a 120 s timeout (sp80 three times, ls20 twice, each followed by
    NameErrors two or three turns later). The harness has the cells and a replay (`KernelClient.replay`,
    used on resume, 20 s per cell, 120 s in all): after a timeout restart, replay the earlier cells
    in that mode, skipping the one that timed out, and say which names are back and which cell was
    dropped (18's message).
    **Decided (v12):** do it.
27. **A state line every turn.** The base rebuilds an opener at every request: what the previous
    sequence executed, "Current state: step S, level L", "Valid actions right now", the retained
    functions, the current board image. Ours restates the state only in PLAN and FIT messages, up to
    13 turns apart, and the budget line only in PLAN. With a 57K drain (22) ten turns is the whole
    window: append one harness line to every turn's last tool output ("[harness] step 57, level 2,
    actions 61 of 500 (level 2: 14), plan round turn 4 of 6, output tokens 213K of 500K"), ~40
    tokens, and the budget line in FIT messages too.
    **Decided (v12):** not every turn (too much); the line goes only in the commit_moves output,
    after a batch, and in the FIT message. The PLAN message keeps its budget line.
28. **What the last batch changed, in the PLAN message.** The base attaches a diff image every turn
    (changed cells since the previous turn, the rest navy) and the game-over opener a fatal-step
    diff. Our PLAN message has the batch's per-move lines but not what the board looks like now
    versus before the batch; the object diff (`step_objects`) appears only in FIT messages. Add the
    object changes from the batch's first frame to its last (the segmentation's `changes` summary,
    at most 12 lines) to the PLAN message, under the batch lines. The sprite list (24) gives the
    "now"; this gives the "since".
    **Decided (v12):** only after a batch was played (the commit_moves output and the PLAN or FIT
    message that follows it), not after a commit-only advance.
29. **A cut reply continues.** The base caps a response at 12,288 tokens and a longer thought
    continues in the next request (the reply is appended, the request repeats); a context-length
    error forces a drain and a retry. Ours asks for 32,768 and never hit the cap in v11; under 22
    the agent must treat finish_reason "length" as a continuation (append the partial reply, ask
    again) and a context-length error as a drain, not as an idle turn or a provider error.
    **Decided (v12):** not for OpenRouter runs, where the response is not capped at 12K; a per-response
    cap, if ever adopted on a self-served model, is a request-level setting and the continuation belongs
    in the request wrapper, not the agent loop.
30. **A batch stops at a board no-op.** The base stops a batch of two or more after the first
    executed action that changed nothing inside the board (a 4 px border, the HUD, excluded) and
    says so with the skipped moves. Ours warns ("changes nothing in your replica") and sends the
    whole batch, and the criterion compares whole frames, so a move that only moves the budget bar
    is not a no-op: the warning fired 4 and 2 times in v11 on games whose HUD changes at every
    move. Compare the frame without the screen-layer sprites (or inside `HUD_BORDER`), and cut the
    batch before the first predicted board no-op when the batch has more than one move (a single
    move is a probe and goes), saying which moves were not sent and why.
    **Decided (v12):** do it.
31. **Not needed now, and why:** the stale-state block guards a second `action()` call inside one
    python snippet after a no-op; our python cannot send moves and commit_moves is one call per turn,
    so the case cannot occur. The known-no-op, known-death and repeat-in-state guards and the death
    ledger are off in the Kaggle config, so the base scores were made without them; the replica does
    their job in advance (a predicted no-op is cut by 30, a predicted game over is warned about), and
    repeat-in-state is left to the model. The priority scheduler allocates the 10 vLLM streams among
    games by per-level pace; it is not model-facing, and at deployment it wraps our harness unchanged.
    The per-turn board image costs 402 tokens a turn (4K of a 57K window) and the model read it badly
    (24); ours shows one at PLAN and FIT with the sprite list. The tool-call markup parser repairs
    vLLM's qwen3_coder parser misses; on OpenRouter the server parses the calls and v11 had none
    malformed in 600 turns.

## Planned experiments (v12, awaiting approval)

Not full runs: forks of the v11 run, resumed at a chosen turn with the change applied, 100 turns at
most, compared with the same 100 turns of the v11 run (the reference window). A fork is a copy of
the run directory truncated at turn T (transcript, tests, trace and visible_trace, engine.py at its
version of turn T, the latest commit before T, notes.md rebuilt by replaying the notes edits of the
cells up to T) that `run_play` resumes as it resumes any run (the real game replays the moves, the
kernel replays the cells, the conversation is rebuilt under the policy in force). Base agent
sampling, for reference: temperature 0.7, top-p 0.95, top-k 20 (the Kaggle config).

- **A. context policy (23 revised):** two forks before a late breakthrough, where v11's prompt was
  already 200K+: sp80 at turn 180 (level 2 solved at turn 210 in v11; window 180-280) and ls20 at
  turn 100 (level 1 solved at turn 126; window 100-200). Measure: the same solve inside the window,
  and the turns it takes.
- **B. prompt and messages, with the harness items:** ls20 forked at turn 148, the first PLAN
  message at level 2 (level 1 was solved at turn 126, the level-2 board drawn by turn 148); window
  148-248, in which v11 never solved level 2. Changes: prompt 13 (support comments), 14, 16, 18, 19,
  20, 21; messages 4, 5, 18's restart message, 24, 25, 27 and 28 (after a batch only); harness 26, 30.
  Forking at 127 instead (the FIT message that opens level 2) would also test the drawing, with 1.
- ~~C. temperature~~: dropped (the base agent runs at 0.7 / 0.95 / top-k 20, the same as v11).

Three forks of 100 turns: about a third of a game each, roughly $2 and 80 minutes per fork at v11's
pace. One subagent implements each experiment on its own worktree branch; a fork tool
(`engine_re/tools/fork_run.py`) is shared.

## Observed on the v12 forks (14:06 UTC check-in)

32. **Sprite list without colours (24).** B's PLAN list names `legend_shape` and `room_shape` but not
    their colours, so the orange-against-blue mismatch that blocks ls20 level 2 is invisible in it.
    Add the main colour (index and name) to each line.
33. **Rebuilt-mode token estimate ~25% low.** `estimated_tokens` (chars / 3.5) gave 67.0K against
    82.8K real at sp80 turn 233 and 53.8K against 67.3K at ls20 turn 120. Calibrate on the real
    prompt_tokens as the base agent does (chars per token from the last response), since the 120K
    warning depends on it.
34. **Rebuilt mode hides kernel side effects and old listings.** A-ls20 turn 155 set
    `replica.LEVELS[2]["cost"]` inside a cell; by turn 169 the turn was out of the window, the file
    said 2 and the kernel 1, and four turns went to a "stale replica" hunt. 9 and 7 read_file calls
    in the A windows against 0 in v11's. Candidates: list the kernel's user names with their defining
    turn in the compacted message, and keep the latest engine listing in it when the current phase
    message does not carry one.
    **Decided (v12):** the compacted message carries the current engine.py listing with its support
    comments (as read_file shows it) whenever the current phase message does not already carry it.
35. **A fork's first plan round runs on v11's text.** The fork resumes at v11's last PLAN message, so
    B's first v12 PLAN message came only at turn 165 (fork at 148). Inherent to forking at a PLAN
    message; fork at the FIT message that opens the level, or regenerate the last phase message
    under the new code at resume. Done with 36: the first resume rebuilds the last PLAN message with
    `_enter_plan` (the old message's last-batch sentence, the frame, the sprite list, notes.md) or the
    FIT message with `_enter_fit` for the same step, in place (`PlayAgent._fork_prompts`); on the v11
    ls20 fork at 148 the PLAN message goes from 20,010 to 26,776 chars, the sprite list included.
36. **A fork resumes with the source's system prompt.** `agent.py` writes a new system message only
    for a fresh run; a resumed run reuses the one saved in its transcript. Fork B therefore ran v11's
    system prompt, character for character (34,325 chars): the v12 plan rules (14, 16, 20, 21) were
    never sent, only the messages, tool descriptions and harness changes were. On a fork's first
    resume (the marker present) rebuild the system message from the current prompts and log it;
    together with 35, regenerate the last phase message too. B must be rerun before its plan rules
    can be judged. Done: `PlayAgent._fork_prompts` replaces the system message with the current
    `_system_message()` and logs `{"fork_prompts": {"system": {chars_before, chars_after, changed},
    "phase": {kind, chars_before, chars_after | error}}}`; the replacements are "message" records with
    `"replaces"`, applied in place by later rebuilds. On the ls20 fork at 148: 34,325 -> 35,106 chars,
    "Every game is solvable" now in it; `--dry-resume` prints both heads.
37. **Plan rule numbering.** `_PLAY_PLAN_RULES` (prompts.py) numbers its rules 1-6 and then 5-11 after
    the v12 insertions. Renumber.
38. **Turns left during fit rounds.** The budget line appears in PLAN messages and, since v12, after
    a batch; in a long fit round the model does not see the turns left (the B rerun believed it had 4
    turns left at turn 248 of 248). Add the turns-left count to the FIT message's budget line and to
    the auto-test append when the round runs past 5 turns.
39. **Probe the untouched object at the first refusal.** The B rerun cited the solvable rule at turn
    181 and stated the real hypothesis (the patch recolours the legend), then chose to test "all
    rings" first; it touched the patch at turn 236. The rule's "touch what you have not touched"
    clause should come first in the refusal case: at a refused move, list the board objects never
    touched (from the recording's segmentation: pieces whose cells the player never overlapped) in
    the FIT or PLAN message.
