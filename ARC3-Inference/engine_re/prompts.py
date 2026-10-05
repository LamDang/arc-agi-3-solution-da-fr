"""System prompt, first user message and tool schemas for the reverse-engineering agent.

The agent has three tools: python (a kernel with the recording, read/edit/undo for engine.py and
helpers to run and look at it), run_tests and finish. ``images`` says whether test reports and
show() come with pictures; the texts follow it.

Two modes. "single" (v5): one session over the whole recording. "step" (v6, engine_re.stepwise):
the harness replays the recording and asks to fix the first step that breaks; the agent sees the
recording up to it (S and `step`; with history=False only `step`) and the tests replay steps 0 to
it. When they pass, the harness replays on and, in the same conversation, names the next step that
breaks (episode_message is the first message, advance_message each next one).
"""

from __future__ import annotations

import copy
from engine_re.tester import MAX_FAILURES
from engine_re.trace import Trace

_SYSTEM = """# Goal
You are given a recording of someone playing a game: every action they took and every frame the game
returned. Write engine.py, a Python model of that game, so that replaying the recorded actions through it
gives the same result after every action. Use the simplest general rules that explain what you see: the
engine is later also played on action sequences nobody recorded, where general rules hold up and special
cases do not.

# Setup
- The game shows a 64x64 screen of colours 0-15 and is played in levels, in order. Actions: 0 RESET,
  1 up, 2 down, 3 left, 4 right, 5 interact, 6 click at a screen pixel (x, y), 7 undo. This game
  advertises a fixed subset of them (given in the first message).
- engine.py already exists. Its top part is a FIXED block that you must not change: the Sprite, Action,
  View and State classes and the rules for drawing a State. Below it you write two functions:
  - make_level(n) -> State: the state at the start of level n (0-based): grid size, every sprite
    (border, background, objects, score or budget displays) and hidden values in state.vars.
  - step(state, action): apply one action to the state, in place. Set state.status = "level_solved"
    or "game_over" when that happens.
- The harness runs your engine. It calls make_level(n) once per level and hands step() a fresh copy of
  that state whenever the level starts (on entering it and after every RESET); RESET never reaches step().
  After step(), "level_solved" starts the next level (WIN after the last one) and "game_over" ends the
  game (then only RESET is accepted). It turns clicks into grid cells and draws the state.

# Drawing (what the harness does; render(state) in python does the same)
1. Start from a 64x64 screen filled with colour 5.
2. Draw every visible sprite: lowest layer first, sprites on the same layer in list order (later on top).
   Pixels -1 (transparent) and -2 (invisible but solid) are not drawn. Each sprite is drawn as
   sprite.render(): its pixels rotated clockwise by .rotation, then flipped (.mirror_ud, .mirror_lr),
   then scaled by .scale.
   - Grid sprites (screen=False) sit on the logical grid (w, h) = state.grid. The grid is scaled by
     s = state.view.scale, or min(64 // w, 64 // h) when that is None, and centred: grid cell (gx, gy)
     covers screen pixels x in [ox + gx*s, ox + gx*s + s), y in [oy + gy*s, oy + gy*s + s), with
     ox = (64 - w*s) // 2 and oy = (64 - h*s) // 2. Anything outside the grid is cut off.
   - Screen sprites (screen=True) sit directly on screen pixels, unscaled (for displays in the border).
3. Turn the finished frame clockwise by state.view.rotation, then flip it if state.view.mirror_ud
   (top-bottom) and state.view.mirror_lr (left-right).
Clicks: action.x, action.y is the clicked screen pixel; action.cell is the grid cell (gx, gy) under it
once step 3 is undone, or None outside the grid. Coordinates: x is the column, y the row, (0, 0) top-left.

# Tests (run_tests; finish runs them too)
- Contract: the fixed block is unchanged; make_level(n) returns a valid State for every recorded level;
  step() accepts every advertised action; the same actions always give the same result.
- Acceptance: the recorded actions are replayed. After every action, your final frame (every pixel) and
  the game status (NOT_FINISHED / WIN / GAME_OVER, levels completed) must equal the recording. When the
  real game animated an action, only its last frame is compared.
- The recording is replayed in the order it was played, and the report stops at the first failing step
  (or after up to 10, if you ask). For the first one you get: __REPORT_IMAGES__; for each region, the colours and your sprites there;
  what your step() printed; and the python command that reproduces the step.

# Built-in python functions
These are python functions: call them in your code inside the python tool (the python tool's
description has the details). They are not separate tools.
- read(path="engine.py", offset=None, limit=None): print engine.py, each line with a LINE#HASH anchor.
- edit(path="engine.py", edits=[...]): change engine.py at those anchors. engine.py changes only
  through edit() and undo() called in the python tool; writing the file any other way is blocked.
- undo(n=1, to=None): put engine.py back as it was n changes ago, or to="best".
- render(state): draw a State as a 64x64 array, exactly as the tests do.
- __SHOW_LINE__
- try_step(i): your State before and after recorded step i, what your step() printed, and where your
  frame differs from the recording.
- auto_sprites(level): sprite code that draws the first frame of a level exactly, ready for edit().
- summarize_levels(): each level's first frame, the steps played in it and how it ended.
They come with S (the recording), np and the classes Sprite, Action, View, State. All these names are
reserved: code that defines or assigns any of them is rejected before it runs.

# How to work
The recording shows only part of what the game can do, so its real rules cannot always be known from
it. Your job is to reproduce what was observed, with the simplest general mechanism that explains it:
one rule that covers many steps rather than special cases, and nothing the recording gives no evidence for.
Work through the recording in order, one step at a time:
1. Before your first turn the harness puts auto_sprites(0)'s code into make_level, so that level 0's first
   frame is drawn, and runs the tests; the first message shows what they report. Start with the first
   step that fails there: usually step 1, the first action of level 0.
2. Make that step pass with the simplest, most logical mechanism, while every earlier step still passes;
   then take the next failing step. Work on the step in front of you, not on later steps or levels.
3. When a step contradicts a rule you wrote, replace the rule with the simplest one that explains all
   the steps so far, instead of adding a special case.
4. All levels are the same game. Keep one set of sprite kinds for the whole game (pixels, tags,
   collision) and describe each level by where those kinds go and how they are shown there: moved,
   turned, mirrored, scaled or recoloured, and the level's grid and view. A new level reuses the kinds
   and rules it shares with earlier levels and only adds what it introduces; the earlier levels' steps
   must keep passing. auto_sprites(n) recognises pieces that are an existing kind turned, mirrored,
   scaled or recoloured, and writes them that way.
Never hard-code recorded frames or anything keyed to the step number. Print whatever helps you debug
inside step(); the test report and try_step show it.

# Example: how a session goes
A made-up game where a blue piece moves on a grid; your game will differ. The reports are shortened.
Before turn 1 the harness put auto_sprites(0)'s code into make_level and ran the tests; the first message
showed: step 1 (ACTION4) is the first failure; 1 step passes before it.
     [1] your #3 "shape_9_2x2_a1b2" at x=4; the recording shows it one cell to the right.
From there every round is the same: read the code, look at the failing step, edit, run the tests.

Turn 1, python:
    read(offset=330, limit=15)      # the lines of step()
    before, after = try_step(1)     # what your step() did at step 1
  -> no sprite changed: step() is still empty.
Turn 2, python: edit() step() so that ACTION4 moves the blue piece one cell right with
  state.try_move(piece, 1, 0); then run_tests() in the same turn.
  -> engine.py: replaced lines 335-336 with 4 lines. Syntax OK.
  -> steps 0-6 pass; step 7 (ACTION4) is the first failure: the piece should not have moved.
     [1] your #3 at x=9; in the recording it stays at x=8, next to a grey block.
Turn 3, python: read() make_level's sprite list; try_step(7).
Turn 4, python: edit() the grey blocks' kind to be collidable (they are walls); then run_tests().
  -> steps 0-15 pass; step 16 is the first failure ...
And so on: take the first failing step, find the simplest rule that explains it and every step before it,
change the code, test again. (When engine.py changed in a turn and you did not run the tests, the harness
runs them at the end of the turn.) When the tests reach level 1, call auto_sprites(1) and add level 1 to
make_level, reusing level 0's sprite kinds. Do not study later levels before the steps in front of you
pass: the recording will still be there when you get to them.
"""

# The show() line of the built-in functions section.
_SHOW_LINE = {
    True: "show(*frames, titles=None, boxes=None): look at frames or States as images, with boxes.",
    False: "show(*frames, titles=None, boxes=None): print frames or States as hex digits, with boxes.",
}

# The third Tests item, which depends on whether reports carry images (the line breaks differ).
_REPORT_IMAGES = {
    True: "your frame and the recorded frame as images\n  with the differing regions boxed and numbered",
    False: "the regions where your frame differs from the\n  recorded one, numbered, with their pixels",
}

_PYTHON = """Run Python in a persistent kernel: variables and imports survive between calls. Prints what your code
prints plus the value of the last expression. Long output is cut, so print compact summaries or small
crops, never whole frames. Preloaded:

The recording
- np (numpy), and the fixed-block classes Sprite, Action, View, State.
- S: the recording, a list of steps. S[i] is step i; S[0] is the RESET that starts the game.
  - S[i].action: an Action. .id is 0 (RESET) to 7; for a click (6), .x and .y are the screen pixel.
  - S[i].frames: numpy int8 array, shape (n, 64, 64): every frame the game returned for this action,
    in order; n > 1 when the action was animated. Index a frame as frame[y, x] (row, column).
  - S[i].last: S[i].frames[-1], the frame after the action: the one the tests compare. The frame
    before step i is S[i-1].last.
  - S[i].state: "NOT_FINISHED", "WIN" or "GAME_OVER" after the action.
  - S[i].levels_completed: levels completed after the action; step i is played in level
    S[i-1].levels_completed.
  - S[i].win_levels, S[i].available_actions.
- summarize_levels(): print one row per level: its first frame (S[k].last), the steps played in it and
  their actions, the animated steps, RESETs and game overs, and the step that solved it.

engine.py (change it by calling edit() or undo() in your python code; writing the file any other way,
such as open('engine.py', 'w'), is blocked)
- read(path="engine.py", offset=None, limit=None): print the file, every line as LINE#HASH:content.
  Those anchors are how edit() addresses lines. offset: first line (1-based); limit: number of lines.
  Long output is cut; it says which offset to continue from. The FIXED block is folded unless offset
  asks for its lines.
- edit(path="engine.py", edits=[...]): change the file at LINE#HASH anchors from the latest read() or
  edit() output. All edits in one call are checked against the same version of the file and applied
  together. Each edit is one of:
    {"op": "replace", "pos": "12#MQ", "end": "15#VR", "lines": [...]}  replace line pos (or pos..end)
    {"op": "append", "pos": "12#MQ", "lines": [...]}   insert after pos (no pos: at the end)
    {"op": "prepend", "pos": "12#MQ", "lines": [...]}  insert before pos (no pos: at the start)
    {"op": "replace_text", "oldText": "...", "newText": "..."}  replace one exact, unique text
  lines is the new content (a list of lines, or one string), with its indentation; [] deletes. It can
  come straight from your code, e.g. lines=auto_sprites(0) or lines=f"RINGS = {rings!r}", so generated
  data is never retyped. Edits in one call must not overlap or touch adjacent lines. A stale anchor (the
  file changed since your read) is rejected: read again. Edits inside the FIXED block are rejected.
  Prints what changed, a syntax check, and fresh anchors around the change.
- undo(n=1, to=None): put engine.py back as it was n changes ago, or to="best": the version that
  passed the most steps before its first failure so far (to=k: version k). Every change (each edit()
  call, and undo itself) is kept as a numbered version, so nothing is lost: undo() right after an
  undo() brings the undone change back. Prints the recent versions (what changed, and the test result
  of each version that was tested), what this undo restored, and a reminder to read() again for fresh
  anchors.

Running your engine
- render(state) -> np.ndarray (64, 64): draws a State exactly as the tests do (the Drawing rules), with
  the same code. Use it to try things on the drawing: change a sprite or state.view and render again.
__SHOW__
- try_step(i, state=None, action=None) -> (before, after): loads engine.py fresh, gives your State
  just before recorded step i (or `state`), applies S[i].action (or `action`: an id, or (6, x, y) for a
  click), prints what your step() printed, what changed in your state (sprites by index #k and name,
  vars), and the regions where your frame differs from S[i].last with your sprites in each. Returns
  copies of both states. try_step(i, level=L) starts at level L as run_tests(level=L) does.

Generating code
- auto_sprites(level, grid=None, frame=None, region=None, merge=False) -> str: Python code for sprites
  that draw the first frame of `level` exactly (or a given frame, or a region (x0, y0, x1, y1) of it):
  the frame split into same-colour connected pieces (merge=True: touching pieces of different colours
  become one sprite), identical pieces sharing one pixel list whose name comes from its content, so the
  same shape gets the same name in every call; pixel lists already defined in engine.py are not repeated,
  and a piece that is an existing one turned, mirrored, scaled or recoloured is written as that one
  with the matching rotation, mirror, scale or colour change. It prints whether the code draws the
  frame exactly. A starting point only: real objects often have several colours, anything hidden or
  covered is missing, transparency is unknown, layers, tags and collidability are guesses, and the
  grid size is guessed unless you pass grid=(w, h)."""

_SHOW = {
    True: """- show(*frames, titles=None, boxes=None): look at frames as images. Each item is a 64x64 frame (e.g.
  S[i].last, render(state)) or a State (rendered first). They are shown side by side, enlarged, in a
  message right after this call's output (at most 4 per call). boxes: [(x0, y0, x1, y1), ...] in screen
  pixels, outlined on every image. Older images are later replaced by a placeholder to save context.""",
    False: """- show(*frames, titles=None, boxes=None): look at frames (images are off, so it prints hex digits). Each
  item is a 64x64 frame (e.g. S[i].last, render(state)) or a State (rendered first), at most 4 per call.
  boxes: [(x0, y0, x1, y1), ...] in screen pixels: it prints the boxed pixels of every frame side by
  side; without boxes, every frame at half resolution.""",
}

_RUN_TESTS = """Run the contract tests, then replay the recording in order: from step 0, or from the start of level L
(make_level(L)) when level=L is given. Stops after `failures` failing steps (1 to 10, default 1).
Reports how many steps pass before the first failure, the first failure in full (images, regions,
your sprites there, what step() printed, the try_step command), and one line per further failure."""

_RUN_TESTS_NO_IMAGES = _RUN_TESTS.replace("(images, regions,\nyour sprites there,", "(regions with their\npixels, your sprites there,")


# --- The stepwise harness (v6) -------------------------------------------------------------------

_SYSTEM_STEP = """# Goal
You are building engine.py, a Python model of a game, from a recording of someone playing it: every
action they took and every frame the game returned. The harness replays the recording through
engine.py step by step. When a step does not give the recorded result, it stops there and asks you to
fix that step. Once the steps up to it pass, it replays on and tells you, in this conversation, how
many more steps passed and which step breaks next; you see the recording only up to that step. This
goes on until the whole recording passes.
Fix each step with the simplest general rule that explains it and keeps the earlier steps passing:
the engine is later also played on action sequences nobody recorded, where general rules hold up and
special cases keyed to step numbers do not.

""" + _SYSTEM[_SYSTEM.index("# Setup") : _SYSTEM.index("# Tests")] + """# Tests (run_tests; finish runs them too)
- Contract: the fixed block is unchanged; make_level(n) returns a valid State for every level reached so
  far; step() accepts every advertised action; the same actions always give the same result.
- Acceptance: the recorded steps from step 0 to the step you are fixing are replayed in order. After
  every action, your final frame (every pixel) and the game status (NOT_FINISHED / WIN / GAME_OVER,
  levels completed) must equal the recording. When the real game animated an action, only its last
  frame is compared.
- The report stops at the first failing step (or after up to 10, if you ask). For it you get:
  __REPORT_IMAGES__;
  for a click, the grid cell it lands on and your sprites there; for each region, the colours and
  your sprites there; what your step() printed; and the python command that reproduces the step.

# Built-in python functions
These are python functions: call them in your code inside the python tool (the python tool's
description has the details). They are not separate tools.
__STEP_BULLET__
- read(path="engine.py", offset=None, limit=None): print engine.py, each line with a LINE#HASH anchor.
- edit(path="engine.py", edits=[...]): change engine.py at those anchors. engine.py changes only
  through edit() and undo() called in the python tool; writing the file any other way is blocked.
- undo(n=1, to=None): put engine.py back as it was n changes ago, or to="best".
- render(state): draw a State as a 64x64 array, exactly as the tests do.
- __SHOW_LINE__
- try_step(i): your State before and after step i (0 to step.index), what your step() printed, and
  where your frame differs from the recording.
- auto_sprites(level) or auto_sprites(frame=...): sprite code that draws a frame exactly, ready for
  edit(); for a level, its first frame.
They come with np and the classes Sprite, Action, View, State. All these names are reserved: code that
defines or assigns any of them is rejected before it runs.

# How to work
1. Look at what the step did: compare step.before with step.after, and read the report's regions and
   what the click hit.
2. Find the simplest rule that explains this step and agrees with what engine.py already does for the
   earlier steps. The recording shows only part of what the game can do: reproduce what you see, with
   no rule the steps give no evidence for.
3. Change engine.py with edit(), run the tests, fix what they report (an earlier step that now breaks
   counts too), and call finish when they pass.
4. When the step starts a new level (the frame after it shows the next level), make_level must draw
   that level: auto_sprites(n) gives code for its first frame. Reuse the sprite kinds engine.py already
   has where they fit.
5. Keep engine.py's comments up to date with the rules you found: older parts of this conversation are
   shortened as it grows, and engine.py is what stays.
Never hard-code recorded frames or anything keyed to the step number. Print whatever helps you debug
inside step(); the test report and try_step show it.
"""

_STEP_BULLET = {
    False: """- step: the step to fix, the only part of the recording shown: step.before, step.action, step.after
  (the frame the tests compare), step.frames, step.level.""",
    True: """- S: the recording so far, steps 0 to the step to fix (later steps are not loaded yet); step = S[-1],
  the step to fix: step.before, step.action, step.after (the frame the tests compare), step.frames,
  step.level.
- summarize_levels(): the levels reached so far, the steps played in each and how they ended.""",
}

_RECORDING_SINGLE = _PYTHON[_PYTHON.index("The recording\n") : _PYTHON.index("engine.py (change it")]
_RECORDING_STEP = """The step to fix
- np (numpy), and the fixed-block classes Sprite, Action, View, State.
- step: the recorded step that breaks, the only part of the recording shown (later steps are not
  loaded; earlier ones already pass, and try_step(i) replays them on your engine).
  - step.index: its number; steps 0 to step.index - 1 pass.
  - step.action: an Action. .id is 0 (RESET) to 7; for a click (6), .x and .y are the screen pixel.
  - step.before: the frame before the step, a numpy int8 array (64, 64) indexed frame[y, x] (row,
    column); your engine already draws it. None for step 0.
  - step.after: the frame after the step, the one the tests compare. step.frames: every frame the game
    returned for it, (n, 64, 64), in order; n > 1 when the action was animated.
  - step.level: the level it is played in; step.state ("NOT_FINISHED", "WIN" or "GAME_OVER") and
    step.levels_completed: after it.

"""
_TRY_STEP_SINGLE = _PYTHON[_PYTHON.index("- try_step(i, state=None"): _PYTHON.index("Generating code")]
_TRY_STEP_STEP = """- try_step(i, state=None, action=None) -> (before, after): loads engine.py fresh, gives your State
  just before recorded step i, 0 to step.index (or `state`), applies that step's action (or `action`:
  an id, or (6, x, y) for a click), prints what your step() printed, what changed in your state
  (sprites by index #k and name, vars), and the regions where your frame differs from the recording
  with your sprites in each. Returns copies of both states.

"""
_AUTO_SPRITES_HEAD_SINGLE = """- auto_sprites(level, grid=None, frame=None, region=None, merge=False) -> str: Python code for sprites
  that draw the first frame of `level` exactly (or a given frame, or a region (x0, y0, x1, y1) of it):
"""
_AUTO_SPRITES_HEAD_STEP = """- auto_sprites(level, grid=None, frame=None, region=None, merge=False) -> str: Python code for sprites
  that draw the first frame of `level`, a level reached so far, exactly (or a given frame such as
  step.after, or a region (x0, y0, x1, y1) of it):
"""
_RECORDING_HISTORY = """The recording so far
- np (numpy), and the fixed-block classes Sprite, Action, View, State.
- S: the recording up to the step to fix, a list of steps: S[0] is the RESET that starts the game and
  S[-1] is the step to fix. Later steps are not loaded yet: each conversation sees the recording up to
  its own breaking step. Steps 0 to len(S) - 2 already pass.
  - S[i].action: an Action. .id is 0 (RESET) to 7; for a click (6), .x and .y are the screen pixel.
  - S[i].frames: numpy int8 array, shape (n, 64, 64): every frame the game returned for this action,
    in order; n > 1 when the action was animated. Index a frame as frame[y, x] (row, column).
  - S[i].last: S[i].frames[-1], the frame after the action: the one the tests compare. The frame
    before step i is S[i-1].last.
  - S[i].state: "NOT_FINISHED", "WIN" or "GAME_OVER" after the action.
  - S[i].levels_completed: levels completed after the action; step i is played in level
    S[i-1].levels_completed.
  - S[i].win_levels, S[i].available_actions.
- step: the step to fix, S[-1]: step.index, step.action, step.before (S[-2].last), step.after
  (step.last), step.frames, step.level (the level it is played in), step.state, step.levels_completed.
- summarize_levels(): print one row per level reached so far: its first frame (S[k].last), the steps
  played in it and their actions, the animated steps, RESETs and game overs, and how it ended.

"""
_PYTHON_HISTORY = (
    _PYTHON.replace(_RECORDING_SINGLE, _RECORDING_HISTORY)
    .replace(_TRY_STEP_SINGLE, _TRY_STEP_STEP)
    .replace(_AUTO_SPRITES_HEAD_SINGLE, _AUTO_SPRITES_HEAD_STEP)
)
_PYTHON_STEP = (
    _PYTHON.replace(_RECORDING_SINGLE, _RECORDING_STEP)
    .replace(_TRY_STEP_SINGLE, _TRY_STEP_STEP)
    .replace(_AUTO_SPRITES_HEAD_SINGLE, _AUTO_SPRITES_HEAD_STEP)
)
assert _PYTHON_STEP.count("S[") == 0 and "summarize_levels" not in _PYTHON_STEP

_RUN_TESTS_STEP = """Run the contract tests, then replay the recording in order from step 0 to the step you are fixing.
Stops after `failures` failing steps (1 to 10, default 1). Reports how many steps pass before the
first failure, the first failure in full (images, regions, what a click hit, your sprites there, what
step() printed, the try_step command), and one line per further failure."""

_FINISH_STEP = """Say the step is fixed. Runs the tests first (steps 0 to the step you are fixing): when they all pass,
the harness replays on and tells you the next step that breaks (or that the whole recording passes);
otherwise you get the report and go on. summary: the rule you added or changed."""

_FINISH = """Ask to end the session. Runs the tests first: if anything fails you get the report and the session
goes on; it ends only when every test passes. summary: what the engine implements."""


def system_prompt(
    match: str = "final", interface: str = "simple", images: bool = True, mode: str = "single", history: bool = True
) -> str:
    """The system prompt. Only the make_level/step interface scored on final frames is offered.
    mode "single": one session over the recording; "step": fix one breaking step (v6), seeing the
    recording up to it (history) or only that step."""
    if interface != "simple":
        raise ValueError("the agent offers only the simple interface (make_level/step)")
    if match != "final":
        raise ValueError("the simple interface produces one frame per action, so it is scored with match='final'")
    text = {"single": _SYSTEM, "step": _SYSTEM_STEP.replace("__STEP_BULLET__", _STEP_BULLET[history])}[mode]
    return text.replace("__REPORT_IMAGES__", _REPORT_IMAGES[images]).replace("__SHOW_LINE__", _SHOW_LINE[images])


SYSTEM_PROMPT = system_prompt()


def _python_description(images: bool, mode: str = "single", history: bool = True) -> str:
    if mode == "step" and history:
        return _PYTHON_HISTORY.replace("__SHOW__", _SHOW[images])
    if mode == "step":
        return _PYTHON_STEP.replace("__SHOW__", _SHOW[images].replace("S[i].last", "step.after"))
    return _PYTHON.replace("__SHOW__", _SHOW[images])


def tools(images: bool = True, mode: str = "single", history: bool = True) -> list[dict]:
    """The tool schemas: python, run_tests and finish (mode "step": the stepwise harness's texts,
    and run_tests without `level`; history: whether python shows the recording so far)."""
    schemas = _tools(images, mode, history)
    if mode == "step":
        run_tests = schemas[1]["function"]
        run_tests["description"] = _RUN_TESTS_STEP if images else _RUN_TESTS_STEP.replace("(images, regions,", "(regions with their pixels,")
        del run_tests["parameters"]["properties"]["level"]
        schemas[2]["function"]["description"] = _FINISH_STEP
        schemas[2]["function"]["parameters"]["properties"]["summary"]["description"] = "The rule you added or changed."
    return schemas


def _tools(images: bool, mode: str, history: bool) -> list[dict]:
    return copy.deepcopy(
        [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _python_description(images, mode, history),
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "Python code to run."}},
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": _RUN_TESTS if images else _RUN_TESTS_NO_IMAGES,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "integer",
                                "description": "Replay only this level, from its start (make_level(level)). Omit to replay from step 0.",
                            },
                            "failures": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_FAILURES,
                                "description": "How many failing steps to report before stopping, 1 to 10 (default 1): "
                                "the first in full, the others one line each.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": _FINISH,
                    "parameters": {
                        "type": "object",
                        "properties": {"summary": {"type": "string", "description": "What the engine implements."}},
                        "required": ["summary"],
                    },
                },
            },
        ]
    )


TOOLS = tools(True)


_ACTION_WORDS = {0: "RESET", 1: "up", 2: "down", 3: "left", 4: "right", 5: "interact", 6: "click", 7: "undo"}


def recording_summary(trace: Trace) -> str:
    """The recording in one sentence; summarize_levels() in python has the levels."""
    steps = trace.steps
    played = len({0} | {steps[i - 1].levels_completed for i in range(1, len(steps))})
    actions = ", ".join(f"{a} ({_ACTION_WORDS.get(a, a)})" for a in steps[0].available_actions)
    return (
        f"{len(steps)} steps over {played} level(s) (S[0] is the RESET that starts the game); it ends {steps[-1].state} "
        f"with {steps[-1].levels_completed} of {steps[0].win_levels} levels completed. The actions this game accepts: "
        f"{actions}. summarize_levels() in python lists each level: its first frame, the steps played in it and how it "
        "ended."
    )


def _first_task(first_fail: int | None) -> str:
    if first_fail is None:
        return "Every test passes already: call finish."
    if first_fail == 0:
        what = "step 0 pass, level 0's first frame (make_level(0) does not draw it exactly yet), then step 1"
    elif first_fail == 1:
        what = "step 1 pass, the first action of level 0"
    else:
        what = f"step {first_fail} pass, the first step that fails"
    return (
        f"Your first task: make {what}. Work on that step only: read the code it runs, look at the step with "
        f"try_step({first_fail}), edit, run the tests. Then go on to the next failing step, one step at a time, in "
        "the order they were played, and add a level with auto_sprites(n) when the tests reach it."
    )


def first_user_message(game: str, trace: Trace, engine_read: str, opening: dict | None = None) -> str:
    """The opening message: the recording in one sentence, what the harness did before the first turn
    (auto_sprites(0) put into make_level, then run_tests: `opening` holds "sprites", the summary
    auto_sprites printed, "report", the test report, and "first_fail"), engine.py as read() shows it,
    and the first task. Without `opening` the model is asked to do that first round itself."""
    head = f"Game: {game}. Write engine.py for it.\n\nThe recording: {recording_summary(trace)}\n"
    shown = f"engine.py now, as read() shows it (the FIXED block folded):\n\n{engine_read}"
    if opening is None:
        return f"""{head}
{shown}

Your first task: put auto_sprites(0)'s code into make_level with edit(), so that level 0's first frame is drawn,
and run the tests. Then make the first failing step pass, then the next one, one step at a time, in the order
they were played."""
    sprites = "\n".join("   " + line if line else "" for line in opening["sprites"].strip().splitlines())
    return f"""{head}
Before your first turn the harness did the first round:
1. auto_sprites(0) wrote sprites that draw level 0's first frame, and make_level now returns them (for every
   level, for now). What it printed:
{sprites}
2. run_tests() then reported:

{opening["report"].strip()}

{shown}

{_first_task(opening.get("first_fail"))}"""


def _action_text(action) -> str:
    if action.id == 6:
        return f"a click at screen pixel ({action.x}, {action.y})"
    return "RESET" if action.id == 0 else f"ACTION{action.id} ({_ACTION_WORDS.get(action.id, '?')})"


def episode_message(game: str, trace: Trace, k: int, report: str, engine_read: str, history: bool = True) -> str:
    """The first message of a stepwise conversation: fix the breaking step k (steps 0..k-1 pass).
    `trace` holds at least steps 0..k; `report` is the test report of the replay up to step k."""
    s = trace.steps[k]
    level = trace.steps[k - 1].levels_completed if k > 0 else 0
    notes = []
    if s.state == "WIN":
        notes.append(f"This step solves level {level}, the last one: the game ends with WIN.")
    elif s.levels_completed > level:
        notes.append(
            f"This step solves level {level}: the frame after it is level {s.levels_completed}'s first frame, which "
            f"make_level({s.levels_completed}) must draw (auto_sprites({s.levels_completed}) gives code for it; reuse the "
            "sprite kinds engine.py already has where they fit)."
        )
    if s.state == "GAME_OVER":
        notes.append("After this step the game is over (GAME_OVER).")
    if k == 0:
        passed = "Step 0, level 0's first frame, does not pass yet."
    else:
        before = "Step 0 of the recording passes" if k == 1 else f"Steps 0-{k - 1} of the recording pass"
        passed = f"{before} with your engine.py; step {k} is the first that does not."
    shown = (f"In python, S holds the recording so far, steps 0-{k} (later steps are not loaded), and `step` is step {k} "
             "(S[-1]): step.before, step.action, step.after, step.frames." if history else
             "In python, `step` holds this step: step.before, step.action, step.after, step.frames.")
    return f"""Fix the breaking test: step {k}.

Game: {game}. {passed}
Step {k}: {_action_text(s.action)}, played in level {level}. The game returned {s.n_frames} frame(s) for it; the tests compare
the last.{(" " + " ".join(notes)) if notes else ""}
{shown}

The test report:

{report.strip()}

engine.py now, as read() shows it (the FIXED block folded):

{engine_read}

Fix step {k}: find the simplest rule that explains it and keeps the earlier steps passing, change engine.py with
edit(), run the tests, and call finish when they pass."""


def advance_message(trace: Trace, fixed: int, k: int, report: str, history: bool = True) -> str:
    """The user message when steps 0..fixed pass and the harness replayed on to step k, the next that
    fails (`trace` is the whole recording; the model now sees it up to k)."""
    s = trace.steps[k]
    level = trace.steps[k - 1].levels_completed if k > 0 else 0
    if k == fixed + 1:
        passed = f"Steps 0-{fixed} pass. The next step, {k}, fails."
    else:
        between = f"step {fixed + 1}" if k == fixed + 2 else f"steps {fixed + 1}-{k - 1}"
        passed = (f"Steps 0-{fixed} pass. The harness replayed on: {between} ({k - fixed - 1} more step"
                  f"{'s' if k - fixed - 1 > 1 else ''}) passed without error. Step {k} is the next that fails.")
    notes = []
    if s.state == "WIN":
        notes.append(f"It solves level {level}, the last one: the game ends with WIN.")
    elif s.levels_completed > level:
        notes.append(
            f"It solves level {level}: the frame after it is level {s.levels_completed}'s first frame, which "
            f"make_level({s.levels_completed}) must draw (auto_sprites({s.levels_completed}) gives code for it; reuse the "
            "sprite kinds engine.py already has where they fit)."
        )
    if s.state == "GAME_OVER":
        notes.append("After it the game is over (GAME_OVER).")
    shown = (f"S now holds the recording up to step {k}, and `step` is step {k}." if history else f"`step` is now step {k}.")
    return f"""{passed}
Step {k}: {_action_text(s.action)}, played in level {level}; {s.n_frames} frame(s), the tests compare the last.{(" " + " ".join(notes)) if notes else ""}
{shown}

The test report:

{report.strip()}

Fix step {k} the same way, keeping steps 0-{k - 1} passing, and call finish when they pass."""


def resume_user_message(game: str, trace: Trace, turns: int, test_report: str, engine_read: str, notes: str = "") -> str:
    notes_part = f"\nWhat that session left behind:\n\n{notes}\n" if notes else ""
    return f"""Game: {game}. Write engine.py for it.

The recording: {recording_summary(trace)}

This continues an earlier session on this game ({turns} turns) that was interrupted. Its conversation is gone and the
python kernel was restarted (its variables are gone), but engine.py and its versions (undo) are kept.
{notes_part}
The current test report of engine.py:

{test_report}

engine.py now, as read() shows it:

{engine_read}"""
