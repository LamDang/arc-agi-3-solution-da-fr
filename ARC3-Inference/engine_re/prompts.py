"""System prompt, first user message and tool schemas for the reverse-engineering agent.

The agent has three tools: python (a kernel with the recording, read/edit/undo for engine.py and
helpers to run and look at it), run_tests and finish. ``images`` says whether test reports and
show() come with pictures; the texts follow it.
"""

from __future__ import annotations

import copy
from collections import Counter

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
They come with S (the recording), np and the classes Sprite, Action, View, State. All these names are
reserved: code that defines or assigns any of them is rejected before it runs.

# How to work
The recording shows only part of what the game can do, so its real rules cannot always be known from
it. Your job is to reproduce what was observed, with the simplest general mechanism that explains it:
one rule that covers many steps rather than special cases, and nothing the recording gives no evidence for.
Work through the recording in order:
1. Start with the first frame of level 0: auto_sprites(0) gives sprite code that draws it exactly. Put it
   into make_level with edit() and run run_tests.
2. Then make the steps pass one at a time, in the order they were played. run_tests shows the first step
   that fails: find the simplest, most logical mechanism that makes it pass while every earlier step
   still passes, then test again.
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
A made-up game where a blue piece moves on a grid; your game will differ. Every round is the same four
moves: read the code, run the tests, edit the code, run the tests again. The reports are shortened.

Turn 1, python:
    read()                          # engine.py with its LINE#HASH anchors
  -> 247#KW:def make_level(n: int) -> State:  ...  251#PR:    return State(  ...  261#KB:    )
Turn 2, run_tests()
  -> step 0 is the first failure; no step passes before it. No sprite of yours draws here.
Turn 3, python (the first edit uses auto_sprites):
    code = auto_sprites(0)          # sprite code that draws level 0's first frame exactly
    edit(edits=[{"op": "prepend", "pos": "247#KW", "lines": code},
                {"op": "replace", "pos": "251#PR", "end": "261#KB",
                 "lines": f"    return State(grid={code.grid}, sprites=level_0_sprites())"}])
  -> engine.py: inserted 37 lines; replaced lines 251-261 with 1 line. Syntax OK.
Turn 4, run_tests()
  -> step 1 (ACTION4) is the first failure; 1 step passes before it.
     [1] your #3 "shape_9_2x2_a1b2" at x=4; the recording shows it one cell to the right.
Turn 5, python:
    read(offset=290, limit=15)      # the lines of step()
    before, after = try_step(1)     # what your step() did at step 1
  -> no sprite changed: step() is still empty.
Turn 6, run_tests()                 # where you stand before changing anything
  -> step 1 (ACTION4) is still the first failure.
Turn 7, python: edit() step() so that ACTION4 moves the blue piece one cell right with
  state.try_move(piece, 1, 0).
Turn 8, run_tests()
  -> steps 0-6 pass; step 7 (ACTION4) is the first failure: the piece should not have moved.
     [1] your #3 at x=9; in the recording it stays at x=8, next to a grey block.
Turn 9, python: read() make_level's sprite list. Turn 10, python: edit() the grey blocks' kind to be
  collidable (they are walls). Turn 11, run_tests() -> steps 0-15 pass; step 16 is the first failure ...
And so on, always read, run the tests, edit, run the tests: take the first failing step, find the simplest
rule that explains it and every step before it, change the code, test again. When the tests reach level 1,
call auto_sprites(1) and add level 1 to make_level, reusing level 0's sprite kinds. Do not study later
levels before the steps in front of you pass: the recording will still be there when you get to them.
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


_FINISH = """Ask to end the session. Runs the tests first: if anything fails you get the report and the session
goes on; it ends only when every test passes. summary: what the engine implements."""


def system_prompt(match: str = "final", interface: str = "simple", images: bool = True) -> str:
    """The system prompt. Only the make_level/step interface scored on final frames is offered."""
    if interface != "simple":
        raise ValueError("the agent offers only the simple interface (make_level/step)")
    if match != "final":
        raise ValueError("the simple interface produces one frame per action, so it is scored with match='final'")
    return _SYSTEM.replace("__REPORT_IMAGES__", _REPORT_IMAGES[images]).replace("__SHOW_LINE__", _SHOW_LINE[images])


SYSTEM_PROMPT = system_prompt()


def _python_description(images: bool) -> str:
    return _PYTHON.replace("__SHOW__", _SHOW[images])


def tools(images: bool = True) -> list[dict]:
    """The tool schemas: python, run_tests and finish."""
    return copy.deepcopy(
        [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _python_description(images),
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


def describe_trace(trace: Trace) -> str:
    steps = trace.steps
    acts = Counter(str(s.action) if s.action.id != 6 else "ACTION6 (click)" for s in steps)
    frames = Counter(s.n_frames for s in steps)
    overs = sum(s.state == "GAME_OVER" for s in steps)
    starts = trace.level_starts()
    return (
        f"- {len(steps)} steps; S[0] is the RESET that starts the game. Actions played: {dict(acts)}\n"
        f"- Advertised actions: {steps[0].available_actions}; win_levels: {steps[0].win_levels}\n"
        f"- Levels reached (level: the first step whose frame shows it): {starts}; the recording ends "
        f"{steps[-1].state} with {steps[-1].levels_completed} level(s) completed\n"
        f"- Steps ending in GAME_OVER: {overs}; frames per step (frames: steps): {dict(sorted(frames.items()))}"
    )


def first_user_message(game: str, trace: Trace, engine_read: str) -> str:
    """The opening message: the recording's facts and engine.py as read() shows it (with anchors)."""
    return f"""Game: {game}. Write engine.py for it.

The recording:
{describe_trace(trace)}

engine.py now, as read() shows it (LINE#HASH anchors for edit()):

{engine_read}

Start as in the example of the system prompt: read(), run_tests, then edit() with auto_sprites(0) to put
level 0 into make_level, then run_tests again."""


def resume_user_message(game: str, trace: Trace, turns: int, test_report: str, engine_read: str, notes: str = "") -> str:
    notes_part = f"\nWhat that session left behind:\n\n{notes}\n" if notes else ""
    return f"""Game: {game}. Write engine.py for it.

The recording:
{describe_trace(trace)}

This continues an earlier session on this game ({turns} turns) that was interrupted. Its conversation is gone and the
python kernel was restarted (its variables are gone), but engine.py and its versions (undo) are kept.
{notes_part}
The current test report of engine.py:

{test_report}

engine.py now, as read() shows it:

{engine_read}"""
