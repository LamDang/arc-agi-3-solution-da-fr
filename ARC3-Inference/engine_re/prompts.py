"""System prompt, first user message and tool schemas for the reverse-engineering agent.

The agent has three tools: python (a kernel with the recording, read_file/edit_file/undo_edit for
engine.py and functions to run and look at it), run_tests and commit_engine. ``images`` says whether test
reports and show_frames() come with pictures; the texts follow it.

Every system prompt has an "# Objects" section (``objects_reference``): a typed reference of what
the model's code handles, written once and shared by the modes: the recorded steps (StepView) and
the recorded action, engine.py's two functions and the FIXED block's classes, and the kernel's
built-in functions. The python tool's description names the preloaded names and points there.

Two modes. "single" (v5): one session over the whole recording. "step" (v6, engine_re.stepwise):
the harness replays the recording and asks to fix the first step that breaks; the agent sees the
recording up to it (`recording` and `step_to_fix`; with history=False only `step_to_fix`) and the
tests replay steps 0 to it. When they pass and the agent submits engine.py with commit_engine, the
harness replays on and, in the same conversation, names the next step that breaks (episode_message is
the first message, advance_message each next one). "play" (v10, engine_re.play_agent): the same agent
playing a live game, with plan rounds (plan_message, the commit_moves tool) and fit rounds
(mismatch_message, then advance_message as in "step").
"""

from __future__ import annotations

import copy
import textwrap

from engine_re import segment
from engine_re import animation  # the ported animation digest
from engine_re.diff_report import COLOR_NAMES
from engine_re.kernel import PRELOADED, PRELOADED_HISTORY, PRELOADED_PLAY, PRELOADED_STEP
from engine_re.tester import MAX_FAILURES, _ranges as _tester_ranges
from engine_re.trace import Trace, action_code

_SYSTEM = """# Goal
You are given a recording of someone playing a game: every action they took and every frame the game
returned. Write engine.py, a Python model of that game, so that replaying the recorded actions through it
gives the same result after every action. Make the tests pass with the most parsimonious model (Occam's
razor): the fewest rules and assumptions that account for every step observed so far.

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

# Drawing (what the harness does; render_state(state) in python does the same)
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

# Sandbox
Your python code runs sandboxed: it can read the workspace, the recording and the Python installation,
and write only inside the workspace (/tmp is refused); no network, no subprocesses. engine.py changes
only through edit_file() and undo_edit().

# Tests (run_tests; commit_engine runs them too)
- Contract: the fixed block is unchanged; make_level(n) returns a valid State for every recorded level;
  step() accepts every advertised action; the same actions always give the same result.
- Acceptance: the recorded actions are replayed. After every action, your final frame (every pixel) and
  the game status (NOT_FINISHED / WIN / GAME_OVER, levels completed) must equal the recording. When the
  real game animated an action, only its last frame is compared.
- The recording is replayed in the order it was played, and the report stops at the first failing step
  (or after up to 10, if you ask). For the first one you get: __REPORT_IMAGES__; for each region, the colours and your sprites there;
  what your step() printed; and the python command that reproduces the step.

__OBJECTS__
# How to work
Reproduce what was observed with the fewest rules and assumptions that account for every step so far,
and nothing the recording gives no evidence for. Per-level constants in the level data (a rate, a budget,
a size) are fine when the steps give no evidence of a formula: do not hunt for one. Do not think about,
model or write code for what has not been observed: for a new level, only draw its first frame, and model
its mechanics when one of its steps fails. Work through the recording in order, one step at a time:
1. Before your first turn the harness puts recording[0].pieces_after.code() into make_level, so that level 0's
   first frame is drawn, and runs the tests; the first message shows what they report. Start with the first
   step that fails there: usually step 1, the first action of level 0.
2. Make that step pass with the simplest mechanism, while every earlier step still passes; then take the
   next failing step. Work on the step in front of you, not on later steps or levels.
3. When a step contradicts a rule you wrote, replace it with the simplest rule that explains all the steps
   so far, instead of adding a special case.
4. All levels are the same game. Keep one set of sprite kinds for the whole game (pixels, tags,
   collision) and describe each level by where those kinds go and how they are shown there: moved,
   turned, mirrored, scaled or recoloured, and the level's grid and view. A new level reuses the kinds
   and rules it shares with earlier levels and only adds what it introduces; the earlier levels' steps
   must keep passing. pieces_after.code() recognises pieces that are one of engine.py's kinds turned,
   mirrored, scaled or recoloured, and writes them that way.
Never hard-code recorded frames or anything keyed to the step number. Print whatever helps you debug
inside step(); the test report and replay_step show it.

# Example: how a session goes
A made-up game where a blue piece moves on a grid; your game will differ. The reports are shortened.
Before turn 1 the harness put recording[0].pieces_after.code() into make_level and ran the tests; the first
message showed: step 1 (ACTION4) is the first failure; 1 step passes before it.
     [1] your #3 "shape_9_2x2_a1b2" at x=4; the recording shows it one cell to the right.
From there every round is the same: read the code, look at the failing step, edit, run the tests.

Turn 1, python:
    read_file(offset=330, limit=15)    # the lines of step()
    print(recording[1].changes)        # what the recorded step 1 changed, piece by piece
    before, after = replay_step(1)     # what your step() did at step 1
  -> moved: SHAPE_9_2x2_a1b2 colour 9 (blue), (4, 3) -> (5, 3), dx=+1 dy=+0
  -> no sprite changed: step() is still empty.
Turn 2, python: edit_file() so that in step() ACTION4 moves the blue piece one cell right with
  state.try_move(piece, 1, 0); then run_tests() in the same turn.
  -> engine.py: replaced lines 335-336 with 4 lines. Syntax OK.
  -> steps 0-6 pass; step 7 (ACTION4) is the first failure: the piece should not have moved.
     [1] your #3 at x=9; in the recording it stays at x=8, next to a grey block.
Turn 3, python: read_file() for make_level's sprite list; replay_step(7).
Turn 4, python: edit_file() to make the grey blocks' kind collidable (they are walls); then run_tests().
  -> steps 0-15 pass; step 16 is the first failure ...
And so on: take the first failing step, find the simplest rule that explains it and every step before it,
change the code, test again. (When engine.py changed in a turn and you did not run the tests, the harness
runs them at the end of the turn.) When the tests reach level 1, add level 1 to make_level from
recording[e].pieces_after.code(), e being the step that enters it, reusing level 0's sprite kinds. Do not study later levels before the steps in front of you
pass: the recording will still be there when you get to them.
"""

# The third Tests item, which depends on whether reports carry images (the line breaks differ).
_REPORT_IMAGES = {
    True: "your frame and the recorded frame as images\n  with the differing regions boxed and numbered",
    False: "the regions where your frame differs from the\n  recorded one, numbered, with their pixels",
}


# --- The Objects reference: everything the model's code handles, shared by every mode -------------
# A typed reference, one entry per name: "name: type  meaning" or "name(args) -> type  meaning" (two
# spaces before the meaning; longer meanings go on on lines indented further). A test checks the
# members listed for each class against the classes themselves.

_OBJECTS_HEAD = """# Objects
A reference of everything your python code handles. In the python tool the namespace starts with np
(numpy), the engine classes Sprite, Action, View and State, and these built-ins, documented below:
  __NAMES__
All these names are reserved: code that defines or assigns any of them is rejected before it runs. The
built-in functions are python functions to call in your code inside the python tool, not separate tools.
The python kernel is persistent for the whole run: every variable, function and import you define stays
until the run ends (moving on to the next step keeps them); define helpers and data once and reuse them
instead of retyping them.
"""

# Which recorded steps python holds, per mode ("history": the stepwise harness with the recording so far).
_RECORDED = {
    "single": """
## The recording: data from the real game
recording: list[StepView]  the recorded steps: recording[i] is step i; recording[0] is the RESET that
    starts the game
""",
    "history": """
## The recording so far: data from the real game
recording: list[StepView]  the recorded steps 0 to the step to fix (later steps are not loaded yet):
    recording[0] is the RESET that starts the game; steps 0 to len(recording) - 2 already pass
step_to_fix: StepView  the step to fix: recording[-1]
""",
    "step": """
## The step to fix: data from the real game
step_to_fix: StepView  the recorded step that breaks, the only one shown (later steps are not loaded;
    steps 0 to step_to_fix.index - 1 already pass, and replay_step(i) replays them on your engine)
""",
    "play": """
## The game so far: data from the game
recording: list[StepView]  every step played so far, in order: recording[0] is the RESET that started the
    game, recording[-1] the last step played; recording[-1].after is the game's current frame
step_to_fix: StepView  in a fit round, the step your replica got wrong: recording[-1]
""",
}

_STEP_VIEW = """StepView: a recorded step
  .index: int  its number i
  .action  the action played: a recorded action (below)
  .before: np.ndarray | None  int8 (64, 64), indexed frame[y, x]: the frame before the action; None for step 0
  .after (= .last): np.ndarray  int8 (64, 64): the frame after the action, the one the tests compare
  .frames: np.ndarray  int8 (n, 64, 64): every frame the action returned, in order; n > 1: animated
  .level: int  the level it was played in
  .outcome: str  the game's status after the action: "NOT_FINISHED", "WIN" or "GAME_OVER" (State.status
      below says how your engine produces it)
  .levels_completed: int  levels completed after the action
  .win_levels: int  the game's number of levels
  .available_actions: list[int]  the action ids the game advertises
  .grid: GridGuess  the logical grid of its level, guessed from the level's frames loaded so far
  .pieces_before: Pieces | None  .before split into pieces (below); None for step 0
  .pieces_after: Pieces  .after split into pieces, on the grid of the level it shows (after a step that solves
      a level: the next level's first frame, and .pieces_after.code() writes sprites for its make_level)
  .changes: list[Change] | None  what the step changed, piece by piece, from .pieces_before to .pieces_after
      (None for step 0); print() shows one line per change
  The last four are computed from the frames when first read. A recorded step has no State or vars:
  frames, and pieces guessed from them; your engine's State comes from replay_step(i) or make_level(n).
The recorded action (a step's .action; it prints as Action(id=6, x=39, y=17) but is not the engine's Action)
  .id: int  0 RESET, 1 up, 2 down, 3 left, 4 right, 5 interact, 6 click, 7 undo
  .x, .y: int | None  a click's screen pixel (column, row); None for other actions
  .name: str  "RESET", or "ACTION1" to "ACTION7"
  It has no .cell: the grid cell under a click depends on your State's grid and view, so the harness
  computes it when it hands the click to step() (Action.cell below).
"""

# --- Ported from the base harness: the animation digest (StepView.animation, engine_re.animation) ----------
_STEP_VIEW_ANIMATION = """  .animation: Animation | None  an animated step's digest, None for one frame; print() shows it. .transient:
      the cells that changed and changed back (in no frame you can otherwise reach), with .transient_bbox (x0,
      y0, x1, y1), .transient_transitions ({"old>new": cells}) and .transient_frames (first, last); .timeline: one
      entry per frame that changed anything against the frame before it (frame 0 against .before): .frame,
      .changed, .bbox, and .cells ("old>new @ (x,y) ...") when few, else .transitions ({"old>new": count})
"""
_STEP_VIEW_ANIMATION_AFTER = "  .frames: np.ndarray  int8 (n, 64, 64): every frame the action returned, in order; n > 1: animated\n"


def _insert(text: str, anchor: str, addition: str, before: bool = False) -> str:
    """`text` with `addition` put right after (or before) `anchor`, which must occur exactly once."""
    if text.count(anchor) != 1:
        raise ValueError(f"prompt anchor not found exactly once: {anchor[:60]!r}")
    at = text.index(anchor) + (0 if before else len(anchor))
    return text[:at] + addition + text[at:]


_STEP_VIEW = _insert(_STEP_VIEW, _STEP_VIEW_ANIMATION_AFTER, _STEP_VIEW_ANIMATION)
# --- end of the animation digest's reference ------------------------------------------------------------------

_ENGINE = """
## engine.py: your two functions and the FIXED block's classes (the classes are preloaded in python too)
make_level(n: int) -> State  level n's first state (0-based): grid, every sprite, vars. The harness calls
    it once per level; every start of the level (entering it, every RESET) gets a deep copy of it, with
    .level = n and .status = "playing"
step(state: State, action: Action) -> None  apply one action to the state, in place; RESET never reaches
    it. A level ends when it sets state.status (below)
After every action the harness draws the state (Drawing) and records the outcome and levels completed;
the tests compare both with the recording.
replica: module  engine.py as it is now, in python: replica.make_level(n), replica.step(state, action), its
    constants and helpers. It is loaded again whenever the file changed since the last use, so it never goes
    stale after an edit. Never `import engine` (that copy would go stale; the kernel refuses it).
    replica.step fills in a click's action.cell when it is None, as the harness does.
Sprite(pixels, x=0, y=0, layer=0, name="", tags=(), visible=True, collidable=True, blocking="pixel",
    rotation=0, mirror_ud=False, mirror_lr=False, scale=1, screen=False)  == compares identity. It prints as
    the code that builds it, with the fields that differ from their defaults, e.g.
    Sprite([[9, 9]], x=3, y=1, tags=("player",)), so print(state.sprites) shows code
  .pixels: list[list[int]]  rows of colours 0-15, -1 transparent, -2 invisible but solid; before
      rotation, mirroring and scale
  .x, .y: int  the top-left pixel: a grid cell, or a screen pixel when screen=True
  .layer: int  higher layers are drawn on top and found first by sprite_at
  .name: str  what by_name finds
  .tags: tuple  labels that by_tag finds, e.g. ("wall",)
  .visible: bool  drawn
  .collidable: bool  takes part in collisions and sprite_at
  .blocking: str  "pixel": collides where pixels other than -1 overlap; "box": where the bounding boxes
      overlap; "none": never collides
  .rotation: int  0, 90, 180 or 270 degrees clockwise
  .mirror_ud, .mirror_lr: bool  flipped top-bottom, left-right, after the rotation
  .scale: int  1; 2 or more: each pixel becomes a scale x scale block; -1, -2, ...: shrunk 2x, 3x, ...,
      each block taking its most common colour
  .screen: bool  True: on screen pixels, unscaled (displays in the border); False: on the logical grid
  .width, .height: int  the size as drawn, after rotation and scale (read-only)
  .render() -> list[list[int]]  the pixels as drawn: rotated, then mirrored, then scaled
  .move(dx, dy) -> None  x += dx, y += dy
  .set_position(x, y) -> None  puts the top-left pixel at (x, y)
  .color_remap(old, new) -> None  colour old becomes new; old=None: every colour (pixels >= 0)
  .clone(**changes) -> Sprite  an independent copy with any fields changed, e.g. wall.clone(x=3, y=4)
  .collides_with(other, ignore_mode=False) -> bool  both collidable, neither blocking "none", both on the
      grid or both on the screen, and overlapping (by pixels other than -1 if either blocks by "pixel");
      ignore_mode=True skips the collidable and "none" checks
Action(id, x=0, y=0, cell=None)  what step() receives: the harness builds it for each action but RESET.
    It prints as the code that builds it, cell left out: Action(4), Action(6, x=12, y=40)
  .id: int  1 up, 2 down, 3 left, 4 right, 5 interact, 6 click, 7 undo
  .x, .y: int  a click's screen pixel (column, row), 0-63; 0 for other actions
  .cell: tuple[int, int] | None  a click's grid cell (gx, gy) under (x, y) in this state's grid and view;
      None outside the grid and for other actions
View(scale=None, rotation=0, mirror_ud=False, mirror_lr=False)  how the level is shown (Drawing 2 and 3)
  .scale: int | None  the grid's scale; None: the largest that fits, min(64 // w, 64 // h)
  .rotation: int  the finished frame turned clockwise by 0, 90, 180 or 270 degrees
  .mirror_ud, .mirror_lr: bool  then flipped top-bottom, then left-right
State(grid, sprites=[], vars={}, status="playing", level=0, view=View())  the current level; == compares
    identity
  .grid: tuple[int, int]  the logical grid (width, height), each 1-64
  .sprites: list[Sprite]  everything drawn: border, background, objects, displays
  .vars: dict  hidden values: budget, counters, modes, what is selected, ...
  .status: str  "playing" (outcome NOT_FINISHED) until step() sets "level_solved": one more level
      completed, and the next level starts from make_level, or the outcome is WIN after the last level;
      or "game_over": the outcome is GAME_OVER, and only RESET is accepted after it
  .level: int  the current level, set by the harness
  .view: View  how the level is shown
  .by_tag(tag) -> list[Sprite]  the sprites carrying tag, in list order
  .by_name(name) -> Sprite | None  the first sprite with this name
  .sprite_at(x, y, tag=None, include_uncollidable=False, screen=False) -> Sprite | None  the sprite at
      grid cell (x, y) (screen=True: a screen sprite at screen pixel (x, y)): highest layer first, then
      list order; only collidable sprites unless include_uncollidable; a "pixel"-blocking sprite only
      where its pixel is not -1; with tag, the first such sprite carrying it
  .sprites_at(x, y, screen=False) -> list[Sprite]  every sprite (on the grid, or on the screen with
      screen=True) whose box holds (x, y), whatever its other flags, in list order
  .collisions(sprite) -> list[Sprite]  the sprites that sprite collides with, in list order
  .try_move(sprite, dx, dy) -> list[Sprite]  moves sprite by (dx, dy) and, if it then collides with
      anything, back; returns the sprites it hit ([]: it moved)
  .add(sprite) -> Sprite  appends sprite to .sprites and returns it
  .remove(sprite) -> None  takes this sprite (the same object) out of .sprites
"""

_PIECES = """
## Pieces: a recorded frame split into sprites (a guess from the pixels, not the game's real sprites)
Pieces: list[Piece]  a frame split as one border sprite, one background sprite, one sprite per 4-connected
    region of one colour on the guessed grid, and screen sprites for what the grid cannot draw; drawn in list
    order they redraw the frame exactly. print() lists them, one line each
  .grid: GridGuess  the grid they sit on
  .code() -> str  Python, usable as edit_file lines, for a sprite list that draws the frame exactly: each shape
      a named constant, one Sprite per piece, in a function level_<n>_sprites (step_<i>_sprites for a frame
      that does not start a level). Pixel constants engine.py has are reused (as they are, turned, mirrored,
      scaled or recoloured) and its names are not defined again
Piece: a Sprite (above, and it prints as one) plus what the segmentation found
  .shape: str  its shape's name, from the pixels: the same pixels get the same name in every frame, e.g.
      "SHAPE_9_2x2_7cf8"; "border" and "background" for those two
  .colour: int  its most common colour
  .size: int  its pixels: grid cells, or screen pixels for a screen piece
  .transform: dict  how it is drawn from its shape: rotation, mirror_ud, mirror_lr, scale, recolour ({old:
      new}); {} when it is the shape as it is
  .children: list[int]  indices of the pieces it encloses (each under the innermost one only)
  .role: str  "border" (the 64x64 screen sprite), "background" (the grid-sized one) or "object"
Change: one piece-level change of a step; it prints as one line
  .kind: str  "moved" (the same pixels elsewhere, nearest first), "recoloured" (same place and shape, other
      colours), "reshaped" (overlapping, other pixels or size), "appeared" or "disappeared"
  .before, .after: Piece | None  the piece in each frame (None: it appeared, disappeared)
  .dx, .dy: int  how far it moved (moved; reshaped: its top-left corner)
  .colours: dict  {old: new} colours (recoloured; reshaped in another colour)
  .note: str  reshaped: how, e.g. "1x31 -> 1x26, lost 5 px at the top"
GridGuess: the logical grid guessed for a level
  .width, .height: int  the grid size
  .grid: tuple[int, int]  (width, height), for State(grid=...)
  .scale: int  screen pixels per cell
  .default_scale: bool  whether scale is min(64 // width, 64 // height); if not, the State needs
      View(scale=scale)
  .x_offset, .y_offset: int  the screen pixel of cell (0, 0)
  .border: int  the colour around the grid
  .frames: int, .note: str  how many frames it was guessed from, and how
"""

_BUILTINS = """
## Built-in functions
read_file(path="engine.py", offset=None, limit=None) -> None  prints the file, every line as
    LINE#HASH:content; those anchors are how edit_file addresses lines. offset: the first line (1-based);
    limit: the number of lines. Long output is cut; it says which offset to continue from. The FIXED
    block is folded unless offset asks for its lines.
edit_file(path="engine.py", edits=[...]) -> None  changes the file. engine.py changes only through
    edit_file() and undo_edit(); writing it any other way, such as open('engine.py', 'w'), is blocked.
    Each edit is one of:
      {"op": "replace_def", "name": "step", "lines": [...]}  replace the whole top-level def, class or
          NAME = ... called name (a method as "Game.step"); a name not defined yet is added at the end
      {"op": "replace_text", "oldText": "...", "newText": "..."}  replace one unique text: an exact
          match, else whole lines matching ignoring whitespace (newText is used as given)
      {"op": "replace", "pos": "12#MQV", "end": "15#VRS", "lines": [...]}  replace line pos (or pos..end)
      {"op": "append", "pos": "12#MQV", "lines": [...]}   insert after pos (no pos: at the end)
      {"op": "prepend", "pos": "12#MQV", "lines": [...]}  insert before pos (no pos: at the start)
    pos and end are LINE#HASH anchors from the latest read_file() or edit_file() output. lines is the
    new content (a list of lines, or one string), with its indentation; [] deletes. To put data into
    engine.py, generate the text with repr() or pprint.pformat(), e.g. lines=f"RINGS = {rings!r}" or
    lines=pieces.code() (a frame's Pieces); never build source text by hand, string by string. All
    edits of one call are checked against the same version of the file: those that are valid are applied
    (edits on adjacent lines are merged, in order; two that change the same line are both refused),
    and each one that fails (a stale anchor, no match, the FIXED block) is reported with the lines as
    they are now. Prints what changed, a syntax check, and fresh anchors around each change (the whole
    region up to 60 lines).
undo_edit(n=1, to=None) -> None  puts engine.py back as it was n changes ago; to="best": the version that
    passed the most steps before its first failure so far; to=k: version k. Every change (each edit_file()
    call, and undo_edit itself) is kept as a numbered version, so nothing is lost: undo_edit() right after
    an undo_edit() brings the undone change back. Prints the recent versions (what changed, and the test
    result of each version that was tested), what it restored, and a reminder to read_file() again.
render_state(state: State) -> np.ndarray  int8 (64, 64): the frame the tests draw for a State (Drawing),
    with the same code. Try things on the drawing: change a sprite or state.view and draw again.
__SHOW_FRAMES__
replay_step(i, state=None, action=None, *, level=None) -> tuple[State | None, State]  loads engine.py
    fresh, gives your State just before recorded step i (replaying steps 0 to i-1, or a copy of `state`),
    applies step i's recorded action (or `action`: an id, (6, x, y) for a click, or an Action), and
    prints what your step() printed, what changed in your State (sprites by index #k and name, vars,
    status) and, with the recorded action, the regions where your frame differs from the recorded frame
    after step i, with your sprites in each. Returns copies (before, after) of your State; before is None
    for step 0. level=L starts at level L's start, as run_tests(level=L) does; replay_step(e, level=L), e
    being the step that entered level L, compares your make_level(L) with the level's recorded first
    frame and returns (None, that State).
"""

_SHOW_FRAMES = {
    True: """show_frames(*frames, titles=None, boxes=None) -> None  shows 64x64 frames (e.g. a recorded step's
    .after, render_state(state)) or States (drawn first) as images, side by side, enlarged, in a message
    right after this call's output (at most 4 per call). boxes: [(x0, y0, x1, y1), ...] in screen pixels,
    outlined and numbered on every image. Older images are later replaced by a placeholder.""",
    False: """show_frames(*frames, titles=None, boxes=None) -> None  prints 64x64 frames (e.g. a recorded step's
    .after, render_state(state)) or States (drawn first) as hex digits, side by side, at most 4 per call.
    boxes: [(x0, y0, x1, y1), ...] in screen pixels: it prints the boxed pixels of every frame; without
    boxes, every frame at half resolution.""",
}

_SUMMARIZE = {
    "single": """summarize_levels() -> None  prints one row per level: its first frame (recording[k].after), its
    grid, the steps played in it and their actions, the animated steps, RESETs and game overs, and the
    step that solved it.
""",
    "history": """summarize_levels() -> None  prints one row per level reached so far: its first frame
    (recording[k].after), its grid, the steps played in it and their actions, the animated steps, RESETs
    and game overs, and how it ended.
""",
    "step": "",
}
_SUMMARIZE["play"] = _SUMMARIZE["history"]

_PLAY_BUILTINS = """state_now() -> State  your replica's State now: engine.py loaded fresh and every step played so far replayed
    through it with the harness rules; prints the level, status and vars; returns a copy (.level is the
    level being played). It is the game as your replica models it; the game's own frame is recording[-1].after.
click_cell(state, x, y) -> tuple | None  the grid cell under screen pixel (x, y) on this state: the action.cell
    the harness gives step() for a click there (replica.step fills it in when an Action's cell is None).
To play moves on your replica, call it directly on copies of a State: s = copy.deepcopy(state_now());
    replica.step(s, Action(1)) plays UP (ids: 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT, 5 SPACE, 6 click, 7 UNDO, 0 RESET).
    The Actions you step your replica with are what commit_moves sends, as python prints them: print(moves)
    shows [Action(4), Action(6, x=12, y=40)], and commit_moves takes ["Action(4)", "Action(6, x=12, y=40)"] or
    that printed list as one string. step() never gets a RESET (Action(0)): a RESET is a fresh
    replica.make_level(n), and after s.status == "level_solved" the next level is replica.make_level(n + 1).
    render_state(s) draws a State and show_frames(...) shows it. Searching over moves (a BFS calling
    replica.step on copies) is a short function in python: write it when the level needs it.
"""

# replay_step's level argument, as the modes whose run_tests has `level` describe it, and as the play mode does.
_REPLAY_LEVEL = "level=L starts at level L's start, as run_tests(level=L) does;"
_REPLAY_LEVEL_PLAY = "level=L starts at make_level(L) and replays only that level's steps;"

_NAMES = {"single": PRELOADED, "history": PRELOADED_HISTORY, "step": PRELOADED_STEP, "play": PRELOADED_PLAY}


def _variant(mode: str, history: bool) -> str:
    if mode == "play":
        return "play"
    return "single" if mode == "single" else "history" if history else "step"


def objects_reference(mode: str = "single", history: bool = True, images: bool = True) -> str:
    """The "# Objects" section of the system prompt: the recorded steps python holds in this mode, then
    the parts every mode shares (StepView and the recorded action, engine.py's functions and classes,
    a frame's pieces, the built-in functions)."""
    variant = _variant(mode, history)
    builtins = _BUILTINS.replace("__SHOW_FRAMES__", _SHOW_FRAMES[images])
    if variant == "play":  # run_tests has no level there
        builtins = builtins.replace(_REPLAY_LEVEL, _REPLAY_LEVEL_PLAY)
    text = (
        _OBJECTS_HEAD.replace("__NAMES__", ", ".join(_NAMES[variant]))
        + _RECORDED[variant] + _STEP_VIEW + _ENGINE + _PIECES
        + builtins + _SUMMARIZE[variant]
        + (_PLAY_BUILTINS if variant == "play" else "")
    )
    return text.replace("your engine", "your replica") if variant == "play" else text  # the play mode's word for engine.py


_PYTHON = """Run Python in a kernel that is persistent for the whole run: every variable, function and import you
define stays until the run ends, so define helpers and data once and reuse them instead of retyping them.
Prints what your code prints plus the value of the last expression. Long output is cut, so print compact
summaries or small crops, never whole frames. Preloaded: np (numpy), the engine classes Sprite, Action,
View, State, and
  __NAMES__
The system prompt's # Objects section documents each of them (fields, signatures, return values). These
names are reserved: code that defines or assigns any of them is rejected before it runs. engine.py
changes only through edit_file() and undo_edit() called here."""

_RUN_TESTS = """Run the contract tests, then replay the recording in order: from step 0, or from the start of level L
(make_level(L)) when level=L is given. Stops after `failures` failing steps (1 to 10, default 1).
Reports how many steps pass before the first failure, the first failure in full (images, regions,
your sprites there, what step() printed, the replay_step command), and one line per further failure."""

_RUN_TESTS_NO_IMAGES = _RUN_TESTS.replace("(images, regions,\nyour sprites there,", "(regions with their\npixels, your sprites there,")


# --- The stepwise harness (v6) -------------------------------------------------------------------

_SYSTEM_STEP = """# Goal
You are building engine.py, a Python model of a game, from a recording of someone playing it: every
action they took and every frame the game returned. The harness replays the recording through
engine.py step by step. When a step does not give the recorded result, it stops there and asks you to
fix that step. Once the steps up to it pass and you submit engine.py with commit_engine, it replays
on and tells you, in this conversation, how many more steps passed and which step breaks next; you see
the recording only up to that step. Passing tests alone do not move on: until you commit you can keep
refining. This goes on until the whole recording passes.
Make the tests pass with the most parsimonious model (Occam's razor): the fewest rules and assumptions
that account for every step observed so far.

""" + _SYSTEM[_SYSTEM.index("# Setup") : _SYSTEM.index("# Tests")] + """# Tests (run_tests; commit_engine runs them too)
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

__OBJECTS__
# How to work
1. Look at what the step did: compare step_to_fix.before with step_to_fix.after, and read the report's
   regions and what the click hit.
2. Find the simplest rule that explains this step and agrees with what engine.py already does for the
   earlier steps: the fewest rules and assumptions that account for every step so far, and no rule the
   steps give no evidence for. Per-level constants in the level data (a rate, a budget, a size) are fine
   when the steps give no evidence of a formula: do not hunt for one. When a step contradicts a rule you
   wrote, replace it with the simplest rule that explains all the steps so far.
   A long line or a strip of small blocks flush against a screen edge, outside the playing grid, that
   shrinks or changes on every action is almost always a step or time budget (a HUD bar), not a game
   mechanic: model it as a per-level budget drawn proportionally and rounded to the nearest pixel,
   e.g. round(length * moves / budget), with the budget a per-level constant that fits the steps seen
   so far. Several budgets may fit; any of them is right at this step, and a later step will narrow
   it. Being one pixel off on such a bar is tolerated by the tests (a warning, not a failure), so do
   not spend turns on it.
3. Change engine.py with edit_file(), run the tests and fix what they report (an earlier step that now
   breaks counts too). When they pass, call commit_engine(message): what you changed and why. The next
   step is shown only after a commit.
4. When the step starts a new level (the frame after it shows the next level), make_level must draw
   that level: step_to_fix.pieces_after.code() gives code for its first frame. Reuse the sprite kinds
   engine.py already has where they fit. Only draw the new level's first frame; model its mechanics
   when one of its steps fails.
5. Do not think about, model or write code for what has not been observed.
6. Keep engine.py's comments up to date with the rules you found: older parts of this conversation are
   shortened as it grows, and engine.py is what stays. Define helpers and data once in python: the
   kernel keeps them for the whole run.
Never hard-code recorded frames or anything keyed to the step number. Print whatever helps you debug
inside step(); the test report and replay_step show it.
"""

_RUN_TESTS_STEP = """Run the contract tests, then replay the recording in order from step 0 to the step you are fixing.
Stops after `failures` failing steps (1 to 10, default 1). Reports how many steps pass before the
first failure, the first failure in full (images, regions, what a click hit, your sprites there, what
step() printed, the replay_step command), and one line per further failure."""

_COMMIT_MESSAGE = (
    "What you changed and the key analysis behind each rule: what in the recording shows it (which steps or "
    "frames, what changed), and which guesses remain."
)

_COMMIT_STEP = """Submit engine.py as your fix of the step. Runs the tests first (steps 0 to the step you are
fixing): if any fails you get the report and nothing moves on; when they all pass, the commit is kept
and the harness replays on and tells you the next step that breaks (or that the whole recording
passes). Passing tests alone (run_tests, or the automatic test after an edit) never move on, so you can
keep refining first. message: """ + _COMMIT_MESSAGE

_COMMIT = """Submit engine.py as your engine. Runs the tests first: if anything fails you get the report and the
session goes on; it ends when every test passes. message: """ + _COMMIT_MESSAGE


# The play mode's third Tests item: the frames compared are the real game's.
_REPORT_IMAGES_PLAY = {
    True: "your frame and the game's frame as images\n  with the differing regions boxed and numbered",
    False: "the regions where your frame differs from the\n  game's, numbered, with their pixels",
}


def system_prompt(
    match: str = "final", interface: str = "simple", images: bool = True, mode: str = "single", history: bool = True
) -> str:
    """The system prompt. Only the make_level/step interface scored on final frames is offered.
    mode "single": one session over the recording; "step": fix one breaking step (v6), seeing the
    recording up to it (history) or only that step. Every mode has the # Objects reference."""
    if interface != "simple":
        raise ValueError("the agent offers only the simple interface (make_level/step)")
    if match != "final":
        raise ValueError("the simple interface produces one frame per action, so it is scored with match='final'")
    text = _SYSTEM_PLAY if mode == "play" else {"single": _SYSTEM, "step": _SYSTEM_STEP}[mode]  # _SYSTEM_PLAY: defined below
    shown = (_REPORT_IMAGES_PLAY if mode == "play" else _REPORT_IMAGES)[images]
    return text.replace("__OBJECTS__", objects_reference(mode, history, images)).replace("__REPORT_IMAGES__", shown)


SYSTEM_PROMPT = system_prompt()


def _python_description(mode: str = "single", history: bool = True) -> str:
    return _PYTHON.replace("__NAMES__", ", ".join(_NAMES[_variant(mode, history)]))


def tools(images: bool = True, mode: str = "single", history: bool = True) -> list[dict]:
    """The tool schemas: python, run_tests and commit_engine (mode "step": the stepwise harness's texts,
    and run_tests without `level`; history: whether python shows the recording so far)."""
    schemas = _tools(images, mode, history)
    if mode == "step":
        run_tests = schemas[1]["function"]
        run_tests["description"] = _RUN_TESTS_STEP if images else _RUN_TESTS_STEP.replace("(images, regions,", "(regions with their pixels,")
        del run_tests["parameters"]["properties"]["level"]
        schemas[2]["function"]["description"] = _COMMIT_STEP
    elif mode == "play":
        run_tests = schemas[1]["function"]
        run_tests["description"] = _RUN_TESTS_PLAY if images else _RUN_TESTS_PLAY.replace("(images, regions,", "(regions with their pixels,")
        del run_tests["parameters"]["properties"]["level"]
        schemas[2]["function"]["description"] = _COMMIT_PLAY
        schemas.append(copy.deepcopy(_COMMIT_MOVES_TOOL))
    return schemas


def _tools(images: bool, mode: str, history: bool) -> list[dict]:
    return copy.deepcopy(
        [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _python_description(mode, history),
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
                    "name": "commit_engine",
                    "description": _COMMIT,
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string", "description": _COMMIT_MESSAGE}},
                        "required": ["message"],
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
        f"{len(steps)} steps over {played} level(s) (recording[0] is the RESET that starts the game); it ends {steps[-1].state} "
        f"with {steps[-1].levels_completed} of {steps[0].win_levels} levels completed. The actions this game accepts: "
        f"{actions}. summarize_levels() in python lists each level: its first frame, the steps played in it and how it "
        "ended."
    )


def _first_task(first_fail: int | None) -> str:
    if first_fail is None:
        return "Every test passes already: call commit_engine(message)."
    if first_fail == 0:
        what = "step 0 pass, level 0's first frame (make_level(0) does not draw it exactly yet), then step 1"
    elif first_fail == 1:
        what = "step 1 pass, the first action of level 0"
    else:
        what = f"step {first_fail} pass, the first step that fails"
    return (
        f"Your first task: make {what}. Work on that step only: read the code it runs, look at the step with "
        f"replay_step({first_fail}), edit, run the tests. Then go on to the next failing step, one step at a time, in "
        "the order they were played, and add a level when the tests reach it, from recording[e].pieces_after.code() "
        "(e: the step that enters it)."
    )


def first_user_message(game: str, trace: Trace, engine_read: str, opening: dict | None = None) -> str:
    """The opening message: the recording in one sentence, what the harness did before the first turn
    (recording[0].pieces_after.code() put into make_level, then run_tests: `opening` holds "sprites", the
    summary printed with that code, "report", the test report, and "first_fail"), engine.py as
    read_file() shows it, and the first task. Without `opening` the model is asked to do that first
    round itself."""
    head = f"Game: {game}. Write engine.py for it.\n\nThe recording: {recording_summary(trace)}\n"
    shown = f"engine.py now, as read_file() shows it (the FIXED block folded):\n\n{engine_read}"
    if opening is None:
        return f"""{head}
{shown}

Your first task: put recording[0].pieces_after.code() into make_level with edit_file(), so that level 0's first frame
is drawn, and run the tests. Then make the first failing step pass, then the next one, one step at a time, in the order
they were played."""
    sprites = "\n".join("   " + line if line else "" for line in opening["sprites"].strip().splitlines())
    return f"""{head}
Before your first turn the harness did the first round:
1. recording[0].pieces_after.code() wrote sprites that draw level 0's first frame, and make_level now returns
   them (for every level, for now). About that code:
{sprites}
2. run_tests() then reported:

{opening["report"].strip()}

{shown}

{_first_task(opening.get("first_fail"))}"""


def _action_text(action) -> str:
    if action.id == 6:
        return f"a click at screen pixel ({action.x}, {action.y})"
    return "RESET" if action.id == 0 else f"ACTION{action.id} ({_ACTION_WORDS.get(action.id, '?')})"


def step_objects(trace: Trace, k: int) -> str:
    """What recorded step k changed, piece by piece and summarised (about 15 lines at most), from steps
    0..k only: the frames' segmentation (engine_re.segment), as step_to_fix.changes has it."""
    try:
        return segment.Segmenter(trace, k).report(k)
    except Exception as exc:  # noqa: BLE001  (a frame the segmentation cannot read must not stop the run)
        return f"What the recorded step changed (objects): not available ({type(exc).__name__}: {exc})."


# The engine.py listing in a stepwise message (the first one and every next-step one): this header, a blank
# line, the listing (read_file() with anchors, the FIXED block folded), a blank line, the closing line.
# Compaction replaces every listing but the latest by ENGINE_ELIDED (elide_engine_listing).
ENGINE_HEADER = "engine.py now (as read_file shows it):"
ENGINE_ELIDED = "[engine.py as it was then: elided; read_file() shows the current file]"


def engine_block(engine_read: str) -> str:
    return f"{ENGINE_HEADER}\n\n{engine_read}"


ENGINE_CLOSING = "\n\nFix step "  # the paragraph after the listing in both messages


def elide_engine_listing(text: str) -> str:
    """`text` with its engine.py listing (ENGINE_HEADER up to the closing paragraph: "Fix step ..." in a fix
    message, "Work out the next moves ..." in a PLAN message (ENGINE_CLOSINGS), or to the end) replaced by
    ENGINE_ELIDED; unchanged when it has none."""
    start = text.find(ENGINE_HEADER)
    if start < 0:
        return text
    end = max(text.rfind(closing) for closing in ENGINE_CLOSINGS)
    if end <= start:
        return text[:start] + ENGINE_ELIDED
    return text[:start] + ENGINE_ELIDED + text[end:]


def episode_message(game: str, trace: Trace, k: int, report: str, engine_read: str, history: bool = True) -> str:
    """The first message of a stepwise conversation: fix the breaking step k (steps 0..k-1 pass).
    `trace` holds at least steps 0..k; `report` is the test report of the replay up to step k. It shows
    what the recorded step changed, piece by piece (step_objects)."""
    s = trace.steps[k]
    level = trace.steps[k - 1].levels_completed if k > 0 else 0
    notes = []
    if s.state == "WIN":
        notes.append(f"This step solves level {level}, the last one: the game ends with WIN.")
    elif s.levels_completed > level:
        notes.append(
            f"This step solves level {level}: the frame after it is level {s.levels_completed}'s first frame, which "
            f"make_level({s.levels_completed}) must draw (step_to_fix.pieces_after.code() gives code for it; reuse the "
            "sprite kinds engine.py already has where they fit)."
        )
    if s.state == "GAME_OVER":
        notes.append("After this step the game is over (GAME_OVER).")
    if k == 0:
        passed = "Step 0, level 0's first frame, does not pass yet."
    else:
        before = "Step 0 of the recording passes" if k == 1 else f"Steps 0-{k - 1} of the recording pass"
        passed = f"{before} with your engine.py; step {k} is the first that does not."
    shown = (f"In python, `recording` holds the recording so far, steps 0-{k} (later steps are not loaded), and "
             f"`step_to_fix` is step {k} (recording[-1]): step_to_fix.before, .action, .after, .frames." if history else
             "In python, `step_to_fix` holds this step: step_to_fix.before, .action, .after, .frames.")
    return f"""Fix the breaking test: step {k}.

Game: {game}. {passed}
Step {k}: {_action_text(s.action)}, played in level {level}. The game returned {s.n_frames} frame(s) for it; the tests compare
the last.{(" " + " ".join(notes)) if notes else ""}{animation_note(trace, k, report)}
{shown}

{step_objects(trace, k)}

The test report:

{report.strip()}

{engine_block(engine_read)}

Fix step {k}: find the simplest rule that explains it and keeps the earlier steps passing, change engine.py with
edit_file(), and run the tests. When they pass, call commit_engine(message) to submit the fix (you may refine it first);
the next steps are shown only after a commit."""


KERNEL_KEEPS = "Your python kernel keeps: "
KERNEL_KEEPS_NOTHING = "Your python kernel keeps nothing you defined yet."


def kernel_names_text(names: list[str], more: int = 0) -> str:
    """The names the model defined in the kernel (kernel.user_names: "name: summary" each), in one line."""
    if not names:
        return KERNEL_KEEPS_NOTHING
    return KERNEL_KEEPS + ", ".join(names) + (f", ... and {more} more" if more else "")


def advance_message(
    trace: Trace, fixed: int, k: int, report: str, history: bool = True, engine_read: str = "", kernel_names: str = "",
) -> str:
    """The user message when a commit of steps 0..fixed was accepted and the harness replayed on to step k, the next that
    fails (`trace` is the whole recording; the model now sees it up to k, and so does step_objects). `engine_read` is
    engine.py as read_file() shows it, listed under ENGINE_HEADER as the first message lists it; `kernel_names` is
    kernel_names_text(), what the model's kernel keeps."""
    s = trace.steps[k]
    level = trace.steps[k - 1].levels_completed if k > 0 else 0
    if k == fixed + 1:
        passed = f"Commit accepted: steps 0-{fixed} pass. The next step, {k}, fails."
    else:
        between = f"step {fixed + 1}" if k == fixed + 2 else f"steps {fixed + 1}-{k - 1}"
        passed = (f"Commit accepted: steps 0-{fixed} pass. The harness replayed on: {between} ({k - fixed - 1} more step"
                  f"{'s' if k - fixed - 1 > 1 else ''}) passed without error. Step {k} is the next that fails.")
    notes = []
    if s.state == "WIN":
        notes.append(f"It solves level {level}, the last one: the game ends with WIN.")
    elif s.levels_completed > level:
        notes.append(
            f"It solves level {level}: the frame after it is level {s.levels_completed}'s first frame, which "
            f"make_level({s.levels_completed}) must draw (step_to_fix.pieces_after.code() gives code for it; reuse the "
            "sprite kinds engine.py already has where they fit)."
        )
    if s.state == "GAME_OVER":
        notes.append("After it the game is over (GAME_OVER).")
    shown = (f"`recording` now holds the recording up to step {k}, and `step_to_fix` is step {k}." if history
             else f"`step_to_fix` is now step {k}.")
    return f"""{passed}
Step {k}: {_action_text(s.action)}, played in level {level}; {s.n_frames} frame(s), the tests compare the last.{(" " + " ".join(notes)) if notes else ""}{animation_note(trace, k, report)}
{shown}

{step_objects(trace, k)}

The test report:

{report.strip()}
{(chr(10) + kernel_names) if kernel_names else ""}{(chr(10) + engine_block(engine_read) + chr(10)) if engine_read else ""}
Fix step {k}, keeping steps 0-{k - 1} passing; commit_engine(message) when the tests pass."""


def resume_user_message(game: str, trace: Trace, turns: int, test_report: str, engine_read: str, notes: str = "") -> str:
    notes_part = f"\nWhat that session left behind:\n\n{notes}\n" if notes else ""
    return f"""Game: {game}. Write engine.py for it.

The recording: {recording_summary(trace)}

This continues an earlier session on this game ({turns} turns) that was interrupted. Its conversation is gone and the
python kernel was restarted (its variables are gone), but engine.py and its versions (undo_edit) are kept.
{notes_part}
The current test report of engine.py:

{test_report}

engine.py now, as read_file() shows it:

{engine_read}"""


# --- The play-and-model agent (v10, engine_re.play_agent) -------------------------------------------

_SYSTEM_PLAY = """# Goal
You are playing a game you have never seen, and the way you play it is to build engine.py, your replica of
the game in Python (in python, the built-in `replica`), as you go. The harness alternates two rounds in this
one conversation:
- Plan: your replica reproduces every step played so far. Look at the game's current frame, use your
  replica to work out what to do (state_now(), then replica.step on copies of it), and send the moves with
  commit_moves(actions, note): the Actions you stepped your replica with, as python prints them.
  The harness predicts each move with your replica, sends it to the game, and compares. As long as
  the game does what your replica predicted it sends the next move; at the first difference it stops the
  batch and opens a fit round on that step.
- Fit: "step k did not go as predicted". You see the game's result, fix engine.py so that it reproduces
  every step so far (the tests replay them all), and submit it with commit_engine(message). Then you plan
  again. commit_moves refuses to send anything while the tests fail, so there is no playing on a wrong
  replica: fix first.
The score is the official one: each level solved scores (human baseline / your actions in that level)^2,
a level not solved scores 0, and later levels weigh more (level n counts n + 1 times). Actions are the
cost; turns and tokens are not. A RESET counts as an action. Win the game in as few actions as you can;
a move you cannot predict is still worth sending when it teaches you a rule, as long as it is cheap.
Make the tests pass with the most parsimonious model (Occam's razor): the fewest rules and assumptions
that account for every step observed so far.

""" + _SYSTEM[_SYSTEM.index("# Setup") : _SYSTEM.index("# Tests")].replace(
    "(given in the first message)", "(every plan message lists them)").replace(
    "the workspace, the recording and the Python installation", "the workspace, the steps played so far and the Python installation",
).replace(
    "- The harness runs your engine. It calls make_level(n) once per level and hands step() a fresh copy of\n"
    "  that state whenever the level starts (on entering it and after every RESET); RESET never reaches step().",
    "- engine.py is your replica of the game, and the harness runs it. It calls make_level(n) once per level\n"
    "  and hands step() a fresh copy of that state whenever the level starts (on entering it and after every\n"
    "  RESET); RESET never reaches step().") + """# Tests (run_tests; commit_engine and commit_moves run them too)
- Contract: the fixed block is unchanged; make_level(n) returns a valid State for every level reached so
  far; step() accepts every advertised action; the same actions always give the same result.
- Acceptance: every step played so far is replayed in order. After every action, your final frame (every
  pixel) and the game status (NOT_FINISHED / WIN / GAME_OVER, levels completed) must equal what the game
  did. When the game animated an action, only its last frame is compared.
- The report stops at the first failing step (or after up to 10, if you ask). For it you get:
  __REPORT_IMAGES__;
  for a click, the grid cell it lands on and your sprites there; for each region, the colours and
  your sprites there; what your step() printed; and the python command that reproduces the step.
- commit_moves checks each move the same way, live: your replica's final frame and status for the move
  against the game's.

__OBJECTS__
# How to work
Plan rounds:
1. Look at the current frame (the message shows it; recording[-1].after holds it) and at your replica's
   state (state_now()). What is the goal of the level? What have your moves changed so far?
2. Try moves on your replica in python: replica.step(s, Action(...)) on copies of state_now() shows what
   your model predicts (render_state(s) draws the result). Search over it in python when the level needs
   it: a BFS over moves calling replica.step on copies is a short function, and your replica is the point
   of having one.
3. Send a batch with commit_moves(actions, note): the moves you are confident about, the shortest way you
   see to the goal. The actions are the Actions you played on your replica, as python prints them: when
   print(moves) shows [Action(4), Action(6, x=12, y=40)], send ["Action(4)", "Action(6, x=12, y=40)"].
   When your replica has never seen a kind of move (a key it has no rule for, a click on
   something it does not model), send that move in a short batch of 1-3 to learn its effect, instead of a
   long plan built on a guess; do not study the frame for many turns first: the game's answer to a move
   shows its rule faster than analysis. The note says what the batch is meant to do.
4. After a game over the harness restarts the level with a RESET (it counts as an action). After a solved
   level the batch stops: the next level is new, plan it afresh.
Fit rounds:
1. Look at what the step did: compare step_to_fix.before with step_to_fix.after, and read the report's
   regions and what the click hit.
2. Find the simplest rule that explains this step and agrees with what engine.py already does for the
   earlier steps: the fewest rules and assumptions that account for every step so far, and no rule the
   steps give no evidence for. Per-level constants in the level data (a rate, a budget, a size) are fine
   when the steps give no evidence of a formula: do not hunt for one. When a step contradicts a rule you
   wrote, replace it with the simplest rule that explains all the steps so far.
   A long line or a strip of small blocks flush against a screen edge, outside the playing grid, that
   shrinks or changes on every action is almost always a step or time budget (a HUD bar), not a game
   mechanic: model it as a per-level budget drawn proportionally and rounded to the nearest pixel,
   e.g. round(length * moves / budget), with the budget a per-level constant that fits the steps seen
   so far. Being one pixel off on such a bar is tolerated by the tests (a warning, not a failure), so do
   not spend turns on it. Such a budget is also the game's way of losing: watch it when you plan.
3. Change engine.py with edit_file(), run the tests and fix what they report (an earlier step that now
   breaks counts too). When they pass, call commit_engine(message): what you changed and why. Then plan.
4. When the step starts a new level (the frame after it shows the next level), make_level must draw
   that level: step_to_fix.pieces_after.code() gives code for its first frame. Reuse the sprite kinds
   engine.py already has where they fit. Only draw the new level's first frame; model its mechanics
   when one of its steps fails.
5. Do not model or write code for what has not been observed. Keep engine.py's comments up to date with
   the rules you found: older parts of this conversation are shortened as it grows, and engine.py is
   what stays. Define helpers and data once in python: the kernel keeps them for the whole run.
Never hard-code frames or anything keyed to the step number. Print whatever helps you debug inside
step(); the test report and replay_step show it.
"""

# --- Ported from the base harness's prompt (inference/agent/prompts.py, inference/agent/tool_agent.py) -----------
# Its sentences, verbatim where they apply, adapted to the replica, commit_moves, numpy frames of colours 0-15 and
# clicks Action(6, x, y) (PLAY_DESIGN.md, "Ported from the base harness"). Each block goes in at an anchor of
# _SYSTEM_PLAY (_insert raises when an anchor is missing, so a change to the text around it is caught at import).

# GAME_OVERVIEW_ADDENDUM's colour legend, and ACTION_INFO_ADDENDUM, UNDO_INFO_ADDENDUM, RESET_INFO_ADDENDUM: in # Setup,
# after the action list.
_PLAY_COLOUR_LEGEND = textwrap.fill(
    "- Colour legend: " + ", ".join(f"{c} {name}" for c, name in sorted(COLOR_NAMES.items())) + ".",
    width=110, subsequent_indent="  ",
) + "\n"
_PLAY_ACTION_MEANINGS = """- Action meanings (use only the actions the game advertises; every plan message lists them):
  - UP, DOWN, LEFT and RIGHT (Action(1) to Action(4)) are directional controls; what they affect depends on the
    game.
  - When available, SPACE (Action(5)) performs a game-specific action, such as interacting, selecting, rotating,
    attaching/detaching, or executing. Test its effect rather than assuming what it does.
  - When available, a click Action(6, x=x, y=y) clicks a board location. Pass integer x and y from 0 to 63.
    Coordinates are zero-based from the top-left: y (the row) increases downward and x (the column) increases
    rightward.
  - When available, UNDO (Action(7)) reverses a previous action, usually the last turn. Check what it restores.
    It cannot recover a failed attempt after game over.
  - RESET (Action(0)) usually restores the current level to its starting state, including the
    remaining-action/time bar, while keeping completed levels. Use it to recover from an unrecoverable position or
    start a different approach. RESET itself counts as one action, and actions already spent still count toward
    your score. Send it with commit_moves like any move; after a game over the harness sends it for you.
"""
_PLAY_SETUP_AFTER = "  advertises a fixed subset of them (every plan message lists them).\n"

# ANIMATION_ADDENDUM and ANIMATION_ADDENDUM_TIMELINE: in # Tests, after the sentence on animated actions.
_PLAY_ANIMATION = """  One action can return a short animation: a step's .frames holds every frame and .after, its final frame, is
  the board it settled on. Transient cells changed and then changed BACK during the animation, so they appear in
  no frame you can otherwise reach - not in .before, not in .after. For an animated step the test report and the
  messages print step.animation: the transient cells (how many, where, their colour changes old>new and the
  frames they differ in) and a diff timeline of the frames, which shows which cells changed at each frame.
  Whatever the action did may be visible only there: read it before concluding that a move did nothing.
"""
_PLAY_ANIMATION_AFTER = "  did. When the game animated an action, only its last frame is compared.\n"

# STEP_VERIFICATION_ADDENDUM (the game is solvable), LEVEL_TRANSFER_SYSTEM_GUIDANCE, VISUAL_GAME_ADDENDUM (a scene, no
# player assumed, no absolute-coordinate goals), PREFER_TOOL_CALLS_LINE, and the memory sections Goal model, Open
# questions and Plan (tool_agent._MEMORY_SECTION_MEANINGS) kept in notes.md: plan rules 5-9, before the fit rules.
NOTES_FILE = "notes.md"
_PLAY_PLAN_RULES = """5. If your search finds no solution under your current model of the game, remember that the game is solvable.
   Reconsider your mechanics, goal, search implementation, or search limits, including interactions with new
   elements. Take a targeted action to test an uncertain rule or overlooked interaction, then update your model
   from the result. A plan far above the human baseline the plan message shows, or no plan at all, means your
   replica is missing a rule, not that the level is hard.
6. Levels usually build on mechanics learned in earlier levels, especially the most recent one. The rules in
   engine.py carry them forward: they are your starting hypothesis on a new level, while you re-check anything
   contradicted by new evidence. New levels often introduce additional mechanics, sometimes through unfamiliar
   board elements. These additions are often important for solving the level. The goal may remain the same but
   require new mechanics to reach it, or the goal itself may change.
7. Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.
   Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do
   not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board
   configuration. Use coordinates only to target actions or describe local evidence. Do not frame the objective
   as reaching a specific absolute row or column.
8. Reading and computing cost nothing; only commit_moves spends the level budget. When you are unsure, prefer
   another python call over more reasoning: the code answers what the reasoning would only guess at, and running
   it is faster than thinking your way to the same answer.
9. engine.py holds the rules (what the objects are, what each action does). Keep what is not code in notes.md, in
   the workspace, under three headings: Goal model: what winning requires. Open questions: unresolved hypotheses.
   Plan: intended next steps. Change it with edit_file(path="notes.md", edits=[...]) (read_file("notes.md") gives
   its anchors); every plan message shows it. Older parts of this conversation will eventually be dropped, so
   anything you leave out of engine.py and notes.md is gone.
"""
_PLAY_PLAN_RULES_BEFORE = "Fit rounds:\n1. Look at what the step did"

_SYSTEM_PLAY = _insert(_SYSTEM_PLAY, _PLAY_SETUP_AFTER, _PLAY_COLOUR_LEGEND + _PLAY_ACTION_MEANINGS)
_SYSTEM_PLAY = _insert(_SYSTEM_PLAY, _PLAY_ANIMATION_AFTER, _PLAY_ANIMATION)
_SYSTEM_PLAY = _insert(_SYSTEM_PLAY, _PLAY_PLAN_RULES_BEFORE, _PLAY_PLAN_RULES, before=True)
# --- end of the ported guidance in the system prompt -----------------------------------------------------------

_RUN_TESTS_PLAY = """Run the contract tests, then replay every step played so far through engine.py, in order. Stops after
`failures` failing steps (1 to 10, default 1). Reports how many steps pass before the first failure, the
first failure in full (images, regions, what a click hit, your sprites there, what step() printed, the
replay_step command), and one line per further failure."""

_COMMIT_PLAY = """Submit engine.py as your fix. Runs the tests first (every step played so far): if any fails you get the
report and the fit round goes on; when they all pass, the fix is kept and the harness asks you to plan the
next moves. message: """ + _COMMIT_MESSAGE

_COMMIT_MOVES = """Send moves to the game, in order. The moves are the Actions you stepped your replica with, as python
prints them: "Action(4)", "Action(6, x=12, y=40)" (a click; the harness computes its cell), "Action(0)" for
RESET. First the tests run on engine.py (every step played so far): if any fails, nothing is sent and you
get the report (fix engine.py first). Otherwise each move is predicted with your replica, sent to the game,
and the game's result is compared with the prediction (final frame, status, levels completed): on a match
the next move is sent; at the first difference the batch stops, the moves after it are not sent, and a fit
round opens on that step. The batch also stops when a level is solved or the game ends. At most __BATCH__
moves per call, one call per turn."""

_COMMIT_MOVES_TOOL = {
    "type": "function",
    "function": {
        "name": "commit_moves",
        "description": _COMMIT_MOVES,
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": 'The moves, in order, as python prints an Action: "Action(4)", "Action(6, x=12, y=40)" (a '
                                   'click at screen pixel x, the column, and y, the row, 0-63), "Action(0)" for RESET; a printed '
                                   'list "[Action(4), Action(0)]" as one string is taken too, and so are the labels "UP", "DOWN", '
                                   '"LEFT", "RIGHT", "SPACE", "RESET", "UNDO" and {"click": [x, y]}. Only the actions this game '
                                   "advertises.",
                    "items": {},
                    "minItems": 1,
                },
                "note": {
                    "type": "string",
                    "description": "One or two sentences: what this batch is meant to do and what your replica predicts.",
                },
            },
            "required": ["actions", "note"],
        },
    },
}


def commit_moves_description(batch_size: int) -> str:
    return _COMMIT_MOVES.replace("__BATCH__", str(int(batch_size)))


def _status_text(step) -> str:
    if step.state == "WIN":
        return "the game is won"
    if step.state == "GAME_OVER":
        return "the game is over (only RESET is accepted now)"
    return "playing"


_KEY_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 7: "UNDO"}


def move_text(action) -> str:
    """A move as the play messages name it: the Action as python prints it (commit_moves takes it back as it is),
    with the key's name: "Action(4) (RIGHT)", "Action(0) (RESET)", "Action(6, x=12, y=40) (a click)"."""
    name = "a click" if action.id == 6 else "RESET" if action.id == 0 else _KEY_NAMES.get(action.id, f"ACTION{action.id}")
    return f"{action_code(action)} ({name})"


def accepted_actions_text(available: list[int]) -> str:
    """The actions a game accepts, as commit_moves takes them (the Action as python prints it), in one sentence."""
    parts = [f"Action({a}) {_KEY_NAMES[a]}" for a in sorted(available) if a in _KEY_NAMES]
    if 6 in available:
        parts.append("clicks Action(6, x=x, y=y) (x the column, y the row, screen pixels 0-63)")
    parts.append("Action(0) RESET (restarts the level; it counts as an action)")
    return "The game accepts: " + ", ".join(parts) + "."


def _ranges(indices: list[int]) -> str:
    return _tester_ranges(sorted(indices))


# The paragraph after the engine.py listing in a PLAN message (elide_engine_listing keeps it).
PLAN_CLOSING = "\n\nWork out the next moves"
ENGINE_CLOSINGS = (ENGINE_CLOSING, PLAN_CLOSING)

# The play agent's notes, appended to the turn's last tool output.
PLAN_NUDGE = (
    "\n\n[harness] {n} turns in this plan round without commit_moves. The goal is to play: send a short batch now, even "
    "a probing one of 1-3 moves; your replica only improves from what the game answers."
)
FIT_ESCAPE = (
    "\n\n[harness] {n} turn(s) on step {k} without an accepted commit. You may now play on with your replica out of step with "
    "the game: commit_moves(actions, note) then sends moves although the tests fail; they are not checked against your "
    "replica, and the steps from {k} on are marked unexplained (the tests replay them but do not compare them). Your "
    "replica is back in step at the first RESET or level change the game makes: a RESET does it at once (it restarts the "
    "level, losing its progress, and costs one action), or you play blind to the end of the level (make_level must then "
    "draw the next level's start). Or keep fixing step {k} and commit_engine(message) as usual."
)
COMMIT_HINT_PLAY = {
    "fit": (
        "\n\nSteps 0-{k} pass{skipped}. You can now call commit_engine(message) to submit the fix, or keep refining first; "
        "after the commit you plan the next moves."
    ),
    "plan": "\n\nSteps 0-{k} pass{skipped} with engine.py as it is now: commit_moves sends moves with it (and commits it).",
}


def plan_message(
    game: str, trace: Trace, *, last_batch: str, budget_line: str, batch_size: int, engine_read: str = "",
    kernel_names: str = "", baseline: list[int] | None = None, unexplained: list[int] | None = None,
    out_of_sync: int | None = None, engine_note: str = "", images: bool = True,
) -> str:
    """The PLAN message: the replica reproduces every step so far (or, out of step since `out_of_sync`, plays
    blind); the game's state, the last batch's outcome, the actions it accepts, the budget, the current frame
    (attached as an image by the agent), and what to do. `unexplained`: the steps played while the replica was
    out of step; `engine_note`: a sentence on engine.py changed since its commit."""
    s = trace.steps[-1]
    n = len(trace.steps)
    level = min(s.levels_completed, max(0, s.win_levels - 1))
    base = f" The human baseline for level {level} is {baseline[level]} actions." if baseline and level < len(baseline) else ""
    if out_of_sync is None:
        skipped = f" (but the unexplained ones, which are not compared: {_ranges(unexplained)})" if unexplained else ""
        head = f"Plan the next moves. Steps 0-{n - 1} pass with your replica as committed{skipped}."
        about_now = "state_now() is your replica's state now"
        unexplained_line = (
            "\nUnexplained steps were played while your replica was out of step with the game; replay_step(k) shows one when "
            "you want to come back to it." if unexplained else ""
        )
    else:
        head = (
            f"Plan the next moves. Your replica is out of step with the game since step {out_of_sync}: moves are sent without "
            "being checked, and steps are unexplained (not compared by the tests) until it is back in step, at the first RESET "
            "(at once; it restarts the level and costs one action) or level change (then make_level must draw the new "
            "level's start)."
        )
        about_now = "state_now() is your replica's state with every step replayed on it, the unexplained ones too, so it may differ from the game"
        unexplained_line = f"\nUnexplained so far: steps {_ranges(unexplained)}." if unexplained else ""
    shown = "(shown below)" if images else "(show_frames(recording[-1].after) shows it)"
    if out_of_sync is None:
        send = (f"commit_moves(actions, note): up to {batch_size} moves, sent one by one and\neach checked against your replica's "
                "prediction; the batch stops at the first difference (a fit round opens), after a\nsolved level and when the game ends.")
    else:
        send = (f"commit_moves(actions, note): up to {batch_size} moves, sent one by one,\nunchecked; the batch stops at the first "
                "RESET or level change (your replica is then back in step) and\nwhen the game ends.")
    return f"""{head}

Game: {game}, at level {level} ({s.levels_completed} of {s.win_levels} levels completed), {_status_text(s)}. {last_batch}{base}
{accepted_actions_text(trace.steps[0].available_actions)}
{budget_line}{unexplained_line}{(chr(10) + engine_note) if engine_note else ""}
In python, `recording` holds every step played so far (steps 0-{n - 1}); recording[-1].after is the game's current frame
{shown}. {about_now}; replica.step(s, Action(...)) on a copy of it plays a move, and commit_moves sends those Actions as
python prints them (Action(4), Action(6, x=12, y=40)).
{(chr(10) + kernel_names) if kernel_names else ""}{(chr(10) + engine_block(engine_read) + chr(10)) if engine_read else ""}
Work out the next moves on your replica, then {send}"""


def mismatch_message(
    trace: Trace, k: int, verdict: str, dropped: int, report: str, engine_read: str = "", kernel_names: str = "",
    auto_reset: bool = True, engine_note: str = "", predicted: bool = True,
) -> str:
    """The FIT message after a move went differently from the replica's prediction: step k (the last step
    played), what differed (`verdict`, one line), how many moves of the batch were not sent, what the step
    changed piece by piece, the test report (the comparison, with its picture), engine.py and the task.
    `predicted` False: no prediction was made (a step played out of step, or found when a run resumed); the
    replica as it is now does not reproduce step k."""
    s = trace.steps[k]
    level = trace.steps[k - 1].levels_completed if k > 0 else 0
    notes = []
    if s.state == "WIN":
        notes.append(f"It solves level {level}, the last one: the game is won.")
    elif s.levels_completed > level:
        notes.append(
            f"It solves level {level}: the frame after it is level {s.levels_completed}'s first frame, which "
            f"make_level({s.levels_completed}) must draw (step_to_fix.pieces_after.code() gives code for it; reuse the "
            "sprite kinds engine.py already has where they fit)."
        )
    if s.state == "GAME_OVER":
        notes.append("After it the game is over; " + (
            "the harness will RESET the level once your replica reproduces it." if auto_reset else "only RESET is accepted now."))
    left = "" if not dropped else f" The {dropped} move{'s' if dropped > 1 else ''} after it in your batch {'were' if dropped > 1 else 'was'} not sent."
    before = "Step 0 matches" if k == 1 else f"Steps 0-{k - 1} match" if k > 1 else "It is the game's first frame"
    keep = f", keeping steps 0-{k - 1} passing" if k > 0 else ""
    title = f"step {k} did not go as your replica predicted" if predicted else f"your replica does not reproduce step {k}"
    return f"""Fix your replica: {title}.

Step {k}: {move_text(s.action)}, played in level {level}; the game returned {s.n_frames} frame(s), the tests compare the last.{animation_note(trace, k, report)}
What differed: {verdict}. {before}.{left}{(" " + " ".join(notes)) if notes else ""}{(chr(10) + engine_note) if engine_note else ""}
`recording` now holds steps 0-{k}, and `step_to_fix` is step {k} (recording[-1]).

{step_objects(trace, k)}

The test report:

{report.strip()}
{(chr(10) + kernel_names) if kernel_names else ""}{(chr(10) + engine_block(engine_read) + chr(10)) if engine_read else ""}
Fix step {k}{keep}; commit_engine(message) when the tests pass, then plan the next moves."""


def batch_lines(trace: Trace, first: int, outcomes: list[dict]) -> list[str]:
    """One line per move of a batch: "#12 Action(1): matches (level 0, 1 frame)" or what differed; `outcomes` are
    the play agent's per-move records ({"index", "label", "ok", "verdict", "frames", "level", "state"}; "ok"
    None: sent while the replica was out of step, not checked)."""
    lines = []
    for o in outcomes:
        if o["ok"] is None:
            what = "sent, not checked (your replica is out of step)"
        elif o["ok"]:
            what = "matches your prediction"
        else:
            what = f"differs from your prediction: {o['verdict']}"
        extra = []
        if o.get("warning"):
            extra.append(o["warning"])
        if o.get("state") == "WIN":
            extra.append("the game is WON")
        elif o.get("state") == "GAME_OVER":
            extra.append("GAME OVER")
        elif o.get("level_solved"):
            extra.append(f"level {o['level']} solved")
        if o.get("resync"):
            extra.append("your replica is back in step with the game here")
        tail = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"  #{o['index']} {o['label']}: {what}{tail}")
    return lines


# --- Ported from the base harness: the animation digest in the step messages, and the PLAN message's additions -------


def animation_note(trace: Trace, k: int, report: str = "") -> str:
    """The animation digest of step k (engine_re.animation.report_lines) on new lines, for a message that says how many
    frames the game returned; "" for a step of one frame, or when `report` (the test report the message carries) has
    it already."""
    s = trace.steps[k]
    if s.n_frames <= 1:
        return ""
    lines = animation.report_lines(trace.steps[k - 1].last if k > 0 else None, s.frames, k, indent="")
    if not lines or lines[0].strip() in report:
        return ""
    return "\n" + "\n".join(lines)


# LEVEL_START_USER_PROMPT (the first message after a level is completed), adapted: the new board is recording[-1].after.
LEVEL_START_PLAY = """You have completed the previous level. recording[-1].after now contains the starting board of the next level; __SHOWN__.
Build a new plan for this layout rather than continuing the previous level's action sequence.
Start from the mechanics you established on the previous level (the rules in engine.py); do not rediscover them without
reason. Inspect the new board for unfamiliar elements, changed arrangements, or interactions your previous understanding
does not explain. New elements often introduce mechanics needed to solve this level, so prioritize small, informative
tests when their behavior is unclear.
Reassess the goal: does the previous objective still apply, now requiring the new mechanics, or does the evidence suggest
a different objective? Combine retained knowledge with new findings to plan for this board."""
UNFAMILIAR_SHOWN = 8  # pieces listed as the new level's unfamiliar elements
UNFAMILIAR_FRAMES = 24  # frames of the previous level whose shapes are compared (distinct ones, evenly spaced)


def unfamiliar_pieces(trace: Trace, limit: int = UNFAMILIAR_SHOWN) -> tuple[int, list[str], int] | None:
    """The pieces of the last step's frame (a new level's first frame) whose shape names occur in none of the previous
    level's frames: (that level, one line per piece, at most `limit`, how many more); None when the segmentation fails."""
    steps = trace.steps
    n = len(steps)
    try:
        seg = segment.Segmenter(trace)
        previous = segment.shown_level(trace, n - 2)
        frames: dict[bytes, int] = {}
        for i in range(n - 1):
            if steps[i].last is not None and steps[i].state != "WIN" and segment.shown_level(trace, i) == previous:
                frames.setdefault(steps[i].last.tobytes(), i)
        chosen = sorted(frames.values())
        if len(chosen) > UNFAMILIAR_FRAMES:
            chosen = [chosen[round(j * (len(chosen) - 1) / (UNFAMILIAR_FRAMES - 1))] for j in range(UNFAMILIAR_FRAMES)]
        known = {p.shape for i in chosen for p in seg.pieces(i) if p.role == "object"}
        new = [(j, p) for j, p in enumerate(seg.pieces(n - 1)) if p.role == "object" and p.shape not in known]
    except Exception:  # noqa: BLE001  (a frame the segmentation cannot read must not stop the run)
        return None
    lines = [f"[{j}] {p.shape}, colour {p.colour} ({COLOR_NAMES.get(p.colour, '?')}), {p.width}x{p.height} at ({p.x}, {p.y})"
             + (" (a screen piece)" if p.screen else "") for j, p in new[:limit]]
    return previous, lines, max(0, len(new) - limit)


def level_start_text(trace: Trace, images: bool = True) -> str:
    """LEVEL_START_PLAY and the new board's unfamiliar elements when the last step played entered a new level (it
    solved one and the game goes on); "" otherwise."""
    steps = trace.steps
    if len(steps) < 2 or steps[-1].levels_completed <= steps[-2].levels_completed or steps[-1].state == "WIN":
        return ""
    shown = "the image below shows this new board" if images else "show_frames(recording[-1].after) shows this new board"
    text = LEVEL_START_PLAY.replace("__SHOWN__", shown)
    found = unfamiliar_pieces(trace)
    if found is None:
        return text
    previous, lines, more = found
    if not lines:
        return text + f"\nEvery piece of the new board (recording[-1].pieces_after) has a shape seen in level {previous}'s frames."
    head = (f"Unfamiliar elements to test first: the pieces of the new board (recording[-1].pieces_after; x, y in grid cells) whose "
            f"shapes occur in none of level {previous}'s frames:")
    return text + "\n" + head + "\n" + "\n".join("  " + line for line in lines) + (f"\n  ... and {more} more" if more else "")


PLAN_NOTES_LINES = 40  # lines of notes.md a PLAN message shows
NOTES_TEMPLATE = "Goal model:\nOpen questions:\nPlan:\n"  # notes.md as the play agent creates it
NOTES_EMPTY = ('notes.md holds nothing yet: keep the goal model, the open questions and the plan there with '
               'edit_file(path="notes.md", edits=[...]).')


def notes_block(notes: str | None, limit: int = PLAN_NOTES_LINES) -> str:
    """notes.md for a PLAN message: its first `limit` lines, with a note when the rest is cut."""
    if notes is None or not notes.strip() or notes.split() == NOTES_TEMPLATE.split():
        return NOTES_EMPTY
    lines = notes.rstrip("\n").splitlines()
    cut = (f'\n  [cut: {len(lines) - limit} more line(s) of notes.md not shown; read_file("notes.md") shows them. Keep it short.]'
           if len(lines) > limit else "")
    return "notes.md (your goal model, open questions and plan):\n" + "\n".join("  " + line for line in lines[:limit]) + cut


def plan_additions(text: str, trace: Trace, notes: str | None = None, images: bool = True) -> str:
    """A PLAN message (plan_message) with the ported additions: after its first paragraph, the level-start paragraph
    when the last step entered a new level (level_start_text); before the engine.py listing (or the closing paragraph
    when there is none), notes.md (notes_block). Compaction (elide_engine_listing) keeps both."""
    level = level_start_text(trace, images)
    if level and "\n\n" in text:
        head, rest = text.split("\n\n", 1)
        text = f"{head}\n\n{level}\n\n{rest}"
    block = notes_block(notes) + "\n\n"
    if ENGINE_HEADER in text:
        return text.replace(ENGINE_HEADER, block + ENGINE_HEADER, 1)
    closing = PLAN_CLOSING.strip()
    at = text.rfind(closing)
    return text if at < 0 else text[:at] + block + text[at:]
