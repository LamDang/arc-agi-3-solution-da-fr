"""System prompt, first user message and tool schemas for the reverse-engineering agent."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from engine_re.trace import Trace

API_NOTES = (Path(__file__).with_name("api_notes.md")).read_text(encoding="utf-8")
GAME_NOTES = (Path(__file__).with_name("game_notes.md")).read_text(encoding="utf-8")

INTERFACES = ("simple", "arcengine")

_SIMPLE_PROMPT = """You are reverse-engineering the game engine of an ARC-AGI-3 game from a recording of someone playing it.
You have every action that was played and every frame the real engine returned. Your job is to write a Python module,
engine.py, that reproduces the game: replaying the recorded actions through it must give the recorded results.

# What you write
engine.py starts with a FIXED INTERFACE block: the Sprite, Action and State classes and the rules for how a State is
drawn. Do not edit that block. It already provides what every game shares, with the same rules as the real games:
sprites with layers, visibility, collidability, blocking modes, rotation, mirroring and scale; collisions
(state.try_move, state.collisions, sprite.collides_with); lookups (state.sprite_at, sprites_at, by_tag, by_name); and
a per-level view (state.view: grid scale, rotation and mirroring of the whole screen). Below it you write two functions:
- make_level(n) -> State: the state at the start of level n: grid size, every sprite (border, background, objects,
  HUD), the hidden variables (state.vars) and, if the screen is shown turned or mirrored, state.view.
- step(state, action): apply one action to the state, in place. Set state.status = "level_solved" or "game_over"
  when that happens.
The harness does the rest. It calls make_level once per level and hands step() a fresh copy of that state whenever the
level starts (on entering it and on every RESET), so make_level may use module-level data and state.vars may hold
Sprite references. It also draws the state, counts completed levels, handles WIN and GAME_OVER, and turns clicks into
grid cells (action.cell). Give the border and background sprites collidable=False so they never block anything.

# What passing means
run_tests runs two suites. The contract tests check that the fixed interface is unchanged, that states are valid, that
step accepts every advertised action, and that the same actions give the same result. The acceptance test replays the
recorded actions and compares, after every action, the FINAL frame (every pixel of your drawn state) and the game state
(NOT_FINISHED / WIN / GAME_OVER, levels_completed). Animation frames are not compared: do each action's whole effect in
one step() call. Look at animation frames only to understand what an action does. The goal is "ALL STEPS MATCH" with
every contract test passing. The session ends as soon as that happens.

# Rules
- Implement the game's real rules and level data, so that your engine would also be right on actions nobody played.
  Do not hard-code recorded frames, per-step outputs or anything keyed to the step number or the action history:
  your engine will later be tested against the real game on new action sequences.
- engine.py must be self-contained: standard library and numpy only, no file reads. Put level data in the file as
  Python literals. Generate those literals with the python tool from the recorded frames rather than typing pixels by
  hand; the python tool may write engine.py (or parts of it) directly.
- The original game source code is not available anywhere you can reach; do not look for it.

# Tools
- python: a persistent Python kernel. The recording is loaded as `trace` / `S` (S[i] is step i) with analysis helpers;
  call help_helpers() to list them. It can also run your engine for debugging (engine, render, new_game, play, replay,
  compare, check_contract, try_step).
- view_engine, write_engine, edit_engine: read and change engine.py.
- run_tests: the contract tests, then the acceptance test. By default it stops at the first failure and explains it:
  the regions of the final frame that differ (numbered__IMAGES_TOOL__), the colours, your sprites there (#12 means
  state.sprites[12]) before and after the step, what engine.py printed during the step, and a try_step(i) command that
  reproduces it in the kernel. level=L tests only level L (your engine starts at make_level(L)), so you can work on a
  later level before earlier ones pass; stop_on_fail=false lists every failing step.
- print() in make_level and step is captured per step: run_tests shows the failing step's output, try_step shows it too.
- finish: stop, when every step matches or you are truly stuck.

# How to work
1. Look before you write: summary(), detect_grid(), show() the first frame of each level, components(), and
   show_step(i) / animation(i) for what each action changed. Work out the logical grid size and scale, the border and
   background colours, the objects, and the HUD drawn in screen pixels.
2. Make step 0 (the RESET) match exactly first: grid size, border and background sprites, the level's objects, the
   HUD. A reliable way to get a layout pixel-exact: downsample the level's first frame to the logical grid
   (logical(frame, geom)), keep everything that never changes as one background sprite, and make separate sprites only
   for the things that move, change or get clicked. render(state) draws a State exactly as the harness does.
3. Model the game the way the real one is built: one tagged sprite per object (walls, pieces, buttons, goals), hidden
   values in state.vars (budget, counters, what is selected), and the HUD as screen sprites that step() updates. Use
   the built-in try_move, collisions, sprite_at and sprites_at rather than writing your own geometry.
4. Then fix the first failing step each time: read the report__IMAGES_STEP__, run the try_step command it prints to see
   your state before and after the step and what changed, implement the rule, re-test. Only the end state of each
   action counts, so skip animations.
5. Keep outputs small: print regions and summaries, not whole 64x64 arrays repeatedly.
6. Write code early and test often: a partial engine plus run_tests tells you exactly what to fix next, faster than
   more analysis. Every change should move the first mismatch later or fix more steps.
7. Keep a short notes.md in the workspace with what you have established (geometry, colours, sprites, rules, open
   questions). Old tool outputs are dropped from your context as it grows; the notes and engine.py persist.

"""

_INTRO = """You are reverse-engineering the game engine of an ARC-AGI-3 game from a recording of someone playing it.
You have every action that was played and every frame the real engine returned. Your job is to write a Python module,
engine.py, that reproduces the real engine: replaying the recorded actions through a fresh instance of your engine
must return the recorded observations at every step.
"""

_PASSING = {
    "final": """
# What passing means
run_tests replays the recorded actions through a fresh instance of your engine and compares, after every action: the
FINAL frame (every pixel of the last frame the action returned), the state (NOT_FINISHED / WIN / GAME_OVER),
levels_completed, win_levels and available_actions. Animation frames and the number of frames are NOT compared: the
real game sometimes animates an action over several frames, but your engine only needs its end result, so do each
action's whole effect in one step() call and then call complete_action(). Look at animation frames only to understand
what an action does. The goal is "ALL STEPS MATCH". The session ends as soon as that happens.
""",
    "all": """
# What passing means
run_tests replays the recorded actions through a fresh instance of your engine and compares, at every step: the number
of frames, every pixel of every frame, the state (NOT_FINISHED / WIN / GAME_OVER), levels_completed, win_levels and
available_actions. The goal is "ALL STEPS MATCH". The session ends as soon as that happens.
""",
}

_RULES_AND_TOOLS = """
# Rules
- Implement the game's real rules and level data, so that your engine would also be right on actions nobody played.
  Do not hard-code recorded frames, per-step outputs or anything keyed to the step number or the action history:
  your engine will later be tested against the real engine on new action sequences.
- engine.py must be self-contained: standard library, numpy and arcengine only, no file reads. Put level data in the
  file as Python literals. Generate those literals with the python tool from the recorded frames rather than typing
  pixels by hand; the python tool may write engine.py (or parts of it) directly.
- The original game source code is not available anywhere you can reach; do not look for it.

# Tools
- python: a persistent Python kernel. The recording is loaded as `trace` / `S` (S[i] is step i) with analysis helpers;
  call help_helpers() to list them. It can also run your engine for debugging (new_game, play, replay, compare).
- view_engine, write_engine, edit_engine: read and change engine.py.
- run_tests: replay and report where your engine deviates (by default up to the first failure), with the differing
  regions of the final frame. level=L tests only level L (your engine starts at set_level(L)), so you can work on a
  later level before earlier ones pass; stop_on_fail=false lists every failing step.
- finish: stop, when every step matches or you are truly stuck.

# How to work
1. Look before you write: summary(), detect_grid(), show() the first frame of each level, components(), and
   show_step(i) / animation(i) for what each action changed. Work out the logical grid size and scale, the background
   and letterbox colours, the objects (sprites), and the HUD drawn in screen pixels.
2. Make step 0 (the RESET) match exactly first: camera size and colours, the level 1 layout, the HUD. A reliable way
   to get a layout pixel-exact: downsample the level's first frame to the logical grid (logical(frame, geom)), keep
   everything that never changes as one background sprite, and make separate sprites only for the things that move,
   change or get clicked.
__STEP3__4. Keep outputs small: print regions and summaries, not whole 64x64 arrays repeatedly.
5. Write code early and test often: a partial engine plus run_tests tells you exactly what to fix next, faster than
   more analysis. Every change should move the first mismatch later or fix more steps.
6. Keep a short notes.md in the workspace with what you have established (geometry, colours, sprites, rules, open
   questions). Old tool outputs are dropped from your context as it grows; the notes and engine.py persist.

"""

_STEP3 = {
    "final": """3. Then fix the first failing step each time: understand what the action did, implement the rule, re-test. Only
   the end state of each action counts, so skip animations.
""",
    "all": """3. Then fix the first failing step each time: understand what the action did, implement the rule, re-test. Count frames:
   every call to step() before complete_action() renders one frame, and entering a new level adds one more.
""",
}


def system_prompt(match: str = "final", interface: str = "simple", images: bool = True) -> str:
    """The system prompt for an engine interface ("simple": make_level/step; "arcengine": an ARCBaseGame
    subclass) and a matching rule ("final": last frame + state per step; "all": every frame, arcengine only).
    images: whether test reports come with pictures of the failing step's frames."""
    if interface == "simple":
        if match != "final":
            raise ValueError("the simple interface produces one frame per action, so it is scored with match='final'")
        text = _SIMPLE_PROMPT.replace("__IMAGES_TOOL__", "; also boxed in an image of your frame next to the original's" if images else "")
        text = text.replace("__IMAGES_STEP__", " and its image" if images else "")
        return text + GAME_NOTES
    if interface != "arcengine":
        raise ValueError(f"interface must be one of {INTERFACES}")
    body = _RULES_AND_TOOLS.replace("__STEP3__", _STEP3[match])
    return _INTRO + _PASSING[match] + body + API_NOTES


SYSTEM_PROMPT = system_prompt("final")


def describe_trace(trace: Trace) -> str:
    steps = trace.steps
    acts = Counter(str(s.action) if s.action.id != 6 else "ACTION6 (click)" for s in steps)
    frames = Counter(s.n_frames for s in steps)
    overs = sum(s.state == "GAME_OVER" for s in steps)
    starts = trace.level_starts()
    return (
        f"- {len(steps)} steps (step 0 is the RESET that starts the game); actions: {dict(acts)}\n"
        f"- available_actions advertised: {steps[0].available_actions}; win_levels: {steps[0].win_levels}\n"
        f"- levels reached: {sorted(starts)} (level -> first step whose final frame shows it: {starts}); "
        f"final state {steps[-1].state} with {steps[-1].levels_completed} level(s) completed\n"
        f"- steps ending in GAME_OVER: {overs}; frames per step (count: steps): {dict(sorted(frames.items()))}"
    )


def first_user_message(game: str, trace: Trace, skeleton: str, interface: str = "simple") -> str:
    intro = (
        "engine.py currently holds this starting module (the fixed interface and empty make_level and step; "
        "it runs but matches nothing yet):"
        if interface == "simple"
        else "engine.py currently holds this skeleton (the structure every real game follows; it runs but matches nothing yet):"
    )
    return f"""Game: {game}. Reproduce its engine in engine.py.

The recording:
{describe_trace(trace)}

{intro}

```python
{skeleton}
```

Start by exploring the recording with the python tool."""


def resume_user_message(game: str, trace: Trace, turns: int, test_report: str, engine_lines: int, notes: str = "") -> str:
    notes_part = f"\nWhat that session left behind:\n\n{notes}\n" if notes else ""
    return f"""Game: {game}. Reproduce its engine in engine.py.

The recording:
{describe_trace(trace)}

This continues an earlier session on this game ({turns} turns) that was interrupted. Its conversation is gone and the
python kernel was restarted (its variables are gone), but engine.py ({engine_lines} lines) holds the work so far.
{notes_part}
The current test result of engine.py:

{test_report}

Read engine.py with view_engine, then continue."""


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Run Python in a persistent kernel (variables survive between calls). The value of a final expression "
                "is printed. Preloaded: np, trace, S, and helpers (help_helpers() lists them). Output is truncated "
                "when long, so print regions and summaries. Can read the trace and write files in the workspace "
                "(including engine.py)."
            ),
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
            "name": "view_engine",
            "description": "Show engine.py with line numbers (optionally a line range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "description": "First line (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line (inclusive)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_engine",
            "description": "Replace the whole of engine.py with new content.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "The complete new engine.py."}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_engine",
            "description": (
                "Replace an exact snippet of engine.py with new text. old_str must occur exactly once "
                "(include surrounding lines to make it unique) unless replace_all is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_str": {"type": "string", "description": "Exact text to replace."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
                },
                "required": ["old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "__RUN_TESTS__",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": (
                            "Test only level L: your engine starts at level L (make_level(L)), its drawing is compared "
                            "with the level's recorded start, then it plays that level's recorded steps. Omit for the "
                            "full replay from step 0."
                        ),
                    },
                    "stop_on_fail": {
                        "type": "boolean",
                        "description": (
                            "true (default): the report stops at the first failing test, explained in detail. false: "
                            "report everything: per-level counts, several failing steps explained, and the list of all "
                            "failing steps."
                        ),
                    },
                    "details": {
                        "type": "integer",
                        "description": "With stop_on_fail=false: how many failing steps to explain in detail (default 2, max 6).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the session. Use when every step matches, or when you cannot make further progress.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "What the engine implements and what is still wrong."}},
                "required": ["summary"],
            },
        },
    },
]

_RUN_TESTS = (
    "Test engine.py: the contract tests (for a make_level/step engine), then the acceptance test, which replays the "
    "recorded actions and compares each step's final frame and state with the recording. By default the report stops "
    "at the first failure: which steps match, then the failing step explained: the regions of the final frame that "
    "differ, numbered{images}, the colours (expected->got), your sprites in each region before and after the step "
    "(#12 is state.sprites[12]), what engine.py printed during the step, and a try_step command that reproduces it "
    "in the python kernel."
)


def tools(images: bool = True) -> list[dict]:
    """The tool schemas; images: whether run_tests reports come with pictures of the frames."""
    note = " and boxed in an image of your frame next to the original's" if images else ""
    out = copy.deepcopy(_TOOLS)
    for tool in out:
        if tool["function"]["name"] == "run_tests":
            tool["function"]["description"] = _RUN_TESTS.format(images=note)
    return out


TOOLS = tools(True)
