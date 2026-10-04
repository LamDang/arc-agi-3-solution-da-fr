"""System prompt, first user message and tool schemas for the reverse-engineering agent."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from engine_re.trace import Trace

API_NOTES = (Path(__file__).with_name("api_notes.md")).read_text(encoding="utf-8")

SYSTEM_PROMPT = """You are reverse-engineering the game engine of an ARC-AGI-3 game from a recording of someone playing it.
You have every action that was played and every frame the real engine returned. Your job is to write a Python module,
engine.py, that reproduces the real engine exactly: replaying the recorded actions through a fresh instance of your
engine must return exactly the recorded observations at every step.

# What passing means
run_tests replays the recorded actions through a fresh instance of your engine and compares, at every step: the number
of frames, every pixel of every frame, the state (NOT_FINISHED / WIN / GAME_OVER), levels_completed, win_levels and
available_actions. The goal is "ALL STEPS MATCH". The session ends as soon as that happens.

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
- run_tests: replay and report the first steps where your engine deviates, with pixel diffs. from_level=L tests only
  level L onwards (your engine starts at set_level(L)), so you can work on a later level before earlier ones pass.
- finish: stop, when every step matches or you are truly stuck.

# How to work
1. Look before you write: summary(), detect_grid(), show() the first frame of each level, components(), and
   show_step(i) / animation(i) for what each action changed. Work out the logical grid size and scale, the background
   and letterbox colours, the objects (sprites), and the HUD drawn in screen pixels.
2. Make step 0 (the RESET) match exactly first: camera size and colours, the level 1 layout, the HUD. A reliable way
   to get a layout pixel-exact: downsample the level's first frame to the logical grid (logical(frame, geom)), keep
   everything that never changes as one background sprite, and make separate sprites only for the things that move,
   change or get clicked.
3. Then fix the first failing step each time: understand what the action did, implement the rule, re-test. Count frames:
   every call to step() before complete_action() renders one frame, and entering a new level adds one more.
4. Keep outputs small: print regions and summaries, not whole 64x64 arrays repeatedly.
5. Test often. Every change should move the first mismatch later or fix more steps.

""" + API_NOTES


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


def first_user_message(game: str, trace: Trace, skeleton: str) -> str:
    return f"""Game: {game}. Reproduce its engine in engine.py.

The recording:
{describe_trace(trace)}

engine.py currently holds this skeleton (the structure every real game follows; it runs but matches nothing yet):

```python
{skeleton}
```

Start by exploring the recording with the python tool."""


def resume_user_message(game: str, trace: Trace, turns: int, test_report: str, engine_lines: int) -> str:
    return f"""Game: {game}. Reproduce its engine in engine.py.

The recording:
{describe_trace(trace)}

This continues an earlier session on this game ({turns} turns) that was interrupted. Its conversation is gone and the
python kernel was restarted (its variables are gone), but engine.py ({engine_lines} lines) holds the work so far.
Its current test result:

{test_report}

Read engine.py with view_engine, then continue."""


TOOLS = [
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
            "description": (
                "Replay the recorded actions through a fresh instance of engine.py and compare with the recording. "
                "Reports how many steps match, the first mismatching steps in detail (pixel diffs, frame counts, "
                "state fields, tracebacks) and the list of all mismatching steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_level": {
                        "type": "integer",
                        "description": "Test only level L onwards: the engine starts with set_level(L). Omit for a full replay.",
                    },
                    "details": {"type": "integer", "description": "How many failing steps to explain in detail (default 2, max 6)."},
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
