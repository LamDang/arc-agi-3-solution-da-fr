"""The functions preloaded in the agent's Python kernel.

The kernel's namespace starts with np, the fixed-block classes (Sprite, Action, View, State),
`recording` (the recorded steps, one StepView each) and these functions; everything else here is
private. In the stepwise harness (the kernel's --focus K) the recording on disk holds only steps
0..K and the namespace also holds `step_to_fix` (the StepView of step K, recording[K]); with
--history it keeps `recording` (steps 0..K) and summarize_levels, without it it has neither:

    read_file(path="engine.py", offset=None, limit=None)   the file with LINE#HASH anchors
    edit_file(path="engine.py", edits=[...])               change it at those anchors
    undo_edit(n=1, to=None)                                go back to an earlier version of engine.py
    render_state(state)                                    draw a State as the tests do
    show_frames(*frames, titles=None, boxes=None)          look at frames as images
    replay_step(i, state=None, action=None)                run one step of engine.py and explain it
    summarize_levels()                                     each level's first frame, steps and end
    replica                                                engine.py as it is now, reloaded after a change
    traced(), support(run=None)                            (play) what code on the replica ran, against the evidence

The kernel's replay mode (REPLAY, set while the harness re-runs the python cells of a resumed
conversation): edit_file() and undo_edit() do nothing and show_frames() makes no image. With REPLAY_FILES
(a forked run, engine_re.tools.fork_run: its notes.md starts over as the template), edit_file() still
applies edits to files other than engine.py, so the cells rebuild notes.md and the other files the model
wrote; engine.py's edits stay off (its versions are kept with the fork).

Each StepView also has the frames' segmentation (engine_re.segment), computed when first used and
only from the steps loaded: .grid, .pieces_before, .pieces_after (whose .code() writes sprites that
draw the frame) and .changes.

engine.py cannot be opened for writing from the kernel (engine_re.guard): edit_file() and
undo_edit() send their arguments to the harness (engine_re.kernel, engine_re.engine_files), which
applies them.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import json
import weakref
import hashlib
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np

from engine_re import diff_report, game_api, hashline, segment, tester
from engine_re import support as _support
from engine_re import animation  # StepView.animation
from engine_re.auto_sprites import kinds_summary
from engine_re.game_api import click_cell  # noqa: F401  (a play built-in: action.cell for a click, as the harness computes it)
from engine_re.trace import Action as _TraceAction, Trace

HEX = "0123456789abcdef"
COLOR_NAMES = {
    0: "white", 1: "light grey", 2: "grey", 3: "dark grey", 4: "darker grey", 5: "black",
    6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "light blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}
MAX_SHOWN = 4  # frames per show_frames() call
# The built-in functions, as the kernel preloads them (the model may call one as a tool by mistake: the
# harness then runs it as python).
FUNCTIONS = ("read_file", "edit_file", "undo_edit", "render_state", "show_frames", "replay_step", "summarize_levels")
# The play-and-model agent (engine_re.play_agent) adds two: the replica's state after everything played, and
# the grid cell a click lands on (the Action.cell the harness computes for a click). Moves are played by
# calling replica.step on copies of a State directly.
PLAY_FUNCTIONS = FUNCTIONS + ("state_now", "click_cell", "traced", "support")

# Set by the kernel (load_trace).
trace: Trace = None  # type: ignore[assignment]
recording: list[StepView] = []  # the model's `recording`: one StepView per loaded step (kept as one list object)
FOCUS: int | None = None  # the step to fix, in the stepwise harness: steps after it are not loaded
_SEGMENTER: segment.Segmenter | None = None  # grids, pieces and changes of the loaded steps (0..FOCUS), made lazily
ENGINE_PATH: Path = Path("engine.py")
IMAGES = True  # False: show_frames() prints hex views instead of making images
_RPC: Callable[[dict], dict] | None = None  # sends edit/undo requests to the harness
_SHOWN: list[dict[str, str]] = []  # images made by show_frames() during the current request
REPLAY = False  # the kernel is re-running earlier cells: no edits, no images (engine_re.kernel.replay_cells)
REPLAY_FILES = False  # in replay mode: edits to files other than engine.py are applied (a fork rebuilds notes.md)
SUPPORT_PATH: Path | None = None  # the committed engine's support map (the play kernel's --support): margins, traced()
_INSTRUMENTED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # module loaded from engine.py -> its support.Instrumented

_API = game_api.canonical()
Sprite, Action, View, State = _API.Sprite, _API.Action, _API.View, _API.State


# --- engine.py ------------------------------------------------------------------------------


def _is_engine(path: str | Path) -> bool:
    return Path(path).resolve() == ENGINE_PATH.resolve()


def read_file(path: str = "engine.py", offset: int | None = None, limit: int | None = None) -> None:
    """Print a file with every line as LINE#HASH:content, from line `offset` for `limit` lines.
    In engine.py the FIXED block is folded unless offset asks for its lines."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"{path} does not exist")
        return
    fold = game_api.fixed_block_lines(text) if _is_engine(path) else None
    smap = _support_map() if _is_engine(path) else None
    margin = _support.margins(smap, text) if smap else None
    try:
        print(hashline.render_read(text, offset, limit, fold=fold, name=str(path), margin=margin))
    except hashline.EditError as exc:
        print(exc)


def edit_file(path: str = "engine.py", edits: Any = None) -> None:
    """Apply anchored edits to a file (engine.py through the harness); prints what changed, a
    syntax check and fresh anchors, or why nothing was applied."""
    if REPLAY and (_is_engine(path) or not REPLAY_FILES):
        print("edit_file(): skipped, the kernel is replaying earlier cells")
        return
    if edits is None:
        print('edit_file(): give edits=[{"op": ..., ...}, ...]; see read_file() for the anchors.')
        return
    edits = _plain(edits)
    if _is_engine(path):
        if _RPC is None:
            print("edit_file(): engine.py can only be changed through the harness, which is not connected.")
            return
        print(_RPC({"op": "edit", "edits": edits})["text"])
        return
    file = Path(path)
    try:
        old = file.read_text(encoding="utf-8") if file.exists() else ""
        result = hashline.apply_edits(old, edits)
    except hashline.EditError as exc:
        print(exc)
        return
    if not result.summary:
        print(f"{path} was not changed: " + "; ".join(result.failed + result.noop))
        return
    file.write_text(result.text, encoding="utf-8")
    applied = f"applied {result.total - len(result.failed)} of {result.total} edits: " if result.failed else ""
    print(f"{path}: {applied}{'; '.join(result.summary)}.")
    print("\n".join(result.failed + [f"Warning: {w}" for w in result.warnings] + hashline.fresh_anchors(result.text, result.regions)))


def _plain(value: Any) -> Any:
    """Edits as plain JSON values (str subclasses become str, numpy integers int)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def undo_edit(n: int = 1, to: Any = None) -> None:
    """Put engine.py back as it was n changes ago, or to="best" (the version that matched the most
    steps), or to=k (version k). The restore is a new version, so undo_edit() again brings the change back."""
    if REPLAY:
        print("undo_edit(): skipped, the kernel is replaying earlier cells")
        return
    if _RPC is None:
        print("undo_edit(): engine.py can only be changed through the harness, which is not connected.")
        return
    print(_RPC({"op": "undo", "n": _plain(n), "to": _plain(to)})["text"])


def _load_engine() -> types.ModuleType:
    """engine.py loaded fresh, as a module: compiled with its conditions wrapped in a recorder that does nothing
    outside traced() (support.instrument; line numbers and behaviour unchanged), or as it is when that fails."""
    name = "candidate_engine_dev"
    source = ENGINE_PATH.read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE_PATH)
    sys.modules[name] = module
    inst = _support.instrument(source, str(ENGINE_PATH))
    if inst is not None:
        module.__dict__[_support.COND_NAME] = _support.no_cond
        _INSTRUMENTED[module] = inst
    exec(inst.code if inst is not None else compile(source, str(ENGINE_PATH), "exec", dont_inherit=True), module.__dict__)
    return module


class _ReplicaModule:
    """The `replica` built-in: engine.py, the model's replica of the game, as it is now. An attribute access
    loads the file again (_load_engine) when its content changed since the last load, else uses the module
    loaded then; so replica.step(...) and replica.make_level(...) never go stale after an edit, as `import
    engine` would. replica.step(state, action) fills in a click's action.cell when it is None, as the harness
    does before it calls step() (click_cell), so an Action written as it prints, Action(6, x=3, y=4), plays
    the same click in python as in the game (the module's own step, engine.py's, is unchanged)."""

    def __init__(self) -> None:
        self._sha: str | None = None
        self._module: types.ModuleType | None = None

    def _current(self) -> types.ModuleType:
        sha = hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest()
        if self._module is None or sha != self._sha:
            self._module = None
            self._module = _load_engine()
            self._sha = sha
        return self._module

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._current(), name)
        if name == "step" and callable(value):
            return _with_click_cell(self._current(), value)
        return value

    def __dir__(self) -> list[str]:
        return [n for n in dir(self._current()) if not n.startswith("__")]

    def __repr__(self) -> str:
        return "<replica: engine.py as it is now (loaded again whenever the file changed)>"


def _with_click_cell(module: types.ModuleType, step: Callable[..., Any]) -> Callable[..., Any]:
    """engine.py's step(state, action), a click without a cell getting the one the harness would give it."""

    def run(state: Any, action: Any) -> Any:
        if getattr(action, "id", None) == 6 and getattr(action, "cell", None) is None and hasattr(state, "grid"):
            cls = getattr(module, "Action", None) or Action
            x, y = int(action.x), int(action.y)
            action = cls(id=6, x=x, y=y, cell=click_cell(state, x, y))
        return step(state, action)

    run.__doc__, run.__name__ = step.__doc__, getattr(step, "__name__", "step")
    return run


replica = _ReplicaModule()


# --- Drawing and looking ----------------------------------------------------------------------


def render_state(state: Any) -> np.ndarray:
    """Draw a State as a 64x64 frame: the same code the tests use."""
    return game_api.render(state)


def _hexrow(row: Any) -> str:
    return "".join(HEX[v] if 0 <= v < 16 else "." for v in row)


def show_frames(*frames: Any, titles: list[str] | None = None, boxes: list[tuple[int, int, int, int]] | None = None) -> None:
    """Show 64x64 frames (or States, rendered first) side by side as one image, enlarged, titled,
    with the boxes (x0, y0, x1, y1, screen pixels, inclusive) outlined and numbered on each. The
    image comes in a message after this call's output. At most MAX_SHOWN frames per call. With
    images off, prints a hex view of the boxes (or of each frame at half resolution) instead."""
    if REPLAY:
        return
    items = list(frames[0]) if len(frames) == 1 and isinstance(frames[0], (list, tuple)) else list(frames)
    if not items:
        print("show_frames(): give one or more frames, e.g. show_frames(recording[3].after, render_state(state))")
        return
    if len(items) > MAX_SHOWN:
        print(f"show_frames(): {len(items)} frames given; showing the first {MAX_SHOWN}.")
        items = items[:MAX_SHOWN]
    arrays = []
    for k, item in enumerate(items):
        array = render_state(item) if hasattr(item, "sprites") else np.asarray(item)
        if array.shape != (64, 64):
            print(f"show_frames(): item {k} has shape {array.shape}, not (64, 64)")
            return
        arrays.append(array.astype(np.int16))
    titles = [str(t) for t in titles] if titles else [str(k + 1) for k in range(len(arrays))]
    titles = (titles + [str(k + 1) for k in range(len(titles), len(arrays))])[: len(arrays)]
    drawn = []
    for n, box in enumerate(boxes or [], 1):
        x0, y0, x1, y1 = (int(v) for v in box)
        drawn.append(diff_report.Box(n, (max(0, min(y0, y1)), max(0, min(x0, x1)), min(63, max(y0, y1)), min(63, max(x0, x1)))))
    if IMAGES:
        scale = diff_report.UPSCALE if len(arrays) <= 2 else 6
        png = diff_report.png_bytes(diff_report.panels_image(arrays, titles, drawn, scale))
        caption = "show_frames(): " + " | ".join(titles) + (f"; boxes {', '.join(str(b.n) for b in drawn)}" if drawn else "")
        _SHOWN.append({"png": base64.b64encode(png).decode("ascii"), "caption": caption})
        print(f"[image: {len(arrays)} frame(s), {', '.join(titles)}; it follows this output]")
        return
    if drawn:
        for b in drawn:
            r0, c0, r1, c1 = b.box
            r1, c1 = min(r1, r0 + 15), min(c1, c0 + 31)
            print(f"box {b.n}: rows {r0}-{r1}, cols {c0}-{c1}, one hex digit per pixel; " + " | ".join(titles))
            for r in range(r0, r1 + 1):
                print(f"{r:>3}  " + "  ".join(_hexrow(a[r, c0 : c1 + 1]) for a in arrays))
    else:
        print("each frame at half resolution (every other pixel), one hex digit per pixel; " + " | ".join(titles))
        for r in range(0, 64, 2):
            print(f"{r:>3}  " + "  ".join(_hexrow(a[r, ::2]) for a in arrays))


def take_shown() -> list[dict[str, str]]:
    """The images show_frames() made since the last call (the kernel sends them to the harness)."""
    out = list(_SHOWN)
    _SHOWN.clear()
    return out


# --- The recorded steps: `recording` and `step_to_fix` ----------------------------------------------

_ACTION_WORDS = {0: "RESET", 1: "up", 2: "down", 3: "left", 4: "right", 5: "interact", 6: "click", 7: "undo"}


class StepView:
    """A recorded step as the model sees it: recording[i], and step_to_fix (in the stepwise harness)
    is one of them. Data from the real game, frames only, never a State.

    index: its number i; action: the Action played; before: the frame before it (64x64 int8,
    frame[y, x]; None for step 0); after (also .last): the frame after it, the one the tests compare;
    frames: every frame it returned, (n, 64, 64); level: the level it is played in; outcome
    ("NOT_FINISHED", "WIN" or "GAME_OVER"; not the engine's State.status), levels_completed: after it;
    win_levels, available_actions.

    And, computed when first read and kept (engine_re.segment, from the loaded steps only): grid, the
    GridGuess of its level; pieces_before (None for step 0) and pieces_after, the frames' pieces;
    changes, what changed between them (None for step 0)."""

    def __init__(self, recorded: Trace, k: int):
        s = recorded.steps[k]
        self.index = k
        self.action = s.action
        self.frames = s.frames
        self.after = self.last = s.last
        self.before = recorded.steps[k - 1].last if k > 0 else None
        self.level = recorded.steps[k - 1].levels_completed if k > 0 else 0
        self.outcome = s.state
        self.levels_completed = s.levels_completed
        self.win_levels = s.win_levels
        self.available_actions = s.available_actions

    @property
    def grid(self) -> Any:
        """The GridGuess of the level it is played in, guessed from that level's loaded frames."""
        _visible(self.index)
        return _SEGMENTER.grid(segment.shown_level(trace, self.index - 1) if self.index else 0)

    @property
    def pieces_before(self) -> segment.Pieces | None:
        """The pieces of .before (None for step 0)."""
        _visible(self.index)
        return _SEGMENTER.pieces(self.index - 1) if self.index else None

    @property
    def pieces_after(self) -> segment.Pieces:
        """The pieces of .after, on the grid of the level it shows (the next level's after a step that
        solves one); .code() writes sprites that draw it."""
        _visible(self.index)
        return _SEGMENTER.pieces(self.index)

    @property
    def changes(self) -> segment.Changes | None:
        """What changed from .pieces_before to .pieces_after (None for step 0)."""
        _visible(self.index)
        return _SEGMENTER.changes(self.index)

    @property
    def animation(self) -> animation.Animation | None:
        """The digest of an animated step's frames (engine_re.animation): its transient cells and its timeline;
        None when the action returned one frame."""
        if not hasattr(self, "_animation"):
            self._animation = animation.digest(self.before, self.frames)
        return self._animation

    def __repr__(self) -> str:
        what = _ACTION_WORDS.get(self.action.id, str(self.action))
        if self.action.id == 6:
            what += f" at ({self.action.x}, {self.action.y})"
        return (f"<step {self.index}: {what} in level {self.level}; {len(self.frames)} frame(s); after it "
                f"{self.outcome}, {self.levels_completed} level(s) completed>")


def load_trace(loaded: Trace, focus: int | None = None) -> None:
    """Make `loaded` the recording the helpers use (the kernel calls this at the start, and again when
    the stepwise harness moves on to step `focus`). `recording` stays the same list object: its
    entries are replaced, so it grows with the trace."""
    global trace, FOCUS, _SEGMENTER
    trace, FOCUS = loaded, focus
    _SEGMENTER = segment.Segmenter(loaded, focus)  # a new one: the grids are guessed again from the steps now loaded
    _SEGMENTER.context = _code_context
    recording[:] = [StepView(loaded, k) for k in range(len(loaded))]


def _visible(i: int) -> None:
    if FOCUS is not None and not 0 <= i <= FOCUS:
        raise ValueError(f"only steps 0-{FOCUS} are loaded: step {FOCUS} is the one to fix, later steps come later")


# --- The recording, level by level ----------------------------------------------------------------

_MOVE_NAMES = {0: "RESET", 1: "up", 2: "down", 3: "left", 4: "right", 5: "interact", 6: "click", 7: "undo"}


def _steps_list(steps: list[int], limit: int = 6) -> str:
    if not steps:
        return "-"
    shown = ", ".join(str(i) for i in steps[:limit])
    return shown + (f", ... ({len(steps)})" if len(steps) > limit else "")


def _levels_text(recorded: Trace) -> str:
    """One row per level the recording plays: where its first frame is, its guessed grid, the steps
    played in it, the actions, the animated steps, RESETs and game overs, and how it ended."""
    steps = recorded.steps
    segmenter = _SEGMENTER if _SEGMENTER is not None and _SEGMENTER.trace is recorded else segment.Segmenter(recorded, FOCUS)
    win = steps[0].win_levels
    played: dict[int, list[int]] = {0: []}
    for i in range(1, len(steps)):
        played.setdefault(steps[i - 1].levels_completed, []).append(i)
    starts = recorded.level_starts()
    rows = [("level", "first frame", "grid", "steps played", "actions", "animated", "RESET", "GAME_OVER", "how it ended")]
    for level in sorted(played):
        indices = played[level]
        moves: dict[str, int] = {}
        for i in indices:
            name = _MOVE_NAMES.get(steps[i].action.id, str(steps[i].action))
            moves[name] = moves.get(name, 0) + 1
        solved = [i for i in indices if steps[i].levels_completed > level]
        if solved:
            ended = f"solved at step {solved[0]}" + (" (WIN)" if steps[solved[0]].state == "WIN" else "")
        elif FOCUS is not None:
            ended = f"not solved by step {len(steps) - 1}, the step to fix (later steps are not loaded)"
        else:
            ended = f"not solved: the recording ends ({steps[-1].state})"
        try:
            g = segmenter.grid(level)
            grid = f"{g.width}x{g.height} s{g.scale}"
        except ValueError:
            grid = "-"
        rows.append((
            str(level),
            f"recording[{starts.get(level, 0)}].after",
            grid,
            f"{indices[0]}-{indices[-1]} ({len(indices)})" if indices else "none",
            ", ".join(f"{name} x{n}" for name, n in moves.items()) or "-",
            _steps_list([i for i in indices if steps[i].n_frames > 1]),
            _steps_list([i for i in indices if steps[i].action.id == 0]),
            _steps_list([i for i in indices if steps[i].state == "GAME_OVER"]),
            ended,
        ))
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    table = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    head = (f"The recording: {len(steps)} steps (recording[0] is the RESET that starts the game), {len(played)} of the game's "
            f"{win} levels played; after step {len(steps) - 1} it is {steps[-1].state} with "
            f"{steps[-1].levels_completed} level(s) completed.")
    if FOCUS is not None:
        head = (f"The recording so far: steps 0-{len(steps) - 1} (recording[0] is the RESET that starts the game; step "
                f"{len(steps) - 1} is the one to fix), {len(played)} of the game's {win} levels reached.")
    note = ("A level's first frame is the last frame of the step that solved the level before (recording[0].after for level 0); "
            "recording[k].pieces_after.code() writes sprites that draw it. \"grid\": the logical grid guessed from the level's "
            "frames (width x height, scale). \"animated\": steps that returned more than one frame; the tests compare only the last one.")
    return "\n".join([head] + table + [note])


def summarize_levels() -> None:
    """Print one row per level of the recording: its first frame, its guessed grid, the steps played
    in it, their actions, animated steps, RESETs and game overs, and how the level ended."""
    print(_levels_text(trace))


# --- Running one step and explaining it ---------------------------------------------------------

_FIELDS = ("state", "levels_completed", "win_levels", "available_actions")


def _as_action(action: Any, i: int) -> _TraceAction:
    """The action to apply: step i's recorded one, or an id, (6, x, y), a dict or an Action."""
    if action is None:
        return trace.steps[i].action
    if isinstance(action, _TraceAction):
        return action
    if isinstance(action, (int, np.integer)):
        return _TraceAction(int(action))
    if isinstance(action, (tuple, list)):
        return _TraceAction(int(action[0]), *(int(v) for v in action[1:3]))
    if isinstance(action, dict):
        return _TraceAction.from_json(action)
    if hasattr(action, "id"):
        return _TraceAction(int(action.id), getattr(action, "x", None), getattr(action, "y", None))
    raise TypeError(f"action must be an action id, (6, x, y) or an Action, not {type(action).__name__}")


def _quietly(fn: Callable[[], Any], what: str) -> Any:
    """Run fn with the engine's prints dropped; on an error say where it happened, then raise."""
    try:
        with contextlib.redirect_stdout(game_api.PrintCapture(0, 0)):
            return fn()
    except Exception:
        print(f"your engine raised an error {what}:")
        raise


def _replay_recorded(game: game_api.GameRunner, k: int) -> Any:
    """Replay recorded step k on `game` as the tests do: at a resync point (the play agent's escape hatch)
    the runner is put back at the level's start; an error at an unexplained step is not raised (the engine
    is out of step with the game there)."""
    ignore, resync = game_api.sync_points(trace.meta)
    action = trace.steps[k].action
    if k in resync:
        return _quietly(lambda: game.resync(resync[k]["level"], resync[k]["score"], action), f"while replaying step {k}")
    if k in ignore:
        try:
            with contextlib.redirect_stdout(game_api.PrintCapture(0, 0)):
                return game.perform(action)
        except Exception:  # noqa: BLE001
            return None
    return _quietly(lambda: game.perform(action), f"while replaying step {k}")


def _steps_text(first: int, last: int) -> str:
    return f"step {first}" if first == last else f"steps {first}-{last}"


def _print_output(capture: game_api.PrintCapture, what: str = "the step") -> None:
    text = capture.getvalue()
    if not text.strip():
        print(f"your engine printed nothing during {what}")
        return
    kept, total = game_api.last_lines(text, lines=40, chars=3000)
    print(f"your engine printed during {what}" + (f" (the last 40 of {total} lines):" if total > 40 else ":"))
    print("\n".join("  " + line for line in kept.splitlines()))


def replay_step(i: int, state: Any = None, action: Any = None, *, level: int | None = None) -> tuple[Any, Any]:
    """Run step i on engine.py and explain it; returns copies (before, after) of your State.

    Loads engine.py fresh. The State before step i comes from replaying the recorded steps 0..i-1
    through the harness rules (RESET, level changes, WIN, GAME_OVER), their prints dropped; or it
    is `state` (a copy), when given. Then it applies step i's recorded action, or `action` (an id,
    (6, x, y) or an Action), and prints:
      - what engine.py printed during the step;
      - what the step changed in your state: sprites moved, changed, shown, hidden, added or
        removed (#k = state.sprites[k]), state.vars and the status;
      - with the recorded action, the comparison with the recording after step i, as run_tests
        explains a failing step: numbered regions that differ and your sprites in each.
    level=L starts at level L's start and replays only that level's steps before i, as
    run_tests(level=L) does; replay_step(e, level=L), e being the step that entered level L, compares
    your make_level(L) with the level's recorded start."""
    _visible(i)
    steps = trace.steps
    capture = game_api.PrintCapture()
    try:
        with contextlib.redirect_stdout(capture):
            module = _load_engine()
    except Exception:
        _print_output(capture, "loading engine.py")
        raise
    if not game_api.is_simple_engine(module):
        raise TypeError("engine.py must define make_level(n) and step(state, action)")
    game = game_api.GameRunner(module, steps[0].win_levels, steps[0].available_actions)
    recorded = action is None
    if state is not None:
        game.state = copy.deepcopy(state)
        game.level = int(getattr(state, "level", 0) or 0)
        game.score = steps[i - 1].levels_completed if 0 < i < len(steps) else game.level
        game.status = "NOT_FINISHED"
        start = "from the state you gave"
    elif level:
        entry = trace.level_starts()[level]
        if i < entry:
            raise ValueError(f"level {level} starts with step {entry}'s final frame; its steps are {entry + 1} onwards")
        capture = game_api.PrintCapture()
        with contextlib.redirect_stdout(capture):
            game.set_level(level)
        game.score = level
        if i == entry:
            print(f"replay_step({i}, level={level}): your make_level({level}) as the test starts it, against the level's recorded start")
            _print_output(capture, f"make_level({level})")
            after = game.state
            lines, _ = diff_report.describe_frames(steps[i].last, render_state(after), None, game_api.state_summary(after), crops=True, images=False)
            print(f"compared with the recording (step {i}'s final frame; expected = the original, got = yours):")
            print("\n".join(lines))
            return None, copy.deepcopy(after)
        for k in range(entry + 1, i):
            _replay_recorded(game, k)
        start = f"started at level {level}, after replaying {_steps_text(entry + 1, i - 1)}" if i > entry + 1 else f"at the start of level {level}"
    else:
        for k in range(i):
            _replay_recorded(game, k)
        start = f"after replaying {_steps_text(0, i - 1)}" if i else "fresh"
    act = _as_action(action, i)
    live = game.state
    alive = list(getattr(live, "sprites", None) or [])  # keeps the sprite ids unique until the summary after the step
    before_summary = game_api.state_summary(live) if live is not None else None
    before = copy.deepcopy(live)
    level_before = game.level
    print(f"replay_step({i}): {act} on your engine {start} (level {level_before})")
    resync = game_api.sync_points(trace.meta)[1].get(i) if recorded and state is None else None
    if resync is not None:
        print(f"step {i} is where your engine was put back in step with the game: the runner restarts level {resync['level']} "
              "here" + ("" if act.id == 0 else " without calling step()"))
    capture = game_api.PrintCapture()
    try:
        with contextlib.redirect_stdout(capture):
            obs = game.resync(resync["level"], resync["score"], act) if resync is not None else game.perform(act)
    except Exception:
        _print_output(capture)
        print("your engine raised an error in this step:")
        raise
    after_state = game.state
    after_summary = game_api.state_summary(after_state) if after_state is not None else None
    del alive
    _print_output(capture)
    print("what the step changed in your state:")
    print("\n".join("  " + line for line in diff_report.state_changes(before_summary, after_summary)))
    if not recorded:
        print(f"not compared with the recording: the action is not step {i}'s recorded one ({steps[i].action})")
    elif i < len(steps):
        fields = {k: obs[k] for k in _FIELDS}
        note = " (from the state you gave)" if state is not None else ""
        frames = obs["frames"]
        if all(fields[k] == getattr(steps[i], k) for k in _FIELDS) and len(frames) and np.array_equal(frames[-1], steps[i].last):
            print(f"your frame matches the recording after step {i}{note}")
            return before, copy.deepcopy(after_state)
        text, _ = tester.describe_step(
            steps[i], fields, obs["frames"], level_before, states={"before": before_summary, "after": after_summary}, crops=True,
            show_vars=False, before_frame=steps[i - 1].last if i else None,
        )
        print(f"compared with the recording after step {i}{note} (expected = the original, got = yours):")
        print("\n".join(text.splitlines()[1:]))
    return before, copy.deepcopy(after_state)


# --- Playing on the replica (the play-and-model agent) --------------------------------------------


def _runner(module: types.ModuleType) -> game_api.GameRunner:
    return game_api.GameRunner(module, trace.steps[0].win_levels, trace.steps[0].available_actions)


def _replay_all(game: game_api.GameRunner) -> None:
    for k in range(len(trace.steps)):
        _replay_recorded(game, k)


def _vars_text(state: Any, limit: int = 160) -> str:
    try:
        text = repr(dict(getattr(state, "vars", {}) or {}))
    except Exception:  # noqa: BLE001
        text = "?"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def state_now() -> Any:
    """Your replica's State now: engine.py loaded fresh, every step played so far replayed through it
    (the harness rules: RESET, level changes, WIN, GAME_OVER). Prints the level, the status and the
    vars; returns a copy of the State (its .level is the level being played). The game as your replica
    models it, not the game itself: the real frame after the last step is recording[-1].after."""
    capture = game_api.PrintCapture()
    try:
        with contextlib.redirect_stdout(capture):
            module = _load_engine()
    except Exception:
        _print_output(capture, "loading engine.py")
        raise
    if not game_api.is_simple_engine(module):
        raise TypeError("engine.py must define make_level(n) and step(state, action)")
    game = _runner(module)
    _replay_all(game)
    state = game.state
    n = len(trace.steps)
    if state is None:
        print(f"state_now(): your replica has no state after replaying steps 0-{n - 1} ({game.status})")
        return None
    print(f"state_now(): your replica after replaying steps 0-{n - 1}: level {game.level}, {game.status}, "
          f"{game.score} level(s) completed, {len(state.sprites)} sprites, vars={_vars_text(state)}")
    return copy.deepcopy(state)


# --- Support: what code on the replica ran, against the evidence (the play agent) ------------------------


def _support_map() -> dict[str, Any] | None:
    """The committed engine's support map (SUPPORT_PATH), or None when there is none yet."""
    if SUPPORT_PATH is None:
        return None
    try:
        return json.loads(Path(SUPPORT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class TracedRun:
    """What a `with traced() as run:` block ran on your replica: .lines (engine.py's lines that ran), .weakest (the
    least support of those lines: how many recorded steps ran it; 0 for an untested or new line; None: no map or no
    line), .untested, .thin and .new (lines), .unseparated (and/or conditions it evaluated that the recorded steps
    never separated). Printing it gives one paragraph; support(run) prints it."""

    def __init__(self) -> None:
        self.lines: list[int] = []
        self.weakest: int | None = None
        self.untested: list[int] = []
        self.thin: list[int] = []
        self.new: list[int] = []
        self.unseparated: list[dict[str, Any]] = []
        self.steps: int | None = None  # the recorded steps the map counts
        self.note = ""

    def __repr__(self) -> str:
        if self.note:
            return f"traced: {len(self.lines)} lines of engine.py ran; {self.note}"
        parts = [f"traced: {len(self.lines)} lines of engine.py ran; weakest support {self.weakest}"
                 + (f" of {self.steps} recorded steps" if self.steps is not None else "")]
        if self.untested:
            parts.append(f"untested (no recorded step ran them): {_support.line_ranges(self.untested)}")
        if self.new:
            parts.append(f"new since your last commit: {_support.line_ranges(self.new)}")
        if self.thin:
            parts.append(f"thin (1-{_support.THIN_SUPPORT - 1} steps): {_support.line_ranges(self.thin)}")
        if self.unseparated:
            parts.append("never separated: " + _support.unseparated_text(self.unseparated, 3))
        return "; ".join(parts)


_LAST_RUN: TracedRun | None = None


def _fill(run: TracedRun, inst: _support.Instrumented, lines: list[int], evaluated: list[list[int]]) -> None:
    run.lines = list(lines)
    smap = _support_map()
    if smap is None:
        run.note = "no support map yet (commit_moves makes one when it commits your replica)"
        return
    current = _support.remap(smap, ENGINE_PATH.read_text(encoding="utf-8"))
    ps = _support.path_support(current, lines)
    table = current["lines"]
    run.steps, run.weakest, run.thin = smap.get("steps"), ps["weakest"], ps["thin"]
    run.new = [line for line in lines if table.get(str(line), {}).get("new")]
    run.untested = [line for line in ps["untested"] if line not in run.new]
    known = _support.compound_lookup(current)
    ran = {k for k, _ in evaluated}
    groups = {inst.conds[k]["group"] for k in ran if k < len(inst.conds) and inst.conds[k]["group"] is not None}
    for g in sorted(groups):
        c = known.get((inst.groups[g]["line"], inst.groups[g]["text"]))
        if c is None or c["separated"]:
            continue
        loose = [k for k in inst.groups[g]["operands"] if inst.conds[k]["text"] in c["missing"]]
        if any(k in ran for k in loose):  # it relied on an operand that never decided the condition (support.path_support)
            run.unseparated.append({"line": c["line"], "text": c["text"], "missing": c["missing"]})


@contextlib.contextmanager
def traced() -> Any:
    """`with traced() as run:` around code that steps your replica (replica.step, replica.make_level): afterwards
    `run` says which lines of engine.py it ran and how many recorded steps support each (TracedRun), so plans can
    be ranked by evidence (run.weakest). Code outside the block, and state_now(), are not counted."""
    global _LAST_RUN
    run = TracedRun()
    _LAST_RUN = run
    module = replica._current()
    inst = _INSTRUMENTED.get(module)
    tracer = _support.Tracer(inst, module.__dict__) if inst is not None else None
    if tracer is None or not tracer.start():
        run.note = "nothing was traced (engine.py could not be instrumented, or a tracer is already running)"
        yield run
        return
    tracer.begin()
    try:
        yield run
    finally:
        tracer.end("run")
        tracer.stop()
        _fill(run, inst, tracer.executed.get("run", []), tracer.evaluated.get("run", []))


def support(run: TracedRun | None = None) -> None:
    """Print what the last `with traced()` block (or `run`) ran on your replica, against the evidence."""
    run = run or _LAST_RUN
    if run is None:
        print("support(): nothing traced yet; run your moves inside `with traced() as run:` first")
        return
    print(repr(run))


# --- Code for a frame's pieces --------------------------------------------------------------------


def _defined_names() -> set[str]:
    """Top-level names engine.py defines (assignments and defs), so generated code does not repeat them."""
    try:
        text = ENGINE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    return set(re.findall(r"^(?:def\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\()", text, re.M))


def _engine_values() -> tuple[dict[str, Any], str | None]:
    """engine.py's module-level values (for the pixel constants generated code can reuse), or why not."""
    if not ENGINE_PATH.exists():
        return {}, None
    try:
        with contextlib.redirect_stdout(game_api.PrintCapture(0, 0)):
            module = _load_engine()
    except (Exception, SystemExit) as exc:  # a broken engine.py only means nothing to reuse
        return {}, f"{type(exc).__name__}: {exc}"
    return dict(vars(module)), None


def _code_context() -> tuple[dict[str, Any], set[str]]:
    """What Pieces.code() takes from engine.py: its module-level values and the names it defines."""
    return _engine_values()[0], _defined_names()


def _level_code(level: int = 0) -> str:
    """The harness's opening (agent.OPENING_CODE), not a built-in: the code of level `level`'s first
    frame, recording[e].pieces_after.code() with e the step that entered it; prints what it assumed
    and found, then the code. Returns the code (a str) with .grid and .view set, for make_level."""
    starts = trace.level_starts()
    if level not in starts:
        raise ValueError(f"the recording never reaches level {level}; levels it reaches: {sorted(starts)}")
    e = starts[level]
    made = recording[e].pieces_after
    _, load_error = _engine_values()
    result = made._sprite_code()
    g = made.grid
    objects = sum(1 for p in made if p.role == "object" and not p.screen)
    print(f"recording[{e}].pieces_after.code(): sprites that draw level {level}'s first frame (recording[{e}].after).")
    print(f"Grid: {g.width}x{g.height} at scale {g.scale}, offset ({g.x_offset}, {g.y_offset}), border colour {g.border} "
          f"({COLOR_NAMES.get(g.border, '?')}), guessed from {g.frames} frame(s) of the level: {g.note}.")
    print(f"Found: background colour {result.background}, {objects} objects (one sprite per single-colour piece), "
          f"{len(made) - objects - 2} screen sprite(s) for displays or pixels off the grid.")
    print(f"Kinds: {kinds_summary(result)}.")
    if load_error:
        print(f"  (engine.py did not load, so its constants were not reused: {load_error})")
    print(f"Renders the frame exactly: {'yes' if result.exact else f'NO ({result.differing} pixels differ)'}.")
    print("Not the real sprites: one colour per sprite, nothing hidden or covered, transparency unknown, layers, tags "
          "and collidability are placeholders, identical-looking objects may be different kinds. The steps decide.")
    print()
    print(result.code)
    code = _LevelCode(result.code)
    code.grid, code.view = g.grid, (None if g.default_scale else g.scale)
    return code


class _LevelCode(str):
    grid: tuple[int, int] = (64, 64)
    view: int | None = None  # the grid's scale when it is not the default one (View(scale=...))


__all__ = [
    "Sprite", "Action", "View", "State", "recording", "read_file", "edit_file", "undo_edit", "render_state", "show_frames",
    "replay_step", "summarize_levels", "replica",
]
