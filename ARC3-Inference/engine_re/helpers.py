"""Analysis helpers preloaded in the agent's Python kernel.

Everything here is in the kernel's namespace: ``trace``, ``S`` (= trace.steps),
``np``, and the functions below. ``help_helpers()`` prints this overview.
"""

from __future__ import annotations

import contextlib
import copy
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from engine_re import auto_sprites as _auto, diff_report, game_api, tester
from engine_re.trace import Action, Step, Trace, new_game as _instantiate, perform

HEX = "0123456789abcdef"
COLOR_NAMES = {
    0: "white", 1: "light grey", 2: "grey", 3: "dark grey", 4: "darker grey", 5: "black",
    6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "light blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}

trace: Trace = None  # type: ignore[assignment]  # set by the kernel
S: list[Step] = []
ENGINE_PATH: Path = Path("engine.py")


def help_helpers() -> None:
    print(
        """Preloaded: np, trace, S (= trace.steps), ENGINE_PATH, Action, COLOR_NAMES.
S[i]: .action (.id, .x, .y), .frames (n,64,64 int8), .last (final frame), .n_frames,
      .state, .levels_completed, .win_levels, .available_actions
summary()                     overview of the trace (actions, levels, game overs, animations)
table(start=0, end=None)      one line per step: action, frames, state, level, pixels changed
before(i)                     final frame before step i (S[i-1].last)
show(grid, r0=0, c0=0, r1=63, c1=63)   print a region as hex digits with a ruler
show_step(i, region=None)     print step i: action, metadata, and the diff it caused
diff(a, b, limit=40)          describe the pixels that differ between two frames
changes(a, b)                 list of (row, col, old, new) for differing pixels
animation(i)                  per-frame diffs inside one multi-frame step
components(grid, color=None, bg=None)   connected regions: color, size, bbox (r0,c0,r1,c1)
colors(grid)                  Counter of colours
bbox(mask)                    (r0, c0, r1, c1) of a boolean mask
detect_grid(grid=None)        guess camera size, scale and offset from a frame (or several)
logical(grid, geom)           downsample a screen frame to the logical grid
screen_to_grid(x, y, geom) / grid_to_screen(gx, gy, geom)   coordinate conversion
find_steps(pred)              indices of steps with pred(step) True
level_starts()                level -> first step whose final frame shows it
new_game()                    load engine.py fresh and return a game to play (for debugging)
play(game, action_or_step)    perform an Action (or step index) on a game; returns observation
replay(n, start_level=None)   new_game() + play the actions of the first n steps (or a level's steps)
compare(i, obs)               diff an observation from play()/replay() against step i
engine()                      load engine.py fresh and return it as a module (engine().make_level(0), ...)
render(state)                 draw a State as a 64x64 frame, exactly as the harness does
game.state                    after play()/replay() on a make_level/step engine: the current State
check_contract()              run the contract tests on engine.py (run_tests runs them too)
before, after = try_step(i, state=None, action=None, level=None)
                              load engine.py fresh, replay steps 0..i-1 (or start from `state`), apply step i's
                              action (or `action`: an id or (6, x, y)) and print what engine.py printed, what the
                              step changed in your state (sprites as #k = state.sprites[k], vars, status) and,
                              for the recorded action, the comparison with the recording as run_tests shows it;
                              returns copies of your State before and after. print() in make_level/step to debug.
                              (It rebinds the name `before`; before(i) above is S[i-1].last.)
code = auto_sprites(level=0, grid=None, step=None, frame=None, merge=False)
                              print code for a sprite list that redraws a level's recorded start (or step i's
                              final frame, or a 64x64 frame) exactly: border, background, one sprite per
                              single-colour region (merge=True: per group of touching regions), identical
                              objects sharing a pixel constant and a tag, HUD as screen sprites. It guesses the
                              grid (pass grid=(w, h) to override) and checks the code renders the frame.
                              A starting point, NOT the real sprites: one colour per sprite, nothing hidden or
                              covered, transparency unknown, layers/tags/names/collidability are placeholders,
                              look-alike objects may differ; the steps decide."""
    )


# --- Display ------------------------------------------------------------------


def _hexrow(row: Iterable[int]) -> str:
    return "".join(HEX[v] if 0 <= v < 16 else ("." if v == -1 else "?") for v in row)


def show(grid: np.ndarray, r0: int = 0, c0: int = 0, r1: int | None = None, c1: int | None = None) -> None:
    """Print grid[r0:r1+1, c0:c1+1] as hex digits (colour 0-15; '.' = -1)."""
    grid = np.asarray(grid)
    r1 = grid.shape[0] - 1 if r1 is None else min(r1, grid.shape[0] - 1)
    c1 = grid.shape[1] - 1 if c1 is None else min(c1, grid.shape[1] - 1)
    cols = range(c0, c1 + 1)
    print("     " + "".join(str(c // 10 % 10) if c % 10 == 0 else " " for c in cols))
    print("     " + "".join(str(c % 10) for c in cols))
    for r in range(r0, r1 + 1):
        print(f"{r:>3}  " + _hexrow(grid[r, c0 : c1 + 1]))


def colors(grid: np.ndarray) -> Counter:
    return Counter(np.asarray(grid).ravel().tolist())


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def changes(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int, int, int]]:
    rows, cols = np.nonzero(np.asarray(a) != np.asarray(b))
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in zip(rows, cols)]


def diff(a: np.ndarray | None, b: np.ndarray | None, limit: int = 40) -> None:
    if a is None or b is None:
        print("(no frame to compare)")
        return
    a, b = np.asarray(a), np.asarray(b)
    mask = a != b
    n = int(mask.sum())
    if not n:
        print("identical")
        return
    trans = Counter(zip(a[mask].tolist(), b[mask].tolist())).most_common(8)
    print(f"{n} pixels differ, bbox (r0,c0,r1,c1)={bbox(mask)}; old->new colours: " + ", ".join(f"{x}->{y} x{k}" for (x, y), k in trans))
    if n <= limit:
        print("  (row, col, old, new):", changes(a, b))


def before(i: int) -> np.ndarray | None:
    return S[i - 1].last if i > 0 else None


def animation(i: int) -> None:
    step = S[i]
    prev = before(i)
    for k, frame in enumerate(step.frames):
        print(f"frame {k}: ", end="")
        diff(prev, frame, limit=12)
        prev = frame


def show_step(i: int, region: tuple[int, int, int, int] | None = None) -> None:
    step = S[i]
    print(repr(step), "available:", step.available_actions)
    print("change from previous final frame: ", end="")
    diff(before(i), step.last)
    if step.n_frames > 1:
        print(f"{step.n_frames} frames (animation):")
        animation(i)
    if region is not None and step.last is not None:
        show(step.last, *region)


def summary() -> None:
    acts = Counter(s.action.name for s in S)
    print(f"game {trace.game_id}: {len(S)} steps; actions used: {dict(acts)}")
    print("advertised available_actions:", sorted({a for s in S for a in s.available_actions}))
    print("level starts (level -> step whose final frame first shows it):", trace.level_starts())
    print("win_levels:", S[-1].win_levels, "| final state:", S[-1].state, "| levels completed:", S[-1].levels_completed)
    overs = [s.index for s in S if s.state == "GAME_OVER"]
    print(f"GAME_OVER steps ({len(overs)}):", overs[:40])
    multi = Counter(s.n_frames for s in S)
    print("frames per step (n_frames: count):", dict(sorted(multi.items())))
    still = [s.index for s in S if s.index and s.last is not None and before(s.index) is not None and np.array_equal(before(s.index), s.last)]
    print(f"steps that changed nothing ({len(still)}):", still[:40])


def table(start: int = 0, end: int | None = None) -> None:
    end = len(S) if end is None else min(end, len(S))
    for s in S[start:end]:
        prev = before(s.index)
        changed = "-" if prev is None or s.last is None else int((prev != s.last).sum())
        print(f"{s.index:>5}  {str(s.action):<22} frames={s.n_frames:<3} {s.state:<12} lvl={s.levels_completed} changed={changed}")


# --- Structure ----------------------------------------------------------------


def components(grid: np.ndarray, color: int | None = None, bg: int | None = None, diagonal: bool = False) -> list[dict[str, Any]]:
    """Connected single-colour regions (4-connected unless diagonal=True).

    color: only that colour. bg: skip that colour. Sorted by size, largest first."""
    from scipy import ndimage

    grid = np.asarray(grid)
    structure = np.ones((3, 3), int) if diagonal else None
    out = []
    for c in sorted(set(np.unique(grid).tolist())):
        if (color is not None and c != color) or (bg is not None and c == bg):
            continue
        labels, n = ndimage.label(grid == c, structure=structure)
        for k in range(1, n + 1):
            mask = labels == k
            out.append({"color": c, "size": int(mask.sum()), "bbox": bbox(mask)})
    return sorted(out, key=lambda d: -d["size"])


def detect_grid(grid: np.ndarray | list[np.ndarray] | None = None, top: int = 3) -> Any:
    """Guess the camera (logical grid width/height), scale and offset.

    The engine renders a width x height grid, scales it by
    s = min(64 // width, 64 // height) and centres it (offset = (64 - size*s) // 2),
    filling the rest with a letterbox colour. A candidate fits when (nearly) every
    s x s block inside the area is one colour and the border is (nearly) one colour;
    UI drawn later in screen pixels (e.g. a budget bar) can break some blocks.

    grid: one frame, or a list of frames from the SAME level (sizes can change
    between levels). With no argument, returns {level: best guesses} using the
    first frame of each level.

    Edge cells that have the letterbox colour make the size ambiguous: candidates
    with the same scale and a uniform border render identically, so any of them
    reproduces the screen until something is drawn in that border."""
    if grid is None:
        return {lvl: detect_grid(S[i].last, top) for lvl, i in trace.level_starts().items() if S[i].last is not None}
    frames = grid if isinstance(grid, list) else [grid]
    frames = [np.asarray(f) for f in frames]
    results = []
    for h in range(5, 65):
        for w in range(5, 65):
            s = min(64 // w, 64 // h)
            if s < 1:
                continue
            ox, oy = (64 - w * s) // 2, (64 - h * s) // 2
            uniform = total = inner_uniform = inner_total = border_main = border_total = 0
            outside = np.ones((64, 64), bool)
            outside[oy : oy + h * s, ox : ox + w * s] = False
            # Blocks touching the outer 2 screen pixels are where HUD bars usually sit.
            ys = oy + np.arange(h) * s
            xs = ox + np.arange(w) * s
            inner = ((ys >= 2) & (ys + s <= 62))[:, None] & ((xs >= 2) & (xs + s <= 62))[None, :]
            for f in frames:
                area = f[oy : oy + h * s, ox : ox + w * s].reshape(h, s, w, s)
                blocks = area.transpose(0, 2, 1, 3).reshape(h, w, s * s)
                ok = (blocks == blocks[:, :, :1]).all(axis=2)
                uniform += int(ok.sum())
                total += h * w
                inner_uniform += int(ok[inner].sum())
                inner_total += int(inner.sum())
                border = f[outside]
                if border.size:
                    border_main += int(np.bincount(border.astype(np.int64) % 16).max())
                    border_total += int(border.size)
            letterbox = round(border_main / border_total, 4) if border_total else 1.0
            inner_score = round(inner_uniform / inner_total, 4) if inner_total else round(uniform / total, 4)
            results.append({"width": w, "height": h, "scale": s, "x_offset": ox, "y_offset": oy,
                            "uniform_blocks": round(uniform / total, 4), "uniform_inner_blocks": inner_score,
                            "letterbox_uniform": letterbox})
    # Best: uniform blocks away from the screen edge and a (nearly) single-colour
    # border (the HUD may sit in it), then the largest scale, then the largest grid.
    results.sort(key=lambda r: (r["uniform_inner_blocks"] >= 0.98 and r["letterbox_uniform"] >= 0.85, r["scale"],
                                r["uniform_inner_blocks"], r["width"] * r["height"]), reverse=True)
    return results[:top]


def logical(grid: np.ndarray, geom: dict[str, Any]) -> np.ndarray:
    """Downsample a 64x64 frame to the logical grid (top-left pixel of each block)."""
    s, ox, oy = geom["scale"], geom["x_offset"], geom["y_offset"]
    return np.asarray(grid)[oy : oy + geom["height"] * s : s, ox : ox + geom["width"] * s : s].copy()


def screen_to_grid(x: int, y: int, geom: dict[str, Any]) -> tuple[int, int] | None:
    gx, gy = (x - geom["x_offset"]) // geom["scale"], (y - geom["y_offset"]) // geom["scale"]
    if 0 <= gx < geom["width"] and 0 <= gy < geom["height"] and x >= geom["x_offset"] and y >= geom["y_offset"]:
        return gx, gy
    return None


def grid_to_screen(gx: int, gy: int, geom: dict[str, Any]) -> tuple[int, int]:
    return gx * geom["scale"] + geom["x_offset"], gy * geom["scale"] + geom["y_offset"]


def find_steps(pred: Callable[[Step], bool]) -> list[int]:
    return [s.index for s in S if pred(s)]


def level_starts() -> dict[int, int]:
    return trace.level_starts()


# --- Running your engine in this kernel ---------------------------------------


def engine() -> types.ModuleType:
    """Load engine.py fresh and return it as a module."""
    name = "candidate_engine_dev"
    source = ENGINE_PATH.read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE_PATH)
    sys.modules[name] = module
    exec(compile(source, str(ENGINE_PATH), "exec", dont_inherit=True), module.__dict__)
    return module


def render(state: Any) -> np.ndarray:
    """Draw a State as a 64x64 frame, exactly as the harness does."""
    return game_api.render(state)


def check_contract() -> None:
    """Run the contract tests on engine.py and print the result."""
    source = ENGINE_PATH.read_text(encoding="utf-8")
    meta_levels = sorted(level for level in trace.level_starts() if level < S[0].win_levels)
    results = game_api.contract_checks(engine(), source, levels=meta_levels, available_actions=list(S[0].available_actions))
    print(game_api.describe_contract(results))


def new_game() -> Any:
    """Load engine.py fresh and return a game to play: a GameRunner for a make_level/step
    engine (its .state is the current State), or an instance of an ARCBaseGame subclass."""
    from arcengine import ARCBaseGame

    module = engine()
    if game_api.is_simple_engine(module):
        return game_api.GameRunner(module, S[0].win_levels, S[0].available_actions)
    name = module.__name__
    classes = [o for o in vars(module).values() if isinstance(o, type) and issubclass(o, ARCBaseGame) and o is not ARCBaseGame and o.__module__ == name]
    if not classes:
        raise TypeError("engine.py defines no ARCBaseGame subclass")
    leaves = [c for c in classes if not any(o is not c and issubclass(o, c) for o in classes)]
    return _instantiate(leaves[-1])


def play(game: Any, action: Action | int) -> dict[str, Any]:
    """Perform an Action (or the action of step index i) and return the observation
    dict: frames, state, levels_completed, win_levels, available_actions."""
    if isinstance(action, (int, np.integer)):
        action = S[int(action)].action
    if isinstance(game, game_api.GameRunner):
        return game.perform(action)
    return perform(game, action)


def replay(n: int | None = None, start_level: int | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Fresh engine; play the first n steps' actions (all if None). With
    start_level=L, call set_level(L) first and play the steps after the one that
    entered level L (n counts from there). Returns (game, observations); the
    observation of trace step i is observations[i] (or observations[i - first])."""
    game = new_game()
    steps = S
    if start_level:
        entry = trace.level_starts()[start_level]
        game.set_level(start_level)
        if isinstance(game, game_api.GameRunner):
            game.score = start_level
        else:
            game._score = start_level
        steps = S[entry + 1 :]
    if n is not None:
        steps = steps[:n]
    return game, [play(game, s.action) for s in steps]


def compare(i: int, obs: dict[str, Any]) -> None:
    """Diff an observation (from play/replay) against trace step i."""
    step = S[i]
    for name in ("state", "levels_completed", "win_levels", "available_actions"):
        if getattr(step, name) != obs[name]:
            print(f"{name}: expected {getattr(step, name)}, got {obs[name]}")
    if step.n_frames != len(obs["frames"]):
        print(f"frames: real {step.n_frames}, yours {len(obs['frames'])} (only the final frame is compared)")
    print("final frame: ", end="")
    diff(step.last, obs["frames"][-1] if len(obs["frames"]) else None)


# --- Reproducing and explaining one step (the command run_tests prints) -----------------------

_FIELDS = ("state", "levels_completed", "win_levels", "available_actions")


def _as_action(action: Any, i: int) -> Action:
    """The action to apply: step i's recorded one, or an id, (6, x, y), a dict or an Action."""
    if action is None:
        return S[i].action
    if isinstance(action, Action):
        return action
    if isinstance(action, (int, np.integer)):
        return Action(int(action))
    if isinstance(action, (tuple, list)):
        return Action(int(action[0]), *(int(v) for v in action[1:3]))
    if isinstance(action, dict):
        return Action.from_json(action)
    if hasattr(action, "id"):
        return Action(int(action.id), getattr(action, "x", None), getattr(action, "y", None))
    raise TypeError(f"action must be an action id, (6, x, y) or an Action, not {type(action).__name__}")


def _quietly(fn: Callable[[], Any], what: str) -> Any:
    """Run fn with your engine's prints dropped; on an error say where it happened, then raise."""
    try:
        with contextlib.redirect_stdout(game_api.PrintCapture(0, 0)):
            return fn()
    except Exception:
        print(f"your engine raised an error {what}:")
        raise


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


def try_step(i: int, state: Any = None, action: Any = None, *, level: int | None = None) -> tuple[Any, Any]:
    """Run step i on your engine and explain it; returns copies (before, after) of your State.

    Loads engine.py fresh. The State before step i comes from replaying the recorded steps 0..i-1
    through the harness rules (RESET, level changes, WIN, GAME_OVER), their prints dropped; or it
    is `state` (a copy), when given. Then it applies step i's recorded action, or `action` (an id,
    (6, x, y) or an Action), and prints:
      - what engine.py printed during the step (print() freely in make_level and step);
      - what the step changed in your state: sprites moved, changed, shown, hidden, added or
        removed (#k = state.sprites[k]), state.vars and the status;
      - with the recorded action, the comparison with the recording after step i, as run_tests
        explains a failing step: numbered regions that differ and your sprites in each.
    level=L starts at level L's start and replays only that level's steps before i, as
    run_tests(level=L) does; try_step(e, level=L), e being the step that entered level L, compares
    your make_level(L) with the level's recorded start."""
    capture = game_api.PrintCapture()
    try:
        with contextlib.redirect_stdout(capture):
            module = engine()
    except Exception:
        _print_output(capture, "loading engine.py")
        raise
    if not game_api.is_simple_engine(module):
        return _try_arcengine_step(i, action)
    game = game_api.GameRunner(module, S[0].win_levels, S[0].available_actions)
    recorded = action is None
    if state is not None:
        game.state = copy.deepcopy(state)
        game.level = int(getattr(state, "level", 0) or 0)
        game.score = S[i - 1].levels_completed if 0 < i < len(S) else game.level
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
            print(f"try_step({i}, level={level}): your make_level({level}) as the test starts it, against the level's recorded start")
            _print_output(capture, f"make_level({level})")
            after = game.state
            lines, _ = diff_report.describe_frames(S[i].last, render(after), None, game_api.state_summary(after), crops=True, images=False)
            print(f"compared with the recording (step {i}'s final frame; expected = the original, got = yours):")
            print("\n".join(lines))
            return None, copy.deepcopy(after)
        for k in range(entry + 1, i):
            _quietly(lambda k=k: game.perform(S[k].action), f"while replaying step {k}")
        start = f"started at level {level}, after replaying {_steps_text(entry + 1, i - 1)}" if i > entry + 1 else f"at the start of level {level}"
    else:
        for k in range(i):
            _quietly(lambda k=k: game.perform(S[k].action), f"while replaying step {k}")
        start = f"after replaying {_steps_text(0, i - 1)}" if i else "fresh"
    act = _as_action(action, i)
    live = game.state
    alive = list(getattr(live, "sprites", None) or [])  # keeps the sprite ids unique until the summary after the step
    before_summary = game_api.state_summary(live) if live is not None else None
    before = copy.deepcopy(live)
    level_before = game.level
    print(f"try_step({i}): {act} on your engine {start} (level {level_before})")
    capture = game_api.PrintCapture()
    try:
        with contextlib.redirect_stdout(capture):
            obs = game.perform(act)
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
        print(f"not compared with the recording: the action is not step {i}'s recorded one ({S[i].action})")
    elif i < len(S):
        fields = {k: obs[k] for k in _FIELDS}
        text, _ = tester.describe_step(
            S[i], fields, obs["frames"], level_before, states={"before": before_summary, "after": after_summary}, crops=True,
            show_vars=False,
        )
        note = " (from the state you gave)" if state is not None else ""
        print(f"compared with the recording after step {i}{note} (expected = the original, got = yours):")
        print("\n".join(text.splitlines()[1:]))
    return before, copy.deepcopy(after_state)


def _try_arcengine_step(i: int, action: Any) -> tuple[None, None]:
    """try_step for an ARCBaseGame engine: the prints and the comparison, without sprites."""
    game = new_game()
    for k in range(i):
        _quietly(lambda k=k: play(game, S[k].action), f"while replaying step {k}")
    capture = game_api.PrintCapture()
    with contextlib.redirect_stdout(capture):
        obs = play(game, _as_action(action, i))
    _print_output(capture)
    if action is None:
        text, _ = tester.describe_step(S[i], {k: obs[k] for k in _FIELDS}, obs["frames"], tester.levels_before(trace)[i])
        print("\n".join(text.splitlines()[1:]))
    return None, None


class GeneratedCode(str):
    """The code auto_sprites printed (a str), with .exact, .grid, .scale and .info; its repr stays short."""

    exact: bool = False
    grid: tuple[int, int] = (64, 64)
    scale: int = 1
    info: Any = None

    def __repr__(self) -> str:
        return f"<generated code: {len(self)} characters; print() it, or write it into engine.py>"


def _level_frames(level: int, limit: int = 60) -> list[np.ndarray]:
    """Final frames of the recorded steps that show level `level` (at most `limit`, evenly spaced)."""
    frames = [s.last for s in S if s.levels_completed == level and s.last is not None and s.state != "WIN"]
    if len(frames) > limit:
        frames = [frames[round(k * (len(frames) - 1) / (limit - 1))] for k in range(limit)]
    return frames


def auto_sprites(
    level: int = 0,
    grid: tuple[int, int] | None = None,
    *,
    step: int | None = None,
    frame: np.ndarray | None = None,
    scale: int | None = None,
    merge: bool = False,
    quiet: bool = False,
) -> GeneratedCode:
    """Print (and return) Python code for a sprite list that redraws a recorded frame exactly: by
    default the start of `level`; step=i: step i's final frame; frame=a 64x64 array. A starting point
    for make_level, NOT the game's real sprites.

    It guesses the grid (grid=(w, h), and scale=s if not the default fit, override it), then makes a
    border screen sprite, a background sprite of the grid's most common colour, one sprite per
    4-connected single-colour region (merge=True: per group of touching non-background regions,
    multi-coloured; default off, since touching objects would fuse), sharing a pixel constant and a
    placeholder tag among identical objects, and screen sprites for what the grid cannot draw
    (HUD). It runs the code and says whether it renders the frame exactly."""


    if frame is not None:
        source_frame, evidence = np.asarray(frame), [np.asarray(frame)]
        what, args, prefix, function = "the frame you gave", "(frame=...)", "F_", "frame_sprites"
    elif step is not None:
        source_frame = S[step].last
        evidence = [source_frame] + _level_frames(S[step].levels_completed)
        what, args, prefix, function = f"step {step}'s final frame", f"(step={step})", f"S{step}_", f"step_{step}_sprites"
    else:
        starts = trace.level_starts()
        if level not in starts:
            raise ValueError(f"the recording never reaches level {level}; levels it reaches: {sorted(starts)}")
        source_frame = S[starts[level]].last
        evidence = [source_frame] + _level_frames(level)
        what = f"the recorded start of level {level} (step {starts[level]}'s final frame)"
        args, prefix, function = f"(level={level})", f"L{level}_", f"level_{level}_sprites"
    if source_frame is None:
        raise ValueError("that step has no frame")
    if grid is None:
        guess = _auto.guess_grid(evidence)
        assumed = f"assumed {guess.width}x{guess.height} at scale {guess.scale}"
        how = (f"guessed from {guess.frames} frame(s): {guess.note}. Conservative: a scale above 1 is taken only if every "
               "block is one colour in all of them, yet a scale-1 game whose objects align on a coarser lattice can still fool it")
    else:
        w, h = int(grid[0]), int(grid[1])
        s, ox, oy = game_api.geometry((w, h), scale)
        ring = np.concatenate([source_frame[0], source_frame[-1], source_frame[:, 0], source_frame[:, -1]])
        guess = _auto.GridGuess(w, h, s, ox, oy, int(np.bincount(ring.astype(np.int64) % 16).argmax()), 1, "as given")
        assumed, how = f"grid {w}x{h} at scale {s} (as given)", ""
    result = _auto.sprite_code(source_frame, guess, merge=merge, prefix=prefix, function=function, source=args)
    code = GeneratedCode(result.code)
    code.exact, code.grid, code.scale, code.info = result.exact, guess.grid, guess.scale, result
    if quiet:
        return code
    region = "one sprite per group of touching regions (merge=True)" if merge else "one per single-colour region; merge=True joins touching ones"
    print(f"auto_sprites{args}: {what}.")
    print(f"Grid: {assumed}, offset ({guess.x_offset}, {guess.y_offset}), border colour {guess.border} "
          f"({COLOR_NAMES.get(guess.border, '?')}); pass grid=(w, h) if that is wrong.")
    if how:
        print(f"  ({how}.)")
    print(f"Found: background colour {result.background}, {result.objects} objects of {result.shapes} shapes ({region}), "
          f"{result.hud} screen sprite(s) for the HUD or pixels off the grid.")
    print(f"Renders the frame exactly: {'yes' if result.exact else f'NO ({result.differing} pixels differ)'}.")
    print("Not the real sprites: one colour per sprite, nothing hidden or covered, transparency unknown, layers, tags, "
          "names and collidability are placeholders, identical-looking objects may be different kinds. The steps decide.")
    print()
    if len(code) <= 6000:
        print(code)
    else:
        print(code[:5000] + f"\n# ... {len(code) - 5000} more characters: the returned string holds all of it "
              "(code = auto_sprites(...); print(code[5000:]))")
    return code


