"""Analysis helpers preloaded in the agent's Python kernel.

Everything here is in the kernel's namespace: ``trace``, ``S`` (= trace.steps),
``np``, and the functions below. ``help_helpers()`` prints this overview.
"""

from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

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
new_game()                    load engine.py fresh and return an instance (for debugging)
play(game, action_or_step)    perform an Action (or step index) on a game; returns observation
replay(n, start_level=None)   new_game() + play the actions of the first n steps (or a level's steps)
compare(i, obs)               diff an observation from play()/replay() against step i"""
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


def new_game() -> Any:
    """Load engine.py fresh and return a new instance of its game class."""
    from arcengine import ARCBaseGame

    name = "candidate_engine_dev"
    source = ENGINE_PATH.read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE_PATH)
    sys.modules[name] = module
    exec(compile(source, str(ENGINE_PATH), "exec"), module.__dict__)
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
