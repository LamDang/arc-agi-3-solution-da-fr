"""The simple game interface: what the agent writes, and how the harness runs it.

The agent's engine.py is a plain module, not an arcengine game:

    make_level(n) -> State      the state at the start of level n
    step(state, action)         apply one action to the state, in place

``FIXED_INTERFACE`` is the block of code at the top of every engine.py that
defines ``Sprite``, ``Action`` and ``State``; the agent must not edit it, and
the contract tests check that it is unchanged. Everything else the real
engines do around those two functions is the harness's job here:

- ``render(state)`` draws a State on the 64x64 screen;
- ``GameRunner`` reproduces arcengine's episode rules (RESET, level changes,
  WIN, GAME_OVER, what is returned after the game ended) and turns screen
  clicks into grid cells, returning observations shaped like the real ones;
- ``contract_checks`` is a small unit-test suite for the module's contract
  (interface untouched, valid states, fresh state on every make_level call,
  every advertised action accepted, determinism). Matching the recording
  (the acceptance test) is the replay in ``tester.replay_test``.

Only the final frame of each action is produced, so engines written this way
are meant to be scored with ``match="final"``.
"""

from __future__ import annotations

import copy
import random
import sys
import types
from typing import Any

import numpy as np

BEGIN_MARKER = "# ==== FIXED INTERFACE: DO NOT EDIT. The harness relies on this block and the tests check it. ===="
END_MARKER = "# ==== END OF FIXED INTERFACE ===="

FIXED_INTERFACE = BEGIN_MARKER + '''
#
# How the harness draws a State (it shows only the final state of each action):
#   1. Start from a 64x64 screen filled with colour 5.
#   2. Draw each visible sprite, lowest layer first; sprites on the same layer
#      are drawn in list order, so later ones end up on top.
#      - A grid sprite (screen=False) is placed on the logical grid. The grid
#        (w, h) = state.grid is scaled up by s = min(64 // w, 64 // h) and centred:
#        grid cell (gx, gy) fills the s x s screen block whose top-left pixel is
#        (ox + gx * s, oy + gy * s), with ox = (64 - w * s) // 2, oy = (64 - h * s) // 2.
#        Parts of a grid sprite outside the grid are not drawn.
#      - A screen sprite (screen=True) is placed in screen pixels, unscaled. Use it
#        for things drawn at screen resolution, such as a budget bar in the border.
#      - Pixels -1 (transparent) and -2 (invisible but solid) are not drawn.
#   So the border colour is a 64x64 screen sprite on the lowest layer, and the
#   background is a grid-sized sprite on the layer above it.
#
# Coordinates: x is the column, y is the row, (0, 0) is the top-left corner.

from dataclasses import dataclass, field


@dataclass(eq=False)
class Sprite:
    """A rectangle of pixels: colours 0-15, -1 transparent, -2 invisible but solid."""

    pixels: list  # list of rows, each a list of ints; edit or replace freely
    x: int = 0  # column of the top-left pixel (grid cells, or screen pixels if screen=True)
    y: int = 0  # row of the top-left pixel
    layer: int = 0  # higher layers are drawn on top
    screen: bool = False  # True: screen pixels, not scaled; False: logical grid
    visible: bool = True  # an invisible sprite is not drawn but is still found by at() and overlapping()
    tags: tuple = ()  # labels to find sprites again, e.g. state.by_tag("wall")
    name: str = ""

    @property
    def width(self) -> int:
        return len(self.pixels[0]) if len(self.pixels) else 0

    @property
    def height(self) -> int:
        return len(self.pixels)

    def solid(self) -> set:
        """The (x, y) positions this sprite covers with a pixel other than -1."""
        return {(self.x + c, self.y + r) for r, row in enumerate(self.pixels) for c, v in enumerate(row) if v != -1}

    def contains(self, x: int, y: int) -> bool:
        """Is (x, y) inside this sprite's bounding box?"""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass
class Action:
    """One action given to step(). RESET (id 0) is handled by the harness and never reaches step()."""

    id: int  # 1 up, 2 down, 3 left, 4 right, 5 interact, 6 click, 7 undo
    x: int = 0  # for a click: the screen pixel column (0-63)
    y: int = 0  # for a click: the screen pixel row (0-63)
    cell: tuple | None = None  # for a click: the grid cell (gx, gy) under it, None if outside the grid


@dataclass(eq=False)
class State:
    """Everything about the current level. make_level() creates it; step() changes it in place."""

    grid: tuple  # logical grid size (width, height), each 1-64
    sprites: list = field(default_factory=list)  # everything drawn: border, background, objects, HUD
    vars: dict = field(default_factory=dict)  # hidden state: budget, counters, modes, what is selected, ...
    status: str = "playing"  # step() sets "level_solved" or "game_over"
    level: int = 0  # index of the current level, set by the harness

    def by_tag(self, tag: str) -> list:
        """Sprites carrying this tag, in list order."""
        return [s for s in self.sprites if tag in s.tags]

    def by_name(self, name: str):
        """The first sprite with this name, or None."""
        return next((s for s in self.sprites if s.name == name), None)

    def at(self, x: int, y: int, screen: bool = False) -> list:
        """Sprites with a pixel other than -1 at (x, y), topmost first."""
        hits = [(s.layer, i, s) for i, s in enumerate(self.sprites) if s.screen == screen and (x, y) in s.solid()]
        return [s for _, _, s in sorted(hits, key=lambda h: (h[0], h[1]), reverse=True)]

    def overlapping(self, sprite: Sprite) -> list:
        """Other sprites in the same space sharing a position where both have a pixel other than -1."""
        mine = sprite.solid()
        return [s for s in self.sprites if s is not sprite and s.screen == sprite.screen and mine & s.solid()]

    def add(self, sprite: Sprite) -> Sprite:
        self.sprites.append(sprite)
        return sprite

    def remove(self, sprite: Sprite) -> None:
        self.sprites = [s for s in self.sprites if s is not sprite]


def rotate_cw(pixels: list, turns: int = 1) -> list:
    """The pixel grid rotated clockwise by 90 degrees, `turns` times."""
    out = [list(row) for row in pixels]
    for _ in range(turns % 4):
        out = [list(row) for row in zip(*out[::-1])]
    return out


def flip_lr(pixels: list) -> list:
    """The pixel grid mirrored left-right."""
    return [list(row)[::-1] for row in pixels]


def flip_ud(pixels: list) -> list:
    """The pixel grid mirrored top-bottom."""
    return [list(row) for row in pixels[::-1]]


def scale_up(pixels: list, k: int) -> list:
    """Each pixel repeated as a k x k block."""
    return [[v for v in row for _ in range(k)] for row in pixels for _ in range(k)]


''' + END_MARKER

STATUSES = ("playing", "level_solved", "game_over")


_CANONICAL: types.ModuleType | None = None


def canonical() -> types.ModuleType:
    """A module holding the interface classes and helpers, defined from FIXED_INTERFACE."""
    global _CANONICAL
    if _CANONICAL is None:
        module = types.ModuleType("engine_re_fixed_interface")
        sys.modules[module.__name__] = module
        # dont_inherit: compile the block as engine.py would, without this file's __future__ imports.
        exec(compile(FIXED_INTERFACE, "<fixed interface>", "exec", dont_inherit=True), module.__dict__)
        _CANONICAL = module
    return _CANONICAL


def is_simple_engine(module: Any) -> bool:
    return callable(getattr(module, "make_level", None)) and callable(getattr(module, "step", None))


def extract_interface(source: str) -> str | None:
    start = source.find(BEGIN_MARKER)
    end = source.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        return None
    return source[start : end + len(END_MARKER)]


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


# --- Rendering ------------------------------------------------------------------


def geometry(grid: tuple) -> tuple[int, int, int]:
    """Scale and top-left offset of the logical grid on the 64x64 screen."""
    w, h = int(grid[0]), int(grid[1])
    s = min(64 // w, 64 // h)
    return s, (64 - w * s) // 2, (64 - h * s) // 2


def to_grid(grid: tuple, x: int, y: int) -> tuple[int, int] | None:
    """The grid cell under screen pixel (x, y), or None outside the grid."""
    w, h = int(grid[0]), int(grid[1])
    s, ox, oy = geometry(grid)
    if not (ox <= x < ox + w * s and oy <= y < oy + h * s):
        return None
    return (x - ox) // s, (y - oy) // s


def _blit(screen: np.ndarray, px: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    """Draw px with its top-left at (top, left), clipped to rows < bottom and columns < right of the region
    starting at the screen origin; pixels < 0 are skipped."""
    h, w = px.shape
    r0, c0 = max(0, -top), max(0, -left)
    r1, c1 = min(h, bottom - top), min(w, right - left)
    if r0 >= r1 or c0 >= c1:
        return
    sub = px[r0:r1, c0:c1]
    region = screen[top + r0 : top + r1, left + c0 : left + c1]
    mask = sub >= 0
    region[mask] = sub[mask]


def render(state: Any) -> np.ndarray:
    """Draw a State as a 64x64 frame, following the rules in FIXED_INTERFACE."""
    w, h = int(state.grid[0]), int(state.grid[1])
    s, ox, oy = geometry((w, h))
    screen = np.full((64, 64), 5, np.int16)
    sprites = list(state.sprites)
    for i in sorted(range(len(sprites)), key=lambda i: (sprites[i].layer, i)):
        sprite = sprites[i]
        if not sprite.visible:
            continue
        px = np.asarray(sprite.pixels, dtype=np.int16)
        if px.ndim != 2 or px.size == 0:
            continue
        x, y = int(sprite.x), int(sprite.y)
        if sprite.screen:
            _blit(screen, px, y, x, 64, 64)
            continue
        # Clip to the grid in grid cells, then scale and place on the screen.
        r0, c0 = max(0, -y), max(0, -x)
        r1, c1 = min(px.shape[0], h - y), min(px.shape[1], w - x)
        if r0 >= r1 or c0 >= c1:
            continue
        sub = px[r0:r1, c0:c1]
        if s > 1:
            sub = np.repeat(np.repeat(sub, s, axis=0), s, axis=1)
        _blit(screen, sub, oy + (y + r0) * s, ox + (x + c0) * s, 64, 64)
    return screen.astype(np.int8)


# --- Running a game -----------------------------------------------------------------


class GameRunner:
    """Plays an engine.py module with arcengine's episode rules (ONLY_RESET_LEVELS=true).

    - RESET rebuilds the current level with make_level (after a WIN: level 0, score 0).
    - Any other action after GAME_OVER or WIN returns no frame, levels_completed 0
      and win_levels 0, like arcengine.
    - Otherwise step(state, action) runs. "level_solved" adds one completed level and
      shows the next level's start (after the last level: WIN, showing the state as is);
      "game_over" ends the game.
    - win_levels and available_actions come from the recording.
    """

    def __init__(self, module: Any, win_levels: int, available_actions: list[int]):
        self.module = module
        self.win_levels = int(win_levels)
        self.available_actions = [int(a) for a in available_actions]
        self.action_cls = getattr(module, "Action", None) or canonical().Action
        self.level = 0
        self.score = 0
        self.status = "NOT_PLAYED"
        self.state: Any = None

    def fresh(self, level: int) -> Any:
        state = self.module.make_level(level)
        state.level = level
        state.status = "playing"
        return state

    def set_level(self, level: int) -> None:
        self.level = level
        self.state = self.fresh(level)

    def _observation(self, frames: list[np.ndarray]) -> dict[str, Any]:
        return {
            "frames": np.stack(frames).astype(np.int8) if frames else np.zeros((0, 64, 64), np.int8),
            "state": self.status,
            "levels_completed": self.score,
            "win_levels": self.win_levels,
            "available_actions": list(self.available_actions),
        }

    def perform(self, action: Any) -> dict[str, Any]:
        """Perform an action given as (id, x, y), e.g. engine_re.trace.Action; returns an observation."""
        action_id = int(action.id)
        if action_id == 0:
            if self.status == "WIN":
                self.level, self.score = 0, 0
            self.state = self.fresh(self.level)
            self.status = "NOT_FINISHED"
            return self._observation([render(self.state)])
        if self.status in ("GAME_OVER", "WIN"):
            obs = self._observation([])
            obs["levels_completed"], obs["win_levels"] = 0, 0
            return obs
        if self.state is None:
            self.state = self.fresh(self.level)
        self.status = "NOT_FINISHED"
        x, y = int(getattr(action, "x", 0) or 0), int(getattr(action, "y", 0) or 0)
        cell = to_grid(self.state.grid, x, y) if action_id == 6 else None
        self.module.step(self.state, self.action_cls(id=action_id, x=x, y=y, cell=cell))
        outcome = self.state.status
        if outcome not in STATUSES:
            raise ValueError(f"state.status must be one of {STATUSES}, got {outcome!r}")
        if outcome == "level_solved":
            self.score += 1
            if self.level + 1 >= self.win_levels:
                self.status = "WIN"
            else:
                self.level += 1
                self.state = self.fresh(self.level)
        elif outcome == "game_over":
            self.status = "GAME_OVER"
        return self._observation([render(self.state)])


# --- Contract tests -----------------------------------------------------------------


def _problems_in_state(state: Any, sprite_cls: Any, state_cls: Any) -> list[str]:
    problems: list[str] = []
    if state_cls is not None and not isinstance(state, state_cls):
        return [f"make_level returned {type(state).__name__}, not State"]
    grid = getattr(state, "grid", None)
    if not (isinstance(grid, (tuple, list)) and len(grid) == 2 and all(isinstance(v, int) and 1 <= v <= 64 for v in grid)):
        problems.append(f"state.grid must be (width, height) with ints 1-64, got {grid!r}")
    sprites = getattr(state, "sprites", None)
    if not isinstance(sprites, list):
        return problems + [f"state.sprites must be a list, got {type(sprites).__name__}"]
    if not isinstance(getattr(state, "vars", None), dict):
        problems.append("state.vars must be a dict")
    if getattr(state, "status", None) not in STATUSES:
        problems.append(f"state.status must be one of {STATUSES}, got {getattr(state, 'status', None)!r}")
    for k, sprite in enumerate(sprites):
        label = f"sprite {k}" + (f" ({sprite.name!r})" if getattr(sprite, "name", "") else "")
        if sprite_cls is not None and not isinstance(sprite, sprite_cls):
            problems.append(f"{label} is a {type(sprite).__name__}, not Sprite")
            continue
        try:
            px = np.asarray(sprite.pixels)
        except Exception:  # noqa: BLE001
            problems.append(f"{label}: pixels are not a rectangular grid")
            continue
        if px.ndim != 2 or px.size == 0 or not np.issubdtype(px.dtype, np.integer):
            problems.append(f"{label}: pixels must be a non-empty rectangular grid of ints")
        elif px.min() < -2 or px.max() > 15:
            problems.append(f"{label}: pixel values must be -2..15, got {int(px.min())}..{int(px.max())}")
        for attr in ("x", "y", "layer"):
            if not isinstance(getattr(sprite, attr, None), (int, np.integer)) or isinstance(getattr(sprite, attr), bool):
                problems.append(f"{label}: {attr} must be an int")
        if len(problems) > 8:
            break
    return problems


def _scribble(state: Any) -> None:
    """Change everything mutable in a state, to reveal objects shared between make_level calls."""

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (list, dict)):
                    walk(item)
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    value[i] = item + 1
            value.append(None)
        elif isinstance(value, dict):
            for key in list(value):
                if isinstance(value[key], (list, dict)):
                    walk(value[key])
                elif isinstance(value[key], (int, float)) and not isinstance(value[key], bool):
                    value[key] = value[key] + 1
            value["__probe__"] = 1

    for sprite in list(state.sprites):
        sprite.x += 1
        sprite.y += 1
        sprite.visible = not sprite.visible
        if isinstance(sprite.pixels, np.ndarray):
            sprite.pixels[...] = 0
        else:
            for row in sprite.pixels:
                if isinstance(row, list):
                    for c in range(len(row)):
                        row[c] = 0
                elif isinstance(row, np.ndarray):
                    row[...] = 0
    walk(state.vars)
    if state.sprites:
        state.sprites.append(type(state.sprites[0])([[0]]))


def contract_checks(
    module: Any,
    source: str,
    *,
    levels: list[int],
    available_actions: list[int],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """The contract tests: a list of {name, ok, detail}. Each check is independent."""
    results: list[dict[str, Any]] = []
    sprite_cls, state_cls, action_cls = (getattr(module, n, None) for n in ("Sprite", "State", "Action"))
    action_cls = action_cls or canonical().Action
    levels = list(levels) or [0]

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            results.append({"name": name, "ok": not detail, "detail": detail or ""})
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "ok": False, "detail": f"raised {type(exc).__name__}: {exc}"})

    def interface_unchanged() -> str:
        block = extract_interface(source)
        if block is None:
            return "the FIXED INTERFACE markers are missing; restore the block from the starting engine.py"
        if _normalise(block) != _normalise(FIXED_INTERFACE):
            return "the FIXED INTERFACE block was edited; restore it exactly (the harness depends on it)"
        return ""

    def defines_functions() -> str:
        missing = [n for n in ("make_level", "step") if not callable(getattr(module, n, None))]
        return f"missing function(s): {', '.join(missing)}" if missing else ""

    def valid_levels() -> str:
        for n in levels:
            state = module.make_level(n)
            problems = _problems_in_state(state, sprite_cls, state_cls)
            if not problems:
                render(state)
            else:
                return f"make_level({n}): " + "; ".join(problems[:4])
        return ""

    def fresh_every_call() -> str:
        for n in levels:
            first = module.make_level(n)
            frame, vars_before = render(first), copy.deepcopy(first.vars)
            _scribble(first)
            again = module.make_level(n)
            changed = int((render(again) != frame).sum())
            if changed or again.vars != vars_before:
                what = f"{changed} pixels differ" if changed else "vars differ"
                return (
                    f"make_level({n}) changed after the previous state was modified ({what}): it must build new "
                    "Sprite objects, new pixel lists and a new vars dict on every call (RESET relies on it)"
                )
        return ""

    def actions_accepted() -> str:
        trials: list[Any] = []
        for a in available_actions:
            if a == 0:
                continue
            if a == 6:
                trials += [(6, 32, 32), (6, 0, 0), (6, 63, 63), (6, 10, 50)]
            else:
                trials.append((a, 0, 0))
        for action_id, x, y in trials:
            state = module.make_level(levels[0])
            cell = to_grid(state.grid, x, y) if action_id == 6 else None
            returned = module.step(state, action_cls(id=action_id, x=x, y=y, cell=cell))
            if returned is not None:
                return f"step() returned {type(returned).__name__}: it must change the given state in place and return None"
            problems = _problems_in_state(state, sprite_cls, state_cls)
            if problems:
                return f"after step with action {action_id}: " + "; ".join(problems[:3])
            render(state)
        return ""

    def deterministic() -> str:
        rng = random.Random(seed)
        choices = [a for a in available_actions if a != 0] or [1]
        level = levels[0]
        a, b = module.make_level(level), module.make_level(level)
        for k in range(30):
            action_id = rng.choice(choices)
            x, y = (rng.randrange(64), rng.randrange(64)) if action_id == 6 else (0, 0)
            for state in (a, b):
                cell = to_grid(state.grid, x, y) if action_id == 6 else None
                module.step(state, action_cls(id=action_id, x=x, y=y, cell=cell))
            if int((render(a) != render(b)).sum()) or a.vars != b.vars or a.status != b.status:
                return f"two fresh states given the same {k + 1} actions differ: step() must not use randomness or shared state"
            if a.status != "playing":
                break
        return ""

    check("the FIXED INTERFACE block is unchanged", interface_unchanged)
    check("make_level and step are defined", defines_functions)
    if results[-1]["ok"]:
        check("make_level(n) builds a valid State for every recorded level", valid_levels)
        check("make_level(n) builds a fresh state on every call", fresh_every_call)
        check("step() accepts every advertised action and keeps the state valid", actions_accepted)
        check("the same actions give the same result", deterministic)
    return results


def describe_contract(results: list[dict[str, Any]]) -> str:
    passed = sum(r["ok"] for r in results)
    lines = [f"  Contract tests: {passed}/{len(results)} pass" + ("." if passed == len(results) else ":")]
    for r in results:
        if not r["ok"]:
            lines.append(f"    FAILED {r['name']}: {r['detail']}")
    return "\n".join(lines)
