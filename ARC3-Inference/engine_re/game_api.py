"""The simple game interface: what the agent writes, and how the harness runs it.

The agent's engine.py is a plain module, not an arcengine game:

    make_level(n) -> State      the state at the start of level n
    step(state, action)         apply one action to the state, in place

``FIXED_INTERFACE`` is the block of code at the top of every engine.py that
defines ``Sprite``, ``Action`` and ``State``. Sprite and State follow
arcengine's sprite model (visible/collidable flags for its interaction modes,
blocking modes, rotation, mirroring, scale) and provide its common mechanisms
(collisions, ``try_move``, ``sprite_at`` lookup order, colour remapping,
cloning), so the agent writes only game rules. The agent must not edit the
block, and the contract tests check that it is unchanged. Everything else the
real engines do around the two functions is the harness's job:

- ``render(state)`` draws a State on the 64x64 screen;
- ``GameRunner`` reproduces arcengine's episode rules (RESET, level changes,
  WIN, GAME_OVER, what is returned after the game ended) and turns screen
  clicks into grid cells, returning observations shaped like the real ones.
  ``make_level(n)`` runs once per level; every level start (entry or RESET)
  gets a deep copy of that first state, like arcengine's pristine levels;
- ``contract_checks`` is a small unit-test suite for the module's contract
  (interface untouched, functions defined, valid states, every advertised
  action accepted, determinism). Matching the recording (the acceptance test)
  is the replay in ``tester.replay_test``.

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
# Sprite and State follow the sprite model of the library the real games are written with: the
# same drawing order, transforms, visibility, collisions and lookups. The harness does the rest:
#   - Levels and RESET: make_level(n) runs once per level. Whenever level n starts (on entering it
#     and on every RESET) the harness hands step() a fresh deep copy of that first state, so nothing
#     a step changes survives a RESET, and make_level may reuse module-level data.
#   - Outcomes: step() sets state.status = "level_solved" (the next level starts, or the game is won
#     after the last one) or "game_over". The harness counts levels and handles WIN and GAME_OVER.
#   - Clicks: action.cell is the grid cell under the click, None outside the grid.
#   - Drawing, of the final state of each action only:
#     1. Start from a 64x64 screen filled with colour 5.
#     2. Draw each visible sprite as sprite.render(), lowest layer first; sprites on the same layer
#        are drawn in list order, so later ones end up on top.
#        - A grid sprite (screen=False) is placed on the logical grid. The grid (w, h) = state.grid
#          is scaled up by s = min(64 // w, 64 // h) and centred: grid cell (gx, gy) fills the s x s
#          screen block whose top-left pixel is (ox + gx * s, oy + gy * s), with
#          ox = (64 - w * s) // 2 and oy = (64 - h * s) // 2. Parts outside the grid are not drawn.
#        - A screen sprite (screen=True) is placed in screen pixels, unscaled: use it for things
#          drawn at screen resolution, such as a budget bar in the border.
#        - Pixels -1 (transparent) and -2 (invisible but solid) are not drawn.
#     So the border colour is a 64x64 screen sprite on the lowest layer and the background is a
#     grid-sized sprite on the layer above; make both collidable=False so nothing bumps into them.
#
# Coordinates: x is the column, y is the row, (0, 0) is the top-left corner.

from copy import deepcopy
from dataclasses import dataclass, field, replace


@dataclass(eq=False)
class Sprite:
    """A rectangle of pixels: colours 0-15, -1 transparent, -2 invisible but solid."""

    pixels: list  # rows of colour indices, before rotation, mirroring and scale
    x: int = 0  # column of the top-left pixel (grid cells, or screen pixels if screen=True)
    y: int = 0  # row of the top-left pixel
    layer: int = 0  # higher layers are drawn on top and found first by sprite_at()
    name: str = ""
    tags: tuple = ()  # labels to find sprites again, e.g. state.by_tag("wall")
    visible: bool = True  # drawn
    collidable: bool = True  # takes part in collisions and sprite_at()
    # visible and collidable: a normal object; visible only: drawn but never hit;
    # collidable only: an invisible wall; neither: hidden and inert, but still in the list.
    blocking: str = "pixel"  # "pixel": collides where pixels other than -1 overlap;
    # "box": collides where bounding boxes overlap; "none": never collides
    rotation: int = 0  # 0, 90, 180 or 270 degrees clockwise
    mirror_ud: bool = False  # flipped top-bottom, after the rotation
    mirror_lr: bool = False  # flipped left-right, after the rotation
    scale: int = 1  # 2 or more: each pixel becomes a scale x scale block; -1, -2, ...: shrunk 2x,
    # 3x, ..., each block taking its most common colour (ties: the highest colour)
    screen: bool = False  # True: screen pixels, not scaled with the grid; False: the logical grid

    def render(self) -> list:
        """The pixels as drawn: rotated, then mirrored, then scaled."""
        out = [list(row) for row in self.pixels]
        for _ in range((self.rotation // 90) % 4):
            out = [list(row) for row in zip(*out[::-1])]
        if self.mirror_ud:
            out = out[::-1]
        if self.mirror_lr:
            out = [row[::-1] for row in out]
        if self.scale > 1:
            out = [[v for v in row for _ in range(self.scale)] for row in out for _ in range(self.scale)]
        elif self.scale < 0:
            k = 1 - self.scale
            shrunk = []
            for r in range(0, len(out), k):
                row = []
                for c in range(0, len(out[0]), k):
                    block = [out[r + i][c + j] for i in range(k) for j in range(k)]
                    opaque = [v for v in block if v != -1]
                    if sum(v < 0 for v in block) > len(opaque):
                        row.append(-1)
                    else:
                        row.append(max(opaque, key=lambda v: (opaque.count(v), v)))
                shrunk.append(row)
            out = shrunk
        return out

    @property
    def width(self) -> int:
        """Width as drawn (after rotation and scale)."""
        h, w = len(self.pixels), len(self.pixels[0]) if len(self.pixels) else 0
        if (self.rotation // 90) % 2:
            h, w = w, h
        return w * self.scale if self.scale > 1 else w // (1 - self.scale) if self.scale < 0 else w

    @property
    def height(self) -> int:
        """Height as drawn (after rotation and scale)."""
        h, w = len(self.pixels), len(self.pixels[0]) if len(self.pixels) else 0
        if (self.rotation // 90) % 2:
            h, w = w, h
        return h * self.scale if self.scale > 1 else h // (1 - self.scale) if self.scale < 0 else h

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy

    def set_position(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def color_remap(self, old, new: int) -> None:
        """Replace colour `old` by `new`; old=None replaces every colour (pixels >= 0)."""
        self.pixels = [[new if (v >= 0 if old is None else v == old) else v for v in row] for row in self.pixels]

    def clone(self, **changes) -> "Sprite":
        """An independent copy, with any fields changed, e.g. wall.clone(x=3, y=4)."""
        return replace(deepcopy(self), **changes)

    def collides_with(self, other: "Sprite") -> bool:
        """Both collidable, neither blocking "none", in the same space, and overlapping: by bounding
        box, or by pixels other than -1 if either sprite blocks by "pixel"."""
        if self is other or not (self.collidable and other.collidable) or self.screen != other.screen:
            return False
        if self.blocking == "none" or other.blocking == "none":
            return False
        x0, x1 = max(self.x, other.x), min(self.x + self.width, other.x + other.width)
        y0, y1 = max(self.y, other.y), min(self.y + self.height, other.y + other.height)
        if x0 >= x1 or y0 >= y1:
            return False
        if self.blocking != "pixel" and other.blocking != "pixel":
            return True
        a, b = self.render(), other.render()
        return any(
            a[y - self.y][x - self.x] != -1 and b[y - other.y][x - other.x] != -1 for y in range(y0, y1) for x in range(x0, x1)
        )


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

    def sprite_at(self, x: int, y: int, tag: str | None = None, include_uncollidable: bool = False, screen: bool = False):
        """The sprite at (x, y), or None: highest layer first, then list order. Only collidable
        sprites count unless include_uncollidable; a "pixel"-blocking sprite counts only where its
        pixel is not -1. With a tag, the first such sprite carrying it."""
        for s in sorted(self.sprites, key=lambda s: s.layer, reverse=True):
            if s.screen != screen or not (include_uncollidable or s.collidable):
                continue
            if not (s.x <= x < s.x + s.width and s.y <= y < s.y + s.height):
                continue
            if s.blocking == "pixel" and s.render()[y - s.y][x - s.x] == -1:
                continue
            if tag is None or tag in s.tags:
                return s
        return None

    def collisions(self, sprite: Sprite) -> list:
        """The sprites that `sprite` collides with, in list order."""
        return [s for s in self.sprites if sprite.collides_with(s)]

    def try_move(self, sprite: Sprite, dx: int, dy: int) -> list:
        """Move `sprite` by (dx, dy). If it then collides with anything, move it back.
        Returns the sprites it hit ([] means it moved)."""
        sprite.move(dx, dy)
        hits = self.collisions(sprite)
        if hits:
            sprite.move(-dx, -dy)
        return hits

    def add(self, sprite: Sprite) -> Sprite:
        self.sprites.append(sprite)
        return sprite

    def remove(self, sprite: Sprite) -> None:
        self.sprites = [s for s in self.sprites if s is not sprite]


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


def sprite_pixels(sprite: Any) -> np.ndarray:
    """A sprite's pixels as drawn: Sprite.render() from the fixed interface (always the canonical code)."""
    if not getattr(sprite, "rotation", 0) and not getattr(sprite, "mirror_ud", False) and not getattr(sprite, "mirror_lr", False) and getattr(sprite, "scale", 1) == 1:
        return np.asarray(sprite.pixels, dtype=np.int16)
    return np.asarray(canonical().Sprite.render(sprite), dtype=np.int16)


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
        px = sprite_pixels(sprite)
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
        self.pristine: dict[int, Any] = {}

    def fresh(self, level: int) -> Any:
        """A deep copy of level `level`'s first state; make_level runs once per level."""
        if level not in self.pristine:
            self.pristine[level] = self.module.make_level(level)
        state = copy.deepcopy(self.pristine[level])
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


_BLOCKING = ("pixel", "box", "none")


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
        for attr in ("x", "y", "layer", "rotation", "scale"):
            value = getattr(sprite, attr, None)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                problems.append(f"{label}: {attr} must be an int")
        if getattr(sprite, "rotation", 0) not in (0, 90, 180, 270):
            problems.append(f"{label}: rotation must be 0, 90, 180 or 270")
        if getattr(sprite, "blocking", None) not in _BLOCKING:
            problems.append(f"{label}: blocking must be one of {_BLOCKING}")
        scale = getattr(sprite, "scale", 1)
        if scale == 0:
            problems.append(f"{label}: scale must not be 0")
        elif isinstance(scale, int) and scale < 0 and px.ndim == 2 and (px.shape[0] % (1 - scale) or px.shape[1] % (1 - scale)):
            problems.append(f"{label}: with scale {scale} the pixel grid must divide by {1 - scale}")
        if len(problems) > 8:
            break
    return problems


def _canonical_vars(state: Any) -> Any:
    """state.vars with sprites replaced by their position in state.sprites, so two states can be compared."""
    index = {id(s): i for i, s in enumerate(state.sprites)}

    def conv(value: Any) -> Any:
        if isinstance(value, dict):
            return {conv(k): conv(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [conv(v) for v in value]
            return tuple(converted) if isinstance(value, tuple) else converted
        if id(value) in index:
            return ("sprite", index[id(value)])
        return value

    return conv(state.vars)


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
    first_states: dict[int, Any] = {}

    def first_state(n: int) -> Any:
        """A deep copy of make_level(n), as the harness hands it to step()."""
        if n not in first_states:
            first_states[n] = module.make_level(n)
        return copy.deepcopy(first_states[n])

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
            state = first_state(n)
            problems = _problems_in_state(state, sprite_cls, state_cls)
            if problems:
                return f"make_level({n}): " + "; ".join(problems[:4])
            render(state)
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
            state = first_state(levels[0])
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
        a, b = first_state(level), copy.deepcopy(module.make_level(level))
        for k in range(30):
            action_id = rng.choice(choices)
            x, y = (rng.randrange(64), rng.randrange(64)) if action_id == 6 else (0, 0)
            for state in (a, b):
                cell = to_grid(state.grid, x, y) if action_id == 6 else None
                module.step(state, action_cls(id=action_id, x=x, y=y, cell=cell))
            same = int((render(a) != render(b)).sum()) == 0 and _canonical_vars(a) == _canonical_vars(b) and a.status == b.status
            if not same:
                return (
                    f"two fresh copies of level {level} given the same {k + 1} actions differ: make_level and step "
                    "must not use randomness, the clock or state kept outside the State"
                )
            if a.status != "playing":
                break
        return ""

    check("the FIXED INTERFACE block is unchanged", interface_unchanged)
    check("make_level and step are defined", defines_functions)
    if results[-1]["ok"]:
        check("make_level(n) builds a valid State for every recorded level", valid_levels)
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
