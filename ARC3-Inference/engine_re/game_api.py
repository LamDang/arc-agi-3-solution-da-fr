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

import base64
import copy
import hashlib
import io
import random
import sys
import types
import zlib
from typing import Any

import numpy as np

BEGIN_MARKER = "# ==== FIXED INTERFACE: DO NOT EDIT. The harness relies on this block and the tests check it. ===="
END_MARKER = "# ==== END OF FIXED INTERFACE ===="

FIXED_INTERFACE = BEGIN_MARKER + '''
#
# The harness runs this module:
#   - Levels and RESET: make_level(n) runs once per level. Whenever level n starts (on entering it
#     and after every RESET) the harness hands step() a fresh copy of that first state, so nothing a
#     step changes survives a RESET, and make_level may reuse module-level data. RESET never
#     reaches step().
#   - Outcomes: step() sets state.status = "level_solved" (the next level starts, or WIN after the
#     last one) or "game_over" (the game ends; then only RESET is accepted).
#   - Drawing, of the state after each action:
#     1. Start from a 64x64 screen filled with colour 5.
#     2. Draw every visible sprite: lowest layer first, sprites on the same layer in list order
#        (later on top). Pixels -1 (transparent) and -2 (invisible but solid) are not drawn. Each
#        sprite is drawn as sprite.render(): its pixels rotated clockwise by .rotation, then flipped
#        (.mirror_ud, .mirror_lr), then scaled by .scale.
#        - Grid sprites (screen=False) sit on the logical grid (w, h) = state.grid. The grid is
#          scaled by s = state.view.scale, or min(64 // w, 64 // h) when that is None, and centred:
#          grid cell (gx, gy) covers screen pixels x in [ox + gx*s, ox + gx*s + s),
#          y in [oy + gy*s, oy + gy*s + s), with ox = (64 - w*s) // 2 and oy = (64 - h*s) // 2.
#          Anything outside the grid is cut off.
#        - Screen sprites (screen=True) sit directly on screen pixels, unscaled (for displays in
#          the border).
#     3. Turn the finished frame clockwise by state.view.rotation, then flip it if
#        state.view.mirror_ud (top-bottom) and state.view.mirror_lr (left-right).
#   - Clicks: action.x, action.y is the clicked screen pixel; action.cell is the grid cell (gx, gy)
#     under it once step 3 is undone, or None outside the grid.
#
# Coordinates: x is the column, y the row, (0, 0) top-left.

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

    def collides_with(self, other: "Sprite", ignore_mode: bool = False) -> bool:
        """Both collidable, neither blocking "none", in the same space, and overlapping: by bounding
        box, or by pixels other than -1 if either sprite blocks by "pixel". ignore_mode=True skips
        the collidable and blocking "none" checks."""
        if self is other or self.screen != other.screen:
            return False
        if not ignore_mode and not (self.collidable and other.collidable):
            return False
        if not ignore_mode and (self.blocking == "none" or other.blocking == "none"):
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

    def __repr__(self) -> str:
        """The Python code that builds this sprite, with the fields that differ from their defaults,
        e.g. Sprite([[8, 8], [8, -1]], x=3, y=4, tags=("wall",)). print() of a sprite list shows code."""

        def code(value) -> str:
            value = value.tolist() if hasattr(value, "tolist") else value  # numpy values as plain ones
            if isinstance(value, str):
                text = repr(value)
                return '"' + text[1:-1] + '"' if text[0] == "'" and '"' not in value else text
            if isinstance(value, tuple):
                return "(" + ", ".join(code(v) for v in value) + ("," if len(value) == 1 else "") + ")"
            if not isinstance(value, list):
                return repr(value)
            items = [code(v) for v in value]
            text = "[" + ", ".join(items) + "]"
            if len(items) > 1 and len(set(items)) == 1:  # equal items: [v] * n, or [row for _ in range(n)], when much shorter
                same = f"[{items[0]} for _ in range({len(items)})]" if isinstance(value[0], list) else f"[{items[0]}] * {len(items)}"
                text = same if len(same) + 8 < len(text) else text
            return text

        parts = [code(self.pixels)]
        for name in ("x", "y", "rotation", "mirror_ud", "mirror_lr", "scale", "screen", "layer", "visible", "collidable",
                     "blocking", "name", "tags"):
            value, default = getattr(self, name), Sprite.__dataclass_fields__[name].default
            try:
                differs = bool(value != default)
            except Exception:  # e.g. a numpy array, which has no single truth value
                differs = True
            if differs:
                parts.append(f"{name}={code(value)}")
        return "Sprite(" + ", ".join(parts) + ")"


@dataclass
class Action:
    """One action given to step(). RESET (id 0) is handled by the harness and never reaches step()."""

    id: int  # 1 up, 2 down, 3 left, 4 right, 5 interact, 6 click, 7 undo
    x: int = 0  # for a click: the screen pixel column (0-63)
    y: int = 0  # for a click: the screen pixel row (0-63)
    cell: tuple | None = None  # for a click: the grid cell (gx, gy) under it, None if outside the grid


@dataclass
class View:
    """How the level is shown on the 64x64 screen (drawing steps 2 and 3)."""

    scale: int | None = None  # grid scale; None: the largest that fits, min(64 // w, 64 // h)
    rotation: int = 0  # the finished frame turned clockwise by 0, 90, 180 or 270 degrees
    mirror_ud: bool = False  # then flipped top-bottom
    mirror_lr: bool = False  # then flipped left-right


@dataclass(eq=False)
class State:
    """Everything about the current level. make_level() creates it; step() changes it in place."""

    grid: tuple  # logical grid size (width, height), each 1-64
    sprites: list = field(default_factory=list)  # everything drawn: border, background, objects, HUD
    vars: dict = field(default_factory=dict)  # hidden state: budget, counters, modes, what is selected, ...
    status: str = "playing"  # step() sets "level_solved" or "game_over"
    level: int = 0  # index of the current level, set by the harness
    view: View = field(default_factory=View)  # scale, rotation and mirroring of the screen

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

    def sprites_at(self, x: int, y: int, screen: bool = False) -> list:
        """Every sprite whose bounding box contains (x, y), in list order, whatever its flags."""
        return [s for s in self.sprites if s.screen == screen and s.x <= x < s.x + s.width and s.y <= y < s.y + s.height]

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


def _code_tokens(text: str) -> list[tuple[int, str]] | None:
    """The tokens of a block without its comments and blank lines (None if it does not tokenize)."""
    import tokenize

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return [(t.type, t.string) for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


# The code of earlier FIXED blocks that engines of earlier runs still have (sha256 of repr(_code_tokens(block))):
# the block before Sprite.__repr__ printed a sprite as the code that builds it.
EARLIER_INTERFACES = frozenset({"151799314d0100f0e6a48fe472bcd1fb4bbcf477e0a7063a1b9d305ddb787319"})


def same_interface(block: str) -> bool:
    """Whether an engine's fixed block is FIXED_INTERFACE. Comments are ignored, so engines of
    earlier runs, whose block had other comments, still count as unchanged; so does an earlier
    block's code (EARLIER_INTERFACES), which the harness works with just as well."""
    if _normalise(block) == _normalise(FIXED_INTERFACE):
        return True
    ours = _code_tokens(block)
    if ours is None:
        return False
    return ours == _code_tokens(FIXED_INTERFACE) or hashlib.sha256(repr(ours).encode()).hexdigest() in EARLIER_INTERFACES


def fixed_block_lines(source: str) -> tuple[int, int] | None:
    """1-based line numbers of the FIXED block's first and last lines in a source, if it has both markers."""
    first = last = None
    for n, line in enumerate(source.splitlines(), 1):
        if first is None and line.strip() == BEGIN_MARKER:
            first = n
        elif first is not None and line.strip() == END_MARKER:
            last = n
            break
    return (first, last) if first and last else None


# --- Rendering ------------------------------------------------------------------


def geometry(grid: tuple, scale: int | None = None) -> tuple[int, int, int]:
    """Scale and top-left offset of the logical grid on the 64x64 screen."""
    w, h = int(grid[0]), int(grid[1])
    s = int(scale) if scale else min(64 // w, 64 // h)
    return s, (64 - w * s) // 2, (64 - h * s) // 2


def _view(state: Any) -> Any:
    return getattr(state, "view", None)


def view_transform(frame: np.ndarray, view: Any) -> np.ndarray:
    """Drawing step 3: rotate clockwise, then mirror top-bottom, then left-right."""
    if view is None:
        return frame
    k = (int(getattr(view, "rotation", 0)) // 90) % 4
    if k:
        frame = np.rot90(frame, k=-k)
    if getattr(view, "mirror_ud", False):
        frame = np.flipud(frame)
    if getattr(view, "mirror_lr", False):
        frame = np.fliplr(frame)
    return np.ascontiguousarray(frame)


_INDEX = np.arange(64 * 64).reshape(64, 64)


def to_grid(grid: tuple, x: int, y: int, view: Any = None) -> tuple[int, int] | None:
    """The grid cell under screen pixel (x, y), undoing the view transform, or None outside the grid."""
    if not (0 <= x < 64 and 0 <= y < 64):
        return None
    if view is not None:
        y, x = divmod(int(view_transform(_INDEX, view)[y, x]), 64)
    w, h = int(grid[0]), int(grid[1])
    s, ox, oy = geometry(grid, getattr(view, "scale", None) if view is not None else None)
    if not (ox <= x < ox + w * s and oy <= y < oy + h * s):
        return None
    return (x - ox) // s, (y - oy) // s


def click_cell(state: Any, x: int, y: int) -> tuple[int, int] | None:
    """action.cell for a click at screen pixel (x, y) on this state."""
    return to_grid(state.grid, x, y, _view(state))


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


def _place(screen: np.ndarray, px: np.ndarray, sprite: Any, w: int, h: int, s: int, ox: int, oy: int) -> None:
    """Draw a sprite's pixels as drawn (values < 0 skipped) onto the unturned screen: screen sprites
    in screen pixels; grid sprites clipped to the grid in grid cells, then scaled and placed."""
    x, y = int(sprite.x), int(sprite.y)
    if sprite.screen:
        _blit(screen, px, y, x, 64, 64)
        return
    r0, c0 = max(0, -y), max(0, -x)
    r1, c1 = min(px.shape[0], h - y), min(px.shape[1], w - x)
    if r0 >= r1 or c0 >= c1:
        return
    sub = px[r0:r1, c0:c1]
    if s > 1:
        sub = np.repeat(np.repeat(sub, s, axis=0), s, axis=1)
    _blit(screen, sub, oy + (y + r0) * s, ox + (x + c0) * s, 64, 64)


def draw_order(state: Any) -> list[int]:
    """Indices of state.sprites in drawing order: lowest layer first, then list order."""
    sprites = list(state.sprites)
    return sorted(range(len(sprites)), key=lambda i: (sprites[i].layer, i))


def render(state: Any) -> np.ndarray:
    """Draw a State as a 64x64 frame, following the rules in FIXED_INTERFACE."""
    w, h = int(state.grid[0]), int(state.grid[1])
    view = _view(state)
    s, ox, oy = geometry((w, h), getattr(view, "scale", None) if view is not None else None)
    screen = np.full((64, 64), 5, np.int16)
    sprites = list(state.sprites)
    for i in draw_order(state):
        sprite = sprites[i]
        if not sprite.visible:
            continue
        px = sprite_pixels(sprite)
        if px.ndim != 2 or px.size == 0:
            continue
        _place(screen, px, sprite, w, h, s, ox, oy)
    return view_transform(screen, view).astype(np.int8)


# --- What the engine prints ---------------------------------------------------------------


class PrintCapture(io.TextIOBase):
    """A stand-in for stdout (``contextlib.redirect_stdout``) that keeps only the first `head` and
    the last `tail` characters written, so an engine that prints a lot cannot exhaust memory or
    swell a result file. ``getvalue()`` marks what was dropped."""

    def __init__(self, head: int = 300, tail: int = 1700):
        super().__init__()
        self.head_limit, self.tail_limit = head, tail
        self.head, self.tail, self.total = "", "", 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        text = str(text)
        written = len(text)
        self.total += written
        room = self.head_limit - len(self.head)
        if room > 0:
            self.head += text[:room]
            text = text[room:]
        if text:
            self.tail = (self.tail + text[-self.tail_limit :])[-self.tail_limit :]
        return written

    def getvalue(self) -> str:
        dropped = self.total - len(self.head) - len(self.tail)
        return self.head + (f"\n[... {dropped} characters not kept ...]\n" if dropped > 0 else "") + self.tail


def last_lines(text: str, lines: int = 20, chars: int = 1500, width: int = 200) -> tuple[str, int]:
    """The end of some printed output: at most its last `lines` lines, each cut to `width`
    characters, and `chars` characters in all; and how many lines it had."""
    all_lines = text.rstrip("\n").splitlines()
    kept = [line if len(line) <= width else line[:width] + f"... [{len(line) - width} more characters]" for line in all_lines[-lines:]]
    while len(kept) > 1 and sum(len(line) + 1 for line in kept) > chars:
        kept.pop(0)
    return "\n".join(kept), len(all_lines)


# --- Where each sprite is drawn (failure reports) ---------------------------------------


def sprite_footprints(state: Any) -> list[np.ndarray | None]:
    """For each sprite of state.sprites, in list order: a 64x64 bool mask of the screen pixels it
    draws, as if it were visible (after rotation, scale, clipping to the grid and the view
    transform), or None when it draws nothing on the screen."""
    w, h = int(state.grid[0]), int(state.grid[1])
    view = _view(state)
    s, ox, oy = geometry((w, h), getattr(view, "scale", None) if view is not None else None)
    masks: list[np.ndarray | None] = []
    for sprite in list(state.sprites):
        try:
            px = sprite_pixels(sprite)
            if px.ndim != 2 or px.size == 0:
                masks.append(None)
                continue
            canvas = np.zeros((64, 64), np.int16)
            _place(canvas, np.where(px >= 0, 1, -1).astype(np.int16), sprite, w, h, s, ox, oy)
            mask = view_transform(canvas, view) == 1
            masks.append(mask if mask.any() else None)
        except Exception:  # noqa: BLE001  (a malformed sprite simply has no footprint)
            masks.append(None)
    return masks


def _short_value(value: Any, index: dict[int, int], depth: int = 0) -> str:
    """repr() of a state.vars value, with the state's sprites written as #index."""
    if id(value) in index:
        return f"#{index[id(value)]}"
    if depth > 2:
        return "..."
    if isinstance(value, dict):
        items = list(value.items())
        body = ", ".join(f"{_short_value(k, index, depth + 1)}: {_short_value(v, index, depth + 1)}" for k, v in items[:6])
        return "{" + body + (", ..." if len(items) > 6 else "") + "}"
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        body = ", ".join(_short_value(v, index, depth + 1) for v in items[:8]) + (", ..." if len(items) > 8 else "")
        if isinstance(value, list):
            return f"[{body}]"
        if isinstance(value, tuple):
            return f"({body}{',' if len(items) == 1 else ''})"
        return "{" + body + "}"
    if hasattr(value, "pixels") and hasattr(value, "layer"):
        return f"Sprite({getattr(value, 'name', '')!r}, not in state.sprites)"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def state_summary(state: Any) -> dict[str, Any]:
    """A JSON-ready description of a State for failure reports: grid, view, status, a short form of
    state.vars, and for every sprite its fields and where it draws (screen bounding box and a
    packed mask, see unpack_footprint). The candidate process computes it; the tester reads it."""
    sprites = list(state.sprites)
    index = {id(s): i for i, s in enumerate(sprites)}
    view = _view(state)
    grid = [int(state.grid[0]), int(state.grid[1])]
    out: dict[str, Any] = {
        "grid": grid,
        "view": {
            "scale": getattr(view, "scale", None),
            "rotation": int(getattr(view, "rotation", 0) or 0),
            "mirror_ud": bool(getattr(view, "mirror_ud", False)),
            "mirror_lr": bool(getattr(view, "mirror_lr", False)),
        },
        "status": str(getattr(state, "status", "")),
        "level": getattr(state, "level", None),
        "vars": {},
        "sprites": [],
    }
    try:
        for k, v in list(dict(state.vars).items())[:16]:
            out["vars"][str(k)] = _short_value(v, index)
    except Exception:  # noqa: BLE001
        out["vars"] = {"?": "state.vars could not be read"}
    masks = sprite_footprints(state)
    for sprite, mask in zip(sprites, masks):
        # oid tells the same sprite object apart in two summaries of one state (before and after a
        # step), as long as the caller keeps the objects alive in between.
        entry: dict[str, Any] = {"oid": id(sprite)}
        for name, default in (("name", ""), ("layer", 0), ("x", 0), ("y", 0), ("visible", True), ("collidable", True),
                              ("screen", False), ("blocking", "pixel"), ("rotation", 0), ("mirror_ud", False),
                              ("mirror_lr", False), ("scale", 1)):
            value = getattr(sprite, name, default)
            entry[name] = value if isinstance(value, (str, bool)) else int(value) if isinstance(value, (int, np.integer)) else repr(value)
        entry["tags"] = [str(t) for t in getattr(sprite, "tags", ()) or ()]
        try:
            entry["w"], entry["h"] = int(sprite.width), int(sprite.height)
            entry["pixels_crc"] = zlib.crc32(np.asarray(sprite.pixels, dtype=np.int16).tobytes())
        except Exception:  # noqa: BLE001
            entry["w"] = entry["h"] = 0
            entry["pixels_crc"] = None
        if mask is not None:
            rows, cols = np.nonzero(mask)
            r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
            entry["box"] = [r0, c0, r1, c1]
            entry["mask"] = base64.b64encode(np.packbits(mask[r0 : r1 + 1, c0 : c1 + 1]).tobytes()).decode("ascii")
        out["sprites"].append(entry)
    return out


def unpack_footprint(entry: dict[str, Any]) -> np.ndarray | None:
    """The 64x64 bool mask of where a sprite of a state_summary draws (None if nowhere)."""
    box = entry.get("box")
    if not box:
        return None
    r0, c0, r1, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    bits = np.unpackbits(np.frombuffer(base64.b64decode(entry["mask"]), np.uint8))[: h * w]
    mask = np.zeros((64, 64), bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = bits.reshape(h, w).astype(bool)
    return mask


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
        cell = click_cell(self.state, x, y) if action_id == 6 else None
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

    def resync(self, level: int, score: int, action: Any) -> dict[str, Any]:
        """Back in step with the real game after unexplained steps (the play agent's escape hatch,
        PLAY_DESIGN.md 3.6): the runner is put at level `level` with `score` levels completed. A RESET is
        then performed as usual (the level restarts from make_level); any other action is not given to
        step(): the real game entered the level with it, so the observation shows the level's start."""
        self.level, self.score, self.status = int(level), int(score), "NOT_FINISHED"
        if int(action.id) == 0:
            return self.perform(action)
        self.state = self.fresh(self.level)
        return self._observation([render(self.state)])


def sync_points(meta: dict[str, Any] | None) -> tuple[set[int], dict[int, dict[str, int]]]:
    """The unexplained steps (`ignore`: replayed, never compared) and the resync points (`resync`: step ->
    {"level", "score"}, GameRunner.resync before that step) a trace's meta holds (the play agent's escape
    hatch); empty for a recording."""
    meta = meta or {}
    ignore = {int(i) for i in meta.get("ignore") or []}
    resync = {int(k): {"level": int(v["level"]), "score": int(v["score"])} for k, v in (meta.get("resync") or {}).items()}
    return ignore, resync


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
    view = getattr(state, "view", None)
    if view is not None:
        if getattr(view, "rotation", 0) not in (0, 90, 180, 270):
            problems.append("state.view.rotation must be 0, 90, 180 or 270")
        scale = getattr(view, "scale", None)
        if scale is not None and not (isinstance(scale, int) and not isinstance(scale, bool) and scale >= 1):
            problems.append("state.view.scale must be None or a positive int")
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
        if not same_interface(block):
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
            cell = click_cell(state, x, y) if action_id == 6 else None
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
                cell = click_cell(state, x, y) if action_id == 6 else None
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
