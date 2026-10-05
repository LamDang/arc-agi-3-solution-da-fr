"""ARC-AGI-3 game "ls20", reverse-engineered from a recorded run.

Write two functions below the fixed interface:

    make_level(n) -> State     the state at the start of level n (0-based)
    step(state, action)        apply one action to the state, in place
"""

# ==== FIXED INTERFACE: DO NOT EDIT. The harness relies on this block and the tests check it. ====
#
# Sprite and State follow the sprite model of the library the real games are written with: the
# same drawing order, transforms, visibility, collisions and lookups. The harness does the rest:
#   - Levels and RESET: make_level(n) runs once per level. Whenever level n starts (on entering it
#     and on every RESET) the harness hands step() a fresh deep copy of that first state, so nothing
#     a step changes survives a RESET, and make_level may reuse module-level data.
#   - Outcomes: step() sets state.status = "level_solved" (the next level starts, or the game is won
#     after the last one) or "game_over". The harness counts levels and handles WIN and GAME_OVER.
#   - Clicks: action.x, action.y is the screen pixel clicked; action.cell is the grid cell under it,
#     found through the inverse of the view transform (drawing step 3 below), or None outside the grid.
#   - Drawing, of the final state of each action only:
#     1. Start from a 64x64 screen filled with colour 5.
#     2. Draw each visible sprite as sprite.render(), lowest layer first; sprites on the same layer
#        are drawn in list order, so later ones end up on top.
#        - A grid sprite (screen=False) is placed on the logical grid. The grid (w, h) = state.grid
#          is scaled up by s = state.view.scale, or by default min(64 // w, 64 // h), and centred:
#          grid cell (gx, gy) fills the s x s screen block whose top-left pixel is
#          (ox + gx * s, oy + gy * s), with ox = (64 - w * s) // 2 and oy = (64 - h * s) // 2.
#          Parts outside the grid are not drawn.
#        - A screen sprite (screen=True) is placed in screen pixels, unscaled: use it for things
#          drawn at screen resolution, such as a budget bar in the border.
#        - Pixels -1 (transparent) and -2 (invisible but solid) are not drawn.
#     3. Turn the whole frame, screen sprites included, by state.view.rotation (clockwise), then
#        flip it if state.view.mirror_ud (top-bottom) and state.view.mirror_lr (left-right).
#     So the border colour is a 64x64 screen sprite on the lowest layer and the background is a
#     grid-sized sprite on the layer above; make both collidable=False so nothing bumps into them.
#
# Coordinates: x is the column, x is the row, (0, 0) is the top-left corner.

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
            if s.blocking == "pixel" and s.render()[y - self_y(s)][x - s.x] == -1:
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


def _self_y(s):
    return s.y


# ==== END OF FIXED INTERFACE ====


# ==== YOUR GAME ====

from .leveldata import LEVELS  # placeholder; replaced below
