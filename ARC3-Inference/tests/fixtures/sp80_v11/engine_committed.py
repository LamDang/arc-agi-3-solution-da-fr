"""ARC-AGI-3 game "sp80", modelled from the game as it is played.

Write two functions below the fixed interface:

    make_level(n) -> State     the state at the start of level n (0-based)
    step(state, action)        apply one action to the state, in place

The fixed interface already provides what every game shares: sprites with layers, visibility,
collidability, blocking modes, rotation, mirroring and scale; collisions (state.try_move,
state.collisions); lookups (state.sprite_at, sprites_at, by_tag, by_name); and a per-level view
(state.view: grid scale, and rotation and mirroring of the whole screen). The harness does the rest: it gives
step() a fresh copy of the level's first state on entering a level and on every RESET (RESET
restarts the current level), draws the state, counts completed levels and ends the game with WIN or
GAME_OVER. This game advertises actions [1, 2, 3, 4, 5, 6].

How it is tested (run_tests):
- contract tests: the fixed interface is unchanged, states are valid, step accepts every advertised
  action, the same actions give the same result;
- acceptance test: every action played so far is replayed; after each one, your final frame and
  the game state must equal the game's. Animation frames are not compared.

Keep this file self-contained: no file reads, all level data written in it. print() in make_level
and step to debug: the test report and replay_step show what a step printed.
"""

# ==== FIXED INTERFACE: DO NOT EDIT. The harness relies on this block and the tests check it. ====
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


# ==== END OF FIXED INTERFACE ====


# ==== YOUR GAME ====


# ---- sp80: pour the source's fluid into every bin without spilling any on the drain ----
#
# The player is the blue 5x1 bar: arrow keys slide it one cell, blocked by solid sprites, the grid
# edge and the row just under the source block (steps 1, 6, 8: it rose once from row 4 to row 3 and
# no further, whichever columns it spanned); a red 3x1 BLOCK in the way is destroyed instead (step
# 25: the bar moved up into the block at grid cols 6-8 row 9 and that block is gone from the frame).
# SPACE pours: the fluid appears in the cell below the
# magenta source block, falls while the cell below is free, and where it rests it runs sideways
# along the obstacle, dropping again wherever the cell below opens up (step 4's 28-frame animation
# and step 11's 20-frame one are exactly that, cell by cell). Fluid that lands on the light grey
# strip is lost (that strip blinked while the step-4 pour drained away). A pour wins only when EVERY
# bin's cavity is wet and nothing was lost - the two halves are separated by the recorded pours:
# step 4 filled both cavities but spilled -> no win; step 24 lost nothing but filled 1 of 3 cavities
# -> no win; step 11 did both -> level 0 solved.
#
# Level 1's first frame is this same kind of level drawn turned 180 degrees, so it is modelled the
# same way in grid coordinates with state.view.rotation = 180: 3 bins, 2 red 3x1 obstacles, the bar
# 2 rows lower, the source in column 5.


def shape_pixels(shape, recolor=None):
    """Fresh pixel rows from a shape: hex strings (one digit per pixel, '.' transparent) or rows of
    colour numbers. recolor={old: new} changes colours."""
    rows = [[-1 if ch == "." else int(ch, 16) for ch in row] if isinstance(row, str) else list(row) for row in shape]
    return [[recolor.get(v, v) for v in row] for row in rows] if recolor else rows


CAP = ("4",)            # dark grey 1x1: the block the source hangs from
SOURCE = ("6",)         # magenta 1x1: pours the fluid downwards
BAR = ("99999",)        # blue 5x1: the player
BIN = ("b.b", "bbb")    # yellow U open at the top; its transparent cell is the cavity to fill
BLOCK = ("888",)        # red 3x1 obstacle
DRAIN = ("1" * 16,)     # light grey 1x16 strip: fluid that lands on it is lost


def cells_of(sprite):
    """The grid cells a grid sprite covers."""
    return [(x, y) for y in range(sprite.y, sprite.y + sprite.height) for x in range(sprite.x, sprite.x + sprite.width)]
def hud_pixels(moves, budget=30):
    """The strip along one edge is a move budget drawn proportionally: green = round(64 * remaining
    / budget) from the left, the rest white. Level 0 (budget 30) ate 2,2,2,3,2,2,2,2,2,2 px over its
    ten actions - exactly round(64*m/30), no animation penalty - and level 1 (budget 45) ate
    1,2,1,2 px - exactly round(64*m/45)."""
    g = max(0, min(64, round(64 * (budget - moves) / budget)))
    return [[14] * g + [0] * (64 - g)]


def board_sprites(objects) -> list:
    """Orange screen border, orange board, the level's objects, and the HUD strip last (on top)."""
    return ([Sprite([[12] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
            Sprite([[12] * 16 for _ in range(16)], layer=-1, collidable=False, name="background")]
            + objects + [Sprite(hud_pixels(0), x=0, y=0, screen=True, layer=1, collidable=False,
                               name="hud", tags=("hud",))])


def obj(pixels, x, y, tag):
    return Sprite(shape_pixels(pixels), x=x, y=y, tags=(tag,))


def level_0_sprites() -> list:
    """Level 0: source in column 9, the bar starts at (3,4), 2 bins, the drain strip along row 15."""
    return board_sprites([
        obj(CAP, 9, 0, "cap"),
        obj(SOURCE, 9, 1, "source"),
        obj(BAR, 3, 4, "player"),
        obj(BIN, 4, 13, "bin"),
        obj(BIN, 10, 13, "bin"),
        obj(DRAIN, 0, 15, "drain"),
    ])


def level_1_sprites() -> list:
    """Level 1 in grid coordinates (its view is turned 180 degrees): 3 bins, 2 red obstacles."""
    return board_sprites([
        obj(CAP, 5, 0, "cap"),
        obj(SOURCE, 5, 1, "source"),
        obj(BAR, 6, 6, "player"),
        obj(BLOCK, 6, 9, "block"),
        obj(BLOCK, 11, 11, "block"),
        obj(BIN, 2, 13, "bin"),
        obj(BIN, 6, 13, "bin"),
        obj(BIN, 10, 13, "bin"),
        obj(DRAIN, 0, 15, "drain"),
    ])


def level_2_sprites() -> list:
    """Level 2, also drawn turned 180 degrees: 3 sources in columns 1, 6 and 14, a 6x1 bar,
    red blocks 4x1 / 5x1 / 6x1, 3 bins (cavities at cols 2, 8 and 13), drain along row 15."""
    return board_sprites([
        obj(CAP, 14, 0, "cap"), obj(CAP, 6, 0, "cap"), obj(CAP, 1, 0, "cap"),
        obj(SOURCE, 14, 1, "source"), obj(SOURCE, 6, 1, "source"), obj(SOURCE, 1, 1, "source"),
        obj(("9" * 6,), 1, 5, "player"),
        obj(("8" * 4,), 10, 10, "block"), obj(("8" * 5,), 1, 8, "block"),
        obj(("8" * 6,), 8, 7, "block"),
        obj(BIN, 1, 13, "bin"), obj(BIN, 7, 13, "bin"), obj(BIN, 12, 13, "bin"),
        obj(DRAIN, 0, 15, "drain"),
    ])


def level_3_sprites() -> list:
    """Level 3: a 20x20 board at scale 3 with a light grey frame instead of a drain strip, one source
    in column 7, 4 bins along row 17 (cavities at cols 3, 9, 13, 17), the 5x1 bar at (5,5) and red
    blocks 5/3/3/4/4. Not turned: the HUD strip is along the top of the screen."""
    return [Sprite([[1] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
            Sprite([[12] * 20 for _ in range(20)], layer=-1, collidable=False, name="background")] + [
        obj(CAP, 7, 0, "cap"),
        obj(SOURCE, 7, 1, "source"),
        obj(BAR, 5, 5, "player"),
        obj(("8" * 5,), 12, 5, "block"),
        Sprite(shape_pixels(("8884888",)), x=2, y=9, tags=("block",)),  # ONE sprite: step 64 clicked
        # bars turned blue together, while clicking the 4x1 at (14,10) (step 58) or the 5x1 at (12,5)
        # (step 52) swapped only that sprite - so the pair is a single piece with an empty cell.
        obj(("8" * 4,), 14, 10, "block"), obj(("8" * 4,), 12, 13, "block"),
        obj(BIN, 2, 17, "bin"), obj(BIN, 8, 17, "bin"), obj(BIN, 12, 17, "bin"), obj(BIN, 16, 17, "bin"),
        obj(("1" * 20,), 0, 19, "drain"),
        Sprite(hud_pixels(0), x=0, y=0, screen=True, layer=1, collidable=False, name="hud", tags=("hud",)),
    ]


BUDGET = {0: 30, 1: 45, 2: 96, 3: 120}
# level 0 ate 2,2,2,3,2,... px = round(64*m/30); level 1 1,2,1,2 = /45; level 2 = /96;
# level 3 lost 1 px every TWO actions plus one extra tick: white = round(m*64/120) matches all 33
# of its actions exactly (1,1,2,2,...,16,17,17,18)
LEVELS = {0: level_0_sprites, 1: level_1_sprites, 2: level_2_sprites, 3: level_3_sprites}
VIEWS = {1: View(rotation=180), 2: View(rotation=180), 3: View(scale=3)}


GRIDS = {3: (20, 20)}


def make_level(n: int) -> State:
    """The state at the start of level n; a level not modelled yet starts like the last known one."""
    build = LEVELS.get(n, LEVELS[max(LEVELS)])
    return State(grid=GRIDS.get(n, (16, 16)), sprites=build(), view=VIEWS.get(n, View()),
                 vars={"budget": BUDGET.get(n, 15 * (n + 2))})


def absorb(state, vacated):
    """The ERODED cell (the red 1x1 the last pour broke off the bar) is still tied to the bar: when
    the bar moves, that tail keeps the cells the bar has just left IF they lie in the tail's own row,
    touching it side by side (step 37: the 1x1 at (6,8) became a 6x1 over (1..6,8) - the bar's five
    vacated cells, same row 8, plus its own; step 38: the bar moved off row 9 and the tail, now rows
    8-9, did NOT take row 9's vacated cells - they stayed empty). Ordinary red blocks never do this:
    level 1 steps 15/16/20 moved the bar off cells beside/below a block and left them empty."""
    for b in state.sprites:
        if "tail" not in b.tags:
            continue
        bc = set(cells_of(b))
        if any((x + dx, y) in bc for (x, y) in vacated for dx in (1, -1)):
            allc = bc | set(vacated)
            x0 = min(c[0] for c in allc); y0 = min(c[1] for c in allc)
            x1 = max(c[0] for c in allc); y1 = max(c[1] for c in allc)
            b.set_position(x0, y0)
            b.pixels = [[8 if (x, y) in allc else -1 for x in range(x0, x1 + 1)]
                        for y in range(y0, y1 + 1)]
            return


def grid_dir(view, sx, sy):
    """A screen direction as a grid direction: the finished frame is turned clockwise by
    view.rotation, so on a turned board an arrow key moves the bar the other way in grid terms.
    Step 12 settled it: RIGHT on level 1 (rotation 180) moved the bar one cell LEFT in grid x."""
    for _ in range((view.rotation // 90) % 4):   # undo one clockwise quarter turn at a time
        sx, sy = sy, -sx
    if view.mirror_lr:
        sx = -sx
    if view.mirror_ud:
        sy = -sy
    return sx, sy


def step(state: State, action: Action) -> None:
    """Apply `action` to `state` in place: the rules are summarised at the top of this section."""
    if action.id in (1, 2, 3, 4):
        p = state.by_tag("player")[0]
        dx, dy = grid_dir(state.view, *{1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action.id])
        nx, ny = p.x + dx, p.y + dy
        top = state.by_tag("source")[0].y + 2  # the bar never enters the row the fluid starts in
        bins = [s for s in state.sprites if "bin" in s.tags]
        floor = (min(s.y for s in bins) - 2) if bins else state.grid[1] - 1
        # nor within one row of the bins: level 2 step 42 pressed UP from grid row 11 (the bins are at
        # rows 13-14) and nothing happened; level 1's bar never went below grid row 11 either.
        if ny >= top and ny + p.height - 1 <= floor and 0 <= nx and nx + p.width <= state.grid[0]:
            old = cells_of(p)
            p.set_position(nx, ny)
            hits = state.collisions(p)
            if hits and all("block" in h.tags for h in hits):
                for h in hits:
                    state.remove(h)  # step 25: moving into a red block destroys it, the bar moves on
            elif hits:
                p.set_position(nx - dx, ny - dy)
            else:
                absorb(state, [c for c in old if c not in set(cells_of(p))])  # only cells the bar left
    elif action.id == 6:
        # A click on a red block exchanges the two roles in place: the block becomes the blue bar you
        # steer and the bar you were steering becomes a red block, neither moving (step 26: clicking
        # the block at grid (11,11) made it blue and turned the 5x1 bar at (4,9) red). Clicking any
        # other sprite does nothing (step 5: a click on the source block changed nothing).
        if action.cell is not None:
            sp = state.sprite_at(action.cell[0], action.cell[1], tag="block")
            if sp is not None:
                p = state.by_tag("player")[0]
                p.tags, sp.tags = sp.tags, p.tags
                p.color_remap(9, 8)
                sp.color_remap(8, 9)
    elif action.id == 5:
        wet, lost, impacts = pour(state)
        cav = cavities(state)
        print("pour:", len(wet), "wet cells, lost", lost, ", cavities", sorted(cav),
              "filled", sorted(cav & wet))
        if cav and not lost and cav <= wet:
            state.status = "level_solved"
        else:
            erode(state, impacts)
    hud = state.by_name("hud")
    if hud is not None:
        hud.pixels = hud_pixels(state.vars.get("moves", 0) + 1, state.vars.get("budget", 30))
    state.vars["moves"] = state.vars.get("moves", 0) + 1


def cavities(state):
    """The cell each bin is open at - the transparent cell inside its box. The fluid has to fill them."""
    out = set()
    for sp in state.by_tag("bin"):
        for j, row in enumerate(sp.pixels):
            for i, v in enumerate(row):
                if v == -1:
                    out.add((sp.x + i, sp.y + j))
    return out


def pour(state):
    """(the cells the poured fluid wets, whether any of it came to rest on the drain).

    The fluid appears in the cell below the source block and falls while the cell below it is free;
    where it comes to rest it runs off sideways, starting one new falling stream for each cell it
    passes that has nothing under it, and stops in that direction there. Reproduces step 4's and
    step 11's animations cell for cell (37 and 29 wet cells).
    """
    w, h = state.grid
    solid, drain = set(), set()
    for sp in state.sprites:
        if sp.screen or not sp.collidable:
            continue
        for j, row in enumerate(sp.render()):  # pixel by pixel: a bin's open cell is not solid
            for i, v in enumerate(row):
                if v != -1:
                    c = (sp.x + i, sp.y + j)
                    solid.add(c)
                    if "drain" in sp.tags:
                        drain.add(c)

    def free(c):
        return 0 <= c[0] < w and 0 <= c[1] < h and c not in solid

    wet, running, lost = set(), set(), [False]
    impacts = set()   # cells that a FALLING stream came to rest on

    def stream(x, y):
        running.add((x, y))
        wet.add((x, y))
        y0 = y
        while free((x, y + 1)):
            y += 1
            wet.add((x, y))
        if y > y0:
            impacts.add((x, y + 1))
        if (x, y + 1) in drain:
            lost[0] = True
            return
        for d in (-1, 1):
            nx = x + d
            while free((nx, y)) and (nx, y) not in wet:
                wet.add((nx, y))
                if free((nx, y + 1)):
                    if (nx, y) not in running:
                        stream(nx, y)
                    break
                nx += d

    for src in state.by_tag("source"):   # every source pours at once (level 2 has three)
        if free((src.x, src.y + 1)):
            stream(src.x, src.y + 1)
    return wet, lost[0], impacts


def erode(state, impacts):
    """A stream that FALLS onto the bar's rightmost cell washes that cell off - it stays where it is
    as a 1x1 red block (step 36: the col-6 stream fell 5 cells onto the bar whose last cell was grid
    (6,8); in the next frame the bar is 5 cells wide and (6,8) is a red 1x1). Level 0 step 4 (the
    fluid starts resting on the bar, so it never falls) and level 1 step 24 (it falls onto the bar's
    middle, not its last cell) both left the bar whole."""
    p = state.by_tag("player")[0]
    x = p.x + p.width - 1
    if (x, p.y) not in impacts:
        return
    for row in p.pixels:
        row.pop()
    state.add(Sprite(shape_pixels(("8",)), x=x, y=p.y, tags=("block", "tail")))
