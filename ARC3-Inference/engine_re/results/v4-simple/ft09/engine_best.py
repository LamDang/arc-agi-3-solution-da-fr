"""ARC-AGI-3 game "ft09", reverse-engineered from a recorded run.

Write two functions below the fixed interface:

    make_level(n) -> State     the state at the start of level n (0-based)
    step(state, action)        apply one action to the state, in place

The fixed interface already provides what every game shares: sprites with layers, visibility,
collidability, blocking modes, rotation, mirroring and scale; collisions (state.try_move,
state.collisions); lookups (state.sprite_at, sprites_at, by_tag, by_name); and a per-level view
(state.view: grid scale, and rotation and mirroring of the whole screen). The harness does the rest: it gives
step() a fresh copy of the level's first state on entering a level and on every RESET (RESET
restarts the current level), draws the state, counts completed levels and ends the game with WIN or
GAME_OVER. This game advertises actions [6].

How it is tested (run_tests):
- contract tests: the fixed interface is unchanged, states are valid, step accepts every advertised
  action, the same actions give the same result;
- acceptance test: the recorded actions are replayed; after each one, your final frame and the game
  state must equal the recording. Animation frames are not compared.

Keep this file self-contained: no file reads, all level data written in it.
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
#     found through the inverse of the view transform (step 3 below), or None outside the grid.
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
#
# ft09: a board of 3x3 tiles on a 32x32 logical grid (scale 2).
#  * "plain" tiles are a solid colour; clicking one cycles its colour through the level's
#    palette (the two/three colours shown in the legend at the top right of the screen).
#    A plain tile whose cells carry magenta (6) marks also cycles the neighbour tile in each
#    marked direction (that cell is 6 instead of the tile colour).
#  * "pattern" tiles are fixed: their centre cell is a colour C and the 8 surrounding cells are
#    flags -- 0 = the neighbour there must be C, 2 = the neighbour must be another colour,
#    3 = don't care. A pattern tile cannot be clicked.
#  * When every pattern holds, the level is solved. Each click that changes a tile spends one
#    move of the budget; the bar along the bottom of the screen (screen row 63) shrinks from
#    orange to blue as the budget is spent. Running out of moves loses the game.

HEX = "0123456789abcdef"


def _rows(text):
    """'999', ... -> [[9,9,9], ...]"""
    return [[HEX.index(ch) for ch in row] for row in text]


def _tile_name(x, y):
    return "tile@%d,%d" % (x, y)


def _is_plain(pixels):
    return len({v for row in pixels for v in row if v != 6}) == 1


def _colour(sprite):
    vals = [v for v in (x for row in sprite.pixels for x in row) if v != 6]
    return max(set(vals), key=vals.count)


def make_level(n: int) -> State:
    d = LEVELS[n]
    sprites = [
        Sprite([[5] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
        Sprite(_rows(d["grid"]), layer=-1, collidable=False, name="background"),
    ]
    for x, y, rows in d["tiles"]:
        pixels = _rows(rows)
        sprites.append(
            Sprite(pixels, x=x, y=y, layer=1, name=_tile_name(x, y),
                   tags=("tile", "plain") if _is_plain(pixels) else ("tile", "pattern"))
        )
    sprites.append(Sprite([[12] * 64], x=0, y=63, screen=True, layer=9, collidable=False,
                          name="budget_bar", tags=("hud",)))
    return State(grid=(32, 32), sprites=sprites,
                 vars={"cycle": list(d["cycle"]), "budget": d["budget"], "moves": 0})


def _draw_bar(state):
    """The budget bar: 64 screen pixels, spent ones (from the right) colour 11."""
    moves, budget = state.vars["moves"], state.vars["budget"]
    used, rem = divmod(64 * moves, budget)
    if 2 * rem > budget or (2 * rem == budget and used % 2 == 1):
        used += 1
    used = max(0, min(64, used))
    state.by_name("budget_bar").pixels = [[12] * (64 - used) + [11] * used]


def _solved(state):
    for p in state.by_tag("pattern"):
        centre = p.pixels[1][1]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                flag = p.pixels[1 + dy][1 + dx]
                if flag not in (0, 2):
                    continue
                t = state.by_name(_tile_name(p.x + 4 * dx, p.y + 4 * dy))
                if t is None or "plain" not in t.tags:
                    continue
                if (_colour(t) == centre) != (flag == 0):
                    return False
    return True


def _cycle(state, sprite):
    cycle = state.vars["cycle"]
    col = _colour(sprite)
    if col not in cycle:
        return
    sprite.color_remap(col, cycle[(cycle.index(col) + 1) % len(cycle)])


def step(state: State, action: Action) -> None:
    if state.status != "playing" or action.id != 6 or action.cell is None:
        return
    clicked = state.sprite_at(action.cell[0], action.cell[1], tag="tile")
    if clicked is None or "plain" not in clicked.tags:
        return  # clicking a pattern tile or the background does nothing
    targets = [clicked]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dx, dy) == (0, 0) or clicked.pixels[1 + dy][1 + dx] != 6:
                continue
            t = state.by_name(_tile_name(clicked.x + 4 * dx, clicked.y + 4 * dy))
            if t is not None and "plain" in t.tags:
                targets.append(t)
    for t in targets:
        _cycle(state, t)
    state.vars["moves"] += 1
    _draw_bar(state)
    if _solved(state):
        state.status = "level_solved"
    elif state.vars["moves"] >= state.vars["budget"]:
        state.status = "game_over"


LEVELS = [
    dict(
        cycle=[9, 8],
        budget=32,
        grid=[
            "55555555555555555555555555555555",
            "55999588859995555559995888599955",
            "55999588859995555559995888599955",
            "55999588859995555559995888599955",
            "55555555555555555555555555555555",
            "55888520259995555558885202588855",
            "55888508259995555558885080588855",
            "55888520059995555558885202588855",
            "55555555555555555555555555555555",
            "55999588858885555559995888599955",
            "55999588858885555559995888599955",
            "55999588858885555559995888599955",
            "55555555555555555555555555555555",
            "55555555555555555555555555555555",
            "55555555555555555555555555555555",
            "55555555555555555555555555555555",
            "55555555555555552224444444442225",
            "55555555555555552444444444444425",
            "55888599959995552499949994999425",
            "55888599959995554499949994999445",
            "55888599959995554499949994999445",
            "55555555555555554444444444444445",
            "55888502258885554499940224999445",
            "55888508058885554499940804999445",
            "55888522058885554499940224999445",
            "55555555555555554444444444444445",
            "55999599958885554499949994999445",
            "55999599958885554499949994999445",
            "55999599958885552499949994999425",
            "55555555555555552444444444444425",
            "55555555555555552224444444442225",
            "55555555555555555555555555555555",
        ],
        tiles=[
            (2, 1, ["999", "999", "999"]),
            (6, 1, ["888", "888", "888"]),
            (10, 1, ["999", "999", "999"]),
            (19, 1, ["999", "999", "999"]),
            (23, 1, ["888", "888", "888"]),
            (27, 1, ["999", "999", "999"]),
            (2, 5, ["888", "888", "888"]),
            (6, 5, ["202", "082", "200"]),
            (10, 5, ["999", "999", "999"]),
            (19, 5, ["888", "888", "888"]),
            (23, 5, ["202", "080", "202"]),
            (27, 5, ["888", "888", "888"]),
            (2, 9, ["999", "999", "999"]),
            (6, 9, ["888", "888", "888"]),
            (10, 9, ["888", "888", "888"]),
            (19, 9, ["999", "999", "999"]),
            (23, 9, ["888", "888", "888"]),
            (27, 9, ["999", "999", "999"]),
            (2, 18, ["888", "888", "888"]),
            (6, 18, ["999", "999", "999"]),
            (10, 18, ["999", "999", "999"]),
            (18, 18, ["999", "999", "999"]),
            (22, 18, ["999", "999", "999"]),
            (26, 18, ["999", "999", "999"]),
            (2, 22, ["888", "888", "888"]),
            (6, 22, ["022", "080", "220"]),
            (10, 22, ["888", "888", "888"]),
            (18, 22, ["999", "999", "999"]),
            (22, 22, ["022", "080", "022"]),
            (26, 22, ["999", "999", "999"]),
            (2, 26, ["999", "999", "999"]),
            (6, 26, ["999", "999", "999"]),
            (10, 26, ["888", "888", "888"]),
            (18, 26, ["999", "999", "999"]),
            (22, 26, ["999", "999", "999"]),
            (26, 26, ["999", "999", "999"]),
        ],
    ),
    dict(
        cycle=[9, 12],
        budget=32,
        grid=[
            "44444444444444444444444444444499",
            "44444444444444444444444444444499",
            "444444444444444444444444444444cc",
            "444444444444444444444444444444cc",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444444444444444444444444444",
            "44444444449994022499944444444444",
            "444444444499940c0499944444444444",
            "44444444449994020499944444444444",
            "44444444444444444444444444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444444444444444444444444444",
            "44444444449994020499944444444444",
            "444444444499942c2499944444444444",
            "44444444449994002499944444444444",
            "44444444444444444444444444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
        ],
        tiles=[
            (10, 7, ["999", "999", "999"]),
            (14, 7, ["999", "999", "999"]),
            (18, 7, ["999", "999", "999"]),
            (10, 11, ["999", "999", "999"]),
            (14, 11, ["022", "0c0", "020"]),
            (18, 11, ["999", "999", "999"]),
            (10, 15, ["999", "999", "999"]),
            (14, 15, ["999", "999", "999"]),
            (18, 15, ["999", "999", "999"]),
            (10, 19, ["999", "999", "999"]),
            (14, 19, ["020", "2c2", "002"]),
            (18, 19, ["999", "999", "999"]),
            (10, 23, ["999", "999", "999"]),
            (14, 23, ["999", "999", "999"]),
            (18, 23, ["999", "999", "999"]),
        ],
    ),
    dict(
        cycle=[8, 12],
        budget=96,
        grid=[
            "44444444444444444444444444444488",
            "44444444444444444444444444444488",
            "444444444488848884888444444444cc",
            "444444444488848884888444444444cc",
            "44444444448884888488844444444444",
            "44444444444444444444444444444444",
            "44444444448884000488844444444444",
            "444444444488840c2488844444444444",
            "44444444448884202488844444444444",
            "44444444444444444444444444444444",
            "44444488848884888488848884444444",
            "44444488848884888488848884444444",
            "44444488848884888488848884444444",
            "44444444444444444444444444444444",
            "44444488842024888420048884444444",
            "44444488842804888408248884444444",
            "44444488840024888420248884444444",
            "44444444444444444444444444444444",
            "44444488848884888488848884444444",
            "44444488848884888488848884444444",
            "44444488848884888488848884444444",
            "44444444444444444444444444444444",
            "44444444448884202488844444444444",
            "444444444488840c2488844444444444",
            "44444444448884000488844444444444",
            "44444444444444444444444444444444",
            "44444444448884888488844444444444",
            "44444444448884888488844444444444",
            "44444444448884888488844444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
        ],
        tiles=[
            (10, 2, ["888", "888", "888"]),
            (14, 2, ["888", "888", "888"]),
            (18, 2, ["888", "888", "888"]),
            (10, 6, ["888", "888", "888"]),
            (14, 6, ["000", "0c2", "202"]),
            (18, 6, ["888", "888", "888"]),
            (6, 10, ["888", "888", "888"]),
            (10, 10, ["888", "888", "888"]),
            (14, 10, ["888", "888", "888"]),
            (18, 10, ["888", "888", "888"]),
            (22, 10, ["888", "888", "888"]),
            (6, 14, ["888", "888", "888"]),
            (10, 14, ["202", "280", "002"]),
            (14, 14, ["888", "888", "888"]),
            (18, 14, ["200", "082", "202"]),
            (22, 14, ["888", "888", "888"]),
            (6, 18, ["888", "888", "888"]),
            (10, 18, ["888", "888", "888"]),
            (14, 18, ["888", "888", "888"]),
            (18, 18, ["888", "888", "888"]),
            (22, 18, ["888", "888", "888"]),
            (10, 22, ["888", "888", "888"]),
            (14, 22, ["202", "0c2", "000"]),
            (18, 22, ["888", "888", "888"]),
            (10, 26, ["888", "888", "888"]),
            (14, 26, ["888", "888", "888"]),
            (18, 26, ["888", "888", "888"]),
        ],
    ),
    dict(
        cycle=[9, 8, 12],
        budget=96,
        grid=[
            "44444444444444444444444444444499",
            "44444444444444444444444444444499",
            "44444444444444444444444444444488",
            "44444444444444444444444444444488",
            "444444444444444444444444444444cc",
            "444444444444444444444444444444cc",
            "44444444444444444444444444444444",
            "44444499949994999499949994444444",
            "44444499949994999499949994444444",
            "44444499949994999499949994444444",
            "44444444444444444444444444444444",
            "44444499942024999420249994444444",
            "44444499942c24999429249994444444",
            "44444499942024999422049994444444",
            "44444444444444444444444444444444",
            "44444499949994999499949994444444",
            "44444499949994999499949994444444",
            "44444499949994999499949994444444",
            "44444444444444444444444444444444",
            "44444444449994022499944444444444",
            "444444444499942c2499944444444444",
            "44444444449994000499944444444444",
            "44444444444444444444444444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444449994999499944444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
        ],
        tiles=[
            (6, 7, ["999", "999", "999"]),
            (10, 7, ["999", "999", "999"]),
            (14, 7, ["999", "999", "999"]),
            (18, 7, ["999", "999", "999"]),
            (22, 7, ["999", "999", "999"]),
            (6, 11, ["999", "999", "999"]),
            (10, 11, ["202", "2c2", "202"]),
            (14, 11, ["999", "999", "999"]),
            (18, 11, ["202", "292", "220"]),
            (22, 11, ["999", "999", "999"]),
            (6, 15, ["999", "999", "999"]),
            (10, 15, ["999", "999", "999"]),
            (14, 15, ["999", "999", "999"]),
            (18, 15, ["999", "999", "999"]),
            (22, 15, ["999", "999", "999"]),
            (10, 19, ["999", "999", "999"]),
            (14, 19, ["022", "2c2", "000"]),
            (18, 19, ["999", "999", "999"]),
            (10, 23, ["999", "999", "999"]),
            (14, 23, ["999", "999", "999"]),
            (18, 23, ["999", "999", "999"]),
        ],
    ),
    dict(
        cycle=[14, 15],
        budget=128,
        grid=[
            "444444444444444444444444444ee444",
            "444444444444444444444444444ee444",
            "44444443334eee4eee444444444ff444",
            "44444443e04eee4eee444444444ff444",
            "44444443024eee4eee44444444444444",
            "44444444444444444444444444444444",
            "4444444eee4e6e4eee40334444444444",
            "4444444eee46e64eee42f34444444444",
            "4444444eee4e6e4eee40204444444444",
            "44444444444444444444444444444444",
            "444eee4eee4eee4eee4eee4eee4eee44",
            "444eee4eee4eee4eee4eee4eee4eee44",
            "444eee4eee4eee4eee4eee4eee4eee44",
            "44444444444444444444444444444444",
            "4443204eee4e6e4eee42024eee420344",
            "4443f24eee46e64eee40e04eee40e344",
            "4443204eee4e6e4eee42024eee420344",
            "44444444444444444444444444444444",
            "444eee4eee40204eee4eee4eee4eee44",
            "444eee4eee42e24eee4eee4eee4eee44",
            "444eee4eee40304eee4eee4eee4eee44",
            "44444444444444444444444444444444",
            "4444444eee42324eee4e6e4eee444444",
            "4444444eee40e04eee46e64eee444444",
            "4444444eee42024eee4e6e4eee444444",
            "44444444444444444444444444444444",
            "4444444eee4eee4eee4eee4203444444",
            "4444444eee4eee4eee4eee40e3444444",
            "4444444eee4eee4eee4eee4333444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
        ],
        tiles=[
            (7, 2, ["333", "3e0", "302"]),
            (11, 2, ["eee", "eee", "eee"]),
            (15, 2, ["eee", "eee", "eee"]),
            (7, 6, ["eee", "eee", "eee"]),
            (11, 6, ["e6e", "6e6", "e6e"]),
            (15, 6, ["eee", "eee", "eee"]),
            (19, 6, ["033", "2f3", "020"]),
            (3, 10, ["eee", "eee", "eee"]),
            (7, 10, ["eee", "eee", "eee"]),
            (11, 10, ["eee", "eee", "eee"]),
            (15, 10, ["eee", "eee", "eee"]),
            (19, 10, ["eee", "eee", "eee"]),
            (23, 10, ["eee", "eee", "eee"]),
            (27, 10, ["eee", "eee", "eee"]),
            (3, 14, ["320", "3f2", "320"]),
            (7, 14, ["eee", "eee", "eee"]),
            (11, 14, ["e6e", "6e6", "e6e"]),
            (15, 14, ["eee", "eee", "eee"]),
            (19, 14, ["202", "0e0", "202"]),
            (23, 14, ["eee", "eee", "eee"]),
            (27, 14, ["203", "0e3", "203"]),
            (3, 18, ["eee", "eee", "eee"]),
            (7, 18, ["eee", "eee", "eee"]),
            (11, 18, ["020", "2e2", "030"]),
            (15, 18, ["eee", "eee", "eee"]),
            (19, 18, ["eee", "eee", "eee"]),
            (23, 18, ["eee", "eee", "eee"]),
            (27, 18, ["eee", "eee", "eee"]),
            (7, 22, ["eee", "eee", "eee"]),
            (11, 22, ["232", "0e0", "202"]),
            (15, 22, ["eee", "eee", "eee"]),
            (19, 22, ["e6e", "6e6", "e6e"]),
            (23, 22, ["eee", "eee", "eee"]),
            (7, 26, ["eee", "eee", "eee"]),
            (11, 26, ["eee", "eee", "eee"]),
            (15, 26, ["eee", "eee", "eee"]),
            (19, 26, ["eee", "eee", "eee"]),
            (23, 26, ["203", "0e3", "333"]),
        ],
    ),
    dict(
        cycle=[11, 14],
        budget=128,
        grid=[
            "444444444444444444444444444444bb",
            "444444444444444444444444444444bb",
            "444444444444444444444444444444ee",
            "44b6b4333444444444444444444444ee",
            "44bbb42e344444444444444444444444",
            "44bbb400244444444444444444444444",
            "44444444444444444444444444444444",
            "44b6b4b6b4b6b4b6b4b6b4b6b4444444",
            "44bbb4bbb4bbb4bbb4bbb4bbb4444444",
            "44bbb4bbb4bbb4bbb4bbb4bbb4444444",
            "44444444444444444444444444444444",
            "444444b6b4b6b4b6b42024b6b4444444",
            "444444bbb4bbb4bbb40e04bbb4444444",
            "444444bbb4bbb4bbb40024bbb4444444",
            "44444444444444444444444444444444",
            "444444b6b42004b6b4b6b4b6b4444444",
            "444444bbb40e04bbb4bbb4bbb4444444",
            "444444bbb42024bbb4bbb4bbb4444444",
            "44444444444444444444444444444444",
            "444444b6b4b6b4b6b4b6b4b6b4b6b444",
            "444444bbb4bbb4bbb4bbb4bbb4bbb444",
            "444444bbb4bbb4bbb4bbb4bbb4bbb444",
            "44444444444444444444444444444444",
            "44444444444444444444442004b6b444",
            "44444444444444444444443e24bbb444",
            "44444444444444444444443334bbb444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
            "44444444444444444444444444444444",
        ],
        tiles=[
            (2, 3, ["b6b", "bbb", "bbb"]),
            (6, 3, ["333", "2e3", "002"]),
            (2, 7, ["b6b", "bbb", "bbb"]),
            (6, 7, ["b6b", "bbb", "bbb"]),
            (10, 7, ["b6b", "bbb", "bbb"]),
            (14, 7, ["b6b", "bbb", "bbb"]),
            (18, 7, ["b6b", "bbb", "bbb"]),
            (22, 7, ["b6b", "bbb", "bbb"]),
            (6, 11, ["b6b", "bbb", "bbb"]),
            (10, 11, ["b6b", "bbb", "bbb"]),
            (14, 11, ["b6b", "bbb", "bbb"]),
            (18, 11, ["202", "0e0", "002"]),
            (22, 11, ["b6b", "bbb", "bbb"]),
            (6, 15, ["b6b", "bbb", "bbb"]),
            (10, 15, ["200", "0e0", "202"]),
            (14, 15, ["b6b", "bbb", "bbb"]),
            (18, 15, ["b6b", "bbb", "bbb"]),
            (22, 15, ["b6b", "bbb", "bbb"]),
            (6, 19, ["b6b", "bbb", "bbb"]),
            (10, 19, ["b6b", "bbb", "bbb"]),
            (14, 19, ["b6b", "bbb", "bbb"]),
            (18, 19, ["b6b", "bbb", "bbb"]),
            (22, 19, ["b6b", "bbb", "bbb"]),
            (26, 19, ["b6b", "bbb", "bbb"]),
            (22, 23, ["200", "3e2", "333"]),
            (26, 23, ["b6b", "bbb", "bbb"]),
        ],
    ),
]
