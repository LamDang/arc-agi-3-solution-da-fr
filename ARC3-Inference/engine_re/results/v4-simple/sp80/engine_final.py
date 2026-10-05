"""ARC-AGI-3 game "sp80", reverse-engineered from a recorded run.

Write two functions below the fixed interface:

    make_level(n) -> State     the state at the start of level n (0-based)
    step(state, action)        apply one action to the state, in place

The fixed interface already provides what every game shares: sprites with layers, visibility,
collidability, blocking modes, rotation, mirroring and scale; collisions (state.try_move,
state.collisions); lookups (state.sprite_at, sprites_at, by_tag, by_name); and a per-level view
(state.view: grid scale, and rotation and mirroring of the whole screen). The harness does the rest:
it gives step() a fresh copy of the level's first state on entering a level and on every RESET (RESET
restarts the current level), draws the state, counts completed levels and ends the game with WIN or
GAME_OVER. This game advertises actions [1, 2, 3, 4, 5, 6].

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


# ==== THE GAME ====
#
# A 16x16 field (scale 4, filling the screen) of orange (12).  Along the top of the field a
# nozzle (grey stem + magenta tip) sits in one column: that is where paint is poured from.
# Horizontal bars (the player's "boards": the blue one is the steered board, the red ones are
# the others) float in the field.  Near the bottom sit yellow U-shaped cups, and the bottom row
# is a light grey floor.  A one-pixel-high budget bar is drawn over internal screen row 0:
# green (14) for the moves left, black (0) for the moves spent.
#
# Actions 1-4 move the blue bar (screen-relative: the finished frame is turned by view.rotation,
# so on a rotated level "up" moves the bar down through the field).  Action 6 clicks: clicking a
# bar makes it the steered one.  Action 5 pours paint: it falls from the nozzle, is blocked by
# bars and cup walls, spreads sideways along the row above the blocker (over the blocker's own
# contiguous run, one cell wider on each side) and drips off the ends of that spread.  A drip
# into the inside of a cup fills it; paint that reaches the floor is wasted; paint that cannot
# even start to drip means the pour is blocked and the game is lost.  The pour itself leaves no
# trace in the final frame, but it hands control to the highest bar the paint touched.
# Solving the level means the pour fills every cup with nothing wasted.  Every action costs one
# move; running out of moves ends the game.

GRID_W = GRID_H = 16
FLOOR_Y = 15
FIELD = 12
CUP_PIXELS = [[11, -1, 11], [11, 11, 11]]
POINTER_PIXELS = [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4],
                  [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]

# Level data: budget (moves), nozzle column, board (x, y, length) with the first one steered,
# cup (x, y) of the 3x2 U, invisible walls (x, y, w, h) and the view rotation.
LEVELS = [
    dict(budget=30, src=9, rot=0,
         boards=[(3, 4, 5)],
         cups=[(4, 13), (10, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1), (0, 3, 1, 9)]),
    dict(budget=45, src=5, rot=180,
         boards=[(6, 6, 5), (6, 9, 3), (11, 11, 3)],
         cups=[(2, 13), (6, 13), (10, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1), (15, 3, 1, 9)]),
    dict(budget=45, src=11, rot=0,
         boards=[(5, 5, 5), (7, 8, 3), (3, 10, 3)],
         cups=[(1, 13), (5, 13), (9, 13), (12, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1)]),
    dict(budget=55, src=7, rot=180,
         boards=[(4, 4, 5), (2, 7, 3), (9, 9, 3), (5, 11, 3)],
         cups=[(1, 13), (4, 13), (8, 13), (11, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1)]),
    dict(budget=60, src=3, rot=0,
         boards=[(6, 3, 5), (10, 6, 3), (1, 8, 3), (8, 10, 3)],
         cups=[(0, 13), (3, 13), (7, 13), (10, 13), (13, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1), (15, 3, 1, 9)]),
    dict(budget=60, src=12, rot=180,
         boards=[(3, 5, 5), (8, 3, 3), (1, 9, 3), (10, 10, 3), (4, 11, 3)],
         cups=[(1, 13), (4, 13), (7, 13), (10, 13), (13, 13)],
         walls=[(0, 2, 16, 1), (0, 12, 16, 1)]),
]


def level_data(n: int) -> dict:
    return LEVELS[n % len(LEVELS)]


def bar_pixels(budget: int, maximum: int) -> list:
    w = max(0, min(64, int(64.0 * budget / maximum + 0.5)))
    return [[14] * w + [0] * (64 - w)]


def make_level(n: int) -> State:
    """The state at the start of level n."""
    data = level_data(n)
    sprites = [
        Sprite([[5] * 64 for _ in range(64)], screen=True, layer=-10, collidable=False, name="border"),
        Sprite([[FIELD] * GRID_W for _ in range(GRID_H)], layer=-1, collidable=False, name="background"),
        Sprite([[1] * GRID_W], x=0, y=FLOOR_Y, layer=0, collidable=True, name="floor", tags=("floor",)),
    ]
    for i, (wx, wy, ww, wh) in enumerate(data["walls"]):
        sprites.append(Sprite([[-2] * ww for _ in range(wh)], x=wx, y=wy, layer=1,
                              visible=False, collidable=True, name="wall%d" % i, tags=("wall",)))
    for i, (cx, cy) in enumerate(data["cups"]):
        sprites.append(Sprite([list(r) for r in CUP_PIXELS], x=cx, y=cy, layer=1,
                              collidable=True, name="cup%d" % i, tags=("cup",)))
    for i, (bx, by, bw) in enumerate(data["boards"]):
        col = 9 if i == 0 else 8
        sprites.append(Sprite([[col] * bw], x=bx, y=by, layer=3 if i == 0 else 2,
                              collidable=True, name="board%d" % i, tags=("board",)))
    sprites.append(Sprite([list(r) for r in POINTER_PIXELS], x=data["src"] * 4, y=1, layer=5,
                          screen=True, collidable=False, name="pointer", tags=("pointer",)))
    sprites.append(Sprite(bar_pixels(data["budget"], data["budget"]), screen=True, layer=9,
                         collidable=False, name="bar", tags=("bar",)))
    return State(
        grid=(GRID_W, GRID_H),
        sprites=sprites,
        vars={"budget": data["budget"], "max_budget": data["budget"], "src": data["src"]},
        view=View(scale=4, rotation=data["rot"]),
    )


def boards_of(state: State) -> list:
    return state.by_tag("board")


def steered(state: State):
    for b in boards_of(state):
        if b.pixels[0][0] == 9:
            return b
    return boards_of(state)[0]


def steer(state: State, board) -> None:
    """The steered bar is blue and is drawn on top of the other bars."""
    for b in boards_of(state):
        b.color_remap(None, 9 if b is board else 8)
        b.layer = 3 if b is board else 2


def screen_to_grid_direction(action_id: int, view: View) -> tuple:
    """Turn a screen-relative arrow direction into grid coordinates."""
    dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action_id]
    if view.mirror_lr:
        dx = -dx
    if view.mirror_ud:
        dy = -dy
    for _ in range((view.rotation // 90) % 4):  # screen = grid turned clockwise
        dx, dy = -dy, dx
    return dx, dy


def spend(state: State) -> None:
    state.vars["budget"] -= 1
    if state.vars["budget"] < 0:
        state.vars["budget"] = 0
    bar = state.by_name("bar")
    bar.pixels = bar_pixels(state.vars["budget"], state.vars["max_budget"])
    # The game is lost once there are fewer moves left than there are bars to place.
    if state.vars["budget"] < len(boards_of(state)):
        state.status = "game_over"


def simulate_pour(state: State) -> dict:
    """Run the paint from the nozzle down the field.

    The paint falls in a column until a bar, a cup wall or the floor stops it.  A bar or a cup
    wall makes it spread sideways along the row above - as far as the blocker reaches plus one
    cell on each side - and it runs off both ends of that spread.  Paint that runs into the
    inside of a cup fills it; paint that reaches the floor is wasted; paint that would run off
    the end of a spread straight onto another bar clogs the pour and the game is lost.

    Returns paint (cells), touched (bars the paint brushed), filled (cups), wasted and jammed.
    """
    boards = boards_of(state)
    cups = state.by_tag("cup")
    floor = state.by_tag("floor")[0]

    def blocker(x, y):
        """The object at (x, y) that stops the paint, or None / "edge"."""
        if not (0 <= x < GRID_W) or not (0 <= y < GRID_H):
            return "edge"
        for b in boards:
            if b.y == y and b.x <= x < b.x + b.width:
                return b
        for c in cups:
            if c.x <= x < c.x + c.width and c.y <= y < c.y + c.height and c.pixels[y - c.y][x - c.x] != -1:
                return c
        if floor.y == y and floor.x <= x < floor.x + floor.width:
            return floor
        return None

    paint = set()
    filled = []
    out = {"wasted": False, "jammed": False}
    seen = set()

    def run_at(y, x, sprite):
        """Contiguous cells of `sprite` along row y that include x."""
        a = b = x
        while a - 1 >= 0 and blocker(a - 1, y) is sprite:
            a -= 1
        while b + 1 < GRID_W and blocker(b + 1, y) is sprite:
            b += 1
        return a, b

    def fall(x, y):
        if out["jammed"] or not (0 <= x < GRID_W) or (x, y) in seen:
            return
        seen.add((x, y))
        if any(b is blocker(x, y) for b in boards):
            out["jammed"] = True
            return
        yy = y
        stopped = None
        while yy < GRID_H:
            stopped = blocker(x, yy)
            if stopped is not None and stopped != "edge":
                break
            paint.add((x, yy))
            yy += 1
        if stopped is floor:
            out["wasted"] = True
            return
        if stopped is None or stopped == "edge":
            return
        if stopped in cups and x == stopped.x + 1 and yy == stopped.y + 1 and stopped not in filled:
            filled.append(stopped)
        a, b = run_at(yy, x, stopped)
        lo, hi = max(0, a - 1), min(GRID_W - 1, b + 1)
        l = r = x
        while l - 1 >= lo and blocker(l - 1, yy - 1) is None:
            l -= 1
        while r + 1 <= hi and blocker(r + 1, yy - 1) is None:
            r += 1
        for cx in range(l, r + 1):
            paint.add((cx, yy - 1))
        fall(l, yy)
        fall(r, yy)

    fall(state.vars["src"], 1)

    touched = []
    for b in boards:
        if any((cx + ox, cy + oy) in paint for cy in [b.y] for cx in range(b.x, b.x + b.width)
               for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            touched.append(b)
    return {"paint": paint, "touched": touched, "filled": filled,
            "wasted": out["wasted"], "jammed": out["jammed"]}


def pour(state: State) -> None:
    cups = state.by_tag("cup")
    if state.vars["budget"] < len(boards_of(state)):
        return  # too few moves left to attempt a pour
    result = simulate_pour(state)
    if result["jammed"]:
        state.status = "game_over"
        return
    if len(result["filled"]) == len(cups) and not result["wasted"]:
        state.status = "level_solved"
        return
    if result["touched"]:
        steer(state, min(result["touched"], key=lambda b: (b.y, b.x)))


def move_board(state: State, board, dx: int, dy: int) -> bool:
    """Move a bar one cell, unless it would leave the field or run into an object.

    Bars slide through one another: only walls, cups and the floor hold a bar back.
    """
    if board.x + dx < 0 or board.x + board.width + dx > GRID_W or board.y + dy < 0 \
            or board.y + board.height + dy > GRID_H:
        return False
    board.move(dx, dy)
    others = [s for s in state.sprites if s not in boards_of(state)]
    hit = [s for s in others if board.collides_with(s)]
    if hit:
        board.move(-dx, -dy)
        return False
    return True


def step(state: State, action: Action) -> None:
    """Apply `action` to `state` in place."""
    if state.status != "playing":
        return
    if action.id in (1, 2, 3, 4):
        move_board(state, steered(state), *screen_to_grid_direction(action.id, state.view))
        spend(state)
    elif action.id == 5:
        spend(state)
        if state.status == "playing":
            pour(state)
    elif action.id == 6:
        spend(state)
        if state.status == "playing" and action.cell is not None:
            hit = state.sprite_at(action.cell[0], action.cell[1], tag="board")
            if hit is not None:
                steer(state, hit)
