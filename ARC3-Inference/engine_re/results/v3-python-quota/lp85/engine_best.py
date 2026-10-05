"""Re-implementation of ARC-AGI-3 game "lp85", reverse-engineered from a recorded run.

NOTES (established from the recording)
======================================
Screen: 64x64, scale 1, background 3 (dark grey), letter_box 3.

HUD (screen pixels, drawn by an interface):
 - col 0, rows 0..63: vertical BUDGET bar. green(14) = remaining, black(5) = used,
   depleting from the TOP, 5 px per successful click.
 - row 1: 8 level slots of 4 px at cols 10-13,15-18,...,45-48 (pitch 5);
   black(5) = not completed, green(14) = completed. Count of green == current level index.

Gameplay model (level 0 confirmed):
 - A rectangular "board" of colour 4 with a grid of cells; cells are kxk blocks at pitch p.
 - The perimeter cells of a sub-rectangle form a RING of 2*(w+h)-4 cells.
 - Two arrow buttons (red=left/counter-clockwise, green=right/clockwise) sit at the ring's
   middle row, one to the left and one to the right of the ring.
 - Clicking an arrow rotates the ring by one step (costs 5 px of budget).
 - Corner brackets (4 small blocks) mark the TARGET cell; the unique blue(11) token must
   be rotated onto the target cell -> level complete.
 - 8 levels; completing the last -> WIN.

Levels observed (first step of each): 0->step0, 1->8, 2->17, 3->36, 4->53, 5->65, 6->96, 7->105
(level 7 completed at step 119 -> WIN, levels_completed=8).
"""

from __future__ import annotations

import numpy as np
from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    GameState,
    InteractionMode,
    Level,
    RenderableUserDisplay,
    Sprite,
)

BG = 3
BOARD = 4
GREEN = 14
BLACK = 5
RED = 8

# ---------------------------------------------------------------------------
# arrow sprite shapes (row-major, -1 transparent)
# ---------------------------------------------------------------------------
def _arrow_left(color=RED):
    """8 tall x 6 wide left-pointing blob (level 0 style)."""
    rows = [
        "..####",
        "..####",
        ".#####",
        "######",
        "######",
        ".#####",
        "..####",
        "..####",
    ]
    return _from_pattern(rows, color)


def _from_pattern(rows, color):
    out = []
    for r in rows:
        out.append([color if c == "#" else -1 for c in r])
    return out


class RingLevel:
    """Data describing one level."""

    def __init__(self, board, origin, pitch, size, rings, targets, arrows):
        self.board = board          # (r0, c0, r1, c1) inclusive rect of colour 4
        self.origin = origin        # (row, col) of cell (0,0) top-left
        self.pitch = pitch
        self.size = size
        self.rings = rings          # list of dict: rows, cols, cells(order), values
        self.targets = targets
        self.arrows = arrows


# ---------------------------------------------------------------------------
# 1. Sprite art
# ---------------------------------------------------------------------------
SPRITES: dict[str, list[list[int]]] = {}


def make_sprite(key, x, y, *, name=None, layer=0, tags=(),
                blocking=BlockingMode.PIXEL_PERFECT, **kwargs):
    return Sprite([row[:] for row in SPRITES[key]], name=name or key, x=x, y=y,
                  layer=layer, tags=list(tags), blocking=blocking, **kwargs)


def block_sprite(size, color, name, x, y, layer=0, tags=()):
    px = [[color if i >= 0 else -1 for i in range(size)] for _ in range(size)]
    return Sprite(px, name=name, x=x, y=y, layer=layer, tags=list(tags),
                  blocking=BlockingMode.NOT_BLOCKED)


# ---------------------------------------------------------------------------
# 2. Levels
# ---------------------------------------------------------------------------
def build_levels():
    levels = []
    for spec in SPECS:
        levels.append(build_level(spec))
    return levels


def build_level(spec):
    sprites = []
    r0, c0, r1, c1 = spec["board"]
    sprites.append(Sprite([[BOARD] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)],
                         name="board", x=c0, y=r0, layer=0,
                         blocking=BlockingMode.NOT_BLOCKED))
    for i, ring in enumerate(spec["rings"]):
        cells = ring_cells(ring)
        vals = ring["values"]
        for j, (r, c) in enumerate(cells):
            sprites.append(block_sprite(spec["size"], vals[j],
                                        f"cell_{i}_{j}", c, r, layer=2))
    # arrows
    for a in spec["arrows"]:
        sprites.append(Sprite(_from_pattern(ARROW_SHAPES[a["shape"]], a["color"]),
                              name=a["name"], x=a["c"], y=a["r"], layer=3,
                              blocking=BlockingMode.NOT_BLOCKED))
    # target brackets
    for t in spec["brackets"]:
        sprites.append(Sprite(_from_pattern(BRACKET_PATTERN, t["color"]),
                              name="bracket", x=t["c"], y=t["r"], layer=4,
                              blocking=BlockingMode.NOT_BLOCKED))
    return Level(sprites=sprites, grid_size=(64, 64), name=spec["name"], data=spec)


def ring_cells(ring):
    """Perimeter cells of an r x c block of cells, clockwise starting at top-left."""
    r, c = ring["rows"], ring["cols"]
    out = []
    for j in range(c):
        out.append((0, j))
    for i in range(1, r - 1):
        out.append((i, c - 1))
    for j in range(c - 1, -1, -1):
        out.append((r - 1, j))
    for i in range(r - 2, 0, -1):
        out.append((i, 0))
    return out


ARROW_SHAPES = {}
BRACKET_PATTERN = []
SPECS = []


# ---------------------------------------------------------------------------
# 3. HUD
# ---------------------------------------------------------------------------
class Hud(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        g = self.game
        used = g.used
        for r in range(64):
            frame[r, 0] = BLACK if r < used * 5 else GREEN
        n = g.current_level_index_for_hud()
        for i in range(8):
            c = 10 + 5 * i
            col = GREEN if i < n else BLACK
            frame[1, c:c + 4] = col
        return frame


# ---------------------------------------------------------------------------
# 4. The game
# ---------------------------------------------------------------------------
class Lp85(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.used = 0
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=BG, letter_box=BG,
                        interfaces=[self.hud])
        super().__init__(game_id="lp85", levels=build_levels(), camera=camera,
                         available_actions=[6])

    def current_level_index_for_hud(self):
        return self.level_index

    def on_set_level(self, level):
        self.used = 0

    def step(self):
        if self.action.id == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            self.handle_click(x, y)
        self.complete_action()

    def handle_click(self, x, y):
        pass
