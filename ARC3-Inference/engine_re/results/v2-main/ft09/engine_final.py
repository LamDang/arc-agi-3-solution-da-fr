"""Re-implementation of ARC-AGI-3 game "ft09", reverse-engineered from a recorded run.

The game is a "colour the board so the mini-map hints are satisfied" puzzle:

* Every level lays out one or more rectangular boards of cells on a 32x32 logical
  grid (scale 2, no letterbox).  A cell is a 3x3 block of logical pixels; cells are
  spaced 4 apart (one background row/column between them).
* A cell is either *clickable* (a flat colour, optionally with a few marker pixels
  of colour 6 = magenta) or a *hint* cell: a noisy 3x3 sprite whose centre pixel is
  the hint's own colour and whose 8 remaining pixels encode the 8 neighbours
  (0 = same colour as the hint, 2 = a different colour, 3 = no clickable cell).
  Hint cells never change and cannot be clicked.
* Clicking a clickable cell cycles its colour through the level palette
  (order given by the legend in the top-right corner).  If the cell's sprite has a
  marker pixel (6) on an edge, the neighbour in that direction is cycled too.
* The level is solved when every hint agrees with its neighbours; then the next
  level starts (an extra frame is rendered).
* The bottom screen row (63) is the action budget bar: remaining actions in orange
  (12) from the left, used ones in yellow (11) at the right.  Only clicks that
  actually change the board and do not finish the level consume budget.
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

# ---------------------------------------------------------------------------
# 1. Level data (extracted from the recorded frames).
#    A cell entry is ("c", palette_index, [(row, col) marker pixels]) for a
#    clickable cell, or ("h", palette_index, 3x3 pixels) for a static hint cell.
# ---------------------------------------------------------------------------
FRAME_SPRITE = [[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2], [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], [2, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 2], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [2, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 2], [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], [2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2]]

LEVELS_DATA = [
  {
    'bg': 5,
    'budget': 32,
    'palette': [9, 8],
    'legend': None,
    'static': [(16, 16, [[2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2], [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], [2, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 2], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [4, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 4], [2, 4, -1, -1, -1, 4, -1, -1, -1, 4, -1, -1, -1, 4, 2], [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2], [2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2]])],
    'boards': [
      {'rows': [1, 5, 9], 'cols': [2, 6, 10], 'cells': [
        [('c', 0, []), ('c', 1, []), ('c', 0, [])],
        [('c', 1, []), ('h', 1, [[2, 0, 2], [0, 8, 2], [2, 0, 0]]), ('c', 0, [])],
        [('c', 0, []), ('c', 1, []), ('c', 1, [])],
      ]},
      {'rows': [1, 5, 9], 'cols': [19, 23, 27], 'cells': [
        [('c', 0, []), ('c', 1, []), ('c', 0, [])],
        [('c', 1, []), ('h', 1, [[2, 0, 2], [0, 8, 0], [2, 0, 2]]), ('c', 1, [])],
        [('c', 0, []), ('c', 1, []), ('c', 0, [])],
      ]},
      {'rows': [18, 22, 26], 'cols': [2, 6, 10], 'cells': [
        [('c', 1, []), ('c', 0, []), ('c', 0, [])],
        [('c', 1, []), ('h', 1, [[0, 2, 2], [0, 8, 0], [2, 2, 0]]), ('c', 1, [])],
        [('c', 0, []), ('c', 0, []), ('c', 1, [])],
      ]},
      {'rows': [18, 22, 26], 'cols': [18, 22, 26], 'cells': [
        [('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('c', 0, []), ('h', 1, [[0, 2, 2], [0, 8, 0], [0, 2, 2]]), ('c', 0, [])],
        [('c', 0, []), ('c', 0, []), ('c', 0, [])],
      ]},
    ],
  },
  {
    'bg': 4,
    'budget': 32,
    'palette': [9, 12],
    'legend': (30, 0),
    'static': [],
    'boards': [
      {'rows': [7, 11, 15, 19, 23], 'cols': [10, 14, 18], 'cells': [
        [('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('c', 0, []), ('h', 1, [[0, 2, 2], [0, 12, 0], [0, 2, 0]]), ('c', 0, [])],
        [('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('c', 0, []), ('h', 1, [[0, 2, 0], [2, 12, 2], [0, 0, 2]]), ('c', 0, [])],
        [('c', 0, []), ('c', 0, []), ('c', 0, [])],
      ]},
    ],
  },
  {
    'bg': 4,
    'budget': 96,
    'palette': [8, 12],
    'legend': (30, 0),
    'static': [],
    'boards': [
      {'rows': [2, 6, 10, 14, 18, 22, 26], 'cols': [6, 10, 14, 18, 22], 'cells': [
        [None, ('c', 0, []), ('c', 0, []), ('c', 0, []), None],
        [None, ('c', 0, []), ('h', 1, [[0, 0, 0], [0, 12, 2], [2, 0, 2]]), ('c', 0, []), None],
        [('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('c', 0, []), ('h', 0, [[2, 0, 2], [2, 8, 0], [0, 0, 2]]), ('c', 0, []), ('h', 0, [[2, 0, 0], [0, 8, 2], [2, 0, 2]]), ('c', 0, [])],
        [('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [None, ('c', 0, []), ('h', 1, [[2, 0, 2], [0, 12, 2], [0, 0, 0]]), ('c', 0, []), None],
        [None, ('c', 0, []), ('c', 0, []), ('c', 0, []), None],
      ]},
    ],
  },
  {
    'bg': 4,
    'budget': 96,
    'palette': [9, 8, 12],
    'legend': (30, 0),
    'static': [],
    'boards': [
      {'rows': [7, 11, 15, 19, 23], 'cols': [6, 10, 14, 18, 22], 'cells': [
        [('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('c', 0, []), ('h', 2, [[2, 0, 2], [2, 12, 2], [2, 0, 2]]), ('c', 0, []), ('h', 0, [[2, 0, 2], [2, 9, 2], [2, 2, 0]]), ('c', 0, [])],
        [('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [None, ('c', 0, []), ('h', 2, [[0, 2, 2], [2, 12, 2], [0, 0, 0]]), ('c', 0, []), None],
        [None, ('c', 0, []), ('c', 0, []), ('c', 0, []), None],
      ]},
    ],
  },
  {
    'bg': 4,
    'budget': 128,
    'palette': [14, 15],
    'legend': (27, 0),
    'static': [],
    'boards': [
      {'rows': [2, 6, 10, 14, 18, 22, 26], 'cols': [3, 7, 11, 15, 19, 23, 27], 'cells': [
        [None, ('h', 0, [[3, 3, 3], [3, 14, 0], [3, 0, 2]]), ('c', 0, []), ('c', 0, []), None, None, None],
        [None, ('c', 0, []), ('c', 0, [[0, 1], [1, 0], [1, 2], [2, 1]]), ('c', 0, []), ('h', 1, [[0, 3, 3], [2, 15, 3], [0, 2, 0]]), None, None],
        [('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [('h', 1, [[3, 2, 0], [3, 15, 2], [3, 2, 0]]), ('c', 0, []), ('c', 0, [[0, 1], [1, 0], [1, 2], [2, 1]]), ('c', 0, []), ('h', 0, [[2, 0, 2], [0, 14, 0], [2, 0, 2]]), ('c', 0, []), ('h', 0, [[2, 0, 3], [0, 14, 3], [2, 0, 3]])],
        [('c', 0, []), ('c', 0, []), ('h', 0, [[0, 2, 0], [2, 14, 2], [0, 3, 0]]), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, [])],
        [None, ('c', 0, []), ('h', 0, [[2, 3, 2], [0, 14, 0], [2, 0, 2]]), ('c', 0, []), ('c', 0, [[0, 1], [1, 0], [1, 2], [2, 1]]), ('c', 0, []), None],
        [None, ('c', 0, []), ('c', 0, []), ('c', 0, []), ('c', 0, []), ('h', 0, [[2, 0, 3], [0, 14, 3], [3, 3, 3]]), None],
      ]},
    ],
  },
  {
    'bg': 4,
    'budget': 128,
    'palette': [11, 14],
    'legend': (30, 0),
    'static': [],
    'boards': [
      {'rows': [3, 7, 11, 15, 19, 23], 'cols': [2, 6, 10, 14, 18, 22, 26], 'cells': [
        [('c', 0, [[0, 1]]), ('h', 1, [[3, 3, 3], [2, 14, 3], [0, 0, 2]]), None, None, None, None, None],
        [('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), None],
        [None, ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('h', 1, [[2, 0, 2], [0, 14, 0], [0, 0, 2]]), ('c', 0, [[0, 1]]), None],
        [None, ('c', 0, [[0, 1]]), ('h', 1, [[2, 0, 0], [0, 14, 0], [2, 0, 2]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), None],
        [None, ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]]), ('c', 0, [[0, 1]])],
        [None, None, None, None, None, ('h', 1, [[2, 0, 0], [3, 14, 2], [3, 3, 3]]), ('c', 0, [[0, 1]])],
      ]},
    ],
  },
]


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
GRID_W = GRID_H = 32
SCALE = 2


def cell_pixels(color, markers):
    """3x3 pixel block of `color` with colour-6 marker pixels punched in."""
    px = [[color] * 3 for _ in range(3)]
    for i, j in markers:
        px[i][j] = 6
    return px


def make_sprite(px, name, x, y, layer):
    return Sprite(np.array(px, dtype=np.int8), name=name, x=x, y=y, layer=layer,
                  interaction=InteractionMode.INTANGIBLE,
                  blocking=BlockingMode.NOT_BLOCKED)


def build_levels():
    levels = []
    for li, d in enumerate(LEVELS_DATA):
        sprites = []
        pal = d["palette"]
        for bi, b in enumerate(d["boards"]):
            rows, cols, mat = b["rows"], b["cols"], b["cells"]
            for i, line in enumerate(mat):
                for j, e in enumerate(line):
                    if e is None:
                        continue
                    if e[0] == "c":
                        px = cell_pixels(pal[e[1]], e[2])
                        nm = "c%d_%d_%d" % (bi, i, j)
                    else:
                        px = e[2]
                        nm = "h%d_%d_%d" % (bi, i, j)
                    sprites.append(make_sprite(px, nm, cols[j], rows[i], 3))
        for si, (sx, sy, px) in enumerate(d["static"]):
            sprites.append(make_sprite(px, "s%d_%d" % (li, si), sx, sy, 2))
        leg = d["legend"]
        if leg is not None:
            lx, ly = leg
            for k, col in enumerate(pal):
                sprites.append(make_sprite([[col, col], [col, col]],
                                           "leg%d_%d" % (li, k), lx, ly + 2 * k, 4))
        levels.append(Level(sprites=sprites, grid_size=(GRID_W, GRID_H),
                            name="level%d" % (li + 1),
                            data={"bg": d["bg"], "budget": d["budget"],
                                  "palette": d["palette"], "boards": d["boards"],
                                  "legend": d["legend"]}))
    return levels


# ---------------------------------------------------------------------------
# 3. Screen-space UI: the action budget bar on the bottom screen row.
# ---------------------------------------------------------------------------
class Hud(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        budget = self.game.budget or 1
        used = min(self.game.used, budget)
        n_used = int(round(used * 64.0 / budget))
        n_used = max(0, min(64, n_used))
        if n_used:
            frame[63, 64 - n_used:] = 11
        frame[63, :64 - n_used] = 12
        return frame


# ---------------------------------------------------------------------------
# 4. The game.
# ---------------------------------------------------------------------------
class Ft09(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.used = 0
        self.budget = 1
        self.palette = []
        self.model = []
        self.constraints = []
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=5, letter_box=5, interfaces=[self.hud])
        super().__init__(game_id="ft09", levels=build_levels(), camera=camera,
                         available_actions=[6])

    # -- (re)initialisation ------------------------------------------------
    def on_set_level(self, level: Level) -> None:
        data = {k: level.get_data(k) for k in
                ("bg", "budget", "palette", "boards", "legend")}
        self.used = 0
        self.budget = data["budget"]
        self.palette = list(data["palette"])
        bg = data["bg"]
        self.camera.background = bg
        self.camera.letter_box = bg
        self.model = []
        for bi, b in enumerate(data["boards"]):
            rows, cols, mat = b["rows"], b["cols"], b["cells"]
            grid = []
            for i, line in enumerate(mat):
                grow = []
                for j, e in enumerate(line):
                    if e is None:
                        grow.append(None)
                        continue
                    nm = ("c%d_%d_%d" if e[0] == "c" else "h%d_%d_%d") % (bi, i, j)
                    spr = level.get_sprites_by_name(nm)[0]
                    grow.append({"kind": e[0], "idx": e[1], "markers": e[2] if e[0] == "c" else [],
                                 "sprite": spr, "bi": bi, "i": i, "j": j,
                                 "row": rows[i], "col": cols[j]})
                grid.append(grow)
            self.model.append(grid)
        # hint constraints: (hint cell, clickable neighbour, code)
        self.constraints = []
        for grid in self.model:
            nr, nc = len(grid), len(grid[0])
            for i in range(nr):
                for j in range(nc):
                    cell = grid[i][j]
                    if cell is None or cell["kind"] != "h":
                        continue
                    pat = cell["markers"] if False else cell["sprite"].pixels
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            code = int(pat[di + 1][dj + 1])
                            ii, jj = i + di, j + dj
                            if not (0 <= ii < nr and 0 <= jj < nc):
                                continue
                            nb = grid[ii][jj]
                            if nb is None or nb["kind"] != "c":
                                continue
                            self.constraints.append((cell, nb, code))

    # -- rules -------------------------------------------------------------
    def paint(self, cell):
        cell["sprite"].pixels = np.array(
            cell_pixels(self.palette[cell["idx"]], cell["markers"]), dtype=np.int8)

    def cycle(self, cell):
        if cell is None or cell["kind"] != "c":
            return
        cell["idx"] = (cell["idx"] + 1) % len(self.palette)
        self.paint(cell)

    def neighbour(self, cell, di, dj):
        grid = self.model[cell["bi"]]
        i, j = cell["i"] + di, cell["j"] + dj
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
            return grid[i][j]
        return None

    def cell_at(self, gx, gy):
        for grid in self.model:
            for row in grid:
                for cell in row:
                    if cell is not None and cell["col"] <= gx <= cell["col"] + 2 \
                            and cell["row"] <= gy <= cell["row"] + 2:
                        return cell
        return None

    def solved(self):
        for hint, cell, code in self.constraints:
            same = hint["idx"] == cell["idx"]
            if code == 0:
                if not same:
                    return False
            elif code == 2:
                if same:
                    return False
            else:
                return False
        return True

    def step(self) -> None:
        if self.action is not None and self.action.id == GameAction.ACTION6:
            x = self.action.data.get("x", -1)
            y = self.action.data.get("y", -1)
            g = self.camera.display_to_grid(x, y)
            if g is not None:
                cell = self.cell_at(g[0], g[1])
                if cell is not None and cell["kind"] == "c":
                    markers = {tuple(m) for m in cell["markers"]}
                    self.cycle(cell)
                    if (0, 1) in markers:
                        self.cycle(self.neighbour(cell, -1, 0))
                    if (2, 1) in markers:
                        self.cycle(self.neighbour(cell, 1, 0))
                    if (1, 0) in markers:
                        self.cycle(self.neighbour(cell, 0, -1))
                    if (1, 2) in markers:
                        self.cycle(self.neighbour(cell, 0, 1))
                    if self.solved():
                        self.next_level()
                    else:
                        self.used += 1
                        if self.used >= self.budget:
                            self.lose()
        self.complete_action()


GAME = Ft09
