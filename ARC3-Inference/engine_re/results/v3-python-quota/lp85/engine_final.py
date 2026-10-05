"""Re-implementation of ARC-AGI-3 game "lp85", reverse-engineered from a recorded run.

The game is a set of "rotating ring" puzzles.  Each level is a grid of cells
(pitch/size depend on the level) drawn on a 64x64 screen at scale 1.  Cells hold
coloured tokens (or are empty).  Red/green arrow sprites sit at the ends of closed
loops of cells; clicking a red arrow shifts every token one step backwards along
its loop, clicking a green arrow shifts them forwards.  Blue corner brackets mark
target cells; when every target cell holds a token of the bracket colour the level
is complete.  A click that moves no token is rejected (and costs nothing).  A green
bar in the left border counts the moves used against the level's budget.

Contract:
- defines one subclass of arcengine.ARCBaseGame, `Lp85`, constructible as `Lp85()`.
- self-contained: stdlib + numpy + arcengine only; all level data lives in this file.
"""

from __future__ import annotations

import numpy as np
from arcengine import (
    ARCBaseGame, BlockingMode, Camera, GameAction, GameState,
    InteractionMode, Level, RenderableUserDisplay, Sprite,
)

# Geometry per level: (nr, nc, pitch, cell_size, origin_y, origin_x) in logical pixels.
GEOM = {0: (5, 7, 6, 4, 19, 12), 1: (10, 10, 3, 2, 17, 17), 2: (7, 11, 3, 2, 19, 15),
        3: (15, 15, 3, 2, 9, 9), 4: (9, 5, 6, 4, 6, 17), 5: (17, 20, 3, 2, 6, 5),
        6: (6, 8, 3, 2, 23, 20), 7: (16, 15, 3, 3, 6, 3)}
BUDGETS = {0: 13, 1: 60, 2: 80, 3: 150, 4: 80, 5: 80, 6: 80, 7: 80}

LEVEL_DATA = {}   # generated; see below


class Lp85(ARCBaseGame):
    ONLY_RESET_LEVELS = True

    def __init__(self, seed: int = 0):
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=3, letter_box=3, interfaces=[self.hud])
        super().__init__(game_id="lp85", levels=build_levels(), camera=camera,
                         available_actions=[6])

    def on_set_level(self, level):
        pass

    def step(self):
        self.complete_action()


def build_levels():
    return [Level(sprites=[], grid_size=(64, 64), name="level%d" % (i + 1), data={})
            for i in range(8)]


class Hud(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        return frame
