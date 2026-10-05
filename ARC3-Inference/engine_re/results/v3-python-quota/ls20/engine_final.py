"""Re-implementation of ARC-AGI-3 game "ls20" (data literal generated below).

Established facts (from the recording):
 * screen == logical grid 64x64, scale 1, no letterbox; camera background 4.
 * Player: 5x5 sprite (rows of 12 orange, then 9 blue). One direction action
   moves it one 5-px cell (x always 4 mod 5, y always 0 mod 5).
 * HUD bottom-left: 10x10 black framed panel holding a 3x3 grid of 2x2 lit
   cells at cols 3-4/5-6/7-8, rows 55-56/57-58/59-60.
 * HUD bottom: budget bar rows 61-62 cols 13..54 (light blue 11, vacated cells
   become 3), shrinking from the left; lives = red pairs at cols 56-57,59-60,
   62-63 rows 61-62.
 * Objects: white/grey "plus" gems advance the panel pattern; light-blue rings
   refill the budget; the 9x9 black goal box shows a 3x3 arrow pattern.
"""
from __future__ import annotations

import json

import numpy as np
from arcengine import (ARCBaseGame, BlockingMode, Camera, GameAction, GameState,
                       InteractionMode, Level, RenderableUserDisplay, Sprite)

HEX = "0123456789abcdef"
INTANG = InteractionMode.INTANGIBLE
NOBLOCK = BlockingMode.NOT_BLOCKED

PLAYER_PX = [[12] * 5, [12] * 5, [9] * 5, [9] * 5, [9] * 5]
GEM_PX = [[3, 0, 3], [1, 0, 0], [3, 1, 3]]
RING_PX = [[11, 11, 11], [11, 3, 11], [11, 11, 11]]
BAR_FULL = 42
MAX_LIVES = 3
SOLID = (4, 5)

DATA = json.loads(r"{}")


def grid(rows):
    return np.array([[int(ch, 16) for ch in r] for r in rows], dtype=np.int8)


def make_levels():
    levels = []
    for i, d in enumerate(DATA["levels"]):
        levels.append(Level(sprites=[], grid_size=(64, 64), name="l%d" % (i + 1),
                            data={"idx": i}))
    return levels


class Hud(RenderableUserDisplay):
    def render_interface(self, frame):
        return frame


class Ls20(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = Hud()
        camera = Camera(0, 0, 64, 64, background=4, letter_box=4, interfaces=[self.hud])
        super().__init__(game_id="ls20", levels=make_levels(), camera=camera,
                         available_actions=[1, 2, 3, 4])

    def on_set_level(self, level):
        pass

    def step(self):
        self.complete_action()
