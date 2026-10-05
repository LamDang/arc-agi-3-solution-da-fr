"""vc33 engine."""
import json
import numpy as np
from arcengine import (ARCBaseGame, BlockingMode, Camera, GameAction, GameState,
                       InteractionMode, Level, RenderableUserDisplay, Sprite)

BUDGET = [50, 50, 75, 50, 200, 50, 200, 200]
BASE = {}   # injected


def hexgrid(rows):
    return np.array([[int(ch, 16) for ch in r] for r in rows], dtype=np.int8)


class Hud(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        b = BUDGET[self.game.level_index]
        k = min(self.game._action_count, b)
        w = 64 - int(np.floor(64.0 * k / b + 0.5))
        frame[0, :] = 7
        if w < 64:
            frame[0, w:] = 4
        return frame
