"""The starting engine module handed to the agent: the structure every real game
follows (sprites, levels, a screen-space HUD, an ARCBaseGame subclass), with no
game-specific content beyond the class name and the advertised actions."""

from __future__ import annotations

TEMPLATE = '''"""Re-implementation of ARC-AGI-3 game "{game}", reverse-engineered from a recorded run.

Contract (the test harness relies on it):
- The module defines one subclass of arcengine.ARCBaseGame, `{cls}`, constructible
  as `{cls}()` (an optional `seed` keyword is allowed).
- It is self-contained: it imports only the standard library, numpy and arcengine,
  and reads no files. All level data lives in this file.
- The harness creates a fresh instance and calls perform_action(...) once per
  recorded action (step 0 is RESET), comparing every frame returned plus state,
  levels_completed, win_levels and available_actions.
- ONLY_RESET_LEVELS=true is set, so RESET restarts the current level.
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
# 1. Sprite art, in logical-grid pixels. -1 = transparent, -2 = invisible but solid.
# ---------------------------------------------------------------------------
SPRITES: dict[str, list[list[int]]] = {{
    # "player": [[9, 9], [9, 9]],
}}


def make_sprite(key: str, x: int, y: int, *, name: str | None = None, layer: int = 0,
                tags: tuple[str, ...] = (), blocking: BlockingMode = BlockingMode.PIXEL_PERFECT,
                **kwargs) -> Sprite:
    return Sprite([row[:] for row in SPRITES[key]], name=name or key, x=x, y=y, layer=layer,
                  tags=list(tags), blocking=blocking, **kwargs)


# ---------------------------------------------------------------------------
# 2. Levels, in order. grid_size=(width, height) is the logical grid; the camera
#    scales it by min(64 // width, 64 // height) and centres it on the 64x64 screen.
#    Each level is rebuilt from this pristine copy on entry and on RESET.
# ---------------------------------------------------------------------------
def build_levels() -> list[Level]:
    return [
        Level(sprites=[], grid_size=(64, 64), name="level1", data={{}}),
    ]


# ---------------------------------------------------------------------------
# 3. Screen-space UI, drawn on the 64x64 frame after the scaled grid
#    (e.g. a move-budget bar in the border).
# ---------------------------------------------------------------------------
class Hud(RenderableUserDisplay):
    def __init__(self, game: "{cls}") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        return frame


# ---------------------------------------------------------------------------
# 4. The game.
# ---------------------------------------------------------------------------
class {cls}(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        # Attributes used by on_set_level must exist before super().__init__,
        # which calls set_level(0) -> on_set_level.
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=0, letter_box=0, interfaces=[self.hud])
        super().__init__(game_id="{game}", levels=build_levels(), camera=camera,
                         available_actions={actions})

    def on_set_level(self, level: Level) -> None:
        """Runs on entering a level and on every level reset: (re)initialise per-level state."""

    def step(self) -> None:
        """Called repeatedly until complete_action(); every call renders one frame."""
        action = self.action.id
        if action == GameAction.ACTION1:  # up
            pass
        elif action == GameAction.ACTION2:  # down
            pass
        elif action == GameAction.ACTION3:  # left
            pass
        elif action == GameAction.ACTION4:  # right
            pass
        elif action == GameAction.ACTION5:  # interact
            pass
        elif action == GameAction.ACTION6:  # click at screen pixel (x, y)
            x, y = self.action.data.get("x", 0), self.action.data.get("y", 0)
            cell = self.camera.display_to_grid(x, y)  # (grid_x, grid_y), or None in the letterbox
        self.complete_action()
'''


def class_name(game: str) -> str:
    return game[:4].capitalize()


def render_skeleton(game: str, available_actions: list[int]) -> str:
    return TEMPLATE.format(game=game[:4], cls=class_name(game), actions=list(available_actions))
