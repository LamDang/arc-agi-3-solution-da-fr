"""The starting engine module handed to the agent.

It has code for only the three things the agent must write (sprites, levels and
the game's step), and explains the rest in comments: where a game keeps its
state, how levels are rebuilt on entry and RESET, screen-space UI, and that only
the final frame of each action is checked. No game-specific content beyond the
class name and the advertised actions.
"""

from __future__ import annotations

TEMPLATE = '''"""Re-implementation of ARC-AGI-3 game "{game}", reverse-engineered from a recorded run.

What is tested: a fresh {cls}() replays the recorded actions (step 0 is RESET). After every action,
the LAST frame your engine returns must equal the recorded final frame, and state, levels_completed,
win_levels and available_actions must match. Animation frames are not compared.

Rules: one ARCBaseGame subclass, constructible as {cls}(); standard library, numpy and arcengine
only; no file reads, so all sprite and level data lives in this file. RESET restarts the current
level (ONLY_RESET_LEVELS=true); after a game over only RESET is accepted (the base class handles it).
"""

from __future__ import annotations

from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# =============================================================================
# 1. SPRITES: one prototype per kind of object.
#
# A Sprite is the picture AND the state of an object: x, y, pixels, rotation,
# layer, visibility and collision are all mutable, and the screen is drawn from
# them. Pixels are colour indices 0-15; -1 is transparent, -2 is invisible but
# solid. Tags let the game find objects again: level.get_sprites_by_tag("wall").
# Collision is pixel-exact by default (blocking=BlockingMode.PIXEL_PERFECT);
# use BlockingMode.NOT_BLOCKED for things you can walk over.
# Levels never use a prototype directly; they place clones of it (place()).
# =============================================================================
SPRITES: dict[str, Sprite] = {{
    # "player": Sprite(pixels=[[9, 9], [9, 9]], name="player", tags=["player"], layer=1),
    # "wall":   Sprite(pixels=[[4]], name="wall", tags=["wall"]),
    # "goal":   Sprite(pixels=[[14]], name="goal", tags=["goal"], blocking=BlockingMode.NOT_BLOCKED),
}}


def place(key: str, x: int, y: int) -> Sprite:
    """A fresh copy of a prototype at logical-grid position (x, y) (x = column, y = row)."""
    return SPRITES[key].clone().set_position(x, y)


# =============================================================================
# 2. LEVELS, in order.
#
# A Level is the list of sprites present when it starts, plus `data`: per-level
# settings (move budget, target colour, ...) read with level.get_data(key).
# grid_size=(width, height) is the logical grid; the camera scales it by
# min(64 // width, 64 // height) and centres it on the 64x64 screen.
#
# The engine keeps a pristine copy of every level. Entering a level and RESET
# both replace the live level with a fresh copy and then call on_set_level(),
# so everything the player changed (positions, removed objects) is undone.
# =============================================================================
LEVELS: list[Level] = [
    Level(
        sprites=[
            # place("player", 1, 1),
            # place("wall", 0, 0),
        ],
        grid_size=(64, 64),
        data={{}},  # e.g. {{"budget": 30}}
        name="level1",
    ),
]


# =============================================================================
# 3. THE GAME
#
# Where the game's state lives:
# - Visible state: the sprites of self.current_level. Move, recolour or hide
#   them (sprite.set_position, sprite.pixels / color_remap, set_visible).
# - Hidden state (counters, modes, what is selected, ...): plain attributes on
#   self. Per-level ones are set in on_set_level(), which runs on level entry
#   and on every RESET, so a RESET restores them. Game-wide ones are set in
#   __init__ BEFORE super().__init__(), because super().__init__() already calls
#   on_set_level() for the first level.
# - Bookkeeping kept by ARCBaseGame (don't duplicate it): self.level_index,
#   self.current_level, self._state, self._score (= levels_completed),
#   self._action_count.
#
# Screen-space UI (a budget bar, lives, ...) is drawn on the 64x64 frame after
# the scaled grid by a RenderableUserDisplay passed in Camera(interfaces=[...]);
# its render_interface(frame) edits and returns the frame (numpy array, [row, col]).
# Add one only if the recording shows such UI.
# =============================================================================
class {cls}(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        camera = Camera(0, 0, 64, 64, background=0, letter_box=0, interfaces=[])
        super().__init__(game_id="{game}", levels=LEVELS, camera=camera, available_actions={actions})

    def on_set_level(self, level: Level) -> None:
        """Level entry and RESET: find this level's sprites and reset per-level state."""
        # self.player = self.current_level.get_sprites_by_tag("player")[0]
        # self.budget = self.current_level.get_data("budget")

    def step(self) -> None:
        """One action: apply its whole effect, then call complete_action().

        Only the final frame of the action is compared, so do not reproduce
        animations: one step() call renders one frame, and that is enough.
        Ends: self.next_level() when the level is solved (after the last level it
        means WIN), self.lose() for game over.
        """
        action = self.action.id
        moves = {{GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}}
        if action in moves:
            dx, dy = moves[action]
            # Built-in collision: try_move_sprite moves the sprite, and if it now overlaps a
            # collidable sprite it moves it back and returns the sprites hit ([] = it moved).
            # hit = self.try_move_sprite(self.player, dx, dy)
        elif action == GameAction.ACTION5:  # interact
            pass
        elif action == GameAction.ACTION6:  # click: screen pixel -> logical grid cell (None in the border)
            cell = self.camera.display_to_grid(self.action.data.get("x", 0), self.action.data.get("y", 0))
        self.complete_action()
'''


def class_name(game: str) -> str:
    return game[:4].capitalize()


def render_skeleton(game: str, available_actions: list[int]) -> str:
    return TEMPLATE.format(game=game[:4], cls=class_name(game), actions=list(available_actions))
