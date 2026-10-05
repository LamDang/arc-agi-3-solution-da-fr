"""Re-implementation of ARC-AGI-3 game "sp80", reverse-engineered from a recorded run.

The game (established from the recording):
  * 16x16 logical grid, scale 4 -> exactly 64x64, background colour 12 (orange).
  * A fixed "player"/emitter: a 1x2 sprite (grey 4 + magenta 6) sitting on the
    wall side.  It never moves; it is the source of the magenta "ink".
  * One or more horizontal bars (5 wide = active/blue 9, 3 wide = inactive/red 8).
    ACTION1-4 move the ACTIVE bar by one cell.  Bars pass through each other but
    cannot leave the level's "field" rows (level 0 rows 3..11, level 1 rows 4..12).
    Only one bar is blue at a time; the others are red.
  * ACTION6 click: if the clicked cell is part of a bar, that bar becomes active.
  * ACTION5: pour ink.  The ink starts at the player cell facing the field and
    grows one cell per frame in the flow direction (down in level 0, up in
    level 1); when its forward cell is blocked it spawns left+right instead;
    cells in the row adjacent to the wall die.  Solid = wall row, baskets,
    bars, player.
  * Baskets are 3x2 yellow (11) cups with a one-cell "opening" (transparent).
    When the ink enters an opening, the basket turns maroon (13) one frame later.
  * If the ink reaches the wall-adjacent row it "leaks": the wall flashes green
    (14) and the unfilled baskets flash black (0) for 7 frames, then everything
    reverts.  Win = all baskets filled and no leak.
  * Each level allows 5 pour attempts; failing the 5th ends the game immediately
    (a single unchanged frame, GAME_OVER).
  * A 1-pixel-tall budget bar in screen row 0 (level 0, drains right-to-left)
    or row 63 (level 1, drains left-to-right): black pixels =
    round(64 * actions_used / budget); running out of budget = GAME_OVER.

Contract: module defines Sp80(arcengine.ARCBaseGame), constructible as Sp80().
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

W = H = 16
BG = 12

# palette helpers
TRANSPARENT = -1

# basket templates (3 wide, 2 tall); True = solid yellow cell
BASKET_GAP_TOP = [[True, False, True], [True, True, True]]     # level 0 (cup opens upward)
BASKET_GAP_BOTTOM = [[True, True, True], [True, False, True]]  # level 1 (cup opens downward)


def _blank():
    return [[TRANSPARENT] * W for _ in range(H)]


def basket_pixels(template, color):
    return [[color if cell else TRANSPARENT for cell in row] for row in template]


# ---------------------------------------------------------------------------
# level descriptions (plain data; sprites are built per level entry)
# ---------------------------------------------------------------------------
LEVELS_DATA = [
    dict(
        name="level1",
        wall_row=15,
        baskets=[(4, 13, "top"), (10, 13, "top")],   # (col, row, gap side)
        player=(9, 0, [4, 6]),                        # col, row, [top colour, bottom colour]
        flow=1,                                       # +1 = ink flows down, -1 = up
        field=(3, 11),                                # rows the bars may occupy
        blocks=[(3, 4, 5)],                           # (col, row, width); last = active
        budget=30,
        bar_row=0,
        bar_from="right",
    ),
    dict(
        name="level2",
        wall_row=0,
        baskets=[(3, 1, "bottom"), (7, 1, "bottom"), (11, 1, "bottom")],
        player=(10, 14, [6, 4]),
        flow=-1,
        field=(4, 12),
        blocks=[(2, 4, 3), (7, 6, 3), (5, 9, 5)],
        budget=45,
        bar_row=63,
        bar_from="left",
    ),
]

# extra levels (never reached in the recording) - plausible continuations
for _i in range(4):
    LEVELS_DATA.append(dict(LEVELS_DATA[0]))
    LEVELS_DATA[-1] = dict(LEVELS_DATA[-1], name="level%d" % (_i + 3))


class Hud(RenderableUserDisplay):
    """The 1-pixel budget bar, drawn in screen pixels."""

    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        st = self.game.st
        if st is None:
            return frame
        row = st["bar_row"]
        black = int(np.floor(64.0 * st["used"] / st["budget"] + 0.5))
        black = max(0, min(64, black))
        for c in range(64):
            frame[row][c] = 14
        if st["bar_from"] == "right":
            cols = range(64 - black, 64)
        else:
            cols = range(0, black)
        for c in cols:
            frame[row][c] = 0
        return frame


class Sp80(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = Hud(self)
        self.st = None
        camera = Camera(0, 0, 64, 64, background=BG, letter_box=BG, interfaces=[self.hud])
        super().__init__(game_id="sp80", levels=build_levels(), camera=camera,
                         available_actions=[1, 2, 3, 4, 5, 6], win_score=6)

    # ------------------------------------------------------------------ setup
    def on_set_level(self, level: Level) -> None:
        cfg = level.get_data("cfg")
        self.k0 = self._action_count
        self.attempts = 0
        self.script = None
        self.st = dict(
            cfg=cfg,
            wall_row=cfg["wall_row"],
            baskets=cfg["baskets"],
            player=cfg["player"],
            flow=cfg["flow"],
            field=cfg["field"],
            blocks=[dict(col=c, row=r, w=w) for (c, r, w) in cfg["blocks"]],
            active=len(cfg["blocks"]) - 1,
            ink=set(),
            filled=set(),
            wall_green=False,
            basket_black=False,
            all_red=False,
            bar_row=cfg["bar_row"],
            bar_from=cfg["bar_from"],
            budget=cfg["budget"],
            used=0,
        )
        self._rebuild_sprites(level)
        self._sync()

    def _rebuild_sprites(self, level: Level) -> None:
        st = self.st
        level.remove_all_sprites()
        s = Sprite([[1] * W], name="wall", x=0, y=st["wall_row"], layer=0,
                   tags=["wall"], interaction=InteractionMode.INTANGIBLE,
                   blocking=BlockingMode.NOT_BLOCKED)
        level.add_sprite(s)
        st["wall_sprite"] = s
        st["basket_sprites"] = []
        for i, (c, r, gap) in enumerate(st["baskets"]):
            tmpl = BASKET_GAP_TOP if gap == "top" else BASKET_GAP_BOTTOM
            spr = Sprite(basket_pixels(tmpl, 11), name="basket%d" % i, x=c, y=r, layer=1,
                         tags=["basket"], interaction=InteractionMode.INTANGIBLE,
                         blocking=BlockingMode.NOT_BLOCKED)
            level.add_sprite(spr)
            st["basket_sprites"].append(spr)
        st["ink_sprite"] = Sprite(_blank(), name="ink", x=0, y=0, layer=2, tags=["ink"],
                                  interaction=InteractionMode.INTANGIBLE,
                                  blocking=BlockingMode.NOT_BLOCKED)
        level.add_sprite(st["ink_sprite"])
        st["block_sprites"] = []
        for i, b in enumerate(st["blocks"]):
            spr = Sprite([[9] * b["w"]], name="block%d" % i, x=b["col"], y=b["row"], layer=3,
                         tags=["block"], interaction=InteractionMode.INTANGIBLE,
                         blocking=BlockingMode.NOT_BLOCKED)
            level.add_sprite(spr)
            st["block_sprites"].append(spr)
        pc, pr, cols = st["player"]
        spr = Sprite([[cols[0]], [cols[1]]], name="player", x=pc, y=pr, layer=5,
                     tags=["player"], interaction=InteractionMode.INTANGIBLE,
                     blocking=BlockingMode.NOT_BLOCKED)
        level.add_sprite(spr)
        st["player_sprite"] = spr

    # ------------------------------------------------------------- pixel sync
    def _sync(self) -> None:
        st = self.st
        # wall colour
        col = 14 if st["wall_green"] else 1
        st["wall_sprite"].pixels = np.array([[col] * W], dtype=np.int8)
        # baskets
        tmpl_map = {"top": BASKET_GAP_TOP, "bottom": BASKET_GAP_BOTTOM}
        for i, spr in enumerate(st["basket_sprites"]):
            c, r, gap = st["baskets"][i]
            if i in st["filled"]:
                bc = 13
            elif st["basket_black"]:
                bc = 0
            else:
                bc = 11
            spr.pixels = np.array(basket_pixels(tmpl_map[gap], bc), dtype=np.int8)
        # ink
        grid = _blank()
        for (r, c) in st["ink"]:
            grid[r][c] = 6
        st["ink_sprite"].pixels = np.array(grid, dtype=np.int8)
        # blocks
        for i, (spr, b) in enumerate(zip(st["block_sprites"], st["blocks"])):
            active = (i == st["active"])
            cc = 9 if (active and not st["all_red"]) else 8
            spr.pixels = np.array([[cc] * b["w"]], dtype=np.int8)
            spr.set_position(b["col"], b["row"])
            spr.set_layer(4 if active else 3)

    # ---------------------------------------------------------------- helpers
    def _solid(self):
        """Cells that the ink cannot enter."""
        st = self.st
        solid = set()
        for c in range(W):
            solid.add((st["wall_row"], c))
        tmpl_map = {"top": BASKET_GAP_TOP, "bottom": BASKET_GAP_BOTTOM}
        for (c, r, gap) in st["baskets"]:
            for dr, row in enumerate(tmpl_map[gap]):
                for dc, v in enumerate(row):
                    if v:
                        solid.add((r + dr, c + dc))
        for b in st["blocks"]:
            for dc in range(b["w"]):
                solid.add((b["row"], b["col"] + dc))
        pc, pr, cols = st["player"]
        solid.add((pr, pc))
        solid.add((pr + 1, pc))
        return solid

    def _openings(self):
        """Basket opening cells -> basket index."""
        st = self.st
        out = {}
        for i, (c, r, gap) in enumerate(st["baskets"]):
            if gap == "top":
                out[(r, c + 1)] = i
            else:
                out[(r + 1, c + 1)] = i
        return out

    def _emitter(self):
        st = self.st
        pc, pr, cols = st["player"]
        return (pr, pc) if st["flow"] < 0 else (pr + 1, pc)

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        st = self.st
        st["used"] = self._action_count - self.k0
        if self.script:
            self.apply_frame(self.script.pop(0))
            if not self.script:
                self.finish_action()
            return
        action = self.action.id
        if action in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                      GameAction.ACTION4):
            d = {GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
                 GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)}[action]
            self.move_block(d)
        elif action == GameAction.ACTION6:
            x, y = self.action.data.get("x", 0), self.action.data.get("y", 0)
            cell = self.camera.display_to_grid(x, y)
            if cell is not None:
                self.click(cell)
        elif action == GameAction.ACTION5:
            self.pour()
            if self.script:
                self.apply_frame(self.script.pop(0))
            return
        self.check_budget()
        self.complete_action()

    def move_block(self, d):
        st = self.st
        b = st["blocks"][st["active"]]
        lo, hi = st["field"]
        nr, nc = b["row"] + d[0], b["col"] + d[1]
        if lo <= nr <= hi and 0 <= nc and nc + b["w"] - 1 < W:
            b["row"], b["col"] = nr, nc
        self._sync()

    def click(self, cell):
        st = self.st
        gx, gy = cell
        for i, b in enumerate(st["blocks"]):
            if b["row"] == gy and b["col"] <= gx < b["col"] + b["w"]:
                st["active"] = i
                break
        self._sync()

    # ------------------------------------------------------------------ pour
    def pour(self) -> None:
        st = self.st
        solid = self._solid()
        openings = self._openings()
        block_of = {}
        for i, b in enumerate(st["blocks"]):
            for dc in range(b["w"]):
                block_of[(b["row"], b["col"] + dc)] = i
        flow = st["flow"]
        wall_adj = st["wall_row"] - flow
        emitter = self._emitter()
        frames = [{emitter}]
        ink = {emitter}
        frontier = [emitter]
        fill_events = []
        leak_frame = None
        touched = None
        t = 0
        pending = []
        def check_touch(cells, when):
            nonlocal touched
            if touched is not None:
                return
            for (r, c) in cells:
                for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nb in block_of:
                        touched = block_of[nb]
                        return
        check_touch([emitter], 0)
        while frontier:
            t += 1
            for (fi, bi) in pending:
                if fi == t:
                    fill_events.append((t, bi))
            pending = []
            added = []
            for (r, c) in frontier:
                if r == wall_adj:
                    continue
                nxt = []
                fr = (r + flow, c)
                if 0 <= fr[0] < H and fr not in solid and fr not in ink:
                    nxt.append(fr)
                else:
                    for side in ((r, c - 1), (r, c + 1)):
                        if 0 <= side[1] < W and side not in solid and side not in ink:
                            nxt.append(side)
                for x in nxt:
                    if x not in ink:
                        ink.add(x)
                        added.append(x)
                        if x in openings:
                            pending.append((t + 1, openings[x]))
                        if x[0] == wall_adj and leak_frame is None:
                            leak_frame = t
            if added:
                check_touch(added, t)
                frames.append(set(added))
                frontier = added
            else:
                break
        N = len(frames) - 1
        last_fill = max([fe[0] for fe in fill_events], default=0)
        F = max(N, last_fill, 1)
        filled = set(bi for (_, bi) in fill_events)
        all_filled = len(filled) == len(st["baskets"])
        win = all_filled and leak_frame is None
        if touched is not None:
            self.new_active = touched
        # ---- build the frame script
        script = []
        running = set()
        for f in range(N + 1):
            running |= frames[f]
            fl = set(bi for (fi, bi) in fill_events if fi <= f)
            wg = leak_frame is not None and f >= leak_frame + 1
            script.append(dict(ink=set(running), filled=fl, wall_green=wg,
                               basket_black=False))
        while len(script) <= F:                       # fill frames past the last growth
            script.append(dict(ink=set(running),
                               filled=set(bi for (fi, bi) in fill_events if fi <= len(script)),
                               wall_green=leak_frame is not None and len(script) >= leak_frame + 1,
                               basket_black=False))
        if win:
            script.append(dict(script[-1]))
        else:
            for f in range(F + 1, F + 8):
                j = f - (F + 1)
                wg = (leak_frame is not None) and (f >= leak_frame + 1) and (
                    f <= F + 1 or j % 2 == 0)
                bb = (f >= F + 2) and ((j % 2 == 1) or f == F + 7)
                script.append(dict(ink=script[-1]["ink"], filled=script[-1]["filled"],
                                   wall_green=bool(wg), basket_black=bool(bb)))
            script.append(dict(ink=set(), filled=set(), wall_green=False,
                               basket_black=False))                      # revert frame
        self.attempts += 1
        self.pending_win = win
        if not win and self.attempts >= 5:
            self.script = None
            self._sync()
            self.lose()
            self.complete_action()
            return
        st["all_red"] = True
        self.script = script

    def apply_frame(self, fr):
        st = self.st
        st["ink"] = fr["ink"]
        st["filled"] = fr["filled"]
        st["wall_green"] = fr["wall_green"]
        st["basket_black"] = fr["basket_black"]
        self._sync()

    def finish_action(self):
        st = self.st
        if getattr(self, "pending_win", False):
            self.script = None
            self.next_level()
            self.check_budget()
            self.complete_action()
        else:
            # revert frame: ink/fills already cleared by the script; restore colours
            st["all_red"] = False
            new_active = getattr(self, "new_active", None)
            if new_active is not None:
                st["active"] = new_active
            self._sync()
            self.check_budget()
            self.complete_action()

    def check_budget(self) -> None:
        st = self.st
        st["used"] = self._action_count - self.k0
        if st["used"] >= st["budget"] and self._state != GameState.GAME_OVER:
            self.lose()


def build_levels():
    levels = []
    for cfg in LEVELS_DATA:
        levels.append(Level(sprites=[], grid_size=(W, H), name=cfg["name"], data={"cfg": cfg}))
    return levels
