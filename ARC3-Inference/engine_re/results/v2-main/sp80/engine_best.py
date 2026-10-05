"""Re-implementation of ARC-AGI-3 game "sp80", reverse-engineered from a recorded run.

The game is a paint-sprayer puzzle on a 16x16 logical grid (scaled x4 to the 64x64
screen, orange background).

 * A sprayer (dark-grey body + magenta nozzle) sits against one wall and shoots a
   stream of magenta paint through the grid when ACTION5 is pressed.
 * The stream travels along the gravity-ish direction of the level (down in level 1,
   up in level 2), one cell per frame.  When a paint cell cannot advance it spreads
   left/right; a side front keeps spreading while the cell in front of it is blocked
   and turns into the open direction as soon as it is free.  A cell that lies in the
   row touching the far wall is terminal (the paint "spills" there).
 * Yellow targets (cups hanging over the floor / arches hanging from the ceiling)
   absorb the paint: when the paint enters a target's opening cell the whole target
   turns maroon one frame later.
 * The blue bar is the roller (the player).  It moves one cell per direction action
   and is confined to a band of rows.  Red bars are dry paint blocks; the roller can
   pass through them.  If the paint wets a dry block that lies downstream of the
   roller and to its left, roller and block swap shapes/places at the end of the
   spray; if such a block exists but is to the roller's right the run is lost.
 * Spray outcome: every target wet and nothing spilled -> level complete;
   otherwise the failure animation plays (the wall bar and the dry targets blink)
   and the grid is restored.
 * ACTION6 clicks a dry block and swaps the roller with it.
 * Every action consumes one unit of the level's action budget, drawn as a green
   bar in the border; running out means game over.
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
# 1. Sprite art, in logical-grid pixels. -1 = transparent.
# ---------------------------------------------------------------------------
CUP = [[11, -1, 11], [11, 11, 11]]          # opening on top row, middle
ARCH = [[11, 11, 11], [11, -1, 11]]         # opening on bottom row, middle
NOZZLE_DOWN = [[4], [6]]                    # body then exit (sprays downwards)
NOZZLE_UP = [[6], [4]]                      # exit then body (sprays upwards)
WALL_ROW = [[1] * 16]

TARGET_NORMAL, TARGET_WET, TARGET_DRY = 11, 13, 0


def rect_cells(x: int, y: int, w: int) -> list:
    return [(y, x + i) for i in range(w)]


# ---------------------------------------------------------------------------
# 2. Levels, in order.
# ---------------------------------------------------------------------------
LEVEL_CONFIGS = [
    # level 1 - cups on the floor, sprayer on the ceiling
    dict(
        name="level1", budget=30, direction=1, ground_row=14,
        nozzle=(9, 0), nozzle_art=NOZZLE_DOWN, wall_row=15,
        stream_start=(2, 9), hud_row=0, hud_anchor="left",
        band=(3, 11),
        player=(3, 4, 5),
        targets=[dict(shape=CUP, x=4, y=13, opening=(13, 5)),
                 dict(shape=CUP, x=10, y=13, opening=(13, 11))],
        bars=[],
    ),
    # level 2 - arches on the ceiling, sprayer on the floor, two dry blocks
    dict(
        name="level2", budget=45, direction=-1, ground_row=1,
        nozzle=(10, 14), nozzle_art=NOZZLE_UP, wall_row=0,
        stream_start=(13, 10), hud_row=63, hud_anchor="right",
        band=(4, 12),
        player=(5, 9, 5),
        targets=[dict(shape=ARCH, x=3, y=1, opening=(2, 4)),
                 dict(shape=ARCH, x=7, y=1, opening=(2, 8)),
                 dict(shape=ARCH, x=11, y=1, opening=(2, 12))],
        bars=[(2, 4, 3), (7, 6, 3)],
    ),
    # level 3
    dict(
        name="level3", budget=40, direction=1, ground_row=14,
        nozzle=(7, 0), nozzle_art=NOZZLE_DOWN, wall_row=15,
        stream_start=(2, 7), hud_row=0, hud_anchor="left",
        band=(3, 11),
        player=(4, 8, 5),
        targets=[dict(shape=CUP, x=1, y=13, opening=(13, 2)),
                 dict(shape=CUP, x=6, y=13, opening=(13, 7)),
                 dict(shape=CUP, x=12, y=13, opening=(13, 13))],
        bars=[(9, 5, 3)],
    ),
    # level 4
    dict(
        name="level4", budget=50, direction=-1, ground_row=1,
        nozzle=(5, 14), nozzle_art=NOZZLE_UP, wall_row=0,
        stream_start=(13, 5), hud_row=63, hud_anchor="right",
        band=(4, 12),
        player=(8, 10, 5),
        targets=[dict(shape=ARCH, x=2, y=1, opening=(2, 3)),
                 dict(shape=ARCH, x=7, y=1, opening=(2, 8)),
                 dict(shape=ARCH, x=12, y=1, opening=(2, 13))],
        bars=[(3, 6, 3), (10, 8, 5)],
    ),
    # level 5
    dict(
        name="level5", budget=50, direction=1, ground_row=14,
        nozzle=(12, 0), nozzle_art=NOZZLE_DOWN, wall_row=15,
        stream_start=(2, 12), hud_row=0, hud_anchor="left",
        band=(3, 11),
        player=(2, 7, 5),
        targets=[dict(shape=CUP, x=1, y=13, opening=(13, 2)),
                 dict(shape=CUP, x=6, y=13, opening=(13, 7)),
                 dict(shape=CUP, x=11, y=13, opening=(13, 12))],
        bars=[(5, 4, 3), (9, 6, 5)],
    ),
    # level 6
    dict(
        name="level6", budget=60, direction=-1, ground_row=1,
        nozzle=(8, 14), nozzle_art=NOZZLE_UP, wall_row=0,
        stream_start=(13, 8), hud_row=63, hud_anchor="right",
        band=(4, 12),
        player=(3, 11, 5),
        targets=[dict(shape=ARCH, x=1, y=1, opening=(2, 2)),
                 dict(shape=ARCH, x=5, y=1, opening=(2, 6)),
                 dict(shape=ARCH, x=9, y=1, opening=(2, 10)),
                 dict(shape=ARCH, x=13, y=1, opening=(2, 14))],
        bars=[(6, 5, 3), (10, 7, 3), (2, 9, 5)],
    ),
]


def make_level(cfg: dict) -> Level:
    sprites = []
    sprites.append(Sprite([row[:] for row in WALL_ROW], name="wall", x=0, y=cfg["wall_row"],
                          layer=0, tags=["sys_static", "wall"]))
    nx, ny = cfg["nozzle"]
    sprites.append(Sprite([row[:] for row in cfg["nozzle_art"]], name="nozzle", x=nx, y=ny,
                          layer=1, tags=["nozzle"],
                          interaction=InteractionMode.INTANGIBLE))
    for i, t in enumerate(cfg["targets"]):
        sprites.append(Sprite([row[:] for row in t["shape"]], name="target%d" % i,
                              x=t["x"], y=t["y"], layer=2, tags=["target"],
                              interaction=InteractionMode.INTANGIBLE))
    for i, (bx, by, bw) in enumerate(cfg["bars"]):
        sprites.append(Sprite([[8] * bw], name="bar%d" % i, x=bx, y=by, layer=3,
                              tags=["bar"], interaction=InteractionMode.INTANGIBLE))
    sprites.append(Sprite([[-1] * 16 for _ in range(16)], name="paint", x=0, y=0, layer=4,
                          tags=["paint"], interaction=InteractionMode.INTANGIBLE))
    px, py, pw = cfg["player"]
    sprites.append(Sprite([[9] * pw], name="roller", x=px, y=py, layer=5,
                          tags=["roller"], interaction=InteractionMode.INTANGIBLE))
    return Level(sprites=sprites, grid_size=(16, 16), name=cfg["name"], data={"cfg": cfg})


def build_levels():
    return [make_level(cfg) for cfg in LEVEL_CONFIGS]


# ---------------------------------------------------------------------------
# 3. Screen-space UI: the action budget bar.
# ---------------------------------------------------------------------------
class Hud(RenderableUserDisplay):
    def __init__(self, game: "Sp80") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        g = self.game
        cfg = g.cfg
        if cfg is None:
            return frame
        total = cfg["budget"]
        left = max(0, total - g.action_count_used)
        width = int(round(64.0 * left / total))
        width = max(0, min(64, width))
        row = cfg["hud_row"]
        if cfg["hud_anchor"] == "left":
            frame[row, :] = 0
            if width:
                frame[row, :width] = 14
        else:
            frame[row, :] = 0
            if width:
                frame[row, 64 - width:] = 14
        return frame


# ---------------------------------------------------------------------------
# 4. The game.
# ---------------------------------------------------------------------------
class Sp80(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.cfg = None
        self.spray = None
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=12, letter_box=12, interfaces=[self.hud])
        super().__init__(game_id="sp80", levels=build_levels(), camera=camera,
                         available_actions=[1, 2, 3, 4, 5, 6])

    # -- helpers ---------------------------------------------------------
    @property
    def action_count_used(self) -> int:
        return self._action_count

    def sprites_by_tag(self, tag):
        return self.current_level.get_sprites_by_tag(tag)

    def roller(self):
        return self.current_level.get_sprites_by_tag("roller")[0]

    def paint_sprite(self):
        return self.current_level.get_sprites_by_tag("paint")[0]

    def sprite_cells(self, s):
        out = []
        px = s.pixels
        for r in range(px.shape[0]):
            for c in range(px.shape[1]):
                if px[r, c] != -1:
                    out.append((s.y + r, s.x + c))
        return out

    def blocked_cells(self, include_roller=True):
        """Cells the paint cannot enter."""
        cells = set()
        for s in self.sprites_by_tag("wall"):
            cells.update(self.sprite_cells(s))
        for s in self.sprites_by_tag("nozzle"):
            cells.update(self.sprite_cells(s))
        for s in self.sprites_by_tag("target"):
            cells.update(self.sprite_cells(s))
        for s in self.sprites_by_tag("bar"):
            cells.update(self.sprite_cells(s))
        if include_roller:
            cells.update(self.sprite_cells(self.roller()))
        return cells

    def set_roller_color(self, color):
        r = self.roller()
        w = r.pixels.shape[1]
        r.pixels = np.array([[color] * w], dtype=np.int8)

    def set_target_color(self, idx, color):
        t = self.sprites_by_tag("target")[idx]
        shape = self.cfg["targets"][idx]["shape"]
        t.pixels = np.array([[color if v != -1 else -1 for v in row] for row in shape],
                            dtype=np.int8)

    def set_paint(self, cells):
        arr = np.full((16, 16), -1, dtype=np.int8)
        for (r, c) in cells:
            arr[r, c] = 6
        self.paint_sprite().pixels = arr

    # -- level lifecycle -------------------------------------------------
    def on_set_level(self, level: Level) -> None:
        self.cfg = level.get_data("cfg")
        self.spray = None
        for i in range(len(self.cfg["targets"])):
            self.set_target_color(i, TARGET_NORMAL)
        self.set_paint([])
        self.set_roller_color(9)

    # -- paint simulation ------------------------------------------------
    def growth_step(self, front, paint, blocked, cfg):
        """Return the children cells of a paint front."""
        r, c = front
        d = cfg["direction"]
        ground = cfg["ground_row"]
        if r == ground:
            return []
        nxt = (r + d, c)
        if nxt not in blocked and nxt not in paint:
            return [nxt]
        out = []
        for dc in (-1, 1):
            nb = (r, c + dc)
            if 0 <= nb[1] < 16 and nb not in blocked and nb not in paint:
                out.append(nb)
        return out

    def simulate(self):
        """Run the whole spray without rendering; returns the analysis."""
        cfg = self.cfg
        blocked = self.blocked_cells()
        paint = set()
        front = "start"
        wet = [False] * len(cfg["targets"])
        spilled = False
        touched = None            # (frame_index, bar sprite)
        frames = 0
        bar_cells = {}
        for s in self.sprites_by_tag("bar"):
            bar_cells[id(s)] = set(self.sprite_cells(s))
        openings = {tuple(t["opening"]): i for i, t in enumerate(cfg["targets"])}
        while True:
            frames += 1
            new = set()
            if front == "start":
                src = tuple(cfg["stream_start"])
                if src not in blocked:
                    new.add(src)
            else:
                for f in front:
                    for ch in self.growth_step(f, paint | new, blocked, cfg):
                        new.add(ch)
            if not new:
                break
            paint |= new
            front = list(new)
            for cell in new:
                if cell in openings:
                    wet[openings[cell]] = True
                if cell[0] == cfg["ground_row"]:
                    spilled = True
            if touched is None:
                for s in self.sprites_by_tag("bar"):
                    bc = bar_cells[id(s)]
                    hit = any(self.adjacent(cell, bc) for cell in new)
                    if hit:
                        touched = (frames, s)
                        break
        return dict(paint=paint, wet=wet, spilled=spilled, touched=touched, frames=frames)

    @staticmethod
    def adjacent(cell, cells):
        r, c = cell
        for (rr, cc) in cells:
            if abs(rr - r) + abs(cc - c) == 1:
                return True
        return False

    def swap_plan(self, analysis):
        """Decide what the spray does to the roller / dry blocks."""
        cfg = self.cfg
        touched = analysis["touched"]
        if touched is None:
            return None, False
        _, bar = touched
        roller = self.roller()
        d = cfg["direction"]
        # the roller must lie downstream of the wetted block
        if (bar.y - roller.y) * d > 0:
            return None, False
        if bar.x < roller.x:
            return bar, False
        return None, True

    # -- main loop -------------------------------------------------------
    def step(self) -> None:
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return

        if self.spray is None and action != GameAction.ACTION5:
            self.handle_simple(action)
            return

        if action == GameAction.ACTION5:
            self.handle_spray()
            return

    def handle_simple(self, action):
        cfg = self.cfg
        roller = self.roller()
        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            dy = {GameAction.ACTION1: -1, GameAction.ACTION2: 1}.get(action, 0)
            dx = {GameAction.ACTION3: -1, GameAction.ACTION4: 1}.get(action, 0)
            lo, hi = cfg["band"]
            ny = roller.y + dy
            nx = roller.x + dx
            w = roller.pixels.shape[1]
            if lo <= ny <= hi and 0 <= nx and nx + w <= 16:
                roller.set_position(nx, ny)
        elif action == GameAction.ACTION6:
            x, y = self.action.data.get("x", 0), self.action.data.get("y", 0)
            cell = self.camera.display_to_grid(x, y)
            if cell is not None:
                gx, gy = cell
                for bar in self.sprites_by_tag("bar"):
                    if any((r, c) == (gy, gx) for (r, c) in self.sprite_cells(bar)):
                        self.do_swap(bar)
                        break
        self.check_budget()

    def do_swap(self, bar):
        roller = self.roller()
        rx, ry, rw = roller.x, roller.y, roller.pixels.shape[1]
        bx, by, bw = bar.x, bar.y, bar.pixels.shape[1]
        roller.pixels = np.array([[9] * bw], dtype=np.int8)
        roller.set_position(bx, by)
        bar.pixels = np.array([[8] * rw], dtype=np.int8)
        bar.set_position(rx, ry)

    def check_budget(self):
        if self._action_count >= self.cfg["budget"]:
            self.lose()
            self.complete_action()
            return True
        self.complete_action()
        return False

    # -- the spray animation ---------------------------------------------
    def handle_spray(self):
        cfg = self.cfg
        st = self.spray
        if st is None:
            analysis = self.simulate()
            bar, lose_now = self.swap_plan(analysis)
            if lose_now:
                self.lose()
                self.complete_action()
                return
            self.spray = dict(
                phase="grow", t=0, paint=set(), front="start",
                wet=[False] * len(cfg["targets"]), pending_wet=[], spilled=False,
                pending_spill=False, wall_on=False, analysis=analysis, swap=bar,
            )
            self.set_roller_color(8)
            return
        if st["phase"] == "grow":
            self.apply_effects(st)
            new = set()
            blocked = self.blocked_cells()
            openings = {tuple(t["opening"]): i for i, t in enumerate(cfg["targets"])}
            if st["front"] == "start":
                src = tuple(cfg["stream_start"])
                if src not in blocked:
                    new.add(src)
            else:
                for f in st["front"]:
                    for ch in self.growth_step(f, st["paint"] | new, blocked, cfg):
                        new.add(ch)
            if not new:
                st["t"] = 0
                won = all(st["wet"]) and not st["spilled"]
                st["phase"] = "win" if won else "tail"
                self.render_tail(st, 0)
                return
            st["paint"] |= new
            st["front"] = list(new)
            self.set_paint(st["paint"])
            for cell in new:
                if cell in openings and not st["wet"][openings[cell]]:
                    st["wet"][openings[cell]] = True
                    st["pending_wet"].append(openings[cell])
                if cell[0] == cfg["ground_row"] and not st["spilled"]:
                    st["spilled"] = True
                    st["pending_spill"] = True
            return
        if st["phase"] == "tail":
            st["t"] += 1
            if st["t"] >= 7:
                self.finish_spray()
                self.check_budget()
                return
            self.render_tail(st, st["t"])
            return
        if st["phase"] == "win":
            st["t"] += 1
            if st["t"] >= 1:
                self.spray = None
                self.next_level()
                self.complete_action()
            return

    def apply_effects(self, st):
        if st["pending_wet"]:
            for i in st["pending_wet"]:
                self.set_target_color(i, TARGET_WET)
            st["pending_wet"] = []
        if st["pending_spill"]:
            st["wall_on"] = True
            st["pending_spill"] = False
            self.set_wall_color(14)

    def render_tail(self, st, t):
        """Apply the blink pattern of frame index t (0 = decision frame)."""
        cfg = self.cfg
        wall_on = st["spilled"] and (t % 2 == 0) and t < 7
        self.set_wall_color(14 if wall_on else 1)
        dry = [i for i, w in enumerate(st["wet"]) if not w]
        blink = (t in (1, 3, 5, 6)) and t < 7
        for i in dry:
            self.set_target_color(i, TARGET_DRY if blink else TARGET_WET if st["wet"][i] else TARGET_NORMAL)

    def set_wall_color(self, color):
        w = self.current_level.get_sprites_by_tag("wall")[0]
        w.pixels = np.array([[color] * 16], dtype=np.int8)

    def finish_spray(self):
        self.set_paint([])
        self.set_roller_color(9)
        self.set_wall_color(1)
        for i in range(len(self.cfg["targets"])):
            self.set_target_color(i, TARGET_NORMAL)
        bar = self.spray["swap"] if self.spray else None
        self.spray = None
        if bar is not None:
            self.do_swap(bar)
