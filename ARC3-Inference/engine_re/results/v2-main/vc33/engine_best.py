"""Re-implementation of ARC-AGI-3 game "vc33", reverse-engineered from a recorded run.

The game is a click-only "hydraulic bands" puzzle rendered on a 64x64 grid:

  * rectangular BANDS of grey material (3) anchored to one side of the screen,
    with a free edge whose position is the band's only degree of freedom;
  * BLACK BARS (5) drawn over the bands between neighbouring bands; some bars
    carry a coloured HOLE (a 2px wide, full-bar-height window);
  * a BALL (arrow shaped, 2px coloured tip on the band's edge, dark-grey body)
    that rides on a band's free edge;
  * BUTTONS (9) inside a band next to a bar; clicking a button moves a fixed
    number of pixels of material from that band into the band on the far side of
    the bar.  A move that would empty the giver past its limit or overflow the
    receiver is refused (nothing happens);
  * the level is solved when every hole is covered by the ball of its colour.

  * A HUD "budget" bar is painted over screen row 0: 64 cells, colour 7, of the
    ones still available; used cells become colour 4.  used = ceil(64*k/budget)
    where k is the number of actions taken since the level was entered.

Contract: module defines Vc33(arcengine.ARCBaseGame), constructible as Vc33().
"""

from __future__ import annotations

import math

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

GREY = 3
BAR = 5
BTN = 9
BODY = 4          # ball body / "used" budget colour
BG_DEFAULT = 0

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def fill(buf, r0, r1, c0, c1, color):
    buf[r0:r1 + 1, c0:c1 + 1] = color


class Band:
    """Grey material band.

    axis 'h': fixed rows [f0,f1]; anchor 'l' -> cols [v0..edge], 'r' -> cols [edge..v1]
    axis 'v': fixed cols [f0,f1]; anchor 't' -> rows [v0..edge], 'b' -> rows [edge..v1]
    """

    def __init__(self, name, axis, f0, f1, anchor, edge, vmin=0, vmax=63, v0=0, v1=63,
                 well=None, split=55):
        self.name = name
        self.axis = axis
        self.f0, self.f1 = f0, f1
        self.anchor = anchor
        self.edge = edge
        self.vmin, self.vmax = vmin, vmax
        self.v0, self.v1 = v0, v1
        self.well = well          # (col0, col1) extra columns below `split`
        self.split = split

    def clone(self):
        return Band(self.name, self.axis, self.f0, self.f1, self.anchor, self.edge,
                    self.vmin, self.vmax, self.v0, self.v1, self.well, self.split)

    def rect(self):
        if self.axis == "h":
            if self.anchor == "l":
                return (self.f0, self.f1, self.v0, self.edge)
            return (self.f0, self.f1, self.edge, self.v1)
        if self.anchor == "t":
            return (self.v0, min(self.edge, self.split), self.f0, self.f1)
        return (self.edge, self.v1, self.f0, self.f1)

    def well_rect(self):
        if self.well is None or self.edge <= self.split:
            return None
        return (self.split + 1, self.edge, self.well[0], self.well[1])

    def surface(self):
        return min(self.edge, self.split)

    def shrink(self, n):
        return self.edge - n if self.anchor in ("l", "t") else self.edge + n

    def grow(self, n):
        return self.edge + n if self.anchor in ("l", "t") else self.edge - n


class Ball:
    """Arrow shaped marker riding on a band's free edge."""

    def __init__(self, color, band, r0, r1, size=6, tail=2, wide=None):
        self.color = color
        self.band = band
        self.r0, self.r1 = r0, r1      # rows occupied by the 2px tip column pair
        self.size = size               # tip block length along the fixed axis
        self.tail = tail               # body thickness (along the moving axis)

    def clone(self, band):
        return Ball(self.color, band, self.r0, self.r1, self.size, self.tail)

    def tip_h(self):
        e = self.band.edge
        return (e - 1, e) if self.band.anchor == "l" else (e, e + 1)

    def tip_v(self):
        e = self.band.edge
        return (e - 1, e) if self.band.anchor == "t" else (e, e + 1)

    def draw(self, buf):
        b = self.band
        e = b.edge
        if b.axis == "h":
            c0, c1 = (e - 1, e) if b.anchor == "l" else (e, e + 1)
            r0, r1 = self.r0, self.r0 + self.size - 1
            fill(buf, r0, r1, c0, c1, self.color)
            t0, t1 = (c0 - self.tail, c0 - 1) if b.anchor == "l" else (c1 + 1, c1 + self.tail)
            fill(buf, r0, r1, t0, t1, BODY)
            m0 = r0 + 2
            m1 = r1 - 2
            if b.anchor == "l":
                fill(buf, m0, m1, t0 - 2, t1, BODY)
            else:
                fill(buf, m0, m1, t0, t1 + 2, BODY)
        else:
            r0, r1 = (e - 1, e) if b.anchor == "t" else (e, e + 1)
            c0, c1 = self.r0, self.r0 + self.size - 1
            fill(buf, r0, r1, c0, c1, self.color)
            t0, t1 = (r0 - self.tail, r0 - 1) if b.anchor == "t" else (r1 + 1, r1 + self.tail)
            fill(buf, t0, t1, c0, c1, BODY)
            m0 = c0 + 2
            m1 = c1 - 2
            if b.anchor == "t":
                fill(buf, t0 - 2, t1, m0, m1, BODY)
            else:
                fill(buf, t0, t1 + 2, m0, m1, BODY)


class Drop:
    """3x3 drop marker riding on a vertical band's free (bottom) edge."""

    def __init__(self, color, band_name, col):
        self.color = color
        self.band_name = band_name
        self.band = None
        self.col = col

    def clone(self, by_name):
        d = Drop(self.color, self.band_name, self.col)
        d.band = by_name[self.band_name]
        return d

    def tip_h(self):
        return (self.band.edge, self.band.edge)

    def tip_v(self):
        return (self.band.edge, self.band.edge)

    def draw(self, buf):
        v = self.band.edge
        c = self.col
        fill(buf, v, v, c, c + 2, self.color)
        fill(buf, v - 1, v - 1, c, c + 2, BODY)
        fill(buf, v - 2, v - 2, c + 1, c + 1, BODY)


# ---------------------------------------------------------------------------
# level description
# ---------------------------------------------------------------------------


class Spec:
    """Immutable description of one level."""

    def __init__(self, name, bg, budget, bands, bars, holes, buttons, balls,
                 step_sizes=None, win=None, ports=None, pre=None):
        self.name = name
        self.bg = bg
        self.budget = budget
        self.bands = bands            # list[Band] (initial edges)
        self.bars = bars              # list[(r0,r1,c0,c1)]
        self.holes = holes            # list[(r0,r1,c0,c1,color)]
        self.buttons = buttons        # list[(r0,r1,c0,c1,src,dst)]
        self.balls = balls            # list[Ball] (band referenced by name)
        self.step_sizes = step_sizes or {}
        self.win = win or []          # list[(hole_index, ball_band_or_None, ball_color)]
        self.ports = ports or []      # list[(r0,r1,c0,c1,color)] drawn after bars
        self.pre = pre or []          # list[(r0,r1,c0,c1,color)] drawn before bands

    def instantiate(self):
        bands = [b.clone() for b in self.bands]
        by_name = {b.name: b for b in bands}
        balls = []
        for bl in self.balls:
            if isinstance(bl, Drop):
                balls.append(bl.clone(by_name))
            else:
                balls.append(bl.clone(by_name[bl.band.name]))
        return World(self, bands, by_name, balls)


class World:
    def __init__(self, spec, bands, by_name, balls):
        self.spec = spec
        self.bands = bands
        self.by_name = by_name
        self.balls = balls
        self.actions = 0

    def band(self, n):
        return self.by_name[n]

    def render(self, buf):
        fill(buf, 0, 63, 0, 63, self.spec.bg)
        for (r0, r1, c0, c1, col) in self.spec.pre:
            fill(buf, r0, r1, c0, c1, col)
        for b in self.bands:
            r0, r1, c0, c1 = b.rect()
            fill(buf, r0, r1, c0, c1, GREY)
            w = b.well_rect()
            if w:
                fill(buf, w[0], w[1], w[2], w[3], GREY)
        for (r0, r1, c0, c1) in self.spec.bars:
            fill(buf, r0, r1, c0, c1, BAR)
        for (r0, r1, c0, c1, col) in self.spec.ports:
            fill(buf, r0, r1, c0, c1, col)
        for (r0, r1, c0, c1, col) in self.spec.holes:
            fill(buf, r0, r1, c0, c1, col)
        for ball in self.balls:
            ball.draw(buf)
        for (r0, r1, c0, c1, s, d) in self.spec.buttons:
            fill(buf, r0, r1, c0, c1, BTN)

    def step_size(self, src, dst):
        return self.spec.step_sizes.get((src.name, dst.name),
                                        self.spec.step_sizes.get("*", 4))

    def transfer(self, src, dst, n):
        snew = src.shrink(n)
        dnew = dst.grow(n)
        if not (src.vmin <= snew <= src.vmax):
            return False
        if not (dst.vmin <= dnew <= dst.vmax):
            return False
        src.edge, dst.edge = snew, dnew
        return True

    def click(self, x, y):
        for (r0, r1, c0, c1, s, d) in self.spec.buttons:
            if r0 <= y <= r1 and c0 <= x <= c1:
                if s is None or d is None:
                    return False
                return self.transfer(self.band(s), self.band(d),
                                     self.step_size(self.band(s), self.band(d)))
        return False

    def solved(self):
        for (h_r0, h_r1, h_c0, h_c1, hcol, bname) in self.spec.win:
            ball = None
            for bl in self.balls:
                if bl.color == hcol and bl.band.name == bname:
                    ball = bl
            if ball is None:
                return False
            b = ball.band
            if b.axis == "h":
                if ball.tip_h() != (h_c0, h_c1):
                    return False
            else:
                if ball.tip_v() != (h_r0, h_r1):
                    return False
        return True


# ---------------------------------------------------------------------------
# the levels
# ---------------------------------------------------------------------------

def level0():
    bands = [
        Band("A", "h", 0, 31, "l", 31, vmin=0, vmax=63),
        Band("B", "h", 28, 63, "l", 51, vmin=0, vmax=63),
    ]
    bars = [(28, 31, 20, 63)]
    holes = [(28, 31, 38, 39, 11)]
    buttons = [(24, 27, 60, 63, "A", "B"), (32, 35, 60, 63, "B", "A")]
    balls = [Ball(11, Band("B", "h", 0, 0, "l", 0), 44, 49, size=6, tail=2)]
    return Spec("level1", BG_DEFAULT, 50, bands, bars, holes, buttons, balls,
                win=[(28, 31, 38, 39, 11, "B")])


def level1():
    bands = [
        Band("A", "h", 0, 19, "r", 52, vmin=0, vmax=60),
        Band("C", "h", 24, 39, "r", 12, vmin=0, vmax=60),
        Band("E", "h", 44, 63, "r", 8, vmin=0, vmax=60),    ]
    bars = [(20, 23, 0, 55), (40, 43, 0, 43)]
    holes = [(40, 43, 28, 29, 14)]
    buttons = [
        (16, 19, 0, 3, "A", "C"),
        (24, 27, 0, 3, "C", "A"),
        (36, 39, 0, 3, "C", "E"),
        (44, 47, 0, 3, "E", "C"),
    ]
    balls = [Ball(14, Band("E", "h", 0, 0, "r", 0), 52, 57, size=6, tail=2)]
    return Spec("level2", BG_DEFAULT, 50, bands, bars, holes, buttons, balls,
                win=[(40, 43, 28, 29, 14, "E")],
                ports=[(20, 23, 56, 63, GREY), (40, 43, 44, 63, GREY)])


def level2():
    bands = [
        Band("s1", "v", 6, 15, "t", 53, vmin=6, vmax=57, v0=6, v1=57, well=(6, 11)),
        Band("s2", "v", 16, 27, "t", 55, vmin=6, vmax=57, v0=6, v1=57, well=(18, 23)),
        Band("s3", "v", 28, 37, "t", 53, vmin=6, vmax=57, v0=6, v1=57, well=(30, 33)),
        Band("s4", "v", 38, 49, "t", 51, vmin=6, vmax=57, v0=6, v1=57, well=(40, 45)),
        Band("s5", "v", 50, 57, "t", 29, vmin=6, vmax=57, v0=6, v1=57, well=(52, 57)),
    ]
    bars = [(38, 57, 14, 15), (52, 57, 26, 27), (36, 57, 36, 37), (30, 57, 48, 49)]
    holes = [(41, 41, 14, 15, 14), (39, 39, 36, 37, 15), (47, 47, 48, 49, 11)]
    buttons = [
        (56, 57, 12, 13, "s1", "s2"), (56, 57, 16, 17, "s2", "s1"),
        (56, 57, 24, 25, "s2", "s3"), (56, 57, 28, 29, "s3", "s2"),
        (56, 57, 34, 35, "s3", "s4"), (56, 57, 38, 39, "s4", "s3"),
        (56, 57, 46, 47, "s4", "s5"), (56, 57, 50, 51, "s5", "s4"),
    ]
    balls = [Drop(14, "s1", 8), Drop(15, "s4", 41), Drop(11, "s5", 52)]
    return Spec("level3", 4, 75, bands, bars, holes, buttons, balls,
                step_sizes={"*": 2},
                win=[(41, 41, 14, 15, 14, "s1"), (39, 39, 36, 37, 15, "s4"),
                     (47, 47, 48, 49, 11, "s5")],
                pre=[(6, 57, 6, 57, 0)])


def placeholder(n):
    return Spec("level%d" % (n + 1), BG_DEFAULT, 64, [], [], [], [], [], win=[])


SPECS = [level0(), level1(), level2()] + [placeholder(i) for i in range(3, 7)]


# ---------------------------------------------------------------------------
# screen UI: the budget bar painted over logical row 0
# ---------------------------------------------------------------------------


class Hud(RenderableUserDisplay):
    def __init__(self, game):
        self.game = game

    def render_interface(self, frame):
        w = self.game.world
        budget = w.spec.budget
        used = min(64, int(64 * w.actions / budget + 0.5)) if budget else 0
        frame[0, :] = 7
        if used:
            frame[0, 64 - used:] = 4
        return frame


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class Vc33(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.world = SPECS[0].instantiate()
        self.hud = Hud(self)
        camera = Camera(0, 0, 64, 64, background=GREY, letter_box=GREY,
                        interfaces=[self.hud])
        super().__init__(game_id="vc33", levels=self._make_levels(), camera=camera,
                         available_actions=[6], win_score=len(SPECS))

    @staticmethod
    def _make_levels():
        out = []
        for i, sp in enumerate(SPECS):
            out.append(Level(sprites=[], grid_size=(64, 64), name=sp.name,
                             data={"index": i}))
        return out

    def on_set_level(self, level) -> None:
        idx = level.get_data("index")
        self.world = SPECS[idx].instantiate()
        self._sync_sprite()

    def _sync_sprite(self):
        buf = np.full((64, 64), GREY, dtype=np.int8)
        self.world.render(buf)
        sprites = self.current_level.get_sprites_by_name("world")
        if not sprites:
            s = Sprite(buf, name="world", x=0, y=0, layer=0,
                       blocking=BlockingMode.NOT_BLOCKED,
                       interaction=InteractionMode.INTANGIBLE)
            self.current_level.add_sprite(s)
            self._world_sprite = s
        else:
            sprites[0].pixels = buf
            self._world_sprite = sprites[0]

    def step(self) -> None:
        act = self.action.id
        if act == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            self.world.actions += 1
            self.world.click(x, y)
        if self.world.solved():
            self.next_level()
        self._sync_sprite()
        self.complete_action()
