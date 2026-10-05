"""lp85: rotating colour rings. Generated engine."""

# ==== FIXED INTERFACE: DO NOT EDIT. ====
from copy import deepcopy
from dataclasses import dataclass, field, replace


@dataclass(eq=False)
class Sprite:
    pixels: list
    x: int = 0
    y: int = 0
    layer: int = 0
    name: str = ""
    tags: tuple = ()
    visible: bool = True
    collidable: bool = True
    blocking: str = "pixel"
    rotation: int = 0
    mirror_ud: bool = False
    mirror_lr: bool = False
    scale: int = 1
    screen: bool = False

    def render(self) -> list:
        out = [list(row) for row in self.pixels]
        for _ in range((self.rotation // 90) % 4):
            out = [list(row) for row in zip(*out[::-1])]
        if self.mirror_ud:
            out = out[::-1]
        if self.mirror_lr:
            out = [row[::-1] for row in out]
        if self.scale > 1:
            out = [[v for v in row for _ in range(self.scale)] for row in out for _ in range(self.scale)]
        elif self.scale < 0:
            k = 1 - self.scale
            shrunk = []
            for r in range(0, len(out), k):
                row = []
                for c in range(0, len(out[0]), k):
                    block = [out[r + i][c + j] for i in range(k) for j in range(k)]
                    opaque = [v for v in block if v != -1]
                    if sum(v < 0 for v in block) > len(opaque):
                        row.append(-1)
                    else:
                        row.append(max(opaque, key=lambda v: (opaque.count(v), v)))
                shrunk.append(row)
            out = shrunk
        return out

    @property
    def width(self) -> int:
        h, w = len(self.pixels), len(self.pixels[0]) if len(self.pixels) else 0
        if (self.rotation // 90) % 2:
            h, w = w, h
        return w * self.scale if self.scale > 1 else w // (1 - self.scale) if self.scale < 0 else w

    @property
    def height(self) -> int:
        h, w = len(self.pixels), len(self.pixels[0]) if len(self.pixels) else 0
        if (self.rotation // 90) % 2:
            h, w = w, h
        return h * self.scale if self.scale > 1 else h // (1 - self.scale) if self.scale < 0 else h

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy

    def set_position(self, x: int, y: int) -> None:
        self.x, self.y = x, y

    def color_remap(self, old, new: int) -> None:
        self.pixels = [[new if (v >= 0 if old is None else v == old) else v for v in row] for row in self.pixels]

    def clone(self, **changes) -> "Sprite":
        return replace(deepcopy(self), **changes)

    def collides_with(self, other: "Sprite", ignore_mode: bool = False) -> bool:
        if self is other or self.screen != other.screen:
            return False
        if not ignore_mode and not (self.collidable and other.collidable):
            return False
        if not ignore_mode and (self.blocking == "none" or other.blocking == "none"):
            return False
        x0, x1 = max(self.x, other.x), min(self.x + self.width, other.x + other.width)
        y0, y1 = max(self.y, other.y), min(self.y + self.height, other.y + other.height)
        if x0 >= x1 or y0 >= y1:
            return False
        if self.blocking != "pixel" and other.blocking != "pixel":
            return True
        a, b = self.render(), other.render()
        return any(
            a[y - self.y][x - self.x] != -1 and b[y - other.y][x - other.x] != -1 for y in range(y0, y1) for x in range(x0, x1)
        )


@dataclass
class Action:
    id: int
    x: int = 0
    y: int = 0
    cell: tuple | None = None


@dataclass
class View:
    scale: int | None = None
    rotation: int = 0
    mirror_ud: bool = False
    mirror_lr: bool = False


@dataclass(eq=False)
class State:
    grid: tuple
    sprites: list = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    status: str = "playing"
    level: int = 0
    view: View = field(default_factory=View)

    def by_tag(self, tag: str) -> list:
        return [s for s in self.sprites if tag in s.tags]

    def by_name(self, name: str):
        return next((s for s in self.sprites if s.name == name), None)

    def sprite_at(self, x: int, y: int, tag: str | None = None, include_uncollidable: bool = False, screen: bool = False):
        for s in sorted(self.sprites, key=lambda s: s.layer, reverse=True):
            if s.screen != screen or not (include_uncollidable or s.collidable):
                continue
            if not (s.x <= x < s.x + s.width and s.y <= y < s.y + s.height):
                continue
            if s.blocking == "pixel" and s.render()[y - s.y][x - s.x] == -1:
                continue
            if tag is None or tag in s.tags:
                return s
        return None

    def sprites_at(self, x: int, y: int, screen: bool = False) -> list:
        return [s for s in self.sprites if s.screen == screen and s.x <= x < s.x + s.width and s.y <= y < s.y + s.height]

    def collisions(self, sprite: Sprite) -> list:
        return [s for s in self.sprites if sprite.collides_with(s)]

    def try_move(self, sprite: Sprite, dx: int, dy: int) -> list:
        sprite.move(dx, dy)
        hits = self.collisions(sprite)
        if hits:
            sprite.move(-dx, -dy)
        return hits

    def add(self, sprite: Sprite) -> Sprite:
        self.sprites.append(sprite)
        return sprite

    def remove(self, sprite: Sprite) -> None:
        self.sprites = [s for s in self.sprites if s is not sprite]


# ==== END OF FIXED INTERFACE ====


# ==== GAME DATA (auto-derived from the recording) ====
# ==== LEVELDATA ====

# ==== END GAME DATA ====


def _pat(shape_str, s):
    rows = [tuple(int(ch) if ch != '.' else 0 for ch in part) for part in shape_str.split('/')]
    if s > 1:
        rows = [[v for v in row for _ in range(s)] for row in rows for _ in range(s)]
    return [[1 if v else -1 for v in row] for row in rows]


def make_level(n: int) -> State:
    lv = LEVELS[n]
    gw, gh, s = lv['grid']
    oy = (64 - gh * s) // 2
    ox = (64 - gw * s) // 2
    sprites = [Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-3, collidable=False, name='border'),
               Sprite([[4] * gw for _ in range(gh)], x=0, y=0, layer=-2, collidable=False, name='background')]
    for col, gx, gy, pat in lv['arrows']:
        pixels = [[col if v else -1 for v in row] for row in _pat(pat, s)]
        sprites.append(Sprite(pixels, x=gx, y=gy, layer=4, collidable=True, blocking='box',
                              tags=('arrow',), name='arrow'))
    cells = {}
    for gx, gy, col in lv['blocks']:
        cells[(gx, gy)] = col
    state = State(grid=(gw, gh), sprites=sprites,
                  vars={'cells': cells, 'used': 0, 'budget': lv['budget'], 'sprites': {}}, level=n)
    sp = {}
    for (gx, gy), col in cells.items():
        pix = [[col] * (2 * s) for _ in range(2 * s)]
        spr = state.add(Sprite(pix, x=gx * 3, y=gy * 3, layer=2, collidable=True, blocking='box',
                              tags=('block',), name='block'))
        sp[(gx, gy)] = spr
    state.vars['sprites'] = sp
    # goal brackets
    for gx, gy, col in lv['goals']:
        px, py = gx * 3, gy * 3
        pix = [[col if (cx in (0, 2) and cy in (0, 2)) else -1 for cx in range(3 * s)] for cy in range(3 * s)]
        state.add(Sprite(pix, x=px - s, y=py - s, layer=5, collidable=False, blocking='none', tags=('goal',)))
    # HUD
    ticks = [[14 if j < n else 5 for _ in range(4)] + ([-1] if j < 8 else []) for j in range(9)]
    flat = [v for row in ticks for v in row]
    state.add(Sprite([flat], x=10, y=1, screen=True, layer=9, collidable=False, blocking='none',
                     tags=('ticks',), name='ticks'))
    state.add(Sprite([[14] for _ in range(64)], x=0, y=0, screen=True, layer=9, collidable=False,
                     blocking='none', tags=('bar',), name='bar'))
    return state


def _draw_bar(state):
    lv = state.vars
    b = lv['budget']
    black = min(64, (64 * lv['used'] + b // 2) // b) if b else 0
    bar = state.by_name('bar')
    bar.pixels = [[5] * black + [[14]] * 0 if False else [5]] * black + [[14]] * (64 - black)


def step(state: State, action: Action) -> None:
    if action.id != 6:
        return
    lv = state.vars
    cells = lv['cells']
    L = state.level
    rings = LEVELS[L]['rings']
    hit = None
    for i, A in enumerate(LEVELS[L]['arrows']):
        gx, gy = A[1], A[2]
        w = len(A[3].split('/')[0]) * LEVELS[L]['scale']
        h = len(A[3].split('/')) * LEVELS[L]['scale']
        if gx <= action.x < gx + w and gy <= action.y < gy + h:
            hit = i
            break
    if hit is None:
        return
    ri, sg = LEVELS[L]['amap'][hit]
    for cyc in rings[ri]:
        n = len(cyc)
        new = {}
        for j, c in enumerate(cyc):
            new[cyc[(j + sg) % n]] = cells.get(c)
        for c in cyc:
            cells[c] = new[c]
    lv['used'] += 1
    for (gx, gy), col in cells.items():
        spr = lv['sprites'].get((gx, gy))
        if spr is not None and col is not None:
            spr.color_remap(None, col)
    for gx, gy, col in LEVELS[L]['goals']:
        if cells.get((gx, gy)) != col:
            break
    else:
        state.status = 'level_solved'
    b = lv['budget']
    black = min(64, (64 * lv['used'] + b // 2) // b) if b else 0
    bar = state.by_name('bar')
    bar.pixels = [[5]] * black + [[14]] * (64 - black)
    if lv['used'] >= b:
        state.status = 'game_over'
