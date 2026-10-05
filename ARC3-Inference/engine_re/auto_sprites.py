"""Sprite code generated from a frame: a starting point for make_level.

``guess_grid(frames)`` guesses the logical grid of a level (size, scale, offset and border colour)
from one or more frames of it. ``sprite_code(frame, grid)`` segments the frame and writes Python
for the level's sprite list that redraws the frame exactly with ``game_api.render``:

- a 64x64 border screen sprite of the border colour (lowest layer, not collidable);
- a grid-sized background sprite of the grid's most common colour (not collidable);
- one sprite per 4-connected region of one colour on the logical grid (the segmentation of
  ``inference/utils/segmentation.py``), or with ``merge=True`` per group of touching non-background
  regions (multi-coloured sprites, transparent around them); objects of identical pixels share
  one module-level constant named from its content (``SHAPE_<colours>_<w>x<h>_<4 hex>``, so the
  same shape gets the same name in every level and constants already in engine.py are not
  written again) and a tag of the same name in lower case;
- screen sprites, on top, for whatever the grid cannot draw: anything outside the grid (HUD)
  and pixels that break the grid's blocks.

The code is executed and rendered to check it reproduces the frame.

What it cannot know: hidden or covered things, transparency, which regions belong to one real
sprite, layers, tags, names and collidability, whether identical-looking objects are the same kind,
and the grid itself is a guess. A pixel-exact start frame does not make the sprites right.

Grid guess. The screen shows a w x h grid scaled by s = min(64 // w, 64 // h) and centred
(``game_api.geometry``), with a border around it. A scale s > 1 is accepted only when every s x s
block of the guessed grid is one colour in every frame given (blocks touching the outer 2 screen
pixels, where HUD bars sit, are not checked) and colour edges exist in both directions; the
largest such scale wins, else 1. A larger scale is a trap when a game at scale 1 happens to align
everything on a coarser lattice (more frames of the level make that less likely). The size: the
smallest centred grid whose surroundings are the border colour, or, when the grid's own main
colour is the border colour (so its edge cannot be seen), the largest one.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass


import numpy as np

from engine_re import game_api

HEX = "0123456789abcdef"
COLOR_NAMES = {
    0: "white", 1: "light grey", 2: "grey", 3: "dark grey", 4: "darker grey", 5: "black",
    6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "light blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}
HUD_MARGIN = 2  # screen pixels at each edge where HUD bars may break the grid's blocks
BAND_PURITY = 0.9  # share of the border colour required on the line just outside the grid
EDGE_SHARE = 0.3  # largest share of the border colour in the grid's outermost cells for its edge to count as visible


@dataclass
class GridGuess:
    width: int
    height: int
    scale: int
    x_offset: int
    y_offset: int
    border: int  # colour around the grid (the most common colour of the screen's outer ring)
    frames: int = 1  # frames the guess was checked against
    note: str = ""

    @property
    def grid(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def default_scale(self) -> bool:
        return self.scale == min(64 // self.width, 64 // self.height)


def _block_bad(frames: list[np.ndarray], s: int, px: int, py: int) -> tuple[np.ndarray, int, int]:
    """For the s x s blocks whose top-left pixels are (py + i*s, px + j*s): True where a block that
    does not touch the HUD margin is not one colour in some frame. Returns (bad, rows, cols)."""
    rows, cols = (64 - py) // s, (64 - px) // s
    bad = np.zeros((rows, cols), bool)
    for f in frames:
        area = f[py : py + rows * s, px : px + cols * s].reshape(rows, s, cols, s)
        bad |= ~(area == area[:, :1, :, :1]).all(axis=(1, 3))
    ys, xs = py + np.arange(rows) * s, px + np.arange(cols) * s
    inner = ((ys >= HUD_MARGIN) & (ys + s <= 64 - HUD_MARGIN))[:, None] & ((xs >= HUD_MARGIN) & (xs + s <= 64 - HUD_MARGIN))[None, :]
    return bad & inner, rows, cols


class _Counts:
    """Pixel counts per colour in any rectangle, summed over the frames (2-D prefix sums)."""

    def __init__(self, frames: list[np.ndarray]):
        onehot = np.zeros((16, 65, 65), np.int64)
        for f in frames:
            onehot[:, 1:, 1:] += (f[None, :, :] % 16 == np.arange(16)[:, None, None])
        self.table = onehot.cumsum(axis=1).cumsum(axis=2)
        self.frames = len(frames)

    def rect(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Counts of each colour in rows r0..r1-1, columns c0..c1-1 (all frames)."""
        t = self.table
        return t[:, r1, c1] - t[:, r0, c1] - t[:, r1, c0] + t[:, r0, c0]


def _sides(s: int, ox: int, oy: int, w: int, h: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """For each side of the grid with screen beyond it: (the 1-pixel line just outside the grid, the
    grid's outermost cell row or column on that side), as (r0, r1, c0, c1) rectangles."""
    x1, y1 = ox + w * s, oy + h * s
    sides = []
    if oy > 0:
        sides.append(((oy - 1, oy, ox, x1), (oy, oy + s, ox, x1)))
    if y1 < 64:
        sides.append(((y1, y1 + 1, ox, x1), (y1 - s, y1, ox, x1)))
    if ox > 0:
        sides.append(((oy, y1, ox - 1, ox), (oy, y1, ox, ox + s)))
    if x1 < 64:
        sides.append(((oy, y1, x1, x1 + 1), (oy, y1, x1 - s, x1)))
    return sides


def _border(counts: _Counts, s: int, ox: int, oy: int, w: int, h: int) -> int | None:
    """The main colour outside the grid (None when the grid fills the screen)."""
    outside = counts.rect(0, 64, 0, 64) - counts.rect(oy, oy + h * s, ox, ox + w * s)
    return int(outside.argmax()) if outside.sum() else None


def _framing(counts: _Counts, border: int | None, s: int, ox: int, oy: int, w: int, h: int) -> int | None:
    """None if a line just outside the grid is not (nearly) the border colour; else the number of
    sides where the grid's edge shows: its outermost cells there are mostly not the border colour."""
    framed = 0
    for outside, inside in _sides(s, ox, oy, w, h):
        line = counts.rect(*outside)
        if line[border] < BAND_PURITY * line.sum():
            return None
        cells = counts.rect(*inside)
        framed += cells[border] <= EDGE_SHARE * cells.sum()
    return framed


def _edges(frames: list[np.ndarray], s: int, ox: int, oy: int, w: int, h: int) -> tuple[int, int]:
    """How many of the grid's inner block boundaries (columns, rows) have a colour change across them."""
    kx = ky = 0
    for k in range(1, w):
        x = ox + k * s
        kx += any(bool((f[oy : oy + h * s, x - 1] != f[oy : oy + h * s, x]).any()) for f in frames)
    for k in range(1, h):
        y = oy + k * s
        ky += any(bool((f[y - 1, ox : ox + w * s] != f[y, ox : ox + w * s]).any()) for f in frames)
    return kx, ky


def guess_grid(frames: np.ndarray | list[np.ndarray], max_scale: int = 8) -> GridGuess:
    """Guess the logical grid from one 64x64 frame or several frames of the same level (more frames
    rule out more coarse scales); see the module docstring."""
    if isinstance(frames, np.ndarray) and frames.ndim == 2:
        frames = [frames]
    frames = [np.asarray(f).astype(np.int16) for f in frames if f is not None]
    if not frames:
        raise ValueError("no frame to guess the grid from")
    counts = _Counts(frames)
    ring = np.concatenate([np.concatenate([f[0], f[-1], f[1:-1, 0], f[1:-1, -1]]) for f in frames])
    ring_colour = int(np.bincount(ring.astype(np.int64) % 16).argmax())
    for s in range(max_scale, 0, -1):
        candidates = []  # (sides framed, area, w, h, ox, oy, border)
        tables: dict[tuple[int, int], np.ndarray] = {}
        for w in range(2, 64 // s + 1):
            for h in range(2, 64 // s + 1):
                if min(64 // w, 64 // h) != s:
                    continue
                ox, oy = (64 - w * s) // 2, (64 - h * s) // 2
                if s > 1:
                    key = (ox % s, oy % s)
                    if key not in tables:
                        tables[key] = _block_bad(frames, s, *key)[0]
                    i0, j0 = (oy - key[1]) // s, (ox - key[0]) // s
                    if tables[key][i0 : i0 + h, j0 : j0 + w].any():
                        continue
                border = _border(counts, s, ox, oy, w, h)
                framed = 0 if border is None else _framing(counts, border, s, ox, oy, w, h)
                if framed is None:
                    continue
                candidates.append((framed, w * h, w, h, ox, oy, ring_colour if border is None else border))
        if not candidates:
            continue
        # A centred grid has a border on both sides of each axis it does not fill, except when the
        # gap is 1 pixel, so a grid framed on fewer than two sides is no evidence: fall back to the
        # largest grid that fits.
        framed = [c for c in candidates if c[0] >= 2]
        if framed:
            _, _, w, h, ox, oy, border = max(framed)
            note = "a grid framed by the border colour"
        else:
            _, _, w, h, ox, oy, border = max(candidates, key=lambda c: c[1])
            note = "no visible edge between grid and border, so the largest grid that fits"
            if s > 1 and (w, h) != (64 // s, 64 // s):
                continue  # a coarse grid floating without a frame: more likely a finer grid's objects
        if s > 1:
            kx, ky = _edges(frames, s, ox, oy, w, h)
            if kx < 2 or ky < 2:
                continue  # too little structure to tell this scale from a finer one
        return GridGuess(w, h, s, ox, oy, border, len(frames), note)
    return GridGuess(64, 64, 1, 0, 0, ring_colour, len(frames), "no grid structure found; the whole screen")


# --- Segmentation ---------------------------------------------------------------------------


def _components(grid: np.ndarray, skip: int | None, merge: bool) -> list[np.ndarray]:
    """Boolean masks of the 4-connected regions of one colour (merge: of any colours but `skip`),
    skipping the colour `skip`, in reading order of their top-left cell."""
    from scipy import ndimage

    masks = []
    if merge:
        labels, n = ndimage.label(grid != skip if skip is not None else np.ones_like(grid, bool))
        masks = [labels == k for k in range(1, n + 1)]
    else:
        for colour in sorted(set(np.unique(grid).tolist()) - {skip}):
            if colour < 0:
                continue
            labels, n = ndimage.label(grid == colour)
            masks += [labels == k for k in range(1, n + 1)]
    def first(mask: np.ndarray) -> tuple[int, int]:
        rows, cols = np.nonzero(mask)
        k = int(np.lexsort((cols, rows))[0])
        return int(rows[k]), int(cols[k])
    return sorted(masks, key=first)


def _cut(values: np.ndarray, mask: np.ndarray) -> tuple[int, int, tuple[str, ...]]:
    """(x, y, rows) of a region: its bounding box as hex strings, '.' outside the region."""
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    sub, inside = values[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]
    text = tuple("".join(HEX[int(v)] if m else "." for v, m in zip(vr, mr)) for vr, mr in zip(sub, inside))
    return int(c0), int(r0), text


def _logical(frame: np.ndarray, g: GridGuess) -> np.ndarray:
    """The frame downsampled to the grid: each block's colour. A block that is not one colour (a HUD
    line drawn over it, say) takes its most common colour; on a tie, the one most of its uniform
    neighbouring blocks have, then its top-left pixel's."""
    s = g.scale
    area = frame[g.y_offset : g.y_offset + g.height * s, g.x_offset : g.x_offset + g.width * s]
    blocks = area.reshape(g.height, s, g.width, s).transpose(0, 2, 1, 3).reshape(g.height, g.width, s * s)
    out = blocks[:, :, 0].copy()
    mixed = ~(blocks == blocks[:, :, :1]).all(axis=2)
    for i, j in zip(*np.nonzero(mixed)):
        counts = Counter(blocks[i, j].tolist())
        best = max(counts.values())
        tied = [v for v, n in counts.items() if n == best]
        if len(tied) > 1:
            around = Counter(
                int(out[a, b]) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                if 0 <= a < g.height and 0 <= b < g.width and not mixed[a, b]
            )
            top = max(around[v] for v in tied)
            tied = [v for v in tied if around[v] == top]
        out[i, j] = int(blocks[i, j, 0]) if int(blocks[i, j, 0]) in tied else tied[0]
    return out


@dataclass
class SpriteCode:
    code: str
    guess: GridGuess
    background: int
    objects: int
    shapes: int
    hud: int
    exact: bool
    differing: int
    function: str
    skipped: list[str]  # names already defined in engine.py, so not emitted again


HEX_PIXELS = '''def hex_pixels(*rows):
    """Pixel rows from hex strings: one digit per pixel (colour 0-15), '.' transparent (-1)."""
    return [[-1 if ch == "." else int(ch, 16) for ch in row] for row in rows]
'''


def shape_name(rows: tuple[str, ...]) -> str:
    """A pixel constant's name from its content: SHAPE_<colours>_<w>x<h>_<4 hex digits of its hash>,
    the same for the same pixels in every call."""
    colours = sorted({ch for row in rows for ch in row if ch != "."}, key=lambda ch: int(ch, 16))
    digest = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:4]
    return f"SHAPE_{'_'.join(str(int(ch, 16)) for ch in colours)}_{len(rows[0])}x{len(rows)}_{digest}"


def _describe_shape(rows: tuple[str, ...], count: int, screen: bool) -> str:
    text = f"{count} sprite{'s' if count > 1 else ''}" + (", screen pixels" if screen else "")
    return text + (", with transparent pixels" if any("." in row for row in rows) else "")


def _shape_lines(name: str, rows: tuple[str, ...], comment: str) -> list[str]:
    """A pixel constant: one hex string per row, so the code shows the shape."""
    if len(rows) == 1:
        return [f'{name} = ("{rows[0]}",)  # {comment}']
    if len(set(rows)) == 1:
        return [f'{name} = ("{rows[0]}",) * {len(rows)}  # {comment}']
    return [f"{name} = (  # {comment}"] + [f'    "{row}",' for row in rows] + [")"]


def _region_mask(region: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros((64, 64), bool)
    if region is None:
        mask[:] = True
    else:
        x0, y0, x1, y1 = (int(v) for v in region)
        mask[max(0, y0) : min(63, y1) + 1, max(0, x0) : min(63, x1) + 1] = True
    return mask


def sprite_code(
    frame: np.ndarray,
    guess: GridGuess,
    *,
    merge: bool = False,
    function: str = "level_0_sprites",
    source: str = "",
    region: tuple[int, int, int, int] | None = None,
    defined: set[str] | None = None,
) -> SpriteCode:
    """Python code for a sprite list that redraws `frame` exactly on the grid `guess` (see the module
    docstring). merge: one sprite per group of touching non-background regions instead of per
    single-colour region. region (x0, y0, x1, y1, screen pixels, inclusive): only the objects inside
    it, without border and background. defined: names engine.py already defines, not emitted again."""
    frame = np.asarray(frame).astype(np.int16)
    g = guess
    defined = set(defined or ())
    inside = _region_mask(region)
    logical = _logical(frame, g)
    background = int(np.bincount(logical.ravel().astype(np.int64) % 16).argmax())
    s = g.scale
    # Cells whose whole block lies in the region.
    cell_in = inside[g.y_offset : g.y_offset + g.height * s, g.x_offset : g.x_offset + g.width * s]
    cell_in = cell_in.reshape(g.height, s, g.width, s).all(axis=(1, 3))
    logical = np.where(cell_in, logical, background)

    names: dict[tuple[str, ...], str] = {}
    order: list[tuple[str, ...]] = []
    counts: Counter = Counter()
    screen_shapes: set[tuple[str, ...]] = set()

    def name_of(rows: tuple[str, ...], screen: bool = False) -> str:
        if rows not in names:
            name = shape_name(rows)
            while name in names.values():  # two different shapes with one name: keep them apart
                name += "_2"
            names[rows] = name
            order.append(rows)
        counts[rows] += 1
        if screen:
            screen_shapes.add(rows)
        return names[rows]

    objects = [(name_of(rows), x, y) for x, y, rows in (_cut(logical, m) for m in _components(logical, background, merge))]

    api = game_api.canonical()
    view = api.View(scale=g.scale) if not g.default_scale else None

    def pixels(rows: tuple[str, ...]) -> list[list[int]]:
        return [[-1 if ch == "." else int(ch, 16) for ch in row] for row in rows]

    base = [
        api.Sprite([[g.border] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False),
        api.Sprite([[background] * g.width for _ in range(g.height)], layer=-1, collidable=False),
    ]
    by_name = {n: r for r, n in names.items()}
    drawn = base + [api.Sprite(pixels(by_name[n]), x=x, y=y) for n, x, y in objects]
    state = api.State(grid=(g.width, g.height), sprites=drawn, **({"view": view} if view else {}))
    residual = (game_api.render(state).astype(np.int16) != frame) & inside
    huds = []
    if residual.any():
        values = np.where(residual, frame, -1)
        huds = [(name_of(rows, True), x, y) for x, y, rows in (_cut(frame, m) for m in _components(values, -1, merge))]

    where = "" if region is None else f", region x {region[0]}-{region[2]}, y {region[1]}-{region[3]}"
    lines = [
        f"# ---- auto_sprites{source}: grid {g.width}x{g.height} at scale {g.scale}{where} ----",
        "# A starting point, not the game's real sprites: rename, merge and retag them, and check them against the steps.",
    ]
    skipped = [n for n in ["hex_pixels", *names.values()] if n in defined]
    if "hex_pixels" not in defined:
        lines += ["", ""] + HEX_PIXELS.rstrip("\n").split("\n")
    constants = [r for r in order if names[r] not in defined]
    if constants:
        lines += ["", ""]
        for rows in constants:
            lines += _shape_lines(names[rows], rows, _describe_shape(rows, counts[rows], rows in screen_shapes))
    view_text = f", view=View(scale={g.scale})" if view else ""
    what = "objects in the region" if region is not None else "sprites of the frame"
    lines += [
        "",
        "",
        f"def {function}() -> list:",
        f'    """The {what} (grid {g.width}x{g.height}): State(grid=({g.width}, {g.height}), sprites={function}(){view_text})."""',
        "    return [",
    ]
    if region is None:
        lines += [
            f"        Sprite([[{g.border}] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name=\"border\"),",
            f"        Sprite([[{background}] * {g.width} for _ in range({g.height})], layer=-1, collidable=False, name=\"background\"),",
        ]
    for name, x, y in objects:
        lines.append(f'        Sprite(hex_pixels(*{name}), x={x}, y={y}, tags=("{name.lower()}",)),')
    for name, x, y in huds:
        lines.append(f'        Sprite(hex_pixels(*{name}), x={x}, y={y}, screen=True, layer=1, collidable=False, tags=("hud", "{name.lower()}")),')
    lines.append("    ]")
    code = "\n".join(lines) + "\n\n\n"  # two blank lines after it, ready to insert before the next definition

    # Check: run the code (with the constants it did not repeat) on the fixed interface and draw it.
    namespace = dict(vars(api))
    exec(compile(HEX_PIXELS, "<auto_sprites>", "exec", dont_inherit=True), namespace)
    namespace.update({n: r for r, n in names.items()})
    exec(compile(code, "<auto_sprites>", "exec", dont_inherit=True), namespace)
    made = namespace[function]()
    check = api.State(grid=(g.width, g.height), sprites=(base + made) if region is not None else made, **({"view": api.View(scale=g.scale)} if view else {}))
    differing = int(((game_api.render(check).astype(np.int16) != frame) & inside).sum())
    return SpriteCode(
        code=code, guess=g, background=background, objects=len(objects), shapes=len({n for n, _, _ in objects}),
        hud=len(huds), exact=differing == 0, differing=differing, function=function, skipped=skipped,
    )
