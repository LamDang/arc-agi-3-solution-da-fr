"""The pieces of a frame, and what changed between two frames: auto_sprites' segmentation as objects.

    pieces(frame, grid=None, known=None) -> Pieces   the sprites that redraw a frame exactly
    changes(before, after) -> list[Change]           what changed between two frames' pieces
    summary(changes, limit=12) -> str                a short summary of them, similar changes grouped

A frame is split as ``auto_sprites.sprite_code`` splits it (the same plan, ``plan_sprites``): on the
guessed logical grid, a border sprite, a background sprite and one sprite per 4-connected region of
one colour (the segmentation of ``inference/utils/segmentation.py``), then screen sprites for what the
grid cannot draw. Each piece is a real fixed-interface Sprite (``Piece``, a subclass) carrying what
the segmentation knows: its shape's stable name (``SHAPE_<colours>_<w>x<h>_<hex>``; a piece that is
an earlier shape turned, mirrored, scaled or recoloured is that shape with a ``transform``), main
colour, size, the pieces it encloses (``children``, as in segmentation.py) and its role.
``Pieces.code()`` is the Python that auto_sprites writes for the frame.

``Segmenter`` holds a recording's grids, pieces and changes for the steps the model may see (0 to
``last``), computed lazily: the kernel's StepView attributes and the step messages use it.
Nothing here reads files or keeps global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from engine_re import game_api
from engine_re.auto_sprites import COLOR_NAMES, GridGuess, Placement, SpriteCode, guess_grid, plan_sprites, sprite_code

_API = game_api.canonical()
KINDS = ("moved", "recoloured", "reshaped", "appeared", "disappeared")


@dataclass(eq=False, repr=False)
class Piece(_API.Sprite):
    """A Sprite of a segmented frame, with what the segmentation knows about it. It prints as the
    code that builds the sprite (Sprite.__repr__)."""

    shape: str = ""  # the shape's stable name, e.g. "SHAPE_9_2x2_7cf8"; "border" or "background" for those
    colour: int = 0  # its most common colour
    size: int = 0  # its opaque pixels: grid cells, or screen pixels for a screen piece
    transform: dict = field(default_factory=dict)  # rotation, mirror_ud, mirror_lr, scale, recolour vs its shape
    children: list = field(default_factory=list)  # indices of the pieces it encloses (innermost encloser only)
    role: str = "object"  # "border", "background" or "object"


def _amount(n: int, screen: bool) -> str:
    """'5 px', '1 cell', '4 cells'."""
    return f"{n} px" if screen else f"{n} cell{'' if n == 1 else 's'}"


def _colour_text(colour: int) -> str:
    return f"{colour} ({COLOR_NAMES.get(colour, '?')})"


def _label(p: Any) -> str:
    """'SHAPE_9_2x2_7cf8', 'screen piece SHAPE_14_1x31_ee7b', 'the border', 'the background'."""
    role = getattr(p, "role", "object")
    if role in ("border", "background"):
        return f"the {role}"
    return ("screen piece " if p.screen else "") + (getattr(p, "shape", "") or "piece")


class Pieces(list):
    """The pieces of one frame (a list of Piece, drawn in list order), with .grid (the GridGuess they
    sit on) and .code() (Python for a sprite list that draws the frame exactly)."""

    def __init__(self, items: list[Piece], grid: GridGuess, frame: np.ndarray, known: dict[str, Any] | None = None,
                 function: str = "frame_sprites", source: str = "the frame"):
        super().__init__(items)
        self.grid = grid
        self._frame = frame
        self._known = dict(known or {})
        self._function, self._source = function, source
        # Set by the kernel: () -> (engine.py's module-level values, the names it defines), so code() reuses its constants.
        self._context: Callable[[], tuple[dict[str, Any], set[str]]] | None = None

    def _sprite_code(self) -> SpriteCode:
        existing, defined = dict(self._known), set()
        if self._context is not None:
            values, defined = self._context()
            existing.update(values)
        return sprite_code(self._frame, self.grid, function=self._function, source=self._source, existing=existing, defined=defined)

    def code(self) -> str:
        """Python code (usable as edit_file lines) for a sprite list that draws the frame exactly: the
        shapes as named constants, one Sprite per piece. In the kernel it reuses engine.py's pixel
        constants (as they are, turned, mirrored, scaled or recoloured) and does not repeat what
        engine.py defines."""
        return self._sprite_code().code

    def __str__(self) -> str:
        g = self.grid
        lines = [f"{len(self)} pieces on a {g.width}x{g.height} grid at scale {g.scale}, offset ({g.x_offset}, {g.y_offset}) "
                 "(x, y, size in grid cells; in screen pixels for screen pieces):"]
        for i, p in enumerate(self):
            name = p.role if p.role != "object" else _label(p)
            text = f"  [{i}] {name}  colour {_colour_text(p.colour)}  {p.width}x{p.height} at ({p.x}, {p.y})"
            if p.role == "object":
                text += f", {_amount(p.size, p.screen)}"
            if p.transform:
                text += "  " + ", ".join(f"{k}={v}" for k, v in p.transform.items())
            if p.children and p.role == "object":
                text += f"  encloses {p.children}"
            lines.append(text)
        return "\n".join(lines)

    __repr__ = __str__


def _given_grid(frame: np.ndarray, grid: tuple[int, int]) -> GridGuess:
    w, h = int(grid[0]), int(grid[1])
    s, ox, oy = game_api.geometry((w, h))
    ring = np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]])
    return GridGuess(w, h, s, ox, oy, int(np.bincount(np.asarray(ring, np.int64) % 16).argmax()), 1, "as given")


def _transform(p: Placement) -> dict:
    out: dict[str, Any] = {}
    for name, default in (("rotation", 0), ("mirror_ud", False), ("mirror_lr", False), ("scale", 1)):
        if getattr(p, name) != default:
            out[name] = getattr(p, name)
    if p.recolor:
        out["recolour"] = dict(p.recolor)
    return out


def _drawn(p: Any) -> np.ndarray:
    """A sprite's pixels as drawn (rotated, mirrored, scaled), -1 where transparent."""
    px = game_api.sprite_pixels(p)
    return np.where(px >= 0, px, -1).astype(np.int16)


def _main_colour(px: np.ndarray) -> int:
    values = px[px >= 0]
    if not values.size:
        return -1
    counts = np.bincount(values.astype(np.int64), minlength=16)
    return int(counts.argmax())


def _enclosures(items: list[Piece]) -> None:
    """Fill each piece's children: the pieces it encloses (every path from them to the edge of their
    space crosses it), each listed only under its innermost encloser, as segmentation.py does. The
    background holds the grid's other top-level pieces, the border the background and the top-level
    screen pieces."""
    from scipy import ndimage

    drawn = {i: _drawn(p) >= 0 for i, p in enumerate(items) if p.role == "object"}
    enclosers: dict[int, set[int]] = {i: set() for i in drawn}
    for b, mask in drawn.items():
        h, w = mask.shape
        if h < 3 or w < 3:
            continue
        holes = ndimage.binary_fill_holes(mask) & ~mask
        if not holes.any():
            continue
        pb = items[b]
        for a in drawn:
            pa = items[a]
            if a == b or pa.screen != pb.screen:
                continue
            rows, cols = np.nonzero(drawn[a])
            if not rows.size:
                continue
            k = int(np.lexsort((cols, rows))[0])
            y, x = pa.y + int(rows[k]) - pb.y, pa.x + int(cols[k]) - pb.x
            if 0 <= y < h and 0 <= x < w and holes[y, x]:
                enclosers[a].add(b)
    for a, around in enclosers.items():
        if around:
            parent = max(around, key=lambda e: (len(enclosers[e]), -e))
            items[parent].children.append(a)
    border = next((i for i, p in enumerate(items) if p.role == "border"), None)
    background = next((i for i, p in enumerate(items) if p.role == "background"), None)
    for a, around in enclosers.items():
        if not around:
            owner = border if items[a].screen else background
            if owner is not None:
                items[owner].children.append(a)
    if border is not None and background is not None:
        items[border].children.insert(0, background)
    for p in items:
        p.children.sort()


def pieces(frame: Any, grid: GridGuess | tuple[int, int] | None = None, known: dict[str, Any] | None = None, *,
           function: str = "frame_sprites", source: str = "the frame") -> Pieces:
    """The pieces of a 64x64 frame: Sprites that redraw it exactly with game_api.render on the grid
    (a GridGuess; a (w, h) size; None: guessed from this frame alone). known: pixel constants
    (name -> hex rows or rows of numbers) a piece may be drawn from instead of a new shape.
    function, source: the name and description of what code() writes."""
    frame = np.asarray(frame).astype(np.int16)
    if frame.shape != (64, 64):
        raise ValueError(f"a frame is 64x64, got {frame.shape}")
    if grid is None:
        g = guess_grid([frame])
    elif isinstance(grid, GridGuess):
        g = grid
    else:
        g = _given_grid(frame, grid)
    plan = plan_sprites(frame, g, existing=known)
    items: list[Piece] = [
        Piece([[g.border] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border",
              shape="border", colour=g.border, size=64 * 64, role="border"),
        Piece([[plan.background] * g.width for _ in range(g.height)], layer=-1, collidable=False, name="background",
              shape="background", colour=plan.background, size=g.width * g.height, role="background"),
    ]
    for p in plan.objects + plan.huds:
        sprite = plan.sprite(p)
        extra: dict[str, Any] = {"tags": ("hud", p.name.lower()) if p.screen else (p.name.lower(),)}
        if p.screen:
            extra.update(screen=True, layer=1, collidable=False)
        piece = Piece(sprite.pixels, x=sprite.x, y=sprite.y, rotation=sprite.rotation, mirror_ud=sprite.mirror_ud,
                      mirror_lr=sprite.mirror_lr, scale=sprite.scale, shape=p.name, transform=_transform(p), **extra)
        px = _drawn(piece)
        piece.colour, piece.size = _main_colour(px), int((px >= 0).sum())
        items.append(piece)
    _enclosures(items)
    return Pieces(items, g, frame, known, function, source)


# --- What changed between two frames ---------------------------------------------------------------


@dataclass(eq=False)
class Change:
    """One object-level change between two frames' pieces."""

    kind: str  # "moved", "recoloured", "reshaped", "appeared" or "disappeared"
    before: Any = None  # the piece before (None: appeared)
    after: Any = None  # the piece after (None: disappeared)
    dx: int = 0  # how far it moved (moved; reshaped: its top-left corner)
    dy: int = 0
    colours: dict = field(default_factory=dict)  # {old: new} colours (recoloured; reshaped in another colour)
    note: str = ""  # reshaped: how, e.g. "1x31 -> 1x26, lost 5 px at the top"

    def __str__(self) -> str:
        p = self.after if self.after is not None else self.before
        if self.kind == "moved":
            return (f"moved: {_label(p)} colour {_colour_text(p.colour)}, ({self.before.x}, {self.before.y}) -> "
                    f"({self.after.x}, {self.after.y}), dx={self.dx:+d} dy={self.dy:+d}")
        if self.kind == "recoloured":
            return f"recoloured: {_label(p)} {p.width}x{p.height} at ({p.x}, {p.y}): {_map_text(self.colours)}"
        if self.kind == "reshaped":
            text = f"reshaped: {_label(self.before)} colour {_colour_text(self.before.colour)} at ({self.before.x}, {self.before.y}): {self.note}"
            return text + (f"; colours {_map_text(self.colours)}" if self.colours else "")
        return f"{self.kind}: {_label(p)} colour {_colour_text(p.colour)}, {p.width}x{p.height} at ({p.x}, {p.y}), {_amount(p.size, p.screen)}"

    __repr__ = __str__


class Changes(list):
    """A list of Change that prints one line per change."""

    def __str__(self) -> str:
        return "\n".join(str(c) for c in self) if self else "no piece changed"

    __repr__ = __str__


def _map_text(colours: dict) -> str:
    return ", ".join(f"{a}->{b}" for a, b in colours.items())


class _Look:
    """A piece as drawn in its space: top-left, drawn pixels, mask. space: "screen", or the grid it
    sits on (pieces of two different grids are never the same piece)."""

    def __init__(self, p: Any, grid: Any):
        self.px = _drawn(p)
        self.mask = self.px >= 0
        self.x, self.y, self.screen = int(p.x), int(p.y), bool(p.screen)
        self.space: Any = "screen" if self.screen else grid
        self.size = int(self.mask.sum())
        self.colour = _main_colour(self.px)

    def box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of its opaque pixels, inclusive."""
        rows, cols = np.nonzero(self.mask)
        if not rows.size:
            return self.x, self.y, self.x - 1, self.y - 1
        return self.x + int(cols.min()), self.y + int(rows.min()), self.x + int(cols.max()), self.y + int(rows.max())

    def overlap(self, other: _Look) -> int:
        if self.space != other.space:
            return 0
        h, w = self.mask.shape
        oh, ow = other.mask.shape
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1, y1 = min(self.x + w, other.x + ow), min(self.y + h, other.y + oh)
        if x0 >= x1 or y0 >= y1:
            return 0
        a = self.mask[y0 - self.y : y1 - self.y, x0 - self.x : x1 - self.x]
        b = other.mask[y0 - other.y : y1 - other.y, x0 - other.x : x1 - other.x]
        return int((a & b).sum())


def _reshape_note(b: _Look, a: _Look) -> str:
    (bx0, by0, bx1, by1), (ax0, ay0, ax1, ay1) = b.box(), a.box()
    size = f"{bx1 - bx0 + 1}x{by1 - by0 + 1} -> {ax1 - ax0 + 1}x{ay1 - ay0 + 1}"
    d = a.size - b.size
    sides = [name for name, moved in (("top", ay0 != by0), ("bottom", ay1 != by1), ("left", ax0 != bx0), ("right", ax1 != bx1)) if moved]
    if d:
        what = f"{'lost' if d < 0 else 'gained'} {_amount(abs(d), b.screen)}"
    else:
        what = f"the same {_amount(b.size, b.screen)}, other pixels"
    if not sides:
        where = " inside its box"
    elif len(sides) == 1:
        where = f" at the {sides[0]}"
    else:
        where = f" (edges moved: {', '.join(sides)})"
    return f"{size}, {what}{where}"


def changes(before: list, after: list) -> Changes:
    """What changed from the pieces `before` to the pieces `after` (unchanged pieces are not listed),
    matched in this order: the same pixels at the same place (unchanged); the same place and shape
    in other colours ("recoloured", with the colour map); the same pixels elsewhere ("moved", the
    nearest first); an overlapping piece of the same space with other pixels or size, sharing at
    least half of the smaller one ("reshaped", same colour first); the rest "disappeared" or
    "appeared". Pieces match only within one space: the screen, or the grid, when both frames have
    the same grid (size, scale and offset; a plain list of sprites counts as on the same grid)."""
    b, a = list(before or []), list(after or [])

    def grid_key(pieces_: Any) -> Any:
        g = getattr(pieces_, "grid", None)
        return None if g is None else (g.width, g.height, g.scale, g.x_offset, g.y_offset)

    looks = {id(p): _Look(p, grid_key(before)) for p in b}
    looks.update({id(p): _Look(p, grid_key(after)) for p in a})
    free_b, free_a = list(range(len(b))), list(range(len(a)))
    out: list[Change] = []

    def pair_by(key: Callable[[_Look], Any]) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        index: dict[Any, list[int]] = {}
        for j in free_a:
            index.setdefault(key(looks[id(a[j])]), []).append(j)
        for i in list(free_b):
            group = index.get(key(looks[id(b[i])]))
            if group:
                j = group.pop(0)
                found.append((i, j))
                free_b.remove(i)
                free_a.remove(j)
        return found

    pair_by(lambda k: (k.space, k.x, k.y, k.px.shape, k.px.tobytes()))  # unchanged
    for i, j in pair_by(lambda k: (k.space, k.x, k.y, k.mask.shape, k.mask.tobytes())):
        lb, la = looks[id(b[i])], looks[id(a[j])]
        pairs = dict(zip(lb.px[lb.mask].tolist(), la.px[la.mask].tolist()))
        colours = {int(o): int(n) for o, n in sorted(pairs.items()) if o != n}
        out.append(Change("recoloured", b[i], a[j], colours=colours))
    # Moved: the same pixels elsewhere, the nearest pairs first.
    same: dict[Any, list[int]] = {}
    for j in free_a:
        la = looks[id(a[j])]
        same.setdefault((la.space, la.px.shape, la.px.tobytes()), []).append(j)
    candidates = []
    for i in free_b:
        lb = looks[id(b[i])]
        for j in same.get((lb.space, lb.px.shape, lb.px.tobytes()), []):
            la = looks[id(a[j])]
            candidates.append((abs(la.x - lb.x) + abs(la.y - lb.y), i, j))
    for _, i, j in sorted(candidates):
        if i in free_b and j in free_a:
            free_b.remove(i)
            free_a.remove(j)
            out.append(Change("moved", b[i], a[j], dx=a[j].x - b[i].x, dy=a[j].y - b[i].y))
    # Reshaped: overlapping, sharing at least half of the smaller piece; the same colour first.
    candidates = []
    for i in free_b:
        lb = looks[id(b[i])]
        for j in free_a:
            la = looks[id(a[j])]
            shared = lb.overlap(la)
            if shared and 2 * shared >= min(lb.size, la.size):
                candidates.append((lb.colour != la.colour, -shared, i, j))
    for _, _, i, j in sorted(candidates):
        if i in free_b and j in free_a:
            free_b.remove(i)
            free_a.remove(j)
            lb, la = looks[id(b[i])], looks[id(a[j])]
            (bx0, by0, _, _), (ax0, ay0, _, _) = lb.box(), la.box()
            colours = {lb.colour: la.colour} if lb.colour != la.colour else {}
            out.append(Change("reshaped", b[i], a[j], dx=ax0 - bx0, dy=ay0 - by0, colours=colours,
                              note=_reshape_note(lb, la)))
    out += [Change("disappeared", b[i], None) for i in free_b]
    out += [Change("appeared", None, a[j]) for j in free_a]

    def order(c: Change) -> tuple:
        p = c.after if c.after is not None else c.before
        return KINDS.index(c.kind), p.screen, p.y, p.x
    return Changes(sorted(out, key=order))


# --- A short summary ----------------------------------------------------------------------------

LINE_CHARS = 170


def _items_line(head: str, items: list[str]) -> str:
    """head + as many items as fit in LINE_CHARS, then how many more."""
    text = head
    for k, item in enumerate(items):
        more = f", ... and {len(items) - k} more"
        piece = (", " if k else "") + item
        if len(text) + len(piece) + (len(more) if k < len(items) - 1 else 0) > LINE_CHARS and k:
            return text + more
        text += piece
    return text


def _where(p: Any) -> str:
    return f"({p.x}, {p.y})"


def summary(found: list[Change], limit: int = 12) -> str:
    """A short summary of changes: one line per group of similar ones (moved by the same step,
    recoloured pieces of one size, appeared or disappeared pieces of one shape; each reshaped piece
    on its own), at most `limit` lines and then how many more groups there are."""
    if not found:
        return "no piece changed"
    groups: dict[tuple, list[Change]] = {}
    for c in found:
        p = c.after if c.after is not None else c.before
        if c.kind == "moved":
            key: tuple = (c.kind, c.dx, c.dy, p.screen)
        elif c.kind == "recoloured":
            key = (c.kind, p.width, p.height, p.screen, getattr(p, "role", "object"))
        elif c.kind == "reshaped":
            key = (c.kind, id(c))
        else:
            key = (c.kind, getattr(p, "shape", ""), p.colour, p.screen)
        groups.setdefault(key, []).append(c)
    lines = []
    for key, group in groups.items():
        kind, n = key[0], len(group)
        first = group[0].after if group[0].after is not None else group[0].before
        screen = " screen" if first.screen else ""
        shapes = sorted({getattr(c.after if c.after is not None else c.before, "shape", "") for c in group})
        shape_text = shapes[0] if len(shapes) == 1 else f"{len(shapes)} shapes"
        if kind == "moved":
            head = f"{n} moved by ({group[0].dx:+d}, {group[0].dy:+d}){screen} ({shape_text}): "
            items = [f"{_where(c.before)}->{_where(c.after)}" for c in group]
        elif kind == "recoloured":
            what = f"the {first.role}" if first.role != "object" else f"{first.width}x{first.height}{screen}"
            head = f"{n} recoloured ({what}{', ' + shape_text if first.role == 'object' else ''}): "
            items = [f"{_where(c.after)} {_map_text(c.colours)}" for c in group]
        elif kind == "reshaped":
            lines.append(_items_line("1 reshaped: " + str(group[0]).split(": ", 1)[1], []))
            continue
        else:
            head = (f"{n} {kind}{screen} ({shape_text}, colour {_colour_text(first.colour)}, {first.width}x{first.height}, "
                    f"{_amount(first.size, first.screen)}): ")
            items = [_where(c.after if c.after is not None else c.before) for c in group]
        lines.append(_items_line(head, items))
    if len(lines) > limit:
        lines = lines[: limit - 1] + [f"... and {len(lines) - limit + 1} more groups of changes"]
    return "\n".join(lines)


# --- A recording's steps ------------------------------------------------------------------------

MAX_GRID_FRAMES = 60  # frames of a level the grid is guessed from (evenly spaced)


def shown_level(trace: Any, i: int) -> int:
    """The level step i's final frame shows: the levels completed after it, but on WIN the last
    level's (the game shows the state as it is)."""
    s = trace.steps[i]
    return max(0, s.levels_completed - 1) if s.state == "WIN" else int(s.levels_completed)


class Segmenter:
    """Grids, pieces and changes of a recording's steps 0 to `last` (default: all), computed when
    first asked for and kept. Only those steps are ever looked at: a step after `last` raises.

    grid(level): the GridGuess of a level, from the final frames of its steps up to `last` (not a
    WIN frame); pieces(i): step i's final frame on the grid of the level it shows; changes(k):
    from step k-1's pieces to step k's (None for step 0). `context`, when set, is given to every
    Pieces for code() (the kernel: engine.py's constants)."""

    def __init__(self, trace: Any, last: int | None = None):
        self.trace = trace
        self.last = len(trace.steps) - 1 if last is None else min(int(last), len(trace.steps) - 1)
        self.context: Callable[[], tuple[dict[str, Any], set[str]]] | None = None
        self._grids: dict[int, GridGuess] = {}
        self._pieces: dict[int, Pieces] = {}
        self._changes: dict[int, Changes] = {}

    def _check(self, i: int) -> None:
        if not 0 <= i <= self.last:
            raise ValueError(f"only steps 0-{self.last} can be segmented, not step {i}")

    def grid(self, level: int) -> GridGuess:
        if level not in self._grids:
            steps = self.trace.steps
            shown = [i for i in range(self.last + 1)
                     if steps[i].last is not None and steps[i].state != "WIN" and shown_level(self.trace, i) == level]
            if not shown:  # a level seen only on a WIN frame
                shown = [i for i in range(self.last + 1) if steps[i].last is not None and shown_level(self.trace, i) == level]
            if not shown:
                raise ValueError(f"no frame of level {level} in steps 0-{self.last}")
            if len(shown) > MAX_GRID_FRAMES:
                shown = [shown[round(k * (len(shown) - 1) / (MAX_GRID_FRAMES - 1))] for k in range(MAX_GRID_FRAMES)]
            self._grids[level] = guess_grid([steps[i].last for i in shown])
        return self._grids[level]

    def enters_level(self, i: int) -> bool:
        """Whether step i's final frame is the first frame of a level it entered (step 0: level 0)."""
        return i == 0 or (self.trace.steps[i].state != "WIN" and shown_level(self.trace, i) != shown_level(self.trace, i - 1))

    def pieces(self, i: int) -> Pieces:
        self._check(i)
        if i not in self._pieces:
            level = shown_level(self.trace, i)
            first = self.enters_level(i)
            function = f"level_{level}_sprites" if first else f"step_{i}_sprites"
            source = f"recording[{i}].after" + (f", level {level}'s first frame" if first else "")
            made = pieces(self.trace.steps[i].last, self.grid(level), function=function, source=source)
            made._context = self.context
            self._pieces[i] = made
        return self._pieces[i]

    def changes(self, k: int) -> Changes | None:
        self._check(k)
        if k == 0:
            return None
        if k not in self._changes:
            self._changes[k] = changes(self.pieces(k - 1), self.pieces(k))
        return self._changes[k]

    def report(self, k: int, name: str = "step_to_fix", limit: int = 12) -> str:
        """The objects block of a step message: what step k changed, summarised (at most `limit` lines)."""
        self._check(k)
        after = self.pieces(k)
        level = shown_level(self.trace, k)
        head = "What the recorded step changed (objects):"
        if k == 0:
            return (f"{head} it starts the game. {name}.pieces_after holds level 0's first frame as {len(after)} pieces; "
                    f"{name}.pieces_after.code() writes sprites that draw it.")
        if self.enters_level(k):
            return (f"{head} it enters level {level}. {name}.pieces_after holds that level's first frame as {len(after)} pieces "
                    f"on a {after.grid.width}x{after.grid.height} grid at scale {after.grid.scale}; {name}.pieces_after.code() "
                    "writes sprites that draw it.")
        text = summary(self.changes(k), limit)
        return (f"{head} {name}.pieces_before -> {name}.pieces_after; {name}.changes lists them.\n"
                + "\n".join("  " + line for line in text.splitlines()))
