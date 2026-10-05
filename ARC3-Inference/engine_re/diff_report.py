"""Explaining a failing step: where the final frames differ, which of the engine's sprites are there,
and a picture of both frames with the differing regions boxed and numbered.

Used by the tester (with state summaries computed in the sandboxed candidate process) and by the
kernel helper ``try_step`` (with summaries of states in the kernel), so both print the same thing.

- ``find_regions``: differing pixels clustered into numbered regions (connected components with a
  small gap tolerance, then bounding boxes with a 1-pixel margin).
- ``describe_frames``: the text for one pair of final frames: per region its box in screen pixels
  and in the engine's grid cells, the colour changes, and the engine's sprites that draw there (by
  their index in ``state.sprites``), before and after the step.
- ``state_changes``: what a step changed in the engine's own state (sprites matched by object
  identity, so additions and removals do not shift them; vars; status), for ``try_step``.
- ``comparison_image``: the engine's frame and the original's side by side, upscaled, with the
  same numbered boxes; ``png_bytes`` / ``data_url`` encode it for a chat message.
"""

from __future__ import annotations

import base64
import json
import io
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine_re import game_api

HEX = "0123456789abcdef"
COLOR_NAMES = {
    0: "white", 1: "light grey", 2: "grey", 3: "dark grey", 4: "darker grey", 5: "black",
    6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "light blue", 11: "yellow",
    12: "orange", 13: "maroon", 14: "green", 15: "purple",
}
# The ARC palette as RGB, the same as inference.agent.vision_context.ARC_COLOR_MAP (a test checks it).
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}
# Box outlines: a saturated cyan, far from every palette colour, edged in black so it shows on light
# and dark pixels alike.
BOX_RGB = (0, 255, 255)
EDGE_RGB = (0, 0, 0)
UPSCALE = 8
MAX_REGIONS = 6  # regions numbered and described; smaller ones beyond this are only counted
MAX_SPRITES = 4  # sprites listed per region


@dataclass
class Region:
    n: int  # number shown in the text and on the boxes, from 1
    core: tuple[int, int, int, int]  # (r0, c0, r1, c1) of the differing pixels, inclusive
    box: tuple[int, int, int, int]  # the same with a 1-pixel margin, clipped to the screen: what is drawn
    mask: np.ndarray  # 64x64 bool: the differing pixels of this region
    changes: list[tuple[int, int, int]]  # (expected colour, got colour, pixels), most common first

    @property
    def pixels(self) -> int:
        return int(self.mask.sum())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _grow(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - margin), max(0, c0 - margin), min(63, r1 + margin), min(63, c1 + margin)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def find_regions(expected: np.ndarray, got: np.ndarray, gap: int = 2, margin: int = 1, limit: int = MAX_REGIONS) -> tuple[list[Region], int]:
    """Cluster the pixels where two 64x64 frames differ into regions.

    Pixels at most ``gap`` empty pixels apart (in any direction) join one region; regions whose
    boxes (with ``margin``) overlap are merged. Returns the ``limit`` largest regions, numbered in
    reading order, and how many smaller ones were left out."""
    from scipy import ndimage

    expected, got = np.asarray(expected), np.asarray(got)
    diff = expected != got
    if not diff.any():
        return [], 0
    grown = ndimage.binary_dilation(diff, structure=np.ones((3, 3), bool), iterations=max(1, (gap + 1) // 2))
    labels, n = ndimage.label(grown, structure=np.ones((3, 3), int))
    groups = [diff & (labels == k) for k in range(1, n + 1)]
    groups = [g for g in groups if g.any()]
    merged = True
    while merged:  # merge clusters whose drawn boxes would overlap
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _overlap(_grow(_bbox(groups[i]), margin), _grow(_bbox(groups[j]), margin)):
                    groups[i] = groups[i] | groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: -int(g.sum()))
    shown, hidden = groups[:limit], len(groups) - min(len(groups), limit)
    shown.sort(key=lambda g: (_bbox(g)[0], _bbox(g)[1]))
    regions = []
    for k, mask in enumerate(shown, 1):
        changes = Counter(zip(expected[mask].tolist(), got[mask].tolist())).most_common()
        regions.append(Region(k, _bbox(mask), _grow(_bbox(mask), margin), mask, [(int(a), int(b), int(c)) for (a, b), c in changes]))
    return regions, hidden


# --- Text ------------------------------------------------------------------------------


def _hexrow(row: np.ndarray) -> str:
    return "".join(HEX[v] if 0 <= v < 16 else "?" for v in row)


def hex_crop(expected: np.ndarray, got: np.ndarray, box: tuple[int, int, int, int], max_rows: int = 12, max_cols: int = 24, indent: str = "      ") -> list[str]:
    """The original's and the engine's pixels side by side in a box, one hex digit per pixel."""
    r0, c0, r1, c1 = box
    r1, c1 = min(r1, r0 + max_rows - 1), min(c1, c0 + max_cols - 1)
    width = c1 - c0 + 1
    lines = [f"{indent}rows {r0}-{r1}, cols {c0}-{c1}, one hex digit per pixel:", f"{indent}{'row':>3}  {'original':<{width}}  {'yours':<{width}}  {'x = differs':<{width}}"]
    for r in range(r0, r1 + 1):
        e, g = expected[r, c0 : c1 + 1], got[r, c0 : c1 + 1]
        lines.append(f"{indent}{r:>3}  {_hexrow(e):<{width}}  {_hexrow(g):<{width}}  " + "".join("x" if a != b else "." for a, b in zip(e, g)))
    return lines


def _range(a: int, b: int) -> str:
    return f"{a}" if a == b else f"{a}-{b}"


def _changes_text(changes: list[tuple[int, int, int]], limit: int = 4) -> str:
    text = ", ".join(f"{a}->{b} x{n}" for a, b, n in changes[:limit])
    return text + (f", ... ({len(changes) - limit} more)" if len(changes) > limit else "")


def cells_text(summary: dict[str, Any] | None, mask: np.ndarray) -> str:
    """Where a set of screen pixels lies on the engine's grid: 'grid cells x 3-5, y 7' or 'outside the grid'."""
    if not summary or "grid" not in summary:
        return ""
    view = summary.get("view") or {}
    view_obj = game_api.canonical().View(
        scale=view.get("scale"), rotation=view.get("rotation", 0), mirror_ud=view.get("mirror_ud", False), mirror_lr=view.get("mirror_lr", False)
    )
    cells = [game_api.to_grid(tuple(summary["grid"]), int(c), int(r), view_obj) for r, c in zip(*np.nonzero(mask))]
    on = [c for c in cells if c is not None]
    if not on:
        return "outside your grid"
    xs, ys = [c[0] for c in on], [c[1] for c in on]
    text = f"your grid cells x {_range(min(xs), max(xs))}, y {_range(min(ys), max(ys))}"
    if len(on) < len(cells):
        text += " and outside the grid"
    return text


def _owner_map(summary: dict[str, Any], masks: list[np.ndarray | None]) -> np.ndarray:
    """Index of the sprite whose pixel shows at each screen pixel (-1: none, the screen colour 5)."""
    owner = np.full((64, 64), -1, np.int32)
    sprites = summary["sprites"]
    for i in sorted(range(len(sprites)), key=lambda i: (sprites[i].get("layer", 0) if isinstance(sprites[i].get("layer"), int) else 0, i)):
        if masks[i] is not None and sprites[i].get("visible") is True:
            owner[masks[i]] = i
    return owner


class SummaryView:
    """A state_summary with its footprints unpacked, for region queries."""

    def __init__(self, summary: dict[str, Any] | None):
        self.summary = summary if summary and "sprites" in summary else None
        self.error = (summary or {}).get("error") if summary else None
        if self.summary is not None:
            self.masks = [game_api.unpack_footprint(e) for e in self.summary["sprites"]]
            self.owner = _owner_map(self.summary, self.masks)

    def __bool__(self) -> bool:
        return self.summary is not None

    def at(self, mask: np.ndarray) -> list[tuple[int, int, int]]:
        """(index, pixels on top, pixels drawn) of every sprite that draws in `mask` (hidden ones as
        if visible), the most visible first."""
        if self.summary is None:
            return []
        out = []
        for i, m in enumerate(self.masks):
            if m is None:
                continue
            drawn = int((m & mask).sum())
            if drawn:
                out.append((i, int(((self.owner == i) & mask).sum()), drawn))
        sprites = self.summary["sprites"]
        out.sort(key=lambda t: (-t[1], -(sprites[t[0]].get("visible") is True), -t[2], -(sprites[t[0]].get("layer", 0) if isinstance(sprites[t[0]].get("layer"), int) else 0)))
        return out


def sprite_text(i: int, e: dict[str, Any]) -> str:
    """One sprite of a state_summary: '#12 "player" tags=(player) layer=1 x=7 y=5 size=2x2 visible collidable'."""
    tags = ", ".join(e.get("tags") or [])
    parts = [f"#{i}", json.dumps(str(e.get("name", ""))), f"tags=({tags})", f"layer={e.get('layer')}"]
    parts += [f"x={e.get('x')}", f"y={e.get('y')}", f"size={e.get('w')}x{e.get('h')}"]
    parts.append("visible" if e.get("visible") is True else "HIDDEN (visible=False)")
    parts.append("collidable" if e.get("collidable") is True else "not collidable")
    if e.get("screen") is True:
        parts.append("screen")
    if e.get("blocking") not in ("pixel", None):
        parts.append(f"blocking={e.get('blocking')}")
    for name, default in (("rotation", 0), ("mirror_ud", False), ("mirror_lr", False), ("scale", 1)):
        if e.get(name, default) != default:
            parts.append(f"{name}={e.get(name)}")
    return " ".join(parts)


_TRACKED = ("x", "y", "layer", "collidable", "screen", "rotation", "mirror_ud", "mirror_lr", "scale", "name", "blocking")


def sprite_change(before: dict[str, Any], after: dict[str, Any]) -> str:
    """How one sprite differs between two summaries: 'x 6->7', 'hidden', 'pixels changed'."""
    changes = [f"{k} {before.get(k)}->{after.get(k)}" for k in _TRACKED if before.get(k) != after.get(k)]
    if before.get("visible") != after.get("visible"):
        changes.append("shown" if after.get("visible") is True else "hidden")
    if before.get("pixels_crc") != after.get("pixels_crc"):
        changes.append("pixels changed")
    if before.get("tags") != after.get("tags"):
        changes.append(f"tags ({', '.join(before.get('tags') or [])})->({', '.join(after.get('tags') or [])})")
    return ", ".join(changes)


def same_sprites(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[int, int]:
    """{index after: index before} for the sprite objects present in both summaries of one state
    (by their oid); empty when the step replaced the state (RESET, a new level)."""
    if not before or not after or "sprites" not in before or "sprites" not in after:
        return {}
    index = {e.get("oid"): j for j, e in enumerate(before["sprites"]) if e.get("oid") is not None}
    return {i: index[e["oid"]] for i, e in enumerate(after["sprites"]) if e.get("oid") in index}


def _name(e: dict[str, Any]) -> str:
    return json.dumps(str(e.get("name", "")))


def state_changes(before: dict[str, Any] | None, after: dict[str, Any] | None, limit: int = 14) -> list[str]:
    """What a step changed in the engine's own state, from summaries taken before and after it:
    sprites moved, changed, shown or hidden, added or removed (#index after the step; removed ones
    by their index before it), state.vars and the status."""
    if not after or "sprites" not in after:
        return ["your engine has no state after the step"]
    if not before or "sprites" not in before:
        return [f"your state: the start of level {after.get('level')} ({len(after['sprites'])} sprites), fresh from make_level"]
    pairs = same_sprites(before, after)
    if not pairs and before["sprites"] and after["sprites"]:
        if before.get("level") != after.get("level"):
            return [f"the level changed from {before.get('level')} to {after.get('level')}: your state is now a fresh copy of make_level({after.get('level')})"]
        return [f"your state was replaced by a fresh copy of make_level({after.get('level')}) (the level restarted)"]
    lines = []
    for i, e in enumerate(after["sprites"]):
        if i not in pairs:
            lines.append(f"added #{i} {_name(e)} at x={e.get('x')} y={e.get('y')}")
            continue
        j = pairs[i]
        change = sprite_change(before["sprites"][j], e)
        if change:
            lines.append(f"#{i} {_name(e)}: {change}" + (f" (it was #{j} before the step)" if i != j else ""))
    kept = set(pairs.values())
    lines += [f"removed: #{j} {_name(e)} (its index before the step)" for j, e in enumerate(before["sprites"]) if j not in kept]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more sprite changes"]
    if not lines:
        lines.append("no sprite changed")
    b, a = before.get("vars") or {}, after.get("vars") or {}
    changed = [f"{k} {b.get(k, '(none)')}->{a.get(k, '(none)')}" for k in dict.fromkeys([*b, *a]) if b.get(k) != a.get(k)]
    lines.append("vars: " + ("; ".join(changed[:8]) + ("; ..." if len(changed) > 8 else "") if changed else "unchanged"))
    if before.get("status") != after.get("status"):
        lines.append(f"status: {before.get('status')} -> {after.get('status')}")
    return lines


def region_lines(region: Region, expected: np.ndarray, got: np.ndarray | None, before: SummaryView, after: SummaryView, crops: bool) -> list[str]:
    """The text for one numbered region."""
    r0, c0, r1, c1 = region.core
    where = cells_text(after.summary if after else None, region.mask)
    lines = [
        f"    [{region.n}] rows {_range(r0, r1)}, cols {_range(c0, c1)}" + (f" ({where})" if where else "")
        + f": {region.pixels} px differ, expected->got {_changes_text(region.changes)}"
    ]
    if after:
        here = after.at(region.mask)
        sprites = after.summary["sprites"]
        pairs = same_sprites(before.summary if before else None, after.summary)
        if not here:
            shown = Counter(expected[region.mask].tolist()).most_common(2)
            lines.append(
                "        no sprite of yours draws here (the empty screen is colour 5); the original shows "
                + ", ".join(f"{c} ({COLOR_NAMES.get(c, '?')}) x{n}" for c, n in shown)
            )
        main = [t for t in here if t[1] or sprites[t[0]].get("visible") is not True]
        under = [t for t in here if not t[1] and sprites[t[0]].get("visible") is True]
        for k, (i, top, drawn) in enumerate(main[:MAX_SPRITES]):
            label = "yours here after the step:" if k == 0 else ""
            if sprites[i].get("visible") is not True:
                where_px = f"(hidden; would draw {drawn} of these px)"
            else:
                where_px = f"(shows at {top} of these px)"
            change = sprite_change(before.summary["sprites"][pairs[i]], sprites[i]) if i in pairs else ""
            lines.append(f"        {label:<27}{sprite_text(i, sprites[i])} {where_px}" + (f"; this step: {change}" if change else ""))
        if len(main) > MAX_SPRITES:
            lines.append(f"        {'':<27}... and {len(main) - MAX_SPRITES} more")
        if under:
            names = ", ".join(f"#{i} {_name(sprites[i])} layer={sprites[i].get('layer')}" for i, _, _ in under[:4])
            lines.append(f"        {'':<27}drawn under them: {names}" + (", ..." if len(under) > 4 else ""))
        if before:
            now = {pairs[i] for i, _, _ in here if i in pairs}
            after_of = {j: i for i, j in pairs.items()}
            gone = [(j, top) for j, top, _ in before.at(region.mask) if top and j not in now]
            for k, (j, top) in enumerate(gone[:2]):
                e = before.summary["sprites"][j]
                if j in after_of:
                    i = after_of[j]
                    change = sprite_change(e, sprites[i])
                    note = f"; this step: {change}" + (f" (now #{i})" if i != j else "") if change else ""
                elif pairs:
                    note = "; removed by the step"
                else:
                    note = ""
                label = "before the step, here:" if k == 0 else ""
                lines.append(f"        {label:<27}{sprite_text(j, e)}{note}")
    if crops and got is not None:
        lines += hex_crop(expected, got, region.box, indent="        ")
    return lines


def vars_line(before: SummaryView, after: SummaryView) -> str | None:
    if not after:
        return None
    a = after.summary.get("vars") or {}
    if not before or before.summary.get("level") != after.summary.get("level"):
        body = ", ".join(f"{k}={v}" for k, v in list(a.items())[:6])
        return f"    your state.vars after the step (level {after.summary.get('level')}): {body or '{}'}"
    b = before.summary.get("vars") or {}
    changed = [f"{k} {b.get(k, '(none)')}->{a.get(k, '(none)')}" for k in list(dict.fromkeys([*b, *a])) if b.get(k) != a.get(k)]
    if not changed:
        return "    your state.vars: unchanged by the step"
    return "    your state.vars changed: " + "; ".join(changed[:6]) + (f"; ... ({len(changed) - 6} more)" if len(changed) > 6 else "")


def describe_frames(
    expected: np.ndarray, got: np.ndarray | None, before: dict[str, Any] | None, after: dict[str, Any] | None, *, crops: bool, images: bool, show_vars: bool = True
) -> tuple[list[str], list[Region]]:
    """Text lines comparing the original's final frame with the engine's, and the numbered regions."""
    before_v, after_v = SummaryView(before), SummaryView(after)
    if got is None:
        return ["    final frame: your engine returned no frame"], []
    regions, hidden = find_regions(expected, got)
    if not regions:
        return ["    final frame: matches"], []
    total = int((np.asarray(expected) != np.asarray(got)).sum())
    where = " (boxed and numbered the same way in the images)" if images else ""
    lines = [f"    final frame: {total} px differ in {len(regions) + hidden} region(s){where}:"]
    for region in regions:
        lines += region_lines(region, np.asarray(expected), np.asarray(got), before_v, after_v, crops)
    if hidden:
        rest = total - sum(r.pixels for r in regions)
        lines.append(f"    ... and {hidden} smaller region(s) ({rest} px) not numbered")
    if after_v.error:
        lines.append(f"    (your sprites could not be listed: {after_v.error})")
    line = vars_line(before_v, after_v) if show_vars else None
    if line:
        lines.append(line)
    return lines, regions


# --- Images ------------------------------------------------------------------------------


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow without scalable default font
        return ImageFont.load_default()


def frame_image(frame: np.ndarray | None, scale: int = UPSCALE):
    """A 64x64 frame as an RGB image, upscaled with nearest neighbour (None: a grey 'no frame' panel)."""
    from PIL import Image, ImageDraw

    if frame is None:
        img = Image.new("RGB", (64 * scale, 64 * scale), (90, 90, 90))
        ImageDraw.Draw(img).text((16, 16), "no frame", fill=(255, 255, 255), font=_font(20))
        return img
    frame = np.asarray(frame)
    lut = np.array([PALETTE[i] for i in range(16)], np.uint8)
    rgb = lut[np.clip(frame.astype(np.int64), 0, 15)]
    return Image.fromarray(rgb, "RGB").resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)


def _draw_boxes(img, regions: list[Region], scale: int, dx: int, dy: int) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    font = _font(16)
    for region in regions:
        r0, c0, r1, c1 = region.box
        x0, y0, x1, y1 = dx + c0 * scale, dy + r0 * scale, dx + (c1 + 1) * scale - 1, dy + (r1 + 1) * scale - 1
        draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=EDGE_RGB, width=5)
        draw.rectangle((x0, y0, x1, y1), outline=BOX_RGB, width=3)
        label = str(region.n)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left + 6, bottom - top + 4
        # Above the box, or below it at the top edge; never on the differing pixels.
        ly = y0 - th - 4 if y0 - th - 4 >= dy else y1 + 4
        if ly + th > dy + 64 * scale:
            ly = y0 + 4
        lx = min(max(x0, dx), dx + 64 * scale - tw - 1)
        draw.rectangle((lx, ly, lx + tw, ly + th), fill=BOX_RGB, outline=EDGE_RGB)
        draw.text((lx + 3 - left, ly + 2 - top), label, fill=EDGE_RGB, font=font)


def comparison_image(got: np.ndarray | None, expected: np.ndarray, regions: list[Region], *, left_title: str, right_title: str, scale: int = UPSCALE):
    """Your frame (left) and the original's (right), side by side with titles, the differing
    regions boxed and numbered on both."""
    from PIL import Image, ImageDraw

    pad, head = 12, 30
    side = 64 * scale
    img = Image.new("RGB", (2 * side + 3 * pad, head + side + pad), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    font = _font(18)
    for k, (frame, title) in enumerate(((got, left_title), (expected, right_title))):
        dx = pad + k * (side + pad)
        img.paste(frame_image(frame, scale), (dx, head))
        draw.text((dx, 6), title, fill=(255, 255, 255), font=font)
        _draw_boxes(img, regions, scale, dx, head)
    return img


def png_bytes(img) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
