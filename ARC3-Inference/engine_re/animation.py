"""The animation digest of a recorded step: what its frames show that its before and after frames do not.

Ported from the base harness (inference/utils/animation.py, and the ANIMATION_ADDENDUM and
ANIMATION_ADDENDUM_TIMELINE texts of inference/agent/prompts.py), adapted to our frames (numpy arrays of
colours 0-15, indexed frame[y, x]) and coordinates (x the column, y the row):

- transient cells: cells that changed and then changed BACK during the animation, so they appear in no frame
  the model can otherwise reach: equal in the step's before and after frames, different in at least one
  frame between. Their count, bounding box, the colour transitions seen (old>new, with counts) and the
  frames in which they differ;
- the timeline: one entry per frame that changed anything relative to the frame before it (frame 0 relative
  to the step's before frame): the frame index, how many cells changed, their bounding box, and either the
  cells (old>new @ (x,y) ...) when there are few, or a count per colour transition when there are many.

``digest(before, frames)`` makes it (None for a step of one frame); StepView.animation (engine_re.helpers)
is the model's handle on it; ``report_lines`` is the one or two lines a test report or a message prints for
an animated step, capped so it never floods the prompt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_CELLS_LISTED = 8  # a timeline entry lists its cells when it has at most this many; else counts per transition
TIMELINE_CHARS = 260  # the timeline is printed in a report when it fits in this many characters, else a pointer


def _span(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


def bbox_text(bbox: tuple[int, int, int, int] | None) -> str:
    """'in x 0-63, y 10' for an inclusive (x0, y0, x1, y1) box; 'at (3, 4)' for one cell."""
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    if x0 == x1 and y0 == y1:
        return f"at ({x0}, {y0})"
    return f"in x {_span(x0, x1)}, y {_span(y0, y1)}"


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{key} x{n}" for key, n in counts.items())


@dataclass
class TimelineEntry:
    """One frame of an animation that changed something relative to the frame before it."""

    frame: int  # its index in the step's .frames
    changed: int  # cells that differ from the frame before (frame 0: from the step's .before)
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), inclusive
    cells: list[str] | None = None  # "old>new @ (x,y) (x,y) ...", one string per colour transition, when few
    transitions: dict[str, int] | None = None  # {"old>new": count}, when too many cells to list

    def __str__(self) -> str:
        what = "; ".join(self.cells) if self.cells is not None else _counts_text(self.transitions or {})
        return f"frame {self.frame}: {self.changed} cell{'s' if self.changed != 1 else ''} {bbox_text(self.bbox)}: {what}"

    __repr__ = __str__


@dataclass
class Animation:
    """The digest of an animated step (StepView.animation)."""

    frames: int  # frames the action returned
    transient: int  # cells that changed and changed back
    transient_bbox: tuple[int, int, int, int] | None  # their (x0, y0, x1, y1), inclusive
    transient_transitions: dict[str, int] = field(default_factory=dict)  # {"old>new": cells}, the colours they took
    transient_frames: tuple[int, int] | None = None  # the first and last frame in which they differ
    timeline: list[TimelineEntry] = field(default_factory=list)

    def transient_text(self) -> str:
        if not self.transient:
            return "no cell changed and changed back (every cell that changed kept its new colour)"
        cells = "1 cell changed and changed back" if self.transient == 1 else f"{self.transient} cells changed and changed back"
        lo, hi = self.transient_frames or (0, 0)
        return (f"{cells} {bbox_text(self.transient_bbox)}, in frame{'s' if lo != hi else ''} {_span(lo, hi)}: "
                f"{_counts_text(self.transient_transitions)}; they are in no frame you can otherwise reach (not in .before, "
                "not in .after)")

    def timeline_text(self) -> str:
        return "; ".join(str(e) for e in self.timeline) if self.timeline else "no frame changed anything"

    def __str__(self) -> str:
        return (f"Animation over {self.frames} frames. Transient: {self.transient_text()}.\nTimeline (each frame against the one "
                f"before it, frame 0 against .before): {self.timeline_text()}.")

    __repr__ = __str__


def digest(before: Any, frames: Any, max_cells: int = MAX_CELLS_LISTED) -> Animation | None:
    """The animation digest of a step whose action returned `frames` from the frame `before` (None for the
    game's first step: frame 0 is then the reference); None when it returned at most one frame."""
    stack = np.asarray(frames) if frames is not None else np.zeros((0, 64, 64), np.int8)
    if stack.ndim != 3 or len(stack) <= 1:
        return None
    stack = stack.astype(np.int16)
    base = stack[0] if before is None else np.asarray(before).astype(np.int16)
    after = stack[-1]
    timeline: list[TimelineEntry] = []
    for i in range(len(stack)):
        previous = base if i == 0 else stack[i - 1]
        if i == 0 and before is None:
            continue
        mask = previous != stack[i]
        n = int(mask.sum())
        if not n:
            continue
        box = _bbox(mask)
        assert box is not None
        ys, xs = np.nonzero(mask)
        pairs = [(int(previous[y, x]), int(stack[i][y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
        if n <= max_cells:
            grouped: dict[str, list[str]] = {}
            for old, new, x, y in pairs:
                grouped.setdefault(f"{old}>{new}", []).append(f"({x},{y})")
            timeline.append(TimelineEntry(i, n, box, cells=[f"{key} @ {' '.join(at)}" for key, at in grouped.items()]))
        else:
            counts = Counter(f"{old}>{new}" for old, new, _, _ in pairs)
            timeline.append(TimelineEntry(i, n, box, transitions=dict(counts.most_common())))
    differs = stack[:-1] != base  # (n-1, 64, 64): each frame before the last against the before frame
    mask = (base == after) & differs.any(axis=0)
    transient = int(mask.sum())
    transitions: Counter = Counter()
    seen_frames: tuple[int, int] | None = None
    if transient:
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            old = int(base[y, x])
            for new in sorted({int(v) for v in stack[:-1, y, x] if int(v) != old}):
                transitions[f"{old}>{new}"] += 1
        hit = [i for i in range(len(stack) - 1) if (differs[i] & mask).any()]
        seen_frames = (hit[0], hit[-1])
    return Animation(
        frames=int(len(stack)), transient=transient, transient_bbox=_bbox(mask),
        transient_transitions=dict(transitions.most_common()), transient_frames=seen_frames, timeline=timeline,
    )


def report_lines(before: Any, frames: Any, k: int | None = None, indent: str = "    ") -> list[str]:
    """One or two lines for a test report or a message on an animated step: the transient cells always, the
    timeline when it fits in TIMELINE_CHARS characters, else where to read it. [] for a step of one frame."""
    found = digest(before, frames)
    if found is None:
        return []
    lines = [f"{indent}Animation: transient cells: {found.transient_text()}."]
    timeline = found.timeline_text()
    if len(timeline) <= TIMELINE_CHARS:
        lines.append(f"{indent}Animation timeline (each frame against the one before it, frame 0 against .before): {timeline}.")
    else:
        where = f"recording[{k}]" if k is not None else "the step's StepView"
        lines.append(f"{indent}Animation timeline: {len(found.timeline)} frames changed something; {where}.animation.timeline "
                     "lists them.")
    return lines
