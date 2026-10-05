"""Line anchors and anchored edits for engine.py, ported from pi-hashline-edit.

``read`` shows every line as ``LINE#HASH:content``; ``edit`` addresses lines by those anchors. A
HASH is two characters of the alphabet ZPMQVRWSNKTXJBYH, computed over the line and its two
neighbours (prev + "\\0" + line + "\\0" + next, each with "\\r" removed and trailing whitespace
stripped), so an anchor goes stale when its line or a neighbour changes, and only then.

pi uses xxh32. This uses it too when the `xxhash` package is installed, else zlib.crc32 (it is not
installed in this repo's environment; crc32 is stable across runs and platforms). Either way the
kernel and the harness share one environment, and anchors are only ever compared with anchors
from this module.

Edit ops (all validated against the same snapshot, then applied bottom-up):

- {"op": "replace", "pos": A, "end": B, "lines": [...]}: replace line A, or lines A..B; [] deletes;
- {"op": "append", "pos": A, "lines": [...]}: insert after A (no pos: at the end);
- {"op": "prepend", "pos": A, "lines": [...]}: insert before A (no pos: at the start);
- {"op": "replace_text", "oldText": ..., "newText": ...}: replace one exact, unique text.

`lines` is a list of strings or one string (split on newlines, one trailing newline ignored), so
generated code goes straight in. An anchor may carry its ":content" suffix, which is cross-checked
against the file (a stale hash whose content still matches after whitespace normalisation is
accepted). Edits that overlap or touch adjacent lines are rejected, as is any edit inside a
protected line range (the FIXED block).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

try:  # xxh32 as in pi-hashline-edit when available; crc32 otherwise (see the module docstring)
    from xxhash import xxh32_intdigest as _digest
except ImportError:
    _digest = zlib.crc32

NIBBLES = "ZPMQVRWSNKTXJBYH"
_ANCHOR_RE = re.compile(r"^([0-9]+)\s*#\s*([^\s:]+)(?:\s*:(.*))?$", re.S)
_DISPLAY_PREFIX_RE = re.compile(rf"^\s*\+?\s*(?:\d+\s*#\s*|#\s*)[{NIBBLES}]{{2,4}}:")
_SIGNIFICANT_RE = re.compile(r"\w")
_KEYS = {"op", "pos", "end", "lines", "oldText", "newText"}
ANCHOR_LINES = 12  # fresh anchors shown per changed region


class EditError(Exception):
    """An edit request that cannot be applied; the message says why (with an [E_...] code)."""


# --- Hashes and display -------------------------------------------------------------------------


def _norm(line: str) -> str:
    return line.replace("\r", "").rstrip()


def line_hash(lines: list[str], index: int) -> str:
    """The 2-character hash of lines[index] (0-based) in its context."""
    prev = _norm(lines[index - 1]) if index > 0 else ""
    nxt = _norm(lines[index + 1]) if index + 1 < len(lines) else ""
    h = _digest((prev + "\0" + _norm(lines[index]) + "\0" + nxt).encode("utf-8"))
    return NIBBLES[(h >> 4) & 15] + NIBBLES[h & 15]


def split_lines(text: str) -> tuple[list[str], bool]:
    """The visible lines of a text and whether it ends with a newline."""
    text = text.replace("\r\n", "\n")
    if not text:
        return [], False
    ends = text.endswith("\n")
    lines = text.split("\n")
    if ends:
        lines.pop()
    return lines, ends


def join_lines(lines: list[str], newline_at_end: bool) -> str:
    return "\n".join(lines) + ("\n" if newline_at_end and lines else "")


def anchor(lines: list[str], line: int) -> str:
    """'LINE#HASH' for a 1-based line."""
    return f"{line}#{line_hash(lines, line - 1)}"


def format_lines(lines: list[str], start: int, end: int, width: int | None = None) -> list[str]:
    """Lines start..end (1-based, inclusive) as 'LINE#HASH:content'."""
    width = width or len(str(end))
    return [f"{n:>{width}}#{line_hash(lines, n - 1)}:{lines[n - 1]}" for n in range(start, end + 1)]


def render_read(
    text: str,
    offset: int | None = None,
    limit: int | None = None,
    max_chars: int = 6000,
    fold: tuple[int, int] | None = None,
    name: str = "engine.py",
) -> str:
    """What read() prints: the lines with anchors, from `offset` (1-based) for `limit` lines, cut to
    about `max_chars` characters with the offset to continue from. With `fold` (first, last line of
    the FIXED block) and no offset, the block's inner lines are folded into one note."""
    lines, _ = split_lines(text)
    total = len(lines)
    if not total:
        return f"{name} is empty. Use edit with an append or prepend op without pos to add content."
    for value, label in ((offset, "offset"), (limit, "limit")):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise EditError(f'[E_BAD_ARG] read: "{label}" must be a positive integer.')
    start = offset or 1
    if start > total:
        return f"Offset {start} is beyond the end of {name} ({total} lines). Use offset=1 to read from the start."
    end = min(total, start - 1 + limit) if limit else total
    width = len(str(end))
    out: list[str] = []
    size = 0
    n = start
    folded = fold if offset is None and fold and fold[1] - fold[0] > 1 else None
    while n <= end:
        if folded and n == folded[0] + 1:
            note = (f"{'':>{width}}  [lines {folded[0] + 1}-{folded[1] - 1}: the FIXED block, folded; it cannot be edited. "
                    f"read(offset={folded[0] + 1}, limit={folded[1] - folded[0] - 1}) shows it]")
            out.append(note)
            size += len(note) + 1
            n = folded[1]
            continue
        line = format_lines(lines, n, n, width)[0]
        if out and size + len(line) + 1 > max_chars:
            out.append(f"\n[Showing lines {start}-{n - 1} of {total}. Use offset={n} to continue.]")
            return "\n".join(out)
        out.append(line)
        size += len(line) + 1
        n += 1
    if end < total:
        out.append(f"\n[Showing lines {start}-{end} of {total}. Use offset={end + 1} to continue.]")
    return "\n".join(out)


# --- Parsing ----------------------------------------------------------------------------------


@dataclass
class Anchor:
    line: int
    hash: str
    hint: str | None = None

    def __str__(self) -> str:
        return f"{self.line}#{self.hash}"


def parse_anchor(ref: object) -> Anchor:
    if not isinstance(ref, str):
        raise EditError(f'[E_BAD_REF] An anchor must be a string like "12#MQ", got {ref!r}.')
    core = re.sub(r"^\s*[>+-]*\s*", "", ref).rstrip()
    match = _ANCHOR_RE.match(core)
    if not match:
        if re.fullmatch(r"\d+", core):
            raise EditError(f'[E_BAD_REF] "{ref}" has no hash: use "LINE#HASH" from read() or edit() output (e.g. "12#MQ").')
        raise EditError(f'[E_BAD_REF] Invalid line reference "{ref}". Expected "LINE#HASH" (e.g. "12#MQ").')
    line, h = int(match.group(1)), match.group(2)
    if line < 1:
        raise EditError(f'[E_BAD_REF] Line numbers start at 1, got "{ref}".')
    if len(h) != 2 or any(c not in NIBBLES for c in h):
        raise EditError(f'[E_BAD_REF] "{ref}": a hash is 2 characters from {NIBBLES}. Copy anchors from read() or edit() output.')
    return Anchor(line, h, match.group(3))


def _as_lines(value: object, index: int) -> list[str]:
    """`lines` given as a list of strings or as one string (split on newlines)."""
    if isinstance(value, str):
        text = value.replace("\r\n", "\n")
        if text.endswith("\n"):
            text = text[:-1]
        out = text.split("\n") if text else []
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        out = [part for v in value for part in v.replace("\r\n", "\n").split("\n")]
    else:
        raise EditError(f'[E_BAD_OP] Edit {index}: "lines" must be a list of strings or one string.')
    for line in out:
        if _DISPLAY_PREFIX_RE.match(line):
            raise EditError(
                f'[E_INVALID_PATCH] Edit {index}: "lines" must be the literal new content, not read() output with '
                f'"LINE#HASH:" prefixes. Offending line: {line!r}'
            )
    return out


def _fuzzy(text: str) -> str:
    text = text.rstrip()
    for chars, plain in (("‘’‚‛", "'"), ("“”„‟", '"'), ("‐‑‒–—―−", "-")):
        for c in chars:
            text = text.replace(c, plain)
    return re.sub(r"[  -   　]", " ", text)


def _hint_matches(hint: str, line: str) -> bool:
    """A ':content' hint matches a line: equal after normalisation, or, cut with '...', its start."""
    h = _fuzzy(hint).strip()
    cut = re.search(r"\.{3}|…", h)
    if cut is None:
        return h == _fuzzy(line).strip()
    return _fuzzy(line).strip().startswith(h[: cut.start()])


# --- Applying -------------------------------------------------------------------------------


@dataclass
class _Op:
    index: int
    kind: str  # "replace" (lines start..end replaced) or "insert" (after line `start`, 0 = the top)
    start: int
    end: int
    lines: list[str]
    label: str


@dataclass
class EditResult:
    text: str
    summary: list[str]  # one entry per applied op, top to bottom
    regions: list[tuple[int, int]]  # (first, last) changed line in the new text, per op; last < first: deletion
    warnings: list[str] = field(default_factory=list)
    noop: list[str] = field(default_factory=list)


def _span_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """1-based first and last line covered by text[start:end]."""
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    return first, last


def apply_edits(text: str, edits: object, protected: tuple[int, int] | None = None) -> EditResult:
    """Apply anchored edits to `text`; raises EditError with an [E_...] message when they cannot be
    applied. `protected` (first, last line, 1-based) is a range no edit may touch."""
    if not isinstance(edits, (list, tuple)) or not edits:
        raise EditError('[E_BAD_OP] edits must be a non-empty list of edit dicts, e.g. [{"op": "replace", "pos": "12#MQ", "lines": [...]}].')
    lines, newline_at_end = split_lines(text)
    total = len(lines)
    stale: list[Anchor] = []
    warnings: list[str] = []

    def check(a: Anchor) -> bool:
        if a.line > total:
            raise EditError(f"[E_RANGE_OOB] Line {a.line} does not exist (the file has {total} lines).")
        actual = line_hash(lines, a.line - 1)
        if actual == a.hash:
            if a.hint is not None and not _hint_matches(a.hint, lines[a.line - 1]):
                stale.append(a)
                return False
            return True
        if a.hint is not None and _fuzzy(a.hint) == _fuzzy(lines[a.line - 1]):
            # The hash is stale but the copied content still matches (whitespace or quote drift).
            prev = lines[a.line - 2] if a.line > 1 else ""
            nxt = lines[a.line] if a.line < total else ""
            if line_hash([prev, a.hint, nxt], 1) == a.hash:
                warnings.append(f"Accepted the anchor for line {a.line}: its hash was stale but its content matched.")
                return True
        stale.append(a)
        return False

    ops: list[_Op] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise EditError(f"[E_BAD_OP] Edit {index} must be a dict with an \"op\".")
        unknown = set(edit) - _KEYS
        if unknown:
            raise EditError(f"[E_BAD_OP] Edit {index} has unknown fields: {', '.join(sorted(unknown))}.")
        op = edit.get("op")
        if op == "replace_text":
            old, new = edit.get("oldText"), edit.get("newText")
            if not isinstance(old, str) or not isinstance(new, str) or set(edit) - {"op", "oldText", "newText"}:
                raise EditError(f'[E_BAD_OP] Edit {index}: replace_text takes only string "oldText" and "newText".')
            old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
            if not old:
                raise EditError(f"[E_BAD_OP] Edit {index}: replace_text needs a non-empty oldText.")
            body = join_lines(lines, newline_at_end)
            found = [m.start() for m in re.finditer(re.escape(old), body)]
            if not found:
                raise EditError(f"[E_NO_MATCH] Edit {index}: replace_text found no exact match. read() and use anchors.")
            if len(found) > 1:
                raise EditError(f"[E_MULTI_MATCH] Edit {index}: replace_text found {len(found)} matches. read() and use anchors.")
            first, last = _span_lines(body, found[0], found[0] + len(old))
            segment_start = sum(len(line) + 1 for line in lines[: first - 1])
            segment = "\n".join(lines[first - 1 : last])
            offset = found[0] - segment_start
            replaced = segment[:offset] + new + segment[offset + len(old) :]
            preview = old.replace("\n", "\\n")
            preview = preview if len(preview) <= 30 else preview[:27] + "..."
            label = f'replace_text "{preview}"'
            before, after = segment.split("\n"), replaced.split("\n")
            # Only the lines that really change count (for the FIXED block and conflict checks):
            # an oldText that starts with the newline ending an unchanged line does not touch it.
            while len(before) > 1 and len(after) > 1 and before[0] == after[0]:
                before, after, first = before[1:], after[1:], first + 1
            while len(before) > 1 and len(after) > 1 and before[-1] == after[-1]:
                before, after, last = before[:-1], after[:-1], last - 1
            if before == [after[0]] and len(after) > 1:
                ops.append(_Op(index, "insert", first, first, after[1:], label))
            elif before == [after[-1]] and len(after) > 1:
                ops.append(_Op(index, "insert", first - 1, first - 1, after[:-1], label))
            else:
                ops.append(_Op(index, "replace", first, last, after, label))
            continue
        if op not in ("replace", "append", "prepend"):
            raise EditError(f'[E_BAD_OP] Edit {index} has op {op!r}; use "replace", "append", "prepend" or "replace_text".')
        if "oldText" in edit or "newText" in edit:
            raise EditError(f'[E_BAD_OP] Edit {index}: "oldText"/"newText" belong to replace_text only.')
        if "lines" not in edit:
            raise EditError(f'[E_BAD_OP] Edit {index} needs "lines" (the new content; [] deletes).')
        new_lines = _as_lines(edit["lines"], index)
        if op == "replace":
            if "pos" not in edit:
                raise EditError(f'[E_BAD_OP] Edit {index}: replace needs a "pos" anchor.')
            pos = parse_anchor(edit["pos"])
            end = parse_anchor(edit["end"]) if edit.get("end") is not None else None
            if end is not None and end.line < pos.line:
                raise EditError(f"[E_BAD_OP] Edit {index}: end line {end.line} comes before pos line {pos.line}.")
            ok = check(pos)
            ok = (check(end) if end is not None else True) and ok
            if not ok:
                continue
            last = end.line if end is not None else pos.line
            if end is None and len(new_lines) > 1:
                warnings.append(f"Edit {index} replaced only line {pos.line} with {len(new_lines)} lines (no end anchor); add end to replace a range.")
            label = f"replace {pos}" + (f"-{end}" if end is not None else "")
            ops.append(_Op(index, "replace", pos.line, last, new_lines, label))
        else:
            if edit.get("end") is not None:
                raise EditError(f'[E_BAD_OP] Edit {index}: {op} takes "pos" only, not "end".')
            if not new_lines:
                raise EditError(f"[E_BAD_OP] Edit {index}: {op} with no lines to insert.")
            pos = parse_anchor(edit["pos"]) if edit.get("pos") is not None else None
            if pos is not None and not check(pos):
                continue
            if op == "append":
                boundary = pos.line if pos is not None else total
                label = f"append after {pos}" if pos is not None else "append at the end"
            else:
                boundary = pos.line - 1 if pos is not None else 0
                label = f"prepend before {pos}" if pos is not None else "prepend at the start"
            ops.append(_Op(index, "insert", boundary, boundary, new_lines, label))

    if stale:
        refs = ", ".join(str(a) for a in stale)
        message = [f"[E_STALE_ANCHOR] {len(stale)} stale anchor{'s' if len(stale) > 1 else ''}: {refs}. engine.py changed since "
                   "you read those lines: read() again and use the new anchors (nothing was applied)."]
        hinted = [a for a in stale if a.hint and re.split(r"\.{3}|…", _fuzzy(a.hint).strip())[0]]
        candidates = []
        for a in hinted:
            matches = [n for n in range(1, total + 1) if _hint_matches(a.hint, lines[n - 1])][:3]
            candidates += [f"  {anchor(lines, n)}:{lines[n - 1]}   <- for stale {a}" for n in matches]
        if candidates:
            message += ["Lines with the content you gave:"] + candidates[:8]
        raise EditError("\n".join(message))

    if protected:
        first, last = protected
        for o in ops:
            inside = (o.start <= last and o.end >= first) if o.kind == "replace" else first <= o.start < last
            if inside:
                raise EditError(
                    f"[E_FIXED_BLOCK] Edit {o.index} ({o.label}) touches the FIXED block (lines {first}-{last}), which must not "
                    "change. Write your code below it."
                )

    noop = []
    kept: list[_Op] = []
    seen = set()
    for o in ops:
        if o.kind == "replace" and o.lines == lines[o.start - 1 : o.end]:
            noop.append(f"edit {o.index} ({o.label}) is identical to the current content")
            continue
        key = (o.kind, o.start, o.end, tuple(o.lines))
        if key not in seen:
            seen.add(key)
            kept.append(o)
    for i, a in enumerate(kept):
        for b in kept[i + 1 :]:
            if a.kind == b.kind == "insert":
                clash = a.start == b.start
            elif a.kind == b.kind == "replace":
                clash = b.start <= a.end + 1 and a.start <= b.end + 1
            else:
                r, ins = (a, b) if a.kind == "replace" else (b, a)
                clash = r.start - 1 <= ins.start <= r.end
            if clash:
                raise EditError(
                    f"[E_EDIT_CONFLICT] Edits {a.index} ({a.label}) and {b.index} ({b.label}) overlap or touch adjacent lines. "
                    "Merge them into one edit (nothing was applied)."
                )
    if not kept:
        return EditResult(text, [], [], warnings, noop or ["the edits change nothing"])

    for o in kept:  # warnings about likely mistakes, as pi gives them
        if o.kind == "replace" and o.lines:
            after = lines[o.end] if o.end < total else None
            before = lines[o.start - 2] if o.start > 1 else None
            if after is not None and _SIGNIFICANT_RE.search(o.lines[-1]) and o.lines[-1].strip() == after.strip():
                warnings.append(f"Edit {o.index} ({o.label}) ends with a copy of the next line, which stays: is it now there twice?")
            if before is not None and _SIGNIFICANT_RE.search(o.lines[0]) and o.lines[0].strip() == before.strip():
                warnings.append(f"Edit {o.index} ({o.label}) starts with a copy of the line before it, which stays: is it now there twice?")
        if o.kind == "insert":
            near = lines[o.start : o.start + len(o.lines)] if o.label.startswith("append") else lines[max(0, o.start - len(o.lines)) : o.start]
            if len(near) == len(o.lines) and any(_SIGNIFICANT_RE.search(x) for x in o.lines) and [x.strip() for x in near] == [x.strip() for x in o.lines]:
                warnings.append(f"Edit {o.index} ({o.label}) inserts lines identical to the ones already next to it: was it applied before?")

    new_lines_all = list(lines)
    for o in sorted(kept, key=lambda o: (o.start, 0 if o.kind == "insert" else 1), reverse=True):
        if o.kind == "replace":
            new_lines_all[o.start - 1 : o.end] = o.lines
        else:
            new_lines_all[o.start : o.start] = o.lines
    if lines and not new_lines_all:
        raise EditError("[E_WOULD_EMPTY] Refusing to empty the file.")
    summary, regions = [], []
    delta = 0
    for o in sorted(kept, key=lambda o: (o.start, 0 if o.kind == "insert" else 1)):
        n_new = len(o.lines)
        if o.kind == "replace":
            n_old = o.end - o.start + 1
            where = f"line {o.start}" if o.start == o.end else f"lines {o.start}-{o.end}"
            summary.append(f"deleted {where}" if not n_new else f"replaced {where} with {n_new} line{'s' if n_new > 1 else ''}")
            first = o.start + delta
            regions.append((first, first + n_new - 1))
            delta += n_new - n_old
        else:
            where = f"after line {o.start}" if o.start else "at the start"
            summary.append(f"inserted {n_new} line{'s' if n_new > 1 else ''} {where}")
            first = o.start + 1 + delta
            regions.append((first, first + n_new - 1))
            delta += n_new
    return EditResult(join_lines(new_lines_all, newline_at_end or not lines), summary, regions, warnings, noop)


def fresh_anchors(text: str, regions: list[tuple[int, int]], max_lines: int = ANCHOR_LINES) -> list[str]:
    """Anchors around each changed region of the new text: the region with one line of context if
    it fits in `max_lines`, else its first and last lines with context."""
    lines, _ = split_lines(text)
    total = len(lines)
    if not total:
        return []
    out: list[str] = []
    for first, last in regions:
        if last < first:  # a deletion: the lines around the gap
            a, b = max(1, first - 1), min(total, first)
            out += format_lines(lines, a, b) + ["   ..."]
            continue
        a, b = max(1, first - 1), min(total, last + 1)
        if b - a + 1 <= max_lines:
            out += format_lines(lines, a, b)
        else:
            out += format_lines(lines, a, min(total, first + 1))
            out.append(f"   ... ({last - first - 3} more new lines) ...")
            out += format_lines(lines, max(1, last - 1), b)
        out.append("   ...")
    return out[:-1] if out and out[-1] == "   ..." else out
