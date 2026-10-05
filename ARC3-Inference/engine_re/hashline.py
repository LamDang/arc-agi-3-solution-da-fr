"""Line anchors and anchored edits for engine.py, ported from pi-hashline-edit and made lenient.

``read`` shows every line as ``LINE#HASH:content``; ``edit`` addresses lines by those anchors. A
HASH is three characters of the alphabet ZPMQVRWSNKTXJBYH, computed over the line and its two
neighbours (prev + "\\0" + line + "\\0" + next, each with "\\r" removed and trailing whitespace
stripped), so an anchor goes stale when its line or a neighbour changes, and only then. Anchors with
2 to 4 hash characters are parsed; a shorter hash is compared with the end of the line's.

pi uses xxh32. This uses it too when the `xxhash` package is installed, else zlib.crc32 (it is not
installed in this repo's environment; crc32 is stable across runs and platforms). Either way the
kernel and the harness share one environment, and anchors are only ever compared with anchors
from this module.

Edit ops (all validated against the same snapshot; the valid ones applied bottom-up):

- {"op": "replace", "pos": A, "end": B, "lines": [...]}: replace line A, or lines A..B; [] deletes;
- {"op": "append", "pos": A, "lines": [...]}: insert after A (no pos: at the end);
- {"op": "prepend", "pos": A, "lines": [...]}: insert before A (no pos: at the start);
- {"op": "replace_text", "oldText": ..., "newText": ...}: replace one unique text: an exact match
  first, else whole lines that match ignoring whitespace (then newText is used as given);
- {"op": "replace_def", "name": "step", "lines": [...]}: replace the whole top-level def, class or
  NAME = ... assignment called name (decorators included; "Game.step" for a method of a top-level
  class). A name that is not defined is added at the end of the file (of the class, for a method).

`lines` is a list of strings or one string (split on newlines, one trailing newline ignored), so
generated code goes straight in. An anchor may carry its ":content" suffix: an anchor whose hash is
stale is still accepted when that content matches the line now there. Edits that touch adjacent
lines are merged (applied in order); two edits that change the same line are refused, both of them.
An edit that fails (a stale anchor, a line that does not exist, no match, the FIXED block, a
conflict) is reported on its own with the lines as they are now, and the other edits are applied;
only a malformed request (E_BAD_OP, E_BAD_REF, E_INVALID_PATCH: not a list of edit dicts, an unknown
op or field, a bad anchor string, read_file() output as lines) applies nothing.
"""

from __future__ import annotations

import ast
import difflib
import re
import zlib
from dataclasses import dataclass, field

try:  # xxh32 as in pi-hashline-edit when available; crc32 otherwise (see the module docstring)
    from xxhash import xxh32_intdigest as _digest
except ImportError:
    _digest = zlib.crc32

NIBBLES = "ZPMQVRWSNKTXJBYH"
HASH_CHARS = 3
_ANCHOR_RE = re.compile(r"^([0-9]+)\s*#\s*([^\s:]+)(?:\s*:(.*))?$", re.S)
_DISPLAY_PREFIX_RE = re.compile(rf"^\s*\+?\s*(?:\d+\s*#\s*|#\s*)[{NIBBLES}]{{2,4}}:")
_SIGNIFICANT_RE = re.compile(r"\w")
_KEYS = {"op", "pos", "end", "lines", "oldText", "newText", "name"}
_OPS = ("replace", "append", "prepend", "replace_text", "replace_def")
ANCHOR_LINES = 60  # a changed region up to this long is shown whole in the fresh anchors
CANDIDATES = 6  # lines suggested when a text or a stale anchor's content is not found
_CLOSE_RATIO = 0.6


class EditError(Exception):
    """An edit request that cannot be applied; the message says why (with an [E_...] code)."""


# --- Hashes and display -------------------------------------------------------------------------


def _norm(line: str) -> str:
    return line.replace("\r", "").rstrip()


def line_hash(lines: list[str], index: int, width: int = HASH_CHARS) -> str:
    """The hash of lines[index] (0-based) in its context: `width` characters (the last `width` of
    the full one, so a 2-character anchor is the end of the 3-character one)."""
    prev = _norm(lines[index - 1]) if index > 0 else ""
    nxt = _norm(lines[index + 1]) if index + 1 < len(lines) else ""
    h = _digest((prev + "\0" + _norm(lines[index]) + "\0" + nxt).encode("utf-8"))
    return "".join(NIBBLES[(h >> (4 * k)) & 15] for k in reversed(range(width)))


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
    """Lines start..end (1-based, inclusive, clipped to the file) as 'LINE#HASH:content'."""
    start, end = max(1, start), min(len(lines), end)
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
    """What read_file() prints: the lines with anchors, from `offset` (1-based) for `limit` lines, cut to
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
                    f"read_file(offset={folded[0] + 1}, limit={folded[1] - folded[0] - 1}) shows it]")
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
        raise EditError(f'[E_BAD_REF] An anchor must be a string like "12#MQV", got {ref!r}.')
    core = re.sub(r"^\s*[>+-]*\s*", "", ref).rstrip()
    match = _ANCHOR_RE.match(core)
    if not match:
        if re.fullmatch(r"\d+", core):
            raise EditError(f'[E_BAD_REF] "{ref}" has no hash: use "LINE#HASH" from read_file() or edit_file() output (e.g. "12#MQV").')
        raise EditError(f'[E_BAD_REF] Invalid line reference "{ref}". Expected "LINE#HASH" (e.g. "12#MQV").')
    line, h = int(match.group(1)), match.group(2)
    if line < 1:
        raise EditError(f'[E_BAD_REF] Line numbers start at 1, got "{ref}".')
    if not 2 <= len(h) <= 4 or any(c not in NIBBLES for c in h):
        raise EditError(f'[E_BAD_REF] "{ref}": a hash is {HASH_CHARS} characters from {NIBBLES}. Copy anchors from read_file() or edit_file() output.')
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
                f'[E_INVALID_PATCH] Edit {index}: "lines" must be the literal new content, not read_file() output with '
                f'"LINE#HASH:" prefixes. Offending line: {line!r}'
            )
    return out


def _fuzzy(text: str) -> str:
    text = text.rstrip()
    for chars, plain in (("‘’‚‛", "'"), ("“”„‟", '"'), ("‐‑‒–—―−", "-")):
        for c in chars:
            text = text.replace(c, plain)
    return re.sub(r"[  -   　]", " ", text)


def _ws(line: str) -> str:
    """A line with its indentation and trailing whitespace dropped and inner whitespace collapsed."""
    return " ".join(_fuzzy(line).split())


def _hint_matches(hint: str, line: str) -> bool:
    """A ':content' hint matches a line: equal after normalisation, or, cut with '...', its start."""
    h = _fuzzy(hint).strip()
    cut = re.search(r"\.{3}|…", h)
    if cut is None:
        return h == _fuzzy(line).strip()
    return _fuzzy(line).strip().startswith(h[: cut.start()])


def _close_lines(lines: list[str], probe: str, limit: int = CANDIDATES) -> list[int]:
    """1-based numbers of the lines most like `probe` (whitespace ignored), best first."""
    target = _ws(probe)
    if not target:
        return []
    scored: list[tuple[float, int]] = []
    for n, line in enumerate(lines, 1):
        cand = _ws(line)
        if not cand:
            continue
        if cand == target:
            scored.append((-1.0, n))
            continue
        matcher = difflib.SequenceMatcher(None, target, cand)
        if matcher.real_quick_ratio() >= _CLOSE_RATIO and matcher.quick_ratio() >= _CLOSE_RATIO:
            ratio = matcher.ratio()
            if ratio >= _CLOSE_RATIO:
                scored.append((-ratio, n))
    scored.sort()
    return [n for _, n in scored[:limit]]


# --- Definitions, for replace_def --------------------------------------------------------------

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _binds(node: ast.AST, name: str) -> bool:
    if isinstance(node, _DEF_NODES):
        return node.name == name
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False


def _span(node: ast.AST) -> tuple[int, int]:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return first, node.end_lineno or node.lineno


def find_definition(text: str, name: str) -> tuple[str, tuple[int, int] | None, int]:
    """Where `name` ("step", "RINGS" or "Game.step") is defined at the top level of `text`: ("found",
    (first, last line), n) with n the number of definitions (the last one is given); ("class", the
    class's span, 0) when the class exists but not the member; ("missing", None, 0). Raises
    SyntaxError when the text does not parse."""
    tree = ast.parse(text)
    head, _, member = name.partition(".")
    if "." in member:
        return "missing", None, 0
    hits = [node for node in tree.body if _binds(node, head)]
    if not hits:
        return "missing", None, 0
    if not member:
        return "found", _span(hits[-1]), len(hits)
    classes = [node for node in hits if isinstance(node, ast.ClassDef)]
    if not classes:
        return "missing", None, 0
    members = [node for node in classes[-1].body if _binds(node, member)]
    if members:
        return "found", _span(members[-1]), len(members)
    return "class", _span(classes[-1]), 0


# --- Applying -------------------------------------------------------------------------------


@dataclass
class _Op:
    index: int
    kind: str  # "replace" (lines start..end replaced) or "insert" (after line `start`, 0 = the top)
    start: int
    end: int
    lines: list[str]
    label: str

    def key(self) -> tuple[float, int]:
        """Where the op sits: an insert after line L sits between L and L + 1; applied in reverse
        order, so the lower lines change first and the line numbers above stay right."""
        return (self.start + 0.5 if self.kind == "insert" else float(self.start), self.index)


@dataclass
class _Failure:
    """An edit that is not applied: its reason, the lines (1-based, in the snapshot's numbering) to
    show as they are now, and content to look for elsewhere."""

    index: int
    label: str
    code: str
    text: str
    show: list[int] = field(default_factory=list)
    hint: str | None = None
    hint_lines: list[int] = field(default_factory=list)  # lines already found for the hint


@dataclass
class EditResult:
    text: str
    summary: list[str]  # one entry per applied op, top to bottom
    regions: list[tuple[int, int]]  # (first, last) changed line in the new text, per op; last < first: deletion
    warnings: list[str] = field(default_factory=list)
    noop: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # one message per edit that was not applied
    total: int = 0  # edits in the request


def _span_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """1-based first and last line covered by text[start:end]."""
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    return first, last


def _preview(text: str, limit: int = 30) -> str:
    preview = text.replace("\n", "\\n")
    return preview if len(preview) <= limit else preview[: limit - 3] + "..."


def _strip_blank_edges(items: list[str]) -> tuple[list[str], int, int]:
    """items without leading and trailing blank lines, and how many were cut at each end."""
    head = 0
    while head < len(items) and not items[head].strip():
        head += 1
    tail = 0
    while tail < len(items) - head and not items[-1 - tail].strip():
        tail += 1
    return items[head : len(items) - tail], head, tail


def _validate(edits: object) -> list[dict]:
    """The request's shape: raises EditError for anything malformed (nothing is applied then)."""
    if not isinstance(edits, (list, tuple)) or not edits:
        raise EditError('[E_BAD_OP] edits must be a non-empty list of edit dicts, e.g. [{"op": "replace", "pos": "12#MQV", "lines": [...]}].')
    out = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise EditError(f"[E_BAD_OP] Edit {index} must be a dict with an \"op\".")
        unknown = set(edit) - _KEYS
        if unknown:
            raise EditError(f"[E_BAD_OP] Edit {index} has unknown fields: {', '.join(sorted(unknown))}.")
        op = edit.get("op")
        if op not in _OPS:
            raise EditError(f'[E_BAD_OP] Edit {index} has op {op!r}; use {", ".join(repr(o) for o in _OPS[:-1])} or {_OPS[-1]!r}.')
        item: dict = {"index": index, "op": op}
        if op == "replace_text":
            old, new = edit.get("oldText"), edit.get("newText")
            if not isinstance(old, str) or not isinstance(new, str) or set(edit) - {"op", "oldText", "newText"}:
                raise EditError(f'[E_BAD_OP] Edit {index}: replace_text takes only string "oldText" and "newText".')
            if not old.strip():
                raise EditError(f"[E_BAD_OP] Edit {index}: replace_text needs a non-empty oldText.")
            item.update(old=old.replace("\r\n", "\n"), new=new.replace("\r\n", "\n"))
            out.append(item)
            continue
        if "oldText" in edit or "newText" in edit:
            raise EditError(f'[E_BAD_OP] Edit {index}: "oldText"/"newText" belong to replace_text only.')
        if "lines" not in edit:
            raise EditError(f'[E_BAD_OP] Edit {index} needs "lines" (the new content; [] deletes).')
        item["lines"] = _as_lines(edit["lines"], index)
        if op == "replace_def":
            name = edit.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)?", name) or set(edit) - {"op", "name", "lines"}:
                raise EditError(f'[E_BAD_OP] Edit {index}: replace_def takes "name" (a top-level name, or Class.member) and "lines".')
            if not item["lines"]:
                raise EditError(f"[E_BAD_OP] Edit {index}: replace_def with no lines (use replace with lines=[] to delete).")
            item["name"] = name
            out.append(item)
            continue
        if "name" in edit:
            raise EditError(f'[E_BAD_OP] Edit {index}: "name" belongs to replace_def only.')
        if op == "replace":
            if "pos" not in edit:
                raise EditError(f'[E_BAD_OP] Edit {index}: replace needs a "pos" anchor.')
            item["pos"] = parse_anchor(edit["pos"])
            item["end"] = parse_anchor(edit["end"]) if edit.get("end") is not None else None
            if item["end"] is not None and item["end"].line < item["pos"].line:
                raise EditError(f"[E_BAD_OP] Edit {index}: end line {item['end'].line} comes before pos line {item['pos'].line}.")
        else:
            if edit.get("end") is not None:
                raise EditError(f'[E_BAD_OP] Edit {index}: {op} takes "pos" only, not "end".')
            if not item["lines"]:
                raise EditError(f"[E_BAD_OP] Edit {index}: {op} with no lines to insert.")
            item["pos"] = parse_anchor(edit["pos"]) if edit.get("pos") is not None else None
        out.append(item)
    return out


def apply_edits(text: str, edits: object, protected: tuple[int, int] | None = None) -> EditResult:
    """Apply anchored edits to `text`. Raises EditError with an [E_...] message when the request is
    malformed or no edit could be applied; otherwise the valid edits are applied and the result's
    `failed` lists the others, each with the lines as they are now. `protected` (first, last line,
    1-based) is a range no edit may touch."""
    items = _validate(edits)
    lines, newline_at_end = split_lines(text)
    total = len(lines)
    warnings: list[str] = []
    ops: list[_Op] = []
    failures: list[_Failure] = []

    def stale(a: Anchor) -> str | None:
        """Why anchor `a` does not fit the file now, or None when it does."""
        if a.line > total:
            return f"line {a.line} does not exist (the file has {total} lines)"
        line = lines[a.line - 1]
        if line_hash(lines, a.line - 1, len(a.hash)) == a.hash:
            if a.hint is not None and _SIGNIFICANT_RE.search(a.hint) and not _hint_matches(a.hint, line):
                return f"{a} is line {a.line}, but its content is not what you gave ({_preview(a.hint.strip())!r})"
            return None
        if a.hint is not None and _SIGNIFICANT_RE.search(a.hint) and _hint_matches(a.hint, line):
            warnings.append(f"Accepted the anchor {a}: its hash was stale but its content matched line {a.line}.")
            return None
        return f"{a} is stale: the file changed there since you read it"

    def anchored(item: dict, anchors: list[Anchor], label: str) -> bool:
        """False, with a failure recorded, when an anchor of `item` does not fit."""
        reasons = [(a, stale(a)) for a in anchors]
        bad = [(a, why) for a, why in reasons if why]
        if not bad:
            return True
        code = "E_RANGE_OOB" if all(a.line > total for a, _ in bad) else "E_STALE_ANCHOR"
        hint = next((a.hint for a, _ in bad if a.hint and _SIGNIFICANT_RE.search(a.hint)), None)
        failures.append(_Failure(item["index"], label, code, "; ".join(why for _, why in bad), [a.line for a, _ in bad], hint))
        return False

    for item in items:
        index, op = item["index"], item["op"]
        if op == "replace_text":
            old, new = item["old"], item["new"]
            label = f'replace_text "{_preview(old)}"'
            body = join_lines(lines, newline_at_end)
            found = [m.start() for m in re.finditer(re.escape(old), body)]
            if len(found) == 1:
                start, stop = found[0], found[0] + len(old)
                # A newline shared at either end of old and new is kept out: it is a line's end, not content.
                if old.startswith("\n") and new.startswith("\n"):
                    old, new, start = old[1:], new[1:], start + 1
                if old.endswith("\n") and (new.endswith("\n") or stop == len(body)):
                    old, new, stop = old[:-1], new[:-1] if new.endswith("\n") else new, stop - 1
                first, last = _span_lines(body, start, stop)
                if old.endswith("\n"):  # the new text joins the next line to the last one changed
                    last += 1
                segment_start = sum(len(line) + 1 for line in lines[: first - 1])
                segment = "\n".join(lines[first - 1 : last])
                offset = start - segment_start
                replaced = segment[:offset] + new + segment[offset + len(old) :]
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
            if len(found) > 1:
                starts = [_span_lines(body, f, f + 1)[0] for f in found]
                failures.append(_Failure(index, label, "E_MULTI_MATCH", f"{len(found)} exact matches; say which with anchors", starts))
                continue
            # Ignoring whitespace: whole lines whose content matches oldText's lines.
            old_lines, head_cut, tail_cut = _strip_blank_edges(old.split("\n"))
            probe = [_ws(line) for line in old_lines]
            matches = [n for n in range(1, total - len(probe) + 2) if [_ws(line) for line in lines[n - 1 : n - 1 + len(probe)]] == probe] if probe else []
            if len(matches) == 1:
                first = matches[0]
                new_lines = new.split("\n")
                if new_lines and new_lines[-1] == "" and new.endswith("\n"):
                    new_lines.pop()
                cut_head = 0
                while cut_head < head_cut and new_lines and not new_lines[0].strip():
                    new_lines, cut_head = new_lines[1:], cut_head + 1
                cut_tail = 0
                while cut_tail < tail_cut and new_lines and not new_lines[-1].strip():
                    new_lines, cut_tail = new_lines[:-1], cut_tail + 1
                if not new.strip():
                    new_lines = []
                warnings.append(f"Edit {index} ({label}) matched ignoring whitespace (lines {first}-{first + len(probe) - 1}); newText was used as given.")
                ops.append(_Op(index, "replace", first, first + len(probe) - 1, new_lines, label))
            elif matches:
                failures.append(_Failure(index, label, "E_MULTI_MATCH", f"{len(matches)} matches (ignoring whitespace); say which with anchors", matches))
            else:
                significant = next((line for line in old_lines if _SIGNIFICANT_RE.search(line)), old_lines[0] if old_lines else old)
                failures.append(_Failure(index, label, "E_NO_MATCH", "no match, even ignoring whitespace", [], significant))
            continue
        new_lines = item["lines"]
        if op == "replace_def":
            name = item["name"]
            label = f"replace_def {name}"
            try:
                where, span, count = find_definition(join_lines(lines, newline_at_end), name)
            except SyntaxError as exc:
                failures.append(_Failure(index, label, "E_NO_DEF", f"the file does not parse (line {exc.lineno}: {exc.msg}), so "
                                         f"{name} cannot be found; fix the syntax with replace or replace_text first", [exc.lineno or 1]))
                continue
            if where == "found":
                if count > 1:
                    warnings.append(f"Edit {index}: {name} is defined {count} times; the last definition (lines {span[0]}-{span[1]}) was replaced.")
                where = f"line {span[0]}" if span[0] == span[1] else f"lines {span[0]}-{span[1]}"
                ops.append(_Op(index, "replace", span[0], span[1], new_lines, f"{label} ({where})"))
            elif where == "class":
                warnings.append(f"Edit {index}: {name} was not defined; added at the end of class {name.split('.')[0]}.")
                ops.append(_Op(index, "insert", span[1], span[1], new_lines, f"{label} (added at the end of the class)"))
            elif "." in name:
                failures.append(_Failure(index, label, "E_NO_DEF", f"class {name.split('.')[0]} is not defined at the top level", []))
            else:
                warnings.append(f"Edit {index}: {name} was not defined; added at the end.")
                lead = [""] if lines and lines[-1].strip() else []
                ops.append(_Op(index, "insert", total, total, lead + new_lines, f"{label} (added at the end)"))
            continue
        pos, end = item["pos"], item.get("end")
        if op == "replace":
            label = f"replace {pos}" + (f"-{end}" if end is not None else "")
            if not anchored(item, [pos] + ([end] if end is not None else []), label):
                continue
            last = end.line if end is not None else pos.line
            if end is None and len(new_lines) > 1:
                warnings.append(f"Edit {index} replaced only line {pos.line} with {len(new_lines)} lines (no end anchor); add end to replace a range.")
            ops.append(_Op(index, "replace", pos.line, last, new_lines, label))
        else:
            if op == "append":
                label = f"append after {pos}" if pos is not None else "append at the end"
            else:
                label = f"prepend before {pos}" if pos is not None else "prepend at the start"
            if pos is not None and not anchored(item, [pos], label):
                continue
            if op == "append":
                boundary = pos.line if pos is not None else total
            else:
                boundary = pos.line - 1 if pos is not None else 0
            ops.append(_Op(index, "insert", boundary, boundary, new_lines, label))

    if protected:
        first, last = protected
        kept_ops = []
        for o in ops:
            inside = (o.start <= last and o.end >= first) if o.kind == "replace" else first <= o.start < last
            if inside:
                failures.append(_Failure(o.index, o.label, "E_FIXED_BLOCK", f"it touches the FIXED block (lines {first}-{last}), "
                                         "which must not change; write your code below it", []))
            else:
                kept_ops.append(o)
        ops = kept_ops

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
    clashing: set[int] = set()
    for i, a in enumerate(kept):
        for b in kept[i + 1 :]:
            if a.kind == b.kind == "insert":
                continue  # inserts at one place go in, in order
            if a.kind == b.kind == "replace":
                clash = b.start <= a.end and a.start <= b.end
                shared = max(a.start, b.start)
            else:
                r, ins = (a, b) if a.kind == "replace" else (b, a)
                clash = r.start <= ins.start < r.end
                shared = ins.start
            if clash:
                clashing.update((a.index, b.index))
                both = f"edits {a.index} ({a.label}) and {b.index} ({b.label}) both change line {shared}: merge them into one edit"
                failures.append(_Failure(a.index, a.label, "E_EDIT_CONFLICT", both, [shared]))
                failures.append(_Failure(b.index, b.label, "E_EDIT_CONFLICT", both, []))
    kept = [o for o in kept if o.index not in clashing]
    failures.sort(key=lambda f: f.index)

    if not kept:
        if failures:
            messages = _failure_messages(failures, lines, lambda n: n)
            raise EditError(f"No edit was applied ({len(failures)} of {len(items)} failed; nothing changed):\n" + "\n".join(messages))
        return EditResult(text, [], [], warnings, noop or ["the edits change nothing"], [], len(items))

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
    for o in sorted(kept, key=_Op.key, reverse=True):
        if o.kind == "replace":
            new_lines_all[o.start - 1 : o.end] = o.lines
        else:
            new_lines_all[o.start : o.start] = o.lines
    if lines and not new_lines_all:
        raise EditError("[E_WOULD_EMPTY] Refusing to empty the file.")
    summary, regions = [], []
    delta = 0
    shifts: list[tuple[_Op, int]] = []  # each applied op with the line delta above it
    for o in sorted(kept, key=_Op.key):
        n_new = len(o.lines)
        shifts.append((o, delta))
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

    def shift(n: int) -> int:
        """Line n of the snapshot in the new text's numbering (inside a replaced range: its start)."""
        d = 0
        for o, above in shifts:
            if o.kind == "replace":
                if o.start <= n <= o.end:
                    return o.start + above
                if o.end < n:
                    d += len(o.lines) - (o.end - o.start + 1)
            elif o.start < n:
                d += len(o.lines)
        return n + d

    failed = _failure_messages(failures, new_lines_all, shift)
    return EditResult(join_lines(new_lines_all, newline_at_end or not lines), summary, regions, warnings, noop, failed, len(items))


def _failure_messages(failures: list[_Failure], lines: list[str], shift) -> list[str]:
    """One message per failed edit: its reason, the lines it pointed at as they are now (one line of
    context each side, fresh anchors) and the lines like the content it gave."""
    out = []
    total = len(lines)
    for f in failures:
        message = [f"Edit {f.index} ({f.label}) not applied: [{f.code}] {f.text}."]
        shown: set[int] = set()
        for n in f.show:
            m = min(max(1, shift(n)), total) if total else 0
            if not m:
                continue
            a, b = max(1, m - 1), min(total, m + 1)
            note = f"  Lines {a}-{b} now:" if a != b else f"  Line {a} now:"
            fresh = [line for line in format_lines(lines, a, b) if int(line.split("#")[0]) not in shown]
            if fresh:
                message.append(note)
                message += ["    " + line for line in fresh]
                shown.update(range(a, b + 1))
        if f.hint:
            close = [n for n in _close_lines(lines, f.hint) if n not in shown]
            if close:
                exact = [n for n in close if _ws(lines[n - 1]) == _ws(f.hint)]
                what = "Lines with the content you gave" if exact else "Lines like the content you gave"
                message.append(f"  {what} ({_preview(f.hint.strip(), 40)!r}):")
                message += ["    " + line for line in format_lines(lines, 1, total) if int(line.split("#")[0]) in close]
        out.append("\n".join(message))
    return out


def fresh_anchors(text: str, regions: list[tuple[int, int]], max_lines: int = ANCHOR_LINES) -> list[str]:
    """Anchors around each changed region of the new text: the region with one line of context when
    it has at most `max_lines` lines, else its first and last lines with context."""
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
        if last - first + 1 <= max_lines:
            out += format_lines(lines, a, b)
        else:
            out += format_lines(lines, a, min(total, first + 1))
            out.append(f"   ... ({last - first - 3} more new lines) ...")
            out += format_lines(lines, max(1, last - 1), b)
        out.append("   ...")
    return out[:-1] if out and out[-1] == "   ..." else out
