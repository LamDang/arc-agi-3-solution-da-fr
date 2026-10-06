"""Support: how much of engine.py the recorded steps exercise (PLAY_DESIGN.md 3.11).

A rule in engine.py is only as good as the steps that ran it. This module measures that, on the harness
side, from the runs the tests and the predictions already make:

- ``instrument(source, path)`` compiles an engine with every condition of the model's part (after the
  FIXED block's end marker) wrapped in a recorder call: each operand of an ``and``/``or`` and the test of
  each ``if``/``elif``/``while``/ternary becomes ``__support_cond__(k, <expr>)``, which returns the value
  unchanged, so short-circuit order and every result are as before. Line numbers are kept (the call takes
  the node's location), so tracebacks, printed output and hashline anchors are unaffected. When the
  rewrite cannot be compiled the plain source is used (no condition data); when that fails too, None.
- ``Tracer`` records, per action position, the engine lines executed (``sys.monitoring`` LINE events set
  on the engine's own code objects only; each location reports once per position, then is disabled until
  the next position) and the conditions evaluated with their truth value.
- ``fold`` turns the positions of the passing steps into a support map (the schema below);
  ``path_support`` reads one position's path against it (its weakest line, the untested lines it runs, the
  compound conditions it evaluates whose operands the recording never separated).
- ``remap``/``margins`` carry a map over to an edited engine.py: lines are matched by the hash of their
  text (a line whose text changed is "new", one that only moved keeps its count); ``comments`` gives
  the trailing ``# support (n): ...`` comment of every line of step() and the functions it calls
  (``step_functions``), which read_file() and the engine listings show.

The support map (``engine_committed.support.json``, ``TestReport.support``)::

    {"schema": 1, "engine_sha": "<sha256 of the engine file>", "steps": 120, "thin": 3,
     "lines": {"512": {"n": 41, "steps": [3, 7, 8, 9, 101, 110, 115, 119]}, ...},
     "conds": {"4": {"line": 512, "kind": "and", "text": "a", "group": 2, "true": 30, "false": 11}, ...},
     "compound": [{"group": 2, "op": "and", "line": 512, "text": "a and b", "operands": [4, 5],
                   "separated": false, "missing": ["b"]}, ...],
     "keys": ["QXZWRM", ...],
     "summary": {"steps": 120, "lines": 210, "untested": 12, "thin": 30, "supported": 168, "compound": 5,
                 "unseparated": 2}}

- ``lines``: every executable line of the model's functions (module-level lines run when the file loads,
  not in a step, and are left out), with ``n`` the number of recorded steps that executed it (0:
  untested; below ``thin``: thin) and the first and last ``STEPS_KEPT`` (5) of those steps.
- ``conds``: per condition, how many steps evaluated it True and how many False (a step can count in
  both); ``kind`` is if/while/ifexp for a test, and/or for an operand of that ``group``.
- ``compound``: per ``and``/``or`` (one per BoolOp node), whether the steps separated its operands: for
  ``A and B`` every operand was evaluated False on some step (B False while A held, A False), for
  ``or`` every operand True on some step. ``missing`` lists the operands that never decided it: the
  recording cannot tell the compound from the simpler rule without them. A None check (``x is not None``,
  ``x is None``) is a guard, not a rule: it is never listed as missing.
- ``keys``: per line of the file, the hash of its text (hashline alphabet), for ``remap``.
"""

from __future__ import annotations

import ast
import dis
import hashlib
import inspect
import re
import sys
import types
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from engine_re import hashline

SCHEMA = 1
THIN_SUPPORT = 3  # a line run by fewer recorded steps is thin; by none, untested
STEPS_KEPT = 5  # step indices kept per line: the first and the last this many (the comments show the last five)
COMMENT_STEPS = 5  # step indices a support comment names, newest first
# A support comment as read_file() and the listings append it to a line (never part of the file: edit_file drops it,
# hashline.strip_support_comments).
COMMENT_RE = hashline.SUPPORT_COMMENT_RE
COND_NAME = "__support_cond__"  # the recorder's name in the engine module (not mangled inside a class)
END_MARKER = "# ==== END OF FIXED INTERFACE ===="
OUTCOMES = ("level_solved", "game_over")
_SETUP_OPS = {"RESUME", "MAKE_CELL", "COPY_FREE_VARS", "RETURN_GENERATOR", "NOP", "CACHE"}
_GUARD = re.compile(r"^[\w.\[\]'\"]+ is (not )?None$")  # an operand that only guards against None
_TOOL_IDS = (1, 3, 4)  # sys.monitoring.COVERAGE_ID first, then two ids no standard tool claims


def first_model_line(source: str) -> int:
    """The first line after the FIXED block's end marker (1 when there is none)."""
    for n, line in enumerate(source.splitlines(), 1):
        if line.strip() == END_MARKER:
            return n + 1
    return 1


def text_key(line: str) -> str:
    """The hash of a line's own text, without its neighbours (hashline's anchors include them)."""
    return hashline.text_hash(line)


def no_cond(k: int, value: Any) -> Any:  # noqa: ARG001  (the recorder when nothing is traced)
    return value


# --- The condition rewrite ----------------------------------------------------------------------------


def _short(text: str | None, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Rewriter(ast.NodeTransformer):
    def __init__(self, source: str, first: int):
        self.source = source
        self.first = first
        self.conds: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []

    def _ours(self, node: ast.AST) -> bool:
        return getattr(node, "lineno", 0) >= self.first

    def _wrap(self, node: ast.expr, kind: str, text: str, group: int | None = None) -> ast.expr:
        k = len(self.conds)
        self.conds.append({"id": k, "line": node.lineno, "kind": kind, "text": text, "group": group})
        call = ast.Call(func=ast.Name(id=COND_NAME, ctx=ast.Load()), args=[ast.Constant(value=k), node], keywords=[])
        ast.copy_location(call, node)
        for child in (call.func, call.args[0]):
            ast.copy_location(child, node)
        return call

    def _test(self, node: ast.AST, kind: str) -> ast.AST:
        ours = self._ours(node)
        text = _short(ast.get_source_segment(self.source, node.test)) if ours else ""
        self.generic_visit(node)
        if ours and not isinstance(node.test, ast.Constant):
            node.test = self._wrap(node.test, kind, text)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        return self._test(node, "if")

    def visit_While(self, node: ast.While) -> ast.AST:
        return self._test(node, "while")

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return self._test(node, "ifexp")

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if not self._ours(node):
            return self.generic_visit(node)
        op = "and" if isinstance(node.op, ast.And) else "or"
        texts = [_short(ast.get_source_segment(self.source, v), 40) for v in node.values]
        g = len(self.groups)
        group = {"id": g, "op": op, "line": node.lineno, "text": _short(ast.get_source_segment(self.source, node)), "operands": []}
        self.groups.append(group)
        self.generic_visit(node)
        values = []
        for value, text in zip(node.values, texts):
            values.append(self._wrap(value, op, text, g))
            group["operands"].append(len(self.conds) - 1)
        node.values = values
        return node


def _codes(code: types.CodeType) -> list[types.CodeType]:
    out = [code]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out += _codes(const)
    return out


def _executable(code: types.CodeType, first: int) -> list[int]:
    """The lines a step can run: those of the model's functions (CO_OPTIMIZED: functions, lambdas, generator
    expressions; not the module or a class body, which run when the file loads), set-up instructions left out."""
    lines: set[int] = set()
    for c in _codes(code):
        if not c.co_flags & inspect.CO_OPTIMIZED:
            continue
        for ins in dis.get_instructions(c):
            line = ins.positions.lineno if ins.positions else None
            if line and line >= first and ins.opname not in _SETUP_OPS:
                lines.add(line)
    return sorted(lines)


@dataclass
class Instrumented:
    """An engine compiled for support: the code object (exec it into a module whose COND_NAME is set), the
    first line of the model's part, its executable lines, its conditions and its and/or groups."""

    code: types.CodeType
    first: int
    lines: list[int]
    conds: list[dict[str, Any]] = field(default_factory=list)
    groups: list[dict[str, Any]] = field(default_factory=list)

    def static(self) -> dict[str, Any]:
        return {"first": self.first, "lines": self.lines, "conds": self.conds, "groups": self.groups}


def instrument(source: str, path: str) -> Instrumented | None:
    """Compile `source` with its conditions wrapped (see the module docstring); the plain source when the
    rewrite does not compile (no conditions recorded); None when the source itself does not compile."""
    first = first_model_line(source)
    try:
        tree = ast.parse(source, path)
        rewriter = _Rewriter(source, first)
        tree = ast.fix_missing_locations(rewriter.visit(tree))
        code = compile(tree, path, "exec", dont_inherit=True)
        return Instrumented(code, first, _executable(code, first), rewriter.conds, rewriter.groups)
    except Exception:  # noqa: BLE001  (an AST the rewrite does not handle: plain code, lines only)
        pass
    try:
        code = compile(source, path, "exec", dont_inherit=True)
    except Exception:  # noqa: BLE001  (the caller compiles it again and reports the error as before)
        return None
    return Instrumented(code, first, _executable(code, first))


# --- Recording ----------------------------------------------------------------------------------------


class Tracer:
    """Records per position the model's lines executed and the conditions evaluated (as k*2 + truth).

    start() claims a sys.monitoring tool id and sets LINE events on the engine's code objects (nothing
    else is instrumented); begin() opens a position (re-enabling the locations disabled during the last
    one), end(key) closes it. The recorder in `namespace` (COND_NAME) records into the open position."""

    def __init__(self, inst: Instrumented, namespace: dict[str, Any] | None = None):
        self.inst = inst
        self.executed: dict[str, list[int]] = {}
        self.evaluated: dict[str, list[list[int]]] = {}
        self._lines: set[int] | None = None
        self._conds: set[int] | None = None
        self.tool: int | None = None
        self.namespaces: list[dict[str, Any]] = []
        if namespace is not None:
            self.attach(namespace)

    def attach(self, namespace: dict[str, Any]) -> None:
        namespace[COND_NAME] = self.cond
        self.namespaces.append(namespace)

    def cond(self, k: int, value: Any) -> Any:
        seen = self._conds
        if seen is not None:
            try:
                seen.add(k + k + (1 if value else 0))
            except Exception:  # noqa: BLE001  (a value with no truth, e.g. an array: the code never asked for it)
                pass
        return value

    def _on_line(self, code: types.CodeType, line: int) -> Any:  # noqa: ARG002
        seen = self._lines
        if seen is not None and line >= self.inst.first:
            seen.add(line)
        return sys.monitoring.DISABLE

    def start(self) -> bool:
        mon = getattr(sys, "monitoring", None)
        if mon is None:
            return False
        for tool in _TOOL_IDS:
            if mon.get_tool(tool) is None:
                mon.use_tool_id(tool, "engine_re.support")
                self.tool = tool
                break
        else:
            return False
        mon.register_callback(self.tool, mon.events.LINE, self._on_line)
        for code in _codes(self.inst.code):
            mon.set_local_events(self.tool, code, mon.events.LINE)
        return True

    def stop(self) -> None:
        mon = getattr(sys, "monitoring", None)
        if mon is None or self.tool is None:
            return
        for code in _codes(self.inst.code):
            mon.set_local_events(self.tool, code, 0)
        mon.register_callback(self.tool, mon.events.LINE, None)
        mon.free_tool_id(self.tool)
        self.tool = None
        for namespace in self.namespaces:
            namespace[COND_NAME] = no_cond

    def begin(self) -> None:
        if self.tool is not None:
            sys.monitoring.restart_events()
        self._lines, self._conds = set(), set()

    def end(self, key: str) -> None:
        if self._lines is None or self._conds is None:
            return
        self.executed[key] = sorted(self._lines)
        self.evaluated[key] = [[c >> 1, c & 1] for c in sorted(self._conds)]
        self._lines = self._conds = None


# --- The support map ------------------------------------------------------------------------------------


def _kept(steps: list[int]) -> list[int]:
    return steps if len(steps) <= 2 * STEPS_KEPT else steps[:STEPS_KEPT] + steps[-STEPS_KEPT:]


def fold(coverage: dict[str, Any], executed: dict[str, list[int]], evaluated: dict[str, list[list[int]]],
         positions: dict[str, int], source: str, sha: str | None = None) -> dict[str, Any]:
    """The support map (module docstring) of the positions `positions` (position key -> step index; the
    passing steps) of one run: its `coverage` (the runner's static data), `executed` and `evaluated`."""
    lines = {int(line): [] for line in coverage.get("lines") or []}
    true: dict[int, int] = {}
    false: dict[int, int] = {}
    for key, step in sorted(positions.items(), key=lambda kv: kv[1]):
        for line in executed.get(key) or []:
            if line in lines:
                lines[line].append(step)
        for k, t in evaluated.get(key) or []:
            (true if t else false)[k] = (true if t else false).get(k, 0) + 1
    thin = THIN_SUPPORT
    conds = {}
    for c in coverage.get("conds") or []:
        k = int(c["id"])
        conds[str(k)] = {"line": c["line"], "kind": c["kind"], "text": c["text"], "group": c.get("group"),
                         "true": true.get(k, 0), "false": false.get(k, 0)}
    compound = []
    for g in coverage.get("groups") or []:
        decide = "false" if g["op"] == "and" else "true"
        missing = [conds[str(k)]["text"] for k in g["operands"]
                   if not conds[str(k)][decide] and not _GUARD.match(conds[str(k)]["text"])]
        compound.append({"group": g["id"], "op": g["op"], "line": g["line"], "text": g["text"], "operands": g["operands"],
                         "separated": not missing, "missing": missing})
    counts = [len(s) for s in lines.values()]
    text_lines = source.splitlines()
    return {
        "schema": SCHEMA,
        "engine_sha": sha or hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "steps": len(positions),
        "thin": thin,
        "lines": {str(line): {"n": len(s), "steps": _kept(s)} for line, s in sorted(lines.items())},
        "conds": conds,
        "compound": compound,
        "keys": [text_key(t) for t in text_lines],
        "summary": {
            "steps": len(positions), "lines": len(counts), "untested": sum(c == 0 for c in counts),
            "thin": sum(0 < c < thin for c in counts), "supported": sum(c >= thin for c in counts),
            "compound": len(compound), "unseparated": sum(not c["separated"] for c in compound),
        },
    }


def fold_result(result: dict[str, Any], positions: dict[str, int], source: str, sha: str | None = None) -> dict[str, Any] | None:
    """fold() on a candidate run's result; None when the run recorded nothing (no tracer, an old runner)."""
    coverage = result.get("coverage")
    if not coverage or "executed" not in result:
        return None
    return fold(coverage, result["executed"], result.get("evaluated") or {}, positions, source, sha)


def summary_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "no support data"
    return (f"{summary['lines']} lines: {summary['untested']} untested, {summary['thin']} thin, {summary['supported']} "
            f"supported; {summary['unseparated']} of {summary['compound']} and/or conditions never separated "
            f"({summary['steps']} steps)")


def path_support(support: dict[str, Any], lines: list[int], evaluated: list[list[int]] | None = None) -> dict[str, Any]:
    """One position's path against a support map: "weakest" (the least support of a line it ran; None when
    it ran none of the model's lines), "weakest_lines", "untested" and "thin" lines it ran, and "unseparated":
    the compound conditions it evaluated that the recording never separated ({"line", "text", "missing"})."""
    table = support.get("lines") or {}
    known = [(int(line), table[str(line)]["n"]) for line in lines if str(line) in table]
    weakest = min((n for _, n in known), default=None)
    conds = support.get("conds") or {}
    ran = {int(k) for k, _ in evaluated or []}
    unseparated = []
    for c in support.get("compound") or []:
        if c["separated"]:
            continue
        # The move relies on it when it evaluates an operand that never decided it (e.g. B of `A and B` once A held).
        loose = [k for k in c["operands"] if str(k) in conds and conds[str(k)]["text"] in c["missing"]]
        if any(k in ran for k in loose):
            unseparated.append({"line": c["line"], "text": c["text"], "missing": c["missing"]})
    return {
        "weakest": weakest,
        "weakest_lines": [line for line, n in known if n == weakest],
        "lines": len(known),
        "untested": [line for line, n in known if n == 0],
        "thin": [line for line, n in known if 0 < n < support.get("thin", THIN_SUPPORT)],
        "unseparated": unseparated,
    }


# --- Words ----------------------------------------------------------------------------------------------


def line_ranges(lines: list[int], executable: Any = None) -> str:
    """"512-518, 530": consecutive lines joined, also across lines that are not executable (blank, comments)."""
    lines = sorted(set(lines))
    if not lines:
        return ""
    exe = {int(x) for x in executable} if executable is not None else None
    parts, start, prev = [], lines[0], lines[0]
    for line in lines[1:]:
        gap = range(prev + 1, line)
        if line == prev + 1 or (exe is not None and not any(g in exe for g in gap)):
            prev = line
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = line
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _lines_word(text: str) -> str:
    return ("lines " if ("," in text or "-" in text) else "line ") + text


def counted_ranges(lines: list[int], table: dict[str, Any], limit: int = 4) -> str:
    """Lines grouped by their support, fewest first: "lines 305-315 (2 steps), line 328 (1 step)"."""
    by_n: dict[int, list[int]] = {}
    for line in lines:
        by_n.setdefault(table[str(line)]["n"], []).append(line)
    parts = [f"{_lines_word(line_ranges(ls, table))} ({n} step{'s' if n != 1 else ''})" for n, ls in sorted(by_n.items())]
    return ", ".join(parts[:limit]) + (f" and {len(parts) - limit} more" if len(parts) > limit else "")


def unseparated_text(items: list[dict[str, Any]], limit: int = 2) -> str:
    """"`a and b` (line 512; never decided by: b)"."""
    shown = [f"`{c['text']}` (line {c['line']}; never decided by: {', '.join(c['missing'])})" for c in items[:limit]]
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return "; ".join(shown) + more


def move_support_text(ps: dict[str, Any], executable: Any = None) -> str:
    """What a move's path rests on, in words: "first to run lines 512-518 (no step so far)", "its path is
    supported by at least 41 steps", the thin lines, the conditions never separated."""
    if ps.get("weakest") is None:
        return "runs none of your replica's step code"
    parts = []
    if ps["untested"]:
        parts.append(f"first to run {_lines_word(line_ranges(ps['untested'], executable))} (no step so far)")
    elif ps["weakest"] < THIN_SUPPORT:
        parts.append(f"its weakest {_lines_word(line_ranges(ps['weakest_lines'], executable))} ran in only {ps['weakest']} "
                     f"step{'s' if ps['weakest'] != 1 else ''} so far")
    else:
        parts.append(f"its path is supported by at least {ps['weakest']} steps")
    if ps.get("unseparated"):
        parts.append("it relies on conditions never separated: " + unseparated_text(ps["unseparated"]))
    return "; ".join(parts)


def mismatch_support_text(ps: dict[str, Any] | None, executable: Any = None) -> str:
    """The mismatch message's sentence on the failing move's path (a compact or full path_support)."""
    if not ps or ps.get("weakest") is None:
        return ""
    if ps["untested"]:
        text = f"this step was the first to run {_lines_word(line_ranges(ps['untested'], executable))} (no earlier step ran them)"
    elif ps["weakest"] < THIN_SUPPORT:
        text = (f"the weakest {_lines_word(line_ranges(ps['weakest_lines'], executable))} on its path had run in only "
                f"{ps['weakest']} earlier step{'s' if ps['weakest'] != 1 else ''}")
    else:
        text = f"every line on its path had run in at least {ps['weakest']} earlier steps"
    if ps.get("unseparated"):
        text += "; it relied on conditions the earlier steps never separated: " + unseparated_text(ps["unseparated"])
    return text


def path_lines_text(support: dict[str, Any], lines: list[int], evaluated: list[list[int]] | None = None,
                    indent: str = "    ") -> list[str]:
    """For a test report's failing step: the lines it ran, with the support of the thin ones."""
    ps = path_support(support, lines, evaluated)
    if ps["weakest"] is None:
        return []
    exe = support.get("lines") or {}
    out = [f"{indent}its path in engine.py ({ps['lines']} lines; support = how many passing steps ran each line):"]
    if ps["untested"]:
        out.append(f"{indent}  untested, no passing step ran them: {_lines_word(line_ranges(ps['untested'], exe))}")
    if ps["thin"]:
        out.append(f"{indent}  thin: {counted_ranges(ps['thin'], exe)}")
    rest = ps["lines"] - len(ps["untested"]) - len(ps["thin"])
    if rest:
        floor = min(exe[str(line)]["n"] for line in lines if str(line) in exe and exe[str(line)]["n"] >= THIN_SUPPORT)
        out.append(f"{indent}  the other {rest} line{'s' if rest > 1 else ''}: at least {floor} steps each")
    if ps["unseparated"]:
        out.append(f"{indent}  conditions it evaluated that no step separated: " + unseparated_text(ps["unseparated"], 3))
    return out


# --- Carrying a map over to an edited engine.py -------------------------------------------------------


def align(support: dict[str, Any], text: str) -> dict[int, int]:
    """Current line -> the line of the engine the map was made on, for every line whose text is unchanged:
    matched in order (difflib on the text hashes), then lines that moved (a text hash left over once on
    each side)."""
    old = list(support.get("keys") or [])
    new = [text_key(t) for t in text.splitlines()]
    matcher = SequenceMatcher(None, old, new, autojunk=False)
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                mapping[j1 + d + 1] = i1 + d + 1
    taken = set(mapping.values())
    left_old: dict[str, list[int]] = {}
    for i, key in enumerate(old, 1):
        if i not in taken:
            left_old.setdefault(key, []).append(i)
    left_new: dict[str, list[int]] = {}
    for j, key in enumerate(new, 1):
        if j not in mapping:
            left_new.setdefault(key, []).append(j)
    for key, js in left_new.items():
        if len(js) == 1 and len(left_old.get(key, [])) == 1:
            mapping[js[0]] = left_old[key][0]
    return mapping


def _code_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def margins(support: dict[str, Any] | None, text: str) -> dict[int, str]:
    """The support margin of each line of `text` (engine.py now): "·" for the FIXED block, the count for a
    line the map knows (0: untested), "new" for a code line of the model's part changed since the map was
    made; lines that no step runs (blank, comments, module level) get no entry."""
    lines = text.splitlines()
    first = first_model_line(text)
    out: dict[int, str] = {n: "·" for n in range(1, first) if first > 1}
    if not support:
        return out
    same = support.get("engine_sha") == hashlib.sha256(text.encode("utf-8")).hexdigest()
    mapping = {n: n for n in range(1, len(lines) + 1)} if same else align(support, text)
    table = support.get("lines") or {}
    for n in range(first, len(lines) + 1):
        old = mapping.get(n)
        if old is None:
            line = lines[n - 1]
            if _code_line(line) and line[:1].isspace() and not line.strip().startswith(("def ", "class ", "@")):
                out[n] = "new"  # an indented code line: in a function, most likely (module-level lines run at load)
        elif str(old) in table:
            out[n] = str(table[str(old)]["n"])
    return out


def remap(support: dict[str, Any], text: str) -> dict[str, Any]:
    """The map carried over to engine.py as it is now (`text`): "lines" keyed by the current line numbers (a new
    line has n 0 and "new": True), "compound" with current line numbers (those whose line changed are left out)."""
    if support.get("engine_sha") == hashlib.sha256(text.encode("utf-8")).hexdigest():
        return support
    mapping = align(support, text)
    table = support.get("lines") or {}
    lines = {}
    for n, label in margins(support, text).items():
        if label == "new":
            lines[str(n)] = {"n": 0, "steps": [], "new": True}
        elif label != "·":
            lines[str(n)] = table[str(mapping[n])]
    back = {old: new for new, old in mapping.items()}
    compound = [{**c, "line": back[c["line"]]} for c in support.get("compound") or [] if c["line"] in back]
    return {**support, "lines": lines, "compound": compound, "remapped": True}


def compound_lookup(support: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(c["line"], c["text"]): c for c in support.get("compound") or []}


# --- Support comments: the step code, line by line -------------------------------------------------------


def step_functions(source: str) -> dict[str, tuple[int, int]]:
    """The top-level functions of the model's part that a step runs: step() and every function it calls,
    transitively (calls by name, resolved from the AST), make_level and the level/sprite functions left
    out unless step calls them: name -> (first, last line). Empty when the source does not parse or has
    no step()."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    first = first_model_line(source)
    defs = {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno >= first}
    if "step" not in defs:
        return {}
    found: dict[str, tuple[int, int]] = {}
    todo = ["step"]
    while todo:
        name = todo.pop()
        if name in found or name not in defs:
            continue
        node = defs[name]
        found[name] = (node.lineno, node.end_lineno or node.lineno)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in defs:
                if child.func.id != "make_level":
                    todo.append(child.func.id)
    return found


def comment_text(entry: dict[str, Any]) -> str:
    """The support comment of a line of the map: "# support (31): 42, 41, 39, 37, 36 and 26 other" (the number
    of passing steps that ran the line, the last COMMENT_STEPS of them newest first), "# support (0): untested",
    "# support: new" for a line changed since the map was made."""
    if entry.get("new"):
        return "# support: new"
    n = int(entry.get("n") or 0)
    if not n:
        return "# support (0): untested"
    shown = [int(s) for s in (entry.get("steps") or [])][-COMMENT_STEPS:][::-1]
    other = n - len(shown)
    return f"# support ({n}): " + ", ".join(str(s) for s in shown) + (f" and {other} other" if other > 0 else "")


def comments(support: dict[str, Any] | None, text: str) -> dict[int, str]:
    """The trailing support comment of each line of `text` (engine.py now) inside step() and the functions it
    calls (step_functions): comment_text of the line's entry in the map carried over to the text (remap); lines
    the map does not know (blank, comments, the def line) get none. Empty without a map."""
    if not support:
        return {}
    spans = step_functions(text)
    if not spans:
        return {}
    table = remap(support, text).get("lines") or {}
    out: dict[int, str] = {}
    for first, last in spans.values():
        for n in range(first, last + 1):
            entry = table.get(str(n))
            if entry is not None:
                out[n] = comment_text(entry)
    return out


strip_comments = hashline.strip_support_comments


# --- Outcome rules ----------------------------------------------------------------------------------------


def plan_items(support: dict[str, Any], source: str, path: tuple[list[int], list[list[int]]] | None,
               limit: int = 4) -> list[str]:
    """The PLAN message's support items: what is thin, untested or never separated on the last batch's path
    (`path`: its lines and evaluated conditions), then the outcome rules (where level_solved or game_over is
    set) that no step ran or whose guarding and/or was never separated; at most `limit` items."""
    exe = support.get("lines") or {}
    items: list[str] = []
    if path:
        ps = path_support(support, path[0], path[1])
        parts = []
        if ps["untested"]:
            parts.append(f"untested {_lines_word(line_ranges(ps['untested'], exe))}")
        if ps["thin"]:
            parts.append("thin " + counted_ranges(ps["thin"], exe))
        if ps["unseparated"]:
            parts.append("never separated: " + unseparated_text(ps["unseparated"]))
        if parts:
            items.append("- your last batch's path: " + "; ".join(parts))
    compounds = support.get("compound") or []
    for line, what, guards in outcome_lines(source):
        n = (exe.get(str(line)) or {}).get("n")
        if n is None:
            continue
        loose = [c for c in compounds if not c["separated"] and c["line"] in guards]
        if n and not loose:
            continue
        ran = "no step ran it (an untested rule)" if not n else (
            f"ran on {n} step{'s' if n > 1 else ''} ({', '.join(str(s) for s in exe[str(line)]['steps'])})")
        text = f"- line {line} sets {what}: {ran}"
        if loose:
            text += "; its condition " + unseparated_text(loose, 1) + " was never separated"
        items.append(text)
    return items[:limit]



def outcome_lines(source: str) -> list[tuple[int, str, list[int]]]:
    """Where the model's part sets an outcome: (line, "level_solved" | "game_over", the lines of the if/elif
    tests that guard it, innermost last). Comparisons with the outcome strings are not settings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    first = first_model_line(source)
    found: list[tuple[int, str, list[int]]] = []

    def walk(node: ast.AST, guards: list[int], in_compare: bool) -> None:
        if isinstance(node, ast.Constant) and node.value in OUTCOMES and not in_compare and node.lineno >= first:
            found.append((node.lineno, node.value, list(guards)))
        for name, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                inner = guards
                if isinstance(node, (ast.If, ast.While)) and name in ("body", "orelse"):
                    inner = guards + [node.test.lineno]
                walk(child, inner, in_compare or isinstance(node, ast.Compare))

    walk(tree, [], False)
    seen, out = set(), []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            out.append(item)
    return out
