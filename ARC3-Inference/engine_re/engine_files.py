"""The harness side of edit_file() and undo_edit(): the only code that writes engine.py.

The analysis kernel cannot open engine.py for writing (engine_re.guard); its edit_file() and
undo_edit() send their arguments to the harness process, which runs ``EngineEditor.handle``: it applies
anchored edits (engine_re.hashline, rejecting any change to the FIXED block) or restores an
earlier version, writes engine.py, and answers with the text the kernel prints.

Every change is kept as a numbered version in ``versions_dir`` (``<game_dir>/engine_versions/``,
outside the kernel's reach): ``v0001.py``, ... and ``versions.jsonl`` with what each change was.
undo_edit(n) restores the version n changes back, undo_edit(to="best") the best engine tested
(``<game_dir>/engine_best.py``: the most steps passing before the first failure in a full replay,
ties broken by the most steps passing in all, see ``best_key``); a restore is itself a new version,
so nothing is lost.
Versions are matched to test results by the sha256 of their content (``engine_sha`` in
``tests.jsonl``).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from engine_re import hashline
from engine_re.game_api import fixed_block_lines

DIFF_LINES = 200  # lines of unified diff kept per change in the transcript
RECENT_VERSIONS = 8


BEST_RULE = "the most steps passing before the first failure, ties broken by the most steps passing in all"


def passing_prefix(entry: dict[str, Any]) -> int:
    """Steps passing before the first failure, from a tests.jsonl entry (also one written before
    the field existed: those were full replays, where it is the first failing step's index)."""
    if entry.get("passing_prefix") is not None:
        return int(entry["passing_prefix"])
    if entry.get("first_fail") is not None:
        return int(entry["first_fail"]) - int(entry.get("first_step") or 0)
    return int(entry.get("total") or 0) if entry.get("exact") == entry.get("total") and not entry.get("error") else 0


def best_key(entry: dict[str, Any]) -> tuple[int, int]:
    """How full-replay results rank for engine_best.py and undo_edit(to="best") (BEST_RULE)."""
    return passing_prefix(entry), int(entry.get("exact") or 0)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def syntax_check(text: str, name: str = "engine.py") -> str:
    try:
        compile(text, name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"SYNTAX ERROR at line {exc.lineno}: {exc.msg}"
    return "Syntax OK"


class EngineEditor:
    def __init__(self, engine_path: Path, versions_dir: Path, game_dir: Path | None = None, log: Callable[[dict], None] | None = None,
                 support: Path | None = None):
        self.engine_path = Path(engine_path)
        self.versions_dir = Path(versions_dir)
        self.game_dir = Path(game_dir) if game_dir else self.engine_path.parent.parent
        self.log = log
        self.support = Path(support) if support else None  # the committed engine's support map (play mode): comments on the fresh anchors
        self.index_path = self.versions_dir / "versions.jsonl"

    def _comments(self, text: str) -> dict[int, str] | None:
        """The support comments of `text`'s step code (engine_re.support.comments), when a map is there."""
        if self.support is None or not self.support.exists():
            return None
        from engine_re import support as sup

        try:
            return sup.comments(json.loads(self.support.read_text(encoding="utf-8")), text)
        except (OSError, ValueError):
            return None

    # --- versions ---------------------------------------------------------------------------

    def versions(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return [json.loads(line) for line in self.index_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def content(self, version: int) -> str:
        return (self.versions_dir / f"v{version:04d}.py").read_text(encoding="utf-8")

    def _record(self, text: str, op: str, summary: str, lines: str = "") -> int:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        number = len(self.versions()) + 1
        (self.versions_dir / f"v{number:04d}.py").write_text(text, encoding="utf-8")
        entry = {"version": number, "op": op, "summary": summary, "lines": lines, "sha": sha256(text), "time": round(time.time(), 3)}
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return number

    def sync(self) -> None:
        """Make the latest version the current engine.py (the first one, or a change made outside edit_file())."""
        if not self.engine_path.exists():
            return
        text = self.engine_path.read_text(encoding="utf-8")
        versions = self.versions()
        if not versions:
            self._record(text, "start", "engine.py at the start")
        elif versions[-1]["sha"] != sha256(text):
            self._record(text, "outside", "engine.py changed outside edit_file()")

    def _tests_by_sha(self) -> dict[str, dict[str, Any]]:
        """The latest full-replay test result of each engine content."""
        path = self.game_dir / "tests.jsonl"
        out: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("engine_sha") and entry.get("level") is None and entry.get("from_level") in (None, 0):
                    out[entry["engine_sha"]] = entry
        return out

    @staticmethod
    def _result_text(entry: dict[str, Any] | None) -> str:
        if entry is None:
            return "not tested"
        if entry.get("passed"):
            return "tested: every test passes"
        contract = ""
        if entry.get("contract_total") and entry.get("contract_passed") != entry.get("contract_total"):
            contract = f"contract {entry.get('contract_passed')}/{entry.get('contract_total')}, "
        prefix, first = passing_prefix(entry), entry.get("first_fail")
        before = f"{prefix} pass before the first failure" + (f" (step {first})" if first is not None else "")
        return f"tested: {contract}{before}, {entry.get('exact')}/{entry.get('total')} in all"

    def history(self, limit: int = RECENT_VERSIONS) -> list[str]:
        tests = self._tests_by_sha()
        lines = []
        for v in self.versions()[-limit:]:
            what = f"{v['op']}: {v['summary']}" + (f" (lines {v['lines']})" if v.get("lines") else "")
            lines.append(f"  v{v['version']:<4} {what:<72} {self._result_text(tests.get(v['sha']))}")
        return lines

    # --- requests from the kernel -------------------------------------------------------------

    def handle(self, request: Any) -> dict[str, Any]:
        """Answer one request from the kernel: {"op": "edit", "edits": [...]} or {"op": "undo", "n": .., "to": ..}."""
        try:
            if not isinstance(request, dict):
                raise hashline.EditError("bad request")
            if request.get("op") == "edit":
                return {"ok": True, "text": self.edit(request.get("edits"))}
            if request.get("op") == "undo":
                return {"ok": True, "text": self.undo(request.get("n", 1), request.get("to"))}
            raise hashline.EditError(f"unknown request {request.get('op')!r}")
        except hashline.EditError as exc:
            return {"ok": False, "text": str(exc)}
        except Exception as exc:  # noqa: BLE001  (reported to the model, never raised in the harness)
            return {"ok": False, "text": f"Error: {type(exc).__name__}: {exc}"}

    def _write(self, old: str, new: str, op: str, summary: str, lines: str) -> int:
        self.engine_path.write_text(new, encoding="utf-8")
        version = self._record(new, op, summary, lines)
        if self.log:
            diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "engine.py", "engine.py", n=2, lineterm=""))
            if len(diff) > DIFF_LINES:
                diff = diff[:DIFF_LINES] + [f"... ({len(diff) - DIFF_LINES} more diff lines)"]
            self.log({"engine_change": {"op": op, "version": version, "summary": summary, "lines": lines, "diff": "\n".join(diff)}})
        return version

    def edit(self, edits: Any) -> str:
        self.sync()
        old = self.engine_path.read_text(encoding="utf-8")
        result = hashline.apply_edits(old, edits, protected=fixed_block_lines(old))
        notes = [f"Warning: {w}" for w in result.warnings] + [f"No change: {n}" for n in result.noop]
        if not result.summary:
            return "engine.py was not changed.\n" + "\n".join(result.failed + notes)
        where = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in result.regions if b >= a) or "deletions only"
        summary = "; ".join(result.summary)
        version = self._write(old, result.text, "edit", summary, where)
        applied = f"applied {result.total - len(result.failed)} of {result.total} edits: " if result.failed else ""
        out = [f"engine.py: {applied}{summary}. {syntax_check(result.text)}. (version {version}; undo_edit() reverts it)"]
        out += result.failed + notes
        anchors = hashline.fresh_anchors(result.text, result.regions, comments=self._comments(result.text))
        if anchors:
            out += ["Fresh anchors around the change:"] + anchors
        return "\n".join(out)

    def undo(self, n: Any = 1, to: Any = None) -> str:
        self.sync()
        versions = self.versions()
        if not versions:
            raise hashline.EditError("[E_NO_VERSION] There is no version of engine.py to go back to.")
        old = self.engine_path.read_text(encoding="utf-8")
        if to == "best":
            best = self.game_dir / "engine_best.py"
            if not best.exists():
                raise hashline.EditError('[E_NO_VERSION] No version has been tested yet, so there is no best one (run_tests first).')
            target_text = best.read_text(encoding="utf-8")
            same = [v["version"] for v in versions if v["sha"] == sha256(target_text)]
            entry = self._tests_by_sha().get(sha256(target_text))
            source = (f"the best tested version ({BEST_RULE})" + (f": v{same[-1]}" if same else "")
                      + (f", {self._result_text(entry)}" if entry else ""))
        elif to is not None:
            if isinstance(to, bool) or not isinstance(to, int) or not 1 <= to <= len(versions):
                raise hashline.EditError(f'[E_BAD_ARG] to must be "best" or a version number 1-{len(versions)}, got {to!r}.')
            target_text, source = self.content(to), f"version {to}"
        else:
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise hashline.EditError(f"[E_BAD_ARG] n must be a positive integer, got {n!r}.")
            if n >= len(versions):
                raise hashline.EditError(f"[E_NO_VERSION] Only {len(versions) - 1} change(s) to go back over (versions 1-{len(versions)}).")
            target = versions[-1 - n]["version"]
            target_text = self.content(target)
            source = f"version {target}, as engine.py was {n} change{'s' if n > 1 else ''} ago"
        if target_text == old:
            restored = f"engine.py already is {source}; nothing changed."
        else:
            version = self._write(old, target_text, "undo", f"restored {source}", "")
            restored = f"Restored {source}, saved as version {version}. {syntax_check(target_text)}."
        return "\n".join(
            [restored, "Recent versions (oldest first):"] + self.history()
            + ["read_file() again before the next edit_file(): the line anchors changed."]
        )
