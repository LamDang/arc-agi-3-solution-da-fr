"""The agent's persistent Python kernel.

    python -m engine_re.kernel WORKSPACE TRACE_DIR [--no-images] [--focus K [--history] [--play]]

Reads one JSON request per line on stdin ({"code": ...}), runs it in a
namespace that persists between requests (a {"focus": k} request, from the stepwise harness, reloads
the trace, which now holds steps 0..k: `recording` grows to it and `step_to_fix` moves to step k,
everything else is kept), and writes one JSON reply per line
({"output": ..., "images": [...]}). As in a notebook, the value of a final
expression is printed. "images" holds the pictures show_frames() made during the
request (base64 PNG and caption), for the harness to attach.

Two more requests. {"names": true} answers {"names": ["RING: list[20]", "cols: function", ...],
"more": n}: what the model has defined in the namespace (not the preloaded built-ins, modules or
dunders), each with a one-word summary, at most NAMES_SHOWN of them. {"replay": [{"turn": t, "code":
...}, ...], "cell_seconds": 20, "total_seconds": 120} re-runs those cells in order after a restart
(a resumed run, agent._resume_conversation) in replay mode: edit_file() and undo_edit() do nothing,
show_frames() makes no image, output is discarded, an exception ends only its cell, and each cell is
cut after cell_seconds (SIGALRM), the whole replay after total_seconds; it answers {"replayed": n,
"failed": [{"turn", "error"}, ...], "skipped": m, "seconds": s}.

The kernel runs sandboxed (engine_re.guard): it can read the workspace and the
trace, write only the workspace, and cannot start processes or open
connections. It cannot write engine.py at all: edit_file() and undo_edit() send their
arguments to the harness as {"rpc": {...}} lines on the same channel and print
the harness's reply, read from stdin ({"ok": ..., "text": ...}). The harness
(``KernelClient`` with an ``engine_files.EngineEditor``) validates and applies
them, so even a forged request can only make a valid edit.

``KernelClient`` is the parent side: it starts the kernel, sends code, answers
edit/undo requests, enforces a timeout and restarts the kernel when it hangs or
dies.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import select
import signal
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

from engine_re.guard import sandbox_env

MAX_OUTPUT_CHARS = 200_000
NAMES_SHOWN = 40  # entries a {"names": true} answer lists before "... and N more"
REPLAY_CELL_SECONDS = 20.0
REPLAY_TOTAL_SECONDS = 120.0
# What the namespace of the model's code starts with (besides np and the fixed-block classes): the built-in
# functions, and `engine`, engine.py as it is now (helpers._EngineModule).
FUNCTIONS = ("read_file", "edit_file", "undo_edit", "render_state", "show_frames", "replay_step")
PRELOADED = ("recording",) + FUNCTIONS + ("summarize_levels", "engine")
# Names the model's code may not rebind: the built-ins, the recording and the fixed-block classes.
RESERVED = PRELOADED + ("Sprite", "Action", "View", "State")
# The stepwise harness (--focus K): `step_to_fix`, the step to fix, instead of the recording, and no
# summarize_levels; with --history also `recording`, the recording so far (steps 0..K, all the trace on disk
# holds, recording[K] being step_to_fix), and summarize_levels.
PRELOADED_STEP = ("step_to_fix",) + FUNCTIONS + ("engine",)
RESERVED_STEP = PRELOADED_STEP + ("Sprite", "Action", "View", "State")
PRELOADED_HISTORY = PRELOADED + ("step_to_fix",)
RESERVED_HISTORY = PRELOADED_HISTORY + ("Sprite", "Action", "View", "State")
# The play-and-model agent (--play, with --focus K --history): the recording so far plus state_now and simulate.
PRELOADED_PLAY = PRELOADED_HISTORY + ("state_now", "simulate")
RESERVED_PLAY = PRELOADED_PLAY + ("Sprite", "Action", "View", "State")
ENGINE_IMPORT_NOTE = (
    "engine is a built-in that always reflects the current engine.py (an import would go stale after an edit): use "
    "engine.step(...), engine.make_level(...) directly"
)


def reserved_bindings(tree: ast.AST, reserved: tuple[str, ...] = RESERVED) -> list[tuple[str, int, str]]:
    """Where the code binds a reserved name: (name, line, how), e.g. ("show_frames", 4, "def show_frames")."""
    found: list[tuple[str, int, str]] = []

    def hit(name: str | None, node: ast.AST, how: str) -> None:
        if name in reserved:
            found.append((name, getattr(node, "lineno", 0), how))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hit(node.name, node, f"def {node.name}")
        elif isinstance(node, ast.ClassDef):
            hit(node.name, node, f"class {node.name}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            hit(node.id, node, f"{'del' if isinstance(node.ctx, ast.Del) else 'assigns'} {node.id}")
        elif isinstance(node, ast.arg):
            hit(node.arg, node, f"a parameter named {node.arg}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Any import of the engine module is refused (that copy would go stale; `engine` is the built-in).
            if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "engine":
                hit("engine", node, "from engine import")
            for alias in node.names:
                if isinstance(node, ast.Import) and alias.name.split(".")[0] == "engine":
                    hit("engine", node, f"import engine as {alias.asname}" if alias.asname else "import engine")
                else:
                    bound = alias.asname or alias.name.split(".")[0]
                    hit(bound, node, f"import as {bound}")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            hit(node.name, node, f"except ... as {node.name}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                hit(name, node, f"global {name}")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            hit(node.name, node, f"case ... {node.name}")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            hit(node.rest, node, f"case **{node.rest}")
    return sorted(set(found), key=lambda f: (f[1], f[0]))


def _reserved_error(found: list[tuple[str, int, str]], reserved: tuple[str, ...] = RESERVED) -> str:
    where = "; ".join(f"line {line}: {how}" for _, line, how in found)
    names = ", ".join(dict.fromkeys(name for name, _, _ in found))
    text = (
        f"Error: nothing was run. This code would replace the harness's built-in {names} ({where}).\n"
        f"These names are reserved: {', '.join(reserved)}. Give your own functions and variables other names.\n"
    )
    if any(name == "engine" and how.startswith(("import", "from engine")) for name, _, how in found):
        text += ENGINE_IMPORT_NOTE + ".\n"
    return text


def _run(code: str, namespace: dict[str, Any], builtins: dict[str, Any] | None = None, status: dict[str, str] | None = None) -> str:
    """Run the model's code in `namespace`. `builtins` (name -> object) are the reserved names: code
    that binds one is rejected before it runs, and any that were changed anyway are put back. `status`,
    when given, gets "error": "Type: message" if the code raised."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            tree = ast.parse(code, "<python>", "exec")
            if builtins:
                found = reserved_bindings(tree, tuple(builtins))
                if found:
                    print(_reserved_error(found, tuple(builtins)), end="")
                    return buffer.getvalue()
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<python>", "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<python>", "eval"), namespace)
                if value is not None:
                    print(repr(value))
        except BaseException as exc:  # noqa: BLE001  (report every failure of model-written code)
            if isinstance(exc, KeyboardInterrupt):
                raise
            frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename != __file__]
            print("Traceback (most recent call last):\n" + "".join(traceback.format_list(frames[-6:])) + f"{type(exc).__name__}: {exc}")
            if status is not None:
                status["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:200]
        finally:
            changed = [name for name, value in (builtins or {}).items() if namespace.get(name) is not value]
            for name in changed:
                namespace[name] = builtins[name]
            if changed:
                print(f"[harness] Your code replaced the built-in {', '.join(changed)}; restored. These names are reserved.")
    return buffer.getvalue()


def _summary(value: Any) -> str:
    """One word about a namespace value: "int", "list[20]", "ndarray(64, 64)", "function", "State", ..."""
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float, complex)):
        return type(value).__name__
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, type):
        return "class"
    if callable(value):
        return "function"
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        return f"{type(value).__name__}{shape}"
    return type(value).__name__


def user_names(namespace: dict[str, Any], preloaded: dict[str, Any], limit: int = NAMES_SHOWN) -> tuple[list[str], int]:
    """What the model defined: every name in the namespace that is not preloaded, a module or a dunder, as
    "name: summary" in definition order, at most `limit` of them, and how many more there are."""
    names = [
        f"{name}: {_summary(value)}"
        for name, value in namespace.items()
        if not name.startswith("__") and name not in preloaded and name != "np" and not isinstance(value, types.ModuleType)
    ]
    return names[:limit], max(0, len(names) - limit)


def replay_cells(cells: list[dict[str, Any]], namespace: dict[str, Any], builtins: dict[str, Any],
                 cell_seconds: float = REPLAY_CELL_SECONDS, total_seconds: float = REPLAY_TOTAL_SECONDS) -> dict[str, Any]:
    """Re-run the model's earlier cells ({"turn", "code"}) in order, in replay mode (helpers.REPLAY: no edits,
    no images), output discarded, an exception ending only its cell, each cell cut after `cell_seconds` and
    the whole replay after `total_seconds`. Returns the counts, the cells that raised and the seconds taken."""
    from engine_re import helpers

    started = time.time()
    failed: list[dict[str, Any]] = []
    skipped = 0

    def alarm(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"the cell ran longer than {cell_seconds:g} s when replayed")

    previous = signal.signal(signal.SIGALRM, alarm)
    helpers.REPLAY = True
    try:
        for i, cell in enumerate(cells):
            remaining = total_seconds - (time.time() - started)
            if remaining <= 0:
                skipped = len(cells) - i
                break
            status: dict[str, str] = {}
            signal.setitimer(signal.ITIMER_REAL, max(0.01, min(cell_seconds, remaining)))
            try:
                _run(str(cell.get("code") or ""), namespace, builtins, status)
            except BaseException as exc:  # noqa: BLE001  (the alarm fired outside the cell's own handler)
                status["error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if status.get("error"):
                failed.append({"turn": cell.get("turn"), "error": status["error"]})
    finally:
        helpers.REPLAY = False
        helpers.take_shown()
        signal.signal(signal.SIGALRM, previous)
    return {"replayed": len(cells) - skipped, "failed": failed, "skipped": skipped, "seconds": round(time.time() - started, 2)}


def main() -> int:
    workspace, trace_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    images = "--no-images" not in sys.argv[3:]
    focus = int(sys.argv[sys.argv.index("--focus") + 1]) if "--focus" in sys.argv[3:] else None
    history = "--history" in sys.argv[3:]
    play = "--play" in sys.argv[3:]
    import numpy as np

    import scipy.ndimage  # noqa: F401
    from engine_re import game_api, guard, helpers
    from engine_re.trace import Trace

    np.set_printoptions(linewidth=200, threshold=4096)
    helpers.load_trace(Trace.load(trace_dir), focus)
    helpers.ENGINE_PATH = workspace / "engine.py"
    helpers.IMAGES = images
    api = game_api.canonical()
    namespace: dict[str, Any] = {"__name__": "__main__", "np": np}
    namespace.update({name: getattr(api, name) for name in ("Sprite", "Action", "View", "State")})
    if focus is None:
        names, reserved = PRELOADED, RESERVED
    else:
        if play:
            names, reserved = PRELOADED_PLAY, RESERVED_PLAY
        else:
            names, reserved = (PRELOADED_HISTORY, RESERVED_HISTORY) if history else (PRELOADED_STEP, RESERVED_STEP)
        namespace["step_to_fix"] = helpers.recording[focus]
    namespace.update({name: getattr(helpers, name) for name in names if name != "step_to_fix"})
    builtins = {name: namespace[name] for name in reserved}
    os.chdir(workspace)
    # Replies go on a private copy of stdout; fd 1 itself goes to /dev/null so
    # code that writes to it directly cannot corrupt the protocol.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)

    def rpc(request: dict[str, Any]) -> dict[str, Any]:
        protocol.write(json.dumps({"rpc": request}) + "\n")
        protocol.flush()
        reply = sys.stdin.readline()
        return json.loads(reply) if reply else {"ok": False, "text": "the harness did not answer"}

    helpers._RPC = rpc
    guard.install(read_roots=[str(trace_dir)], write_roots=[str(workspace)], protected=[str(workspace / "engine.py")])

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if not line.strip():
            continue
        request = json.loads(line)
        if "focus" in request:  # the stepwise harness moved on: the trace on disk now ends at the new step
            helpers.load_trace(Trace.load(trace_dir), int(request["focus"]))  # `recording` grows in place
            namespace["step_to_fix"] = builtins["step_to_fix"] = helpers.recording[helpers.FOCUS]
            if "recording" in builtins:
                namespace["recording"] = builtins["recording"] = helpers.recording
            protocol.write(json.dumps({"output": "", "images": []}) + "\n")
            protocol.flush()
            continue
        if "names" in request:
            shown, more = user_names(namespace, builtins)
            protocol.write(json.dumps({"names": shown, "more": more}) + "\n")
            protocol.flush()
            continue
        if "replay" in request:
            result = replay_cells(
                list(request["replay"] or []), namespace, builtins,
                float(request.get("cell_seconds") or REPLAY_CELL_SECONDS), float(request.get("total_seconds") or REPLAY_TOTAL_SECONDS),
            )
            protocol.write(json.dumps(result) + "\n")
            protocol.flush()
            continue
        output = _run(request["code"], namespace, builtins)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[: MAX_OUTPUT_CHARS // 2] + "\n...[output truncated]...\n" + output[-MAX_OUTPUT_CHARS // 2 :]
        shown = helpers.take_shown()
        protocol.write(json.dumps({"output": output, "images": shown}) + "\n")
        protocol.flush()
    return 0


class KernelClient:
    """Parent side of the kernel: start, execute with a timeout, answer edit/undo requests, restart.

    focus: the step to fix in the stepwise harness (the kernel then shows `step_to_fix` instead of
    `recording`); history: with focus, also `recording`, the recording so far (steps 0..focus), and
    summarize_levels.

    editor: what applies edit_file()/undo_edit() (an engine_files.EngineEditor; by default one with
    versions in <workspace>/../engine_versions). After execute(), ``last_images`` holds what
    show_frames() made: a list of (PNG bytes, caption)."""

    def __init__(
        self,
        workspace: Path,
        trace_dir: Path,
        timeout: float = 120.0,
        editor: Any = None,
        images: bool = True,
        log: Callable[[dict], None] | None = None,
        focus: int | None = None,
        history: bool = False,
        play: bool = False,
    ):
        from engine_re.engine_files import EngineEditor

        self.workspace = Path(workspace).resolve()
        self.trace_dir = Path(trace_dir).resolve()
        self.timeout = timeout
        self.images = images
        self.focus = focus
        self.history = history
        self.play = play
        self.editor = editor or EngineEditor(self.workspace / "engine.py", self.workspace.parent / "engine_versions", self.workspace.parent, log)
        self.proc: subprocess.Popen | None = None
        self.last_images: list[tuple[bytes, str]] = []

    def start(self) -> None:
        cmd = [sys.executable, "-m", "engine_re.kernel", str(self.workspace), str(self.trace_dir)]
        if not self.images:
            cmd.append("--no-images")
        if self.focus is not None:
            cmd += ["--focus", str(self.focus)] + (["--history"] if self.history else []) + (["--play"] if self.play else [])
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.workspace,
            env=sandbox_env(str(self.workspace)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_log(),
            text=True,
            bufsize=1,
        )

    def _stderr_log(self):
        log_dir = self.workspace.parent
        return open(log_dir / "kernel_stderr.log", "a", encoding="utf-8")  # noqa: SIM115

    def refocus(self, focus: int) -> None:
        """Stepwise harness: the trace on disk now ends at step `focus`; a running kernel reloads it,
        moves `step_to_fix` there and grows `recording` to it, keeping its variables; a kernel started
        later starts there."""
        self.focus = focus
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None or self.proc.stdout is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"focus": focus}) + "\n")
            self.proc.stdin.flush()
            ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
            if not ready or not self.proc.stdout.readline():
                self.stop()
        except (BrokenPipeError, OSError):
            self.stop()

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def execute(self, code: str) -> str:
        self.last_images = []
        message = self._request({"code": code}, self.timeout)
        if "error" in message:
            return message["error"]
        for item in message.get("images") or []:
            try:
                self.last_images.append((base64.b64decode(item["png"]), str(item.get("caption", ""))))
            except (KeyError, ValueError, TypeError):
                continue
        return message.get("output", "")

    def names(self) -> tuple[list[str], int]:
        """What the model has defined in the kernel: "name: summary" entries (at most NAMES_SHOWN) and how
        many more there are; nothing when the kernel cannot answer."""
        message = self._request({"names": True}, self.timeout)
        return list(message.get("names") or []), int(message.get("more") or 0)

    def replay(self, cells: list[dict[str, Any]], cell_seconds: float = REPLAY_CELL_SECONDS,
               total_seconds: float = REPLAY_TOTAL_SECONDS) -> dict[str, Any]:
        """Re-run earlier python cells ({"turn", "code"}) in the kernel's replay mode (see the module). Returns
        {"replayed", "failed": [{"turn", "error"}], "skipped", "seconds"}; "error" says why when the kernel
        could not do it."""
        if not cells:
            return {"replayed": 0, "failed": [], "skipped": 0, "seconds": 0.0}
        request = {"replay": cells, "cell_seconds": cell_seconds, "total_seconds": total_seconds}
        message = self._request(request, total_seconds + max(30.0, cell_seconds))
        if "error" in message:
            return {"replayed": 0, "failed": [], "skipped": len(cells), "seconds": 0.0, "error": message["error"]}
        return message

    def _request(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send one request and return the kernel's reply, answering its edit/undo requests meanwhile; on a
        kernel that dies, hangs or breaks the protocol, {"error": text} and the kernel is restarted next time."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return {"error": "The Python kernel had died; it was restarted and all variables were lost. Run your code again."}
        deadline = time.time() + timeout
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], max(0.0, deadline - time.time()))
            if not ready:
                self.stop()
                return {"error": f"Timed out after {timeout:g}s. The kernel was restarted and all variables were lost."}
            line = self.proc.stdout.readline()
            if not line:
                log = self.workspace.parent / "kernel_stderr.log"
                tail = log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else ""
                self.stop()
                return {"error": tail + "\nThe Python kernel crashed (out of memory or a fatal error); it was restarted and all variables were lost."}
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.stop()
                return {"error": "Kernel protocol error; the kernel was restarted and all variables were lost."}
            if "rpc" in message:
                reply = self.editor.handle(message["rpc"])
                try:
                    self.proc.stdin.write(json.dumps(reply) + "\n")
                    self.proc.stdin.flush()
                except BrokenPipeError:
                    pass
                continue
            return message


if __name__ == "__main__":
    raise SystemExit(main())
