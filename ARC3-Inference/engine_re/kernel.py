"""The agent's persistent Python kernel.

    python -m engine_re.kernel WORKSPACE TRACE_DIR [--no-images] [--focus K [--history]]

Reads one JSON request per line on stdin ({"code": ...}), runs it in a
namespace that persists between requests (a {"focus": k} request, from the stepwise harness, reloads
the trace, which now holds steps 0..k, and moves `step` and S to it, keeping everything else), and writes one JSON reply per line
({"output": ..., "images": [...]}). As in a notebook, the value of a final
expression is printed. "images" holds the pictures show() made during the
request (base64 PNG and caption), for the harness to attach.

The kernel runs sandboxed (engine_re.guard): it can read the workspace and the
trace, write only the workspace, and cannot start processes or open
connections. It cannot write engine.py at all: edit() and undo() send their
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
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from engine_re.guard import sandbox_env

MAX_OUTPUT_CHARS = 200_000
# What the namespace of the model's code starts with (besides np and the fixed-block classes).
PRELOADED = ("S", "read", "edit", "undo", "render", "show", "try_step", "auto_sprites", "summarize_levels")
# Names the model's code may not rebind: the built-in functions, the recording and the fixed-block classes.
RESERVED = PRELOADED + ("Sprite", "Action", "View", "State")
# The stepwise harness (--focus K): `step`, the step to fix, instead of the recording, and no summarize_levels;
# with --history also S, the recording so far (steps 0..K, all the trace on disk holds), and summarize_levels.
PRELOADED_STEP = ("step", "read", "edit", "undo", "render", "show", "try_step", "auto_sprites")
RESERVED_STEP = PRELOADED_STEP + ("Sprite", "Action", "View", "State")
PRELOADED_HISTORY = PRELOADED + ("step",)
RESERVED_HISTORY = PRELOADED_HISTORY + ("Sprite", "Action", "View", "State")


def reserved_bindings(tree: ast.AST, reserved: tuple[str, ...] = RESERVED) -> list[tuple[str, int, str]]:
    """Where the code binds a reserved name: (name, line, how), e.g. ("show", 4, "def show")."""
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
        elif isinstance(node, ast.alias):
            bound = node.asname or node.name.split(".")[0]
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
    return (
        f"Error: nothing was run. This code would replace the harness's built-in {names} ({where}).\n"
        f"These names are reserved: {', '.join(reserved)}. Give your own functions and variables other names.\n"
    )


def _run(code: str, namespace: dict[str, Any], builtins: dict[str, Any] | None = None) -> str:
    """Run the model's code in `namespace`. `builtins` (name -> object) are the reserved names: code
    that binds one is rejected before it runs, and any that were changed anyway are put back."""
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
        finally:
            changed = [name for name, value in (builtins or {}).items() if namespace.get(name) is not value]
            for name in changed:
                namespace[name] = builtins[name]
            if changed:
                print(f"[harness] Your code replaced the built-in {', '.join(changed)}; restored. These names are reserved.")
    return buffer.getvalue()


def main() -> int:
    workspace, trace_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    images = "--no-images" not in sys.argv[3:]
    focus = int(sys.argv[sys.argv.index("--focus") + 1]) if "--focus" in sys.argv[3:] else None
    history = "--history" in sys.argv[3:]
    import numpy as np

    import scipy.ndimage  # noqa: F401
    from engine_re import game_api, guard, helpers
    from engine_re.trace import Trace

    np.set_printoptions(linewidth=200, threshold=4096)
    helpers.trace = Trace.load(trace_dir)
    helpers.S = helpers.trace.steps
    helpers.ENGINE_PATH = workspace / "engine.py"
    helpers.IMAGES = images
    api = game_api.canonical()
    namespace: dict[str, Any] = {"__name__": "__main__", "np": np}
    namespace.update({name: getattr(api, name) for name in ("Sprite", "Action", "View", "State")})
    if focus is None:
        namespace.update({name: getattr(helpers, name) for name in PRELOADED})
        reserved = RESERVED
    else:
        helpers.FOCUS = focus
        names = PRELOADED_HISTORY if history else PRELOADED_STEP
        namespace["step"] = helpers.StepView(helpers.trace, focus)
        namespace.update({name: getattr(helpers, name) for name in names if name != "step"})
        reserved = RESERVED_HISTORY if history else RESERVED_STEP
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
            helpers.trace = Trace.load(trace_dir)
            helpers.S = helpers.trace.steps
            helpers.FOCUS = int(request["focus"])
            namespace["step"] = builtins["step"] = helpers.StepView(helpers.trace, helpers.FOCUS)
            if "S" in builtins:
                namespace["S"] = builtins["S"] = helpers.S
            protocol.write(json.dumps({"output": "", "images": []}) + "\n")
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
    """Parent side of the kernel: start, execute with a timeout, answer edit/undo, restart.

    focus: the step to fix in the stepwise harness (the kernel then shows `step` instead of S);
    history: with focus, also S, the recording so far (steps 0..focus), and summarize_levels.

    editor: what applies edit()/undo() (an engine_files.EngineEditor; by default one with versions
    in <workspace>/../engine_versions). After execute(), ``last_images`` holds what show() made:
    a list of (PNG bytes, caption)."""

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
    ):
        from engine_re.engine_files import EngineEditor

        self.workspace = Path(workspace).resolve()
        self.trace_dir = Path(trace_dir).resolve()
        self.timeout = timeout
        self.images = images
        self.focus = focus
        self.history = history
        self.editor = editor or EngineEditor(self.workspace / "engine.py", self.workspace.parent / "engine_versions", self.workspace.parent, log)
        self.proc: subprocess.Popen | None = None
        self.last_images: list[tuple[bytes, str]] = []

    def start(self) -> None:
        cmd = [sys.executable, "-m", "engine_re.kernel", str(self.workspace), str(self.trace_dir)]
        if not self.images:
            cmd.append("--no-images")
        if self.focus is not None:
            cmd += ["--focus", str(self.focus)] + (["--history"] if self.history else [])
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
        """Stepwise harness: the trace on disk now ends at step `focus`; a running kernel reloads it and
        moves `step` (and S) there, keeping its variables; a kernel started later starts there."""
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
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return "The Python kernel had died; it was restarted and all variables were lost. Run your code again."
        deadline = time.time() + self.timeout
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], max(0.0, deadline - time.time()))
            if not ready:
                self.stop()
                return f"Timed out after {self.timeout:g}s. The kernel was restarted and all variables were lost."
            line = self.proc.stdout.readline()
            if not line:
                log = self.workspace.parent / "kernel_stderr.log"
                tail = log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else ""
                self.stop()
                return tail + "\nThe Python kernel crashed (out of memory or a fatal error); it was restarted and all variables were lost."
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self.stop()
                return "Kernel protocol error; the kernel was restarted and all variables were lost."
            if "rpc" in message:
                reply = self.editor.handle(message["rpc"])
                try:
                    self.proc.stdin.write(json.dumps(reply) + "\n")
                    self.proc.stdin.flush()
                except BrokenPipeError:
                    pass
                continue
            for item in message.get("images") or []:
                try:
                    self.last_images.append((base64.b64decode(item["png"]), str(item.get("caption", ""))))
                except (KeyError, ValueError, TypeError):
                    continue
            return message.get("output", "")


if __name__ == "__main__":
    raise SystemExit(main())
