"""The agent's persistent Python kernel.

    python -m engine_re.kernel WORKSPACE TRACE_DIR

Reads one JSON request per line on stdin ({"code": ...}), runs it in a
namespace that persists between requests, and writes one JSON reply per line
({"output": ...}). As in a notebook, the value of a final expression is
printed. The kernel runs sandboxed (engine_re.guard): it can read the
workspace and the trace, write only the workspace, and cannot start processes
or open connections.

``KernelClient`` is the parent side: it starts the kernel, sends code, enforces
a timeout and restarts the kernel when it hangs or dies.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import select
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from engine_re.guard import sandbox_env

MAX_OUTPUT_CHARS = 200_000


def _run(code: str, namespace: dict[str, Any]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            tree = ast.parse(code, "<python>", "exec")
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
    return buffer.getvalue()


def main() -> int:
    workspace, trace_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    import numpy as np

    import arcengine  # noqa: F401
    import scipy.ndimage  # noqa: F401
    from engine_re import guard, helpers
    from engine_re.trace import Trace

    np.set_printoptions(linewidth=200, threshold=4096)
    helpers.trace = Trace.load(trace_dir)
    helpers.S = helpers.trace.steps
    helpers.ENGINE_PATH = workspace / "engine.py"
    namespace: dict[str, Any] = {"__name__": "__main__", "np": np}
    namespace.update({k: v for k, v in vars(helpers).items() if not k.startswith("_") and k not in ("annotations",)})
    os.chdir(workspace)
    # Replies go on a private copy of stdout; fd 1 itself goes to /dev/null so
    # code that writes to it directly cannot corrupt the protocol.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    guard.install(read_roots=[str(trace_dir)], write_roots=[str(workspace)])

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        output = _run(request["code"], namespace)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[: MAX_OUTPUT_CHARS // 2] + "\n...[output truncated]...\n" + output[-MAX_OUTPUT_CHARS // 2 :]
        protocol.write(json.dumps({"output": output}) + "\n")
        protocol.flush()
    return 0


class KernelClient:
    """Parent side of the kernel: start, execute with a timeout, restart."""

    def __init__(self, workspace: Path, trace_dir: Path, timeout: float = 120.0):
        self.workspace = Path(workspace).resolve()
        self.trace_dir = Path(trace_dir).resolve()
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "engine_re.kernel", str(self.workspace), str(self.trace_dir)],
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

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def execute(self, code: str) -> str:
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return "The Python kernel had died; it was restarted and all variables were lost. Run your code again."
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
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
            return json.loads(line)["output"]
        except (json.JSONDecodeError, KeyError):
            self.stop()
            return "Kernel protocol error; the kernel was restarted and all variables were lost."


if __name__ == "__main__":
    raise SystemExit(main())
