"""Sandbox for code the model writes: an audit hook (PEP 578) installed in the
analysis kernel and in the process that runs the candidate engine.

Once installed it cannot be removed. It:

- allows reads only under the Python installation, the system directories and
  the roots the caller passes (the workspace, the trace), so the real game
  sources, the repo and the credentials stay unreadable;
- allows writes only under the given write roots, and none to the protected
  files (engine.py in the kernel, which changes it only through the harness);
- blocks starting processes and opening network connections, so the model
  cannot shell out or download the game files.

This keeps an honest model honest; it is not a defence against deliberate
interpreter-level escapes (ctypes, frame inspection). `scan_engine_source`
flags those patterns in the final engines for the report.
"""

from __future__ import annotations

import os
import re
import sys
import sysconfig

SYSTEM_READ_ROOTS = ("/usr", "/lib", "/lib64", "/etc", "/dev", "/proc", "/sys")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_BLOCKED_EVENTS = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.bind",
    "sys.remote_exec",
)
_READ_PATH_EVENTS = ("os.listdir", "os.scandir", "os.chdir", "glob.glob", "os.walk")
_WRITE_PATH_EVENTS = ("os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.truncate", "os.chmod", "os.chown", "os.link", "os.symlink", "shutil.rmtree", "shutil.move", "shutil.copyfile", "shutil.copytree")


def python_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(path)
    return sorted(roots)


def _norm(path: object) -> str | None:
    if isinstance(path, int):
        return None  # already-open file descriptor
    try:
        return os.path.realpath(os.fsdecode(path))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def install(read_roots: list[str], write_roots: list[str], protected: list[str] | None = None) -> None:
    """Install the guard. ``write_roots`` are readable too. ``protected`` files (engine.py in the
    kernel) stay readable but cannot be opened for writing, removed, renamed, replaced, linked or
    have a parent directory renamed, by any code in the process: the kernel changes engine.py only
    through the harness (see engine_re.kernel)."""
    writes = tuple(os.path.realpath(p) for p in write_roots)
    reads = tuple(os.path.realpath(p) for p in [*python_roots(), *SYSTEM_READ_ROOTS, *read_roots]) + writes
    guarded = tuple(os.path.realpath(p) for p in protected or [])

    def deny(what: str) -> None:
        raise PermissionError(f"sandbox: {what} is not allowed")

    def is_protected(path: str) -> bool:
        # The file itself, or a directory containing it (renaming that would move it).
        return any(path == p or p.startswith(path.rstrip(os.sep) + os.sep) for p in guarded)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = _norm(args[0])
            if path is None:
                return
            mode, flags = args[1], args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool((flags or 0) & _WRITE_FLAGS)
            if writing and not _under(path, writes):
                deny(f"writing {path}")
            if writing and guarded and path in guarded:
                deny(f"writing {path} (change it with edit() or undo())")
            if not _under(path, reads):
                deny(f"reading {path}")
        elif event in _READ_PATH_EVENTS:
            path = _norm(args[0]) if args and args[0] is not None else os.path.realpath(".")
            if path is not None and not _under(path, reads):
                deny(f"listing {path}")
        elif event in _WRITE_PATH_EVENTS:
            for k, arg in enumerate(args[:2]):
                path = _norm(arg) if isinstance(arg, (str, bytes, os.PathLike)) else None
                if path is not None and not _under(path, writes):
                    deny(f"{event} on {path}")
                copying_from = k == 0 and event in ("shutil.copyfile", "shutil.copytree")
                if path is not None and guarded and not copying_from and is_protected(path):
                    deny(f"{event} on {path} (change engine.py with edit() or undo())")
        elif event.startswith(_BLOCKED_EVENTS):
            deny(event)

    sys.addaudithook(hook)


def sandbox_env(home: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for sandboxed subprocesses: no API keys or cloud credentials."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": home,
        "TMPDIR": home,
        "MPLCONFIGDIR": home,
        "LANG": "C.UTF-8",
        "PYTHONPATH": package_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "ONLY_RESET_LEVELS": "true",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    env.update(extra or {})
    return env


_SUSPICIOUS = {
    "frame inspection": r"\b(sys\._getframe|inspect\.(stack|currentframe)|f_back|f_locals|f_globals)\b",
    "garbage-collector walk": r"\bgc\.get_(objects|referrers|referents)\b",
    "ctypes": r"\bctypes\b",
    "file access": r"\bopen\(|\bnp\.load\(|\bPath\(|\bos\.(listdir|scandir|walk)\b",
    "subprocess/network": r"\b(subprocess|socket|urllib|requests|http\.client)\b",
    "reads the real engine": r"\b(arc_agi|environment_files)\b",
}


def scan_engine_source(source: str) -> dict[str, list[int]]:
    """Line numbers of patterns that would let an engine read the answers instead
    of computing them."""
    hits: dict[str, list[int]] = {}
    for label, pattern in _SUSPICIOUS.items():
        regex = re.compile(pattern)
        lines = [i for i, line in enumerate(source.splitlines(), 1) if regex.search(line)]
        if lines:
            hits[label] = lines
    return hits
