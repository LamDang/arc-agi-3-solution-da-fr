"""Helpers for per-run artifact directories, git metadata, and file logging."""
from __future__ import annotations

import html
import logging
import lzma
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TextIO


log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_DIR_NAME_RE = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})(?:_(?P<suffix>\d{2}))?(?:(?P<legacy>_run)|_(?P<label>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)



def _match_run_dir_name(name: str) -> re.Match[str] | None:
    match = _RUN_DIR_NAME_RE.fullmatch(name)
    if match is None:
        return None
    if str(match.group("label") or "").lower() == "run":
        return None
    return match



def is_run_dir_name(name: str) -> bool:
    """Return whether a directory name matches the current run format."""
    return _match_run_dir_name(name) is not None


def is_selectable_run_dir_name(name: str) -> bool:
    """Return whether a run directory should participate in automatic discovery."""
    match = _match_run_dir_name(name)
    return match is not None and match.group("legacy") is None


def run_dir_sort_key(path_or_name: str | Path) -> tuple[str, int, str]:
    """Return a deterministic sort key for run directory names."""
    name = Path(path_or_name).name
    match = _match_run_dir_name(name)
    if match is None:
        raise ValueError(f"Unsupported run directory name: {name!r}")
    return (
        match.group("timestamp"),
        int(match.group("suffix") or 0),
        0 if match.group("legacy") else 1,
        match.group("label") or "",
    )


def sanitize_run_name(name: str | None) -> str:
    """Normalize a user-provided run name for filesystem-safe run directories."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("._-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = re.sub(r"_{2,}", "_", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    if not normalized:
        return ""
    if normalized.lower() == "run":
        return "named-run"
    return normalized


def get_git_info() -> tuple[str, str]:
    """Return the current git commit hash and working tree diff."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        text=True,
    ).strip()
    try:
        diff = subprocess.check_output(
            ["git", "diff"],
            cwd=_PROJECT_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        diff = ""
    return commit, diff


def save_git_info(base_dir: Path) -> Path:
    """Save git commit and diff to a file inside the run directory."""
    path = base_dir / "git_info.txt"
    try:
        commit, diff = get_git_info()
        path.write_text(f"commit: {commit}\n\n{diff}", encoding="utf-8")
    except Exception as exc:
        log.warning("failed to capture git info: %s", exc)
        path.write_text(f"git info unavailable: {exc}\n", encoding="utf-8")
    return path


def setup_experiment_directory(base_output_dir: str | Path = "runs", *, run_name: str | None = None) -> tuple[Path, Path]:
    """Create a timestamped run directory and save git metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = sanitize_run_name(run_name)
    root_dir = Path(base_output_dir)
    attempt_index = 0
    while True:
        if label:
            candidate_name = f"{timestamp}_{label}" if attempt_index == 0 else f"{timestamp}_{attempt_index:02d}_{label}"
        else:
            candidate_name = timestamp if attempt_index == 0 else f"{timestamp}_{attempt_index:02d}"
        if (root_dir / f"{candidate_name}_run").exists():
            attempt_index += 1
            continue

        base_dir = root_dir / candidate_name
        try:
            base_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            attempt_index += 1
            continue

    log_file = base_dir / "logs.log"
    save_git_info(base_dir)
    return base_dir, log_file


def setup_logging_for_experiment(log_file_path: str | Path, fmt: str) -> Path:
    """Attach a file handler for the current run's log file."""
    log_path = Path(log_file_path)
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(root_logger.level or logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_path


# A finished game run's request log is kept as <name>.xz. Each request repeats
# the conversation so far, which xz's 8 MB window finds again: a 115 MB log
# becomes about 0.5 MB. gzip's 32 KB window misses it and only reaches 5x.
COMPRESSED_LOG_SUFFIX = ".xz"


def existing_log(path: Path) -> Path | None:
    """`path` if it exists, else its compressed copy, else None."""
    for candidate in (path, path.with_name(path.name + COMPRESSED_LOG_SUFFIX)):
        if candidate.exists():
            return candidate
    return None


def open_log(path: Path) -> TextIO:
    """Open a JSONL log for reading, plain or xz-compressed."""
    if path.name.endswith(COMPRESSED_LOG_SUFFIX):
        return lzma.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def compress_log(path: Path) -> Path | None:
    """Replace a finished log with an xz copy. Returns the copy, or None if `path` is missing."""
    if not path.exists():
        return None
    target = path.with_name(path.name + COMPRESSED_LOG_SUFFIX)
    partial = target.with_name(target.name + ".partial")
    with path.open("rb") as source, lzma.open(partial, "wb", preset=6) as sink:
        shutil.copyfileobj(source, sink, 1 << 20)
    partial.replace(target)
    path.unlink()
    return target


def artifact_stem(value: str) -> str:
    """A game id made safe for file names; game run files are <stem>_p<pass>."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def render_transcript_html(text: str, title: str) -> str:
    """The solver_analysis/ page of a transcript: the text, escaped, in a <pre>."""
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
