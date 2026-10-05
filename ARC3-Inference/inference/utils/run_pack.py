"""Pack a finished run directory into a small lossless form, and unpack it.

Most of a run directory can be rebuilt from a small part of it:

- The games are deterministic, so every board in the viewer's event logs
  (``artifacts/<stem>_events.jsonl``, as numbers and as ASCII) follows from the
  actions in ``benchmark.json``, replayed through the game engine.
- The transcript text in those event logs is slices of
  ``transcripts/<stem>.txt``, and ``solver_analysis/<stem>.html`` is that
  transcript rendered as a page.

``pack_run`` keeps each event log with its boards replaced by the number of the
replayed state they equal and its transcript text by offsets, drops the
transcript pages, and compresses every other large file with xz. ``pack.json``
records the sha256 of every file it replaced. Before removing a file, pack
rebuilds it from the packed form and compares the bytes; a file that does not
rebuild exactly is left as it is. ``unpack_run`` rebuilds the files and checks
their hashes, so an unpacked run is byte-identical to the original.

Request logs are not restored: they stay ``.jsonl.xz``, which every reader
opens. Unpacking replays the games, so it needs their files
(``environment_files/``); ``pack.json`` records their hashes.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import lzma
import os
import tarfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii
from inference.utils.run_artifacts import (
    COMPRESSED_LOG_SUFFIX,
    artifact_stem,
    compress_log,
    render_transcript_html,
)

log = logging.getLogger(__name__)

MANIFEST = "pack.json"
FORMAT = 1
HARNESS_DIR = Path(__file__).resolve().parents[2]
# Other files at least this large are compressed with xz.
MIN_COMPRESS_BYTES = 64 * 1024
# Left as they are: read without unpacking (scores, resume, run lists), or
# already compressed.
KEEP_FILES = {"benchmark.json", MANIFEST}
KEEP_DIRS = {"analyses"}
COMPRESSED_SUFFIXES = (".xz", ".gz", ".zip", ".mp4", ".png", ".jpg", ".gif", ".webp")
SRC_DIR = "src"
SRC_ARCHIVE = "src.tar.xz"
EVENTS_SUFFIX = "_events.jsonl"
PACKED_EVENTS_SUFFIX = "_events.jsonl.pack.xz"
# Rebuild order: the transcripts (xz) come before the pages and events built from them.
METHOD_ORDER = {"xz": 0, "tar": 0, "transcript_html": 1, "events": 2}

_unpack_lock = threading.Lock()
_ENGINE_LOG = logging.getLogger(__name__ + ".replay")
_ENGINE_LOG.setLevel(logging.WARNING)


class PackError(RuntimeError):
    pass


@dataclass
class PackReport:
    packed: dict[str, str] = field(default_factory=dict)  # path -> method
    kept: dict[str, str] = field(default_factory=dict)  # path -> why it was left as is
    compressed_logs: list[str] = field(default_factory=list)
    bytes_before: int = 0
    bytes_after: int = 0


@dataclass
class _GameRun:
    index: int
    game_id: str
    stem: str
    history: list[dict[str, Any]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _xz(data: bytes) -> bytes:
    return lzma.compress(data, preset=6)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def _dir_bytes(run_dir: Path) -> int:
    return sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / MANIFEST
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_packed(run_dir: Path) -> bool:
    return (Path(run_dir) / MANIFEST).exists()


def packed_sources(run_dir: Path) -> set[Path]:
    """The files that hold packed data (to skip when copying a run's files)."""
    manifest = load_manifest(run_dir)
    if manifest is None:
        return set()
    sources = {Path(run_dir) / MANIFEST}
    for entry in manifest["files"].values():
        if entry["method"] in ("xz", "tar", "events"):
            sources.add(Path(run_dir) / entry["source"])
    return sources


def _environments_dir(value: str | Path | None) -> Path:
    if value:
        return Path(value)
    configured = os.environ.get("ENVIRONMENTS_DIR", "").strip()
    return Path(configured) if configured else HARNESS_DIR / "environment_files"


def _game_file(environments_dir: Path, game_id: str) -> Path:
    # arc_agi stores downloads as {environments_dir}/{base_id}/{version}/{base_id}.py
    base_id, _, version = game_id.partition("-")
    return environments_dir / base_id / version / f"{base_id}.py"


def _game_runs(run_dir: Path) -> list[_GameRun]:
    path = run_dir / "benchmark.json"
    if not path.exists():
        return []
    runs = json.loads(path.read_text(encoding="utf-8"))["game_runs"]
    game_ids = list(dict.fromkeys(run["game_id"] for run in runs))
    # named as HarnessSolver._run_stem names them
    return [
        _GameRun(
            index=position,
            game_id=run["game_id"],
            stem=f"{artifact_stem(run['game_id'])}_p{position // len(game_ids)}",
            history=list(run.get("history") or []),
        )
        for position, run in enumerate(runs)
    ]


@contextmanager
def _replay_settings() -> Iterator[None]:
    # As taaf.game_api sets it for every run: RESET restarts the current level.
    previous = os.environ.get("ONLY_RESET_LEVELS")
    os.environ["ONLY_RESET_LEVELS"] = "true"
    # arc_agi's scorecard module logs each scorecard it creates
    scorecard_log = logging.getLogger("arc_agi.scorecard")
    level = scorecard_log.level
    scorecard_log.setLevel(logging.WARNING)
    try:
        yield
    finally:
        scorecard_log.setLevel(level)
        if previous is None:
            os.environ.pop("ONLY_RESET_LEVELS", None)
        else:
            os.environ["ONLY_RESET_LEVELS"] = previous


def replay_boards(
    game_id: str, history: list[dict[str, Any]], environments_dir: Path
) -> list[list[list[int]]]:
    """The visible board after reset and after each recorded action."""
    import arc_agi
    import numpy as np
    from arcengine import GameAction

    with _replay_settings():
        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=str(environments_dir),
            # without one, Arcade logs every game it loads to stdout
            logger=_ENGINE_LOG,
        )
        env = arcade.make(game_id)
        if env is None:
            raise PackError(f"{game_id}: game not found in {environments_dir}")
        frame = env.reset()
        boards = [np.asarray(frame.frame[-1]).tolist()]
        for record in history:
            action = record["action"]
            frame = env.step(GameAction[action["id"]], data=action.get("data") or {})
            if frame is None:
                raise PackError(f"{game_id}: the engine rejected action {len(boards)}")
            boards.append(np.asarray(frame.frame[-1]).tolist())
    return boards


def _board_key(board: list[list[int]]) -> str:
    return json.dumps(board, separators=(",", ":"))


def _dumps(event: Any) -> str:
    # as inference.utils.viewer_artifacts.append_raw_events_sidecar writes them
    return json.dumps(event, separators=(",", ":"))


def _placeholder(value: Any, key: str) -> Any:
    return value[key] if isinstance(value, dict) and set(value) == {key} else None


def _unpack_event_line(line: str, boards: list[list[list[int]]] | None, transcript: str | None) -> str:
    packed = json.loads(line)
    literal = _placeholder(packed, "$line")
    if literal is not None:
        return literal
    event = dict(packed)
    state = _placeholder(event.get("board"), "$state")
    if state is not None:
        if boards is None:
            raise PackError("event log needs the replayed boards")
        event["board"] = boards[state]
    if _placeholder(event.get("board_ascii"), "$ascii") is not None:
        event["board_ascii"] = format_grid_ascii(event["board"])
    span = _placeholder(event.get("transcript"), "$slice")
    if span is not None:
        if transcript is None:
            raise PackError("event log needs its transcript")
        event["transcript"] = transcript[span[0] : span[1]]
    return _dumps(event)


def _pack_events(text: str, boards: list[list[list[int]]], transcript: str | None) -> str:
    """The event log with boards and transcript text replaced by references.

    A line that does not rebuild exactly from its references is kept literally.
    """
    state_of: dict[str, int] = {}
    for number, board in enumerate(boards):
        state_of.setdefault(_board_key(board), number)
    cursor = 0
    packed_lines = []
    for line in text.split("\n"):
        packed: dict[str, Any] | None = None
        try:
            event = json.loads(line) if line else None
        except ValueError:
            event = None
        if isinstance(event, dict) and "$line" not in event:
            packed = dict(event)
            board = event.get("board")
            if isinstance(board, list):
                state = state_of.get(_board_key(board))
                if state is not None:
                    packed["board"] = {"$state": state}
                if event.get("board_ascii") and event.get("board_ascii") == format_grid_ascii(board):
                    packed["board_ascii"] = {"$ascii": True}
            chunk = event.get("transcript")
            if transcript is not None and isinstance(chunk, str) and chunk:
                start = transcript.find(chunk, cursor)
                if start < 0:
                    start = transcript.find(chunk)
                if start >= 0:
                    packed["transcript"] = {"$slice": [start, start + len(chunk)]}
                    cursor = start + len(chunk)
            if _unpack_event_line(_dumps(packed), boards, transcript) != line:
                packed = None
        packed_lines.append(_dumps(packed if packed is not None else {"$line": line}))
    return "\n".join(packed_lines)


def _unpack_events(packed: str, boards: list[list[list[int]]] | None, transcript: str | None) -> str:
    return "\n".join(_unpack_event_line(line, boards, transcript) for line in packed.split("\n"))


def _src_archive(files: list[tuple[str, bytes]]) -> bytes:
    """A reproducible tar.xz of the given files (no times, owners or modes)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return _xz(buffer.getvalue())


def _src_members(archive: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(archive)), mode="r") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
            if member.isfile()
        }


def pack_run(run_dir: str | Path, *, environments_dir: str | Path | None = None) -> PackReport:
    """Pack a finished run in place. Packing a packed run removes files unpacked since."""
    run_dir = Path(run_dir)
    report = PackReport(bytes_before=_dir_bytes(run_dir))
    manifest = load_manifest(run_dir)
    if manifest is not None:
        _remove_unpacked(run_dir, manifest, report)
        report.bytes_after = _dir_bytes(run_dir)
        return report

    game_runs = _game_runs(run_dir)
    benchmark = run_dir / "benchmark.json"
    if benchmark.exists():
        states = [r.get("state") for r in json.loads(benchmark.read_text(encoding="utf-8"))["game_runs"]]
        if "playing" in states:
            raise PackError(f"{run_dir}: a game run is still playing; pack it when the run has ended")

    for path in sorted(run_dir.glob("*requests.jsonl")):
        compress_log(path)
        report.compressed_logs.append(path.name + COMPRESSED_LOG_SUFFIX)

    environments = _environments_dir(environments_dir)
    entries: dict[str, dict[str, Any]] = {}
    sources: dict[str, bytes] = {}  # packed path -> bytes to write
    games: dict[str, dict[str, str]] = {}

    def add(path: Path, data: bytes, method: str, rebuilt: bytes, **extra: Any) -> bool:
        relative = path.relative_to(run_dir).as_posix()
        if rebuilt != data:
            report.kept[relative] = f"{method}: rebuilt file differs; compressed whole if large"
            return False
        entries[relative] = {"sha256": _sha256(data), "bytes": len(data), "method": method, **extra}
        report.packed[relative] = method
        return True

    for game_run in game_runs:
        transcript_path = run_dir / "transcripts" / f"{game_run.stem}.txt"
        transcript = (
            transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else None
        )
        page = run_dir / "solver_analysis" / f"{game_run.stem}.html"
        if transcript is not None and page.exists():
            title = f"{game_run.game_id} analysis"
            add(
                page,
                page.read_bytes(),
                "transcript_html",
                render_transcript_html(transcript, title).encode("utf-8"),
                source=transcript_path.relative_to(run_dir).as_posix(),
                title=title,
            )
        events = run_dir / "artifacts" / f"{game_run.stem}{EVENTS_SUFFIX}"
        game_file = _game_file(environments, game_run.game_id)
        if events.exists() and game_file.exists():
            data = events.read_bytes()
            try:
                boards = replay_boards(game_run.game_id, game_run.history, environments)
            except Exception as exc:  # noqa: BLE001  (any replay failure: compress instead)
                report.kept[events.relative_to(run_dir).as_posix()] = (
                    f"replay failed ({exc}); compressed whole if large"
                )
                continue
            packed = _pack_events(data.decode("utf-8"), boards, transcript).encode("utf-8")
            rebuilt = _unpack_events(packed.decode("utf-8"), boards, transcript).encode("utf-8")
            source = events.with_name(f"{game_run.stem}{PACKED_EVENTS_SUFFIX}")
            if add(
                events,
                data,
                "events",
                rebuilt,
                source=source.relative_to(run_dir).as_posix(),
                game_run=game_run.index,
                transcript=transcript_path.relative_to(run_dir).as_posix() if transcript else None,
            ):
                sources[entries[events.relative_to(run_dir).as_posix()]["source"]] = _xz(packed)
                games[game_run.game_id] = {
                    "file": game_file.relative_to(environments).as_posix(),
                    "sha256": _sha256(game_file.read_bytes()),
                }

    src = run_dir / SRC_DIR
    if src.is_dir():
        files = sorted(
            (path.relative_to(run_dir).as_posix(), path.read_bytes())
            for path in src.rglob("*")
            if path.is_file()
        )
        archive = _src_archive(files)
        members = _src_members(archive)
        for name, data in files:
            add(run_dir / name, data, "tar", members.get(name, b""), source=SRC_ARCHIVE)
        sources[SRC_ARCHIVE] = archive

    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if (
            not path.is_file()
            or relative.as_posix() in entries
            or relative.as_posix() in KEEP_FILES
            or relative.parts[0] in KEEP_DIRS | {SRC_DIR}
            or path.name.endswith(COMPRESSED_SUFFIXES)
            or path.stat().st_size < MIN_COMPRESS_BYTES
        ):
            continue
        data = path.read_bytes()
        packed = _xz(data)
        source = relative.as_posix() + COMPRESSED_LOG_SUFFIX
        if add(path, data, "xz", lzma.decompress(packed), source=source):
            sources[source] = packed

    if not entries:
        report.bytes_after = _dir_bytes(run_dir)
        return report
    for relative, data in sources.items():
        _write_atomic(run_dir / relative, data)
    manifest = {
        "format": FORMAT,
        "arcengine": metadata.version("arcengine"),
        "games": games,
        "files": entries,
    }
    _write_atomic(run_dir / MANIFEST, (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    _remove_unpacked(run_dir, manifest, report)
    report.bytes_after = _dir_bytes(run_dir)
    return report


def _remove_unpacked(run_dir: Path, manifest: dict[str, Any], report: PackReport) -> None:
    for relative, entry in manifest["files"].items():
        path = run_dir / relative
        if not path.exists():
            continue
        if _sha256(path.read_bytes()) != entry["sha256"]:
            raise PackError(f"{path} changed since the run was packed; not removing it")
        path.unlink()
        report.packed.setdefault(relative, entry["method"])
    src = run_dir / SRC_DIR
    for directory in sorted((p for p in src.rglob("*") if p.is_dir()), reverse=True) if src.is_dir() else []:
        if not any(directory.iterdir()):
            directory.rmdir()
    if src.is_dir() and not any(src.iterdir()):
        src.rmdir()


def unpack_run(run_dir: str | Path, *, environments_dir: str | Path | None = None) -> list[str]:
    """Rebuild the files a packed run left out. Returns the rebuilt paths."""
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    if manifest is None:
        return []
    missing = {
        relative: entry
        for relative, entry in manifest["files"].items()
        if not (run_dir / relative).exists()
    }
    if not missing:
        return []
    environments = _environments_dir(environments_dir)
    game_runs = {game_run.index: game_run for game_run in _game_runs(run_dir)}
    src_members: dict[str, bytes] | None = None
    for relative, entry in sorted(missing.items(), key=lambda item: METHOD_ORDER[item[1]["method"]]):
        method = entry["method"]
        if method == "xz":
            data = lzma.decompress((run_dir / entry["source"]).read_bytes())
        elif method == "tar":
            if src_members is None:
                src_members = _src_members((run_dir / entry["source"]).read_bytes())
            data = src_members[relative]
        elif method == "transcript_html":
            text = (run_dir / entry["source"]).read_text(encoding="utf-8")
            data = render_transcript_html(text, entry["title"]).encode("utf-8")
        elif method == "events":
            game_run = game_runs[entry["game_run"]]
            game = manifest["games"][game_run.game_id]
            game_file = environments / game["file"]
            if not game_file.exists():
                raise PackError(
                    f"{relative}: needs {game_file} to replay the game; get it with "
                    f"`uv run --no-sync python scripts/fetch_games.py {game_run.game_id}`"
                )
            if _sha256(game_file.read_bytes()) != game["sha256"]:
                raise PackError(f"{relative}: {game_file} is not the game version this run played")
            boards = replay_boards(game_run.game_id, game_run.history, environments)
            transcript = (
                (run_dir / entry["transcript"]).read_text(encoding="utf-8")
                if entry.get("transcript")
                else None
            )
            packed = lzma.decompress((run_dir / entry["source"]).read_bytes()).decode("utf-8")
            data = _unpack_events(packed, boards, transcript).encode("utf-8")
        else:
            raise PackError(f"{relative}: unknown pack method {method!r}")
        if _sha256(data) != entry["sha256"]:
            raise PackError(f"{relative}: rebuilt file differs from the original")
        _write_atomic(run_dir / relative, data)
    return sorted(missing)


def ensure_unpacked(path: str | Path, *, environments_dir: str | Path | None = None) -> bool:
    """Unpack the packed run that contains `path`, if any file is missing.

    Returns False when the run could not be unpacked; the reason is logged.
    """
    for candidate in [Path(path), *Path(path).parents][:4]:
        if (candidate / MANIFEST).exists():
            with _unpack_lock:
                try:
                    restored = unpack_run(candidate, environments_dir=environments_dir)
                except (PackError, OSError, ValueError) as exc:
                    log.warning("could not unpack %s: %s", candidate, exc)
                    return False
            if restored:
                log.info("unpacked %d files in %s", len(restored), candidate)
            return True
    return True
