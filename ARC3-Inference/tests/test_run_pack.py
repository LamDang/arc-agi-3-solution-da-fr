"""Packing a run removes only what it rebuilds exactly; unpacking restores every byte."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inference.utils.grid_utils import format_grid_ascii
from inference.utils.run_artifacts import open_log, render_transcript_html
from inference.utils.run_pack import (
    MANIFEST,
    PackError,
    pack_run,
    replay_boards,
    unpack_run,
)
from inference.utils.viewer_artifacts import load_raw_events

ENVIRONMENTS = Path(__file__).resolve().parents[1] / "environment_files"
GAME_ID = "ft09-0d8bbf25"
STEM = f"{GAME_ID}_p0"
HISTORY = [
    {"action": {"id": "ACTION6", "data": {"x": 38, "y": 38}}},
    {"action": {"id": "ACTION6", "data": {"x": 38, "y": 46}}},
    {"action": {"id": "ACTION6", "data": {"x": 54, "y": 46}}},
]
TRANSCRIPT = "--- analysis_step=1 ---\n[THINKING]\nClick the tiles.\n" + "filler line\n" * 50

pytestmark = pytest.mark.skipif(
    not (ENVIRONMENTS / "ft09").exists(), reason="needs environment_files/ft09 (scripts/fetch_games.py)"
)


def _event_line(event: dict) -> str:
    # as inference.utils.viewer_artifacts.append_raw_events_sidecar writes them
    return json.dumps(event, separators=(",", ":"))


def _make_run(run_dir: Path, *, state: str = "gave_up") -> None:
    boards = replay_boards(GAME_ID, HISTORY, ENVIRONMENTS)
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "transcripts").mkdir()
    (run_dir / "solver_analysis").mkdir()
    (run_dir / "src" / "pkg").mkdir(parents=True)
    benchmark = {"game_runs": [{"game_id": GAME_ID, "state": state, "history": HISTORY}]}
    (run_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    (run_dir / "transcripts" / f"{STEM}.txt").write_text(TRANSCRIPT, encoding="utf-8")
    (run_dir / "solver_analysis" / f"{STEM}.html").write_text(
        render_transcript_html(TRANSCRIPT, f"{GAME_ID} analysis"), encoding="utf-8"
    )
    stray = [row[:] for row in boards[0]]
    stray[0][0] = (stray[0][0] + 1) % 16  # no replayed state has this board
    events = [
        {"type": "initial", "board": boards[0], "board_ascii": format_grid_ascii(boards[0])},
        {"type": "analysis", "action_num": 1, "board": boards[0], "transcript": TRANSCRIPT[24:60]},
        *(
            {"type": "action", "action_num": n, "board": boards[n], "board_ascii": format_grid_ascii(boards[n])}
            for n in (1, 2, 3)
        ),
        {"type": "action", "action_num": 4, "board": stray, "board_ascii": "not the board"},
    ]
    (run_dir / "artifacts" / f"{STEM}_events.jsonl").write_text(
        "".join(_event_line(event) + "\n" for event in events), encoding="utf-8"
    )
    (run_dir / "artifacts" / f"{STEM}_viewer_data.json").write_text("{}", encoding="utf-8")
    (run_dir / "intermediate_states.pkl").write_bytes(bytes(range(256)) * 400)
    (run_dir / "src" / "pkg" / "module.py").write_text("print('ran')\n", encoding="utf-8")
    (run_dir / f"{STEM}_requests.jsonl").write_text('{"event": "request"}\n', encoding="utf-8")


def _hashes(run_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_pack_then_unpack_restores_every_file(tmp_path: Path) -> None:
    _make_run(tmp_path)
    before = _hashes(tmp_path)
    log_text = (tmp_path / f"{STEM}_requests.jsonl").read_text(encoding="utf-8")

    report = pack_run(tmp_path, environments_dir=ENVIRONMENTS)

    assert report.kept == {}
    assert report.packed == {
        f"solver_analysis/{STEM}.html": "transcript_html",
        f"artifacts/{STEM}_events.jsonl": "events",
        "intermediate_states.pkl": "xz",
        "src/pkg/module.py": "tar",
    }
    assert all(not (tmp_path / path).exists() for path in report.packed)
    assert (tmp_path / "transcripts" / f"{STEM}.txt").exists()  # under 64 KB: left as is
    assert not (tmp_path / "src").exists()
    assert report.bytes_after < report.bytes_before

    restored = unpack_run(tmp_path, environments_dir=ENVIRONMENTS)

    assert sorted(restored) == sorted(report.packed)
    after = _hashes(tmp_path)
    for path, digest in before.items():
        if path.endswith("_requests.jsonl"):
            continue  # request logs stay compressed; every reader opens .jsonl.xz
        assert after[path] == digest, path
    with open_log(tmp_path / f"{STEM}_requests.jsonl.xz") as handle:
        assert handle.read() == log_text


def test_event_lines_that_do_not_replay_are_kept_literally(tmp_path: Path) -> None:
    _make_run(tmp_path)
    pack_run(tmp_path, environments_dir=ENVIRONMENTS)

    with open_log(tmp_path / "artifacts" / f"{STEM}_events.jsonl.pack.xz") as handle:
        packed = [json.loads(line) for line in handle.read().split("\n")]

    assert packed[0]["board"] == {"$state": 0} and packed[0]["board_ascii"] == {"$ascii": True}
    assert packed[1]["transcript"] == {"$slice": [24, 60]}
    assert packed[4]["board"] == {"$state": 3}
    # no replayed state has this board, and its ASCII is not the board's: both stay
    assert isinstance(packed[5]["board"], list) and packed[5]["board_ascii"] == "not the board"
    assert packed[6] == {"$line": ""}  # the final newline


def test_a_file_that_does_not_rebuild_is_left_as_is(tmp_path: Path) -> None:
    _make_run(tmp_path)
    page = tmp_path / "solver_analysis" / f"{STEM}.html"
    page.write_text(page.read_text(encoding="utf-8") + "<!-- edited -->", encoding="utf-8")

    report = pack_run(tmp_path, environments_dir=ENVIRONMENTS)

    assert f"solver_analysis/{STEM}.html" in report.kept
    assert page.exists()


def test_pack_again_removes_what_was_unpacked(tmp_path: Path) -> None:
    _make_run(tmp_path)
    first = pack_run(tmp_path, environments_dir=ENVIRONMENTS)
    unpack_run(tmp_path, environments_dir=ENVIRONMENTS)

    second = pack_run(tmp_path, environments_dir=ENVIRONMENTS)

    assert second.packed == first.packed
    assert all(not (tmp_path / path).exists() for path in first.packed)


def test_unpack_needs_the_game_version_the_run_played(tmp_path: Path) -> None:
    _make_run(tmp_path)
    pack_run(tmp_path, environments_dir=ENVIRONMENTS)
    manifest = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    manifest["games"][GAME_ID]["sha256"] = "0" * 64
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackError, match="not the game version"):
        unpack_run(tmp_path, environments_dir=ENVIRONMENTS)


def test_a_running_run_is_not_packed(tmp_path: Path) -> None:
    _make_run(tmp_path, state="playing")
    with pytest.raises(PackError, match="still playing"):
        pack_run(tmp_path, environments_dir=ENVIRONMENTS)


def test_the_viewer_loader_unpacks_a_packed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_run(tmp_path)
    expected = (tmp_path / "artifacts" / f"{STEM}_events.jsonl").read_text(encoding="utf-8")
    pack_run(tmp_path, environments_dir=ENVIRONMENTS)
    monkeypatch.setenv("ENVIRONMENTS_DIR", str(ENVIRONMENTS))

    events = load_raw_events({}, viewer_data_path=tmp_path / "artifacts" / f"{STEM}_viewer_data.json")

    assert [event["type"] for event in events] == ["initial", "analysis", "action", "action", "action", "action"]
    assert (tmp_path / "artifacts" / f"{STEM}_events.jsonl").read_text(encoding="utf-8") == expected
