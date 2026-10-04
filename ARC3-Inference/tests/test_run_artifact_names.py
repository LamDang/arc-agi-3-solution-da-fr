"""A game run's request and prompt logs keep one name for the whole run."""
from __future__ import annotations

from pathlib import Path

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
from inference.agent.tool_agent import _resolve_prompt_log_path, _resolve_request_log_path
from viewer import data as viewer_data


def _write_state_file(artifacts: Path, run_stem: str) -> Path:
    # named as HarnessSolver._play_one names it
    path = artifacts / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
    path.write_text("{}", encoding="utf-8")
    return path


def _log_paths(state_path: Path) -> tuple[Path, Path]:
    return _resolve_request_log_path(state_path), _resolve_prompt_log_path(state_path)


def test_logs_stay_per_game_after_other_games_finish(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    ls20 = _write_state_file(artifacts, "ls20-9607627b_p0")
    sp80 = _write_state_file(artifacts, "sp80-589a99af_p0")
    expected = (
        tmp_path / "ls20-9607627b_p0_requests.jsonl",
        tmp_path / "prompts" / "ls20-9607627b_p0.log",
    )
    assert _log_paths(ls20) == expected

    # the solver deletes a game's state file when that game finishes,
    # leaving ls20's as the only one
    sp80.unlink()
    assert _log_paths(ls20) == expected


def test_single_game_run_uses_per_game_logs(tmp_path: Path) -> None:
    # also the first game of a multi-game run, before the others start
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    state_path = _write_state_file(artifacts, "ls20-9607627b_p0")
    assert _log_paths(state_path) == (
        tmp_path / "ls20-9607627b_p0_requests.jsonl",
        tmp_path / "prompts" / "ls20-9607627b_p0.log",
    )


def test_state_file_not_named_for_a_game_run_uses_run_level_logs(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_state_file(artifacts, "sp80-589a99af_p0")
    assert _log_paths(artifacts / RUNTIME_STATE_FILENAME) == (
        tmp_path / "requests.jsonl",
        tmp_path / "prompts" / "prompt.log",
    )
    assert _log_paths(tmp_path / "standalone" / RUNTIME_STATE_FILENAME) == (
        tmp_path / "standalone" / "requests.jsonl",
        tmp_path / "standalone" / "prompts" / "prompt.log",
    )


def test_viewer_finds_per_game_and_run_level_request_logs(tmp_path: Path) -> None:
    viewer_data_path = tmp_path / "artifacts" / "ls20-9607627b_p0_viewer_data.json"

    # runs from before this change name a single game's log requests.jsonl
    run_level = tmp_path / "requests.jsonl"
    run_level.touch()
    assert viewer_data._resolve_request_log_path(
        run_dir=tmp_path, viewer_data_path=viewer_data_path, game_id="ls20-9607627b"
    ) == run_level

    run_level.unlink()
    per_game = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    per_game.touch()
    assert viewer_data._resolve_request_log_path(
        run_dir=tmp_path, viewer_data_path=viewer_data_path, game_id="ls20-9607627b"
    ) == per_game
