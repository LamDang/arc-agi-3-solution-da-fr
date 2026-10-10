"""--game resolves community game ids found under a local ENVIRONMENTS_DIR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from inference.framework.run import _local_game_ids, _resolve_game_ids


def _add_game(environments_dir: Path, game_id: str) -> None:
    base_id, _, version = game_id.partition("-")
    game_dir = environments_dir / base_id / version
    game_dir.mkdir(parents=True)
    (game_dir / "metadata.json").write_text(json.dumps({"game_id": game_id}), encoding="utf-8")


def _args(game: str, environments_dir: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        game=game,
        dataset="",
        include_tags="",
        exclude_tags="",
        environments_dir=None if environments_dir is None else str(environments_dir),
    )


@pytest.fixture
def environments_dir(tmp_path: Path) -> Path:
    for game_id in ("mm01-63be02fb", "sg01-63be02fb", "sg01-4ac0337b8857411e"):
        _add_game(tmp_path, game_id)
    return tmp_path


def test_local_game_ids_lists_every_metadata_file(environments_dir: Path) -> None:
    assert sorted(_local_game_ids(str(environments_dir))) == [
        "mm01-63be02fb", "sg01-4ac0337b8857411e", "sg01-63be02fb",
    ]
    assert _local_game_ids(None) == []
    assert _local_game_ids("__auto__") == []


def test_official_and_community_games_resolve_together(environments_dir: Path) -> None:
    game_ids = _resolve_game_ids(_args("ls20,mm01,sg01-4ac0337b8857411e", environments_dir))
    assert game_ids == ["ls20-9607627b", "mm01-63be02fb", "sg01-4ac0337b8857411e"]


def test_shared_prefix_needs_the_full_id(environments_dir: Path) -> None:
    with pytest.raises(ValueError, match="Ambiguous ARC-AGI3 game 'sg01'"):
        _resolve_game_ids(_args("sg01", environments_dir))


def test_community_game_needs_an_environments_dir() -> None:
    with pytest.raises(ValueError, match="Unknown ARC-AGI3 game: mm01"):
        _resolve_game_ids(_args("mm01", None))
