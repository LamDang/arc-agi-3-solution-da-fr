"""Download ARC-AGI-3 game files for offline harness runs.

This build bundles no game files: the harness plays from a local directory
passed as ENVIRONMENTS_DIR / --environments-dir. This downloads the files once
from the ARC-AGI-3 API (three.arcprize.org) into that directory and skips games
that are already there. ARC_API_KEY is used when set; otherwise arc_agi
requests an anonymous key.

    uv run --no-sync python scripts/fetch_games.py            # all 25 official games
    uv run --no-sync python scripts/fetch_games.py ls20 ft09  # ids or prefixes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arc_agi
import requests

from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS

DEFAULT_ENVIRONMENTS_DIR = Path(__file__).resolve().parents[1] / "environment_files"


def _resolve(requested: str) -> str:
    matches = [
        game_id
        for game_id in DUCK_HARNESS_PUBLIC_GAME_IDS
        if game_id == requested or game_id.startswith(requested)
    ]
    if len(matches) != 1:
        raise SystemExit(f"{requested!r} does not match exactly one official game id.")
    return matches[0]


def _is_present(environments_dir: Path, game_id: str) -> bool:
    # arc_agi stores downloads as {environments_dir}/{base_id}/{version}/.
    base_id, _, version = game_id.partition("-")
    return (environments_dir / base_id / version / "metadata.json").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "games", nargs="*", help="Game ids or prefixes. Default: all 25 official games."
    )
    parser.add_argument(
        "--environments-dir", type=Path, default=DEFAULT_ENVIRONMENTS_DIR
    )
    args = parser.parse_args()
    environments_dir: Path = args.environments_dir.resolve()

    game_ids = [_resolve(game) for game in args.games] or list(DUCK_HARNESS_PUBLIC_GAME_IDS)
    missing = [game_id for game_id in game_ids if not _is_present(environments_dir, game_id)]
    if not missing:
        print(f"All {len(game_ids)} games already present in {environments_dir}")
        return 0

    environments_dir.mkdir(parents=True, exist_ok=True)
    try:
        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.NORMAL,
            environments_dir=str(environments_dir),
        )
        for game_id in missing:
            arcade.make(game_id)
    except requests.RequestException as exc:
        print(
            f"Could not reach the ARC-AGI-3 API ({type(exc).__name__}). "
            "Is three.arcprize.org allowed by the network policy?",
            file=sys.stderr,
        )
        return 1

    failed = [game_id for game_id in missing if not _is_present(environments_dir, game_id)]
    if failed:
        print(f"Failed to download: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"Downloaded {len(missing)} game(s) into {environments_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
