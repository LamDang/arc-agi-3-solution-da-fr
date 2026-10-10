"""Merge the community game submodules into one ENVIRONMENTS_DIR.

The community collections under community_games/ are git submodules in the
arcengine game format ({base_id}/{version}/metadata.json + {base_id}.py), the
same format as the official games in environment_files/. arc_agi reads games
from a single directory, so this links every game of every source, and the
official games, into environment_files_community/ (git-ignored): a real
directory per game whose files are symlinks to the submodule's files.

Skipped: community copies of an official game id prefix (arc-interactive ships
old ft09, ls20 and vc33 versions), so `ls20` still means the official game.
Two sources may share a prefix (sg01 and sg04 are in both collections); their
full ids differ, so pass the full id to the harness for those.

The NVIDIA DreamTeam games vendored in arc3-synthetic-games import
`arc_agi_3.game_creator.arcengine_adapter`; put NVIDIA_RUNTIME on PYTHONPATH to
play them (the script prints the export line).

Each community game is loaded and RESET once, which also gives its level count.
taaf requires one baseline action count per level; when a game's metadata.json
lists a different number (most arc-interactive games), the linked metadata.json
is replaced by a copy without baseline_actions, and the harness plays the game
without partial credit for unfinished levels.

catalog.json in the output lists each game's id, source, title, tags, level
count, baselines ("ok", "none" or "dropped: <n> for <m> levels") and the load
check ("ok" or the error).

    git submodule update --init ARC3-Inference/community_games
    uv run --no-sync python scripts/build_community_envs.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
COMMUNITY_DIR = HARNESS_DIR / "community_games"
NVIDIA_RUNTIME = COMMUNITY_DIR / "arc3-synthetic-games/third_party/nvidia/runtime_support"
DEFAULT_OUTPUT_DIR = HARNESS_DIR / "environment_files_community"
OFFICIAL_DIR = HARNESS_DIR / "environment_files"
CATALOG_FILENAME = "catalog.json"

# source name -> game directory inside the submodules
SOURCES: dict[str, Path] = {
    "arc-interactive": COMMUNITY_DIR / "arc-interactive/environment_files",
    "arc3-synthetic-games": COMMUNITY_DIR / "arc3-synthetic-games/environment_files",
    "nvidia-dreamteam": COMMUNITY_DIR / "arc3-synthetic-games/third_party/nvidia/environment_files",
}


def _official_prefixes() -> set[str]:
    from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS

    return {game_id.split("-")[0] for game_id in DUCK_HARNESS_PUBLIC_GAME_IDS}


def _games(source_dir: Path) -> list[tuple[dict, Path]]:
    games = []
    for metadata_file in sorted(source_dir.rglob("metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        games.append((metadata, metadata_file.parent))
    return games


def _link_game(game_dir: Path, output_dir: Path, game_id: str) -> Path:
    # arc_agi loads {class_name.lower()}.py from the directory holding
    # metadata.json; linking files, not the directory, keeps rglob finding it.
    base_id, _, version = game_id.partition("-")
    target = output_dir / base_id / version
    target.mkdir(parents=True)
    for path in sorted(game_dir.iterdir()):
        if path.is_file():
            (target / path.name).symlink_to(path.resolve())
    return target


def _entry(metadata: dict, *, source: str, game_dir: Path) -> dict:
    return {
        "game_id": metadata["game_id"],
        "source": source,
        "title": metadata.get("title"),
        "tags": metadata.get("tags") or [],
        "levels": len(metadata.get("baseline_actions") or []) or None,
        "baselines": "ok" if metadata.get("baseline_actions") else "none",
        "path": os.path.relpath(game_dir, HARNESS_DIR),
    }


def _load(output_dir: Path, game_ids: list[str]) -> dict[str, tuple[str, int | None]]:
    """Load each game and take a RESET: game id -> (status, level count)."""
    import arc_agi
    from arcengine import GameAction

    if NVIDIA_RUNTIME.is_dir() and str(NVIDIA_RUNTIME) not in sys.path:
        sys.path.insert(0, str(NVIDIA_RUNTIME))
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(output_dir),
        logger=logging.getLogger("build_community_envs.arc_agi"),
    )
    results: dict[str, tuple[str, int | None]] = {}
    for game_id in game_ids:
        try:
            env = arcade.make(game_id)
            if env is None or env.observation_space is None:
                raise RuntimeError("no environment or observation after make()")
            if env.step(GameAction.RESET) is None:
                raise RuntimeError("RESET returned no frame")
            results[game_id] = ("ok", env.observation_space.win_levels)
        except Exception as exc:  # noqa: BLE001 - report every broken game
            results[game_id] = (f"{type(exc).__name__}: {exc}"[:300], None)
    return results


def _drop_baselines(output_dir: Path, game_id: str) -> None:
    base_id, _, version = game_id.partition("-")
    metadata_file = output_dir / base_id / version / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata.pop("baseline_actions", None)
    metadata_file.unlink()
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source", action="append", choices=sorted(SOURCES),
        help="Sources to link (repeatable). Default: all.",
    )
    parser.add_argument(
        "--no-official", action="store_true",
        help="Do not link the official games from environment_files/.",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()

    if output_dir.exists():
        if not (output_dir / CATALOG_FILENAME).is_file():
            raise SystemExit(f"{output_dir} exists and was not built by this script; not replacing it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    official_prefixes = _official_prefixes()
    catalog: list[dict] = []
    seen: dict[str, str] = {}
    skipped: list[str] = []

    if not args.no_official and OFFICIAL_DIR.is_dir():
        for metadata, game_dir in _games(OFFICIAL_DIR):
            _link_game(game_dir, output_dir, metadata["game_id"])
            seen[metadata["game_id"]] = "official"
            catalog.append(_entry(metadata, source="official", game_dir=game_dir))

    for source in args.source or list(SOURCES):
        source_dir = SOURCES[source]
        if not source_dir.is_dir():
            raise SystemExit(
                f"{source_dir} is missing. Run: git submodule update --init ARC3-Inference/community_games"
            )
        for metadata, game_dir in _games(source_dir):
            game_id = metadata["game_id"]
            if game_id.split("-")[0] in official_prefixes:
                skipped.append(f"{game_id} ({source}: copy of an official game)")
                continue
            if game_id in seen:
                skipped.append(f"{game_id} ({source}: duplicate of {seen[game_id]})")
                continue
            _link_game(game_dir, output_dir, game_id)
            seen[game_id] = source
            catalog.append(_entry(metadata, source=source, game_dir=game_dir))

    community_ids = [entry["game_id"] for entry in catalog if entry["source"] != "official"]
    results = _load(output_dir, community_ids)
    for entry in catalog:
        if entry["game_id"] not in results:
            continue
        entry["check"], levels = results[entry["game_id"]]
        if levels is None:
            continue
        if entry["baselines"] == "ok" and entry["levels"] != levels:
            entry["baselines"] = f"dropped: {entry['levels']} for {levels} levels"
            _drop_baselines(output_dir, entry["game_id"])
        entry["levels"] = levels

    (output_dir / CATALOG_FILENAME).write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for entry in catalog:
        counts[entry["source"]] = counts.get(entry["source"], 0) + 1
    print(f"Linked {len(catalog)} games into {output_dir}:")
    for source, count in counts.items():
        print(f"  {source}: {count}")
    for line in skipped:
        print(f"  skipped {line}")
    prefixes: dict[str, list[str]] = {}
    for game_id in seen:
        prefixes.setdefault(game_id.split("-")[0], []).append(game_id)
    shared = {prefix: ids for prefix, ids in prefixes.items() if len(ids) > 1}
    if shared:
        print("Prefixes shared by several games (pass the full id): " + ", ".join(
            f"{prefix}: {' / '.join(ids)}" for prefix, ids in sorted(shared.items())
        ))
    failed = [entry for entry in catalog if entry.get("check", "ok") != "ok"]
    print(f"Loaded {len(community_ids)} community games: {len(community_ids) - len(failed)} ok, {len(failed)} failed")
    for entry in failed:
        print(f"  {entry['game_id']} ({entry['source']}): {entry['check']}")
    for source in counts:
        dropped = sum(1 for e in catalog if e["source"] == source and e["baselines"].startswith("dropped"))
        missing = sum(1 for e in catalog if e["source"] == source and e["baselines"] == "none")
        if dropped or missing:
            print(f"  {source}: baselines dropped (count != levels) for {dropped}, absent for {missing}")
    if "nvidia-dreamteam" in counts:
        print(f'NVIDIA DreamTeam games need: export PYTHONPATH="{NVIDIA_RUNTIME}${{PYTHONPATH:+:$PYTHONPATH}}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
