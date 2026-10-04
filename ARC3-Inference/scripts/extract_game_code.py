"""Copy the exact source of games and of arcengine into a game-code directory.

With ARC3_GAME_CODE_DIR set to that directory, the agent can read its game's
code from the python tool (see inference/agent/game_code.py). Each game's
module is copied byte for byte from the file the arc_agi loader runs, found
under ENVIRONMENTS_DIR. The arcengine package the games import is copied from
the installed version. manifest.json records each file's source and sha256.

    uv run --no-sync python scripts/extract_game_code.py ls20 ft09 vc33 sp80 lp85
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from importlib import metadata
from pathlib import Path

import arcengine

from inference.agent.game_code import ENGINE_DIRNAME, MANIFEST_FILENAME

HARNESS_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _game_dirs(environments_dir: Path, requested: str) -> Path:
    # arc_agi stores downloads as {environments_dir}/{base_id}/{version}/.
    matches = [
        path.parent
        for path in sorted(environments_dir.glob("*/*/metadata.json"))
        if (game_id := json.loads(path.read_text(encoding="utf-8"))["game_id"]) == requested
        or game_id.startswith(requested)
    ]
    if len(matches) != 1:
        raise SystemExit(f"{requested!r} matches {len(matches)} games in {environments_dir}.")
    return matches[0]


def _copy(source: Path, destination: Path, root: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(source) != _sha256(destination):
        raise SystemExit(f"Copy of {source} differs from its source.")
    return {
        "path": destination.relative_to(root).as_posix(),
        "source": os.path.relpath(source, HARNESS_DIR),
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "lines": len(destination.read_text(encoding="utf-8").splitlines()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("games", nargs="+", help="Game ids or prefixes.")
    parser.add_argument(
        "--environments-dir", type=Path, default=HARNESS_DIR / "environment_files"
    )
    parser.add_argument("--out", type=Path, default=HARNESS_DIR / "game_code")
    args = parser.parse_args()
    out: Path = args.out.resolve()

    games: dict[str, list[dict[str, object]]] = {}
    for requested in args.games:
        game_dir = _game_dirs(args.environments_dir.resolve(), requested)
        game_id = json.loads((game_dir / "metadata.json").read_text(encoding="utf-8"))["game_id"]
        # The loader runs {class_name.lower()}.py; every official game dir
        # holds exactly that one module, named after the game's base id.
        module = game_dir / f"{game_id.split('-', 1)[0]}.py"
        if sorted(game_dir.glob("*.py")) != [module]:
            raise SystemExit(f"Expected exactly {module.name} in {game_dir}.")
        games[game_id] = [_copy(module, out / game_id / module.name, out)]

    engine_dir = Path(arcengine.__file__).parent
    engine = [
        _copy(source, out / ENGINE_DIRNAME / source.name, out)
        for source in sorted(engine_dir.glob("*.py"))
    ]

    manifest = {
        "arcengine_version": metadata.version("arcengine"),
        "games": games,
        "engine": engine,
    }
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for game_id, files in games.items():
        print(f"{game_id}: {files[0]['path']} ({files[0]['lines']} lines)")
    print(f"arcengine {manifest['arcengine_version']}: {len(engine)} files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
