"""Read-only access to the game's own source code, for the engine-code experiment.

ARC3_GAME_CODE_DIR names a directory written by scripts/extract_game_code.py:
``<dir>/<game_id>/<module>.py`` holds each game's module and
``<dir>/arcengine/*.py`` the engine package the games import. When it is set,
the python tool gains ``game_code_files``, ``game_code(file)`` and
``read_game_code(start, end, file)``, and the system prompt describes them.
When it is unset, the agent is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from inference.utils.grid_utils import ARC_COLOR_CHARS

ENV_VAR = "ARC3_GAME_CODE_DIR"
ENGINE_DIRNAME = "arcengine"
MANIFEST_FILENAME = "manifest.json"

GAME_CODE_ADDENDUM = (
    "\n\nGame source code:\n"
    "- This game's source code can be read from the python tool. It is the code the "
    "game runs, so its levels, rules and win conditions are exactly as written there. "
    "Reading it costs no game actions.\n"
    "- `game_code_files` lists the files: `{module}` ({lines} lines), the game itself, "
    "an `ARCBaseGame` subclass with its sprites and levels; then the `arcengine` "
    "package it is built on.\n"
    "- `read_game_code(start=1, end=None, file=None)` returns lines `start` to `end` "
    "(1-based, inclusive) of a file, each prefixed with its line number. `file` "
    "defaults to `{module}`. `game_code(file=None)` returns a file's full text, for "
    "searching with `re` or string methods.\n"
    "- The game module is long and mostly sprite pixel tables. Search for `class `, "
    "`def ` and names first, then print at most about 150 lines per call: tool output "
    "is cut to about {tool_output_tokens} tokens.\n"
    "- Colors in the code are integers 0-15. In `.ascii`, color i is the character "
    "`\"{color_chars}\"[i]`. `row` and `col` are cells of the 64x64 display frame; "
    "the game's `Camera` (`arcengine/camera.py`) maps game coordinates to them.\n"
)


def game_code_dir() -> Path | None:
    raw = os.environ.get(ENV_VAR, "").strip()
    return Path(raw) if raw else None


def load_game_code(root: Path, game_id: str) -> dict[str, str]:
    """The files the agent may read, by name, the game module first.

    Raises when the game has no extracted module: a run that was asked to
    expose the code must not silently play without it.
    """
    modules = sorted((root / game_id).glob("*.py"))
    if len(modules) != 1:
        raise FileNotFoundError(
            f"{ENV_VAR}={root}: expected one module in {root / game_id}, found "
            f"{len(modules)}. Run scripts/extract_game_code.py {game_id.split('-', 1)[0]}."
        )
    files = {modules[0].name: modules[0].read_text(encoding="utf-8")}
    engine = sorted((root / ENGINE_DIRNAME).glob("*.py"))
    if not engine:
        raise FileNotFoundError(f"{ENV_VAR}={root}: no {ENGINE_DIRNAME}/*.py files.")
    for path in engine:
        files[f"{ENGINE_DIRNAME}/{path.name}"] = path.read_text(encoding="utf-8")
    return files


def game_code_addendum(files: dict[str, str], *, tool_output_tokens: int) -> str:
    module = next(iter(files))
    return GAME_CODE_ADDENDUM.format(
        module=module,
        lines=len(files[module].splitlines()),
        tool_output_tokens=tool_output_tokens,
        color_chars=ARC_COLOR_CHARS,
    )
