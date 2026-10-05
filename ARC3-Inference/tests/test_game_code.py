"""The python tool can read the game's source when ARC3_GAME_CODE_DIR is set."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference.agent.game_code import load_game_code
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import _build_system_prompt

GAME_CODE = {
    "ls20.py": "import numpy as np\nclass Ls20:\n    pass\n",
    "arcengine/camera.py": "class Camera:\n    MAX_DIMENSION = 64\n",
}


def _run(code: str, game_code: dict[str, str] | None) -> dict:
    def no_actions(*args, **kwargs):
        raise AssertionError("no action expected")

    return run_sandboxed_python(
        code=code,
        timeout_seconds=10,
        initial_state={},
        action_handler=no_actions,
        game_code=game_code,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_game_code_puts_the_game_module_first(tmp_path: Path) -> None:
    _write(tmp_path / "arcengine" / "camera.py", "class Camera: ...\n")
    _write(tmp_path / "arcengine" / "base_game.py", "class ARCBaseGame: ...\n")
    _write(tmp_path / "ls20-9607627b" / "ls20.py", "class Ls20: ...\n")

    files = load_game_code(tmp_path, "ls20-9607627b")

    assert list(files) == ["ls20.py", "arcengine/base_game.py", "arcengine/camera.py"]
    assert files["ls20.py"] == "class Ls20: ...\n"


def test_load_game_code_fails_for_a_game_without_extracted_code(tmp_path: Path) -> None:
    _write(tmp_path / "arcengine" / "camera.py", "class Camera: ...\n")
    with pytest.raises(FileNotFoundError, match="extract_game_code.py ft09"):
        load_game_code(tmp_path, "ft09-0d8bbf25")


def test_sandbox_reads_numbered_lines_of_the_game_module() -> None:
    result = _run("print(game_code_files)\nread_game_code(2, 3)", GAME_CODE)

    assert result["error"] == ""
    assert result["stdout"] == (
        "['ls20.py', 'arcengine/camera.py']\n"
        "2| class Ls20:\n"
        "3|     pass\n"
    )


def test_sandbox_returns_exact_text_of_any_listed_file() -> None:
    code = (
        "assert game_code() == " + repr(GAME_CODE["ls20.py"]) + "\n"
        "import re\n"
        "print(re.findall(r'MAX_DIMENSION = (\\d+)', game_code('arcengine/camera.py')))"
    )
    result = _run(code, GAME_CODE)

    assert result["error"] == ""
    assert result["stdout"] == "['64']\n"


def test_sandbox_rejects_unknown_files() -> None:
    result = _run("game_code('../metadata.json')", GAME_CODE)
    assert "Unknown file '../metadata.json'" in result["error"]


def test_sandbox_has_no_game_code_functions_when_disabled() -> None:
    result = _run("read_game_code()", None)
    assert "NameError" in result["error"]


def test_system_prompt_describes_game_code_only_when_given() -> None:
    with_code = _build_system_prompt(tool_output_tokens=3072, game_code=GAME_CODE)
    without = _build_system_prompt(tool_output_tokens=3072)

    assert "`ls20.py` (3 lines)" in with_code
    assert "read_game_code(" in with_code
    assert "read_game_code" not in without
