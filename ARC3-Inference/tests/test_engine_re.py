"""The engine reverse-engineering harness: traces, sandbox, tester and kernel.

Uses a tiny synthetic game, so no downloaded game files are needed."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from engine_re.kernel import KernelClient
from engine_re.tester import replay_test, run_candidate
from engine_re.trace import Action, Trace, actions_from_events, parse_action_label, record_trace

TINY_GAME = '''
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

MOVES = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, DOWN), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


class Tiny(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        player = Sprite([[9]], name="player", x=1, y=1)
        wall = Sprite([[5] * 8], name="wall", x=0, y=0)
        level = Level(sprites=[player, wall], grid_size=(8, 8))
        super().__init__(game_id="tiny", levels=[level], camera=Camera(0, 0, 8, 8, 0, 3), available_actions=[1, 2, 3, 4])

    def step(self) -> None:
        dx, dy = MOVES.get(self.action.id, (0, 0))
        if dx or dy:
            self.try_move("player", dx, dy)
        self.complete_action()
'''

ACTIONS = [Action(0), Action(2), Action(2), Action(4), Action(1), Action(1), Action(1), Action(3)]


def _game_class(source: str) -> type:
    module = types.ModuleType("tiny_game")
    exec(compile(source, "tiny.py", "exec"), module.__dict__)
    return module.Tiny


@pytest.fixture()
def tiny_trace() -> Trace:
    return record_trace(_game_class(TINY_GAME.replace("DOWN", "1")), "tiny", ACTIONS)


def _engine(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "engine.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_action_labels_from_run_logs() -> None:
    assert parse_action_label("MOUSE(row=46, col=12)") == Action(6, x=12, y=46)
    assert parse_action_label("LEFT") == Action(3)
    assert parse_action_label("UNDO") == Action(7)
    events = [{"type": "initial"}, {"type": "action", "action_display": "UP"}, {"type": "action", "action_display": "SPACE"}]
    assert actions_from_events(events) == [Action(0), Action(1), Action(5)]


def test_trace_round_trip(tmp_path: Path, tiny_trace: Trace) -> None:
    tiny_trace.save(tmp_path / "trace")
    loaded = Trace.load(tmp_path / "trace")
    assert [s.record() for s in loaded.steps] == [s.record() for s in tiny_trace.steps]
    assert all(np.array_equal(a.frames, b.frames) for a, b in zip(loaded.steps, tiny_trace.steps))


def test_exact_engine_passes(tmp_path: Path, tiny_trace: Trace) -> None:
    report = replay_test(_engine(tmp_path, TINY_GAME.replace("DOWN", "1")), tiny_trace, scratch_root=tmp_path)
    assert report.passed, report.text
    assert "ALL STEPS MATCH" in report.text


def test_divergence_is_located(tmp_path: Path, tiny_trace: Trace) -> None:
    # Moving down by 2 instead of 1 first shows at step 1.
    report = replay_test(_engine(tmp_path, TINY_GAME.replace("DOWN", "2")), tiny_trace, scratch_root=tmp_path)
    assert not report.passed
    assert report.first_fail == 1
    assert "--- Step 1: ACTION2" in report.text


def test_engine_crash_is_reported(tmp_path: Path, tiny_trace: Trace) -> None:
    source = TINY_GAME.replace("DOWN", "1").replace("self.complete_action()", "raise RuntimeError('boom')")
    report = replay_test(_engine(tmp_path, source), tiny_trace, scratch_root=tmp_path)
    assert report.error is not None and "boom" in report.error
    assert report.exact == 0


def test_candidate_cannot_read_files(tmp_path: Path) -> None:
    secret = tmp_path / "answers.txt"
    secret.write_text("frames", encoding="utf-8")
    source = TINY_GAME.replace("DOWN", "1").replace(
        "    def step(self) -> None:", f"    def step(self) -> None:\n        open({str(secret)!r}).read()"
    )
    result, _ = run_candidate(_engine(tmp_path, source), [Action(0).to_json()], scratch_root=tmp_path)
    assert result["error"] is not None and "sandbox: reading" in result["error"]


def test_kernel_persists_state_and_is_sandboxed(tmp_path: Path, tiny_trace: Trace) -> None:
    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=30)
    try:
        assert kernel.execute("x = len(S); x").strip() == str(len(ACTIONS))
        assert kernel.execute("x + 1").strip() == str(len(ACTIONS) + 1)
        assert "sandbox: reading" in kernel.execute(f"open({str(outside)!r}).read()")
        assert "subprocess.Popen is not allowed" in kernel.execute("import subprocess; subprocess.run(['true'])")
        assert kernel.execute("open('notes.txt', 'w').write('ok')").strip() == "2"
    finally:
        kernel.stop()


class _ScriptedModel:
    """Stands in for OpenRouter: returns the given tool calls, one turn each."""

    def __init__(self, turns: list[list[tuple[str, dict]]]):
        self.turns = turns

    def chat(self, messages, tools):  # noqa: ARG002
        import json

        calls = self.turns.pop(0)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                            for i, (name, args) in enumerate(calls)
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def test_agent_tests_a_changed_engine_automatically(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    exact = TINY_GAME.replace("DOWN", "1")
    model = _ScriptedModel(
        [
            [("python", {"code": "x = 1"})],
            [("python", {"code": f"open('engine.py', 'w').write({exact!r})"})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), client=model)
    result = agent.run()
    # The second turn rewrote engine.py without calling run_tests: the harness tested it, and it passes.
    assert result.auto_tests == 1
    assert result.status == "passed"
    assert any("[harness] engine.py changed" in m.get("content", "") for m in agent.messages if m["role"] == "tool")


def test_python_quota_pauses_until_engine_changes(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("python", {"code": "1"})],
            [("python", {"code": "2"})],
            [("python", {"code": "3"})],  # over the quota: paused
            [("edit_engine", {"old_str": "pass", "new_str": "pass  # changed", "replace_all": True})],
            [("python", {"code": "4"})],  # engine changed: runs again
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5, python_quota=2), client=model)
    agent.run()
    outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert outputs[0].strip() == "1" and outputs[1].strip() == "2"
    assert outputs[2].startswith("[harness] Python is paused")
    assert outputs[4].strip() == "4"
    assert agent.result.python_paused == 1
