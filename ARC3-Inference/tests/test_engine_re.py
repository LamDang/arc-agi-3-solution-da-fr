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


ANIMATED_STEP = """    def step(self) -> None:
        if not getattr(self, "ticks", 0):
            dx, dy = MOVES.get(self.action.id, (0, 0))
            if dx or dy:
                self.try_move("player", dx, dy)
            self.ticks = 3 if (dx or dy) else 1
        self.ticks -= 1
        if self.ticks == 0:
            self.complete_action()
"""


def test_animation_frames_are_compared_only_with_match_all(tmp_path: Path) -> None:
    # The real game spends 3 frames on each move; the candidate does it in one.
    plain_step = TINY_GAME[TINY_GAME.index("    def step(self)") :]
    animated = TINY_GAME.replace("DOWN", "1").replace(plain_step, ANIMATED_STEP)
    trace = record_trace(_game_class(animated), "tiny", ACTIONS)
    assert trace[1].n_frames == 3
    engine = _engine(tmp_path, TINY_GAME.replace("DOWN", "1"))
    final = replay_test(engine, trace, scratch_root=tmp_path)
    assert final.passed, final.text
    strict = replay_test(engine, trace, scratch_root=tmp_path, match="all")
    assert not strict.passed and strict.first_fail == 1
    assert "frame count" in strict.text


@pytest.mark.parametrize("interface", ["simple", "arcengine"])
def test_skeleton_runs_as_an_engine(tmp_path: Path, tiny_trace: Trace, interface: str) -> None:
    from engine_re.skeleton import render_skeleton

    report = replay_test(_engine(tmp_path, render_skeleton("tiny", [1, 2, 3, 4], interface)), tiny_trace, scratch_root=tmp_path)
    assert report.error is None, report.error
    assert "--- Step 0: RESET" in report.text
    if interface == "simple":
        assert report.contract_passed == report.contract_total == 5, report.text


SIMPLE_TINY_GAME = """
def make_level(n):
    return State(
        grid=(8, 8),
        sprites=[
            Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
            Sprite([[0] * 8 for _ in range(8)], layer=-1, collidable=False, name="background"),
            PLAYER,
            Sprite([[5] * 8], x=0, y=0, tags=("wall",)),
        ],
        vars={"player": PLAYER},
    )


def step(state, action):
    moves = {1: (0, -1), 2: (0, DOWN), 3: (-1, 0), 4: (1, 0)}
    if action.id in moves:
        state.try_move(state.vars["player"], *moves[action.id])
"""

# Module-level objects are fine: the harness deep-copies the first state of a level on every level start.
SIMPLE_TINY_GAME = 'PLAYER = Sprite([[9]], x=1, y=1, tags=("player",))\n' + SIMPLE_TINY_GAME


def _simple_engine(tmp_path: Path, game_code: str) -> Path:
    from engine_re.skeleton import render_skeleton

    skeleton = render_skeleton("tiny", [1, 2, 3, 4])
    source = skeleton[: skeleton.index("# ==== YOUR GAME ====")] + game_code
    return _engine(tmp_path, source)


def test_simple_engine_passes_contract_and_acceptance(tmp_path: Path, tiny_trace: Trace) -> None:
    report = replay_test(_simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "1")), tiny_trace, scratch_root=tmp_path)
    assert report.passed, report.text
    assert report.contract_total and report.contract_passed == report.contract_total
    assert "Contract tests: 5/5 pass." in report.text


def test_simple_engine_divergence_is_located(tmp_path: Path, tiny_trace: Trace) -> None:
    report = replay_test(_simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "2")), tiny_trace, scratch_root=tmp_path)
    assert not report.passed and report.first_fail == 1


def test_contract_catches_an_edited_interface(tmp_path: Path, tiny_trace: Trace) -> None:
    path = _simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "1"))
    path.write_text(path.read_text().replace("    layer: int = 0  # higher", "    layer: int = 1  # higher"), encoding="utf-8")
    report = replay_test(path, tiny_trace, scratch_root=tmp_path)
    assert not report.passed
    assert "FAILED the FIXED INTERFACE block is unchanged" in report.text


def test_reset_restores_the_first_state_of_the_level(tmp_path: Path) -> None:
    # Moving, then RESET, then moving again must start from the original position, even though
    # make_level returns a module-level sprite that step() moves.
    actions = [Action(0), Action(4), Action(4), Action(0), Action(2)]
    trace = record_trace(_game_class(TINY_GAME.replace("DOWN", "1")), "tiny", actions)
    report = replay_test(_simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "1")), trace, scratch_root=tmp_path)
    assert report.passed, report.text


def _arc_and_fixed_sprites(rng, api):
    """The same random sprite as an arcengine Sprite and as a fixed-interface Sprite."""
    from arcengine import BlockingMode, Sprite as ArcSprite

    blocking = {"pixel": BlockingMode.PIXEL_PERFECT, "box": BlockingMode.BOUNDING_BOX, "none": BlockingMode.NOT_BLOCKED}
    h, w = rng.randint(1, 4), rng.randint(1, 4)
    pixels = [[rng.choice([-1, -1, -2, 0, 3, 7, 9]) for _ in range(w)] for _ in range(h)]
    kw = dict(
        x=rng.randint(-2, 8), y=rng.randint(-2, 8), layer=rng.randint(-1, 2), rotation=rng.choice([0, 90, 180, 270]),
        mirror_ud=rng.random() < 0.3, mirror_lr=rng.random() < 0.3, scale=rng.choice([1, 1, 2, 3]),
    )
    visible, collidable, block = rng.random() < 0.8, rng.random() < 0.8, rng.choice(["pixel", "pixel", "box", "none"])
    tags = tuple(rng.sample(["a", "b"], rng.randint(0, 2)))
    name = f"s{rng.randrange(10**6)}"
    ours = api.Sprite([r[:] for r in pixels], name=name, tags=tags, visible=visible, collidable=collidable, blocking=block, **kw)
    theirs = ArcSprite([r[:] for r in pixels], name=name, tags=list(tags), blocking=blocking[block], visible=visible, collidable=collidable, **kw)
    return ours, theirs


def test_fixed_interface_follows_arcengine_sprites() -> None:
    import random

    from arcengine import Level

    from engine_re.game_api import canonical

    api, rng = canonical(), random.Random(0)
    for _ in range(300):
        pairs = [_arc_and_fixed_sprites(rng, api) for _ in range(rng.randint(2, 6))]
        for ours, theirs in pairs:
            assert np.asarray(ours.render()).tolist() == np.asarray(theirs.render()).tolist()
            assert (ours.width, ours.height) == (theirs.width, theirs.height)
        for a in pairs:
            for b in pairs:
                assert a[0].collides_with(b[0]) == a[1].collides_with(b[1])
        state, level = api.State(grid=(16, 16), sprites=[o for o, _ in pairs]), Level(sprites=[t for _, t in pairs])
        for _ in range(10):
            x, y, tag, everything = rng.randint(-2, 14), rng.randint(-2, 14), rng.choice([None, "a"]), rng.random() < 0.3
            ours = state.sprite_at(x, y, tag=tag, include_uncollidable=everything)
            theirs = level.get_sprite_at(x, y, tag=tag, ignore_collidable=everything)
            assert (ours.name if ours else None) == (theirs.name if theirs else None)


def test_render_matches_arcengine_camera() -> None:
    import random

    from arcengine import Camera, Level

    from engine_re.game_api import canonical, render

    api, rng = canonical(), random.Random(1)
    for _ in range(50):
        pairs = [_arc_and_fixed_sprites(rng, api) for _ in range(rng.randint(1, 6))]
        w, h = rng.randint(4, 20), rng.randint(4, 20)
        level = Level(sprites=[t for _, t in pairs], grid_size=(w, h))
        camera = Camera(0, 0, w, h, background=4, letter_box=2)
        expected = np.asarray(camera.render(level.get_sprites()))
        state = api.State(
            grid=(w, h),
            sprites=[
                api.Sprite([[2] * 64 for _ in range(64)], screen=True, layer=-100, collidable=False),
                api.Sprite([[4] * w for _ in range(h)], layer=-99, collidable=False),
            ]
            + [o for o, _ in pairs],
        )
        assert np.array_equal(render(state), expected)


def test_game_runner_follows_the_episode_rules() -> None:
    import types as _types

    from engine_re.game_api import GameRunner, canonical

    api = canonical()

    def make_level(n):
        return api.State(grid=(4, 4), sprites=[api.Sprite([[n] * 4 for _ in range(4)])], vars={"n": n})

    def step(state, action):
        state.status = {1: "level_solved", 2: "game_over"}.get(action.id, "playing")

    calls = []
    make = lambda n: calls.append(n) or make_level(n)  # noqa: E731
    runner = GameRunner(_types.SimpleNamespace(make_level=make, step=step, Action=api.Action), 2, [1, 2, 3])
    assert runner.perform(Action(0))["state"] == "NOT_FINISHED"
    obs = runner.perform(Action(1))
    assert (obs["levels_completed"], runner.level, int(obs["frames"][-1][0, 0])) == (1, 1, 1)
    assert runner.perform(Action(2))["state"] == "GAME_OVER"
    ended = runner.perform(Action(3))
    assert len(ended["frames"]) == 0 and ended["levels_completed"] == 0 and ended["win_levels"] == 0
    assert runner.perform(Action(0))["levels_completed"] == 1 and runner.level == 1  # RESET restarts the level
    assert runner.perform(Action(1))["state"] == "WIN"
    restarted = runner.perform(Action(0))
    assert (restarted["levels_completed"], runner.level) == (0, 0)  # RESET after WIN starts over
    assert calls == [0, 1]  # make_level runs once per level; level starts get copies


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
            [("edit_engine", {"old_str": "# ==== YOUR GAME ====", "new_str": "# ==== YOUR GAME ==== (changed)"})],
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


def test_compaction_keeps_tool_argument_keys() -> None:
    import json

    from engine_re.agent import _elide_arguments

    elided = json.loads(_elide_arguments(json.dumps({"content": "x" * 5000})))
    assert set(elided) == {"content"}
    assert len(elided["content"]) < 400 and "elided" in elided["content"]
