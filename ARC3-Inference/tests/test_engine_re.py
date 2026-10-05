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


def _game_class(source: str, name: str = "Tiny") -> type:
    module = types.ModuleType("tiny_game")
    exec(compile(source, "tiny.py", "exec"), module.__dict__)
    return getattr(module, name)


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


def _engine_source(game_code: str) -> str:
    """A make_level/step engine.py: the starting module with `game_code` as the game."""
    from engine_re.skeleton import render_skeleton

    skeleton = render_skeleton("tiny", [1, 2, 3, 4])
    return skeleton[: skeleton.index("# ==== YOUR GAME ====")] + game_code


def _simple_engine(tmp_path: Path, game_code: str) -> Path:
    return _engine(tmp_path, _engine_source(game_code))


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


def test_determinism_check_accepts_sprites_in_vars() -> None:
    from engine_re.game_api import _canonical_vars, canonical

    api = canonical()

    def state():
        a, b = api.Sprite([[1]], name="a"), api.Sprite([[2]], name="b")
        return api.State(grid=(4, 4), sprites=[a, b], vars={"pairs": {a: (a, b)}, "held": [b], "n": 3})

    assert _canonical_vars(state()) == _canonical_vars(state())


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
                assert a[0].collides_with(b[0], ignore_mode=True) == a[1].collides_with(b[1], ignoreMode=True)
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


def test_view_turns_the_frame_and_clicks_follow() -> None:
    import random

    from engine_re.game_api import canonical, click_cell, render

    api, rng = canonical(), random.Random(2)
    for _ in range(40):
        w, h = rng.randint(3, 20), rng.randint(3, 20)
        cells = [api.Sprite([[rng.randint(0, 15)]], x=gx, y=gy) for gx in range(w) for gy in range(h)]
        hud = api.Sprite([[11] * 5], x=2, y=0, screen=True, layer=5)
        plain = api.State(grid=(w, h), sprites=cells + [hud])
        view = api.View(scale=rng.choice([None, 1, 2]), rotation=rng.choice([0, 90, 180, 270]),
                        mirror_ud=rng.random() < 0.5, mirror_lr=rng.random() < 0.5)
        turned = api.State(grid=(w, h), sprites=cells + [hud], view=view)
        base = render(api.State(grid=(w, h), sprites=cells + [hud], view=api.View(scale=view.scale)))
        expected = np.rot90(base, k=-(view.rotation // 90))
        expected = np.flipud(expected) if view.mirror_ud else expected
        expected = np.fliplr(expected) if view.mirror_lr else expected
        assert np.array_equal(render(turned), expected)  # the HUD turns with the frame
        # Clicking anywhere on a drawn cell gives that cell back.
        frame = render(turned)
        marker = cells[rng.randrange(len(cells))]
        marker.pixels = [[16 - 1 if marker.pixels[0][0] != 15 else 0]]
        changed = np.argwhere(render(turned) != frame)
        for y, x in changed[:5]:
            assert click_cell(turned, int(x), int(y)) == (marker.x, marker.y)
        del plain


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


def _tail(source: str) -> str:
    """Everything after the FIXED block of an engine source."""
    from engine_re.game_api import END_MARKER

    return source[source.index(END_MARKER) + len(END_MARKER) :]


def _rewrite_call(game_code: str, actions: tuple[int, ...] = (1, 2, 3, 4), game: str = "tiny") -> str:
    """Kernel code that turns the starting engine.py into one with `game_code` below the FIXED block,
    through edit(), as the model would (python cannot write engine.py)."""
    from engine_re.skeleton import render_skeleton

    old = _tail(render_skeleton(game, list(actions)))
    new = _tail(_engine_source(game_code))
    return f"edit(edits=[{{'op': 'replace_text', 'oldText': {old!r}, 'newText': {new!r}}}])"


def test_agent_tests_a_changed_engine_automatically(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("python", {"code": "x = 1"})],
            [("python", {"code": _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "1"))})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), opening=False, client=model)
    result = agent.run()
    # The second turn changed engine.py without calling run_tests: the harness tested it, and it passes.
    assert result.auto_tests == 1 and result.engine_changes == 1
    assert result.status == "passed"
    assert any("[harness] engine.py changed" in m.get("content", "") for m in agent.messages if m["role"] == "tool")


def test_python_quota_pauses_until_engine_changes(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    change = "edit(edits=[{'op': 'replace_text', 'oldText': '# ==== YOUR GAME ====', 'newText': '# ==== YOUR GAME ==== (changed)'}])"
    model = _ScriptedModel(
        [
            [("python", {"code": "1"})],
            [("python", {"code": "2"})],
            [("python", {"code": "3"})],  # over the quota: paused
            [("python", {"code": change})],  # a call that changes engine.py still runs
            [("python", {"code": "4"})],  # engine changed: runs again
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5, python_quota=2), opening=False, client=model)
    agent.run()
    outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert outputs[0].strip() == "1" and outputs[1].strip() == "2"
    assert outputs[2].startswith("[harness] Python is paused")
    assert outputs[3].startswith("engine.py: replaced line")
    assert outputs[4].startswith("4")
    assert agent.result.python_paused == 1


def test_compaction_keeps_tool_argument_keys() -> None:
    import json

    from engine_re.agent import _elide_arguments

    elided = json.loads(_elide_arguments(json.dumps({"content": "x" * 5000})))
    assert set(elided) == {"content"}
    assert len(elided["content"]) < 400 and "elided" in elided["content"]


# --- Failure reports: failures, one level, regions, sprites, images, reproduction ------------

TWO_LEVELS = '''
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

MOVES = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


class Two(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(sprites=[Sprite([[9]], name="player", x=1, y=1), Sprite([[5] * 8], name="wall", x=0, y=0)], grid_size=(8, 8)),
            Level(sprites=[Sprite([[9]], name="player", x=1, y=3), Sprite([[8] * 8], name="wall", x=0, y=7)], grid_size=(8, 8)),
        ]
        super().__init__(game_id="two", levels=levels, camera=Camera(0, 0, 8, 8, 0, 3), available_actions=[1, 2, 3, 4])

    def step(self) -> None:
        dx, dy = MOVES.get(self.action.id, (0, 0))
        if dx or dy:
            self.try_move("player", dx, dy)
        if self.current_level.get_sprites_by_name("player")[0].x >= 4:
            self.next_level()
        self.complete_action()
'''
# Level 0 is solved at step 3 (the player reaches x=4); steps 4-8 play level 1, with a RESET at step 7.
TWO_ACTIONS = [Action(0), Action(4), Action(4), Action(4), Action(2), Action(4), Action(1), Action(0), Action(2)]

SIMPLE_TWO = """
LAYOUT = {0: ((1, 1), 5, 0), 1: ((1, 3), WALL1, 7)}


def make_level(n):
    (px, py), colour, wall_y = LAYOUT[n]
    player = Sprite([[9]], x=px, y=py, name="player", tags=("player",))
    return State(
        grid=(8, 8),
        sprites=[
            Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
            Sprite([[0] * 8 for _ in range(8)], layer=-1, collidable=False, name="background"),
            player,
            Sprite([[colour] * 8], x=0, y=wall_y, name="wall", tags=("wall",)),
        ],
        vars={"player": player},
    )


def step(state, action):
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    if action.id in moves:
        state.try_move(state.vars["player"], *moves[action.id])
    if state.vars["player"].x >= 4:
        state.status = "level_solved"
"""


@pytest.fixture()
def two_level_trace() -> Trace:
    return record_trace(_game_class(TWO_LEVELS, "Two"), "two", TWO_ACTIONS)


def _repro_commands(text: str) -> str:
    """The python lines the report prints under "Reproduce in python:", as they would be pasted."""
    lines = text.splitlines()
    start = lines.index("  Reproduce in python:")
    return "\n".join(line[4:] for line in lines[start + 1 :] if line.startswith("    "))


def test_the_report_stops_at_the_first_failure_by_default(tmp_path: Path, tiny_trace: Trace) -> None:
    engine = _simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "2"))
    full = replay_test(engine, tiny_trace, scratch_root=tmp_path)
    stop = replay_test(engine, tiny_trace, failures=1, scratch_root=tmp_path)
    # The bookkeeping covers the whole replay either way; only the text stops.
    text_only = ("failures", "mode", "seconds", "detail_steps")
    assert {k: v for k, v in stop.summary().items() if k not in text_only} == {
        k: v for k, v in full.summary().items() if k not in text_only
    }
    assert stop.first_fail == 1 and stop.passing_prefix == 1 and stop.exact < stop.total
    assert "step 1 is the first failure; 1 step passes before it (step 0)." in stop.text
    assert "--- Step 1: ACTION2" in stop.text and "--- Step 2" not in stop.text and "    step 2 " not in stop.text
    assert "All mismatching steps" not in stop.text and "Per level" not in stop.text
    assert "Reproduce in python:" in stop.text
    assert "The report stops after 1 failing step; later steps are not reported (run_tests(failures=n) lists up to 10)." in stop.text
    assert replay_test(engine, tiny_trace, stop_on_fail=True, scratch_root=tmp_path).text == stop.text
    # failures=None (the final test) reports everything, as before.
    assert "All mismatching steps" in full.text and "Per level" in full.text and "--- Step 2" in full.text


def test_further_failures_get_one_line_each(tmp_path: Path, tiny_trace: Trace) -> None:
    engine = _simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "2"))
    full = replay_test(engine, tiny_trace, scratch_root=tmp_path)
    failing = [c.index for c in full.checks if not c.ok]
    assert len(failing) >= 3, full.text
    three = replay_test(engine, tiny_trace, failures=3, scratch_root=tmp_path, images=True)
    lines = three.text.splitlines()
    assert "--- Step 1: ACTION2" in three.text and len(three.images) == 1  # images: the first failure only
    head = lines.index("  Further failures (2), one line each:")
    for k, line in zip(failing[1:3], lines[head + 1 : head + 3]):
        assert line.startswith(f"    step {k} ACTION"), line
        assert "px differ in" in line and "[1] rows" in line and ("yours: #" in line or "no sprite of yours" in line), line
    assert lines.index("  Reproduce in python:") < head  # the command belongs to the first failure
    last = lines[head + 3]
    assert last == "  No other step fails." if len(failing) == 3 else last.startswith("  The report stops after 3 failing steps")
    everything = replay_test(engine, tiny_trace, failures=99, scratch_root=tmp_path)  # clamped to 10
    assert everything.failures == 10 and f"Further failures ({min(9, len(failing) - 1)})" in everything.text
    assert replay_test(engine, tiny_trace, failures=0, scratch_root=tmp_path).failures == 1


def test_a_contract_failure_is_reported_with_the_replay(tmp_path: Path, tiny_trace: Trace) -> None:
    path = _simple_engine(tmp_path, SIMPLE_TINY_GAME.replace("DOWN", "2"))
    path.write_text(path.read_text().replace("    layer: int = 0  # higher", "    layer: int = 1  # higher"), encoding="utf-8")
    stop = replay_test(path, tiny_trace, failures=1, scratch_root=tmp_path)
    assert "FAILED the FIXED INTERFACE block is unchanged" in stop.text
    assert "--- Step 1" in stop.text and "step 1 is the first failure" in stop.text
    assert stop.total == len(tiny_trace) and stop.first_fail == 1


def test_level_only_replays_that_level(tmp_path: Path, two_level_trace: Trace) -> None:
    from engine_re.tester import level_span

    assert two_level_trace.level_starts() == {0: 0, 1: 3}
    assert level_span(two_level_trace, 0) == (0, 0, 4) and level_span(two_level_trace, 1) == (3, 4, 9)
    engine = _simple_engine(tmp_path, SIMPLE_TWO.replace("WALL1", "8"))
    assert replay_test(engine, two_level_trace, scratch_root=tmp_path).passed
    one = replay_test(engine, two_level_trace, level=1, failures=1, scratch_root=tmp_path)
    assert one.passed and (one.first_step, one.total, one.level) == (4, 5, 1), one.text
    assert one.start_frame_diff == 0
    assert "level 1 only" in one.text and "ALL 5 STEPS OF LEVEL 1 MATCH (and its start frame)" in one.text
    zero = replay_test(engine, two_level_trace, level=0, scratch_root=tmp_path)
    assert zero.passed and (zero.first_step, zero.total) == (0, 4)
    with pytest.raises(ValueError, match="not a level"):
        replay_test(engine, two_level_trace, level=2, scratch_root=tmp_path)


def test_level_start_failure_is_explained_first(tmp_path: Path, two_level_trace: Trace) -> None:
    engine = _simple_engine(tmp_path, SIMPLE_TWO.replace("WALL1", "11"))
    report = replay_test(engine, two_level_trace, level=1, failures=1, scratch_root=tmp_path, images=True)
    assert not report.passed and report.start_frame_diff == 64 * 8 and report.passing_prefix == 0
    assert "the level 1 start frame differs (512 px): the first failure, so 0 steps pass before it." in report.text
    assert "--- Level 1 start" in report.text and "--- Step" not in report.text
    assert "[1] rows 56-63, cols 0-63 (your grid cells x 0-7, y 7): 512 px differ, expected->got 8->11 x512" in report.text
    assert '#3 "wall" tags=(wall) layer=0 x=0 y=7 size=8x1 visible collidable (shows at 512 of these px)' in report.text
    assert "before, after = try_step(3, level=1)" in report.text
    assert report.detail_steps == [3] and len(report.images) == 1 and report.images[0].png.startswith(b"\x89PNG")


def test_find_regions_clusters_differences_with_a_gap_tolerance() -> None:
    from engine_re import diff_report

    expected = np.zeros((64, 64), np.int8)
    got = expected.copy()
    got[10:12, 10:12] = 9
    got[10:12, 14] = 9  # two empty pixels from the block before: the same region
    got[40, 40] = 3
    got[60:62, 2:6] = 8
    regions, hidden = diff_report.find_regions(expected, got)
    assert hidden == 0 and [r.n for r in regions] == [1, 2, 3]
    assert [r.core for r in regions] == [(10, 10, 11, 14), (40, 40, 40, 40), (60, 2, 61, 5)]
    assert regions[0].box == (9, 9, 12, 15) and regions[0].changes == [(0, 9, 6)] and regions[0].pixels == 6
    many = expected.copy()
    many[::8, ::8] = 1  # 64 isolated pixels: only the largest few are numbered
    regions, hidden = diff_report.find_regions(expected, many)
    assert len(regions) == diff_report.MAX_REGIONS and hidden == 64 - diff_report.MAX_REGIONS


def test_comparison_image_boxes_every_region_on_both_frames() -> None:
    from engine_re import diff_report

    expected = np.full((64, 64), 4, np.int8)
    got = expected.copy()
    got[20:24, 30:34] = 9
    got[50, 5] = 11
    regions, _ = diff_report.find_regions(expected, got)
    img = diff_report.comparison_image(got, expected, regions, left_title="YOUR ENGINE", right_title="ORIGINAL GAME")
    scale, pad, head = diff_report.UPSCALE, 12, 30
    assert img.size == (2 * 64 * scale + 3 * pad, head + 64 * scale + pad)
    pixels = np.asarray(img)
    for left in (pad, 2 * pad + 64 * scale):  # both panels
        for region in regions:
            r0, c0, r1, c1 = region.box
            top, mid = head + r0 * scale + 1, left + (c0 + c1 + 1) * scale // 2
            assert tuple(pixels[top, mid]) == diff_report.BOX_RGB
        inside = pixels[head + 22 * scale, left + 32 * scale]  # a differing pixel is not covered
        assert tuple(inside) == (diff_report.PALETTE[9] if left == pad else diff_report.PALETTE[4])
    png = diff_report.png_bytes(img)
    assert png.startswith(b"\x89PNG") and diff_report.data_url(png).startswith("data:image/png;base64,")


def test_palette_matches_the_main_harness() -> None:
    from engine_re.diff_report import PALETTE
    from inference.agent.vision_context import ARC_COLOR_MAP

    assert PALETTE == ARC_COLOR_MAP


def test_regions_list_the_sprites_drawn_there() -> None:
    from engine_re import diff_report
    from engine_re.game_api import canonical, render, state_summary

    api = canonical()

    def state(x: int, ghost: bool = True):
        sprites = [
            api.Sprite([[0] * 6 for _ in range(6)], layer=-1, collidable=False, name="background"),
            api.Sprite([[9]], x=x, y=1, name="player", tags=("player",)),
        ]
        if ghost:
            sprites.append(api.Sprite([[7]], x=3, y=1, visible=False, name="ghost"))
        return api.State(grid=(6, 6), sprites=sprites, vars={"moves": x})  # scale 10, offset (2, 2)

    live = state(1)
    before = state_summary(live)
    live.sprites[1].move(1, 0)  # the step, in place, as step() does
    live.vars["moves"] = 2
    expected = render(state(3, ghost=False))
    expected[0, 30] = 11  # in the border, where nothing of the engine draws
    lines, regions = diff_report.describe_frames(expected, render(live), before, state_summary(live), crops=False, images=True)
    text = "\n".join(lines)
    assert len(regions) == 2
    assert "[1] rows 0, cols 30 (outside your grid)" in text
    assert "no sprite of yours draws here (the empty screen is colour 5); the original shows 11 (yellow) x1" in text
    assert "[2] rows 12-21, cols 22-41 (your grid cells x 2-3, y 1)" in text
    assert '#1 "player" tags=(player) layer=0 x=2 y=1 size=1x1 visible collidable (shows at 100 of these px); this step: x 1->2' in text
    assert '#2 "ghost" tags=() layer=0 x=3 y=1 size=1x1 HIDDEN (visible=False) collidable (hidden; would draw 100 of these px)' in text
    assert "#0 \"background\" tags=() layer=-1 x=0 y=0 size=6x6 visible not collidable (shows at 100 of these px)" in text
    assert "your state.vars changed: moves 1->2" in text


def test_state_summary_follows_the_view() -> None:
    from engine_re.game_api import canonical, state_summary, unpack_footprint

    api = canonical()
    sprite = api.Sprite([[9, 9]], x=1, y=1)
    plain = state_summary(api.State(grid=(8, 8), sprites=[sprite]))
    turned = state_summary(api.State(grid=(8, 8), sprites=[sprite], view=api.View(rotation=180)))
    assert plain["sprites"][0]["box"] == [8, 8, 15, 23]
    assert turned["sprites"][0]["box"] == [48, 40, 55, 55]
    assert unpack_footprint(turned["sprites"][0]).sum() == 2 * 64


PRINTING_STEP = """
def step(state, action):
    moves = {1: (0, -1), 2: (0, DOWN), 3: (-1, 0), 4: (1, 0)}
    print("action", action.id, "player at", state.vars["player"].x, state.vars["player"].y)
    if action.id in moves:
        state.try_move(state.vars["player"], *moves[action.id])
"""


def _printing_tiny_game(down: str) -> str:
    game = SIMPLE_TINY_GAME[: SIMPLE_TINY_GAME.index("def step(state, action):")] + PRINTING_STEP
    return game.replace("DOWN", down)


def test_printed_command_reproduces_the_failure_in_the_kernel(tmp_path: Path, tiny_trace: Trace) -> None:
    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = _simple_engine(workspace, _printing_tiny_game("2"))
    report = replay_test(engine, tiny_trace, failures=1, scratch_root=tmp_path)
    assert "your engine printed during this step:\n      action 2 player at 1 1" in report.text
    command = _repro_commands(report.text)
    assert command.startswith("before, after = try_step(1)\n# replays steps 0-0")
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        out = kernel.execute(command)  # exactly as printed
        assert "Traceback" not in out, out
        assert out.startswith("try_step(1): ACTION2 on your engine after replaying step 0 (level 0)\n")
        assert "your engine printed during the step:\n  action 2 player at 1 1\n" in out
        assert 'what the step changed in your state:\n  #2 "": y 1->3\n  vars: unchanged\n' in out
        assert "compared with the recording after step 1 (expected = the original, got = yours):" in out
        assert "[1] rows 16-31, cols 8-15 (your grid cells x 1, y 2-3)" in out and '#2 "" tags=(player)' in out
        assert kernel.execute("print(type(before).__name__, after.sprites[2].y - before.sprites[2].y)").strip() == "State 2"
        out = kernel.execute("b, a = try_step(1, action=4)")
        assert '#2 "": x 1->2' in out and "not compared with the recording: the action is not step 1's recorded one (ACTION2)" in out
        out = kernel.execute("b, a = try_step(2, state=after); print(a.sprites[2].y)")
        assert "on your engine from the state you gave" in out and "(from the state you gave)" in out
        assert out.strip().endswith("5")  # y=3 after step 1, and this engine moves down by 2
    finally:
        kernel.stop()


def test_try_step_compares_a_level_start(tmp_path: Path, two_level_trace: Trace) -> None:
    two_level_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = _simple_engine(workspace, SIMPLE_TWO.replace("WALL1", "11"))
    report = replay_test(engine, two_level_trace, level=1, failures=1, scratch_root=tmp_path)
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        out = kernel.execute(_repro_commands(report.text) + "\nprint(before, after.level)")
        assert "Traceback" not in out, out
        assert "try_step(3, level=1): your make_level(1) as the test starts it" in out
        assert '#3 "wall"' in out and "expected->got 8->11 x512" in out and out.strip().endswith("None 1")
        out = kernel.execute("b, a = try_step(5, level=1)")
        assert "on your engine started at level 1, after replaying step 4 (level 1)" in out and '#2 "player": x 1->2' in out
        out = kernel.execute("b, a = try_step(3)")  # the full replay: step 3 completes level 0
        assert "the level changed from 0 to 1: your state is now a fresh copy of make_level(1)" in out
    finally:
        kernel.stop()


def test_engine_prints_are_captured_per_step_and_capped(tmp_path: Path) -> None:
    from engine_re.candidate_runner import RUN_PRINT_LIMIT

    noisy = SIMPLE_TINY_GAME.replace("DOWN", "1").replace(
        "def step(state, action):", "def step(state, action):\n    print('step', action.id)\n    print('z' * 9000)"
    )
    engine = _simple_engine(tmp_path, noisy)
    actions = [Action(0).to_json()] + [Action(2 + k % 2).to_json() for k in range(300)]
    meta = {"win_levels": 1, "available_actions": [1, 2, 3, 4], "levels": [0]}
    result, frames = run_candidate(engine, actions, scratch_root=tmp_path, meta=meta, inspect=[300])
    assert result["error"] is None and len(frames) == len(actions)
    prints = result["prints"]
    assert "0" not in prints and prints["1"].startswith("step 2\nzzz") and "characters not kept" in prints["1"]
    assert all(len(text) < 2100 for text in prints.values())
    assert sum(len(text) for text in prints.values()) <= RUN_PRINT_LIMIT + 2100
    assert result.get("prints_dropped", 0) > 0 and prints["300"].startswith("step 3")  # inspected: always kept
    assert result["stdout"] == ""  # nothing reached the process's own stdout


def test_state_changes_track_sprites_by_identity() -> None:
    from engine_re.diff_report import state_changes
    from engine_re.game_api import canonical, state_summary

    api = canonical()
    a, b, c = (api.Sprite([[1]], name=n) for n in "abc")
    state = api.State(grid=(4, 4), sprites=[a, b, c], vars={"n": 1})
    before = state_summary(state)
    state.remove(a)  # b and c move up one index
    b.move(1, 0)
    c.visible = False
    state.add(api.Sprite([[2]], x=3, y=3, name="d"))
    state.vars["n"] = 2
    state.status = "level_solved"
    lines = state_changes(before, state_summary(state))
    assert lines == [
        '#0 "b": x 0->1 (it was #1 before the step)',
        '#1 "c": hidden (it was #2 before the step)',
        'added #2 "d" at x=3 y=3',
        'removed: #0 "a" (its index before the step)',
        "vars: n 1->2",
        "status: playing -> level_solved",
    ]


def _image_messages(agent) -> list[tuple[int, dict]]:
    return [(k, m) for k, m in enumerate(agent.messages) if m["role"] == "user" and isinstance(m["content"], list)]


def test_agent_sends_the_latest_test_images_after_the_tool_messages(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    wrong = _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "2"))
    model = _ScriptedModel(
        [
            [("python", {"code": wrong})],  # tested automatically
            [("run_tests", {})],
            [("run_tests", {"level": 0, "failures": 3})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=3), opening=False, client=model)
    agent.run()
    images = _image_messages(agent)
    assert len(images) == 3 and agent.result.image_messages == 3
    for k, _ in images:
        assert agent.messages[k - 1]["role"] == "tool"  # after the turn's tool messages
    assert all(isinstance(m["content"], str) for m in agent.messages if m["role"] == "tool")
    latest = images[-1][1]["content"]
    assert sum(p["type"] == "image_url" for p in latest) == 1  # the first failure's picture only
    assert "Further failures (" in agent.messages[images[-1][0] - 1]["content"]
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in latest if p["type"] == "image_url")
    for _, earlier in images[:-1]:  # only the latest test keeps its images
        assert all(p["type"] == "text" for p in earlier["content"]) and any("omitted" in p["text"] for p in earlier["content"])
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    saved = [path for r in records if "images" in r for path in r["images"]]
    assert saved == ["images/turn001_step1_auto.png", "images/turn002_step1.png", "images/turn003_step1.png"]
    assert all((tmp_path / path).read_bytes().startswith(b"\x89PNG") for path in saved)
    assert "base64" not in (tmp_path / "transcript.jsonl").read_text()


def test_agent_without_images_keeps_a_text_diff(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    wrong = _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "2"))
    model = _ScriptedModel([[("python", {"code": wrong})]])
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=1), opening=False, client=model, images=False)
    agent.run()
    assert not _image_messages(agent) and not (tmp_path / "images").exists()
    auto = agent.messages[-1]["content"]
    assert "[harness] engine.py changed" in auto and "one hex digit per pixel" in auto and "--- Step 1" in auto
    assert len(auto) < 4000


def test_agent_level_tests_are_kept_apart_from_full_replays(tmp_path: Path, two_level_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig

    two_level_trace.save(tmp_path / "trace")
    wrong = _rewrite_call(SIMPLE_TWO.replace("WALL1", "11"), game="two")
    model = _ScriptedModel(
        [
            [("python", {"code": wrong})],
            [("run_tests", {"level": 1})],
            [("run_tests", {"level": "1", "failures": 30})],  # clamped to 10
        ]
    )
    agent = EngineAgent("two", tmp_path, ModelConfig(), Budget(max_turns=3), opening=False, client=model)
    agent.run()
    tests = [json.loads(line) for line in (tmp_path / "tests.jsonl").read_text().splitlines()]
    assert [(t["level"], t["from_level"], t["failures"], t["total"]) for t in tests] == [
        (None, None, 1, 9), (1, 1, 1, 5), (1, 1, 10, 5),
    ]
    assert [t["passing_prefix"] for t in tests] == [tests[0]["first_fail"], 0, 0]  # level 1's start frame differs
    assert agent.result.best["total"] == 9 and agent.best_key == (tests[0]["passing_prefix"], tests[0]["exact"])
    final = (tmp_path / "final_test.txt").read_text()
    assert "All mismatching steps" in final  # the authoritative final test is the full report


def test_compaction_handles_image_messages(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    agent = EngineAgent("tiny", tmp_path, ModelConfig(keep_recent_tool_outputs=1), Budget(), opening=False, client=_ScriptedModel([]))
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    agent.messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "a", "content": "x" * 1000},
        {"role": "user", "content": [{"type": "text", "text": "images"}, picture]},
        {"role": "tool", "tool_call_id": "b", "content": "y" * 1000},
    ]
    agent._compact()
    assert agent.messages[2]["content"][1] == picture and "elided" in agent.messages[1]["content"]


def test_the_tools_are_python_run_tests_and_finish() -> None:
    from engine_re.prompts import TOOLS, system_prompt, tools

    assert [t["function"]["name"] for t in TOOLS] == ["python", "run_tests", "finish"]
    assert [t["function"]["name"] for t in tools(False)] == ["python", "run_tests", "finish"]
    run_tests = TOOLS[1]["function"]
    assert set(run_tests["parameters"]["properties"]) == {"level", "failures"}
    assert run_tests["description"].startswith("Run the contract tests, then replay the recording in order")
    python = TOOLS[0]["function"]["description"]
    for name in ("read(", "edit(", "undo(", "render(", "show(", "try_step(", "auto_sprites(", "S[i].last"):
        assert name in python
    assert "as images" in python and "images are off" in tools(False)[0]["function"]["description"]
    assert "image" not in system_prompt(images=False).lower() and "as images" in system_prompt(images=True)
    for term in ("camera", "letterbox", "letter_box", "ARCBaseGame", "arcengine", "library"):
        assert term not in system_prompt() and term not in python


def test_the_fixed_block_comment_matches_the_prompt() -> None:
    from engine_re.game_api import FIXED_INTERFACE, same_interface
    from engine_re.prompts import system_prompt

    for term in ("camera", "letterbox", "ARCBaseGame", "library"):
        assert term not in FIXED_INTERFACE
    comment = " ".join(line.lstrip("# ").strip() for line in FIXED_INTERFACE.splitlines() if line.startswith("#"))
    prompt = " ".join(system_prompt().split())
    for phrase in ("lowest layer first, sprites on the same layer in list order (later on top)",
                   "covers screen pixels x in [ox + gx*s, ox + gx*s + s), y in [oy + gy*s, oy + gy*s + s)",
                   "or min(64 // w, 64 // h) when that is None", "Turn the finished frame clockwise by state.view.rotation"):
        assert phrase in " ".join(comment.split()) and phrase in prompt
    # An engine whose block differs only in comments (an earlier run's) still has the same interface.
    old = FIXED_INTERFACE.replace("# The harness runs this module:", "# An older comment.")
    assert same_interface(old) and not same_interface(FIXED_INTERFACE.replace("layer: int = 0", "layer: int = 1"))


# --- auto_sprites ----------------------------------------------------------------------------

LEVEL_STARTS = Path(__file__).with_name("fixtures") / "engine_re_level_starts.npz"


def test_auto_sprites_redraws_every_reference_level_start_exactly() -> None:
    from engine_re.auto_sprites import guess_grid, sprite_code

    with np.load(LEVEL_STARTS) as data:
        frames, grids, names = data["frames"], data["grids"], data["names"]
    assert len(frames) == 34
    right = 0
    for frame, (w, h, s), name in zip(frames, grids, names):
        guess = guess_grid(frame)
        right += (guess.width, guess.height, guess.scale) == (w, h, s)
        assert guess.scale <= s, f"{name}: guessed scale {guess.scale}, the real one is {s}"
        code = sprite_code(frame, guess)
        assert code.exact, f"{name}: {code.differing} pixels differ"
    assert right >= 33  # sp80 level 5 draws its grid's outer cells in the border colour


def _level_start(game: str, level: int) -> np.ndarray:
    with np.load(LEVEL_STARTS) as data:
        names = [str(n) for n in data["names"]]
        return data["frames"][next(k for k, n in enumerate(names) if n.startswith(f"{game}:{level}:"))]


def test_auto_sprites_reuses_an_earlier_levels_kinds_turned() -> None:
    from engine_re import game_api
    from engine_re.auto_sprites import SHAPE_PIXELS, guess_grid, kinds_summary, sprite_code

    # vc33 draws each level as one picture turned by a multiple of 90 degrees: level 0 by 270, level 2 not at all.
    namespace = dict(vars(game_api.canonical()))
    exec(SHAPE_PIXELS, namespace)
    first = sprite_code(_level_start("vc33", 0), guess_grid(_level_start("vc33", 0)))
    exec(first.code, namespace)  # as if level 0's code had been put into engine.py
    existing = {k: v for k, v in namespace.items() if k.startswith("SHAPE_")}
    frame = _level_start("vc33", 2)
    second = sprite_code(frame, guess_grid(frame), function="level_2_sprites", existing=existing, defined={*existing, "shape_pixels"})
    assert second.exact and second.reuse.get("turned", 0) >= 5, kinds_summary(second)
    assert kinds_summary(second).startswith(f"{second.pieces} pieces: {sum(second.reuse.values())} reuse existing kinds (")
    piece = next(n for n in existing if n.startswith("SHAPE_4_2x3_"))  # the floating piece's top
    assert f"Sprite(shape_pixels({piece}), x=46, y=21, rotation=90, tags=(" in second.code and piece in second.from_engine
    defined_again = [line for line in second.code.splitlines() if line.split(" = ")[0] in existing]
    assert not defined_again and "def shape_pixels" not in second.code


def test_auto_sprites_matches_turned_mirrored_scaled_and_recoloured_pieces() -> None:
    from engine_re.auto_sprites import GridGuess, kinds_summary, sprite_code

    f = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])  # chiral: its mirror image is not a rotation of it
    cells = np.zeros((16, 16), np.int16)

    def put(x: int, y: int, shape: np.ndarray, colour: int) -> None:
        cells[y : y + shape.shape[0], x : x + shape.shape[1]][shape > 0] = colour

    put(1, 1, f, 10)
    put(6, 1, np.rot90(f, -1), 10)  # turned 90 degrees clockwise
    put(11, 1, f[:, ::-1], 10)  # mirrored left-right
    put(1, 6, np.kron(f, np.ones((2, 2), int)), 10)  # scaled 2x
    put(9, 6, f, 12)  # recoloured
    put(1, 13, np.ones((1, 3), int), 9)  # a solid bar, and the same bar turned
    put(6, 12, np.ones((3, 1), int), 9)
    put(9, 13, np.ones((1, 6), int), 9)  # a longer bar: a new kind, not the first one scaled
    put(12, 10, np.ones((2, 1), int), 5)  # engine.py's WALL (rows of numbers), turned
    frame = np.repeat(np.repeat(cells, 4, axis=0), 4, axis=1).astype(np.int8)
    guess = GridGuess(16, 16, 4, 0, 0, 0, 1, "as given")
    code = sprite_code(frame, guess, existing={"WALL": [[5, 5]], "SPEED": 3, "NAMES": ["a", "b"]})
    assert code.exact, code.code
    assert kinds_summary(code) == "9 pieces: 6 reuse existing kinds (4 turned, 1 scaled, 1 recoloured; 1 kind from engine.py), 3 new kinds"
    lines = [line.strip() for line in code.code.splitlines() if line.strip().startswith("Sprite(shape_pixels(")]
    first = lines[0].split("(")[2].split(")")[0]
    assert lines[0].startswith(f"Sprite(shape_pixels({first}), x=1, y=1, tags=(")
    assert any(line.startswith(f"Sprite(shape_pixels({first}), x=6, y=1, rotation=90,") for line in lines)
    assert any(line.startswith(f"Sprite(shape_pixels({first}), x=11, y=1, mirror_lr=True,") for line in lines)
    assert any(line.startswith(f"Sprite(shape_pixels({first}), x=1, y=6, scale=2,") for line in lines)
    assert any(line.startswith(f"Sprite(shape_pixels({first}, {{10: 12}}), x=9, y=6,") for line in lines)
    assert any(line.startswith("Sprite(shape_pixels(WALL), x=12, y=10, rotation=90,") for line in lines)
    assert code.from_engine == ["WALL"] and "WALL =" not in code.code


def test_guess_grid_takes_a_coarse_scale_only_with_evidence() -> None:
    from engine_re.auto_sprites import guess_grid

    rng = np.random.default_rng(0)
    cells = rng.integers(6, 12, size=(10, 20))  # a 20x10 grid at scale 3, centred, on a border of 5
    frame = np.full((64, 64), 5, np.int8)
    frame[17:47, 2:62] = np.repeat(np.repeat(cells, 3, axis=0), 3, axis=1)
    g = guess_grid(frame)
    assert (g.width, g.height, g.scale, g.x_offset, g.y_offset, g.border) == (20, 10, 3, 2, 17, 5)
    detail = frame.copy()
    detail[30, 30] = 0 if detail[30, 30] != 0 else 1  # one pixel off the 3x3 blocks: not a scale-3 grid
    assert guess_grid(detail).scale == 1
    moved = np.roll(frame, 1, axis=1)  # another frame of the "same level" breaking the blocks
    assert guess_grid([frame, moved]).scale == 1


def test_auto_sprites_helper_in_the_kernel(tmp_path: Path, tiny_trace: Trace) -> None:
    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        out = kernel.execute("code = auto_sprites(0)")
        assert "Traceback" not in out and "did not load" not in out, out
        assert "assumed 8x8 at scale 8" in out and "Renders the frame exactly: yes" in out
        assert "def level_0_sprites() -> list:" in out and "Not the real sprites" in out
        assert "SHAPE_9_1x1_" in out and "def shape_pixels" in out and "Kinds: 2 pieces: none reuses an existing kind, 2 new kinds." in out
        check = (
            "ns = {'Sprite': Sprite, 'State': State, 'View': View}; exec(code, ns)\n"
            "st = State(grid=(8, 8), sprites=ns['level_0_sprites']())\n"
            "print(bool((render(st) == S[0].last).all()), code.exact, auto_sprites(frame=S[3].last).grid)"
        )
        assert kernel.execute(check).strip().endswith("True True (8, 8)")
        out = kernel.execute("c = auto_sprites(frame=S[2].last, grid=(16, 16), region=(8, 8, 23, 31)); print(c.exact)")
        assert "Renders the region exactly: yes" in out and "def frame_sprites_region()" in out and "border" not in out.split("def frame_sprites_region")[1]
        # A pixel constant in engine.py is reused, not written again.
        (workspace / "engine.py").write_text("WALL = [[5] * 8]\n", encoding="utf-8")
        out = kernel.execute("code = auto_sprites(0)")
        assert "Kinds: 2 pieces: 1 reuse existing kinds (1 as they are; 1 kind from engine.py), 1 new kind." in out
        assert "  From engine.py: WALL." in out and "Sprite(shape_pixels(WALL), x=0, y=0, tags=(\"wall\",))" in out
        (workspace / "engine.py").write_text("WALL = [[5] * 8]\nraise SystemExit\n", encoding="utf-8")
        assert "(engine.py did not load, so its constants were not reused: SystemExit" in kernel.execute("code = auto_sprites(0)")
    finally:
        kernel.stop()


# --- read / edit / undo, the tools, finish and show ---------------------------------------------


def test_anchors_are_stable_and_stale_ones_are_rejected() -> None:
    from engine_re import hashline

    text = "".join(f"line {k}\n" for k in range(1, 21))
    lines, _ = hashline.split_lines(text)
    shown = hashline.render_read(text)
    assert shown.splitlines()[0].strip() == f"{hashline.anchor(lines, 1)}:line 1"
    assert all(len(a.split("#")[1].split(":")[0]) == 2 for a in shown.splitlines())
    assert all(c in hashline.NIBBLES for line in shown.splitlines() for c in line.split("#")[1][:2])
    before = {n: hashline.anchor(lines, n) for n in range(1, 21)}
    edited = hashline.apply_edits(text, [{"op": "replace", "pos": before[10], "lines": ["LINE 10"]}]).text
    after_lines, _ = hashline.split_lines(edited)
    after = {n: hashline.anchor(after_lines, n) for n in range(1, 21)}
    # Only the edited line and its neighbours get new hashes; distant anchors stay valid.
    changed = {n for n in before if before[n] != after[n]}
    assert 10 in changed and changed <= {9, 10, 11}
    assert hashline.split_lines(text)[0] == lines and hashline.anchor(lines, 5) == before[5]  # same input, same anchor
    with pytest.raises(hashline.EditError, match=r"\[E_STALE_ANCHOR\] 1 stale anchor: " + before[10]):
        hashline.apply_edits(edited, [{"op": "replace", "pos": before[10], "lines": ["x"]}])
    # A ":content" suffix is cross-checked: the right hash with the wrong content is stale too.
    with pytest.raises(hashline.EditError, match="E_STALE_ANCHOR"):
        hashline.apply_edits(text, [{"op": "replace", "pos": before[3] + ":line 4", "lines": ["x"]}])
    assert hashline.apply_edits(text, [{"op": "replace", "pos": before[3] + ":line 3", "lines": ["x"]}]).text.count("x\n") == 1
    # Paging: read() says where to continue.
    assert "Use offset=16 to continue." in hashline.render_read(text, offset=11, limit=5)


def test_every_edit_op() -> None:
    from engine_re import hashline

    text = "a\nb\nc\nd\ne\n"
    lines, _ = hashline.split_lines(text)
    A = {n: hashline.anchor(lines, n) for n in range(1, 6)}

    def run(*edits):
        return hashline.apply_edits(text, list(edits)).text

    assert run({"op": "replace", "pos": A[2], "lines": ["B"]}) == "a\nB\nc\nd\ne\n"
    assert run({"op": "replace", "pos": A[2], "end": A[4], "lines": "X\nY\n"}) == "a\nX\nY\ne\n"  # lines as one string
    assert run({"op": "replace", "pos": A[2], "end": A[3], "lines": []}) == "a\nd\ne\n"
    assert run({"op": "append", "pos": A[1], "lines": ["a2"]}) == "a\na2\nb\nc\nd\ne\n"
    assert run({"op": "append", "lines": "f\ng"}) == "a\nb\nc\nd\ne\nf\ng\n"
    assert run({"op": "prepend", "pos": A[5], "lines": ["d2"]}) == "a\nb\nc\nd\nd2\ne\n"
    assert run({"op": "prepend", "lines": ["start"]}) == "start\na\nb\nc\nd\ne\n"
    assert run({"op": "replace_text", "oldText": "c\nd", "newText": "C\nD"}) == "a\nb\nC\nD\ne\n"
    # Several edits validated against one snapshot, applied together.
    assert run({"op": "replace", "pos": A[1], "lines": ["A"]}, {"op": "replace", "pos": A[4], "lines": ["D"]}) == "A\nb\nc\nD\ne\n"
    for bad, code in (
        ([{"op": "replace", "pos": A[2], "lines": []}, {"op": "replace", "pos": A[3], "lines": []}], "E_EDIT_CONFLICT"),
        ([{"op": "replace", "pos": A[2], "lines": ["x"]}, {"op": "append", "pos": A[2], "lines": ["y"]}], "E_EDIT_CONFLICT"),
        ([{"op": "replace_text", "oldText": "zzz", "newText": "y"}], "E_NO_MATCH"),
        ([{"op": "append", "pos": A[2], "lines": []}], "E_BAD_OP"),
        ([{"op": "replace", "pos": "2", "lines": ["x"]}], "E_BAD_REF"),
        ([{"op": "replace", "pos": A[1], "lines": [f"{A[1]}:a"]}], "E_INVALID_PATCH"),
        ([{"op": "move", "pos": A[1]}], "E_BAD_OP"),
    ):
        with pytest.raises(hashline.EditError, match=code):
            hashline.apply_edits(text, bad)
    assert hashline.apply_edits(text, [{"op": "replace", "pos": A[1], "lines": ["a"]}]).noop


def test_edits_inside_the_fixed_block_are_rejected(tmp_path: Path) -> None:
    from engine_re import hashline
    from engine_re.engine_files import EngineEditor
    from engine_re.game_api import fixed_block_lines
    from engine_re.skeleton import render_skeleton

    engine = tmp_path / "workspace" / "engine.py"
    engine.parent.mkdir()
    source = render_skeleton("tiny", [1, 2, 3, 4])
    engine.write_text(source)
    first, last = fixed_block_lines(source)
    lines, _ = hashline.split_lines(source)
    editor = EngineEditor(engine, tmp_path / "engine_versions", tmp_path)
    for edit in (
        {"op": "replace", "pos": hashline.anchor(lines, first + 40), "lines": ["# mine"]},
        {"op": "append", "pos": hashline.anchor(lines, first), "lines": ["# mine"]},
        {"op": "replace_text", "oldText": "    layer: int = 0", "newText": "    layer: int = 1"},
    ):
        reply = editor.handle({"op": "edit", "edits": [edit]})
        assert not reply["ok"] and "[E_FIXED_BLOCK]" in reply["text"]
    assert engine.read_text() == source
    below = editor.handle({"op": "edit", "edits": [{"op": "append", "pos": hashline.anchor(lines, last), "lines": ["", "X = 1"]}]})
    assert below["ok"] and "Syntax OK" in below["text"] and "X = 1" in engine.read_text()


def test_python_cannot_write_engine_py(tmp_path: Path, tiny_trace: Trace) -> None:
    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = workspace / "engine.py"
    engine.write_text("ORIGINAL = 1\n")
    (workspace / "other.py").write_text("x = 1\n")
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        for code in (
            "open('engine.py', 'w').write('x')",
            "open('engine.py', 'a').write('x')",
            "open('engine.py', 'r+').write('x')",
            "import os; os.open('engine.py', os.O_WRONLY)",
            "import os; os.replace('other.py', 'engine.py')",
            "import os; os.remove('engine.py')",
            "import os; os.link('engine.py', 'hard.py')",
            "import os; os.symlink('engine.py', 'soft.py'); open('soft.py', 'w').write('x')",
            "import shutil; shutil.copyfile('other.py', 'engine.py')",
            "import os; os.rename('.', '../moved')",
            "import pathlib; pathlib.Path('engine.py').write_text('x')",
        ):
            out = kernel.execute(code)
            assert "PermissionError: sandbox" in out, (code, out)
        assert engine.read_text() == "ORIGINAL = 1\n"
        assert kernel.execute("print(open('engine.py').read().strip())").strip() == "ORIGINAL = 1"  # reading is fine
        assert kernel.execute("import shutil; shutil.copyfile('engine.py', 'copy.py'); print('ok')").strip() == "ok"
        out = kernel.execute("edit(edits=[{'op': 'replace_text', 'oldText': 'ORIGINAL = 1', 'newText': 'CHANGED = 2'}])")
        assert "engine.py: replaced line 1 with 1 line. Syntax OK. (version 2" in out
        assert engine.read_text() == "CHANGED = 2\n"
        assert "engine_versions" not in kernel.execute("import os; print(os.listdir('.'))")
        assert "sandbox: listing" in kernel.execute("import os; os.listdir('../engine_versions')")
    finally:
        kernel.stop()


def test_undo_restores_earlier_versions_and_the_best(tmp_path: Path) -> None:
    import json

    from engine_re.engine_files import EngineEditor, best_key, sha256

    # The best version: the most steps passing before the first failure, then the most in all.
    assert best_key({"passing_prefix": 5, "exact": 6}) > best_key({"passing_prefix": 4, "exact": 9}) > best_key({"passing_prefix": 4, "exact": 8})
    assert best_key({"first_fail": 4, "exact": 7, "total": 9}) == (4, 7)  # entries written before passing_prefix
    assert best_key({"first_fail": None, "exact": 9, "total": 9}) == (9, 9)

    engine = tmp_path / "workspace" / "engine.py"
    engine.parent.mkdir()
    engine.write_text("A = 1\n")
    log: list[dict] = []
    editor = EngineEditor(engine, tmp_path / "engine_versions", tmp_path, log=log.append)

    def edit(old: str, new: str) -> str:
        return editor.handle({"op": "edit", "edits": [{"op": "replace_text", "oldText": old, "newText": new}]})["text"]

    edit("A = 1", "A = 2")  # v2 (v1 is the start)
    edit("A = 2", "A = 3")  # v3
    (tmp_path / "engine_best.py").write_text("A = 2\n")
    tested = {"engine_sha": sha256("A = 2\n"), "level": None, "from_level": None, "exact": 7, "total": 9, "first_fail": 4, "passed": False}
    (tmp_path / "tests.jsonl").write_text(json.dumps(tested) + "\n")
    out = editor.undo()
    assert engine.read_text() == "A = 2\n" and "Restored version 2" in out and "saved as version 4" in out
    assert "tested: 4 pass before the first failure (step 4), 7/9 in all" in out and "read() again" in out
    editor.undo()  # undo the undo: back to A = 3
    assert engine.read_text() == "A = 3\n" and [v["version"] for v in editor.versions()] == [1, 2, 3, 4, 5]
    out = editor.undo(3)  # the state 3 changes ago: version 2
    assert engine.read_text() == "A = 2\n" and "Restored version 2" in out and "saved as version 6" in out
    editor.undo(to=1)
    assert engine.read_text() == "A = 1\n"
    out = editor.undo(to="best")
    assert engine.read_text() == "A = 2\n"
    assert out.startswith(
        "Restored the best tested version (the most steps passing before the first failure, ties broken by the most "
        "steps passing in all): v6, tested: 4 pass before the first failure (step 4), 7/9 in all, saved as version 8."
    )
    assert [r["engine_change"]["op"] for r in log] == ["edit", "edit", "undo", "undo", "undo", "undo", "undo"]
    assert log[0]["engine_change"]["diff"].splitlines()[-2:] == ["-A = 1", "+A = 2"]
    assert not editor.handle({"op": "undo", "n": 99})["ok"]


def test_finish_runs_the_tests_and_ends_only_when_they_pass(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("finish", {"summary": "nothing yet"})],  # fails: the session goes on
            [("python", {"code": _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "1"))}), ("finish", {"summary": "moves"})],
            [("python", {"code": "1"})],  # never reached
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), opening=False, client=model)
    result = agent.run()
    tool_outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_outputs[0].startswith("Not finished: the tests still fail, so the session goes on.")
    assert "--- Step 0: RESET" in tool_outputs[0]
    assert tool_outputs[-1] == "Every test passes. Session finished."
    assert result.status == "passed" and result.turns == 2 and result.finish_calls == 2 and result.finish_summary == "moves"
    assert result.tests_run == 2 and result.auto_tests == 0


def test_show_images_join_the_turns_image_message(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import IMAGE_PLACEHOLDER, Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("python", {"code": "show(S[0].last, S[1].last, titles=['a', 'b'], boxes=[(8, 8, 15, 15)])"}), ("run_tests", {})],
            [("python", {"code": "show(S[2].last)"})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), opening=False, client=model)
    agent.run()
    images = _image_messages(agent)
    assert len(images) == 2
    first, second = images[0][1]["content"], images[1][1]["content"]
    captions = [p["text"] for p in first if p["type"] == "text"]
    assert captions[1] == "show(): a | b; boxes 1" and any("From the latest run_tests report" in c for c in captions)
    assert captions.count(IMAGE_PLACEHOLDER) >= 2
    assert all(p["type"] == "text" for p in first)  # stripped once the next turn's images came
    assert sum(p["type"] == "image_url" for p in second) == 1
    assert "[image: 2 frame(s), a, b; it follows this output]" in agent.messages[3]["content"]
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert [r["show_images"] for r in records if "show_images" in r] == [["images/turn001_show1.png"], ["images/turn002_show1.png"]]
    assert (tmp_path / "images" / "turn001_show1.png").read_bytes().startswith(b"\x89PNG")


def _message_text(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(p["text"] for p in content if p["type"] == "text")


def test_the_harness_plays_the_first_round(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re import hashline
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=0), client=_ScriptedModel([]))
    result = agent.run()
    engine = (tmp_path / "workspace" / "engine.py").read_text()
    # auto_sprites(0)'s code sits above make_level, which returns its sprites.
    assert engine.index("def level_0_sprites()") < engine.index("def make_level(")
    assert "    return State(grid=(8, 8), sprites=level_0_sprites())" in engine
    # Level 0's first frame is drawn, so the first failure is the first action.
    assert result.opening == {"exact": True, "first_fail": 1, "passing_prefix": 1}
    # The harness's edit and test are not the model's.
    assert result.engine_changes == 0 and result.tests_run == 0
    tests = [json.loads(line) for line in (tmp_path / "tests.jsonl").read_text().splitlines()]
    assert [t["auto"] for t in tests] == ["opening"] and tests[0]["first_fail"] == 1
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert [r["by"] for r in records if "engine_change" in r] == ["harness"]
    # The first message: the recording in one sentence, what was done, the report, engine.py and the task.
    content = agent.messages[1]["content"]
    text = _message_text(agent.messages[1])
    assert "The recording: 8 steps over 1 level(s)" in text and "summarize_levels()" in text
    assert "Levels reached" not in text and "frames per step" not in text
    assert "Before your first turn the harness did the first round" in text
    assert "Renders the frame exactly: yes." in text and "def shape_pixels" not in text.split("TEST RESULT")[0]
    assert "step 1 is the first failure; 1 step passes before it (step 0)" in text
    assert "the FIXED block, folded" in text and "class Sprite:" not in text
    lines, _ = hashline.split_lines(engine)
    assert f"{hashline.anchor(lines, len(lines))}:{lines[-1]}" in text
    assert content[0]["text"].rstrip().split("\n")[-1].startswith("Your first task: make step 1 pass, the first action of level 0.")
    assert any(p["type"] == "image_url" for p in content)
    assert (tmp_path / "images" / "turn000_step1_opening.png").exists()


def test_first_message_without_the_opening(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig
    from engine_re.skeleton import render_skeleton

    tiny_trace.save(tmp_path / "trace")
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=0), opening=False, client=_ScriptedModel([]))
    agent.run()
    opening = agent.messages[1]["content"]
    assert (tmp_path / "workspace" / "engine.py").read_text() == render_skeleton("tiny", [1, 2, 3, 4])
    assert "The actions this game accepts: 1 (up), 2 (down), 3 (left), 4 (right)." in opening
    assert "the FIXED block, folded" in opening and "class Sprite:" not in opening
    assert "Your first task: put auto_sprites(0)'s code into make_level with edit()" in opening
    assert agent.messages[0]["content"].startswith("# Goal")
    assert not agent.result.opening


def test_the_opening_does_not_count_as_the_models_work_after_a_resume(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=1), client=_ScriptedModel([[("python", {"code": "1"})]])).run()
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), client=_ScriptedModel([[("python", {"code": "2"})]]))
    result = agent.run()
    assert result.resumes == 1 and result.engine_changes == 0 and result.tests_run == 0
    assert "This continues an earlier session" in _message_text(agent.messages[1])


def test_summarize_levels_lists_each_level(two_level_trace: Trace) -> None:
    from engine_re import helpers
    from engine_re.kernel import PRELOADED

    rows = helpers._levels_text(two_level_trace).splitlines()
    assert rows[0].startswith("The recording: 9 steps") and "2 of the game's 2 levels played" in rows[0]
    assert rows[2].split() == ["0", "S[0].last", "1-3", "(3)", "right", "x3", "3", "-", "-", "solved", "at", "step", "3"]
    assert rows[3].startswith("1      S[3].last    4-8 (5)") and rows[3].rstrip().endswith("not solved: the recording ends (NOT_FINISHED)")
    assert "RESET x1" in rows[3] and "summarize_levels" in PRELOADED


def test_a_resumed_session_keeps_the_versions_and_shows_anchors(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re import hashline
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    wrong = _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "2"))
    EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=1), opening=False, client=_ScriptedModel([[("python", {"code": wrong})]])).run()
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), opening=False, client=_ScriptedModel([[("python", {"code": "undo()"})]]))
    agent.run()
    opening = agent.messages[1]["content"]
    assert "This continues an earlier session on this game (1 turns)" in opening and "--- Step " in opening
    shown = opening.split("engine.py now, as read() shows it:")[1]
    assert "the FIXED block, folded; it cannot be edited" in shown and "class Sprite:" not in shown
    out = next(m["content"] for m in agent.messages if m["role"] == "tool")
    assert out.startswith("Restored version 1, as engine.py was 1 change ago, saved as version 3.")
    assert agent.result.resumes == 1 and agent.result.engine_changes == 2
    # The resume showed the engine as the first session left it (version 2), with valid anchors.
    lines, _ = hashline.split_lines((tmp_path / "engine_versions" / "v0002.py").read_text())
    assert f"{hashline.anchor(lines, len(lines))}:{lines[-1]}" in shown


def test_kernel_rejects_code_that_rebinds_a_builtin():
    from engine_re import kernel

    def show(*frames):
        return "the harness show"

    namespace = {"show": show, "print": print}
    builtins = {"show": show}
    out = kernel._run("def show(f):\n    pass\nprint('ran')", namespace, builtins)
    assert "nothing was run" in out and "line 1: def show" in out and "ran" not in out
    assert namespace["show"] is show
    for code in ("show = 3", "for show in range(2): pass", "import os as show", "f = lambda show: show", "del show"):
        assert "nothing was run" in kernel._run(code, namespace, builtins), code
    # Other names, calls and keyword arguments are fine.
    assert kernel._run("x = show()\nprint(x)\ndef f(frames, show_all=True): return frames", namespace, builtins).strip() == "the harness show"
    # A rebinding the check cannot see is undone after the run, and reported.
    out = kernel._run("globals()['show'] = 1", namespace, builtins)
    assert namespace["show"] is show and "restored" in out


def test_system_prompt_names_every_builtin_function():
    from engine_re.kernel import RESERVED
    from engine_re.prompts import system_prompt

    for images in (True, False):
        prompt = system_prompt(images=images)
        section = prompt[prompt.index("# Built-in python functions") : prompt.index("# How to work")]
        for name in RESERVED:
            assert name in section, name
        assert "reserved" in section


def test_openrouter_client_asks_again_after_a_provider_error(monkeypatch):
    from engine_re import agent as agent_mod

    answers = [
        {"choices": [{"finish_reason": "error", "error": {"message": "upstream failed"}, "message": {"content": ""}}]},
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": []}}], "usage": {}},
    ]

    class Resp:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    client = agent_mod.OpenRouterClient(agent_mod.ModelConfig(), api_key="test")
    monkeypatch.setattr(client.session, "post", lambda *a, **k: Resp(answers.pop(0)))
    monkeypatch.setattr(agent_mod.time, "sleep", lambda s: None)
    data = client.chat([], [])
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert len(client.provider_errors) == 1 and "upstream failed" in client.provider_errors[0]
