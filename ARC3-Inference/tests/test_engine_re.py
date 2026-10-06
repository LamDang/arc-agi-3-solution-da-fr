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
        assert kernel.execute("x = len(recording); x").strip() == str(len(ACTIONS))
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
    through edit_file(), as the model would (python cannot write engine.py)."""
    from engine_re.skeleton import render_skeleton

    old = _tail(render_skeleton(game, list(actions)))
    new = _tail(_engine_source(game_code))
    return f"edit_file(edits=[{{'op': 'replace_text', 'oldText': {old!r}, 'newText': {new!r}}}])"


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


def test_builtin_functions_called_as_tools_run_as_python(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig, builtin_call_code

    assert builtin_call_code("read_file", {"offset": 240, "limit": 110}) == "read_file(offset=240, limit=110)"
    assert builtin_call_code("edit_file", {"edits": '[{"op": "append", "lines": ["x"]}]'}) == "edit_file(edits=[{'op': 'append', 'lines': ['x']}])"
    assert builtin_call_code("show_frames", {"titles": "[not json"}) == "show_frames(titles='[not json')"
    assert builtin_call_code("read_file", {"path": "engine.py", "offset": "240"}) == "read_file(path='engine.py', offset=240)"
    tiny_trace.save(tmp_path / "trace")
    edits = json.dumps([{"op": "replace_text", "oldText": "# ==== YOUR GAME ====", "newText": "# ==== YOUR GAME ==== (changed)"}])
    model = _ScriptedModel(
        [
            [("read_file", {"offset": 1, "limit": 2})],
            [("edit_file", {"edits": edits}), ("nonsense", {})],  # edits as a JSON string
            [("python", {"code": "print(open('engine.py').read().count('(changed)'))"})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=3), opening=False, client=model)
    result = agent.run()
    outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    note = "[harness] read_file is a python function, not a tool; this call ran as python: read_file(offset=1, limit=2)\n"
    lines = outputs[0].splitlines()
    assert outputs[0].startswith(note) and lines[1].startswith("1#") and lines[2].startswith("2#") and "[Showing lines 1-2 of" in lines[-1]
    assert outputs[1].startswith("[harness] edit_file is a python function, not a tool; this call ran as python: edit_file(edits=[{'op': 'replace_text'")
    assert "engine.py: replaced line" in outputs[1] and result.engine_changes == 1
    assert outputs[2].startswith("Error: unknown tool 'nonsense'. The tools are python, run_tests and commit_engine.\n\n[harness] engine.py changed")
    assert outputs[3].startswith("1\n")
    assert result.tool_calls == {"python": 3, "nonsense": 1}  # counted as python calls
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert [(r["tool"], r.get("called_as")) for r in records if "tool" in r] == [
        ("python", "read_file"), ("python", "edit_file"), ("nonsense", None), ("python", None),
    ]


def test_the_automatic_test_repeats_the_same_failure_in_one_line(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    comment = "edit_file(edits=[{'op': 'append', 'lines': ['# a comment']}])"
    model = _ScriptedModel(
        [
            [("python", {"code": _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "2"))})],  # step 1 fails: the full report
            [("python", {"code": comment})],  # the same failure: one line
            [("python", {"code": _rewrite_now(SIMPLE_TINY_GAME.replace("DOWN", "3"))})],  # another failure: the report
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=3), opening=False, client=model)
    result = agent.run()
    assert result.auto_tests == 3
    outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert "[harness] engine.py changed, so it was tested automatically" in outputs[0] and "--- Step 1: ACTION2" in outputs[0]
    assert "--- Step 1" not in outputs[1]
    assert ("[harness] engine.py changed, tested automatically: the same result as the last test (step 1 fails the same way: "
            "final frame: ") in outputs[1]
    assert "[harness] engine.py changed, so it was tested automatically" in outputs[2] and "--- Step 1: ACTION2" in outputs[2]
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    logged = [r["auto_test"] for r in records if "auto_test" in r]
    assert len(logged) == 3 and all("--- Step 1: ACTION2" in text for text in logged)  # the log keeps the full reports
    tests = [json.loads(line) for line in (tmp_path / "tests.jsonl").read_text().splitlines()]
    assert tests[0]["signature"] == tests[1]["signature"] != tests[2]["signature"] and len(tests[0]["signature"]) == 12


def test_python_quota_pauses_until_engine_changes(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    change = "edit_file(edits=[{'op': 'replace_text', 'oldText': '# ==== YOUR GAME ====', 'newText': '# ==== YOUR GAME ==== (changed)'}])"
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


def test_an_engine_listing_is_elided_up_to_its_closing_line() -> None:
    from engine_re.prompts import ENGINE_ELIDED, elide_engine_listing, engine_block

    listing = "1#ABC:x = 1\n2#DEF:\n\n[Showing lines 1-2 of 9. Use offset=3 to continue.]"
    text = f"The report.\n\nFix step 2 was done.\n\n{engine_block(listing)}\n\nFix step 3; commit when the tests pass."
    assert elide_engine_listing(text) == f"The report.\n\nFix step 2 was done.\n\n{ENGINE_ELIDED}\n\nFix step 3; commit when the tests pass."
    assert elide_engine_listing(f"Head.\n\n{engine_block(listing)}") == f"Head.\n\n{ENGINE_ELIDED}"
    assert elide_engine_listing("nothing listed\n\nhere") == "nothing listed\n\nhere"


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
    """The python command the report prints after "Reproduce:", as it would be pasted."""
    line = next(line for line in text.splitlines() if line.startswith("  Reproduce: "))
    return line[len("  Reproduce: ") :].split("   #")[0]


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
    assert "  Reproduce: before, after = replay_step(1)" in stop.text.splitlines()
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
    assert lines.index("  Reproduce: before, after = replay_step(1)") < head  # the command belongs to the first failure
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
    assert "before, after = replay_step(3, level=1)" in report.text
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
    assert command == "before, after = replay_step(1)"
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        out = kernel.execute(command)  # exactly as printed
        assert "Traceback" not in out, out
        assert out.startswith("replay_step(1): ACTION2 on your engine after replaying step 0 (level 0)\n")
        assert "your engine printed during the step:\n  action 2 player at 1 1\n" in out
        assert 'what the step changed in your state:\n  #2 "": y 1->3\n  vars: unchanged\n' in out
        assert "compared with the recording after step 1 (expected = the original, got = yours):" in out
        assert "[1] rows 16-31, cols 8-15 (your grid cells x 1, y 2-3)" in out and '#2 "" tags=(player)' in out
        assert kernel.execute("print(type(before).__name__, after.sprites[2].y - before.sprites[2].y)").strip() == "State 2"
        out = kernel.execute("b, a = replay_step(1, action=4)")
        assert '#2 "": x 1->2' in out and "not compared with the recording: the action is not step 1's recorded one (ACTION2)" in out
        out = kernel.execute("b, a = replay_step(2, state=after); print(a.sprites[2].y)")
        assert "on your engine from the state you gave" in out and "(from the state you gave)" in out
        assert out.strip().endswith("5")  # y=3 after step 1, and this engine moves down by 2
        out = kernel.execute("b, a = replay_step(0)")  # step 0 matches: one line, no comparison
        assert "what the step changed in your state:" in out and out.strip().endswith("your frame matches the recording after step 0")
        assert "compared with the recording" not in out
    finally:
        kernel.stop()


def test_replay_step_compares_a_level_start(tmp_path: Path, two_level_trace: Trace) -> None:
    two_level_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = _simple_engine(workspace, SIMPLE_TWO.replace("WALL1", "11"))
    report = replay_test(engine, two_level_trace, level=1, failures=1, scratch_root=tmp_path)
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        out = kernel.execute(_repro_commands(report.text) + "\nprint(before, after.level)")
        assert "Traceback" not in out, out
        assert "replay_step(3, level=1): your make_level(1) as the test starts it" in out
        assert '#3 "wall"' in out and "expected->got 8->11 x512" in out and out.strip().endswith("None 1")
        out = kernel.execute("b, a = replay_step(5, level=1)")
        assert "on your engine started at level 1, after replaying step 4 (level 1)" in out and '#2 "player": x 1->2' in out
        out = kernel.execute("b, a = replay_step(3)")  # the full replay: step 3 completes level 0
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


def test_the_tools_are_python_run_tests_and_commit_engine() -> None:
    import re

    from engine_re.prompts import TOOLS, system_prompt, tools

    for mode, history in (("single", True), ("step", True), ("step", False)):
        schemas = tools(False, mode, history)
        assert [t["function"]["name"] for t in schemas] == ["python", "run_tests", "commit_engine"]
        commit = schemas[2]["function"]
        assert commit["parameters"]["required"] == ["message"] and "which guesses remain" in commit["description"]
        assert ("never move on" in commit["description"]) == (mode == "step")
    for mode in ("single", "step"):
        assert "commit_engine" in system_prompt(mode=mode) and not re.search(r"\bfinish\b", system_prompt(mode=mode))
    run_tests = TOOLS[1]["function"]
    assert set(run_tests["parameters"]["properties"]) == {"level", "failures"}
    assert run_tests["description"].startswith("Run the contract tests, then replay the recording in order")
    python = TOOLS[0]["function"]["description"]
    assert "# Objects" in python and "edit_file() and undo_edit()" in python
    objects = system_prompt()[system_prompt().index("# Objects") : system_prompt().index("# How to work")]
    assert "replica: module  engine.py as it is now" in objects and "Never `import engine`" in objects
    for text in (python, objects):  # the kernel's persistence, said plainly
        assert "persistent for the whole run" in text and "define helpers and data once and reuse them" in text
    for name in ("read_file(", "edit_file(", "undo_edit(", "render_state(", "show_frames(", "replay_step(",
                 "summarize_levels(", "recording[i]", "recording[k].after"):
        assert name in objects and name.split("(")[0].split("[")[0] in python, name
    assert "auto_sprites" not in system_prompt() and "auto_sprites" not in python
    for name in (".pieces_after: Pieces", ".changes: list[Change]", ".code() -> str", "Piece: a Sprite", "GridGuess:"):
        assert name in objects, name
    assert "as images" in objects and "as hex digits" in system_prompt(images=False)
    assert "image" not in system_prompt(images=False).lower() and "as images" in system_prompt(images=True)
    for term in ("camera", "letterbox", "letter_box", "ARCBaseGame", "arcengine", "library"):
        assert term not in system_prompt() and term not in python


def test_the_prompts_ask_for_parsimony_not_generality() -> None:
    from engine_re.agent import COMMIT_HINT
    from engine_re.prompts import system_prompt, tools

    for mode, history in (("single", True), ("step", True), ("step", False)):
        prompt = " ".join(system_prompt(mode=mode, history=history).split())  # line wraps differ between the modes
        texts = [prompt, COMMIT_HINT] + [t["function"]["description"] for t in tools(True, mode, history)]
        for gone in ("nobody recorded", "general rules hold up", "more general", "unrecorded", "unseen", "special cases do not",
                     "as simple and general"):
            assert not any(gone in text for text in texts), (mode, gone)
        for kept in ("most parsimonious model (Occam's razor): the fewest rules and assumptions", "every step observed so far",
                     "Per-level constants in the level data (a rate, a budget, a size) are fine", "do not hunt for one",
                     "replace it with the simplest rule that explains all the steps so far",
                     "Do not think about, model or write code for what has not been observed",
                     "only draw its first frame" if mode == "single" else "Only draw the new level's first frame",
                     "Never hard-code recorded frames or anything keyed to the step number.",
                     "# Sandbox Your python code runs sandboxed: it can read the workspace, the recording and the Python installation",
                     "write only inside the workspace (/tmp is refused); no network, no subprocesses", "only through edit_file() and undo_edit()"):
            assert kept in prompt, (mode, kept)


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


def test_pieces_in_the_kernel_replace_auto_sprites(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.kernel import FUNCTIONS, RESERVED, RESERVED_HISTORY, RESERVED_STEP

    assert all("auto_sprites" not in names for names in (FUNCTIONS, RESERVED, RESERVED_HISTORY, RESERVED_STEP))
    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        assert "NameError: name 'auto_sprites' is not defined" in kernel.execute("auto_sprites(0)")
        out = kernel.execute("code = recording[0].pieces_after.code(); print(code)")
        assert "Traceback" not in out, out
        assert "# ---- Sprites that draw recording[0].after, level 0's first frame: grid 8x8 at scale 8 ----" in out
        assert "def level_0_sprites() -> list:" in out and "SHAPE_9_1x1_" in out and "def shape_pixels" in out
        check = (
            "ns = {'Sprite': Sprite, 'State': State, 'View': View}; exec(code, ns)\n"
            "st = State(grid=(8, 8), sprites=ns['level_0_sprites']())\n"
            "p = recording[2].pieces_after\n"
            "print(bool((render_state(st) == recording[0].after).all()), recording[3].grid.grid, isinstance(p[2], Sprite),\n"
            "      bool((render_state(State(grid=p.grid.grid, sprites=list(p))) == recording[2].after).all()),\n"
            "      recording[0].pieces_before, recording[0].changes, recording[2].pieces_before is recording[1].pieces_after)"
        )
        assert kernel.execute(check).split() == ["True", "(8,", "8)", "True", "True", "None", "None", "True"]
        out = kernel.execute("print(recording[1].changes)")
        assert out.startswith("moved: SHAPE_9_1x1_") and "(1, 1) -> (1, 2), dx=+0 dy=+1" in out, out
        out = kernel.execute("recording[1].pieces_after")
        assert out.startswith("4 pieces on a 8x8 grid at scale 8, offset (0, 0)") and "  [3] SHAPE_9_1x1_" in out, out
        # A pixel constant in engine.py is reused, not written again.
        (workspace / "engine.py").write_text("WALL = [[5] * 8]\n", encoding="utf-8")
        out = kernel.execute("print(recording[0].pieces_after.code())")
        assert "Sprite(shape_pixels(WALL), x=0, y=0, tags=(\"wall\",))" in out and "WALL =" not in out
    finally:
        kernel.stop()


def test_step_views_never_segment_beyond_the_focus(monkeypatch, two_level_trace: Trace) -> None:
    from engine_re import auto_sprites, helpers, segment

    seen: list[int] = []  # the steps whose final frame was segmented or used to guess a grid
    where = {s.last.__array_interface__["data"][0]: i for i, s in enumerate(two_level_trace.steps)}

    def spy(found):
        def wrapped(frame, *args, **kwargs):
            seen.extend(where.get(f.__array_interface__["data"][0], -1) for f in (frame if isinstance(frame, list) else [frame]))
            return found(frame, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(segment, "guess_grid", spy(auto_sprites.guess_grid))
    monkeypatch.setattr(segment, "pieces", spy(segment.pieces))
    helpers.load_trace(two_level_trace, focus=4)  # the whole recording in memory, focused on step 4
    for view in helpers.recording[:5]:
        view.grid, view.pieces_before, view.pieces_after, view.changes
    helpers.summarize_levels()
    assert seen and max(seen) <= 4 and -1 not in seen
    for attribute in ("grid", "pieces_before", "pieces_after", "changes"):
        with pytest.raises(ValueError, match="only steps 0-4 are loaded"):
            getattr(helpers.recording[5], attribute)
    # Level 1 starts at step 3: its grid comes from steps 3 and 4 only; moving the focus guesses it again.
    assert helpers.recording[4].pieces_after.grid is helpers.recording[4].grid
    helpers.load_trace(two_level_trace, focus=6)
    seen.clear()
    helpers.recording[6].changes
    assert max(seen) == 6
    helpers.load_trace(two_level_trace)


def _sprite_fields(sprite) -> dict:
    import dataclasses

    out = {}
    for f in dataclasses.fields(sprite):
        value = getattr(sprite, f.name)
        out[f.name] = np.asarray(value).tolist() if f.name == "pixels" else value
    return out


def test_sprites_print_as_the_code_that_builds_them() -> None:
    import dataclasses
    import random

    from engine_re import game_api, segment
    from engine_re.skeleton import render_skeleton

    api = game_api.canonical()
    Sprite = api.Sprite
    rng = random.Random(0)
    names = ["", "player", "it's", 'say "hi"', "both ' and \"", "back\\slash", "new\nline", "naïve"]
    for _ in range(400):
        h, w = rng.randint(1, 6), rng.randint(1, 70)
        if rng.random() < 0.3:
            pixels = [[rng.choice([-2, -1, 3])] * w for _ in range(h)]
        elif rng.random() < 0.3:
            row = [rng.randint(-2, 15) for _ in range(w)]
            pixels = [list(row) for _ in range(h)]
        else:
            pixels = [[rng.randint(-2, 15) for _ in range(w)] for _ in range(h)]
        if rng.random() < 0.2:
            pixels = np.array(pixels, dtype=np.int8)
        kw = {}
        for name, choices in (("x", [0, 3, -2, 63]), ("y", [0, 1, -5]), ("layer", [0, -2, 9]), ("name", names),
                              ("tags", [(), ("wall",), ("a", 'b"c', "d'e"), ["x"]]), ("visible", [True, False]),
                              ("collidable", [True, False]), ("blocking", ["pixel", "box", "none"]),
                              ("rotation", [0, 90, 180, 270]), ("mirror_ud", [False, True]), ("mirror_lr", [False, True]),
                              ("scale", [1, 2, 3, -1]), ("screen", [False, True])):
            if rng.random() < 0.4:
                kw[name] = rng.choice(choices)
        if rng.random() < 0.1:
            kw["x"] = np.int64(7)
        sprite = Sprite(pixels, **kw)
        text = str(sprite)
        assert text == repr(sprite) and text.startswith("Sprite(") and "pixels=" not in text
        again = eval(text, {"Sprite": Sprite})  # noqa: S307
        assert _sprite_fields(again) == _sprite_fields(sprite), text
        for name, value in kw.items():  # only the fields that differ from their defaults are written
            default = Sprite.__dataclass_fields__[name].default
            assert (f" {name}=" in text) == (value != default), (name, text)
    assert str(Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border")) == (
        'Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border")'
    )
    shape = Sprite([[8, 8, -1], [8, 8, 8], [8, 8, 8], [8, 8, -1]], x=28, y=8, rotation=180, tags=("shape_8_3x4_79b9",))
    assert str(shape) == 'Sprite([[8, 8, -1], [8, 8, 8], [8, 8, 8], [8, 8, -1]], x=28, y=8, rotation=180, tags=("shape_8_3x4_79b9",))'
    assert str([shape, Sprite([[1]])]) == f"[{shape}, Sprite([[1]])]"
    # A Piece prints as the Sprite it is.
    piece = segment.Piece([[9, 9]], x=1, shape="SHAPE_9_2x1_abcd", colour=9, size=2)
    assert str(piece) == "Sprite([[9, 9]], x=1)" and dataclasses.fields(piece)
    # engine.py's FIXED block has it too, and an engine prints its sprites as code.
    source = render_skeleton("tiny", [1, 2, 3, 4])
    assert "def __repr__(self) -> str:" in source[: source.index(game_api.END_MARKER)]
    module: dict = {}
    exec(compile(source, "engine.py", "exec", dont_inherit=True), module)
    assert repr(module["Sprite"]([[1, 2]], tags=("a",))) == 'Sprite([[1, 2]], tags=("a",))'
    # An engine of an earlier run, whose block has no __repr__, still has an unchanged interface.
    block = game_api.FIXED_INTERFACE
    start = block.index("    def __repr__(self) -> str:")
    end = block.index('        return "Sprite(" + ", ".join(parts) + ")"\n') + len('        return "Sprite(" + ", ".join(parts) + ")"\n')
    earlier = block[:start].rstrip("\n") + "\n" + block[end:]
    assert game_api.same_interface(earlier) and not game_api.same_interface(earlier.replace("layer: int = 0", "layer: int = 1"))


def test_pieces_redraw_every_reference_level_start_exactly() -> None:
    import time

    from engine_re import game_api, segment
    from engine_re.auto_sprites import guess_grid, sprite_code

    api = game_api.canonical()
    with np.load(LEVEL_STARTS) as data:
        frames, names = data["frames"], data["names"]
    slowest = 0.0
    for frame, name in zip(frames, names):
        guess = guess_grid(frame)
        started = time.perf_counter()
        found = segment.pieces(frame, guess)
        slowest = max(slowest, time.perf_counter() - started)
        view = {} if guess.default_scale else {"view": api.View(scale=guess.scale)}
        state = api.State(grid=guess.grid, sprites=list(found), **view)
        assert np.array_equal(game_api.render(state), frame), name
        assert [p.role for p in found[:2]] == ["border", "background"] and {p.role for p in found[2:]} <= {"object"}
        assert all(isinstance(p, api.Sprite) and p.size > 0 for p in found)
        for p in found[2:]:
            assert p.tags[-1] == p.shape.lower() and p.screen == ("hud" in p.tags)
            shown = {k: getattr(p, k) for k, default in (("rotation", 0), ("mirror_ud", False), ("mirror_lr", False), ("scale", 1))
                     if getattr(p, k) != default}
            assert {k: v for k, v in p.transform.items() if k != "recolour"} == shown
        assert found.code() == sprite_code(frame, guess, function="frame_sprites", source="the frame").code
        children = sorted(c for p in found for c in p.children)
        assert children == list(range(1, len(found)))  # a tree: every piece but the border has one parent
    assert slowest < 0.5


def _ring_frame(colours: list[int], bar: int) -> np.ndarray:
    """A 16x16 grid at scale 4: 20 tiles of 1 cell around a square, each its own colour, a 3x3 frame
    enclosing a dot in the middle, and a screen bar (pixels off the grid's blocks) `bar` pixels tall."""
    cells = np.zeros((16, 16), np.int16)
    ring = [(x, 2) for x in range(2, 14, 2)] + [(12, y) for y in range(4, 14, 2)] + [(x, 12) for x in range(10, 0, -2)]
    ring += [(2, y) for y in range(10, 2, -2)]
    ring = ring[:20]
    for (x, y), colour in zip(ring, colours):
        cells[y, x] = colour
    cells[6:9, 6:9] = 7
    cells[7, 7] = 8
    frame = np.repeat(np.repeat(cells, 4, axis=0), 4, axis=1).astype(np.int8)
    frame[64 - bar :, 1] = 11
    return frame


def test_changes_find_recoloured_moved_reshaped_and_new_pieces() -> None:
    from engine_re import segment
    from engine_re.auto_sprites import GridGuess

    grid = GridGuess(16, 16, 4, 0, 0, 0, 1, "as given")
    colours = [1 + k % 6 for k in range(20)]
    before = segment.pieces(_ring_frame(colours, 30), grid)
    after = segment.pieces(_ring_frame(colours[1:] + colours[:1], 25), grid)  # the ring turns by one tile; the bar shrinks
    found = segment.changes(before, after)
    kinds = [c.kind for c in found]
    assert kinds.count("recoloured") == 20 and kinds.count("reshaped") == 1 and len(found) == 21, found
    first = found[0]
    assert first.colours == {colours[0]: colours[1]} and first.before.x == first.after.x and (first.dx, first.dy) == (0, 0)
    reshaped = next(c for c in found if c.kind == "reshaped")
    assert reshaped.note == "1x30 -> 1x25, lost 5 px at the top" and (reshaped.dx, reshaped.dy) == (0, 5)
    text = segment.summary(found)
    assert text.splitlines()[0].startswith("20 recoloured (1x1, 6 shapes): ") and "... and" in text.splitlines()[0]
    assert text.splitlines()[1] == "1 reshaped: screen piece " + reshaped.before.shape + " colour 11 (yellow) at (1, 34): 1x30 -> 1x25, lost 5 px at the top"
    assert len(text.splitlines()) == 2 and all(len(line) <= segment.LINE_CHARS + 30 for line in text.splitlines())
    # The 3x3 frame encloses the dot.
    frame_piece = next(i for i, p in enumerate(before) if p.colour == 7)
    dot = next(i for i, p in enumerate(before) if p.colour == 8)
    assert before[frame_piece].children == [dot] and dot not in before[1].children and frame_piece in before[1].children
    # Appeared, disappeared, and a piece filled in; unchanged pieces are not listed.
    other = _ring_frame(colours, 30)
    other[28:32, 28:32] = 7  # the dot takes the frame's colour: the frame is now a solid square
    other[8:12, 0:4] = 13  # a new piece at cell (0, 2)
    other[8:12, 8:12] = 0  # the first tile is gone
    found = segment.changes(before, segment.pieces(other, grid))
    assert [c.kind for c in found] == ["reshaped", "appeared", "disappeared", "disappeared"], found
    assert found[0].note == "3x3 -> 3x3, gained 1 cell inside its box" and found[1].after.colour == 13
    assert str(segment.changes(before, before)) == "no piece changed" and segment.summary([]) == "no piece changed"
    # Pieces of two different grids (a new level) are never the same piece; screen pieces still compare.
    coarse = segment.pieces(_ring_frame(colours, 30), GridGuess(32, 32, 2, 0, 0, 0, 1, "as given"))
    found = segment.changes(before, coarse)
    assert {c.kind for c in found if not (c.after or c.before).screen} == {"appeared", "disappeared"}
    assert len([c for c in found if c.kind == "disappeared"]) == len(before) - 2  # all but the border and the bar
    step = segment.pieces(_ring_frame(colours, 30), grid)
    step_moved = _ring_frame(colours, 30)
    step_moved[:, 1] = 0
    step_moved[64 - 30 :, 2] = 11  # the bar moves one pixel right
    found = segment.changes(step, segment.pieces(step_moved, grid))
    assert [(c.kind, c.dx, c.dy) for c in found] == [("moved", 1, 0)] and str(found[0]).startswith("moved: screen piece SHAPE_11_1x30_")
    assert segment.summary(found).startswith("1 moved by (+1, +0) screen (SHAPE_11_1x30_")


def test_the_step_messages_show_what_the_step_changed(monkeypatch, two_level_trace: Trace) -> None:
    from engine_re import auto_sprites, prompts, segment

    def visible(k: int) -> Trace:
        return Trace(two_level_trace.game_id, two_level_trace.steps[: k + 1])

    text = prompts.episode_message("two", visible(1), 1, "REPORT", "ENGINE", history=False)
    block = text[text.index("What the recorded step changed (objects):") : text.index("The test report:")].strip()
    assert block.splitlines()[0] == ("What the recorded step changed (objects): step_to_fix.pieces_before -> "
                                     "step_to_fix.pieces_after; step_to_fix.changes lists them.")
    assert block.splitlines()[1].startswith("  1 moved by (+1, +0) (SHAPE_9_1x1_") and len(block.splitlines()) == 2
    assert "it starts the game" in prompts.episode_message("two", visible(0), 0, "REPORT", "ENGINE")
    # advance_message gets the whole recording, but the block looks only at steps 0..k.
    seen: list[int] = []
    where = {s.last.__array_interface__["data"][0]: i for i, s in enumerate(two_level_trace.steps)}

    def spy(found):
        def wrapped(frame, *args, **kwargs):
            seen.extend(where.get(f.__array_interface__["data"][0], -1) for f in (frame if isinstance(frame, list) else [frame]))
            return found(frame, *args, **kwargs)
        return wrapped

    monkeypatch.setattr(segment, "guess_grid", spy(auto_sprites.guess_grid))
    monkeypatch.setattr(segment, "pieces", spy(segment.pieces))
    text = prompts.advance_message(two_level_trace, 1, 3, "REPORT", history=True)
    assert seen and max(seen) <= 3
    assert "What the recorded step changed (objects): it enters level 1. step_to_fix.pieces_after holds that level's" in text
    assert "step_to_fix.pieces_after.code() gives code for it" in text and "auto_sprites" not in text
    seen.clear()
    text = prompts.advance_message(two_level_trace, 3, 4, "REPORT")
    assert max(seen) == 4 and "1 moved by (+0, +1)" in text
    # A frame the segmentation cannot read leaves a note, not a failed run.
    monkeypatch.setattr(segment, "pieces", lambda *a, **k: 1 / 0)
    assert "(objects): not available (ZeroDivisionError" in prompts.advance_message(two_level_trace, 3, 4, "REPORT")


LP85_TRACE = Path("/home/user/arc-agi-3-solution-da-fr/ARC3-Inference/runs/engine-re/qwen38flash-v6c-lp85/lp85/trace")


@pytest.mark.skipif(not LP85_TRACE.exists(), reason="the lp85 run is not on this machine")
def test_pieces_and_changes_on_a_real_recording() -> None:
    from engine_re import game_api, segment

    trace = Trace.load(LP85_TRACE)
    api = game_api.canonical()
    segmenter = segment.Segmenter(trace)
    for level, k in trace.level_starts().items():
        found = segmenter.pieces(k)
        view = {} if found.grid.default_scale else {"view": api.View(scale=found.grid.scale)}
        assert np.array_equal(game_api.render(api.State(grid=found.grid.grid, sprites=list(found), **view)), trace.steps[k].last), level
    kinds = [c.kind for c in segmenter.changes(1)]
    # The ring of 20 tiles turns: 18 change colour (2 keep theirs); the bar at the left loses its top 5 pixels to black.
    assert (kinds.count("recoloured"), kinds.count("reshaped"), kinds.count("appeared"), len(kinds)) == (18, 1, 1, 20)
    assert "lost 5 px at the top" in segment.summary(segmenter.changes(1))
    assert "it enters level 1" in segmenter.report(8)


# --- read_file / edit_file / undo_edit, the tools, finish and show_frames ---------------------------------------------


def test_anchors_are_stable_and_stale_ones_are_rejected() -> None:
    from engine_re import hashline

    text = "".join(f"line {k}\n" for k in range(1, 21))
    lines, _ = hashline.split_lines(text)
    shown = hashline.render_read(text)
    assert shown.splitlines()[0].strip() == f"{hashline.anchor(lines, 1)}:line 1"
    assert all(len(a.split("#")[1].split(":")[0]) == 3 for a in shown.splitlines())
    assert all(c in hashline.NIBBLES for line in shown.splitlines() for c in line.split("#")[1][:3])
    before = {n: hashline.anchor(lines, n) for n in range(1, 21)}
    edited = hashline.apply_edits(text, [{"op": "replace", "pos": before[10], "lines": ["LINE 10"]}]).text
    after_lines, _ = hashline.split_lines(edited)
    after = {n: hashline.anchor(after_lines, n) for n in range(1, 21)}
    # Only the edited line and its neighbours get new hashes; distant anchors stay valid.
    changed = {n for n in before if before[n] != after[n]}
    assert 10 in changed and changed <= {9, 10, 11}
    assert hashline.split_lines(text)[0] == lines and hashline.anchor(lines, 5) == before[5]  # same input, same anchor
    # A stale anchor: the error shows the lines there now, with fresh anchors, and nothing is applied.
    with pytest.raises(hashline.EditError, match=r"\[E_STALE_ANCHOR\] " + before[10] + " is stale") as info:
        hashline.apply_edits(edited, [{"op": "replace", "pos": before[10], "lines": ["x"]}])
    assert "No edit was applied (1 of 1 failed" in str(info.value) and "Lines 9-11 now:" in str(info.value)
    assert all(f"{after[n]}:{line}" in str(info.value) for n, line in ((9, "line 9"), (10, "LINE 10"), (11, "line 11")))
    # A ":content" suffix is cross-checked: the right hash with the wrong content is stale too.
    with pytest.raises(hashline.EditError, match="E_STALE_ANCHOR"):
        hashline.apply_edits(text, [{"op": "replace", "pos": before[3] + ":line 4", "lines": ["x"]}])
    assert hashline.apply_edits(text, [{"op": "replace", "pos": before[3] + ":line 3", "lines": ["x"]}]).text.count("x\n") == 1
    # A stale hash whose content still matches the line is accepted, with a warning; a 2-character
    # anchor (the end of the hash) is accepted too.
    result = hashline.apply_edits(text, [{"op": "replace", "pos": "3#ZZZ:line 3", "lines": ["x"]}])
    assert result.text.count("x\n") == 1 and "stale but its content matched" in result.warnings[0]
    short = "3#" + before[3].split("#")[1][-2:]
    assert hashline.apply_edits(text, [{"op": "replace", "pos": short, "lines": ["x"]}]).text.count("x\n") == 1
    with pytest.raises(hashline.EditError, match="E_STALE_ANCHOR"):
        hashline.apply_edits(text, [{"op": "replace", "pos": "3#ZZZ:line 7", "lines": ["x"]}])
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
    # Edits on adjacent lines, and an insert touching a replaced range, are merged in order.
    assert run({"op": "replace", "pos": A[2], "lines": []}, {"op": "replace", "pos": A[3], "lines": []}) == "a\nd\ne\n"
    assert run({"op": "replace", "pos": A[2], "lines": ["x"]}, {"op": "append", "pos": A[2], "lines": ["y"]}) == "a\nx\ny\nc\nd\ne\n"
    assert run({"op": "append", "pos": A[3], "lines": ["y"]}, {"op": "replace", "pos": A[2], "end": A[3], "lines": ["X"]}) == "a\nX\ny\nd\ne\n"
    assert run({"op": "prepend", "pos": A[3], "lines": ["y"]}, {"op": "replace", "pos": A[3], "lines": ["C"]}) == "a\nb\ny\nC\nd\ne\n"
    assert run({"op": "append", "pos": A[1], "lines": ["1"]}, {"op": "append", "pos": A[1], "lines": ["2"]}) == "a\n1\n2\nb\nc\nd\ne\n"
    for bad, code in (
        ([{"op": "replace", "pos": A[2], "end": A[3], "lines": ["x"]}, {"op": "replace", "pos": A[3], "lines": ["y"]}], "E_EDIT_CONFLICT"),
        ([{"op": "replace", "pos": A[2], "end": A[4], "lines": ["x"]}, {"op": "append", "pos": A[3], "lines": ["y"]}], "E_EDIT_CONFLICT"),
        ([{"op": "replace_text", "oldText": "zzz", "newText": "y"}], "E_NO_MATCH"),
        ([{"op": "append", "pos": A[2], "lines": []}], "E_BAD_OP"),
        ([{"op": "replace", "pos": "2", "lines": ["x"]}], "E_BAD_REF"),
        ([{"op": "replace", "pos": A[1], "lines": [f"{A[1]}:a"]}], "E_INVALID_PATCH"),
        ([{"op": "move", "pos": A[1]}], "E_BAD_OP"),
        ([{"op": "replace_def", "name": "a-b", "lines": ["x"]}], "E_BAD_OP"),
    ):
        with pytest.raises(hashline.EditError, match=code):
            hashline.apply_edits(text, bad)
    assert hashline.apply_edits(text, [{"op": "replace", "pos": A[1], "lines": ["a"]}]).noop


CODE = "import os\n\nX = 1\n\n\n@deco\ndef f(a):\n    return a\n\n\nclass G:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n"


def test_edits_are_lenient_and_applied_one_by_one() -> None:
    from engine_re import hashline

    lines, _ = hashline.split_lines(CODE)
    A = {n: hashline.anchor(lines, n) for n in range(1, len(lines) + 1)}
    # replace_text: an exact match first; else whole lines matched ignoring whitespace, newText as given.
    result = hashline.apply_edits(CODE, [{"op": "replace_text", "oldText": "def f(a):\n  return   a  ", "newText": "def f(a):\n    return a + 1"}])
    assert result.text.splitlines()[6:8] == ["def f(a):", "    return a + 1"] and "matched ignoring whitespace" in result.warnings[0]
    result = hashline.apply_edits(CODE, [{"op": "replace_text", "oldText": "\n    return a\n", "newText": "\n    return -a\n"}])
    assert result.text.splitlines()[7] == "    return -a" and result.text.count("\n") == CODE.count("\n")
    # No match: the error suggests lines like the first significant line given, with anchors.
    with pytest.raises(hashline.EditError) as info:
        hashline.apply_edits(CODE, [{"op": "replace_text", "oldText": "def f(b):\n    return b", "newText": "x"}])
    assert "[E_NO_MATCH]" in str(info.value) and f"{A[7]}:def f(a):" in str(info.value)
    with pytest.raises(hashline.EditError, match="E_MULTI_MATCH"):
        hashline.apply_edits(CODE, [{"op": "replace_text", "oldText": "return", "newText": "x"}])
    # Partial application: the valid edit is applied, the stale one reported with the lines there now.
    result = hashline.apply_edits(CODE, [{"op": "replace", "pos": "3#ZZZ", "lines": ["X = 2"]}, {"op": "append", "pos": A[1], "lines": ["import re"]}])
    assert result.summary == ["inserted 1 line after line 1"] and result.total == 2 and len(result.failed) == 1
    new_lines, _ = hashline.split_lines(result.text)
    assert result.failed[0].startswith("Edit 0 (replace 3#ZZZ) not applied: [E_STALE_ANCHOR] 3#ZZZ is stale")
    assert f"{hashline.anchor(new_lines, 4)}:X = 1\n" in result.failed[0]  # line 3 is line 4 after the insert
    # A true overlap refuses both edits, showing both; the rest is applied.
    result = hashline.apply_edits(CODE, [
        {"op": "replace", "pos": A[7], "end": A[8], "lines": ["def f(a):", "    return 0"]},
        {"op": "replace", "pos": A[8], "lines": ["    return 1"]},
        {"op": "replace", "pos": A[3], "lines": ["X = 3"]},
    ])
    assert result.summary == ["replaced line 3 with 1 line"] and len(result.failed) == 2
    assert all("[E_EDIT_CONFLICT] edits 0 (replace " in f and "and 1 (replace " in f and "both change line 8" in f for f in result.failed)
    # A line beyond the end.
    result = hashline.apply_edits(CODE, [{"op": "replace", "pos": "99#ZZZ", "lines": ["x"]}, {"op": "replace", "pos": A[3], "lines": ["X = 3"]}])
    assert result.failed[0].startswith("Edit 0 (replace 99#ZZZ) not applied: [E_RANGE_OOB] line 99 does not exist (the file has 16 lines)")


def test_replace_def_replaces_a_whole_definition() -> None:
    from engine_re import hashline

    def run(name: str, *new: str, protected=None) -> hashline.EditResult:
        return hashline.apply_edits(CODE, [{"op": "replace_def", "name": name, "lines": list(new)}], protected=protected)

    result = run("f", "def f(a):", "    return -a")
    assert result.summary == ["replaced lines 6-8 with 2 lines"] and "@deco" not in result.text  # decorators included
    assert run("X", "X = 5").text.splitlines()[2] == "X = 5"
    result = run("G.n", "    def n(self):", "        return 22")
    assert result.summary == ["replaced lines 15-16 with 2 lines"] and result.text.endswith("        return 22\n")
    assert run("G", "class G:", "    pass").text.endswith("class G:\n    pass\n")
    result = run("h", "def h():", "    pass")
    assert result.text.endswith("        return 2\n\ndef h():\n    pass\n") and result.warnings == ["Edit 0: h was not defined; added at the end."]
    result = run("G.z", "    def z(self):", "        pass")
    assert result.text.endswith("        return 2\n    def z(self):\n        pass\n") and "added at the end of class G" in result.warnings[0]
    with pytest.raises(hashline.EditError, match=r"\[E_NO_DEF\] class Q is not defined"):
        run("Q.z", "x")
    with pytest.raises(hashline.EditError, match=r"\[E_FIXED_BLOCK\]"):
        run("X", "X = 5", protected=(1, 4))
    with pytest.raises(hashline.EditError, match=r"\[E_NO_DEF\] the file does not parse \(line 1"):
        hashline.apply_edits("def (:\n", [{"op": "replace_def", "name": "f", "lines": ["x"]}])
    # Fresh anchors show a changed region whole up to 60 lines, with one line of context.
    big = [f"    x{k} = {k}" for k in range(59)]
    result = run("f", "def f():", *big)
    anchors = hashline.fresh_anchors(result.text, result.regions)
    assert len(anchors) == 62 and anchors[0].endswith(":") and anchors[-1].endswith(":") and not any("more new lines" in a for a in anchors)
    result = run("f", "def f():", *big, "    return 1")
    assert any("more new lines" in a for a in hashline.fresh_anchors(result.text, result.regions))


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
        out = kernel.execute("edit_file(edits=[{'op': 'replace_text', 'oldText': 'ORIGINAL = 1', 'newText': 'CHANGED = 2'}])")
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
    assert "tested: 4 pass before the first failure (step 4), 7/9 in all" in out and "read_file() again" in out
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


def test_commit_engine_runs_the_tests_and_ends_only_when_they_pass(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("commit_engine", {})],  # no message: refused, nothing runs
            [("commit_engine", {"message": "nothing yet"})],  # fails: the session goes on
            [("python", {"code": _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "1"))}), ("commit_engine", {"message": "moves"})],
            [("python", {"code": "1"})],  # never reached
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), opening=False, client=model)
    result = agent.run()
    tool_outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_outputs[0].startswith("Error: commit_engine needs a message")
    assert tool_outputs[1].startswith("Not committed: the tests still fail, so the session goes on.")
    assert "--- Step 0: RESET" in tool_outputs[1]
    assert tool_outputs[-1] == "Every test passes. Session finished."
    assert result.status == "passed" and result.turns == 3 and result.commit_calls == 2 and result.commit_message == "moves"
    assert result.tests_run == 2 and result.auto_tests == 0


def test_show_images_join_the_turns_image_message(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import IMAGE_PLACEHOLDER, Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel(
        [
            [("python", {"code": "show_frames(recording[0].after, recording[1].after, titles=['a', 'b'], boxes=[(8, 8, 15, 15)])"}), ("run_tests", {})],
            [("python", {"code": "show_frames(recording[2].after)"})],
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), opening=False, client=model)
    agent.run()
    images = _image_messages(agent)
    assert len(images) == 2
    first, second = images[0][1]["content"], images[1][1]["content"]
    captions = [p["text"] for p in first if p["type"] == "text"]
    assert captions[1] == "show_frames(): a | b; boxes 1" and any("From the latest run_tests report" in c for c in captions)
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
    assert "1. recording[0].pieces_after.code() wrote sprites that draw level 0's first frame" in text
    assert "recording[0].pieces_after.code(): sprites that draw level 0's first frame (recording[0].after)." in text
    assert "    # recording[e].pieces_after.code(), e being the step that enters level n." in engine
    assert "auto_sprites" not in engine and "auto_sprites" not in text
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
    assert "Your first task: put recording[0].pieces_after.code() into make_level with edit_file()" in opening
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
    assert rows[2].split() == ["0", "recording[0].after", "8x8", "s8", "1-3", "(3)", "right", "x3", "3", "-", "-", "solved", "at",
                               "step", "3"]
    assert rows[3].startswith("1      recording[3].after  8x8 s8  4-8 (5)") and rows[3].rstrip().endswith("not solved: the recording ends (NOT_FINISHED)")
    assert "RESET x1" in rows[3] and "summarize_levels" in PRELOADED


def test_a_resumed_session_keeps_the_versions_and_shows_anchors(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re import hashline
    from engine_re.agent import Budget, EngineAgent, ModelConfig

    tiny_trace.save(tmp_path / "trace")
    wrong = _rewrite_call(SIMPLE_TINY_GAME.replace("DOWN", "2"))
    EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=1), opening=False, client=_ScriptedModel([[("python", {"code": wrong})]])).run()
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), opening=False, client=_ScriptedModel([[("python", {"code": "undo_edit()"})]]))
    agent.run()
    opening = agent.messages[1]["content"]
    assert "This continues an earlier session on this game (1 turns)" in opening and "--- Step " in opening
    shown = opening.split("engine.py now, as read_file() shows it:")[1]
    assert "the FIXED block, folded; it cannot be edited" in shown and "class Sprite:" not in shown
    out = next(m["content"] for m in agent.messages if m["role"] == "tool")
    assert out.startswith("Restored version 1, as engine.py was 1 change ago, saved as version 3.")
    assert agent.result.resumes == 1 and agent.result.engine_changes == 2
    # The resume showed the engine as the first session left it (version 2), with valid anchors.
    lines, _ = hashline.split_lines((tmp_path / "engine_versions" / "v0002.py").read_text())
    assert f"{hashline.anchor(lines, len(lines))}:{lines[-1]}" in shown


def test_the_kernel_lists_what_the_model_defined_and_replays_cells(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re import kernel as kernel_mod
    from engine_re.prompts import KERNEL_KEEPS_NOTHING, kernel_names_text
    from engine_re.skeleton import render_skeleton

    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = workspace / "engine.py"
    engine.write_text(render_skeleton("tiny", [1, 2, 3, 4]), encoding="utf-8")
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        assert kernel.names() == ([], 0) and kernel_names_text([], 0) == KERNEL_KEEPS_NOTHING
        kernel.execute("import os, numpy\nRING = list(range(20))\ndef cols(a): return a\nL1 = {1: 2, 2: 3, 3: 4}\n"
                       "f0 = np.zeros((64, 64))\nbest = (1, 2)\nst = replica.make_level(0)\nclass K: pass\nn = None\ns = 'ab'")
        names, more = kernel.names()
        assert names == ["RING: list[20]", "cols: function", "L1: dict[3]", "f0: ndarray(64, 64)", "best: tuple[2]", "st: State",
                         "K: class", "n: None", "s: str[2]"] and more == 0  # not the built-ins, np, modules or dunders
        assert kernel_names_text(names, 0) == "Your python kernel keeps: " + ", ".join(names)
        assert kernel_names_text(names[:2], 7) == "Your python kernel keeps: RING: list[20], cols: function, ... and 7 more"
        kernel.execute("\n".join(f"v{k} = {k}" for k in range(45)))
        names, more = kernel.names()
        assert len(names) == kernel_mod.NAMES_SHOWN == 40 and more == 14
        # Replay: edits are skipped, images not made, an error ends only its cell, a slow cell is cut, the rest goes on.
        before = engine.read_text()
        result = kernel.replay([
            {"turn": 1, "code": "kept = 41\nedit_file(edits=[{'op': 'append', 'lines': ['Y_MARK = 8']}])\nundo_edit()"},
            {"turn": 2, "code": "1 / 0"},
            {"turn": 3, "code": "show_frames(recording[0].after)\nafter = 5"},
            {"turn": 4, "code": "while True: pass"},
            {"turn": 5, "code": "late = 1"},
        ], cell_seconds=1.0, total_seconds=10.0)
        assert result["replayed"] == 5 and result["skipped"] == 0 and 1.0 <= result["seconds"] < 5.0
        assert [(f["turn"], f["error"].split(":")[0]) for f in result["failed"]] == [(2, "ZeroDivisionError"), (4, "TimeoutError")]
        assert engine.read_text() == before and kernel.last_images == []
        assert kernel.execute("print(kept, after, late)").split() == ["41", "5", "1"]
        # After the replay, edits work again, and the overall cap skips the cells it cannot reach.
        assert "engine.py: inserted" in kernel.execute("edit_file(edits=[{'op': 'append', 'lines': ['Z_MARK = 9']}])")
        result = kernel.replay([{"turn": 6, "code": "while True: pass"}, {"turn": 7, "code": "x = 1"}], cell_seconds=5.0, total_seconds=1.0)
        assert result["replayed"] == 1 and result["skipped"] == 1 and [f["turn"] for f in result["failed"]] == [6]
        assert kernel.replay([]) == {"replayed": 0, "failed": [], "skipped": 0, "seconds": 0.0}
    finally:
        kernel.stop()


def test_the_replica_builtin_always_reflects_the_current_engine_py(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.kernel import ENGINE_IMPORT_NOTE
    from engine_re.skeleton import render_skeleton

    tiny_trace.save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text(render_skeleton("tiny", [1, 2, 3, 4]), encoding="utf-8")
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60)
    try:
        assert kernel.execute("print(replica.make_level(0).grid, callable(replica.step), 'make_level' in dir(replica))").split() == ["(64,", "64)", "True", "True"]
        for code in ("import engine", "from engine import step", "import engine as e", "import engine as replica", "replica = 3"):
            out = kernel.execute(code)
            assert "nothing was run" in out and "built-in replica" in out, code
            assert (ENGINE_IMPORT_NOTE in out) == ("import" in code), code
        assert ENGINE_IMPORT_NOTE.startswith("replica is a built-in") and "use replica.step(...)" in ENGINE_IMPORT_NOTE
        kernel.execute("edit_file(edits=[{'op': 'append', 'lines': ['X_MARK = 7']}])")
        assert kernel.execute("replica.X_MARK").strip() == "7"  # reloaded after the change
        kernel.execute("undo_edit()")
        assert "AttributeError" in kernel.execute("replica.X_MARK")
        assert kernel.execute("replica").startswith("<replica: engine.py as it is now")
        assert "NameError" in kernel.execute("engine")  # one name for the built-in, in every mode
    finally:
        kernel.stop()


def test_kernel_rejects_code_that_rebinds_a_builtin():
    from engine_re import kernel

    def show_frames(*frames):
        return "the harness show_frames"

    namespace = {"show_frames": show_frames, "print": print}
    builtins = {"show_frames": show_frames}
    out = kernel._run("def show_frames(f):\n    pass\nprint('ran')", namespace, builtins)
    assert "nothing was run" in out and "line 1: def show_frames" in out and "ran" not in out
    assert namespace["show_frames"] is show_frames
    for code in ("show_frames = 3", "for show_frames in range(2): pass", "import os as show_frames", "f = lambda show_frames: show_frames",
                 "del show_frames"):
        assert "nothing was run" in kernel._run(code, namespace, builtins), code
    # Other names, calls and keyword arguments are fine.
    code = "x = show_frames()\nprint(x)\ndef f(frames, show_frames_all=True): return frames"
    assert kernel._run(code, namespace, builtins).strip() == "the harness show_frames"
    # A rebinding the check cannot see is undone after the run, and reported.
    out = kernel._run("globals()['show_frames'] = 1", namespace, builtins)
    assert namespace["show_frames"] is show_frames and "restored" in out


def test_system_prompt_names_every_builtin_function():
    from engine_re.kernel import RESERVED
    from engine_re.prompts import system_prompt

    for images in (True, False):
        prompt = system_prompt(images=images)
        section = prompt[prompt.index("# Objects") : prompt.index("# How to work")]
        for name in RESERVED:
            assert name in section, name
        assert "reserved" in section and "step_to_fix" not in prompt


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


def test_openrouter_client_can_pin_providers(monkeypatch):
    from engine_re import agent as agent_mod

    sent = []

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {}}

    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return Resp()

    for providers, expected in ((None, None), (["z-ai"], {"order": ["z-ai"], "allow_fallbacks": False})):
        client = agent_mod.OpenRouterClient(agent_mod.ModelConfig(providers=providers), api_key="test")
        monkeypatch.setattr(client.session, "post", post)
        client.chat([], [])
        assert sent[-1].get("provider") == expected



def test_openrouter_client_sends_reasoning_effort_and_sampling(monkeypatch):
    from engine_re import agent as agent_mod

    sent = []

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {}}

    def post(*args, **kwargs):
        sent.append(kwargs["json"])
        return Resp()

    client = agent_mod.OpenRouterClient(agent_mod.ModelConfig(), api_key="test")
    monkeypatch.setattr(client.session, "post", post)
    client.chat([], [])
    assert sent[-1]["reasoning"] == {"enabled": True} and "top_k" not in sent[-1]

    config = agent_mod.ModelConfig(reasoning_effort="low", temperature=1.0, top_p=0.95, top_k=20)
    client = agent_mod.OpenRouterClient(config, api_key="test")
    monkeypatch.setattr(client.session, "post", post)
    client.chat([], [])
    assert sent[-1]["reasoning"] == {"effort": "low"}
    assert (sent[-1]["temperature"], sent[-1]["top_p"], sent[-1]["top_k"]) == (1.0, 0.95, 20)

# --- The stepwise harness (v6) ----------------------------------------------------------------


def _rewrite_now(game_code: str) -> str:
    """Kernel code that puts `game_code` below the FIXED block of engine.py as it is now, through edit_file()."""
    from engine_re.game_api import END_MARKER

    new = _tail(_engine_source(game_code))
    return (
        "from pathlib import Path as _P\n"
        "_t = _P('engine.py').read_text()\n"
        f"_m = {END_MARKER!r}\n"
        f"edit_file(edits=[{{'op': 'replace_text', 'oldText': _t[_t.index(_m) + len(_m):], 'newText': {new!r}}}])"
    )


class _RecordingModel(_ScriptedModel):
    """A scripted model that keeps the first user message of every conversation."""

    def __init__(self, turns):
        super().__init__(turns)
        self.openings: list[str] = []

    def chat(self, messages, tools):
        if len(messages) == 2:
            content = messages[1]["content"]
            self.openings.append(content if isinstance(content, str) else content[0]["text"])
            self.tools = tools
        self.last_messages = list(messages)
        import json

        self.calls = getattr(self, "calls", []) + [json.loads(json.dumps(messages))]
        return super().chat(messages, tools)


def test_stepwise_moves_on_only_after_a_commit(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig
    from engine_re.stepwise import StepwiseRun

    tiny_trace.save(tmp_path / "trace")
    no_up = SIMPLE_TINY_GAME.replace("DOWN", "1").replace("1: (0, -1)", "1: (0, 0)")
    first, last = "Moves: steps 1-3 show the piece one cell per key. Guess: up does nothing.", "Up moves the piece too (step 4)."
    model = _RecordingModel(
        [
            [("python", {"code": "kept = 41\n" + _rewrite_now(no_up)})],  # steps 0..0 pass (automatic test): no move on
            [("run_tests", {})],  # they still pass: still no move on
            [("commit_engine", {"message": first})],  # the commit moves on, to step 4
            [("python", {"code": "print(kept + 1, step_to_fix.index, step_to_fix.level, len(recording), recording[-1] is step_to_fix)"})],
            [("commit_engine", {"message": "nothing yet"})],  # step 4 still fails: no move on
            [("python", {"code": _rewrite_now(SIMPLE_TINY_GAME.replace("DOWN", "1"))})],  # steps 0..4 pass: no move on
            [("commit_engine", {"message": last})],  # the whole recording passes
        ]
    )
    result = StepwiseRun("tiny", tmp_path, ModelConfig(), Budget(max_turns=10), client=model, opening=False).run()
    assert result.status == "passed" and result.turns == 7 and result.engine_changes == 2 and result.commit_calls == 3
    assert [(a["fixed"], a["next"], a["message"], a["turn"]) for a in result.advances] == [(0, 4, first, 3), (4, None, last, 7)]
    assert all(len(a["engine_sha"]) == 64 and isinstance(a["version"], int) for a in result.advances)
    assert result.final["exact"] == 8 and result.passing_prefix == 8 and result.mode == "stepwise" and result.commit_message == last
    # One conversation: one first message, then the harness's message about the next step, in the same history.
    assert len(model.openings) == 1 and model.openings[0].startswith("Fix the breaking test: step 0.")
    users = [m["content"] if isinstance(m["content"], str) else m["content"][0]["text"] for m in model.last_messages if m["role"] == "user"]
    advance = next(u for u in users if u.startswith("Commit accepted"))
    assert advance.startswith("Commit accepted: steps 0-0 pass. The harness replayed on: steps 1-3 (3 more steps) passed without "
                              "error. Step 4 is the next that fails.")
    assert "Step 4: ACTION1 (up), played in level 0" in advance and "`recording` now holds the recording up to step 4" in advance
    # Every next-step message lists engine.py as the first one does, under one fixed header, before its closing line.
    from engine_re.prompts import ENGINE_HEADER

    listing = advance[advance.index(ENGINE_HEADER) :]
    assert ENGINE_HEADER in model.openings[0] and "the FIXED block, folded" in listing and "def make_level(" in listing
    assert listing.splitlines()[-1] == "Fix step 4, keeping steps 0-3 passing; commit_engine(message) when the tests pass."
    # And what the model's kernel keeps, just before the listing.
    assert "\nYour python kernel keeps: kept: int, " in advance and advance.index("kernel keeps") < advance.index(ENGINE_HEADER)
    outputs = [m["content"] for m in model.last_messages if m["role"] == "tool"]
    hint = "Steps {} pass. You can now call commit_engine(message) to submit the fix, or keep refining first"
    assert "[harness] engine.py changed, so it was tested automatically" in outputs[0] and hint.format("0-0") in outputs[0]
    assert "ALL STEPS MATCH" in outputs[1] and hint.format("0-0") in outputs[1]
    assert outputs[2] == "Committed: steps 0-0 pass. The harness now replays the rest of the recording."
    assert outputs[3].split() == ["42", "4", "0", "5", "True"]  # the kernel kept its variables; recording grew to step 4
    assert outputs[4].startswith("Not committed: steps 0-4 do not all pass yet, so nothing moves on.")
    assert hint.format("0-4") in outputs[5]
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    commits = [r["commit"] for r in records if "commit" in r]
    assert [(c["fixed"], c["next"], c["message"]) for c in commits] == [(0, 4, first), (4, None, last)]
    tests = [json.loads(line) for line in (tmp_path / "tests.jsonl").read_text().splitlines()]
    assert [(t["auto"], t["focus"]) for t in tests] == [
        ("episode", 0), (True, 0), (False, 0), (False, 0), ("advance", 4), (False, 4), (True, 4), (False, 4),
    ]
    assert len(Trace.load(tmp_path / "visible_trace")) == 5  # the model saw steps 0..4, never 5..7
    # An interrupted run would get its commits back from the transcript.
    again = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(), client=_ScriptedModel([]), stepwise=True)
    again._restore()
    assert again.result.advances == result.advances and again.result.commit_calls == 3 and again.result.commit_message == last


def test_an_interrupted_stepwise_run_continues_its_conversation(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig, resume_note
    from engine_re.prompts import KERNEL_KEEPS

    tiny_trace.save(tmp_path / "trace")
    no_up = SIMPLE_TINY_GAME.replace("DOWN", "1").replace("1: (0, -1)", "1: (0, 0)")
    first = _RecordingModel(
        [
            [("python", {"code": "kept = 41\n" + _rewrite_now(no_up)})],
            [("python", {"code": "show_frames(recording[0].after)\nbroken = 1 / 0"})],  # raises after the image
            [("commit_engine", {"message": "moves"})],  # on to step 4; then the run stops (3 turns)
        ]
    )
    agent = EngineAgent("tiny", tmp_path, ModelConfig(compact_prompt_tokens=0), Budget(max_turns=3), client=first, stepwise=True)
    result = agent.run()
    assert result.status == "budget_turns" and result.step == 4
    sent = agent.messages  # everything the model was sent, shortened as it was
    log = tmp_path / "transcript.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("compact" in r for r in records) and any("hide_images" in r for r in records) and any("append" in r for r in records)
    # Compaction keeps only the latest engine.py listing: the first message's is elided, the next-step message's stays.
    from engine_re.prompts import ENGINE_ELIDED, ENGINE_HEADER

    users = [_message_text(m) for m in sent if m["role"] == "user"]
    assert f"{ENGINE_ELIDED}\n\nFix step 0:" in users[0] and ENGINE_HEADER not in users[0]
    assert ENGINE_HEADER in users[-1] and ENGINE_ELIDED not in users[-1] and sum(ENGINE_HEADER in u for u in users) == 1

    # transcript.jsonl alone gives that conversation back, exactly (images included).
    rebuilt = EngineAgent("tiny", tmp_path, ModelConfig(compact_prompt_tokens=0), Budget(), client=_ScriptedModel([]), stepwise=True)
    state = rebuilt._rebuild_conversation()
    assert state["focus"] == 4 and state["messages"] == sent
    assert [(c["turn"], c["code"][:12]) for c in state["cells"]] == [(1, "kept = 41\nfr"), (2, "show_frames(")]  # the python cells
    # A turn cut off before its tools answered is left out (its cells too).
    kept_log = log.read_text()
    cut = {"turn": 4, "finish_reason": "tool_calls", "content": "", "usage": {},
           "tool_calls": [{"id": "x", "type": "function", "function": {"name": "python", "arguments": "{}"}}]}
    log.write_text(kept_log + json.dumps(cut) + "\n")
    assert rebuilt._rebuild_conversation()["messages"] == sent and len(rebuilt._rebuild_conversation()["cells"]) == 2
    log.write_text(kept_log)
    versions = len((tmp_path / "engine_versions" / "versions.jsonl").read_text().splitlines())

    # Running again continues that conversation: no new first message; the kernel restarted and re-ran the cells.
    second = _RecordingModel(
        [
            [("python", {"code": "print(kept)"})],
            [("python", {"code": _rewrite_now(SIMPLE_TINY_GAME.replace("DOWN", "1"))})],
            [("commit_engine", {"message": "up moves too"})],
        ]
    )
    again = EngineAgent("tiny", tmp_path, ModelConfig(compact_prompt_tokens=0), Budget(max_turns=10), client=second, stepwise=True)
    result = again.run()
    assert result.status == "passed" and result.turns == 6 and result.resumes == 1
    assert second.openings == []  # never a fresh two-message conversation
    first_call = second.calls[0]
    note = first_call[len(sent)]["content"]
    assert first_call[: len(sent)] == sent and first_call[len(sent)]["role"] == "user"
    assert note.startswith("[harness] The run was interrupted here and has now resumed, in this same conversation.")
    assert ("re-ran your 2 python cells in order with file edits disabled, so your variables and functions are back; "
            "cells that raised when re-run (as before, or because engine.py changed later): turn 2.") in note
    assert note.splitlines()[-1].startswith(KERNEL_KEEPS + "kept: int")  # the names it keeps, after the replay
    replays = [r["replay"] for r in (json.loads(line) for line in log.read_text().splitlines()) if "replay" in r]
    assert len(replays) == 1 and replays[0]["cells"] == replays[0]["replayed"] == 2 and replays[0]["skipped"] == 0
    assert [f["turn"] for f in replays[0]["failed"]] == [2] and "ZeroDivisionError" in replays[0]["failed"][0]["error"]
    assert note == resume_note(replays[0], note.splitlines()[-1])
    outputs = [m["content"] for m in second.last_messages[len(sent):] if m["role"] == "tool"]
    assert outputs[0].strip() == "41"  # the variable is back
    # The replayed edit_file cell did not change engine.py (no new version), and the replayed show_frames made no image.
    assert len((tmp_path / "engine_versions" / "versions.jsonl").read_text().splitlines()) == versions + 1  # the second run's edit
    assert result.engine_changes == 2
    assert [(a["fixed"], a["next"], a["message"]) for a in result.advances] == [(0, 4, "moves"), (4, None, "up moves too")]
    # And the resumed part is in the transcript too: rebuilding now gives the whole conversation.
    assert EngineAgent("tiny", tmp_path, ModelConfig(compact_prompt_tokens=0), Budget(), client=_ScriptedModel([]),
                       stepwise=True)._rebuild_conversation()["messages"] == again.messages


def test_an_older_transcript_is_rebuilt_and_written_back_in_full(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, EngineAgent, ModelConfig, resume_note
    from engine_re.prompts import KERNEL_KEEPS

    tiny_trace.save(tmp_path / "trace")
    no_up = SIMPLE_TINY_GAME.replace("DOWN", "1").replace("1: (0, -1)", "1: (0, 0)")
    model = _RecordingModel([[("python", {"code": _rewrite_now(no_up)})], [("commit_engine", {"message": "moves"})]])
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), client=model, stepwise=True)
    agent.run()
    sent = agent.messages
    log = tmp_path / "transcript.jsonl"
    new_kinds = ("message", "append", "hide_images", "compact")
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()
                           if not any(k in json.loads(line) for k in new_kinds)))  # as logged before these records

    def texts(messages):  # the older records do not keep what the kernel held at a next step: that line is left out
        def plain(text):
            return "\n".join(line for line in text.splitlines() if not line.startswith(KERNEL_KEEPS)) if isinstance(text, str) else text

        return [(m["role"], plain(m["content"]) if isinstance(m["content"], str) else [plain(p.get("text")) for p in m["content"]])
                for m in messages]

    older = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(), client=_ScriptedModel([]), stepwise=True)
    state = older._rebuild_conversation()
    assert state["legacy"] and texts(state["messages"]) == texts(sent) and len(state["cells"]) == 1
    second = _RecordingModel([[("python", {"code": "1"})]])
    EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=3), client=second, stepwise=True).run()
    note = resume_note({"replayed": 1, "failed": [], "skipped": 0}, KERNEL_KEEPS + "_P: class, _t: str[" )
    sent_note = second.calls[0][len(sent)]["content"]
    assert texts(second.calls[0])[: len(sent)] == texts(sent) and sent_note.startswith(note)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum("rebased" in r for r in records) == 1
    # Written back in full: from now on the transcript alone gives the exact conversation.
    state = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(), client=_ScriptedModel([]), stepwise=True)._rebuild_conversation()
    assert "legacy" not in state and texts(state["messages"])[: len(sent) + 1] == texts(sent) + texts([{"role": "user", "content": sent_note}])


def test_a_commit_dropped_when_engine_changes_after_it(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, ModelConfig
    from engine_re.stepwise import StepwiseRun

    tiny_trace.save(tmp_path / "trace")
    no_up = SIMPLE_TINY_GAME.replace("DOWN", "1").replace("1: (0, -1)", "1: (0, 0)")
    model = _RecordingModel(
        [
            [("python", {"code": _rewrite_now(no_up)}), ("commit_engine", {"message": "moves"}),
             ("python", {"code": _rewrite_now(no_up.replace("3: (-1, 0)", "3: (-2, 0)"))})],
            [("python", {"code": "1"})],
        ]
    )
    result = StepwiseRun("tiny", tmp_path, ModelConfig(), Budget(max_turns=2), client=model, opening=False).run()
    assert result.advances == [] and result.step == 0 and result.status == "budget_turns"
    outputs = [m["content"] for m in model.last_messages if m["role"] == "tool"]
    assert outputs[1].startswith("Committed") and "so the commit was not kept: call commit_engine again" in outputs[2]


def test_stepwise_has_no_limit_per_step(tmp_path: Path, tiny_trace: Trace) -> None:
    import json

    from engine_re.agent import Budget, ModelConfig
    from engine_re.stepwise import StepwiseRun

    tiny_trace.save(tmp_path / "trace")
    model = _ScriptedModel([[("python", {"code": "print('recording' in globals(), step_to_fix.index)"})]] + [[("python", {"code": "1"})]] * 4)
    result = StepwiseRun("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), client=model, opening=False, history=False).run()
    assert result.status == "budget_turns" and result.turns == 5 and result.step == 0 and result.final["first_fail"] == 0
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert next(r["output"] for r in records if r.get("tool") == "python").split() == ["False", "0"]  # only the step


def test_stepwise_starts_with_the_opening(tmp_path: Path, tiny_trace: Trace) -> None:
    from engine_re.agent import Budget, ModelConfig
    from engine_re.stepwise import StepwiseRun

    tiny_trace.save(tmp_path / "trace")
    model = _RecordingModel([
        [("python", {"code": _rewrite_now(SIMPLE_TINY_GAME.replace("DOWN", "1"))})],
        [("commit_engine", {"message": "Each key moves the piece one cell."})],
    ])
    result = StepwiseRun("tiny", tmp_path, ModelConfig(), Budget(max_turns=5), client=model).run()
    assert result.opening == {"exact": True, "first_fail": 1, "passing_prefix": 1}
    assert result.status == "passed" and result.engine_changes == 1 and result.tests_run == 2  # the automatic test, the commit's
    assert [(a["fixed"], a["next"]) for a in result.advances] == [(1, None)]
    assert model.openings[0].startswith("Fix the breaking test: step 1.")
    assert "lands on your grid cell" not in model.openings[0]  # not a click game


def test_the_report_says_what_a_click_lands_on() -> None:
    from types import SimpleNamespace

    from engine_re import diff_report

    def sprite(name, x, y, w, h, layer=0, collidable=True, screen=False):
        return {"name": name, "x": x, "y": y, "w": w, "h": h, "layer": layer, "collidable": collidable, "screen": screen,
                "visible": True, "tags": [], "blocking": "pixel"}

    before = {"grid": [8, 8], "view": {"scale": None, "rotation": 0, "mirror_ud": False, "mirror_lr": False},
              "sprites": [sprite("background", 0, 0, 8, 8, layer=-1, collidable=False), sprite("button", 2, 3, 2, 2)]}
    lines = diff_report.click_lines(SimpleNamespace(id=6, x=2 * 8 + 3, y=3 * 8 + 1), before)  # scale 8: cell (2, 3)
    assert lines[0].startswith("    the click (19, 25) lands on your grid cell (2, 3): step() gets action.cell == (2, 3); ")
    assert '#1 "button"' in lines[1] and lines[1].endswith("<- state.sprite_at(*action.cell)")
    assert '#0 "background"' in lines[2] and "sprite_at" not in lines[2]
    assert diff_report.click_lines(SimpleNamespace(id=1, x=None, y=None), before) == []


def test_the_step_prompts_name_every_builtin():
    from engine_re.kernel import RESERVED_HISTORY, RESERVED_STEP
    from engine_re.prompts import system_prompt, tools

    for images in (True, False):
        for history, reserved in ((True, RESERVED_HISTORY), (False, RESERVED_STEP)):
            prompt = system_prompt(images=images, mode="step", history=history)
            section = prompt[prompt.index("# Objects") : prompt.index("# How to work")]
            assert all(name in section for name in reserved), (history, [n for n in reserved if n not in section])
            assert ("recording[" in prompt) == history and ("summarize_levels" in prompt) == history
            assert "step_to_fix: StepView" in section and "step_to_fix.before" in prompt
            python = tools(images, "step", history)[0]["function"]["description"]
            assert ("recording" in python) == history and "step_to_fix" in python


def _objects_members(section: str) -> dict[str, set[str]]:
    """The members the # Objects reference lists under each heading line ("Sprite(...)", "StepView: ...",
    ...): every `.name` before the two spaces that start a member line's meaning."""
    import re

    members: dict[str, set[str]] = {}
    current = None
    for line in section.splitlines():
        if line and not line.startswith(" "):
            heading = re.match(r"(\w+(?: \w+)*)", line)
            current = heading.group(1) if heading else None
            if current is not None:
                members[current] = set()
        elif current is not None and line.startswith("  .") and not line.startswith("   "):
            head = line[2:].split("  ")[0]
            members[current] |= set(re.findall(r"(?:^|[\s,(=])\.([A-Za-z_]\w*)", head))
    return members


def test_the_objects_reference_matches_the_code(tiny_trace: Trace) -> None:
    """The # Objects reference names every field and method of the engine classes, the recorded step
    and the recorded action, and nothing they do not have, in every mode."""
    import dataclasses
    import inspect
    import re

    from engine_re import game_api, helpers
    from engine_re.prompts import objects_reference, system_prompt
    from engine_re.trace import Action as RecordedAction

    api = game_api.canonical()
    text = objects_reference("single", True, True)
    members = _objects_members(text)
    for name in ("Sprite", "Action", "View", "State"):
        cls = getattr(api, name)
        fields = [f.name for f in dataclasses.fields(cls)]
        public = set(fields) | {k for k in vars(cls) if not k.startswith("_") and k not in fields}
        assert members[name] == public, (name, members[name] ^ public)
        # The signature line lists the constructor's arguments in order, with their defaults.
        signature = re.search(rf"^{name}\((.*?)\)  ", text, re.M | re.S).group(1)
        assert [a.split("=")[0].strip() for a in signature.split(",")] == fields
        for member in public:  # each method's arguments as the code has them
            value = vars(cls).get(member)
            if inspect.isfunction(value):
                args = [p for p in inspect.signature(value).parameters if p != "self"]
                documented = re.search(rf"^  \.{member}\((.*?)\) ->", text, re.M).group(1)
                assert [a.split("=")[0].strip().lstrip("*") for a in documented.split(",") if a.strip()] == args, member
    from engine_re.kernel import FUNCTIONS

    for name in FUNCTIONS + ("summarize_levels",):  # the built-in functions' arguments, as helpers has them
        documented = re.search(rf"^{name}\((.*?)\) ->", text, re.M).group(1)
        args = [a.split("=")[0].split(":")[0].strip().lstrip("*") for a in documented.split(",")]
        assert [a for a in args if a] == list(inspect.signature(getattr(helpers, name)).parameters), name
    view = helpers.StepView(tiny_trace, 1)
    lazy = {k for k, v in vars(helpers.StepView).items() if isinstance(v, property)}
    assert lazy == {"grid", "pieces_before", "pieces_after", "changes"}
    public = {k for k in vars(view) if not k.startswith("_")} | lazy
    assert members["StepView"] == public, members["StepView"] ^ public
    # A frame's pieces: Piece lists what it adds to Sprite; Pieces, Change and GridGuess everything they have.
    from engine_re import segment
    from engine_re.auto_sprites import GridGuess

    def own(cls: type, instance: object = None) -> set[str]:
        names = {k for k in vars(cls) if not k.startswith("_")}
        if dataclasses.is_dataclass(cls):
            names |= {f.name for f in dataclasses.fields(cls)}
        return names | {k for k in vars(instance or object()) if not k.startswith("_")} if instance is not None else names

    sprite_fields = {f.name for f in dataclasses.fields(api.Sprite)}
    assert members["Piece"] == {f.name for f in dataclasses.fields(segment.Piece)} - sprite_fields
    assert members["Pieces"] == own(segment.Pieces, segment.pieces(tiny_trace.steps[0].last)) == {"grid", "code"}
    assert members["Change"] == own(segment.Change)
    assert members["GridGuess"] == own(GridGuess)
    recorded = {f.name for f in dataclasses.fields(RecordedAction)} | {"name"}
    assert members["The recorded action"] == recorded and not hasattr(view.action, "cell")
    # The play mode (engine_re.play_agent) has the same classes and two more built-ins, documented as helpers has them.
    from engine_re.kernel import PRELOADED_PLAY

    play = objects_reference("play", True, True)
    play_members = _objects_members(play)
    assert all(play_members[k] == v for k, v in members.items())
    for name in FUNCTIONS + ("summarize_levels", "state_now", "click_cell", "traced", "support"):
        documented = re.search(rf"^{name}\((.*?)\) ->", play, re.M).group(1)
        args = [a.split("=")[0].split(":")[0].strip().lstrip("*") for a in documented.split(",")]
        assert [a for a in args if a] == list(inspect.signature(getattr(helpers, name)).parameters), name
    assert set(PRELOADED_PLAY) - {"recording", "step_to_fix", "replica"} == set(helpers.PLAY_FUNCTIONS)
    assert all(re.search(rf"^{name}[(:]", play, re.M) for name in PRELOADED_PLAY), PRELOADED_PLAY
    assert "run_tests(level=L)" not in play and "run_tests(level=L)" in text
    # Every mode shares these parts; the recorded steps python holds differ.
    for mode, history in (("single", True), ("step", True), ("step", False), ("play", True)):
        prompt = system_prompt(mode=mode, history=history)
        for part in ("StepView: a recorded step", "A recorded step has no State or vars:", "Piece: a Sprite",
                     "State(grid, sprites=[]", "replay_step(i, state=None, action=None, *, level=None) -> tuple[State | None, State]"):
            assert part in prompt, (mode, history, part)


def test_recorded_steps_are_step_views_and_step_is_a_free_name(tmp_path: Path, tiny_trace: Trace) -> None:
    import ast

    from engine_re.kernel import RESERVED_HISTORY, RESERVED_STEP, reserved_bindings

    # The engine's own step(state, action), and `step` as a loop variable, are the model's names to use.
    free = "def step(state, action):\n    return action\nfor step in recording:\n    pass\nstep = 3\n"
    for reserved in (RESERVED_HISTORY, RESERVED_STEP):
        assert reserved_bindings(ast.parse(free), reserved) == []
        assert [name for name, _, _ in reserved_bindings(ast.parse("step_to_fix = 1"), reserved)] == ["step_to_fix"]
    assert tiny_trace.steps[3].outcome == tiny_trace.steps[3].state == "NOT_FINISHED"

    # The stepwise harness's kernel with the recording so far: steps 0..3, then 0..5.
    Trace(tiny_trace.game_id, tiny_trace.steps[:4]).save(tmp_path / "trace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = KernelClient(workspace, tmp_path / "trace", timeout=60, focus=3, history=True)
    try:
        out = kernel.execute(
            "print(len(recording), recording[0].before is None, recording[2].outcome, recording[2].level, "
            "bool((recording[3].before == recording[2].after).all()), recording[3].after is recording[3].last, "
            "recording[-1] is step_to_fix, step_to_fix.index, recording[1].frames.shape)"
        )
        assert out.strip().split(maxsplit=8) == ["4", "True", "NOT_FINISHED", "0", "True", "True", "True", "3", "(1, 64, 64)"], out
        out = kernel.execute(
            "n = 0\nfor step in recording:\n    n += 1\nlast = step.index\n"
            "def step(state, action):\n    return action\nkept = recording\nprint(n, last)"
        )
        assert out.split() == ["4", "3"], out
        out = kernel.execute("step_to_fix = 1")
        assert "nothing was run" in out and "step_to_fix" in out
        assert "nothing was run" in kernel.execute("recording = []")
        Trace(tiny_trace.game_id, tiny_trace.steps[:6]).save(tmp_path / "trace")
        kernel.refocus(5)
        out = kernel.execute("print(len(recording), kept is recording, step_to_fix.index, recording[-1] is step_to_fix, step(0, 7))")
        assert out.split() == ["6", "True", "5", "True", "7"], out
    finally:
        kernel.stop()


def test_openrouter_client_waits_out_rate_limits(monkeypatch):
    from engine_re import agent as agent_mod

    class Resp:
        def __init__(self, status, data=None):
            self.status_code, self.data, self.headers, self.text = status, data, {}, ""

        def json(self):
            return self.data

    ok = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {}}
    answers = [Resp(429)] * 40 + [Resp(503), Resp(200, ok)]
    waits = []
    client = agent_mod.OpenRouterClient(agent_mod.ModelConfig(), api_key="test")
    monkeypatch.setattr(client.session, "post", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(agent_mod.time, "sleep", waits.append)
    monkeypatch.delenv("ARC3_HTTP_RETRIES", raising=False)
    assert client.chat([], [])["choices"][0]["message"]["content"] == "ok"
    assert len(waits) == 41 and not client.provider_errors  # 41 retries, no limit, none counted as a provider error
    answers[:] = [Resp(400)]
    with pytest.raises(RuntimeError, match="OpenRouter HTTP 400"):
        client.chat([], [])


def _trace_with_pixel(trace: Trace, step: int, row: int, col: int, n: int = 1) -> Trace:
    """A copy of ``trace`` whose step ``step`` final frame has ``n`` pixels changed from (row, col) rightwards."""
    import copy

    out = copy.deepcopy(trace)
    frames = out.steps[step].frames.copy()
    for i in range(n):
        frames[-1][row, col + i] = (int(frames[-1][row, col + i]) + 1) % 16
    out.steps[step].frames = frames
    return out


def test_one_border_pixel_is_tolerated_with_a_warning(tmp_path: Path, tiny_trace: Trace) -> None:
    engine = _engine(tmp_path, TINY_GAME.replace("DOWN", "1"))
    report = replay_test(engine, _trace_with_pixel(tiny_trace, 3, 0, 5), scratch_root=tmp_path)
    assert report.passed, report.text
    assert report.tolerated == [3]
    assert "WARNING (tolerated" in report.text and "step 3: 1 px differs at the frame border (row 0, col 5" in report.text
    assert "ALL STEPS MATCH" in report.text
    # The stepwise report (stops at the first failure) carries the same warning.
    stop = replay_test(engine, _trace_with_pixel(tiny_trace, 3, 63, 63), failures=1, scratch_root=tmp_path)
    assert stop.passed and stop.tolerated == [3] and "tolerated" in stop.text


def test_interior_or_two_border_pixels_still_fail(tmp_path: Path, tiny_trace: Trace) -> None:
    engine = _engine(tmp_path, TINY_GAME.replace("DOWN", "1"))
    inside = replay_test(engine, _trace_with_pixel(tiny_trace, 3, 5, 5), scratch_root=tmp_path)
    assert not inside.passed and inside.first_fail == 3 and inside.tolerated == []
    two = replay_test(engine, _trace_with_pixel(tiny_trace, 3, 0, 5, n=2), scratch_root=tmp_path)
    assert not two.passed and two.first_fail == 3 and "tolerated" not in two.text


def test_the_thinking_budget_goes_into_the_request_body(monkeypatch) -> None:
    import sys

    from engine_re import run_experiment
    from engine_re.agent import NO_BUDGET_WARNING, ModelConfig, OpenRouterClient, thinking_budget_warning

    messages, tool_list = [{"role": "user", "content": "hi"}], []
    body = OpenRouterClient(ModelConfig(thinking_budget=1024), api_key="test").body(messages, tool_list)
    assert body["reasoning"] == {"max_tokens": 1024}
    plain = OpenRouterClient(ModelConfig(), api_key="test").body(messages, tool_list)
    assert plain["reasoning"] == {"enabled": True} and "max_tokens" not in plain["reasoning"]
    effort = OpenRouterClient(ModelConfig(reasoning_effort="low"), api_key="test").body(messages, tool_list)
    assert effort["reasoning"] == {"effort": "low"}
    # A budget and an effort together are refused, by the config and by run_experiment.
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelConfig(thinking_budget=1024, reasoning_effort="low")
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(thinking_budget=0)
    monkeypatch.setattr(sys, "argv", ["run_experiment", "--run-dir", "r", "--games", "g", "--out", "o",
                                      "--thinking-budget", "1024", "--reasoning-effort", "low"])
    with pytest.raises(SystemExit) as exit_info:
        run_experiment.main()
    assert exit_info.value.code == 2

    # The model listing: a warning only for a model whose reasoning settings do not say supports_max_tokens.
    listing = {"data": [
        {"id": "a/budget", "reasoning": {"mandatory": False, "supports_max_tokens": True}},
        {"id": "b/efforts", "reasoning": {"mandatory": True, "supported_efforts": ["max", "high", "low"]}},
        {"id": "c/none", "reasoning": None},
    ]}
    assert thinking_budget_warning("a/budget", 1024, fetch=lambda: listing) is None
    assert thinking_budget_warning("b/efforts", 1024, fetch=lambda: listing) == NO_BUDGET_WARNING.format(model="b/efforts", budget=1024)
    assert thinking_budget_warning("c/none", 1024, fetch=lambda: listing) is None
    assert thinking_budget_warning("d/missing", 1024, fetch=lambda: listing) is None

    def offline():
        raise OSError("no network")

    assert thinking_budget_warning("b/efforts", 1024, fetch=offline) is None
