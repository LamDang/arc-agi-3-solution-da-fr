"""The play-and-model agent (engine_re.play_agent) on a tiny two-level game, with a scripted model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_re.agent import Budget, ModelConfig
from engine_re.live_game import LiveGame
from engine_re.play_agent import PlayAgent
from engine_re.trace import Action, parse_move

# Two levels of 8x8; RIGHT x3 solves each level (player.x >= 4); LEFT at x == 1 loses the game.
GAME = '''
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

MOVES = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


class Twol(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(sprites=[Sprite([[9]], name="player", x=1, y=1), Sprite([[5] * 8], name="wall", x=0, y=0)], grid_size=(8, 8)),
            Level(sprites=[Sprite([[9]], name="player", x=1, y=3), Sprite([[8] * 8], name="wall", x=0, y=7)], grid_size=(8, 8)),
        ]
        super().__init__(game_id="twol", levels=levels, camera=Camera(0, 0, 8, 8, 0, 3), available_actions=[1, 2, 3, 4])

    def step(self) -> None:
        player = self.current_level.get_sprites_by_name("player")[0]
        if self.action.id == GameAction.ACTION3 and player.x == 1:
            self.lose()
            self.complete_action()
            return
        dx, dy = MOVES.get(self.action.id, (0, 0))
        if dx or dy:
            self.try_move("player", dx, dy)
        if player.x >= 4:
            self.next_level()
        self.complete_action()
'''

ENGINE = """
LAYOUT = {0: ((1, 1), 5, 0), 1: ((1, 3), 8, 7)}


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
    player = state.vars["player"]
    if action.id == 3 and player.x == 1:
        state.status = "game_over"
        return
    moves = {1: (0, -1), 2: (0, DOWN), 3: (-1, 0), 4: (1, 0)}
    if action.id in moves:
        state.try_move(player, *moves[action.id])
    if player.x >= 4:
        state.status = "level_solved"
"""


@pytest.fixture()
def environments(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    folder = root / "twol" / "0000"
    folder.mkdir(parents=True)
    (folder / "twol.py").write_text(GAME, encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"game_id": "twol-0000", "baseline_actions": [3, 3]}), encoding="utf-8")
    return root


class _ScriptedModel:
    def __init__(self, turns: list[list[tuple[str, dict]]]):
        self.turns = turns
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools):  # noqa: ARG002
        self.seen.append([dict(m) for m in messages])
        calls = self.turns.pop(0) if self.turns else []
        return {
            "choices": [{
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "content": "" if calls else "done",
                    "tool_calls": [
                        {"id": f"c{len(self.seen)}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                        for i, (name, args) in enumerate(calls)
                    ],
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def _install(engine_code: str) -> tuple[str, dict]:
    """A python call that puts `engine_code`'s LAYOUT, make_level and step into engine.py through edit_file()
    (the opening has already put level 0's sprite code into make_level, so the defs are replaced by name)."""
    parts = engine_code.split("\n\n\n")
    edits = [{"op": "replace_def", "name": name, "lines": part.strip("\n")} for name, part in zip(("LAYOUT", "make_level", "step"), parts)]
    return ("python", {"code": f"edit_file(edits={edits!r})"})


def _texts(agent: PlayAgent, role: str = "user") -> list[str]:
    out = []
    for m in agent.messages:
        if m["role"] != role:
            continue
        c = m["content"]
        out.append(c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text"))
    return out


def _agent(tmp_path: Path, environments: Path, model: _ScriptedModel, turns: int = 8, **kw) -> PlayAgent:
    return PlayAgent("twol", tmp_path / "run", ModelConfig(), Budget(max_turns=turns), environments, client=model,
                     images=False, batch_size=4, **kw)


def test_live_game_scores_like_taaf(environments: Path) -> None:
    game = LiveGame("twol", environments)
    game.start()
    for move in ("RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "DOWN", "RIGHT"):
        game.perform(parse_move(move))
    assert game.won and game.levels_completed == 2
    assert game.actions_per_level() == [3, 4]
    # level 0: (3/3)^2 * 100 = 100, weight 1; level 1: (3/4)^2 * 100 = 56.25, weight 2 -> (100 + 112.5) / 3
    assert game.score() == pytest.approx((100 + 2 * 56.25) / 3)
    assert [e["type"] for e in game.events()][:2] == ["initial", "action"] and game.events()[-1]["run_complete"]


def test_a_right_engine_wins_the_game_in_the_baseline_actions(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel([
        [_install(ENGINE.replace("DOWN", "1"))],  # the harness tests it automatically: step 0 passes
        [("commit_engine", {"message": "the player moves; a level is solved at x >= 4"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT", "RIGHT"], "note": "reach x=4"})],  # stops after level 0 is solved
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "and again"})],
    ])
    agent = _agent(tmp_path, environments, model)
    result = agent.run()
    assert result.status == "won" and result.outcome == "won"
    assert result.actions == 6 and result.levels_completed == 2 and result.score == pytest.approx(100.0)
    assert result.batches == 2 and result.moves_sent == 6 and result.mismatches == 0
    assert result.batch_log[0]["sent"] == 3 and result.batch_log[0]["mismatch"] is None
    users = _texts(agent)
    assert users[0].startswith("Plan the next moves. Steps 0-0 pass")
    assert any("Commit accepted" in u for u in users)
    assert any("level 0 solved, so the batch stopped there (1 move(s) not sent)" in u for u in users)
    tool_outputs = _texts(agent, "tool")
    assert any(o.startswith("Sent 3 of 4 move(s) (steps 1-3):") and "#3 RIGHT: matches your prediction (level 0 solved)" in o for o in tool_outputs)
    assert any("The game is won." in o for o in tool_outputs)
    # the records the viewer and scoring read
    assert (tmp_path / "run" / "artifacts" / "twol-0000_p0_viewer_data_events.jsonl").exists()
    run = agent.game_run()
    assert run["state"] == "won" and run["actions_per_level"] == [3, 3] and run["final_score"] == pytest.approx(100.0)
    assert len(run["history"]) == 6


def test_a_wrong_prediction_opens_a_fit_round_and_blocks_moves_until_fixed(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel([
        [_install(ENGINE.replace("DOWN", "-1"))],  # DOWN moves up: blocked by the wall at y=0 in level 0
        [("commit_engine", {"message": "moves"})],
        [("commit_moves", {"actions": ["DOWN", "RIGHT"], "note": "down then right"})],  # step 1 differs; RIGHT not sent
        [("commit_moves", {"actions": ["RIGHT"], "note": "try anyway"})],  # refused: the tests fail
        [("python", {"code": "edit_file(edits=[{'op': 'replace_text', 'oldText': '2: (0, -1)', 'newText': '2: (0, 1)'}])"})],
        [("commit_engine", {"message": "DOWN moves down"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "solve level 0"})],
    ])
    agent = _agent(tmp_path, environments, model)
    result = agent.run()
    assert result.status == "budget_turns"
    assert result.mismatches == 1 and result.refused_batches == 1 and result.batches == 2
    users = _texts(agent)
    fit = next(u for u in users if u.startswith("Fix the engine: step 1 did not go as your engine predicted."))
    assert "What differed: the final frame differs. Step 0 matches. The 1 move after it in your batch was not sent." in fit
    assert "TEST RESULT" in fit and "step 1 is the first failure" in fit
    tool_outputs = _texts(agent, "tool")
    assert any(o.startswith("Not sent: your engine does not reproduce the game so far (steps 0-1)") for o in tool_outputs)
    assert any("#1 DOWN: differs from your prediction: the final frame differs" in o for o in tool_outputs)
    assert result.fit_rounds[0]["step"] == 1 and result.fit_rounds[0]["end_turn"] == 6 and result.fit_rounds[0]["commits"] == 1
    assert result.levels_completed == 1 and result.actions == 4
    # a transcript line per move and per batch
    records = [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines()]
    assert sum("move" in r for r in records) == 4 and sum("batch" in r for r in records) == 2


def test_bad_batches_are_refused_before_anything_is_sent(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel([
        [_install(ENGINE.replace("DOWN", "1"))],
        [("commit_engine", {"message": "ok"})],
        [("commit_moves", {"actions": ["RIGHT"] * 5, "note": "too many"})],
        [("commit_moves", {"actions": ["SPACE"], "note": "not advertised"})],
        [("commit_moves", {"actions": [{"click": [70, 2]}], "note": "off screen"})],
        [("commit_moves", {"actions": ["fly"], "note": "nonsense"})],
        [("commit_moves", {"actions": ["RIGHT"]})],  # no note
        [("commit_moves", {"actions": ["RIGHT"], "note": "one"}), ("commit_moves", {"actions": ["RIGHT"], "note": "two"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=8)
    result = agent.run()
    outputs = _texts(agent, "tool")
    assert any(o.startswith("Not sent: at most 4 moves per call") for o in outputs)
    assert any(o.startswith("Not sent: this game does not accept SPACE") for o in outputs)
    assert any(o.startswith("Not sent: the click MOUSE(row=2, col=70) is off the 64x64 screen") for o in outputs)
    assert any(o.startswith("Not sent: unrecognised action 'fly'") for o in outputs)
    assert any(o.startswith("Not sent: commit_moves needs a note") for o in outputs)
    assert any(o.startswith("Not sent: one commit_moves call per turn") for o in outputs)
    assert result.actions == 1 and result.refused_batches == 6


def test_a_game_over_is_followed_by_the_harness_reset(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel([
        [_install(ENGINE.replace("DOWN", "1"))],
        [("commit_engine", {"message": "ok"})],
        [("commit_moves", {"actions": ["LEFT", "RIGHT"], "note": "LEFT at x=1 loses"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "solve level 0"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=4)
    result = agent.run()
    users = _texts(agent)
    assert any("The game was over, so the harness sent a RESET (step 2, an action)" in u for u in users)
    assert result.auto_resets == 1 and result.mismatches == 0 and result.actions == 5 and result.levels_completed == 1
    assert [s.action.id for s in agent.live.trace.steps] == [0, 3, 0, 4, 4, 4]
    assert result.actions_per_level == [5, 0]


def test_a_resumed_run_continues_the_game_and_the_conversation(tmp_path: Path, environments: Path) -> None:
    first = _ScriptedModel([
        [_install(ENGINE.replace("DOWN", "1"))],
        [("commit_engine", {"message": "ok"})],
        [("commit_moves", {"actions": ["RIGHT"], "note": "one step"})],
    ])
    agent = _agent(tmp_path, environments, first, turns=3)
    result = agent.run()
    assert result.status == "budget_turns" and result.actions == 1
    second = _ScriptedModel([
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "finish level 0"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "finish level 1"})],
    ])
    agent2 = _agent(tmp_path, environments, second, turns=5)
    result2 = agent2.run()
    assert result2.resumes == 1 and result2.status == "won" and result2.actions == 6
    assert len(agent2.live.trace) == 7 and result2.batches == 3  # the earlier batch is restored from result.json
    assert any("The run was interrupted here and has now resumed" in u for u in _texts(agent2))
