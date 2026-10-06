"""The play-and-model agent (engine_re.play_agent) on tiny games, with a scripted model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_re.agent import Budget, ModelConfig
from engine_re.live_game import LiveGame, benchmark_json
from engine_re.play_agent import PlayAgent
from engine_re.prompts import ENGINE_ELIDED, ENGINE_HEADER, PLAN_CLOSING, elide_engine_listing
from engine_re.trace import Action, action_code, parse_move, parse_moves

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
RIGHT_ENGINE = ENGINE.replace("DOWN", "1")
WRONG_DOWN = ENGINE.replace("DOWN", "-1")  # DOWN moves up: blocked by the wall at y=0 in level 0

# A click game (sp80 advertises keys and clicks): SPACE does nothing; a click on the target cell solves the level.
CLICK_GAME = '''
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite


class Clik(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(sprites=[Sprite([[9]], name="target", x=2, y=2), Sprite([[4]], name="mark", x=5, y=5)], grid_size=(8, 8)),
            Level(sprites=[Sprite([[9]], name="target", x=6, y=1)], grid_size=(8, 8)),
        ]
        super().__init__(game_id="clik", levels=levels, camera=Camera(0, 0, 8, 8, 0, 3), available_actions=[5, 6])

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            cell = self.camera.display_to_grid(self.action.data.get("x", 0), self.action.data.get("y", 0))
            target = self.current_level.get_sprites_by_name("target")[0]
            if cell is not None and tuple(cell) == (target.x, target.y):
                self.next_level()
        self.complete_action()
'''

CLICK_ENGINE = """
TARGETS = {0: (2, 2), 1: (6, 1)}


def make_level(n):
    tx, ty = TARGETS[n]
    sprites = [
        Sprite([[3] * 64 for _ in range(64)], screen=True, layer=-2, collidable=False, name="border"),
        Sprite([[0] * 8 for _ in range(8)], layer=-1, collidable=False, name="background"),
        Sprite([[9]], x=tx, y=ty, name="target"),
    ]
    if n == 0:
        sprites.append(Sprite([[4]], x=5, y=5, name="mark"))
    return State(grid=(8, 8), sprites=sprites)


def step(state, action):
    if action.id == 6 and action.cell == TARGETS[state.level]:
        state.status = "level_solved"
"""


def _write_game(root: Path, name: str, code: str, baseline: list[int]) -> None:
    folder = root / name / "0000"
    folder.mkdir(parents=True)
    (folder / f"{name}.py").write_text(code, encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"game_id": f"{name}-0000", "baseline_actions": baseline}), encoding="utf-8")


@pytest.fixture()
def environments(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    _write_game(root, "twol", GAME, [3, 3])
    _write_game(root, "clik", CLICK_GAME, [2, 1])
    return root


class _ScriptedModel:
    def __init__(self, turns: list[list[tuple[str, dict]]]):
        self.turns = turns
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools):  # noqa: ARG002
        self.seen.append([dict(m) for m in messages])
        calls = self.turns.pop(0) if self.turns else []
        if callable(calls):  # a turn written from what the model has seen (e.g. a python output)
            calls = calls(messages)
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


def _install(engine_code: str, names: tuple[str, ...] = ("LAYOUT", "make_level", "step")) -> tuple[str, dict]:
    """A python call that puts `engine_code`'s top-level definitions into engine.py through edit_file() (the opening
    has already put level 0's sprite code into make_level, so the defs are replaced by name)."""
    parts = engine_code.split("\n\n\n")
    edits = [{"op": "replace_def", "name": name, "lines": part.strip("\n")} for name, part in zip(names, parts)]
    return ("python", {"code": f"edit_file(edits={edits!r})"})


def _edit(old: str, new: str) -> tuple[str, dict]:
    return ("python", {"code": f"edit_file(edits=[{{'op': 'replace_text', 'oldText': {old!r}, 'newText': {new!r}}}])"})


NOTHING = ("python", {"code": "x = 1"})
MOVES = "    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}"  # step()'s line in RIGHT_ENGINE


def _texts(agent: PlayAgent, role: str = "user") -> list[str]:
    out = []
    for m in agent.messages:
        if m["role"] != role:
            continue
        c = m["content"]
        out.append(c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text"))
    return out


def _records(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines()]


def _agent(tmp_path: Path, environments: Path, model: _ScriptedModel, turns: int = 8, game: str = "twol", **kw) -> PlayAgent:
    kw.setdefault("images", False)
    kw.setdefault("batch_size", 4)
    return PlayAgent(game, tmp_path / "run", ModelConfig(), Budget(max_turns=turns), environments, client=model, **kw)


def _start(engine: str = RIGHT_ENGINE, **install) -> list[list[tuple[str, dict]]]:
    """The first two turns: install the engine (tested automatically), then commit it."""
    return [[_install(engine, **install)], [("commit_engine", {"message": "the rules"})]]


# --- the live game, its score and records -------------------------------------------------------


def test_live_game_scores_like_taaf(environments: Path) -> None:
    from taaf.game import GameRun

    game = LiveGame("twol", environments)
    game.start()
    for move in ("RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "DOWN", "RIGHT"):
        game.perform(parse_move(move))
    assert game.won and game.levels_completed == 2
    assert game.actions_per_level() == [3, 4]
    # level 0: (3/3)^2 * 100 = 100, weight 1; level 1: (3/4)^2 * 100 = 56.25, weight 2 -> (100 + 112.5) / 3
    assert game.score() == pytest.approx((100 + 2 * 56.25) / 3)
    events = game.events()
    assert [e["type"] for e in events][:2] == ["initial", "action"] and events[-1]["run_complete"]
    assert [e["level"] for e in events] == [1, 1, 1, 2, 2, 2, 2, 2]  # the level shown after each step, 1-based
    # TAAF's own formula and counts on the record written for it
    run = GameRun.from_json_dict(game.game_run(state="won"))
    assert run._compute_final_score() == pytest.approx(game.score()) and run.actions_per_level == [3, 4]
    assert len(run.history) == sum(run.actions_per_level) == game.actions


def test_live_game_counts_a_game_over_and_its_reset_in_the_level(environments: Path) -> None:
    game = LiveGame("twol", environments)
    game.start()
    for move in ("LEFT", "RESET", "RIGHT", "RIGHT", "RIGHT"):
        game.perform(parse_move(move))
    assert [s.state for s in game.trace.steps][1:3] == ["GAME_OVER", "NOT_FINISHED"]
    assert game.actions_per_level() == [5, 0] and game.score() == pytest.approx((3 / 5) ** 2 * 100 / 3)
    no_baseline = LiveGame("twol", environments)
    no_baseline.metadata.pop("baseline_actions")
    no_baseline.start()
    assert no_baseline.score() is None and no_baseline.game_run()["final_score"] == 0.0  # as TAAF: 0 without baselines


def test_parse_move_forms() -> None:
    assert parse_move("UP") == Action(1) and parse_move("reset") == Action(0) and parse_move("ACTION5") == Action(5)
    assert parse_move({"click": [12, 40]}) == Action(6, 12, 40) == parse_move("MOUSE(row=40, col=12)") == parse_move("click(12, 40)")
    assert parse_move({"action": "MOUSE", "row": 40, "col": 12}) == Action(6, 12, 40) == parse_move([6, 12, 40])
    with pytest.raises(ValueError):
        parse_move("fly")


def test_the_fixed_block_action_prints_as_commit_moves_takes_it() -> None:
    """The kernel's Action (the fixed block's, game_api.canonical) prints as the code that builds it, cell left out;
    that text, the dataclass form an engine's own Action prints, and the objects themselves parse back to the move."""
    from engine_re.game_api import FIXED_INTERFACE, canonical, same_interface

    api = canonical()
    moves = [api.Action(4), api.Action(6, 12, 40, cell=(1, 5)), api.Action(0), api.Action(id=1)]
    assert repr(moves) == "[Action(4), Action(6, x=12, y=40), Action(0), Action(1)]" and str(moves[1]) == "Action(6, x=12, y=40)"
    want = [Action(4), Action(6, 12, 40), Action(0), Action(1)]
    assert parse_moves(repr(moves)) == parse_moves([repr(m) for m in moves]) == parse_moves(moves) == want
    assert [action_code(a) for a in want] == [repr(m) for m in moves]  # how the play messages name the moves
    assert parse_move("Action(id=6, x=12, y=40, cell=(9, 9))") == Action(6, 12, 40)  # the cell is the harness's to compute
    assert parse_move("Action(id=4, x=0, y=0, cell=None)") == Action(4) == parse_move({"id": 4, "x": 0, "y": 0})
    assert parse_move("Action(id=1, x=None, y=None)") == Action(1)  # a recorded action, as it prints
    assert parse_moves('[UP, "RESET", {"click": [3, 4]}, Action(6, x=5, y=6)]') == [Action(1), Action(0), Action(6, 3, 4), Action(6, 5, 6)]
    for bad in ("Action(6)", "Action(9)", "Action(4, z=1)", "[Action(4), fly]"):
        with pytest.raises(ValueError):
            parse_moves(bad)
    action_block = FIXED_INTERFACE.split("class Action:")[1].split("class View:")[0]
    assert "__repr__" not in action_block and same_interface(FIXED_INTERFACE)  # the repr is the harness's: the block is as it was


# --- the loop -------------------------------------------------------------------------------------


def test_a_right_engine_wins_the_game_in_the_baseline_actions(tmp_path: Path, environments: Path) -> None:
    from engine_re.trace import trace_from_run

    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT", "RIGHT"], "note": "reach x=4"})],  # stops after level 0 is solved
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT", "RIGHT"], "note": "and again"})],  # WIN at the third
    ])
    agent = _agent(tmp_path, environments, model)
    result = agent.run()
    assert result.status == "won" and result.outcome == "won"
    assert result.actions == 6 and result.levels_completed == 2 and result.score == pytest.approx(100.0)
    assert result.batches == 2 and result.moves_sent == 6 and result.mismatches == 0
    assert result.batch_log[0]["sent"] == 3 and result.batch_log[0]["mismatch"] is None
    assert result.batch_log[1]["sent"] == 3 and len(result.batch_log[1]["moves"]) == 4  # WIN inside the batch: the 4th not sent
    users = _texts(agent)
    assert users[0].startswith("Plan the next moves. Steps 0-0 pass") and "The game has just started" in users[0]
    assert "The game accepts: Action(1) UP, Action(2) DOWN, Action(3) LEFT, Action(4) RIGHT, Action(0) RESET" in users[0]
    assert any("Commit accepted" in u for u in users)
    assert any("level 0 solved, and your make_level(1) drew the new level's start as the game did, so the batch stopped there "
               "(1 move(s) not sent)" in u for u in users)
    tool_outputs = _texts(agent, "tool")
    assert any(o.startswith("Sent 3 of 4 move(s) (steps 1-3):") and "#3 Action(4): matches your prediction (level 0 solved)" in o for o in tool_outputs)
    assert any("The game is won." in o for o in tool_outputs)
    assert result.final["passed"] and result.committed_sha and (tmp_path / "run" / "engine_committed.py").exists()
    # the records the viewer and scoring read
    events = tmp_path / "run" / "artifacts" / "twol-0000_p0_events.jsonl"
    assert events.exists()
    trace, mismatches = trace_from_run(tmp_path / "run", "twol", environments)
    assert mismatches == [] and len(trace) == len(agent.live.trace) == 7
    run = agent.game_run()
    assert run["state"] == "won" and run["actions_per_level"] == [3, 3] and run["final_score"] == pytest.approx(100.0)
    assert len(run["history"]) == 6 and sum(h["generated_tokens"] for h in run["history"]) + run["final_generated_tokens"] == 20
    # every record carries the phase it was logged in
    records = _records(tmp_path)
    assert all(r.get("phase") in ("plan", "fit") for r in records)
    assert sum("plan" in r for r in records) == 3  # the opening, after the commit, after level 0


def test_a_wrong_prediction_opens_a_fit_round_and_blocks_moves_until_fixed(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["DOWN", "RIGHT"], "note": "down then right"})],  # step 1 differs; RIGHT not sent
        [("commit_moves", {"actions": ["RIGHT"], "note": "try anyway"})],  # refused: the tests fail
        [_edit("2: (0, -1)", "2: (0, 1)")],
        [("commit_engine", {"message": "DOWN moves down"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "solve level 0"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=7)
    result = agent.run()
    assert result.status == "budget_turns"
    assert result.mismatches == 1 and result.refused_batches == 1 and result.batches == 2
    users = _texts(agent)
    fit = next(u for u in users if u.startswith("Fix your replica: step 1 did not go as your replica predicted."))
    assert "What differed: the final frame differs. Step 0 matches. The 1 move after it in your batch was not sent." in fit
    assert "TEST RESULT" in fit and "step 1 is the first failure" in fit
    tool_outputs = _texts(agent, "tool")
    assert any(o.startswith("Not sent: your replica does not reproduce the game so far (steps 0-1)") for o in tool_outputs)
    assert any("#1 Action(2): differs from your prediction: the final frame differs" in o for o in tool_outputs)
    assert "after the commit you plan the next moves" in tool_outputs[4]  # the fit round's commit hint
    assert len(result.fit_rounds) == 1
    assert result.fit_rounds[0]["step"] == 1 and result.fit_rounds[0]["end_turn"] == 6 and result.fit_rounds[0]["commits"] == 1
    assert result.fit_rounds[0]["accepted"] and result.fit_rounds[0]["turns"] == 3
    assert result.batch_log[0]["diff"] == "the final frame differs"
    assert result.levels_completed == 1 and result.actions == 4
    assert result.phase_turns == {"plan": 4, "fit": 3}
    records = _records(tmp_path)
    assert sum("move" in r for r in records) == 4 and sum("batch" in r for r in records) == 2


def test_bad_batches_are_refused_before_anything_is_sent(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [
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
    assert any(o.startswith("Not sent: this game does not accept Action(5) (SPACE)") for o in outputs)
    assert any(o.startswith("Not sent: the click Action(6, x=70, y=2) is off the 64x64 screen") for o in outputs)
    assert any(o.startswith("Not sent: unrecognised action 'fly'") for o in outputs)
    assert any(o.startswith("Not sent: commit_moves needs a note") for o in outputs)
    assert any(o.startswith("Not sent: one commit_moves call per turn") for o in outputs)
    assert result.actions == 1 and result.refused_batches == 6


def test_a_predicted_game_over_is_followed_by_the_harness_reset(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [
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


def test_an_unpredicted_game_over_is_fitted_then_reset(tmp_path: Path, environments: Path) -> None:
    no_loss = RIGHT_ENGINE.replace('    if action.id == 3 and player.x == 1:\n        state.status = "game_over"\n        return\n', "")
    model = _ScriptedModel(_start(no_loss) + [
        [("commit_moves", {"actions": ["LEFT", "RIGHT"], "note": "probe LEFT"})],  # the game is lost; the engine says not
        [_install(RIGHT_ENGINE)],
        [("commit_engine", {"message": "LEFT at x=1 loses"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5)
    result = agent.run()
    users = _texts(agent)
    fit = next(u for u in users if u.startswith("Fix your replica: step 1"))
    assert "outcome: the game says 'GAME_OVER', your replica 'NOT_FINISHED'" in fit
    assert "After it the game is over; the harness will RESET the level once your replica reproduces it." in fit
    assert "The game was over, so the harness sent a RESET (step 2, an action): the level restarted as your replica predicted." in users[-1]
    assert result.auto_resets == 1 and result.mismatches == 1 and [s.action.id for s in agent.live.trace.steps] == [0, 3, 0]


def test_no_auto_reset_leaves_the_reset_to_the_model(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["LEFT"], "note": "lose"})],
        [("commit_moves", {"actions": ["RIGHT"], "note": "refused: the game is over"})],
        [("commit_moves", {"actions": ["RESET", "RIGHT"], "note": "restart"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5, auto_reset=False)
    result = agent.run()
    assert "The game is over: only RESET is accepted now." in _texts(agent)[-2]
    assert any(o.startswith("Not sent: the game is over, so only RESET is accepted now.") for o in _texts(agent, "tool"))
    assert result.auto_resets == 0 and [s.action.id for s in agent.live.trace.steps] == [0, 3, 0, 4]


def test_a_commit_and_a_batch_in_one_turn(tmp_path: Path, environments: Path) -> None:
    """commit_engine then commit_moves: the commit is the batch's (one advance with its message, no implicit one); a
    batch then commit_engine after a difference: the fit round is recorded and closed by the commit."""
    model = _ScriptedModel([
        [_install(WRONG_DOWN)],
        [("commit_engine", {"message": "explicit"}), ("commit_moves", {"actions": ["RIGHT"], "note": "go"})],
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"}), _edit("2: (0, -1)", "2: (0, 1)"),
         ("commit_engine", {"message": "DOWN moves down"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "solve"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=4)
    result = agent.run()
    messages = [a["message"] for a in result.advances]
    assert messages == ["explicit", "DOWN moves down"] and not any(a.get("implicit") for a in result.advances)
    assert result.mismatches == 1 and len(result.fit_rounds) == 1 and result.fit_rounds[0]["accepted"]
    users = _texts(agent)
    assert sum(u.startswith("Plan the next moves") for u in users) == 4 and not any(u.startswith("Fix the engine") for u in users)
    assert "Your last batch: 1 move(s) sent, all as your replica predicted." in users[1]
    assert ("Your last batch: 1 move(s) sent; step 2, Action(2), differed from your replica's prediction (the final frame differs). "
            "Commit accepted: your replica now reproduces steps 0-2.") in users[2]
    assert result.levels_completed == 1 and result.actions == 4


def test_a_batch_then_an_edit_in_one_turn(tmp_path: Path, environments: Path) -> None:
    """The batch was sent with the committed engine; engine.py edited after it in the same turn is tested
    automatically, and the next message says where it stands."""
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["RIGHT"], "note": "one"}), _edit(MOVES, MOVES.replace("4: (1, 0)", "4: (2, 0)"))],  # matched, then broken
        [("commit_moves", {"actions": ["RIGHT"], "note": "refused"})],
        [("python", {"code": "undo_edit()"}), ("commit_moves", {"actions": ["DOWN"], "note": "probe"}),
         _edit(MOVES, MOVES.replace("2: (0, 1)", "2: (0, 2)"))],
    ])
    agent = _agent(tmp_path, environments, model, turns=5)
    agent.run()
    users = _texts(agent)
    plan = users[2]
    assert plan.startswith("Plan the next moves. Steps 0-1 pass with your replica as committed.")
    assert "engine.py has changed since your last commit and does not reproduce every step played" in plan
    tools = _texts(agent, "tool")
    assert tools[2].startswith("Sent 1 of 1 move(s)") and "[harness] engine.py changed, so it was tested automatically" in tools[3]
    assert tools[4].startswith("Not sent: your replica does not reproduce the game so far")
    # after undo_edit() the DOWN batch is sent with the committed engine and matches; the edit after it is reported
    assert any("#2 Action(2): matches your prediction" in t for t in tools)
    assert "engine.py has changed since your last commit and does not reproduce every step played" in users[-1]


def test_a_commit_that_passes_up_to_its_step_but_fails_a_later_one(tmp_path: Path, environments: Path) -> None:
    broken = _edit("moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}", "moves = {1: (0, -1), 2: (0, -1), 3: (-1, 0), 4: (2, 0)}")
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["DOWN", "RIGHT", "RIGHT"], "note": "three"})],
        [broken],
        [("commit_moves", {"actions": ["RIGHT"], "note": "refused: step 1 fails"})],  # a fit round on step 1
        [_edit("2: (0, -1)", "2: (0, 1)")],
        [("commit_engine", {"message": "DOWN fixed"})],  # steps 0-1 pass, step 2 does not
        [_edit("4: (2, 0)", "4: (1, 0)")],
        [("commit_engine", {"message": "RIGHT fixed"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=9)
    result = agent.run()
    users = _texts(agent)
    advance = next(u for u in users if u.startswith("Commit accepted: steps 0-1 pass."))
    assert "The next step, 2, fails." in advance and "Fix step 2" in advance
    assert [(r["step"], r["end"]) for r in result.fit_rounds] == [(1, "accepted"), (2, "accepted")]
    assert [(a["fixed"], a["next"]) for a in result.advances if not a.get("implicit")] == [(0, None), (1, 2), (2, None)]
    assert users[-1].startswith("Plan the next moves. Steps 0-3 pass")


def test_engine_errors_during_the_prediction(tmp_path: Path, environments: Path) -> None:
    # (an engine that raised on level 0 would fail the contract tests, which play level 0, and send nothing)
    raising = RIGHT_ENGINE.replace("    moves = {", "    if action.id == 1 and state.level == 1:\n        raise ValueError('UP is not modelled')\n    moves = {")
    model = _ScriptedModel(_start(raising) + [[("commit_moves", {"actions": ["RIGHT"] * 3, "note": "level 0"})],
                                              [("commit_moves", {"actions": ["UP", "RIGHT"], "note": "probe UP"})]])
    agent = _agent(tmp_path, environments, model, turns=4)
    result = agent.run()
    tools = _texts(agent, "tool")
    assert "#4 Action(1): differs from your prediction: your replica raised an error (ValueError: UP is not modelled)" in tools[-1]
    assert result.mismatches == 1 and result.actions == 4  # the move was still sent; RIGHT was not
    assert _texts(agent)[-1].startswith("Fix your replica: step 4 did not go as your replica predicted.")


def test_a_hud_pixel_difference_matches_with_its_warning(tmp_path: Path, environments: Path) -> None:
    pixel = RIGHT_ENGINE.replace(
        "    if player.x >= 4:", "    if action.id == 1:\n        state.add(Sprite([[7]], x=63, y=0, screen=True, layer=5))\n    if player.x >= 4:")
    model = _ScriptedModel(_start(pixel) + [[("commit_moves", {"actions": ["UP", "RIGHT"], "note": "UP is blocked"})]])
    agent = _agent(tmp_path, environments, model, turns=3)
    result = agent.run()
    out = _texts(agent, "tool")[-1]
    assert "#1 Action(1): matches your prediction (1 px differs at the frame border" in out and "tolerated as HUD-bar rounding" in out
    assert result.mismatches == 0 and result.actions == 2


def test_a_click_game(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(CLICK_ENGINE, names=("TARGETS", "make_level", "step")) + [
        [("commit_moves", {"actions": ["UP"], "note": "not accepted"})],
        [("commit_moves", {"actions": ["SPACE", {"click": [40, 40]}, "MOUSE(row=20, col=20)", {"click": [50, 10]}],
                           "note": "space, a miss, the target"})],
        [("commit_moves", {"actions": [{"click": [50, 10]}], "note": "level 1's target at cell (6, 1)"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5, game="clik")
    result = agent.run()
    tools = _texts(agent, "tool")
    assert tools[2].startswith("Not sent: this game does not accept Action(1) (UP). The game accepts: Action(5) SPACE, clicks "
                               "Action(6, x=x, y=y) (x the column, y the row, screen pixels 0-63), Action(0) RESET")
    assert "#3 Action(6, x=20, y=20): matches your prediction (level 0 solved)" in tools[3]
    assert "clicks Action(6, x=x, y=y) (x the column, y the row, screen pixels 0-63)" in _texts(agent)[0]
    assert result.status == "won" and result.actions == 4 and result.actions_per_level == [3, 1]
    assert [s.action for s in agent.live.trace.steps][1:] == [Action(5), Action(6, 40, 40), Action(6, 20, 20), Action(6, 50, 10)]


def test_actions_stepped_on_the_replica_are_sent_as_printed(tmp_path: Path, environments: Path) -> None:
    """A click game played the way the prompt says: moves tried on the replica in python (a click's cell filled
    in by replica.step, as the harness does), the printed list pasted into commit_moves as one string, sent."""
    plan = ("import copy\nt = copy.deepcopy(state_now())\nmoves = [Action(5), Action(0), Action(6, x=20, y=20)]\n"
            "replica.step(t, moves[0])\nt = replica.make_level(0)\nreplica.step(t, moves[2])\nprint(t.status)\nprint(moves)")
    model = _ScriptedModel(_start(CLICK_ENGINE, names=("TARGETS", "make_level", "step")) + [
        [("python", {"code": plan})],
        lambda messages: [("commit_moves", {"actions": _printed_moves(messages), "note": "space, reset, the target"})],
        [("python", {"code": "t = copy.deepcopy(state_now())\nmoves = [Action(6, x=50, y=10)]\nreplica.step(t, moves[0])\n"
                             "print(t.status)\nprint(moves)"})],
        lambda messages: [("commit_moves", {"actions": _printed_moves(messages), "note": "level 1's target"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6, game="clik")
    result = agent.run()
    tools = _texts(agent, "tool")
    assert "level_solved\n[Action(5), Action(0), Action(6, x=20, y=20)]" in tools[2]
    assert tools[3].startswith("Sent 3 of 3 move(s) (steps 1-3):") and "#2 Action(0): matches your prediction" in tools[3]
    assert "#3 Action(6, x=20, y=20): matches your prediction (level 0 solved)" in tools[3]
    assert "level_solved\n[Action(6, x=50, y=10)]" in tools[4] and "#4 Action(6, x=50, y=10): matches your prediction" in tools[5]
    assert result.status == "won" and result.mismatches == 0 and result.actions_per_level == [3, 1]
    assert [s.action for s in agent.live.trace.steps][1:] == [Action(5), Action(0), Action(6, 20, 20), Action(6, 50, 10)]


def test_the_action_budget_cuts_a_batch_and_ends_the_run_cleanly(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "DOWN"], "note": "three, one cut"})],
        [("python", {"code": "x = 1"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6, max_actions=2)
    result = agent.run()
    assert result.status == "budget_actions" and result.actions == 2
    assert "The last 1 move(s) of the batch were cut: the run allows 2 actions in all." in _texts(agent, "tool")[-1]
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["status"] == "budget_actions" and saved["actions"] == 2 and saved["final"]["passed"]
    assert (tmp_path / "run" / "artifacts" / "twol-0000_p0_events.jsonl").exists()
    assert len(agent.game_run()["history"]) == 2


def test_the_action_budget_mid_fit_round(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [[("commit_moves", {"actions": ["DOWN"], "note": "probe"})], [NOTHING]])
    agent = _agent(tmp_path, environments, model, turns=6, max_actions=1)
    result = agent.run()
    assert result.status == "budget_actions" and result.turns == 3 and result.phase == "fit"
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["final"]["first_fail"] == 1 and saved["fit_rounds"][0]["end_turn"] is None
    assert json.loads((tmp_path / "run" / "trace" / "trace.json").read_text())["steps"][-1]["action"] == {"id": 2}


# --- nudges and the escape hatch ------------------------------------------------------------------


def test_the_plan_nudge(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [[NOTHING]] * 4 + [[("commit_moves", {"actions": ["RIGHT"], "note": "go"})], [NOTHING]])
    agent = _agent(tmp_path, environments, model, turns=8, plan_turns=2)
    result = agent.run()
    tools = _texts(agent, "tool")
    nudged = [i for i, t in enumerate(tools) if "turns in this plan round without commit_moves" in t]
    # turn 1 (the install) and 2 (the commit) count, then a new plan round starts: turns 4 and 6 are nudged, not 7-8
    assert nudged == [3, 5] and result.plan_nudges == 2
    assert "send a short batch now, even a probing one of 1-3 moves" in tools[3]
    assert sum("plan_nudge" in r for r in _records(tmp_path)) == 2


def test_the_pure_loop_never_offers_the_escape_hatch(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [[("commit_moves", {"actions": ["DOWN"], "note": "probe"})]] + [[NOTHING]] * 4
                           + [[("commit_moves", {"actions": ["RIGHT"], "note": "refused"})]])
    agent = _agent(tmp_path, environments, model, turns=8)
    result = agent.run()
    assert not any("out of step" in t for t in _texts(agent, "tool"))
    assert result.escapes == 0 and result.unexplained == [] and _texts(agent, "tool")[-1].startswith("Not sent: your replica does not")


def test_the_escape_hatch_plays_blind_and_resyncs_at_a_reset(tmp_path: Path, environments: Path) -> None:
    raising_down = WRONG_DOWN.replace("    moves = {", "    if action.id == 2 and player.y > 1:\n        raise ValueError('lost')\n    moves = {")
    model = _ScriptedModel(_start(raising_down) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})],  # step 1 differs: a fit round
        [NOTHING],
        [NOTHING],  # 2 turns: the escape is offered
        [("commit_moves", {"actions": ["DOWN", "DOWN", "RESET", "RIGHT"], "note": "blind"})],  # sent unchecked up to the RESET
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "in step again"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=7, fit_turns=2)
    result = agent.run()
    tools = _texts(agent, "tool")
    assert "You may now play on with your replica out of step" in tools[4]
    blind = tools[5]
    assert blind.startswith("Sent 3 of 4 move(s) (steps 2-4), not checked: your replica is out of step with the game since step 1.")
    assert "#4 Action(0): sent, not checked (your replica is out of step) (your replica is back in step with the game here)" in blind
    assert result.unexplained == [1, 2, 3] and result.resync == {"4": {"level": 0, "score": 0}} and result.out_of_sync is None
    plan = next(u for u in _texts(agent) if "Every step played passes with your replica again" in u)
    assert plan.startswith("Plan the next moves. Steps 0-4 pass with your replica as committed (but the unexplained ones, which are "
                           "not compared: 1-3).")
    assert "#7 Action(4): matches your prediction (level 0 solved)" in tools[-1]
    assert result.levels_completed == 1 and result.escapes == 1 and result.fit_rounds[0]["end"] == "escaped"
    # the tests: the unexplained steps never fail, even where the engine raised; the final test passes
    assert result.final["passed"] and result.final["ignored"] == [1, 2, 3] and result.final["total"] == 5
    text = (tmp_path / "run" / "final_test.txt").read_text()
    assert "Unexplained steps 1-3 (played while your replica was out of step with the game)" in text


def test_the_escape_hatch_through_a_level_change(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})],
        [NOTHING],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "blind"})],
        [("commit_moves", {"actions": ["RIGHT", "UP"], "note": "blind, solves level 0"})],  # back in step at the level change
        [("commit_moves", {"actions": ["RIGHT"], "note": "checked again"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=7, fit_turns=1)
    result = agent.run()
    users = _texts(agent)
    assert any(u.startswith("Plan the next moves. Your replica is out of step with the game since step 1") for u in users)
    assert result.unexplained == [1, 2, 3] and result.resync == {"4": {"level": 1, "score": 1}}
    assert "#5 Action(4): matches your prediction" in _texts(agent, "tool")[-1]
    assert result.final["passed"] and result.levels_completed == 1


def test_the_escape_hatch_resyncs_at_the_automatic_reset(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})],
        [NOTHING],
        [("commit_moves", {"actions": ["LEFT"], "note": "blind: loses"})],
        [("commit_moves", {"actions": ["RIGHT"], "note": "checked"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6, fit_turns=1)
    result = agent.run()
    assert result.unexplained == [1, 2] and result.resync == {"3": {"level": 0, "score": 0}} and result.auto_resets == 1
    assert any("The game was over, so the harness sent a RESET (step 3, an action), which puts your replica back in step." in u
               for u in _texts(agent))
    assert "#4 Action(4): matches your prediction" in _texts(agent, "tool")[-1]


def test_a_resync_the_engine_does_not_reproduce_opens_a_fit_round(tmp_path: Path, environments: Path) -> None:
    wrong_level_1 = WRONG_DOWN.replace("1: ((1, 3), 8, 7)", "1: ((2, 3), 8, 7)")
    model = _ScriptedModel(_start(wrong_level_1) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})],
        [NOTHING],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "blind to the level change"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5, fit_turns=1)
    result = agent.run()
    fit = _texts(agent)[-1]
    assert fit.startswith("Fix your replica: your replica does not reproduce step 4.")
    assert "now that it is back in step with the game" in fit and result.phase == "fit" and result.fit_rounds[-1]["step"] == 4


# --- resume ---------------------------------------------------------------------------------------


def test_a_resumed_run_continues_the_game_and_the_conversation(tmp_path: Path, environments: Path) -> None:
    first = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT"], "note": "one step"})]])
    agent = _agent(tmp_path, environments, first, turns=3, images=True)
    result = agent.run()
    assert result.status == "budget_turns" and result.actions == 1
    second = _ScriptedModel([
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "finish level 0"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "finish level 1"})],
    ])
    agent2 = _agent(tmp_path, environments, second, turns=5, images=True)
    result2 = agent2.run()
    assert result2.resumes == 1 and result2.status == "won" and result2.actions == 6
    assert len(agent2.live.trace) == 7 and result2.batches == 3  # the earlier batch is restored from result.json
    assert any("The run was interrupted here and has now resumed" in u for u in _texts(agent2))
    def texts(messages: list[dict]) -> list:
        return [m["content"] if isinstance(m["content"], str) else [p.get("text") for p in m["content"]] for m in messages]

    # the same conversation, rebuilt with its images: only the latest PLAN message's frame is still a live image
    assert texts(second.seen[0][: len(first.seen[-1])])[:-2] == texts(first.seen[-1])[:-2]
    live = [i for i, m in enumerate(second.seen[0]) if isinstance(m["content"], list)
            and any(p.get("type") == "image_url" for p in m["content"])]
    assert len(live) == 1 and second.seen[0][live[0]]["content"][0]["text"].startswith("Plan the next moves. Steps 0-1 pass")
    assert result2.committed_sha == result.committed_sha and len(result2.step_tokens) == 6


def test_a_run_resumed_in_its_first_plan_round_keeps_its_conversation(tmp_path: Path, environments: Path) -> None:
    first = _ScriptedModel([[NOTHING]])
    _agent(tmp_path, environments, first, turns=1).run()
    second = _ScriptedModel([[("commit_moves", {"actions": ["RIGHT"], "note": "go"})]])
    agent = _agent(tmp_path, environments, second, turns=2)
    agent.run()
    assert len(second.seen[0]) == len(first.seen[0]) + 3  # the assistant turn, its tool output, the resume note
    assert "x = 1" in json.dumps(second.seen[0])


def test_a_run_resumed_in_a_fit_round(tmp_path: Path, environments: Path) -> None:
    first = _ScriptedModel(_start(WRONG_DOWN) + [[("commit_moves", {"actions": ["DOWN"], "note": "probe"})], [NOTHING]])
    _agent(tmp_path, environments, first, turns=4, fit_turns=3).run()
    second = _ScriptedModel([[NOTHING], [_edit("2: (0, -1)", "2: (0, 1)")], [("commit_engine", {"message": "fixed"})]])
    agent = _agent(tmp_path, environments, second, turns=7, fit_turns=3)
    result = agent.run()
    assert len(result.fit_rounds) == 1 and result.fit_rounds[0]["step"] == 1 and result.fit_rounds[0]["accepted"]
    assert result.fit_rounds[0]["escape_offered"] == 6  # three turns after the round opened at turn 3, across the resume
    assert _texts(agent)[-1].startswith("Plan the next moves. Steps 0-1 pass")


def test_a_run_interrupted_in_the_middle_of_a_batch(tmp_path: Path, environments: Path, monkeypatch) -> None:
    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "two"})]])
    agent = _agent(tmp_path, environments, model, turns=5)
    real_play = PlayAgent._play

    def crash(self, acts, note):
        real_play(self, acts[:1], note)  # the first move is played and saved, then the process dies
        raise KeyboardInterrupt

    monkeypatch.setattr(PlayAgent, "_play", crash)
    with pytest.raises(KeyboardInterrupt):
        agent.run()
    monkeypatch.setattr(PlayAgent, "_play", real_play)
    second = _ScriptedModel([[("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "finish level 0"})]])
    agent2 = _agent(tmp_path, environments, second, turns=4)
    result = agent2.run()
    users = _texts(agent2)
    assert "The run was interrupted while moves were being played: steps 1-1 were played after the last message" in users[-2]
    assert users[-2].startswith("Plan the next moves. Steps 0-1 pass") and result.levels_completed == 1
    assert len(agent2.live.trace) == 4 and len(result.step_tokens) == 3


def test_a_finished_game_gives_its_benchmark_record_again(tmp_path: Path, environments: Path) -> None:
    from taaf.benchmark import Benchmark

    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT"] * 3, "note": "a"})],
                                       [("commit_moves", {"actions": ["RIGHT"] * 3, "note": "b"})]])
    agent = _agent(tmp_path, environments, model)
    agent.run()
    record = agent.game_run()
    again = _agent(tmp_path, environments, _ScriptedModel([]))  # what run_play does for a finished game
    again._restore()
    assert again.game_run() | {"final_wallclock_seconds": 0, "started_at": 0} == record | {"final_wallclock_seconds": 0, "started_at": 0}
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark_json("t", [record], 0.0)))
    loaded = Benchmark.from_json(path)
    assert loaded.game_runs[0].final_score == pytest.approx(100.0) and loaded.game_runs[0].state == "won"


# --- context --------------------------------------------------------------------------------------


def test_compaction_elides_the_listing_of_older_plan_messages(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT"], "note": "one"})]])
    agent = _agent(tmp_path, environments, model, turns=3, images=True)
    agent.run()
    plans = [m for m in agent.messages if m["role"] == "user" and isinstance(m["content"], list)
             and m["content"][0]["text"].startswith("Plan the next moves")]
    assert len(plans) == 3 and plans[-1]["content"][-1]["type"] == "image_url"
    texts = [p["content"][0]["text"] for p in plans]
    assert [ENGINE_HEADER in t for t in texts] == [True, True, False]  # listed when it changed
    agent._compact()
    texts = [p["content"][0]["text"] for p in plans]
    assert ENGINE_ELIDED in texts[0] and ENGINE_HEADER not in texts[0]
    assert ENGINE_HEADER in texts[1] and ENGINE_ELIDED not in texts[1]  # the latest listing stays
    assert all(PLAN_CLOSING.strip() in t for t in texts)  # the paragraph after the listing is kept
    text = "Head.\n\n" + ENGINE_HEADER + "\n\n  1#ABC:x = 1\n" + PLAN_CLOSING + " on your replica."
    assert elide_engine_listing(text) == f"Head.\n\n{ENGINE_ELIDED}{PLAN_CLOSING} on your replica."


def test_the_play_agent_keeps_compaction(tmp_path: Path, environments: Path) -> None:
    with pytest.raises(ValueError, match="compact"):
        PlayAgent("twol", tmp_path / "run", ModelConfig(context="condense"), Budget(), environments, client=_ScriptedModel([]))


def test_the_play_prompts_say_nothing_of_a_recording_or_run_tests_levels() -> None:
    from engine_re.prompts import system_prompt, tools

    for images in (True, False):
        text = system_prompt(mode="play", images=images)
        assert "run_tests(level" not in text and "first message" not in text and "the recorded frame as images" not in text
        assert "state_now() -> State" in text and "click_cell(state, x, y) -> tuple | None" in text
        assert "replica.step(s, Action(1))" in text and "engine.step" not in text and "simulate" not in text
        names = [t["function"]["name"] for t in tools(images, "play", True)]
        assert names == ["python", "run_tests", "commit_engine", "commit_moves"]
        run_tests = tools(images, "play", True)[1]["function"]
        assert "level" not in run_tests["parameters"]["properties"]


# --- the kernel's play built-ins ------------------------------------------------------------------


def _printed_moves(messages: list[dict]) -> str:
    """The printed list of Actions ("[Action(4), ...]") in the last python output the model got."""
    out = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
    return next(line for line in out.splitlines() if line.startswith("[Action("))


def test_state_now_and_the_replica_in_the_kernel(tmp_path: Path, environments: Path) -> None:
    """The play kernel: state_now() is the replica's State after everything played, and moves are played by
    calling replica.step on copies of it; the Actions print as the code that builds them, and that printed list,
    pasted into commit_moves, is what is sent (a RESET among them), each move matching the prediction."""
    from engine_re.kernel import KernelClient

    plan = ("import copy\ns = state_now()\nt = copy.deepcopy(s)\nmoves = [Action(4), Action(0), Action(4), Action(4)]\n"
            "replica.step(t, moves[0])\nt = replica.make_level(t.level)\nreplica.step(t, moves[2])\nprint('x', t.vars['player'].x, t.status)\n"
            "print(moves)")
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["RIGHT"], "note": "one"})],
        [("python", {"code": plan})],
        lambda messages: [("commit_moves", {"actions": [m.strip() for m in _printed_moves(messages)[1:-1].split(",")], "note": "as printed"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5)
    agent.run()
    tools = _texts(agent, "tool")
    assert "x 2 playing\n[Action(4), Action(0), Action(4), Action(4)]" in tools[-2]
    assert tools[-1].startswith("Sent 4 of 4 move(s) (steps 2-5):") and "#3 Action(0): matches your prediction" in tools[-1]
    assert "#5 Action(4): matches your prediction" in tools[-1] and "differs" not in tools[-1]
    assert [s.action for s in agent.live.trace.steps][2:] == [Action(4), Action(0), Action(4), Action(4)]
    assert agent.result.batch_log[-1]["moves"] == ["Action(4)", "Action(0)", "Action(4)", "Action(4)"]
    run = tmp_path / "run"
    kernel = KernelClient(run / "workspace", run / "visible_trace", timeout=60, images=False, focus=5, history=True, play=True)
    try:
        out = kernel.execute(
            "import copy\ns = state_now()\nprint('x', s.vars['player'].x)\n"
            "t = copy.deepcopy(s)\nreplica.step(t, Action(4))\nprint('after RIGHT', t.vars['player'].x, t.status, s.vars['player'].x)\n"
            "print('cell', click_cell(t, 3, 4))\nreplica.step(t, Action(6, x=3, y=4))\n"
            "nxt = replica.make_level(1)\nprint('level 1 sprites', len(nxt.sprites))"
        )
        assert "state_now(): your replica after replaying steps 0-5: level 0, NOT_FINISHED" in out and "x 3" in out
        assert "after RIGHT 4 level_solved 3" in out and "cell (0, 0)" in out and "level 1 sprites 4" in out
        for name in ("state_now", "click_cell", "replica"):
            assert "nothing was run" in kernel.execute(f"{name} = 1"), name
    finally:
        kernel.stop()


def test_the_kernel_replays_unexplained_steps_and_resyncs(tmp_path: Path, environments: Path) -> None:
    from engine_re.kernel import KernelClient

    raising_down = WRONG_DOWN.replace("    moves = {", "    if action.id == 2 and player.y > 1:\n        raise ValueError('lost')\n    moves = {")
    model = _ScriptedModel(_start(raising_down) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})], [NOTHING],
        [("commit_moves", {"actions": ["DOWN", "RESET"], "note": "blind"})],
    ])
    _agent(tmp_path, environments, model, turns=5, fit_turns=1).run()
    run = tmp_path / "run"
    kernel = KernelClient(run / "workspace", run / "visible_trace", timeout=60, images=False, focus=3, history=True, play=True)
    try:
        out = kernel.execute("s = state_now()\nb, a = replay_step(3)")
        assert "your replica after replaying steps 0-3: level 0, NOT_FINISHED" in out
        assert "step 3 is where your engine was put back in step with the game" in out and "your frame matches the recording after step 3" in out
    finally:
        kernel.stop()


def test_a_run_resumed_out_of_step(tmp_path: Path, environments: Path) -> None:
    first = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["DOWN"], "note": "probe"})], [NOTHING],
        [("commit_moves", {"actions": ["RIGHT"], "note": "blind"})],
    ])
    result = _agent(tmp_path, environments, first, turns=5, fit_turns=1).run()
    assert result.out_of_sync == 1 and result.unexplained == [1, 2]
    second = _ScriptedModel([[("commit_moves", {"actions": ["RESET", "RIGHT"], "note": "back in step"})],
                             [("commit_moves", {"actions": ["RIGHT"], "note": "checked"})]])
    agent = _agent(tmp_path, environments, second, turns=7, fit_turns=1)
    result = agent.run()
    assert result.unexplained == [1, 2] and result.resync == {"3": {"level": 0, "score": 0}} and result.out_of_sync is None
    assert "#4 Action(4): matches your prediction" in _texts(agent, "tool")[-1] and result.final["passed"]
    assert [r["end"] for r in result.fit_rounds] == ["escaped"]


def test_run_play_writes_what_score_run_reads_and_skips_finished_games(tmp_path: Path, environments: Path, monkeypatch) -> None:
    import sys

    from engine_re import run_play
    from inference.tools.eval import _run_evaluations_from_benchmark

    scripts = {
        "twol": _start() + [[("commit_moves", {"actions": ["RIGHT"] * 3, "note": "a"})], [("commit_moves", {"actions": ["RIGHT"] * 4, "note": "b"})]],
        "clik": _start(CLICK_ENGINE, names=("TARGETS", "make_level", "step")) + [[("commit_moves", {"actions": [{"click": [20, 20]}], "note": "a"})]],
    }

    class Scripted(PlayAgent):
        def __init__(self, game, *args, **kw):
            super().__init__(game, *args, client=_ScriptedModel([list(t) for t in scripts[game]]), **kw)

    monkeypatch.setattr(run_play, "PlayAgent", Scripted)
    out = tmp_path / "out"
    argv = ["run_play", "--games", "twol,clik", "--out", str(out), "--environments-dir", str(environments), "--max-turns", "5",
            "--no-images", "--batch-size", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_play.main() == 0
    first = json.loads((out / "benchmark.json").read_text())
    assert [r["state"] for r in first["game_runs"]] == ["won", "gave_up"]
    assert "| twol | won | 100.0 | 2/2 | 6 |" in (out / "summary.md").read_text()
    [evaluation] = _run_evaluations_from_benchmark(out)
    scores = {g.game_id: g.score for g in evaluation.games}
    assert scores == {"twol-0000": pytest.approx(100.0), "clik-0000": pytest.approx(100 / 3)}
    assert (out / "twol" / "artifacts" / "twol-0000_p0_events.jsonl").exists()
    # again: the finished games are skipped and give the same records
    assert run_play.main() == 0
    again = json.loads((out / "benchmark.json").read_text())
    strip = ("final_wallclock_seconds", "started_at")
    assert [{k: v for k, v in r.items() if k not in strip} for r in again["game_runs"]] == \
        [{k: v for k, v in r.items() if k not in strip} for r in first["game_runs"]]
