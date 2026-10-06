"""The play-and-model agent (engine_re.play_agent) on tiny games, with a scripted model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_re.agent import (
    IMAGE_NOTE, IMAGE_PLACEHOLDER, IMAGE_TOKENS_FALLBACK, REBUILT_CLOSING, Budget, ModelConfig, TurnMessage, image_part_tokens,
    is_context_length_error, rebuilt_context, render_request, shrink_step, split_for_estimate,
)
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

# A click game (sp80 advertises keys and clicks): SPACE does nothing; a click on the target cell solves the level; a
# click on the mark (level 0) pushes it one cell to the right.
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
            marks = self.current_level.get_sprites_by_name("mark")
            if cell is not None and tuple(cell) == (target.x, target.y):
                self.next_level()
            elif cell is not None and marks and tuple(cell) == (marks[0].x, marks[0].y):
                self.try_move("mark", 1, 0)
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
    mark = state.by_name("mark")
    if action.id == 6 and action.cell == TARGETS[state.level]:
        state.status = "level_solved"
    elif action.id == 6 and mark is not None and action.cell == (mark.x, mark.y):
        mark.x += 1
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
        [("commit_moves", {"actions": ["DOWN"], "note": "down"})],  # step 1 differs (a probe of one: the replica predicts no change)
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
    assert "What differed: the final frame differs. Step 0 matches." in fit
    assert "TEST RESULT" in fit and "step 1 is the first failure" in fit
    assert "\nActions played: 1 of at most 500 (level 0: 1). Turns: 3 of 7;" in fit  # the budget line, after a batch
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
    model = _ScriptedModel(_start(pixel) + [[("commit_moves", {"actions": ["UP"], "note": "UP is blocked"})]])
    agent = _agent(tmp_path, environments, model, turns=3)
    result = agent.run()
    out = _texts(agent, "tool")[-1]
    assert "#1 Action(1): matches your prediction (1 px differs at the frame border" in out and "tolerated as HUD-bar rounding" in out
    assert result.mismatches == 0 and result.actions == 1


def test_a_click_game(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(CLICK_ENGINE, names=("TARGETS", "make_level", "step")) + [
        [("commit_moves", {"actions": ["UP"], "note": "not accepted"})],
        [("commit_moves", {"actions": ["SPACE"], "note": "a probe: the replica predicts nothing"})],
        [("commit_moves", {"actions": [{"click": [40, 40]}, "MOUSE(row=20, col=20)", {"click": [50, 10]}],
                           "note": "the mark, the target"})],
        [("commit_moves", {"actions": [{"click": [50, 10]}], "note": "level 1's target at cell (6, 1)"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6, game="clik")
    result = agent.run()
    tools = _texts(agent, "tool")
    assert tools[2].startswith("Not sent: this game does not accept Action(1) (UP). The game accepts: Action(5) SPACE, clicks "
                               "Action(6, x=x, y=y) (x the column, y the row, screen pixels 0-63), Action(0) RESET")
    assert tools[3].startswith("Sent 1 of 1 move(s) (steps 1-1):\n  #1 Action(5): matches your prediction")
    assert "#3 Action(6, x=20, y=20): matches your prediction (level 0 solved)" in tools[4]
    assert "What the batch changed on the board (steps 1 -> 2" not in tools[4]  # the batch entered level 1: no change summary
    assert "clicks Action(6, x=x, y=y) (x the column, y the row, screen pixels 0-63)" in _texts(agent)[0]
    assert result.status == "won" and result.actions == 4 and result.actions_per_level == [3, 1]
    assert [s.action for s in agent.live.trace.steps][1:] == [Action(5), Action(6, 40, 40), Action(6, 20, 20), Action(6, 50, 10)]


def test_actions_stepped_on_the_replica_are_sent_as_printed(tmp_path: Path, environments: Path) -> None:
    """A click game played the way the prompt says: moves tried on the replica in python (a click's cell filled
    in by replica.step, as the harness does), the printed list pasted into commit_moves as one string, sent."""
    plan = ("import copy\nt = copy.deepcopy(state_now())\nmoves = [Action(6, x=40, y=40), Action(0), Action(6, x=20, y=20)]\n"
            "replica.step(t, moves[0])\nt = replica.make_level(0)\nreplica.step(t, moves[2])\nprint(t.status)\nprint(moves)")
    model = _ScriptedModel(_start(CLICK_ENGINE, names=("TARGETS", "make_level", "step")) + [
        [("python", {"code": plan})],
        lambda messages: [("commit_moves", {"actions": _printed_moves(messages), "note": "the mark, reset, the target"})],
        [("python", {"code": "t = copy.deepcopy(state_now())\nmoves = [Action(6, x=50, y=10)]\nreplica.step(t, moves[0])\n"
                             "print(t.status)\nprint(moves)"})],
        lambda messages: [("commit_moves", {"actions": _printed_moves(messages), "note": "level 1's target"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6, game="clik")
    result = agent.run()
    tools = _texts(agent, "tool")
    assert "level_solved\n[Action(6, x=40, y=40), Action(0), Action(6, x=20, y=20)]" in tools[2]
    assert tools[3].startswith("Sent 3 of 3 move(s) (steps 1-3):") and "#2 Action(0): matches your prediction" in tools[3]
    assert "#3 Action(6, x=20, y=20): matches your prediction (level 0 solved)" in tools[3]
    assert "level_solved\n[Action(6, x=50, y=10)]" in tools[4] and "#4 Action(6, x=50, y=10): matches your prediction" in tools[5]
    assert result.status == "won" and result.mismatches == 0 and result.actions_per_level == [3, 1]
    assert [s.action for s in agent.live.trace.steps][1:] == [Action(6, 40, 40), Action(0), Action(6, 20, 20), Action(6, 50, 10)]


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

    def crash(self, acts, note, **kw):
        real_play(self, acts[:1], note, **kw)  # the first move is played and saved, then the process dies
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
    assert len(plans) == 3 and plans[-1]["content"][-2]["type"] == "image_url"
    assert plans[-1]["content"][-1]["text"].startswith("Your replica's sprites now")  # the sprite list, under the image
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
    agent = PlayAgent("twol", tmp_path / "run2", ModelConfig(context="rebuilt"), Budget(), environments, client=_ScriptedModel([]))
    assert agent.rebuilt and not agent.condense


# --- the rebuilt context ----------------------------------------------------------------------------


def _synthetic_conversation(turns: int) -> list[TurnMessage]:
    """A play conversation of `turns` model turns, tagged as the agent tags it: the opening PLAN message (turn 0), a
    commit_engine at turn 3, a commit_moves at turn 5 followed by a PLAN message with an image (turn 5's tail), an
    image message at turn 7, a commit_engine at turn 12; every other turn a python call. Every reply has reasoning."""
    png = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [
        TurnMessage({"role": "system", "content": "SYSTEM PROMPT"}, turn=0),
        TurnMessage({"role": "user", "content": [{"type": "text", "text": "Plan the next moves. OPENING"}, png]}, turn=0, phase="plan"),
    ]
    for t in range(1, turns + 1):
        if t == 3 or t == 12:
            call = {"id": f"c{t}", "type": "function", "function": {"name": "commit_engine", "arguments": json.dumps({"message": f"COMMIT MESSAGE {t}"})}}
            output = f"Committed at turn {t}."
        elif t == 5:
            call = {"id": f"c{t}", "type": "function", "function": {"name": "commit_moves", "arguments": json.dumps({"actions": ["Action(4)"], "note": "NOTE 5"})}}
            output = "Sent 1 of 1 move(s)."
        else:
            call = {"id": f"c{t}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": f"x = {t}\nprint(x)"})}}
            output = f"OUTPUT {t}"
        messages.append(TurnMessage({"role": "assistant", "content": f"TEXT {t}", "reasoning": f"REASONING {t}", "tool_calls": [call]}, turn=t))
        messages.append(TurnMessage({"role": "tool", "tool_call_id": call["id"], "content": output}, turn=t))
        if t == 5:
            messages.append(TurnMessage({"role": "user", "content": [{"type": "text", "text": "Plan the next moves. PHASE 5"}, png]}, turn=t, phase="plan"))
        if t == 7:
            messages.append(TurnMessage({"role": "user", "content": [{"type": "text", "text": IMAGE_NOTE}, {"type": "text", "text": "CAPTION 7"}, png]}, turn=t))
    return messages


def _text_of(message: dict) -> str:
    c = message["content"]
    return c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text")


def _images_of(message: dict) -> int:
    c = message["content"]
    return 0 if isinstance(c, str) else sum(p.get("type") == "image_url" for p in c)


def test_the_rebuilt_context_with_the_phase_message_among_the_last_turns() -> None:
    conversation = _synthetic_conversation(14)  # the window is turns 5-14: the PLAN message of turn 5 is in it
    view, stats = rebuilt_context(conversation, keep_turns=10)
    assert view[0] is conversation[0] and view[0]["role"] == "system"
    compacted = _text_of(view[1])
    assert view[1]["role"] == "user" and not isinstance(view[1], TurnMessage)
    assert compacted.endswith(REBUILT_CLOSING)
    assert "Turn 3 (commit_engine):" in compacted and "COMMIT MESSAGE 3" in compacted and "Committed at turn 3." in compacted
    assert "Turn 5" not in compacted and "the current PLAN message" not in compacted  # inside the window: sent as it is
    assert "REASONING" not in compacted and "OUTPUT 1" not in compacted and "Turn 1" not in compacted
    assert _images_of(view[1]) == 0
    real = view[2:]
    assert real == [m for m in conversation if m.turn >= 5] and all(m is n for m, n in zip(real, [m for m in conversation if m.turn >= 5]))
    assert any(m.phase == "plan" and m.turn == 5 for m in real) and real[0]["role"] == "assistant"  # the window starts at a turn
    assert [m["reasoning"] for m in real if m["role"] == "assistant"] == [f"REASONING {t}" for t in range(5, 15)]
    assert stats["commit_turns"] == 1 and stats["phase_turn"] == 5 and stats["older_turns"] == 0 and not stats["phase_compacted"]
    assert stats["recent_turns"] == 10 and stats["chars"]["phase"] == 0 and stats["chars"]["commits"] == len(compacted.split("\n\n", 1)[1].rsplit("\n\n", 1)[0])
    assert stats["images"] == 2  # the PLAN message's and the image message's, both real


def test_the_rebuilt_context_with_an_older_phase_message() -> None:
    conversation = _synthetic_conversation(20)  # the window is turns 11-20: the PLAN message of turn 5 is older
    view, stats = rebuilt_context(conversation, keep_turns=10)
    assert view[0]["role"] == "system" and view[1]["role"] == "user"
    parts = view[1]["content"]
    text = _text_of(view[1])
    heads = [line for line in text.splitlines() if line.startswith("Turn ")]
    assert heads == ["Turn 3 (commit_engine):", "Turn 5 (commit_moves):", "Turn 5 (the current PLAN message):",
                     "Turn 6:", "Turn 7:", "Turn 8:", "Turn 9:", "Turn 10:"]
    assert text.endswith(REBUILT_CLOSING)
    assert "REASONING" not in text  # never in the compacted message
    assert "NOTE 5" in text and "Sent 1 of 1 move(s)." in text and "PHASE 5" in text
    assert "x = 6\nprint(x)" in text and "OUTPUT 6" in text and "TEXT 6" in text  # an older turn: calls, arguments and outputs
    assert "CAPTION 7" not in text and IMAGE_NOTE not in text  # an image message is not rendered
    assert "Turn 1" not in heads and "OUTPUT 2" not in text  # a turn before the phase message without a commit is dropped
    # The phase message's image is the compacted message's only one, right after its header.
    assert _images_of(view[1]) == 1
    i = next(i for i, p in enumerate(parts) if p.get("type") == "image_url")
    assert parts[i - 1]["text"].endswith("Turn 5 (the current PLAN message):\n\nPlan the next moves. PHASE 5")
    assert len(parts) == 3 and parts[2]["text"].startswith("The turns since that message")  # text, image, text
    real = view[2:]
    assert real == [m for m in conversation if m.turn >= 11] and real[0]["role"] == "assistant"
    assert [m["reasoning"] for m in real if m["role"] == "assistant"] == [f"REASONING {t}" for t in range(11, 21)]
    assert "Turn 12" not in text and any("COMMIT MESSAGE 12" in c["function"]["arguments"]  # the commit of turn 12 is real
                                         for m in real for c in m.get("tool_calls") or [])
    assert stats["commit_turns"] == 2 and stats["phase_turn"] == 5 and stats["phase_compacted"] and stats["older_turns"] == 5
    assert stats["recent_turns"] == 10 and stats["images"] == 1 and stats["chars"]["phase"] == len("Plan the next moves. PHASE 5")
    assert stats["messages"] == 2 + len(real)
    # Every commit turn is somewhere: the older ones in the compacted message, the recent one real.
    for t in (3, 5, 12):
        assert f"Turn {t} (commit_" in text or any(m.turn == t for m in real)
    # A short conversation needs no compacted message: the request is the conversation.
    short = _synthetic_conversation(6)
    view, stats = rebuilt_context(short, keep_turns=10)
    assert view == short and stats["commit_turns"] == 0 and stats["recent_turns"] == 7  # turns 0-6


class _StubCounter:
    """A tokenizer stub: one token per `chars` characters of the request's json, the images at their vision cost."""

    def __init__(self, chars: float = 4.0):
        self.chars = chars
        self.calls = 0

    def count(self, messages, tools=None):
        self.calls += 1
        rendered, image_tokens = render_request(messages, tools)
        text = int(len(rendered) / self.chars)
        return {"tokens": text + image_tokens, "text_tokens": text, "image_tokens": image_tokens}

    def describe(self) -> str:
        return "stub"


def _png_part(width: int, height: int) -> dict:
    """An image part whose data URL is a PNG header of that size (enough for png_dimensions)."""
    import base64
    import struct

    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00" * 8
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(header).decode()}}


def test_image_parts_count_at_their_vision_cost() -> None:
    assert image_part_tokens(_png_part(536, 554)) == 17 * 18 + 2 == 308  # the PLAN frame
    assert image_part_tokens(_png_part(1060, 554)) == 34 * 18 + 2 == 614  # the test comparison
    assert image_part_tokens({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}) == IMAGE_TOKENS_FALLBACK
    scrubbed, tokens = split_for_estimate([{"role": "user", "content": [{"type": "text", "text": "t"}, _png_part(64, 64)]}])
    assert tokens == 6 and scrubbed[0]["content"][1]["image_url"]["url"] == "<image>"
    assert is_context_length_error("OpenRouter HTTP 400: This model's maximum context length is 131072 tokens")
    assert is_context_length_error("HTTP 400: too many tokens") and not is_context_length_error("HTTP 500: internal error")


def test_the_shrink_steps_in_order() -> None:
    conversation = _synthetic_conversation(20)  # window 11-20, the PLAN message of turn 5 older, 2 commit turns before it
    shrink: dict = {}
    taken = []
    for _ in range(20):
        view, stats = rebuilt_context(conversation, keep_turns=10, shrink=shrink)
        following = shrink_step(shrink, stats, keep_turns=10)
        if following is None:
            break
        shrink = following
        taken.append({k: v for k, v in shrink.items()})
    # reasoning of the window's oldest turns first (7 of the 10, never the last 3), then the older turns, then the
    # commit turns (only 2 here, under the floor of 5: untouched), then the phase message's image.
    assert [t.get("reasoning") for t in taken[:7]] == [1, 2, 3, 4, 5, 6, 7]
    assert taken[7] == {"reasoning": 7, "older": True} and taken[8] == {"reasoning": 7, "older": True, "phase_image": True}
    assert len(taken) == 9
    view, stats = rebuilt_context(conversation, keep_turns=10, shrink=taken[-1])
    real = view[2:]
    assistants = [m for m in real if m["role"] == "assistant"]
    assert [bool(m.get("reasoning")) for m in assistants] == [False] * 7 + [True] * 3  # the last 3 keep theirs
    assert all(m.get("reasoning") for m in conversation if m["role"] == "assistant")  # the conversation is untouched
    text = _text_of(view[1])
    assert "The turns since that message" not in text and "Turn 3 (commit_engine):" in text and "PHASE 5" in text
    assert _images_of(view[1]) == 0 and IMAGE_PLACEHOLDER in text and stats["images"] == 0  # (the turn-7 image message is older)
    assert stats["shrink"] == {"reasoning": 7, "older": True, "commits": 0, "phase_image": True}


def test_the_rebuilt_request_is_counted_and_shrunk_to_the_budget(tmp_path: Path, environments: Path) -> None:
    agent = PlayAgent("twol", tmp_path / "run", ModelConfig(context="rebuilt", context_window=40_000, reply_reserve=1_000),
                      Budget(), environments, client=_ScriptedModel([]), images=False)
    assert agent._context_budget() == 40_000 - 1_000 - 512
    agent.messages = _synthetic_conversation(20)
    for m in agent.messages:
        if m["role"] == "assistant":
            m["reasoning"] = "reasoning " * 1500  # 15K chars a turn: the window alone is far over the budget
    agent.counter = _StubCounter(chars=4.0)
    view, stats = agent._rebuilt_view()
    assert stats["exact"] and stats["budget"] == 38_488 and stats["margin"] == stats["estimated_tokens"] * 2 // 100
    assert stats["estimated_tokens"] + stats["margin"] <= stats["budget"]
    assert stats["shrink"]["reasoning"] >= 1 and not stats["shrink"].get("older")  # stopped as soon as it fitted
    assistants = [m for m in view if m["role"] == "assistant"]
    assert assistants[-1].get("reasoning") and not assistants[0].get("reasoning")
    # Too big to ever fit: every step is taken, the request is sent as it is with a warning, nothing truncated.
    agent.model.context_window = 12_000
    view, stats = agent._rebuilt_view()
    assert stats["shrink"] == {"reasoning": 7, "older": True, "commits": 0, "phase_image": True} and "warning" in stats
    assert all(m.get("reasoning") for m in [m for m in view if m["role"] == "assistant"][-3:])
    assert "PHASE 5" in _text_of(view[1])
    # Without a counter: the calibrated estimate, no margin, and the same shrink loop.
    agent.counter = None
    agent.model.context_window = 40_000
    view, stats = agent._rebuilt_view()
    assert not stats["exact"] and stats["chars_per_token"] == 3.0 and stats["margin"] == 0 and stats["estimated_tokens"] <= stats["budget"]


def test_the_calibration_clamps_and_logs(tmp_path: Path, environments: Path) -> None:
    agent = PlayAgent("twol", tmp_path / "run", ModelConfig(context="rebuilt"), Budget(), environments, client=_ScriptedModel([]), images=False)
    messages = [{"role": "system", "content": "s" * 4000}, {"role": "user", "content": [{"type": "text", "text": "u"}, _png_part(536, 554)]}]
    rendered, image_tokens = render_request(messages, agent._tools())
    assert image_tokens == 308
    agent._calibrate_from_usage(messages, agent._tools(), {"prompt_tokens": 308 + len(rendered) // 10})  # 10 chars a token
    assert agent.chars_per_token == 3.3  # the ceiling
    agent._calibrate_from_usage(messages, agent._tools(), {"prompt_tokens": 308 + len(rendered) * 5})
    assert agent.chars_per_token == 1.0  # the floor
    agent._calibrate_from_usage(messages, agent._tools(), {"prompt_tokens": 308 + int(len(rendered) / 2.8)})
    assert abs(agent.chars_per_token - 2.8) < 0.01 and agent._calibrations == 3
    records = [r["token_calibration"] for r in agent.records if "token_calibration" in r]
    assert [r["chars_per_token"] for r in records] == [3.3, 1.0, pytest.approx(2.8, abs=0.01)]
    assert records[0]["measured"] == pytest.approx(10.0, abs=0.01) and records[0]["image_tokens"] == 308
    agent._calibrate_from_usage(messages, agent._tools(), {"prompt_tokens": 308 + int(len(rendered) / 2.82)})
    assert len([r for r in agent.records if "token_calibration" in r]) == 3  # a move under 0.05 is not logged
    agent._calibrate_from_usage(messages, agent._tools(), {})  # no usage: nothing changes
    assert abs(agent.chars_per_token - 2.82) < 0.01
    # A rejected request lowers the ceiling for the run and the figure in use with it.
    agent._context_overflow("OpenRouter HTTP 400: maximum context length exceeded")
    assert agent.chars_ceiling == pytest.approx(2.82 * 0.9, abs=0.01) and agent.chars_per_token == agent.chars_ceiling
    assert agent.context_overflows == 1 and any("context_overflow" in r for r in agent.records)


def test_a_context_length_error_is_retried_once_after_a_further_shrink(tmp_path: Path, environments: Path) -> None:
    class Rejecting(_ScriptedModel):
        def __init__(self, turns):
            super().__init__(turns)
            self.rejected = False

        def chat(self, messages, tools):
            if len(self.seen) == 2 and not self.rejected:  # the third request: rejected once
                self.rejected = True
                raise RuntimeError('OpenRouter HTTP 400: {"error": "This model\'s maximum context length is 131072 tokens"}')
            return super().chat(messages, tools)

    model = Rejecting(_start() + [[("commit_moves", {"actions": ["RIGHT"], "note": "one"})], [NOTHING]])
    agent = PlayAgent("twol", tmp_path / "run2", ModelConfig(context="rebuilt"), Budget(max_turns=4), environments, client=model,
                      images=False, batch_size=4)
    result = agent.run()
    assert result.status == "budget_turns" and agent.context_overflows == 1 and len(model.seen) == 4
    records = _records_at(tmp_path / "run2")
    overflow = [r for r in records if "context_overflow" in r]
    # (the scripted usage says 10 prompt tokens, so the calibration sits at the 3.3 ceiling: the rejection takes it to 2.97)
    assert len(overflow) == 1 and overflow[0]["turn"] == 2 and overflow[0]["context_overflow"]["ceiling"] == pytest.approx(2.97, abs=0.01)
    rebuilt = [r for r in records if "rebuilt" in r]
    assert [r["turn"] for r in rebuilt] == [0, 1, 2, 2, 3]  # the rejected request's record, then the retry's
    assert rebuilt[3]["rebuilt"]["chars_per_token"] <= 2.97 and not rebuilt[3]["rebuilt"]["exact"]
    calibrations = [r for r in records if "token_calibration" in r]
    assert calibrations and calibrations[0]["turn"] == 1  # from the first response on


def _records_at(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "transcript.jsonl").read_text().splitlines()]


def test_the_real_tokenizer_counts_a_request() -> None:
    from engine_re.tokens import TokenCounter, tokenizer_folder

    folder = tokenizer_folder(None)
    if folder is None:
        pytest.skip("no tokenizer files (ARC3_TOKENIZER unset, the default not cached)")
    counter = TokenCounter(folder)
    assert counter.count_text("hello world") == 2
    messages = [{"role": "system", "content": "Be brief."}, {"role": "assistant", "content": "ok", "reasoning": "why"},
                {"role": "user", "content": [{"type": "text", "text": "Look:"}, _png_part(536, 554)]}]
    text, blocks, images = counter.render(messages, None)
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n") and text.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>assistant\nok<|im_end|>" in text and blocks == ["<think>\nwhy\n</think>\n\n"] and len(images) == 1
    counted = counter.count(messages, None)
    assert counted["image_tokens"] == 308 and counted["reasoning_tokens"] == counter.count_text(blocks[0]) > 0
    assert counted["tokens"] == counter.count_text(text) + counted["reasoning_tokens"] + 308
    tools = [{"type": "function", "function": {"name": "python", "description": "run", "parameters": {"type": "object", "properties": {}}}}]
    with_tools, _, _ = counter.render(messages, tools)
    assert "# Tools" in with_tools and '"name": "python"' in with_tools
    assert TokenCounter(folder, reasoning="template").count(messages, None)["reasoning_tokens"] == 0


def test_the_play_agent_in_rebuilt_mode(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT"], "note": "one"})], [NOTHING]])
    config = ModelConfig(context="rebuilt", compact_prompt_tokens=0, rebuilt_keep_turns=1)
    agent = PlayAgent("twol", tmp_path / "run", config, Budget(max_turns=4), environments, client=model, images=False, batch_size=4)
    result = agent.run()
    assert result.status == "budget_turns" and result.context == "rebuilt"
    records = _records(tmp_path)
    assert not any("compact" in r for r in records)  # never compacted, whatever the prompt size
    rebuilt = [r for r in records if "rebuilt" in r]
    assert [r["turn"] for r in rebuilt] == [0, 1, 2, 3] and len(model.seen) == 4
    assert all(set(r["rebuilt"]) >= {"commit_turns", "phase_turn", "older_turns", "chars", "estimated_tokens"} for r in rebuilt)
    assert rebuilt[-1]["rebuilt"]["commit_turns"] == 1 and rebuilt[-1]["rebuilt"]["phase_turn"] == 3
    assert rebuilt[-1]["rebuilt"]["chars"]["total"] == sum(rebuilt[-1]["rebuilt"]["chars"][k] for k in ("system", "commits", "phase", "older", "recent"))
    # Every message knows its turn; the transcript logs plain messages.
    assert [m.turn for m in agent.messages] == [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    assert [m.phase for m in agent.messages if m.phase] == ["plan", "plan", "plan"]
    assert all("turn" not in r["message"] and "phase" not in r["message"] for r in records if "message" in r)
    # The last request: the system prompt, the compacted context with the commit turn, then turn 3 as it is.
    last = model.seen[-1]
    assert last[0]["role"] == "system" and last[1]["role"] == "user"
    text = last[1]["content"]
    assert text.startswith("Earlier turns of this game in which you committed") and text.endswith(REBUILT_CLOSING)
    assert "Turn 2 (commit_engine):" in text and "the rules" in text and "Turn 1" not in text and "Turn 3" not in text
    assert [m["role"] for m in last[2:]] == ["assistant", "tool", "user"] and _texts(agent)[-1] == _text_of(last[-1])
    # Resumed: the conversation is rebuilt with the same tags, and the next request is rebuilt from it.
    second = _ScriptedModel([[NOTHING]])
    agent2 = PlayAgent("twol", tmp_path / "run", config, Budget(max_turns=5), environments, client=second, images=False, batch_size=4)
    agent2.run()
    n = len(agent.messages)
    assert agent2.messages[:n] == agent.messages
    assert [(m.turn, m.phase) for m in agent2.messages[:n]] == [(m.turn, m.phase) for m in agent.messages]
    assert agent2.messages[n].turn == 4 and "has now resumed" in agent2.messages[n]["content"]
    first = second.seen[0]
    text = first[1]["content"]
    assert first[0]["role"] == "system" and "Turn 2 (commit_engine):" in text and "Turn 3 (the current PLAN message):" in text
    assert "Plan the next moves. Steps 0-1 pass" in text and text.endswith(REBUILT_CLOSING)
    assert [m["role"] for m in first[2:]] == ["assistant", "tool", "user"] and "has now resumed" in first[-1]["content"]


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
    # config.json names the harness version from one constant, with the git short sha when git gives it (v11 follow-up 9).
    from engine_re import PLAY_VERSION

    harness = json.loads((out / "config.json").read_text())["harness"]
    assert harness.startswith(f"engine_re.play_agent ({PLAY_VERSION}") and "(v10)" not in harness and harness == run_play.harness_label()
    assert harness == f"engine_re.play_agent ({PLAY_VERSION})" or harness.startswith(f"engine_re.play_agent ({PLAY_VERSION}, git ")
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


# --- support: what the recorded steps ran (PLAY_DESIGN.md 3.11) -----------------------------------

SUPPORT_SOURCE = """FIXED = 1
# ==== END OF FIXED INTERFACE ====
import numpy as np

CALLS = []


def probe(name, value):
    CALLS.append(name)
    return value


def rule(a, b, c):
    if probe("a", a) and probe("b", b):
        out = "both"
    elif a or c:
        out = "either"
    else:
        out = "none"
    keep = c and np.array([1, 2])
    while False:
        pass
    return out if keep is not None else "never"


def boom():
    x = 1
    raise ValueError("here")
"""


def _line(source: str, text: str) -> int:
    """The first line holding `text` after the FIXED block (the whole file when it has none)."""
    lines = source.splitlines()
    start = next((n for n, line in enumerate(lines, 1) if line.strip() == "# ==== END OF FIXED INTERFACE ===="), 0)
    return next(n for n, line in enumerate(lines, 1) if n > start and text in line)


def test_the_condition_rewrite_keeps_values_order_and_lines_and_the_tracer_records_them() -> None:
    import traceback
    import types

    from engine_re import support

    inst = support.instrument(SUPPORT_SOURCE, "/x/engine.py")
    assert inst is not None and inst.first == 3
    kinds = [(c["kind"], c["text"]) for c in inst.conds]
    assert ("and", 'probe("a", a)') in kinds and ("if", 'probe("a", a) and probe("b", b)') in kinds
    assert ("or", "a") in kinds and ("ifexp", "keep is not None") in kinds and ("and", "np.array([1, 2])") in kinds
    assert not any(c["kind"] == "while" for c in inst.conds)  # a constant test is left alone
    assert [g["op"] for g in inst.groups] == ["and", "or", "and"]
    assert 1 not in inst.lines and _line(SUPPORT_SOURCE, 'out = "both"') in inst.lines
    assert _line(SUPPORT_SOURCE, "CALLS = []") not in inst.lines  # module level: runs at load, not in a step
    module = types.ModuleType("m")
    module.__dict__[support.COND_NAME] = support.no_cond
    exec(inst.code, module.__dict__)
    tracer = support.Tracer(inst, module.__dict__)
    assert tracer.start()
    try:
        for key, args in (("0", (False, True, False)), ("1", (True, False, True)), ("2", (True, True, False))):
            tracer.begin()
            module.CALLS.clear()
            got = module.rule(*args)
            tracer.end(key)
            if key == "0":  # short-circuit kept: b is not evaluated once a is false
                assert module.CALLS == ["a"] and got == "none"
        tracer.begin()
        try:
            module.boom()
        except ValueError as exc:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            assert frame.lineno == _line(SUPPORT_SOURCE, 'raise ValueError("here")')  # line numbers unchanged
        tracer.end("3")
    finally:
        tracer.stop()
    assert module.rule(True, True, True) == "both" and module.__dict__[support.COND_NAME] is support.no_cond
    assert _line(SUPPORT_SOURCE, 'out = "none"') in tracer.executed["0"] and _line(SUPPORT_SOURCE, 'out = "both"') not in tracer.executed["0"]
    assert _line(SUPPORT_SOURCE, 'out = "either"') in tracer.executed["1"] and _line(SUPPORT_SOURCE, 'out = "both"') in tracer.executed["2"]
    a_and = next(c["id"] for c in inst.conds if c["text"] == 'probe("a", a)')
    b_and = next(c["id"] for c in inst.conds if c["text"] == 'probe("b", b)')
    assert [a_and, 0] in tracer.evaluated["0"] and not any(k == b_and for k, _ in tracer.evaluated["0"])
    assert [a_and, 1] in tracer.evaluated["1"] and [b_and, 0] in tracer.evaluated["1"]
    assert tracer.executed["3"] == [_line(SUPPORT_SOURCE, "x = 1"), _line(SUPPORT_SOURCE, 'raise ValueError("here")')]
    # A file the rewrite cannot read is reported as before: instrument gives None and the runner compiles it itself.
    assert support.instrument("def f(:\n", "/x/engine.py") is None


def test_the_support_map_counts_steps_and_finds_conditions_never_separated() -> None:
    import types

    from engine_re import support

    inst = support.instrument(SUPPORT_SOURCE, "/x/engine.py")
    module = types.ModuleType("m")
    exec(inst.code, module.__dict__)
    tracer = support.Tracer(inst, module.__dict__)
    tracer.start()
    calls = [(True, True, False), (True, True, True), (False, False, True), (True, False, False)]
    try:
        for k, args in enumerate(calls):
            tracer.begin()
            module.rule(*args)
            tracer.end(str(k))
    finally:
        tracer.stop()
    data = {"coverage": inst.static(), "executed": tracer.executed, "evaluated": tracer.evaluated}
    both = _line(SUPPORT_SOURCE, 'out = "both"')
    # Steps 0-2 only: `a and b` was never decided by b (b was true whenever a held); `a or c`, reached only when the
    # `and` failed, never by a.
    smap = support.fold_result(data, {"0": 10, "1": 11, "2": 12}, SUPPORT_SOURCE)
    assert smap["steps"] == 3 and smap["lines"][str(both)] == {"n": 2, "steps": [10, 11]}
    assert smap["lines"][str(_line(SUPPORT_SOURCE, 'out = "none"'))]["n"] == 0  # untested
    first = next(c for c in smap["compound"] if c["text"].startswith('probe("a", a) and'))
    assert not first["separated"] and first["missing"] == ['probe("b", b)']
    either = next(c for c in smap["compound"] if c["text"] == "a or c")
    assert not either["separated"] and either["missing"] == ["a"]
    assert smap["summary"]["untested"] >= 2 and smap["summary"]["unseparated"] >= 2 and smap["thin"] == support.THIN_SUPPORT
    # Step 3 (a held, b did not) separates `a and b`.
    smap = support.fold_result(data, {"0": 10, "1": 11, "2": 12, "3": 13}, SUPPORT_SOURCE)
    assert next(c for c in smap["compound"] if c["text"].startswith('probe("a", a) and'))["separated"]
    # Step 3's path against the first map: "either" ran on one earlier step (thin), and it relied on both loose
    # compounds (it evaluated b of the `and` and a of the `or`, the operands that never decided them).
    early = support.fold_result(data, {"0": 10, "1": 11, "2": 12}, SUPPORT_SOURCE)
    ps = support.path_support(early, tracer.executed["3"], tracer.evaluated["3"])
    assert ps["weakest"] == 1 and ps["untested"] == [] and _line(SUPPORT_SOURCE, 'out = "either"') in ps["weakest_lines"]
    assert [c["text"] for c in ps["unseparated"]] == ['probe("a", a) and probe("b", b)', "a or c"]
    assert "ran in only 1 step so far" in support.move_support_text(ps) and "never separated" in support.move_support_text(ps)
    assert support.mismatch_support_text(ps).startswith("the weakest lines")
    # A path through the untested branch.
    ps = support.path_support(early, tracer.executed["2"], tracer.evaluated["2"])
    assert support.path_support(early, [_line(SUPPORT_SOURCE, 'out = "none"')])["untested"] == [_line(SUPPORT_SOURCE, 'out = "none"')]
    none_path = support.path_support(early, tracer.executed["2"] + [_line(SUPPORT_SOURCE, 'out = "none"')])
    assert none_path["weakest"] == 0 and support.mismatch_support_text(none_path).startswith("this step was the first to run line")


def test_the_support_margin_follows_lines_through_an_edit() -> None:
    from engine_re import hashline, support

    text = SUPPORT_SOURCE
    smap = {"engine_sha": "old", "steps": 5, "keys": [support.text_key(t) for t in text.splitlines()],
            "lines": {str(_line(text, 'out = "both"')): {"n": 4, "steps": [1, 2, 3, 4]},
                      str(_line(text, 'out = "none"')): {"n": 0, "steps": []}}, "compound": []}
    # Two lines inserted above (the counted lines move), one counted line changed.
    edited = text.replace("def rule(a, b, c):\n", "def rule(a, b, c):\n    # a note\n    a = bool(a)\n").replace(
        'out = "none"', 'out = "nothing"')
    margin = support.margins(smap, edited)
    assert margin[_line(edited, 'out = "both"')] == "4" and margin[_line(edited, 'out = "nothing"')] == "new"
    assert margin[_line(edited, "a = bool(a)")] == "new" and _line(edited, "# a note") not in margin
    assert margin[1] == "·" and margin[2] == "·"  # the FIXED block
    listing = hashline.render_read(edited, margin=margin)
    assert listing.startswith(hashline.MARGIN_NOTE)
    shown = next(line for line in listing.splitlines() if 'out = "both"' in line)
    assert shown.startswith("   4| ") and hashline.parse_anchor(shown.split(":")[0]).line == _line(edited, 'out = "both"')
    with pytest.raises(hashline.EditError, match="E_INVALID_PATCH"):  # a listing line pasted as code is still refused
        hashline.apply_edits(edited, [{"op": "append", "lines": [shown]}])
    moved = support.remap(smap, edited)
    assert moved["lines"][str(_line(edited, 'out = "both"'))]["n"] == 4
    assert moved["lines"][str(_line(edited, 'out = "nothing"'))] == {"n": 0, "steps": [], "new": True}


def test_the_runner_records_executed_lines_per_step(tmp_path: Path) -> None:
    from engine_re.hashline import apply_edits
    from engine_re.skeleton import render_skeleton
    from engine_re.tester import run_candidate

    edits = [{"op": "replace_def", "name": n, "lines": p.strip("\n")} for n, p in zip(("LAYOUT", "make_level", "step"), RIGHT_ENGINE.split("\n\n\n"))]
    text = apply_edits(render_skeleton("twol", [1, 2, 3, 4]), edits).text
    text = text.replace("    if player.x >= 4:", "    if action.id == 2 and player.y > 1:\n        raise ValueError('deep')\n    if player.x >= 4:")
    engine = tmp_path / "engine.py"
    engine.write_text(text, encoding="utf-8")
    meta = {"win_levels": 2, "available_actions": [1, 2, 3, 4], "levels": [0]}
    actions = [{"id": 0}, {"id": 4}, {"id": 3}, {"id": 3}, {"id": 0}, {"id": 2}]  # RIGHT, LEFT, LEFT at x 1 (lost), RESET, DOWN
    result, _ = run_candidate(engine, actions, meta=meta, contract=False)
    executed = result["executed"]
    assert set(executed) == {"0", "1", "2", "3", "4", "5"} and result["error_step"] == 5
    assert _line(text, "state.try_move(player") in executed["1"] and _line(text, 'state.status = "game_over"') in executed["3"]
    assert _line(text, "state.try_move(player") not in executed["3"] and _line(text, "player = Sprite(") in executed["0"]
    assert executed["4"] == []  # a RESET of a level already built runs none of the model's code
    from engine_re.support import first_model_line

    assert all(line >= first_model_line(text) for lines in executed.values() for line in lines)
    assert f"line {_line(text, 'raise ValueError')}" in result["error"]  # the traceback's line is the file's
    assert any(g["text"] == "action.id == 3 and player.x == 1" for g in result["coverage"]["groups"])
    plain, _ = run_candidate(engine, actions, meta=meta, contract=False, trace=False)
    assert "executed" not in plain and plain["error"] == result["error"] and plain["steps"] == result["steps"]


def test_commit_moves_says_what_each_prediction_rests_on(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT", "RIGHT"], "note": "a"})],
        [("commit_moves", {"actions": ["DOWN", "UP", "RIGHT"], "note": "b"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6)
    result = agent.run()
    run = tmp_path / "run"
    engine = (run / "engine_committed.py").read_text()
    tools = _texts(agent, "tool")
    first = next(t for t in tools if t.startswith("Sent 3 of 4"))
    assert "Support of your replica's predictions (how many of the 1 recorded steps ran the code each move runs" in first
    head = _line(engine, "player = state.vars")
    # The moves are named as the Sent lines name them, by step and action (v11 follow-up 10): the first batch is steps 1-4.
    assert f"\n  #1 Action(4), #2 Action(4), #4 Action(4): first to run lines {head}-" in first  # step() had never run
    assert f"\n  #3 Action(4): first to run lines {head}-" in first and "(no step so far)" in first  # it solves the level too
    assert "Sent 3 of 4 move(s) (steps 1-3):\n  #1 Action(4): matches" in first and "move 1" not in first and "moves 1" not in first
    assert "\n  #4 Action(2), #5 Action(1), #6 Action(4): their paths are supported by at least 3 steps each" in tools[-1]
    assert "Sent 3 of 3 move(s) (steps 4-6):\n  #4 Action(2): matches" in tools[-1]
    support = result.batch_log[1]["support"]
    assert [s["weakest"] for s in support] == [3, 3, 3] and all(s["untested"] == [] for s in support)
    assert result.batch_log[0]["support"][0]["untested"] and result.batch_log[0]["support"][0]["weakest"] == 0
    # the sidecar: the committed engine's map, grown with the moves that matched
    import hashlib

    smap = json.loads((run / "engine_committed.support.json").read_text())
    assert smap["engine_sha"] == hashlib.sha256((run / "engine_committed.py").read_bytes()).hexdigest() == result.committed_sha
    assert smap["steps"] == 7 and smap["lines"][str(_line(engine, "state.try_move(player"))]["n"] == 6
    tests = [json.loads(line) for line in (run / "tests.jsonl").read_text().splitlines()]
    assert tests[-1]["support"]["steps"] == 4 and set(tests[-1]["support"]) == {"steps", "lines", "untested", "thin", "supported",
                                                                                "compound", "unseparated"}
    # the PLAN message names the outcome rule no step ran and the thin lines of the last batch's path
    plans = [u for u in _texts(agent) if u.startswith("Plan the next moves")]
    solved = _line(engine, 'state.status = "level_solved"')
    assert "Rules with little support" in plans[2]
    assert f"- your last batch's path: thin line {solved} (1 step), lines " in plans[2]  # and make_level(1)'s lines (2 steps)
    assert (f"- line {_line(engine, 'state.status = \"game_over\"')} sets game_over: no step ran it (an untested rule); its "
            "condition `action.id == 3 and player.x == 1`") in plans[-1]
    from engine_re.hashline import MARGIN_NOTE

    listing = plans[1].split(ENGINE_HEADER)[1]  # listed after the commit, with the support margin
    assert listing.lstrip().startswith(MARGIN_NOTE)
    assert next(line for line in listing.splitlines() if "state.try_move(player" in line).startswith("   0| ")


def test_a_mismatch_names_the_untested_lines_and_cut_untested_cuts_the_batch(tmp_path: Path, environments: Path) -> None:
    model = _ScriptedModel(_start(WRONG_DOWN) + [
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "a"})],  # cut after the first: it runs untested code
        [("commit_moves", {"actions": ["DOWN"], "note": "b"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5, cut_untested=True)
    result = agent.run()
    assert result.cut_untested and result.batch_log[0]["sent"] == 1 and result.batch_log[0]["cut_untested"] == 1
    assert result.batch_log[0]["moves"] == ["Action(4)", "Action(4)"] and result.actions == 2
    tools = _texts(agent, "tool")
    assert ("The batch was cut after move 1, the first to run code no recorded step has run: it is the experiment "
            "(the harness's cut-untested rule); the 1 move(s) after it were not sent.") in tools[2]
    fit = next(u for u in _texts(agent) if u.startswith("Fix your replica: step 2"))
    assert "Support: the weakest lines" in fit and "on its path had run in only 1 earlier step." in fit
    report = fit[fit.index("TEST RESULT"):]
    assert "its path in engine.py" in report and "thin: lines" in report and "(1 step)" in report


def test_the_kernel_margin_and_support_comments(tmp_path: Path, environments: Path) -> None:
    """read_file() in the play kernel shows the committed engine's support in the margin and, on the lines of step()
    and the functions it calls, as a trailing comment (the last five steps that ran the line, newest first); an edited
    line shows as new; edit_file drops a comment pasted back; traced() and support() are gone (free names now)."""
    from engine_re.hashline import MARGIN_NOTE
    from engine_re.kernel import KernelClient

    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "a"})]])
    agent = _agent(tmp_path, environments, model, turns=3)
    agent.run()
    run = tmp_path / "run"
    kernel = KernelClient(run / "workspace", run / "visible_trace", timeout=60, images=False, focus=2, history=True, play=True,
                          support=run / "engine_committed.support.json")
    try:
        out = kernel.execute("read_file()")
        assert out.startswith(MARGIN_NOTE) and "`# support (n): ...` comment" in MARGIN_NOTE
        moves = next(line for line in out.splitlines() if "state.try_move(player" in line)
        assert moves.startswith("   2| ") and moves.endswith("        state.try_move(player, *moves[action.id])  # support (2): 2, 1")
        solved = next(line for line in out.splitlines() if 'state.status = "level_solved"' in line)
        assert solved.startswith("   0| ") and solved.endswith("  # support (0): untested")
        assert not any("# support" in line for line in out.splitlines() if "def make_level" in line or "LAYOUT[n]" in line)
        # an edited line shows as new, in the margin and in its comment; the fresh anchors of the edit carry the comments too
        out = kernel.execute("edit_file(edits=[{'op': 'replace_text', 'oldText': 'if player.x >= 4:', 'newText': 'if player.x >= 5:'}])")
        assert "    if player.x >= 5:  # support: new" in out and "state.try_move(player, *moves[action.id])  # support (2): 2, 1" in out
        out = kernel.execute("read_file()")
        assert next(line for line in out.splitlines() if "player.x >= 5" in line).startswith(" new| ")
        assert next(line for line in out.splitlines() if "player.x >= 5" in line).endswith("  # support: new")
        assert next(line for line in out.splitlines() if "state.try_move(player" in line).startswith("   2| ")
        # a comment copied from the listing into an edit is dropped: it is the harness's, not the file's
        kernel.execute("edit_file(edits=[{'op': 'replace_text', 'oldText': 'if player.x >= 5:  # support: new', "
                       "'newText': 'if player.x >= 6:  # support (2): 2, 1'}])")
        text = (run / "workspace" / "engine.py").read_text()
        assert "if player.x >= 6:\n" in text and "# support" not in text
        for name in ("traced", "support"):
            assert "nothing was run" not in kernel.execute(f"{name} = 1"), name
    finally:
        kernel.stop()


def test_support_comments_on_the_step_code() -> None:
    """support.comments on a small engine with a synthetic map: step() and the helper it calls (transitively) get a
    comment, make_level and the level function do not; the comment's form; a changed line is new."""
    from engine_re import hashline
    from engine_re import support as sup

    source = """# ==== END OF FIXED INTERFACE ====
def level_0_sprites():
    return []


def make_level(n):
    return level_0_sprites()


def helper(state):
    state.vars["n"] = state.vars.get("n", 0) + 1
    return deeper(state)


def deeper(state):
    return state.vars["n"]


def step(state, action):
    helper(state)
    if action.id == 1:
        state.status = "level_solved"
"""
    lines = source.splitlines()
    assert sup.step_functions(source) == {"step": (19, 22), "helper": (10, 12), "deeper": (15, 16)}
    n = {text: next(i for i, line in enumerate(lines, 1) if line.strip() == text) for text in
         ("helper(state)", "if action.id == 1:", 'state.status = "level_solved"', 'state.vars["n"] = state.vars.get("n", 0) + 1',
          "return deeper(state)", 'return state.vars["n"]', "return level_0_sprites()", "return []")}
    table = {
        str(n["helper(state)"]): {"n": 31, "steps": [1, 2, 3, 4, 5, 36, 37, 39, 41, 42]},
        str(n["if action.id == 1:"]): {"n": 31, "steps": [1, 2, 3, 4, 5, 36, 37, 39, 41, 42]},
        str(n['state.status = "level_solved"']): {"n": 0, "steps": []},
        str(n['state.vars["n"] = state.vars.get("n", 0) + 1']): {"n": 3, "steps": [1, 2, 3]},
        str(n["return deeper(state)"]): {"n": 3, "steps": [1, 2, 3]},
        str(n['return state.vars["n"]']): {"n": 3, "steps": [1, 2, 3]},
        str(n["return level_0_sprites()"]): {"n": 2, "steps": [0, 7]},
        str(n["return []"]): {"n": 2, "steps": [0, 7]},
    }
    import hashlib

    smap = {"schema": 1, "engine_sha": hashlib.sha256(source.encode()).hexdigest(), "steps": 42, "thin": 3, "lines": table,
            "conds": {}, "compound": [], "keys": [sup.text_key(t) for t in lines]}
    comments = sup.comments(smap, source)
    assert comments == {
        n["helper(state)"]: "# support (31): 42, 41, 39, 37, 36 and 26 other",
        n["if action.id == 1:"]: "# support (31): 42, 41, 39, 37, 36 and 26 other",
        n['state.status = "level_solved"']: "# support (0): untested",
        n['state.vars["n"] = state.vars.get("n", 0) + 1']: "# support (3): 3, 2, 1",
        n["return deeper(state)"]: "# support (3): 3, 2, 1",
        n['return state.vars["n"]']: "# support (3): 3, 2, 1",
    }
    shown = hashline.render_read(source, margin=sup.margins(smap, source), comments=comments)
    assert ":    helper(state)  # support (31): 42, 41, 39, 37, 36 and 26 other" in shown
    assert "return level_0_sprites()\n" in shown and "return level_0_sprites()  #" not in shown
    # a changed line: new in the margin and in its comment; a moved one keeps its count
    edited = source.replace("    helper(state)\n", "    helper(state)\n    pass\n").replace("if action.id == 1:", "if action.id == 2:")
    comments = sup.comments(smap, edited)
    assert comments[n["if action.id == 1:"] + 1] == "# support: new" and comments[n["helper(state)"]].startswith("# support (31)")
    assert sup.STEPS_KEPT == 5 and sup._kept(list(range(20))) == [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]
    assert hashline.strip_support_comments("x = 1  # support (3): 3, 2, 1\ny = 2  # support: new\nz = 3  # mine") == "x = 1\ny = 2\nz = 3  # mine"
    assert sup.comments(None, source) == {} and sup.comments(smap, "def f():\n    pass\n") == {}


def test_the_play_prompt_explains_support_and_its_rules() -> None:
    from engine_re.prompts import system_prompt, tools

    text = system_prompt(mode="play", images=False)
    assert "# Support: how much of your replica the steps played have tested" in text
    assert "0 is untested" in text and "fewer than 3 is thin" in text and "Trust these counts over comments of your own in engine.py." in text
    assert "Important rules should be tested in different conditions a few times." in text
    assert "A rule written as `A and B` or `A or B` is only established once the steps separate its parts" in text
    assert "Do not over-engineer your replica on one observation: one step supports one rule, not a general mechanism" in text
    assert "a rule is worth generalising when its support\n   comes from steps in different conditions, not before." in text
    assert ("on every line of step() and of the functions it calls, in a trailing comment the harness adds to the listing:\n"
            "`# support (31): 42, 41, 39, 37, 36 and 26 other` is how many steps ran the line and the last five of them, newest\n"
            "first; `# support (0): untested` a line no step ran; `# support: new` a line changed since the map was made.") in text
    assert "These comments are not in the file (edit_file drops them from anything you paste)." in text
    assert "traced" not in text and "support(run" not in text and "TracedRun" not in text
    commit_moves = tools(False, "play", True)[3]["function"]["description"]
    assert "what each move's prediction rested on" in commit_moves
    python = tools(False, "play", True)[0]["function"]["description"]
    assert ("A cell has 120 s: a longer one is killed and the kernel\nrestarts without your variables (the harness then re-runs "
            "your earlier cells, not the one that timed out),\nso bound searches by time (time.time()) and keep the best result "
            "so far in a variable you print.") in python
    assert "traced" not in python and "support" not in python
# --- ported from the base harness's prompt ---------------------------------------------------------

# Two levels of 8x8 as twol; SPACE flashes the wall (5 -> 14 -> 5: an animation of two frames) and moves the player down.
FLASH_GAME = '''
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

MOVES = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1), GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}


class Flash(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.flashing = False
        levels = [
            Level(sprites=[Sprite([[9]], name="player", x=1, y=1), Sprite([[5] * 8], name="wall", x=0, y=0)], grid_size=(8, 8)),
            Level(sprites=[Sprite([[9]], name="player", x=1, y=3), Sprite([[8] * 8], name="wall", x=0, y=7)], grid_size=(8, 8)),
        ]
        super().__init__(game_id="flsh", levels=levels, camera=Camera(0, 0, 8, 8, 0, 3), available_actions=[1, 2, 3, 4, 5])

    def step(self) -> None:
        wall = self.current_level.get_sprites_by_name("wall")[0]
        if self.action.id == GameAction.ACTION5:
            if not self.flashing:
                self.flashing = True
                wall.color_remap(None, 14)
                return
            self.flashing = False
            wall.color_remap(None, 5)
            self.try_move("player", 0, 1)
            self.complete_action()
            return
        player = self.current_level.get_sprites_by_name("player")[0]
        dx, dy = MOVES.get(self.action.id, (0, 0))
        if dx or dy:
            self.try_move("player", dx, dy)
        if player.x >= 4:
            self.next_level()
        self.complete_action()
'''


@pytest.fixture()
def flash_env(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    _write_game(root, "flsh", FLASH_GAME, [3, 3])
    return root


def _flash_frames() -> tuple:
    """A synthetic animated step: row 10 flashes 1 -> 14 -> 1 over three frames, and the pixel at (5, 3) turns 9 in the
    second frame and stays."""
    import numpy as np

    before = np.zeros((64, 64), np.int8)
    before[10, :] = 1
    flash = before.copy()
    flash[10, :] = 14
    settled = before.copy()
    settled[3, 5] = 9
    return before, np.stack([flash, settled, settled])


def test_the_animation_digest_of_a_flash() -> None:
    import numpy as np

    from engine_re import animation

    before, frames = _flash_frames()
    found = animation.digest(before, frames)
    assert found.frames == 3 and found.transient == 64 and found.transient_bbox == (0, 10, 63, 10)
    assert found.transient_transitions == {"1>14": 64} and found.transient_frames == (0, 0)
    assert [(e.frame, e.changed, e.bbox) for e in found.timeline] == [(0, 64, (0, 10, 63, 10)), (1, 65, (0, 3, 63, 10))]
    assert found.timeline[0].transitions == {"1>14": 64} and found.timeline[0].cells is None  # too many cells to list
    assert str(found.timeline[1]) == "frame 1: 65 cells in x 0-63, y 3-10: 14>1 x64, 0>9 x1"
    one = before.copy()
    one[3, 5] = 9
    few = animation.digest(before, np.stack([one, before]))  # one pixel blinks: its cells are listed, as (x,y)
    assert few.timeline[0].cells == ["0>9 @ (5,3)"] and few.timeline[1].cells == ["9>0 @ (5,3)"]
    assert few.transient == 1 and few.transient_bbox == (5, 3, 5, 3) and "1 cell changed and changed back at (5, 3)" in str(few)
    assert animation.digest(before, frames[:1]) is None and animation.digest(None, frames).transient == 0
    lines = animation.report_lines(before, frames, 7)
    assert lines == [
        "    Animation: transient cells: 64 cells changed and changed back in x 0-63, y 10, in frame 0: 1>14 x64; they are in "
        "no frame you can otherwise reach (not in .before, not in .after).",
        "    Animation timeline (each frame against the one before it, frame 0 against .before): frame 0: 64 cells in x 0-63, "
        "y 10: 1>14 x64; frame 1: 65 cells in x 0-63, y 3-10: 14>1 x64, 0>9 x1.",
    ]
    # a timeline too long for a report is a pointer
    noisy = np.stack([np.full((64, 64), k % 16, np.int8) if k % 2 else before for k in range(12)])
    long = animation.report_lines(before, noisy, 7)
    assert len(long) == 2 and long[1] == "    Animation timeline: 11 frames changed something; recording[7].animation.timeline lists them."


def test_the_animation_digest_in_step_views_reports_and_messages() -> None:
    from engine_re import helpers, tester
    from engine_re.prompts import mismatch_message
    from engine_re.trace import Step, Trace

    before, frames = _flash_frames()
    trace = Trace("flash", [Step(0, Action(0), before[None], "NOT_FINISHED", 0, 1, [1, 2, 3, 4]),
                            Step(1, Action(1), frames, "NOT_FINISHED", 0, 1, [1, 2, 3, 4])])
    helpers.load_trace(trace)
    assert helpers.recording[0].animation is None and helpers.recording[1].animation.transient == 64
    assert "Animation over 3 frames. Transient: 64 cells changed and changed back" in str(helpers.recording[1].animation)
    got = {"state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 1, "available_actions": [1, 2, 3, 4]}
    text, _ = tester.describe_step(trace[1], got, before[None], 0, before_frame=before)
    assert "(the original animated this action over 3 frames; only the last is compared)" in text
    assert "    Animation: transient cells: 64 cells changed and changed back in x 0-63, y 10, in frame 0: 1>14 x64" in text
    assert "Animation" not in tester.describe_step(trace[0], got, before[None], 0)[0]
    # the message says it after the frame count, unless the report it carries has it already
    message = mismatch_message(trace, 1, "the final frame differs", 0, "TEST RESULT (stub)")
    assert ("the game returned 3 frame(s), the tests compare the last.\nAnimation: transient cells: 64 cells changed and changed "
            "back") in message
    assert mismatch_message(trace, 1, "the final frame differs", 0, text).count("Animation: transient cells") == 1


def test_an_animated_step_in_the_play_loop(tmp_path: Path, flash_env: Path) -> None:
    """SPACE animates (the wall flashes) and moves the player: the fit message carries the digest once; when the replica
    models the move, the next SPACE matches and commit_moves' output names its transient cells."""
    space = RIGHT_ENGINE.replace("moves = {1: (0, -1)", "moves = {5: (0, 1), 1: (0, -1)")
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["SPACE"], "note": "probe SPACE"})],
        [_install(space)],
        [("commit_engine", {"message": "SPACE moves the player down"})],
        [("commit_moves", {"actions": ["SPACE"], "note": "again"})],
    ])
    agent = _agent(tmp_path, flash_env, model, turns=6, game="flsh")
    agent.run()
    fit = next(u for u in _texts(agent) if u.startswith("Fix your replica: step 1"))
    assert "the game returned 2 frame(s), the tests compare the last." in fit
    assert fit.count("Animation: transient cells: 512 cells changed and changed back in x 0-63, y 0-7, in frame 0: 5>14 x512") == 1
    out = _texts(agent, "tool")[-1]
    assert "#2 Action(5): matches your prediction" in out
    assert "[harness] Step 2 (Action(5)) animated over 2 frames: 512 cells changed and changed back" in out


def test_a_batch_stops_before_a_predicted_board_noop_and_warns_of_a_predicted_game_over(tmp_path: Path, environments: Path) -> None:
    """twol: UP at y=1 is blocked by the wall (nothing changes on the board) and LEFT at x=1 loses. A batch of two or
    more is cut before its first predicted board no-op (nothing sent when that is move 1: a refusal); a single move
    goes as a probe; a predicted game over is a warning, never a refusal, and the warnings are in the batch's log entry."""
    model = _ScriptedModel(_start() + [
        [("commit_moves", {"actions": ["RIGHT", "UP", "LEFT", "RIGHT"], "note": "cut before UP"})],
        [("commit_moves", {"actions": ["UP", "RIGHT"], "note": "refused: UP first"})],
        [("commit_moves", {"actions": ["UP"], "note": "a probe of one goes"})],
        [("commit_moves", {"actions": ["LEFT", "LEFT", "RIGHT"], "note": "the second LEFT loses"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=6)
    result = agent.run()
    tools = _texts(agent, "tool")
    cut = ("[harness] Moves 2-4 not sent: your replica predicts move 2 (Action(1)) changes nothing on the board; a probe of that "
           "rule goes as a batch of one.")
    assert tools[2].startswith("Sent 1 of 4 move(s) (steps 1-1):\n  #1 Action(4): matches your prediction\nActions played: 1 of at most 500")
    assert cut in tools[2] and "Every move matched your replica." in tools[2] and result.batch_log[0]["warnings"] == [cut]
    assert "What the batch changed on the board (steps 0 -> 1; x, y in the screen grid, recording[k].changes has each step):\n" \
           "  1 moved by (+1, +0) (SHAPE_9_1x1" in tools[2]
    assert tools[3] == ("Not sent: your replica predicts move 1 (Action(1)) changes nothing on the board, so the batch was cut "
                        "before it and nothing was sent; a probe of that rule goes as a batch of one.")
    assert tools[4].startswith("Sent 1 of 1 move(s) (steps 2-2):\n  #2 Action(1): matches your prediction") and "[harness]" not in tools[4]
    assert "What the batch changed" not in tools[4]  # the frames are equal
    death = ("[harness] Warning: your replica predicts a game over at move 2 (Action(3)); the 1 move(s) after it would not be "
             "sent, and the harness then RESETs the level (one more action). Sent anyway: a deliberate probe is fine.")
    assert tools[5].startswith("Sent 2 of 3 move(s) (steps 3-4):") and death in tools[5]
    assert result.batch_log[2]["warnings"] == [death] and result.refused_batches == 1 and result.batches == 3
    assert "warnings" not in result.batch_log[3] and len(result.batch_log) == 4  # the automatic RESET; the refusal is not a batch
    assert [s.action.id for s in agent.live.trace.steps] == [0, 4, 1, 3, 3, 0]
    users = _texts(agent)
    assert "Your last batch: 1 move(s) sent, all as your replica predicted (3 move(s) not sent)." in users[2]
    assert not any("changes nothing in your replica" in t for t in tools)


def test_the_board_noop_rule_on_a_synthetic_prediction() -> None:
    """A move is a board no-op when its predicted frame equals the previous one outside the screen-layer sprites' boxes
    (a whole-screen border left in; a HUD_BORDER ring without a summary) with the status and levels unchanged."""
    import numpy as np

    from engine_re.tester import HUD_BORDER

    acts = [Action(4), Action(1), Action(2)]  # at positions 1, 2, 3 after one recorded step
    f0 = np.zeros((64, 64), np.int8)
    f1 = f0.copy()
    f1[0, :] = 7  # position 1: the HUD strip along the top changed, nothing else
    f2 = f1.copy()
    f2[30, 30] = 9  # position 2: the board changed
    f3 = f2.copy()
    f3[63, 5] = 4  # position 3: a pixel inside the HUD_BORDER ring only
    steps = [{"state": "NOT_FINISHED", "levels_completed": 0}] * 4
    hud = {"name": "hud", "screen": True, "box": [0, 0, 0, 63]}
    border = {"name": "border", "screen": True, "box": [0, 0, 63, 63]}  # the whole screen: not a HUD, left in the board
    with_hud = {"sprites": [border, hud]}
    frames = [np.stack([f0]), np.stack([f1]), np.stack([f2]), np.stack([f3])]
    prediction = {"steps": steps, "inspect": {"1": {"before": with_hud, "after": with_hud}, "2": {"before": with_hud, "after": with_hud}}}
    first = PlayAgent._first_board_noop
    assert first(acts, prediction, frames, 1) == 0  # move 1 changes only the HUD
    assert HUD_BORDER == 2 and first(acts[1:], prediction, frames, 2) == 1  # move 2 changes the board; move 3 only the ring (no summary)
    frames[3][0][5, 5] = 4  # now move 3 changes the board too
    assert first(acts[1:], prediction, frames, 2) is None
    assert first(acts[:1], prediction, frames, 1) is None  # a single move is a probe, sent regardless
    solved = {"steps": [steps[0], {"state": "NOT_FINISHED", "levels_completed": 1}] + steps[2:], "inspect": prediction["inspect"]}
    assert first(acts, solved, frames, 1) is None  # a level solved is never a no-op, whatever the frame
    assert first(acts, {"steps": steps[:1]}, frames, 1) is None  # no prediction for the move: the raise warning's case
    assert not PlayAgent._board_mask([with_hud])[0, 10] and PlayAgent._board_mask([with_hud])[1, 10]
    assert not PlayAgent._board_mask([None])[1, 10] and PlayAgent._board_mask([None])[2, 10]


def test_the_raise_warning_and_the_verdict_on_levels(tmp_path: Path, environments: Path) -> None:
    """Before a batch is sent, a replica that raises at some move is said so (the batch goes as it is); a replica
    that solves the level when the game does not, or the converse, gets a verdict that says so, in the batch lines
    and the FIT message, instead of "raised an error" when make_level(n + 1) is what raised."""
    early = RIGHT_ENGINE.replace("if player.x >= 4:", "if player.x >= 3:").replace("LAYOUT = {0: ((1, 1), 5, 0), 1: ((1, 3), 8, 7)}",
                                                                                   "LAYOUT = {0: ((1, 1), 5, 0)}")
    model = _ScriptedModel(_start(early) + [[("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "solve early"})]])
    agent = _agent(tmp_path, environments, model, turns=3)
    result = agent.run()
    out = _texts(agent, "tool")[-1]
    assert ("[harness] Warning: your replica raises at move 2 (Action(4)): KeyError: 1; the batch is sent as it is and a mismatch "
            "there opens a fit round.") in out
    verdict = ("your replica predicts level 0 solved; the game did not (levels completed: the game says 0, your replica 1); your "
               "replica then raised an error starting level 1 (KeyError: 1)")
    assert f"#2 Action(4): differs from your prediction: {verdict}" in out and result.batch_log[0]["diff"] == verdict
    fit = _texts(agent)[-1]
    assert fit.startswith("Fix your replica: step 2 did not go as your replica predicted.") and f"What differed: {verdict}." in fit
    late = RIGHT_ENGINE.replace("if player.x >= 4:", "if player.x >= 5:")
    model = _ScriptedModel(_start(late) + [[("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "the game solves first"})]])
    agent = _agent(tmp_path / "late", environments, model, turns=3)
    agent.run()
    out = _texts(agent, "tool")[-1]
    assert ("#3 Action(4): differs from your prediction: the final frame differs; the game solved level 0; your replica did not "
            "(levels completed: the game says 1, your replica 0)") in out
    # The level-opening FIT message names the engine.py constants the new board's pieces match (v11 follow-up 1): the
    # player's 1x1 blue constant of level 0 (as it is); the red wall matches none (a solid bar is never recoloured).
    from engine_re.auto_sprites import shape_name

    fit = " ".join(_texts(agent)[-1].split())
    assert "It solves level 0: the frame after it is level 1's first frame, which make_level(1) must draw" in fit
    assert (f"(recording[-1]). The new board's pieces match these engine.py constants: {shape_name(('9',))}; 1 piece matches none "
            "(8x1 red at (0, 7)). Combine the constants into the new level's sprites and draw the rest from "
            "recording[-1].pieces_after.code(). What the recorded step changed (objects): it enters level 1.") in fit


def test_the_play_system_prompt_has_the_base_harness_guidance() -> None:
    from engine_re.prompts import system_prompt

    for images in (True, False):
        text = system_prompt(mode="play", images=images)
        setup = text[text.index("# Setup"):text.index("# Drawing")]
        assert ("- Colour legend: 0 white, 1 light grey, 2 grey, 3 dark grey, 4 darker grey, 5 black, 6 magenta, 7 pink, 8 red,\n"
                "  9 blue, 10 light blue, 11 yellow, 12 orange, 13 maroon, 14 green, 15 purple.") in setup
        flat = " ".join(setup.split())
        for part in ("UP, DOWN, LEFT and RIGHT (Action(1) to Action(4)) are directional controls; what they affect depends on "
                     "the game.",
                     "When available, SPACE (Action(5)) performs a game-specific action, such as interacting, selecting, "
                     "rotating, attaching/detaching, or executing. Test its effect rather than assuming what it does.",
                     "a click Action(6, x=x, y=y) clicks a board location. Pass integer x and y from 0 to 63. Coordinates are "
                     "zero-based from the top-left: y (the row) increases downward and x (the column) increases rightward.",
                     "When available, UNDO (Action(7)) reverses a previous action, usually the last turn. Check what it "
                     "restores. It cannot recover a failed attempt after game over.",
                     "RESET (Action(0)) usually restores the current level to its starting state, including the "
                     "remaining-action/time bar, while keeping completed levels. Use it to recover from an unrecoverable "
                     "position or start a different approach. RESET itself counts as one action, and actions already spent "
                     "still count toward your score."):
            assert part in flat, part
        assert "row` and `col`" not in text and "MOUSE" not in text and "first game action in a Python snippet" not in text
        tests = " ".join(text[text.index("# Tests"):text.index("# Objects")].split())
        assert ("Transient cells changed and then changed BACK during the animation, so they appear in no frame you can "
                "otherwise reach - not in .before, not in .after.") in tests
        plan = " ".join(text[text.index("Plan rounds:"):text.index("Fit rounds:")].split())
        for part in (
            "If your search finds no solution under your current model of the game, remember that the game is solvable. "
            "Reconsider your mechanics, goal, search implementation, or search limits, including interactions with new "
            "elements. Take a targeted action to test an uncertain rule or overlooked interaction, then update your model "
            "from the result.",
            "A plan far above the human baseline the plan message shows, or no plan at all, means your replica is missing a "
            "rule, not that the level is hard.",
            "Levels usually build on mechanics learned in earlier levels, especially the most recent one.",
            "they are your starting hypothesis on a new level, while you re-check anything contradicted by new evidence.",
            "New levels often introduce additional mechanics, sometimes through an unfamiliar board element or a visual "
            "change. These additions are often important for solving the level. The goal may remain the same but require "
            "new mechanics to reach it, or the goal itself may change.",
            "Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.",
            # the v12 rules: stuck means a missing or wrong rule; plan from the end state; colour is part of a comparison
            "6. Every game is solvable. When you are stuck (no plan, a plan far above the human baseline, or the same refusal "
            "twice), you are probably missing a rule or one of your written rules is wrong: explore more of the game's "
            "mechanics (touch what you have not touched, repeat a refused move under a changed condition) rather than "
            "searching harder on the rules you have.",
            "7. When the goal is a configuration of the board, enumerate the winning end states from your rules first (often "
            "a few lines over the replica's State), then plan the route to the nearest one; search over moves only when the "
            "end state is unknown.",
            "A comparison the game makes (a piece against a legend, a key against a lock, a pattern against a target) may "
            "involve colour as well as shape and rotation; \"recoloured\" in an object diff is a change in its own right, not "
            "a rotation.",
            "Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do "
            "not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board "
            "configuration.",
            "Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a "
            "specific absolute row or column.",
            "Reading and computing cost nothing; only commit_moves spends the level budget. When you are unsure, prefer "
            "another python call over more reasoning: the code answers what the reasoning would only guess at",
            "Goal model: what winning requires. Open questions: unresolved hypotheses. Plan: intended next steps.",
            'edit_file(path="notes.md", edits=[...])',
            "Older parts of this conversation will eventually be dropped, so anything you leave out of engine.py and "
            "notes.md is gone.",
        ):
            assert part in plan, part
        assert ".animation: Animation | None" in text  # the Objects reference
    assert "Colour legend" not in system_prompt(mode="single") and "notes.md" not in system_prompt(mode="step")
    assert ".animation: Animation | None" in system_prompt(mode="single")


def test_notes_md_round_trips_through_the_kernel_and_shows_in_the_plan_message(tmp_path: Path, environments: Path) -> None:
    """notes.md starts empty (v11 follow-up 11: the seeded headings were duplicated by the model's own); the model writes
    its headings with edit_file, here called as a tool with its edits as a JSON string of one dict (the shim of follow-up
    6 parses and wraps it, and says so), and the PLAN message shows the file as it is, each heading once."""
    first = ("edit_file", {"path": "notes.md", "edits": '{"op": "append", "lines": ["Goal model: reach x >= 4", "Open questions:", "Plan: RIGHT x3"]}'})
    many = "edit_file(path='notes.md', edits=[{'op': 'append', 'lines': [f'line {i}' for i in range(50)]}])"
    model = _ScriptedModel(_start() + [
        [("python", {"code": notes}), ("commit_moves", {"actions": ["RIGHT"], "note": "one"})],
        [("python", {"code": many}), ("commit_moves", {"actions": ["RIGHT"], "note": "two"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=4)
    from engine_re.prompts import NOTES_TEMPLATE

    assert NOTES_TEMPLATE == ""  # seeded empty, no headings
    agent.run()
    users = _texts(agent)
    assert "notes.md holds nothing yet" in users[0] and 'edits=[{"op": "append", "lines": [...]}]' in users[0] and "Goal model" not in users[0]
    tools = _texts(agent, "tool")
    assert tools[2].startswith("[harness] edit_file is a python function, not a tool; this call ran as python: edit_file(path='notes.md', "
                               "edits=[{'op': 'append', 'lines': ['Goal model: reach x >= 4', 'Open questions:', 'Plan: RIGHT x3']}])\n"
                               "[harness] edits given as a JSON string: parsed\n[harness] edits given as one dict: wrapped in a list\n")
    assert "notes.md: " in tools[2] and "Goal model: reach x >= 4" in tools[3]
    plan = users[2]
    assert "notes.md (your goal model, open questions and plan):\n  Goal model: reach x >= 4\n  Open questions:\n  Plan: RIGHT x3" in plan
    assert plan.count("Goal model:") == 1 and plan.count("Open questions:") == 1 and plan.count("Plan:") == 1  # no duplicated headings
    assert plan.index("notes.md (your goal") < plan.index("Work out the next moves")
    text = (tmp_path / "run" / "workspace" / "notes.md").read_text()
    assert text.startswith("Goal model: reach x >= 4\nOpen questions:\nPlan: RIGHT x3\n") and len(text.splitlines()) == 53
    last = users[3]
    assert '  line 36\n  [cut: 13 more line(s) of notes.md not shown; read_file("notes.md") shows them. Keep it short.]' in last
    # edit_file on notes.md writes the file itself: engine.py's versions are untouched
    assert not any("reach x >= 4" in v.read_text() for v in (tmp_path / "run" / "engine_versions").glob("*.py"))


def test_the_plan_message_after_a_solved_level(tmp_path: Path, environments: Path) -> None:
    """The level-start paragraph (the base harness's LEVEL_START_USER_PROMPT, without its list of unfamiliar pieces) and,
    under the frame, the replica's sprites of the new level (level 1's wall, red and at the bottom)."""
    model = _ScriptedModel(_start() + [[("commit_moves", {"actions": ["RIGHT"] * 3, "note": "solve level 0"})]])
    agent = _agent(tmp_path, environments, model, turns=3)
    agent.run()
    users = _texts(agent)
    plan = users[-1]
    assert plan.startswith("Plan the next moves. Steps 0-3 pass")
    assert ("You have completed the previous level. recording[-1].after now contains the starting board of the next level; "
            "show_frames(recording[-1].after) shows this new board.\nBuild a new plan for this layout rather than continuing the "
            "previous level's action sequence.") in plan
    assert "Inspect the new board for an unfamiliar element, a visual change (a colour, a frame, a legend), a changed" in plan
    assert "Reassess the goal: does the previous objective still apply, now requiring the new mechanics" in plan
    assert "Unfamiliar elements" not in plan and "You have completed the previous level" not in users[1]
    # Under it, the level-start nudge (v11 follow-up 1): which engine.py constants the new board's pieces match (the
    # opening's 1x1 blue player constant, as it is) and which match none (the red wall: a solid bar is never recoloured).
    from engine_re.auto_sprites import shape_name
    from engine_re.prompts import KINDS_HEAD

    flat = " ".join(plan.split())
    assert (f"Combine retained knowledge with new findings to plan for this board. {KINDS_HEAD} {shape_name(('9',))}; 1 piece "
            "matches none (8x1 red at (0, 7)). Combine the constants into the new level's sprites and draw the rest from "
            "recording[-1].pieces_after.code(). ") in flat, flat
    assert plan.count(KINDS_HEAD) == 1 and KINDS_HEAD not in users[1]
    assert plan.index("You have completed the previous level") < plan.index(KINDS_HEAD) < plan.index("Work out the next moves")
    sprites = plan[plan.index("Your replica's sprites now"):]
    assert sprites == ("Your replica's sprites now (state_now().sprites; x, y in the engine's grid), vars={'player': #2}:\n"
                       '  [0] "border" tags=[] 64x64 colour 3 (dark grey) at (0, 0) layer -2 inert screen\n'
                       '  [1] "background" tags=[] 8x8 colour 0 (white) at (0, 0) layer -1 inert\n'
                       '  [2] "player" tags=[\'player\'] 1x1 colour 9 (blue) at (1, 3) layer 0\n'
                       '  [3] "wall" tags=[\'wall\'] 8x1 colour 8 (red) at (0, 7) layer 0')
    assert plan.index("Work out the next moves") < plan.index("Your replica's sprites now")  # under the frame, after the text


# --- v12: the kernel restart replay, the sprite list and the reconciliation on the v11 sp80 fixture ----------------

SP80 = Path(__file__).with_name("fixtures") / "sp80_v11"  # the v11 run's committed engine, support map and trace (steps 0-122)


def test_a_timed_out_cell_restarts_the_kernel_and_the_earlier_cells_are_replayed(tmp_path: Path, environments: Path) -> None:
    """A cell past the kernel's time is killed; the message names what was lost (the names the kernel held), the cell
    (its first line), that the built-ins are back, which cells were re-run and what the kernel keeps now; the cell
    that timed out is not re-run, now or when the run resumes."""
    from engine_re.prompts import KERNEL_KEEPS

    model = _ScriptedModel(_start() + [
        [("python", {"code": "def helper(s):\n    return s.level\nDATA = [1, 2]"})],
        [("python", {"code": "import time\ntime.sleep(4)\nlate = 1"})],
        [("python", {"code": "print(helper(state_now()), DATA)"})],
    ])
    agent = _agent(tmp_path, environments, model, turns=5)
    agent.kernel.timeout = 1.5
    result = agent.run()
    tools = _texts(agent, "tool")
    out = tools[3]
    assert out.startswith("Timed out after 1.5s. The kernel was restarted and all variables were lost.\n\n[harness] This cell ran "
                          "longer than 1.5 s, so it was killed and the kernel was restarted. Lost: helper: function, DATA: list[2] "
                          "(2 names). The cell that timed out (starting `import time`) was not re-run: bound searches by time "
                          "(time.time()) and keep the best result so far in a variable you print. The harness built-ins and "
                          "`recording` are back. Your earlier 2 python cells were re-run in order with file edits disabled, so what "
                          "they defined is back.\n" + KERNEL_KEEPS + "helper: function, DATA: list[2]")
    assert tools[4].strip().endswith("0 [1, 2]") and result.kernel_restarts == 1
    records = _records(tmp_path)
    restart = next(r["kernel_restart"] for r in records if "kernel_restart" in r)
    assert restart["reason"] == "timeout" and restart["lost"] == ["helper: function", "DATA: list[2]"] and restart["replayed"] == 2
    assert [c["turn"] for c in agent.cells] == [1, 3, 5]  # the install cell, the helper cell, the print; not the sleep
    # a resumed run re-runs the same cells: the one that timed out is left out of them
    second = _ScriptedModel([[("python", {"code": "print(helper(state_now()), DATA)"})]])
    agent2 = _agent(tmp_path, environments, second, turns=6)
    agent2.run()
    assert [c["turn"] for c in agent2.cells] == [1, 3, 5, 6] and _texts(agent2, "tool")[-1].strip().endswith("0 [1, 2]")
    assert any("re-ran your 3 python cells" in u for u in _texts(agent2))


def test_the_plan_messages_sprite_list_on_the_sp80_engine() -> None:
    """The replica's sprite list (24) from the v11 sp80 committed engine after steps 0-121 (level 3, in step): one line per
    sprite in the engine's grid with its flags, vars first; out of step, the segmentation's list of the game's frame."""
    from engine_re.prompts import PIECES_HEAD, pieces_list_text, sprite_list_text
    from engine_re.tester import replica_state
    from engine_re.trace import Trace

    trace = Trace.load(SP80 / "trace")
    sub = Trace(trace.game_id, trace.steps[:122], trace.meta)  # steps 0-121: the last the committed engine reproduces
    summary = replica_state(SP80 / "engine_committed.py", sub)
    text = sprite_list_text(summary)
    assert text.splitlines()[:3] == [
        "Your replica's sprites now (state_now().sprites; x, y in the engine's grid), vars={'budget': 120, 'moves': 70}:",
        '  [0] "border" tags=[] 64x64 colour 1 (light grey) at (0, 0) layer -2 inert screen',
        '  [1] "background" tags=[] 20x20 colour 12 (orange) at (0, 0) layer -1 inert',
    ]
    assert "  [8] \"\" tags=['player'] 4x1 colour 9 (blue) at (0, 10) layer 0" in text
    assert "  [9] \"\" tags=['bin'] 3x2 colour 11 (yellow) at (2, 17) layer 0" in text
    assert text.splitlines()[-1] == '  [14] "hud" tags=[\'hud\'] 64x1 multi (0 white, 14 green) at (0, 0) layer 1 inert screen'
    assert len(text.splitlines()) == 16
    flags = sprite_list_text({"vars": {}, "sprites": [{"name": "k", "tags": ["a", "b"], "w": 2, "h": 3, "x": 4, "y": 5, "layer": 1,
                                                       "rotation": 90, "mirror_ud": True, "mirror_lr": True, "scale": 2, "visible": False,
                                                       "collidable": False, "screen": True, "colours": {"12": 3, "9": 2, "5": 1}}]})
    assert flags.splitlines()[1] == ("  [0] \"k\" tags=['a', 'b'] 2x3 multi (12 orange, 9 blue) at (4, 5) layer 1 rot=90 mirror_ud "
                                     "mirror_lr scale=2 hidden inert screen")
    from engine_re.diff_report import colour_text

    assert colour_text({"12": 6, "9": 4}) == "colour 12 (orange)" and colour_text({"12": 5, "9": 4, "5": 1}) == "multi (12 orange, 9 blue)"
    assert colour_text({}) == "" and colour_text(None) == ""
    pieces = pieces_list_text(sub)
    assert pieces.startswith(PIECES_HEAD + "\n  ") and "pieces on a 20x20 grid at scale 3, offset (2, 2)" in pieces.splitlines()[1]


def test_the_level_start_nudge_names_the_engine_constants_the_new_board_matches() -> None:
    """v11 follow-up 1 on the sp80 fixture: engine.py's pixel constants read from the file (auto_sprites.pixel_constants,
    no code run) and, at a level start, which of them the new board's pieces match (prompts.level_kinds_text, the matcher
    of pieces_after.code(): as they are, turned, scaled, recoloured), and the pieces that match none."""
    from engine_re.auto_sprites import pixel_constants
    from engine_re.prompts import KINDS_HEAD, entered_level, level_kinds_text, level_start_text
    from engine_re.trace import Trace

    constants = pixel_constants((SP80 / "engine_committed.py").read_text())
    assert set(constants) == {"CAP", "SOURCE", "BAR", "BIN", "BLOCK"} and constants["BIN"] == ["b.b", "bbb"] and constants["BAR"] == ["99999"]
    source = 'A = ("9",)\nB = ("55",) * 2\nC = [[1, -1], [1, 1]]\n_D = ("9",)\nE = ("9", "99")\nF = 3\nG = [str(i) for i in range(2)]\n'
    assert pixel_constants(source) == {"A": ["9"], "B": ["55", "55"], "C": [[1, -1], [1, 1]]}  # not _D, a ragged E, F, a comprehension
    assert pixel_constants("def f(:\n") == {}
    trace = Trace.load(SP80 / "trace")
    starts = trace.level_starts()
    assert starts[1] == 11 and starts[4] == 122
    sub = Trace(trace.game_id, trace.steps[:12], trace.meta)  # steps 0-11: step 11 enters level 1
    assert entered_level(sub) and not entered_level(Trace(trace.game_id, trace.steps[:11], trace.meta))
    text = level_kinds_text(sub, constants)
    flat = " ".join(text.split())
    assert flat == (f"{KINDS_HEAD} BIN x3 (three turned 180), BLOCK x2, BAR, SOURCE, CAP; 2 pieces match none (16x1 light grey at "
                    "(0, 0), 64x1 green screen piece at (0, 63)). Combine the constants into the new level's sprites and draw the rest "
                    "from recording[-1].pieces_after.code().")
    assert len(text.splitlines()) <= 8 and all(len(line) <= 118 for line in text.splitlines())
    assert level_start_text(sub, images=False, kinds=text).endswith("to plan for this board.\n\n" + text)
    last = Trace(trace.game_id, trace.steps[:123], trace.meta)  # step 122 enters level 4
    flat = " ".join(level_kinds_text(last, constants).split())
    assert flat.startswith(f"{KINDS_HEAD} BIN x4 (three turned 180, one turned 90), BLOCK, BAR, SOURCE x2, CAP x2; 4 pieces match none (")
    assert flat.endswith("2x2 purple at (10, 13), ...). Combine the constants into the new level's sprites and draw the rest from "
                         "recording[-1].pieces_after.code().")
    assert level_kinds_text(sub, {}) == "" and level_kinds_text(Trace(trace.game_id, [], trace.meta), constants) == ""
    none = " ".join(level_kinds_text(sub, {"ODD": ["7777", "7..7"]}).split())
    assert none.startswith("None of the new board's ") and none.endswith("draw them from recording[-1].pieces_after.code().")


def test_the_fit_reports_sprite_by_sprite_reconciliation_on_the_sp80_engine(tmp_path: Path) -> None:
    """The reconciliation (25) on a constructed mismatch at sp80 step 121: the game's frame with the player one row lower,
    a block recoloured, a bin missing and an extra 2x2 piece, against the committed engine's prediction."""
    import numpy as np

    from engine_re.diff_report import RECONCILE_HEAD
    from engine_re.tester import describe_step, predict
    from engine_re.trace import Step, Trace

    trace = Trace.load(SP80 / "trace")
    sub = Trace(trace.game_id, trace.steps[:121], trace.meta)
    step = trace[121]
    prediction, frames = predict(SP80 / "engine_committed.py", sub, [step.action], scratch_root=tmp_path)
    got, states = prediction["steps"][121], prediction["inspect"]["121"]
    assert np.array_equal(frames[121][-1], step.last)  # the replica reproduces the step as played
    sprites = states["after"]["sprites"]
    expected = step.last.copy()

    def cells(x: int, y: int, w: int, h: int, colour: int) -> None:  # the level's 20x20 grid at scale 3, offset (2, 2)
        expected[2 + y * 3: 2 + (y + h) * 3, 2 + x * 3: 2 + (x + w) * 3] = colour

    player, block, bin_ = sprites[8], sprites[5], sprites[9]
    assert (player["tags"], block["tags"], bin_["tags"]) == (["player"], ["block"], ["bin"])
    cells(player["x"], player["y"], 4, 1, 12)
    cells(player["x"], player["y"] + 1, 4, 1, 9)  # the blue bar one row lower
    cells(block["x"], block["y"], 5, 1, 11)  # the red block yellow
    cells(bin_["x"], bin_["y"], 3, 2, 12)  # the bin gone
    cells(10, 3, 2, 2, 9)  # a blue 2x2 nothing draws
    fake = Step(index=121, action=step.action, frames=np.stack([expected]), state=step.state, levels_completed=step.levels_completed,
                win_levels=step.win_levels, available_actions=step.available_actions)
    text, regions = describe_step(fake, got, frames[121], 3, states=states, crops=False)
    assert len(regions) == 3 and "px differ in 3 region(s)" in text
    tail = text[text.index(RECONCILE_HEAD):].splitlines()
    assert tail == [
        "    sprite by sprite (your replica's sprites against the game's frame; x, y in your grid):",
        '      #5 "" 5x1 colour 8 (red): yours colour 8; the game shows it colour 11 (yellow) at the same place',
        '      #8 "" 4x1 colour 9 (blue): yours at (0, 10); the game shows this shape at (0, 11)',
        '      #9 "" 3x2 colour 11 (yellow): yours visible at (2, 17); the game shows nothing there',
        "      the game shows a 2x2 piece (blue, 4 cells) at (10, 3) that none of your sprites draws",
    ]
    same, _ = describe_step(step, got, frames[121], 3, states=states, crops=False)
    assert RECONCILE_HEAD not in same and "final frame: matches" in same


def test_support_comments_on_the_sp80_engine() -> None:
    """The committed engine's listing with the v11 map: step() and the functions it calls carry the comments; the
    level functions and make_level do not. (The v11 map kept four steps at each end, so the fifth named is the
    fourth of the first ones; a v12 map keeps five.)"""
    from engine_re import hashline
    from engine_re import support as sup
    from engine_re.game_api import fixed_block_lines

    text = (SP80 / "engine_committed.py").read_text()
    smap = json.loads((SP80 / "engine_committed.support.json").read_text())
    spans = sup.step_functions(text)
    assert set(spans) == {"step", "cells_of", "erode", "shape_pixels", "absorb", "cavities", "pour", "hud_pixels", "grid_dir"}
    comments = sup.comments(smap, text)
    listing = hashline.render_read(text, max_chars=30000, fold=fixed_block_lines(text), margin=sup.margins(smap, text), comments=comments)
    lines = listing.splitlines()
    assert next(line for line in lines if "if action.id in (1, 2, 3, 4):" in line) == \
        " 121| 453#VSX:    if action.id in (1, 2, 3, 4):  # support (121): 121, 120, 119, 118, 4 and 116 other"
    assert next(line for line in lines if 'state.status = "level_solved"' in line).endswith("  # support (3): 51, 30, 11")
    assert not any("# support" in line for line in lines if "level_3_sprites" in line or "obj(BIN, 2, 17" in line)
    assert sum("# support" in line for line in lines[1:]) == len(comments) == 116  # (the margin note names the comment too)
