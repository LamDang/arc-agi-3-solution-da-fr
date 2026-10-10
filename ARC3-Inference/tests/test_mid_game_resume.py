"""--resume-mid-game: a game run killed mid-game continues where it stopped."""
from __future__ import annotations

import json
import lzma
import threading
from pathlib import Path
from typing import Any

import arcengine
import pytest
import taaf.game
import taaf.game_api

from inference.agent import tool_agent
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
from inference.agent.tool_agent import (
    AnalyzerTurnResult,
    ToolAgent,
    _append_request_snapshot,
    _ChatCompletionResult,
)
from inference.framework import mid_game_resume as mgr
from inference.framework import run as run_module
from inference.framework.solver import HarnessSolver, _HarnessGameSession
from inference.utils.run_artifacts import compress_log
from tests.test_play import CLICK_GAME, GAME, _write_game

UP, DOWN, LEFT, RIGHT, RESET = "ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"
STEM = "twol-0000_p0"
CLICK_STEM = "clik-0000_p0"


@pytest.fixture()
def environments(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    _write_game(root, "twol", GAME, [3, 3])
    _write_game(root, "clik", CLICK_GAME, [2, 1])
    return root


class _FakeAnalyzer:
    """Stands in for ToolAgent: records restore_session and every analyze call."""

    def __init__(self, on_analyze=None) -> None:
        self.generated_tokens = 0
        self.restored: dict[str, Any] | None = None
        self.analyze_calls: list[dict[str, Any]] = []
        self.on_analyze = on_analyze

    def restore_session(self, state_path: Path, **kwargs: Any) -> None:
        self.restored = {"state_path": state_path, **kwargs}
        self.generated_tokens = kwargs["generated_tokens"]

    def analyze(self, state_path, action_num, **kwargs):  # noqa: ARG002
        self.analyze_calls.append({"action_num": action_num, **kwargs})
        if self.on_analyze is not None:
            return self.on_analyze()
        raise AssertionError("no turn expected")


def _game(environments: Path, name: str = "twol") -> taaf.game_api.GameAPI:
    return taaf.game_api.GameAPI(
        env_name=f"{name}-0000",
        arcade_spec=taaf.game_api.ArcadeSpec(environments_dir=str(environments)),
    )


def _session(solver: HarnessSolver, game, analyzer, stem: str) -> _HarnessGameSession:
    artifacts = solver._artifacts_dir()
    return _HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=artifacts / f"{stem}_{RUNTIME_STATE_FILENAME}",
        transcript_path=solver._transcripts_dir() / f"{stem}.txt",
        analysis_html_relpath=f"solver_analysis/{stem}.html",
        stop_event=solver._stop_event,
        viewer_data_path=artifacts / f"{stem}_viewer_data.json",
    )


def _solver(job_dir: Path, **kwargs: Any) -> HarnessSolver:
    solver = HarnessSolver(**kwargs)
    solver.job_dir = job_dir
    return solver


def _old_run(
    tmp_path: Path,
    environments: Path,
    moves: list[str | tuple[int, int]],
    *,
    name: str = "twol",
    stem: str = STEM,
    tokens: int = 100,
) -> tuple[Path, dict[str, Any]]:
    """A run killed mid-game: the artifacts the solver writes while playing
    ``moves`` (a game over is followed by the automatic RESET, as in play),
    a request log of one turn per action, and the game run's benchmark entry."""
    old_dir = tmp_path / "old"
    game = _game(environments, name)
    game.start_game()
    session = _session(_solver(old_dir), game, _FakeAnalyzer(), stem)
    session.seed_initial_history()
    session.write_runtime_state()
    session._append_initial_viewer_event()
    session.write_viewer_payload()
    log_path = old_dir / f"{stem}_requests.jsonl"
    history: list[dict] = []
    for number, move in enumerate(moves, start=1):
        if isinstance(move, tuple):
            action = arcengine.ActionInput(id=arcengine.GameAction.ACTION6, data={"x": move[1], "y": move[0]})
        else:
            action = arcengine.ActionInput(id=arcengine.GameAction.from_name(move), data={})
        turn = {"analysis_step": number, "action": session.action_count + 1, "request_index_within_turn": 1}
        messages = [
            {"role": "system", "content": "rules"},
            *history,
            {"role": "user", "content": f"Current state: step {session.action_count + 1}, level 1."},
        ]
        _append_request_snapshot(log_path, messages=messages, tools=[], event="request", **turn)
        reply = {
            "role": "assistant",
            "content": f"World model: move {number}",
            "tool_calls": [{"id": f"c{number}", "type": "function", "function": {"name": "python", "arguments": "{}"}}],
        }
        _append_request_snapshot(
            log_path, messages=None, tools=None, event="response", reply=reply,
            usage={"completion_tokens": tokens, "total_tokens": tokens * 10}, **turn,
        )
        history = [*messages[1:], reply, {"role": "tool", "tool_call_id": f"c{number}", "content": "ok"}]
        session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=tokens)
        if session.game.current_state.raw.state == arcengine.GameState.GAME_OVER:
            session._execute_auto_reset()
    session.write_viewer_payload()
    # the turn in flight at the kill: its request went out, no reply came back
    _append_request_snapshot(
        log_path,
        messages=[
            {"role": "system", "content": "rules"},
            *history,
            {"role": "user", "content": f"Current state: step {session.action_count + 1}, level 2."},
        ],
        tools=[], event="request", analysis_step=len(moves) + 1,
        action=session.action_count + 1, request_index_within_turn=1,
    )
    prior = game.game_run.to_json_dict()
    return old_dir, prior


TWOL_MOVES = [RIGHT, RIGHT, RIGHT, LEFT, RIGHT]  # level 1 solved, then a death and its RESET on level 2


# --- reading the earlier run ------------------------------------------------------------------


def test_recorded_actions_include_the_automatic_reset(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)

    recorded = mgr.recorded_actions(old_dir, STEM)

    assert recorded["source"] == "runtime_state"
    steps = recorded["steps"]
    assert [s["id"] for s in steps] == [RIGHT, RIGHT, RIGHT, LEFT, RESET, RIGHT]
    assert [s["automatic"] for s in steps] == [False] * 4 + [True, False]
    assert [s["levels_completed"] for s in steps] == [0, 0, 1, 1, 1, 1]
    mgr.check_against_benchmark(steps, prior)


def test_a_click_is_stored_as_row_col_and_replayed_as_x_y(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, [(44, 45), (20, 21)], name="clik", stem=CLICK_STEM)

    steps = mgr.recorded_actions(old_dir, CLICK_STEM)["steps"]

    assert steps[0]["result"]["action_data"] == {"row": 44, "col": 45}
    assert steps[1]["data"] == {"x": 21, "y": 20} and steps[1]["display"] == "MOUSE(row=20, col=21)"
    action = mgr.engine_action({"id": "ACTION6", "data": {"row": 4, "col": 7}})
    assert action.id == arcengine.GameAction.ACTION6 and dict(action.data) == {"x": 7, "y": 4}
    game = _game(environments, "clik")
    game.start_game()
    assert mgr.replay(game, steps, initial_grid_sha=mgr.recorded_actions(old_dir, CLICK_STEM)["initial_grid_sha"]) is None
    assert game.current_state.levels_completed == 1
    mgr.check_against_benchmark(steps, prior)


def test_the_event_log_is_the_fallback(tmp_path: Path, environments: Path) -> None:
    old_dir, _prior = _old_run(tmp_path, environments, [(44, 45), (20, 21)], name="clik", stem=CLICK_STEM)
    primary = mgr.recorded_actions(old_dir, CLICK_STEM)
    mgr.runtime_state_path(old_dir, CLICK_STEM).unlink()

    fallback = mgr.recorded_actions(old_dir, CLICK_STEM)

    assert fallback["source"] == "events"
    assert fallback["initial_grid_sha"] == primary["initial_grid_sha"]
    keys = ("id", "data", "display", "grid_sha", "levels_completed")
    assert [{k: s[k] for k in keys} for s in fallback["steps"]] == [
        {k: s[k] for k in keys} for s in primary["steps"]
    ]


def test_benchmark_json_must_agree_with_the_recorded_actions(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    steps = mgr.recorded_actions(old_dir, STEM)["steps"]
    prior["history"][1]["action"]["id"] = "ACTION1"
    with pytest.raises(mgr.MidGameResumeError, match="action 2"):
        mgr.check_against_benchmark(steps, prior)


def test_conversation_is_the_last_request_without_system_and_opener(tmp_path: Path) -> None:
    log = tmp_path / "g_p0_requests.jsonl"
    history = [
        {"role": "user", "content": "Current state: step 1, level 1."},
        {"role": "assistant", "content": "Plan: go right", "tool_calls": [], "reasoning_details": [{"type": "x"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    _append_request_snapshot(log, messages=[{"role": "system", "content": "rules"}, *history[:1]], tools=[],
                             event="request", analysis_step=1, action=1, request_index_within_turn=1)
    _append_request_snapshot(log, messages=None, tools=None, event="response", analysis_step=1, action=1,
                             request_index_within_turn=1, reply={"role": "assistant", "content": "Plan: go right"},
                             usage={"completion_tokens": 30, "total_tokens": 130})
    opener = {"role": "user", "content": "Current state: step 2, level 1."}
    _append_request_snapshot(log, messages=[{"role": "system", "content": "rules"}, *history, opener], tools=[],
                             event="request", analysis_step=2, action=2, request_index_within_turn=1)
    # a compaction note request after it is not what the model saw last
    _append_request_snapshot(log, messages=[{"role": "system", "content": "rules"}, *history, opener], tools=[],
                             event="request", analysis_step=3, action=2, request_index_within_turn=0,
                             kind="note_compaction")
    _append_request_snapshot(log, messages=None, tools=None, event="response", analysis_step=3, action=2,
                             request_index_within_turn=0, kind="note_compaction",
                             reply={"role": "assistant", "content": "World model: note"},
                             usage={"completion_tokens": 7, "prompt_tokens": 50})
    with log.open("a") as handle:
        handle.write('{"event": "request", "messages": [{"role": "sys')  # torn by the kill

    conversation = mgr.restored_conversation(log)

    assert conversation["messages"] == history
    assert conversation["generated_tokens"] == 37
    assert conversation["total_tokens"] == 130 + 57
    assert conversation["analysis_step"] == 2
    assert conversation["assistant_texts"] == ["Plan: go right"]
    assert conversation["tokens_by_action"] == {1: 30, 2: 7}

    compress_log(log)
    assert mgr.restored_conversation(log.with_name(log.name + ".xz")) == conversation


def test_a_mid_turn_tail_ending_in_tool_results_is_kept(tmp_path: Path) -> None:
    log = tmp_path / "g_p0_requests.jsonl"
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Current state: step 1, level 1."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "python"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    _append_request_snapshot(log, messages=messages, tools=[], event="request", analysis_step=1, action=1,
                             request_index_within_turn=2)
    assert mgr.restored_conversation(log)["messages"] == messages[1:]


def test_prior_elapsed_trusts_only_plausible_file_times(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    events = tmp_path / "events.jsonl"
    events.write_text("{}\n")
    started = datetime.fromtimestamp(events.stat().st_mtime) - timedelta(seconds=500)
    assert mgr.prior_elapsed_seconds(last_wallclock=400, started_at=started, files=[events]) == pytest.approx(500)
    # a file written long after the game (a copy, a checkout) is not this run's
    assert mgr.prior_elapsed_seconds(
        last_wallclock=10, started_at=started, files=[events], max_gap_seconds=100
    ) == 10
    assert mgr.prior_elapsed_seconds(last_wallclock=600, started_at=started, files=[events]) == 600
    assert mgr.prior_elapsed_seconds(last_wallclock=5, started_at=None, files=[events]) == 5


def test_copy_for_continuation_cuts_a_torn_line_and_decompresses(tmp_path: Path) -> None:
    source = tmp_path / "a_requests.jsonl"
    source.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
    assert mgr.copy_log_for_continuation(source, tmp_path / "out" / source.name).read_text() == '{"a": 1}\n{"b": 2}\n'
    whole = tmp_path / "b_requests.jsonl"
    whole.write_text('{"a": 1}\n')
    packed = compress_log(whole)
    copied = mgr.copy_log_for_continuation(packed, tmp_path / "out" / "b_requests.jsonl")
    assert copied.read_text() == '{"a": 1}\n'


# --- the replay -------------------------------------------------------------------------------


def test_replay_reproduces_every_board(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    plan = mgr.build_plan(old_dir, STEM, prior)
    game = _game(environments)
    game.start_game()

    assert mgr.replay(game, plan["steps"], initial_grid_sha=plan["initial_grid_sha"]) is None
    assert len(game.game_run.history) == 6 and game.current_state.levels_completed == 1
    # each step carries its earlier record's tokens; the RESET none
    assert [s["generated_tokens"] for s in plan["steps"]] == [100, 100, 100, 100, 0, 100]
    assert plan["prior_generated_tokens"] == 500 and plan["recorded_tokens"] == 500
    assert plan["conversation"]["analysis_step"] == 6


def test_a_tampered_board_is_a_divergence(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    path = mgr.runtime_state_path(old_dir, STEM)
    state = json.loads(path.read_text())
    state["history"][3]["frame"]["grid"][2][2] = 7  # the board after action 3
    path.write_text(json.dumps(state))
    plan = mgr.build_plan(old_dir, STEM, prior)
    game = _game(environments)
    game.start_game()

    assert mgr.replay(game, plan["steps"], initial_grid_sha=plan["initial_grid_sha"]) == 2
    assert len(game.game_run.history) == 3


def test_lagging_benchmark_actions_take_their_tokens_from_the_request_log(
    tmp_path: Path, environments: Path
) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES, tokens=40)
    prior["history"] = prior["history"][:2]  # benchmark.json saved two actions before the kill

    plan = mgr.build_plan(old_dir, STEM, prior)

    assert [s["generated_tokens"] for s in plan["steps"]] == [40, 40, 40, 40, 0, 40]
    walls = [s["wallclock_seconds"] for s in plan["steps"]]
    assert walls == sorted(walls)


# --- the solver -------------------------------------------------------------------------------


def _resumed(tmp_path: Path, environments: Path, analyzer, **solver_kwargs: Any):
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    plan = mgr.build_plan(old_dir, STEM, prior)
    new_dir = tmp_path / "new"
    for name in (f"{STEM}_events.jsonl", f"{STEM}_viewer_data.json"):
        mgr.copy_log_for_continuation(old_dir / "artifacts" / name, new_dir / "artifacts" / name)
    mgr.copy_log_for_continuation(old_dir / f"{STEM}_requests.jsonl", new_dir / f"{STEM}_requests.jsonl")
    solver = _solver(new_dir, mid_game_resume={STEM: plan}, **solver_kwargs)
    solver._warmup_remaining = 1
    solver._warmup_lock = threading.Lock()
    game = _game(environments)
    game.start_game()
    solver.analyzer_factory = lambda _game, _index: analyzer
    solver._play_one(game, 0, 0)
    return game, plan, new_dir


def test_resumed_tokens_over_the_cap_give_up_at_once(tmp_path: Path, environments: Path) -> None:
    analyzer = _FakeAnalyzer()
    game, plan, new_dir = _resumed(tmp_path, environments, analyzer, max_generated_tokens_per_game=400)

    run = game.game_run
    assert analyzer.analyze_calls == []
    assert run.state == "gave_up" and run.solver_note == "tokens=500"
    # replayed, with no warmup RESET in front, and the earlier records' costs
    assert [r.action.id.name for r in run.history] == [RIGHT, RIGHT, RIGHT, LEFT, RESET, RIGHT]
    assert [r.generated_tokens for r in run.history] == [s["generated_tokens"] for s in plan["steps"]]
    assert [r.wallclock_seconds for r in run.history] == [s["wallclock_seconds"] for s in plan["steps"]]
    restored = analyzer.restored
    assert restored["generated_tokens"] == 500 and restored["step"] == 7 and restored["level"] == 2
    assert restored["actions_at_level_start"] == 3
    assert restored["history_messages"][-1] == {"role": "tool", "tool_call_id": "c5", "content": "ok"}
    transcript = (new_dir / "transcripts" / f"{STEM}.txt").read_text()
    assert f"--- resumed from {plan['source_dir']} at action 6 ---" in transcript


def test_resumed_game_continues_the_clock_counters_and_viewer_log(tmp_path: Path, environments: Path) -> None:
    seen: dict[str, Any] = {}

    def turn():
        seen["session_tokens"] = analyzer.generated_tokens
        analyzer.generated_tokens += 25
        with Path(analyzer.analyze_calls[-1]["transcript_path"]).open("a") as handle:
            handle.write("thinking\n")
        game.game_run.solver_note = None
        stop.set()
        return AnalyzerTurnResult(step_executed=False)

    analyzer = _FakeAnalyzer(turn)
    stop = threading.Event()
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    plan = mgr.build_plan(old_dir, STEM, prior)
    plan["prior_elapsed_seconds"] = 1234.0
    new_dir = tmp_path / "new"
    for name in (f"{STEM}_events.jsonl",):
        mgr.copy_log_for_continuation(old_dir / "artifacts" / name, new_dir / "artifacts" / name)
    mgr.copy_log_for_continuation(old_dir / f"{STEM}_requests.jsonl", new_dir / f"{STEM}_requests.jsonl")
    # killed mid-batch: the last action is in the runtime state, not yet in the event log
    events_file = new_dir / "artifacts" / f"{STEM}_events.jsonl"
    old_events = events_file.read_text().splitlines()[:-1]
    events_file.write_text("\n".join(old_events) + "\n")
    solver = _solver(new_dir, mid_game_resume={STEM: plan}, max_runtime_s_per_game=10_000)
    solver._stop_event = stop
    game = _game(environments)
    game.start_game()
    solver.analyzer_factory = lambda _game, _index: analyzer
    solver._play_one(game, 0, 0)

    (call,) = analyzer.analyze_calls
    assert call["action_num"] == 6 and call["analysis_step"] == 7
    assert seen["session_tokens"] == 500
    assert game.game_run.solver_note == "tokens=525"
    events = (new_dir / "artifacts" / f"{STEM}_events.jsonl").read_text().splitlines()
    assert events[: len(old_events)] == old_events and len(events) > len(old_events)
    added = [json.loads(line) for line in events[len(old_events):]]
    assert [(e["type"], e["action_num"]) for e in added] == [("action", 6), ("analysis", 6)]
    assert added[0]["action_display"] == "RIGHT"
    assert game.game_run.final_wallclock_seconds >= 1234.0


def test_a_divergent_replay_continues_with_a_fresh_conversation(tmp_path: Path, environments: Path) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    plan = mgr.build_plan(old_dir, STEM, prior)
    plan["steps"][3]["grid_sha"] = "0" * 40
    analyzer = _FakeAnalyzer()
    solver = _solver(tmp_path / "new", mid_game_resume={STEM: plan}, max_generated_tokens_per_game=1)
    game = _game(environments)
    game.start_game()
    solver.analyzer_factory = lambda _game, _index: analyzer
    solver._play_one(game, 0, 0)

    assert analyzer.restored is None
    assert len(game.game_run.history) == 4
    assert game.game_run.solver_note.startswith("mid-game resume diverged at action 4 of 6")


# --- the agent --------------------------------------------------------------------------------


def test_restore_session_keeps_history_and_puts_the_note_in_the_opener(
    tmp_path: Path, environments: Path, monkeypatch
) -> None:
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    conversation = mgr.restored_conversation(old_dir / f"{STEM}_requests.jsonl")
    # the state the solver writes before each turn
    state_path = old_dir / "artifacts" / f"{STEM}_{RUNTIME_STATE_FILENAME}"
    agent = ToolAgent(model="test/model")
    agent._tool_steps = 1
    sent: list[list[dict]] = []

    def chat(messages, **kwargs):  # noqa: ARG001
        sent.append(json.loads(json.dumps(messages)))
        return _ChatCompletionResult(
            message={"role": "assistant", "content": "thinking"}, finish_reason="stop",
            usage={"completion_tokens": 11, "total_tokens": 21},
        )

    monkeypatch.setattr(agent, "_chat_completion", chat)
    agent.restore_session(
        state_path,
        history_messages=conversation["messages"],
        generated_tokens=conversation["generated_tokens"],
        total_tokens=conversation["total_tokens"],
        assistant_texts=conversation["assistant_texts"],
        step=7,
        level=2,
    )
    assert agent.generated_tokens == 500

    agent.analyze(state_path, 6, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
                  transcript_path=tmp_path / "t.txt", analysis_step=7)

    (request,) = sent
    assert request[0]["role"] == "system"
    assert request[1:-1] == tool_agent._strip_control_keys(conversation["messages"])
    opener = request[-1]["content"]
    text = opener if isinstance(opener, str) else opener[0]["text"]
    assert text.startswith(
        "[harness] The run was interrupted and has now resumed in this same conversation at step 7, level 2."
    )
    assert agent.generated_tokens == 511
    assert agent._pending_restart_note == ""
    assert agent._history_messages[: len(conversation["messages"])] == conversation["messages"]


def test_restore_session_can_strip_encrypted_reasoning(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARC3_RESUME_STRIP_REASONING_DETAILS", "1")
    agent = ToolAgent(model="test/model")
    message = {"role": "assistant", "content": "x", "reasoning_details": [{"type": "reasoning.encrypted"}]}
    agent.restore_session(tmp_path / "s.json", history_messages=[message], generated_tokens=3, step=2, level=1)
    assert agent._history_messages == [{"role": "assistant", "content": "x"}]
    assert "reasoning_details" in message


# --- run.py -------------------------------------------------------------------------------------


def _two_run_dir(tmp_path: Path, environments: Path) -> tuple[Path, list, dict]:
    """An earlier run of [clik, twol]: clik won (kept), twol killed while playing."""
    old_dir, prior = _old_run(tmp_path, environments, TWOL_MOVES)
    won = {**prior, "game_id": "clik-0000", "state": "won", "solver_note": "tokens=9", "history": []}
    (old_dir / "benchmark.json").write_text(json.dumps({"game_runs": [won, prior]}))
    (old_dir / "artifacts" / f"{CLICK_STEM}_viewer_data.json").write_text("{}")
    (old_dir / "transcripts").mkdir(exist_ok=True)
    (old_dir / "transcripts" / f"{STEM}.txt").write_text("old transcript\n")
    log = old_dir / f"{STEM}_requests.jsonl"
    with log.open("a") as handle:
        handle.write('{"torn')
    prior_runs = [taaf.game.GameRun.from_json_dict(won), None]
    return old_dir, prior_runs, prior


def test_default_resume_is_unchanged(tmp_path: Path, environments: Path) -> None:
    old_dir, prior_runs, _prior = _two_run_dir(tmp_path, environments)
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    run_module._prepare_resume(old_dir, run_dir=new_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"])

    record = json.loads((new_dir / "resume.json").read_text())
    assert "continued" not in record
    assert [e["game_id"] for e in record["kept"]] == ["clik-0000"]
    assert [e["game_id"] for e in record["replayed"]] == ["twol-0000"]
    assert (new_dir / "artifacts" / f"{CLICK_STEM}_viewer_data.json").exists()
    assert not list(new_dir.rglob(f"{STEM}*"))


def test_mid_game_resume_continues_the_playing_run(tmp_path: Path, environments: Path) -> None:
    old_dir, prior_runs, _prior = _two_run_dir(tmp_path, environments)
    plans = run_module._plan_mid_game_resume(
        old_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"],
        environments_dir=str(environments),
    )
    assert list(plans) == [STEM] and plans[STEM]["level"] == 2
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    run_module._prepare_resume(
        old_dir, run_dir=new_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"], continued=plans,
    )

    record = json.loads((new_dir / "resume.json").read_text())
    assert record["replayed"] == []
    (continued,) = record["continued"]
    assert continued["game_id"] == "twol-0000" and continued["actions"] == 6 and continued["level"] == 2
    assert (new_dir / "transcripts" / f"{STEM}.txt").read_text() == "old transcript\n"
    assert (new_dir / "artifacts" / f"{STEM}_events.jsonl").exists()
    assert not (new_dir / "artifacts" / f"{STEM}_{RUNTIME_STATE_FILENAME}").exists()
    copied = (new_dir / f"{STEM}_requests.jsonl").read_text()
    assert copied.endswith("\n") and '{"torn' not in copied


def test_a_compressed_request_log_is_continued_uncompressed(tmp_path: Path, environments: Path) -> None:
    old_dir, prior_runs, _prior = _two_run_dir(tmp_path, environments)
    compress_log(old_dir / f"{STEM}_requests.jsonl")
    plans = run_module._plan_mid_game_resume(
        old_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"],
        environments_dir=str(environments),
    )
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    run_module._prepare_resume(
        old_dir, run_dir=new_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"], continued=plans,
    )
    assert not (new_dir / f"{STEM}_requests.jsonl.xz").exists()
    with lzma.open(old_dir / f"{STEM}_requests.jsonl.xz", "rt") as handle:
        original = handle.read()
    assert original.startswith((new_dir / f"{STEM}_requests.jsonl").read_text())


def test_a_run_that_does_not_replay_is_replayed_from_level_1(tmp_path: Path, environments: Path) -> None:
    old_dir, prior_runs, _prior = _two_run_dir(tmp_path, environments)
    path = mgr.runtime_state_path(old_dir, STEM)
    state = json.loads(path.read_text())
    state["history"][2]["frame"]["grid"][0][0] = 7
    path.write_text(json.dumps(state))

    plans = run_module._plan_mid_game_resume(
        old_dir, prior_runs=prior_runs, game_ids=["clik-0000", "twol-0000"],
        environments_dir=str(environments),
    )

    assert plans == {}
