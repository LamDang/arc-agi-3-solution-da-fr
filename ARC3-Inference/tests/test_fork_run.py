"""engine_re.tools.fork_run: a finished play run truncated at one of its turns, resumed by the play agent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine_re.agent import RESUME_NOTE, Budget, ModelConfig
from engine_re.kernel import writes_files
from engine_re.play_agent import COMMITTED_FILE, FORK_FILE, SUPPORT_FILE, PlayAgent
from engine_re.prompts import NOTES_TEMPLATE
from engine_re.tools.fork_run import ForkError, cut_records, cut_tests, cut_trace, fork_run, fork_state
from engine_re.trace import Action, Step, Trace
from tests.test_play import GAME, _ScriptedModel, _start, _write_game

# --- the truncation of a synthetic transcript -----------------------------------------------------


def _turn(t: int, *calls: tuple[str, dict], cost: float = 0.1) -> dict:
    return {"turn": t, "finish_reason": "tool_calls", "content": "", "reasoning": "", "elapsed_min": float(t),
            "tool_calls": [{"id": f"c{t}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": cost}}


RESUME = RESUME_NOTE.format(replay="The python kernel restarted.", names="names")
RECORDS = [
    {"turn": 0, "plan": {"step": 0, "steps": 1, "level": 0}, "phase": "plan"},
    {"turn": 0, "engine_change": {"op": "edit", "version": 2, "summary": "opening"}, "by": "harness"},
    {"turn": 0, "message": {"role": "system", "content": "system"}},
    {"turn": 0, "message": {"role": "user", "content": "Plan the next moves. Steps 0-0 pass"}},
    _turn(1, ("python", {"code": "edit_file(edits=[])"})),
    {"turn": 1, "engine_change": {"op": "edit", "version": 3, "summary": "the rules"}, "phase": "plan"},
    {"turn": 1, "tool": "python", "id": "c1_0", "output": "engine.py: edited", "phase": "plan"},
    {"turn": 1, "append": "\n[harness] tested", "phase": "plan"},
    {"turn": 1, "auto_test": "TEST RESULT", "phase": "plan"},
    _turn(2, ("python", {"code": "probe = 1"}), ("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "go"})),
    {"turn": 2, "tool": "python", "id": "c2_0", "output": "", "phase": "plan"},
    {"turn": 2, "commit": {"fixed": 0, "next": None, "message": "(commit_moves) go", "engine_sha": "B", "version": 3}, "phase": "plan"},
    {"turn": 2, "move": {"index": 1, "label": "Action(4)", "ok": True, "state": "NOT_FINISHED"}, "phase": "plan"},
    {"turn": 2, "move": {"index": 2, "label": "Action(4)", "ok": False, "verdict": "differs", "state": "NOT_FINISHED"}, "phase": "plan"},
    {"turn": 2, "batch": {"turn": 2, "first_step": 1, "moves": ["Action(4)", "Action(4)"], "sent": 2, "matched": 1, "mismatch": 2, "note": "go"}, "phase": "plan"},
    {"turn": 2, "tool": "commit_moves", "id": "c2_1", "output": "Sent 2 of 2 move(s)", "phase": "plan"},
    {"turn": 2, "step_start": {"step": 2, "steps": 3, "report": "TEST RESULT", "verdict": "differs"}, "phase": "fit", "step": 2},
    {"turn": 2, "hide_images": True, "phase": "fit", "step": 2},
    {"turn": 3, "images": ["images/turn002_step2_advance.png"], "captions": ["step 2"], "phase": "fit", "step": 2},  # (logged with turn + 1)
    {"turn": 2, "message": {"role": "user", "content": [{"type": "text", "text": "Fix your replica: step 2"},
                                                       {"type": "image_file", "path": "images/turn002_step2_advance.png"}]}, "phase": "fit", "step": 2},
    {"turn": 2, "replay": {"cells": 2, "replayed": 2, "failed": [], "skipped": 0, "seconds": 0.1}, "phase": "fit", "step": 2},
    {"turn": 2, "message": {"role": "user", "content": RESUME}, "phase": "fit", "step": 2},
    {"turn": 2, "resumed": {"step": 2, "messages": 9}, "phase": "fit", "step": 2},
    _turn(3, ("python", {"code": "edit_file(path='notes.md', edits=[])"})),
    {"turn": 3, "engine_change": {"op": "edit", "version": 4, "summary": "fix"}, "phase": "fit", "step": 2},
    {"turn": 3, "tool": "python", "id": "c3_0", "output": "notes.md: edited", "phase": "fit", "step": 2},
    _turn(4, ("commit_engine", {"message": "fixed"})),
    {"turn": 4, "tool": "commit_engine", "id": "c4_0", "output": "Committed: steps 0-2 pass.", "phase": "fit", "step": 2},
    {"turn": 4, "commit": {"fixed": 2, "next": None, "message": "fixed", "engine_sha": "C", "version": 4}, "phase": "fit", "step": 2},
    {"turn": 4, "plan": {"step": 2, "steps": 3, "level": 0}, "phase": "plan", "step": 2},
    {"turn": 4, "hide_images": True, "phase": "plan", "step": 2},
    {"turn": 4, "images": ["images/turn004_frame2.png"], "captions": ["frame"], "phase": "plan", "step": 2},
    {"turn": 4, "message": {"role": "user", "content": [{"type": "text", "text": "Plan the next moves. Steps 0-2 pass"},
                                                       {"type": "image_file", "path": "images/turn004_frame2.png"}]}, "phase": "plan", "step": 2},
    {"turn": 4, "compact": True, "phase": "plan", "step": 2},
    _turn(5, ("python", {"code": "y = 2"})),
    {"turn": 5, "tool": "python", "id": "c5_0", "output": "", "phase": "plan", "step": 2},
]
TESTS = [{"turn": 0, "auto": "opening", "engine_sha": "A", "level": None, "from_level": None, "passed": True, "exact": 1, "total": 1, "passing_prefix": 1},
         {"turn": 1, "auto": True, "engine_sha": "B", "level": None, "from_level": None, "passed": True, "exact": 1, "total": 1, "passing_prefix": 1},
         {"turn": 2, "auto": False, "engine_sha": "B", "level": None, "from_level": None, "passed": True, "exact": 1, "total": 1, "passing_prefix": 1},
         {"turn": 3, "auto": True, "engine_sha": "C", "level": None, "from_level": None, "passed": False, "exact": 2, "total": 3, "passing_prefix": 2},
         {"turn": 4, "auto": False, "engine_sha": "C", "level": None, "from_level": None, "passed": True, "exact": 3, "total": 3, "passing_prefix": 3}]


def _text(content) -> str:
    """A message's text: the string, or the text parts of a list (a PLAN message carries the sprite list as a part)."""
    return content if isinstance(content, str) else "\n".join(p.get("text", "") for p in content if p.get("type") == "text")


def _kinds(records: list[dict]) -> list[tuple[int, str]]:
    return [(r["turn"], next(k for k in r if k not in ("turn", "phase", "step", "elapsed_min"))) for r in records]


def test_the_cut_keeps_what_the_model_read_at_the_next_turn() -> None:
    kept = cut_records(RECORDS, 2)
    # ends with the FIT message created at the end of turn 2 and its pictures (logged with turn 3); the source's resume there is dropped
    assert _kinds(kept)[-3:] == [(2, "hide_images"), (3, "images"), (2, "message")]
    assert not any("resumed" in r or "replay" in r for r in kept) and not any(r.get("message", {}).get("content") == RESUME for r in kept)
    state = fork_state(kept, 2)
    assert state == {"turn": 2, "steps": 3, "focus": 2, "phase": "fit", "version": 3, "committed_sha": "B", "out_of_sync": None, "resumes": 0}
    # a cut after the resume keeps it (it is in the middle of what the model read)
    kept3 = cut_records(RECORDS, 3)
    assert any("resumed" in r for r in kept3) and _kinds(kept3)[-1] == (3, "tool")
    assert fork_state(kept3, 3) == {"turn": 3, "steps": 3, "focus": 2, "phase": "fit", "version": 4, "committed_sha": "B", "out_of_sync": None, "resumes": 1}
    kept4 = cut_records(RECORDS, 4)
    assert _kinds(kept4)[-2:] == [(4, "message"), (4, "compact")]
    assert fork_state(kept4, 4)["phase"] == "plan" and fork_state(kept4, 4)["committed_sha"] == "C" and fork_state(kept4, 4)["focus"] == 2


def test_the_cut_before_any_commit_takes_the_openings_engine() -> None:
    kept = cut_records(RECORDS, 1)
    assert _kinds(kept)[-1] == (1, "auto_test")
    state = fork_state(kept, 1)
    assert state["steps"] == 1 and state["focus"] == 0 and state["phase"] == "plan" and state["version"] == 3
    assert state["committed_sha"] == {"version": 2}  # the opening's edit, committed without a record


def test_the_cut_refuses_a_turn_the_run_never_reached() -> None:
    with pytest.raises(ForkError):
        cut_records(RECORDS, 9)
    assert [t["turn"] for t in cut_tests(TESTS, 2)] == [0, 1, 2]


def test_the_cut_of_an_out_of_step_run() -> None:
    records = cut_records(RECORDS, 2) + [
        {"turn": 2, "out_of_sync": {"from": 2, "unexplained": [2]}, "phase": "plan"},
        _turn(3, ("commit_moves", {"actions": ["RIGHT"], "note": "blind"})),
        {"turn": 3, "move": {"index": 3, "label": "Action(4)", "ok": None, "state": "NOT_FINISHED", "resync": False}, "phase": "plan"},
        {"turn": 3, "tool": "commit_moves", "id": "c3_0", "output": "Sent 1 of 1 move(s), not checked", "phase": "plan"},
        {"turn": 3, "plan": {"step": 3, "steps": 4, "level": 0, "out_of_sync": 2}, "phase": "plan"},
        _turn(4, ("commit_moves", {"actions": ["RESET"], "note": "resync"})),
        {"turn": 4, "move": {"index": 4, "label": "Action(0)", "ok": None, "state": "NOT_FINISHED", "resync": True}, "phase": "plan"},
        {"turn": 4, "resync": {"step": 4, "level": 0, "score": 0}, "phase": "plan"},
        {"turn": 4, "tool": "commit_moves", "id": "c4_0", "output": "Sent", "phase": "plan"},
    ]
    assert fork_state(cut_records(records, 3), 3)["out_of_sync"] == 2 and fork_state(cut_records(records, 3), 3)["phase"] == "plan"
    assert fork_state(cut_records(records, 4), 4)["out_of_sync"] is None
    steps = [Step(k, Action(0 if k in (0, 4) else 4), np.zeros((1, 64, 64), np.int8), "NOT_FINISHED", 0, 2, [1, 2, 3, 4]) for k in range(6)]
    trace = Trace("twol", steps, {"source": "play", "ignore": [2, 3, 5], "resync": {"4": {"level": 0, "score": 0}}, "out_of_sync": 5})
    cut = cut_trace(trace, 4, 2)
    assert len(cut) == 4 and cut.meta == {"source": "play", "ignore": [2, 3], "out_of_sync": 2}
    cut = cut_trace(trace, 5, None)
    assert len(cut) == 5 and cut.meta == {"source": "play", "ignore": [2, 3], "resync": {"4": {"level": 0, "score": 0}}}


def test_cells_that_write_files_are_told_apart() -> None:
    assert writes_files("edit_file(path='notes.md', edits=[])") and writes_files('edit_file("notes.md", edits=[])')
    assert writes_files("open('plan.txt', 'w').write('x')") and writes_files("Path('a.txt').write_text('x')")
    assert not writes_files("edit_file(edits=[{'op': 'append', 'lines': ['x']}])") and not writes_files("print(open('a.txt').read())")


# --- a run forked and resumed --------------------------------------------------------------------


class _CostedModel(_ScriptedModel):
    def chat(self, messages, tools):
        response = super().chat(messages, tools)
        response["usage"]["cost"] = 0.01
        return response


NOTES_1 = "edit_file(path='notes.md', edits=[{'op': 'replace_text', 'oldText': 'Goal model:', 'newText': 'Goal model: reach x >= 4'}])\nprobe = 42"
NOTES_2 = ("edit_file(path='notes.md', edits=[{'op': 'replace_text', 'oldText': 'Plan:', 'newText': 'Plan: RIGHT x3 (written at turn 4)'}])\n"
           "open('scratch.txt', 'w').write('turn 4')")


@pytest.fixture()
def environments(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    _write_game(root, "twol", GAME, [3, 3])
    return root


def _play_agent(game_dir: Path, environments: Path, model: _ScriptedModel, turns: int) -> PlayAgent:
    return PlayAgent("twol", game_dir, ModelConfig(), Budget(max_turns=turns), environments, client=model, images=False, batch_size=4)


def _source_run(tmp_path: Path, environments: Path) -> Path:
    """A run of five turns: the engine installed and committed, notes written and one move at turn 3, a file and more notes
    and two moves (level 0 solved) at turn 4, nothing at turn 5."""
    model = _CostedModel(_start() + [
        [("python", {"code": NOTES_1}), ("commit_moves", {"actions": ["RIGHT"], "note": "one"})],
        [("python", {"code": NOTES_2}), ("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "two"})],
        [("python", {"code": "late = 1"})],
    ])
    src = tmp_path / "v11" / "twol"
    result = _play_agent(src, environments, model, turns=5).run()
    assert result.status == "budget_turns" and result.actions == 3 and result.levels_completed == 1
    assert (src / "workspace" / "scratch.txt").exists() and "turn 4" in (src / "workspace" / "notes.md").read_text()
    return src


def test_a_fork_holds_the_state_after_its_turn_and_nothing_later(tmp_path: Path, environments: Path) -> None:
    src = _source_run(tmp_path, environments)
    before = sorted(str(p.relative_to(src)) + str(p.stat().st_mtime_ns) for p in src.rglob("*"))
    out = tmp_path / "fork" / "twol"
    summary = fork_run(src, out, 3, environments)
    assert sorted(str(p.relative_to(src)) + str(p.stat().st_mtime_ns) for p in src.rglob("*")) == before  # the source is untouched
    assert summary["steps"] == 2 and summary["phase"] == "plan" and summary["focus"] == 1 and not summary["warnings"]
    assert summary["game"] == {"step": 1, "level": 0, "levels_completed": 0, "state": "NOT_FINISHED", "actions": 1, "score": 0.0}
    assert summary["committed_test"]["passes"] and summary["engine_test"]["passes"] and summary["support_steps"] == 2
    assert len(Trace.load(out / "trace")) == 2 and len(Trace.load(out / "visible_trace")) == 2
    records = [json.loads(line) for line in (out / "transcript.jsonl").read_text().splitlines()]
    assert all(r["turn"] <= 3 for r in records) and "fork" in records[-1] and records[-1]["fork"]["turn"] == 3
    assert _text(records[-2]["message"]["content"]).startswith("Plan the next moves. Steps 0-1 pass")
    assert all(json.loads(line)["turn"] <= 3 for line in (out / "tests.jsonl").read_text().splitlines())
    assert sorted(p.name for p in (out / "workspace").iterdir()) == ["engine.py", "notes.md"]
    assert (out / "workspace" / "notes.md").read_text() == NOTES_TEMPLATE
    assert json.loads((out / FORK_FILE).read_text())["turn"] == 3
    assert (out / COMMITTED_FILE).exists() and (out / SUPPORT_FILE).exists() and (out / "engine_best.py").exists()
    source = json.loads((src / "result.json").read_text())
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "running" and result["turns"] == 3 and result["minutes"] == 0 and result["usage"]["cost_usd"] == 0
    assert result["usage"]["completion_tokens"] == 15 and result["usage"]["requests"] == 3  # the tokens of the kept turns stay
    assert result["forked_from"]["source"] == str(src) and result["forked_from"]["turn"] == 3 and result["forked_from"]["cost_usd"] == pytest.approx(0.03)
    assert result["moves_sent"] == 1 and result["batches"] == 1 and result["trace_steps"] == 2 and result["committed_sha"] == source["committed_sha"]
    assert result["phase_turns"] == {"plan": 3, "fit": 0} and len(result["step_tokens"]) == 1 and result["advances"][-1]["turn"] <= 3
    versions = [json.loads(line) for line in (out / "engine_versions" / "versions.jsonl").read_text().splitlines()]
    assert [v["version"] for v in versions] == [1, 2, 3] and sorted(p.name for p in (out / "engine_versions").glob("*.py")) == ["v0001.py", "v0002.py", "v0003.py"]
    assert json.loads((out.parent / "config.json").read_text())["forks"]["twol"]["fork_turn"] == 3
    with pytest.raises(ForkError):
        fork_run(src, out, 3, environments)  # exists already


def test_a_dry_resume_of_a_fork_rebuilds_the_notes_from_the_kept_cells(tmp_path: Path, environments: Path) -> None:
    src = _source_run(tmp_path, environments)
    out = tmp_path / "fork" / "twol"
    fork_run(src, out, 3, environments)
    summary = _play_agent(out, environments, _ScriptedModel([]), turns=8).dry_resume()
    assert summary["resumed"] and summary["turn"] == 3 and summary["steps"] == 2 and summary["phase"] == "plan" and summary["step"] == 1
    assert summary["level"] == 0 and summary["actions"] == 1 and summary["budget"]["cost_usd"] == 0 and summary["budget"]["over"] is None
    assert summary["engine"]["passes"] and summary["committed"]["passes"] and summary["committed"]["support_steps"] == 2
    assert summary["last_message"]["role"] == "user" and "The run was interrupted here and has now resumed" in summary["last_message"]["text"]
    assert "with edits to engine.py disabled (its versions are kept) and the edits to notes.md" in summary["last_message"]["text"]
    assert summary["messages"] >= 2 and summary["replay"]["files"] is True and summary["replay"]["replayed"] == 2
    assert "probe: int" in summary["kernel"]["names"]
    assert summary["notes"] == "Goal model: reach x >= 4\nOpen questions:\nPlan:\n"  # turn 3's edit, not turn 4's
    assert summary["fork"]["turn"] == 3 and not (out / FORK_FILE).exists()  # the marker is consumed
    assert not (out / "workspace" / "scratch.txt").exists()
    records = [json.loads(line) for line in (out / "transcript.jsonl").read_text().splitlines()]
    assert [r for r in records if "fork" in r][-1]["fork"]["notes_rebuilt"] is True
    # the second message before the resume note is the PLAN message the model read at turn 4 of the source
    agent = _play_agent(out, environments, _ScriptedModel([]), turns=8)
    state = agent._rebuild_conversation()
    users = [m for m in state["messages"] if m["role"] == "user"]
    assert _text(users[-2]["content"]).startswith("Plan the next moves. Steps 0-1 pass")


def test_a_fork_resumes_with_its_own_budget_and_plays_on(tmp_path: Path, environments: Path) -> None:
    src = _source_run(tmp_path, environments)
    out = tmp_path / "fork" / "twol"
    fork_run(src, out, 3, environments, verify=False)
    assert not (out / SUPPORT_FILE).exists() and not (out / "artifacts").exists()
    model = _CostedModel([
        [("commit_moves", {"actions": ["RIGHT", "RIGHT"], "note": "finish level 0"})],
        [("commit_moves", {"actions": ["RIGHT", "RIGHT", "RIGHT"], "note": "finish level 1"})],
    ])
    agent = _play_agent(out, environments, model, turns=5)  # absolute: two turns after the fork's three
    result = agent.run()
    assert result.status == "won" and result.turns == 5 and result.resumes == 1 and result.actions == 6
    assert result.usage.cost_usd == pytest.approx(0.02) and result.minutes < 1  # the fork's own spending only
    assert result.usage.completion_tokens == 25 and result.forked_from["turn"] == 3
    assert "reach x >= 4" in (out / "workspace" / "notes.md").read_text() and "turn 4" not in (out / "workspace" / "notes.md").read_text()
    saved = json.loads((out / "result.json").read_text())
    assert saved["forked_from"]["source"] == str(src) and saved["usage"]["cost_usd"] == pytest.approx(0.02)
    assert not (out / FORK_FILE).exists()
    # restored once more: the budget still counts from the fork record
    again = _play_agent(out, environments, _ScriptedModel([]), turns=9)
    again._restore()
    assert again.result.usage.cost_usd == pytest.approx(0.02) and again.prior_minutes < 1 and again.result.turns == 5


def test_a_fork_at_a_turn_the_budget_has_reached_stops_at_once(tmp_path: Path, environments: Path) -> None:
    src = _source_run(tmp_path, environments)
    out = tmp_path / "fork" / "twol"
    fork_run(src, out, 4, environments, verify=False)
    model = _CostedModel([[("python", {"code": "z = 1"})]])
    result = _play_agent(out, environments, model, turns=4).run()
    assert result.status == "budget_turns" and result.turns == 4 and not model.seen
    assert "turn 4" in (out / "workspace" / "notes.md").read_text()  # the turn-4 cells belong to this fork
