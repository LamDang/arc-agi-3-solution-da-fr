"""engine_re.condense: the iteration condenser, on a scripted stepwise run and on its own."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from engine_re.agent import CONTINUE, IMAGE_PLACEHOLDER, Budget, EngineAgent, ModelConfig
from engine_re.condense import (
    FAILED_KEPT,
    LISTING_ELIDED,
    NO_COMMIT_MESSAGE,
    annotate,
    condense,
    failed_command,
    iteration_diff,
    iterations,
    message_text,
    net_diff,
)
from engine_re.trace import Trace
import test_engine_re as base
from test_engine_re import SIMPLE_TINY_GAME, _RecordingModel, _rewrite_now, _ScriptedModel

tiny_trace = base.tiny_trace  # the fixture

MOVES_LINE = "    moves = {1: (0, -1), 2: (0, DOWN), 3: (-1, 0), 4: (1, 0)}\n"


def _engine(up: str, down: str, left: str, right: str) -> str:
    """The tiny game with each direction's move given as an expression (the recording: 0 reset, 2 down, 2 down,
    4 right, 1 up, 1 up, 1 up, 3 left)."""
    return SIMPLE_TINY_GAME.replace(MOVES_LINE, f"    moves = {{1: {up}, 2: {down}, 3: {left}, 4: {right}}}\n")


DOWN_ONCE = _engine("(0, 0)", '(0, 1 if state.vars["player"].y == 1 else 0)', "(0, 0)", "(0, 0)")  # fixes step 0; step 2 fails
DOWN = _engine("(0, 0)", "(0, 1)", "(0, 0)", "(0, 0)")  # step 3 fails
DOWN_RIGHT = _engine("(0, 0)", "(0, 1)", "(0, 0)", "(1, 0)")  # step 4 fails
NO_LEFT = _engine("(0, -1)", "(0, 1)", "(0, 0)", "(1, 0)")  # step 7 fails
BAD_LEFT = _engine("(0, -1)", "(0, 1)", "(-2, 0)", "(1, 0)")  # step 7 still fails


class _ThinkingModel(_RecordingModel):
    """The scripted model with a reasoning field on every turn."""

    def chat(self, messages, tools):
        response = super().chat(messages, tools)
        response["choices"][0]["message"]["reasoning"] = f"thinking at turn {len(self.calls)}"
        return response


# Five iterations: four fixed by a commit (each with a failed call, a successful call and an edit), the fifth open with
# seven turns (two of them older than the last five), a failed call next to a successful one, and an edit followed by a
# failed call that carries the automatic test.
SCRIPT = [
    [("python", {"code": "1/0"})],  # turn 1: a Traceback
    [("python", {"code": "print('probe1')"}), ("python", {"code": _rewrite_now(DOWN_ONCE)})],  # turn 2
    [("commit_engine", {"message": "down once"})],  # turn 3: fixes step 0; on to step 2
    [("python", {"code": "edit_file(edits=[{'op': 'replace_text', 'oldText': 'nope-nope', 'newText': 'x'}])"})],  # turn 4: [E_NO_MATCH]
    [("python", {"code": "print('probe2')"}), ("python", {"code": _rewrite_now(DOWN)})],  # turn 5
    [("commit_engine", {"message": "down always"})],  # turn 6: on to step 3
    [("read_file", {})],  # turn 7: unknown tool
    [("python", {"code": "print('probe3')"}), ("python", {"code": _rewrite_now(DOWN_RIGHT)})],  # turn 8
    [("commit_engine", {"message": "right too"})],  # turn 9: on to step 4
    [("commit_engine", {"message": "too early"})],  # turn 10: not committed
    [("python", {"code": "print('probe4')"}), ("python", {"code": _rewrite_now(NO_LEFT)})],  # turn 11
    [("commit_engine", {"message": "up too"})],  # turn 12: on to step 7
    [("python", {"code": "1/0"}), ("python", {"code": "print('both')"})],  # turn 13: one failed, one fine
    [("python", {"code": _rewrite_now(BAD_LEFT)}), ("python", {"code": "raise ValueError('after the edit')"})],  # turn 14
    [("python", {"code": "print('fifteen')"})],
    [("python", {"code": "print('sixteen')"})],
    [("run_tests", {})],  # turn 17: a report with a failing step is not a failed command
    [("python", {"code": "print('eighteen')"})],
    [("python", {"code": "print('nineteen')"})],
]


def _run(tmp_path: Path, trace: Trace) -> tuple[list[dict], list[dict], list[dict]]:
    """Run the script; returns (the live conversation, the full one with every image live, the records)."""
    trace.save(tmp_path / "trace")
    model = _ThinkingModel(copy.deepcopy(SCRIPT))
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=len(SCRIPT)), client=model, stepwise=True)
    result = agent.run()
    assert result.status == "budget_turns" and result.step == 7, result
    assert [(a["fixed"], a["next"]) for a in result.advances] == [(0, 2), (2, 3), (3, 4), (4, 7)]
    log = tmp_path / "transcript.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    # The conversation with every image live: the transcript replayed without its hide_images records.
    kept = log.read_text()
    log.write_text("".join(json.dumps(r) + "\n" for r in records if "hide_images" not in r))
    full = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(), client=_ScriptedModel([]), stepwise=True)._rebuild_conversation()["messages"]
    log.write_text(kept)
    return agent.messages, full, records


def _texts(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(p.get("text") or "" for p in content if p.get("type") == "text")


def _without_placeholders(message: dict) -> str:
    return "\n".join(line for line in _texts(message).split("\n") if line != IMAGE_PLACEHOLDER)


def _images(message: dict) -> int:
    content = message["content"]
    return sum(p.get("type") == "image_url" for p in content) if isinstance(content, list) else 0


def test_condense_on_a_scripted_run(tmp_path: Path, tiny_trace: Trace) -> None:
    live, full, records = _run(tmp_path, tiny_trace)
    versions = tmp_path / "engine_versions"
    notes = annotate(full, records)
    its = iterations(notes, records)
    assert [(it.step, it.first_turn, it.last_turn, it.finished) for it in its] == [
        (0, 1, 3, True), (2, 4, 6, True), (3, 7, 9, True), (4, 10, 12, True), (7, 13, 19, False),
    ]
    assert [n.failed for n in notes if n.kind == "tool"] == [
        True, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False,
        True, False, False, True, False, False, False, False, False,
    ]

    before = copy.deepcopy(full)
    condensed = condense(full, records, versions)
    assert full == before  # a pure function
    assert condense(full, records, versions) == condensed  # and a deterministic one
    out = condensed.messages
    assert condensed.cap is None
    assert out[0] == full[0]  # the system prompt unchanged
    assert [m["role"] for m in out[:5]] == ["system", "user", "user", "user", "user"]

    # The oldest iteration (step 0, turns 1-3): text + the net diff + the commit; its listing elided, no images, no calls.
    oldest = out[1]
    assert isinstance(oldest["content"], str) and oldest["content"].startswith("Fix the breaking test: step 0.")
    assert LISTING_ELIDED in oldest["content"] and "engine.py now, as read_file() shows it" not in oldest["content"]
    assert "[harness] Condensed record of the turns that fixed step 0 (turns 1-3): the net change to engine.py and the commit" in oldest["content"]
    assert "Net change to engine.py (version 1 -> 2):" in oldest["content"]
    assert "+    moves = {1: (0, 0), 2: (0, 1 if state.vars[\"player\"].y == 1 else 0), 3: (0, 0), 4: (0, 0)}" in oldest["content"]
    assert "commit_engine (turn 3): down once\n-> Committed: steps 0-0 pass." in oldest["content"]
    assert "probe1" not in oldest["content"] and "1/0" not in oldest["content"]

    # The last three finished iterations: the failing-step message with its image, the successful calls, the diff, the commit.
    for block, (step, a, b, probe, failed, message, old, new) in zip(out[2:5], [
        (2, 4, 6, "probe2", "[E_NO_MATCH]", "down always", "1 if state.vars", "2: (0, 1), 3: (0, 0)"),
        (3, 7, 9, "probe3", "Error: unknown tool", "right too", "4: (0, 0)", "4: (1, 0)"),
        (4, 10, 12, "probe4", "Not committed", "up too", "1: (0, 0)", "1: (0, -1)"),
    ]):
        assert isinstance(block["content"], list) and _images(block) == 1
        text = _texts(block)
        assert text.startswith("Commit accepted:") and f"Step {step}:" in text
        assert f"[harness] Condensed record of the turns that fixed step {step} (turns {a}-{b}): reasoning and failed commands dropped" in text
        assert f"turn {a + 1}, python:\nprint('{probe}')\n->\n{probe}\n" in text
        assert f"turn {a + 1}: edit_file (replaced line 292 with 1 line) -> tested automatically: ALL STEPS MATCH." in text
        assert f"turn {b}, commit_engine" not in text  # the accepted commit is the block's last line, once
        assert failed not in text and "thinking at turn" not in text
        assert f"Net change to engine.py (version {step - 1 if step < 4 else 4} -> " in text or "Net change to engine.py (version" in text
        assert "-    moves = {" in text and old in text.split("-    moves = {")[1].split("\n")[0]
        assert new in text.split("+    moves = {")[1].split("\n")[0]
        assert f"commit_engine (turn {b}): {message}\n-> Committed: steps 0-" in text

    # The current iteration (step 7, turns 13-19): its message as is, turns 13-14 stripped, 15-19 untouched.
    current = out[5]
    assert _texts(current).startswith("Commit accepted: steps 0-4 pass.") and "Step 7:" in _texts(current)
    assistants = [m for m in out[5:] if m["role"] == "assistant"]
    assert len(assistants) == 7
    assert "reasoning" not in assistants[0] and "reasoning" not in assistants[1]
    assert [m["reasoning"] for m in assistants[2:]] == [f"thinking at turn {t}" for t in range(15, 20)]
    turn13 = assistants[0]
    assert [json.loads(c["function"]["arguments"])["code"] for c in turn13["tool_calls"]] == ["print('both')"]
    tools = [m for m in out[5:] if m["role"] == "tool"]
    assert tools[0]["content"] == "both\n" and not any("ZeroDivisionError" in m["content"] for m in tools)
    # Turn 14: the edit's result kept; the failed call after it kept only for the automatic test it carries.
    assert tools[1]["content"].startswith("engine.py: replaced") and "tested automatically" not in tools[1]["content"]
    assert tools[2]["content"].startswith("ValueError: after the edit\n" + FAILED_KEPT) and "tested automatically" in tools[2]["content"]
    assert len(assistants[1]["tool_calls"]) == 2
    assert [m["content"] for m in tools[3:5]] == ["fifteen\n", "sixteen\n"]
    assert "step 7 is the first failure" in tools[5]["content"]  # the run_tests report, not a failed command
    # Images: the three recent blocks keep theirs; in the current iteration only the latest set is live.
    assert sum(_images(m) for m in out) == 4
    with_images = [i for i, m in enumerate(out) if _images(m)]
    assert with_images[:3] == [2, 3, 4] and out[with_images[3]]["role"] == "user" and _texts(out[with_images[3]]).startswith("[harness] The images")
    assert IMAGE_PLACEHOLDER in _texts(current)
    assert CONTINUE not in [m.get("content") for m in out]

    # The live conversation (its older images already hidden by the agent) condenses to the same text.
    as_live = condense(live, records, versions).messages
    assert [(m["role"], _without_placeholders(m)) for m in as_live] == [(m["role"], _without_placeholders(m)) for m in out]

    # The safety cap: the stripped results cut first, then the recent blocks reduced to the older form.
    capped = condense(full, records, versions, cap_tokens=1)
    assert capped.cap and capped.cap["results_cut"] == 1 and capped.cap["blocks_reduced"] == 3
    assert capped.cap["after"] < capped.cap["before"] == condensed.estimate
    assert all(isinstance(m["content"], str) and NO_COMMIT_MESSAGE not in m["content"] for m in capped.messages[1:5])
    assert "commit_engine (turn 12): up too" in capped.messages[4]["content"] and "probe4" not in capped.messages[4]["content"]
    cut = [m for m in capped.messages if m["role"] == "tool" and "cut to 600 characters by the context cap" in m["content"]]
    assert len(cut) == 1 and "tested automatically" in cut[0]["content"][:700]

    # Fewer iterations kept in full: the three recent blocks become old ones; more turns kept: nothing stripped.
    fewer = condense(full, records, versions, keep_iterations=1).messages
    assert [isinstance(m["content"], str) for m in fewer[1:5]] == [True, True, True, False]
    whole = condense(full, records, versions, keep_turns=10).messages
    assert all("reasoning" in m for m in whole if m["role"] == "assistant") and sum(m["role"] == "tool" for m in whole) == 9


def test_a_legacy_transcript_condenses_the_same(tmp_path: Path, tiny_trace: Trace) -> None:
    """The report's loader rebuilds the older transcript format into the same full conversation."""
    from engine_re.condense_report import current_scheme, load_run, new_scheme

    live, full, records = _run(tmp_path, tiny_trace)
    log = tmp_path / "transcript.jsonl"
    new_kinds = ("message", "append", "hide_images", "compact")
    log.write_text("".join(json.dumps(r) + "\n" for r in records if not any(k in r for k in new_kinds)))
    run = load_run("tiny", tmp_path)
    assert run.turns == 19 and len(run.messages) == len(full)
    assert [(m["role"], _texts(m), _images(m)) for m in run.messages] == [(m["role"], _texts(m), _images(m)) for m in full]
    condensed = new_scheme(run)
    assert [(m["role"], _texts(m)) for m in condensed[19].messages] == [
        (m["role"], _texts(m)) for m in condense(run.before(19), run.records, run.versions_dir).messages
    ]
    # (a) replays the live scheme: the prompt of the last turn is what the model was sent (images hidden the same way).
    prompts = current_scheme(run, ModelConfig())
    sent = [(m["role"], _texts(m), _images(m)) for m in live[: len(prompts[19])]]
    assert [(m["role"], _texts(m), _images(m)) for m in prompts[19]] == sent


def test_failed_commands() -> None:
    report = "TEST RESULT (full replay)\n  Acceptance test: step 3 is the first failure; 3 steps pass before it."
    assert not failed_command(report) and not failed_command("Committed: steps 0-3 pass.") and not failed_command("42\n")
    for output in (
        'Traceback (most recent call last):\n  File "<python>", line 1\nZeroDivisionError: division by zero',
        "Error: unknown tool 'read_file'. The tools are python, run_tests and commit_engine.",
        "Error: nothing was run. This code would replace the harness's built-in edit_file.",
        "Not committed: steps 0-4 do not all pass yet, so nothing moves on. The report:\n\n" + report,
        "[E_STALE_ANCHOR] 2 stale anchors: 12#MQ, 14#ZZ. engine.py changed since.",
    ):
        assert failed_command(output), output
    # The harness's appended text does not count: an automatic test that shows a crash is not the call's error.
    appended = "ok\n\n\n[harness] engine.py changed, so it was tested automatically (run_tests with its defaults):\nTraceback (most recent call last): boom"
    assert not failed_command(appended)


def test_net_diff_is_capped_and_names_the_defs_beyond() -> None:
    old = "\n".join(["def a():", "    return 1", "", "", "def b():", "    return 2"] + [f"x{i} = {i}" for i in range(400)] + ["", "def z():", "    return 0"])
    new = old.replace("return 1", "return 10").replace("x399 = 399", "x399 = 0").replace("return 0", "return 9")
    full = net_diff(old, new, "old", "new", max_lines=10_000)
    assert full.startswith("--- old\n+++ new\n@@ ") and "+    return 10" in full and "+    return 9" in full
    capped = net_diff(old, new, "old", "new", max_lines=8)
    lines = capped.splitlines()
    assert len(lines) == 9 and lines[-1].startswith("... ") and lines[-1].endswith(" more lines (changed further down: z)")
    assert int(lines[-1].split()[1]) == len(full.splitlines()) - 8
    assert net_diff(old, old, "old", "new") == "(none: engine.py is the same at the start and at the commit)"
    assert net_diff(None, new, "old", "new") == "(engine.py versions not available for the diff)"


def test_an_iteration_whose_commit_changed_nothing(tmp_path: Path) -> None:
    from engine_re.condense import Iteration

    versions = tmp_path / "engine_versions"
    versions.mkdir()
    (versions / "v0001.py").write_text("def step(state, action):\n    pass\n")
    records = [{"turn": 0, "engine_change": {"version": 1, "op": "start"}}, {"turn": 3, "advance": {"fixed": 2, "next": 5}}]
    it = Iteration(index=0, start=1, end=9, first_turn=1, last_turn=3, finished=True, step=2)
    assert iteration_diff(it, records, versions) == "Net change to engine.py: none (version 1 at the start and at the commit)."
    assert message_text({"role": "user", "content": [{"type": "text", "text": "hello"}]}) == "hello"
