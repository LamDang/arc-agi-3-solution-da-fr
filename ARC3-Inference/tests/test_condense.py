"""engine_re.condense: the iteration condenser, on a scripted stepwise run and on its own."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from engine_re.agent import CONTINUE, IMAGE_PLACEHOLDER, Budget, EngineAgent, ModelConfig
from engine_re.condense import (
    FAILED_KEPT,
    NO_COMMIT_MESSAGE,
    RESUME_HEAD,
    annotate,
    condense,
    failed_command,
    iteration_diff,
    iterations,
    message_text,
    net_diff,
)
from engine_re.prompts import ENGINE_ELIDED, ENGINE_HEADER, KERNEL_KEEPS, KERNEL_KEEPS_NOTHING
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
BAD_LEFT = _engine("(0, -1)", "(0, 1)", "(1, 0)", "(1, 0)")  # step 7 still fails, differently (not the same failure)


class _ThinkingModel(_RecordingModel):
    """The scripted model with a reasoning field on every turn."""

    def chat(self, messages, tools):
        response = super().chat(messages, tools)
        turn = sum(m["role"] == "assistant" for m in messages) + 1  # the same text whether the run was resumed or not
        response["choices"][0]["message"]["reasoning"] = f"thinking at turn {turn}"
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
    [("no_such_tool", {})],  # turn 7: unknown tool (a python built-in called as a tool would run as python)
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


def _names_dropped(message: dict) -> str:
    return "\n".join(line for line in _texts(message).split("\n") if not line.startswith(KERNEL_KEEPS) and line != KERNEL_KEEPS_NOTHING)


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
    condensed = condense(full, records, versions, keep_turns=5)
    assert full == before  # a pure function
    assert condense(full, records, versions, keep_turns=5) == condensed  # and a deterministic one
    out = condensed.messages
    assert condensed.cap is None
    assert out[0] == full[0]  # the system prompt unchanged
    assert [m["role"] for m in out[:5]] == ["system", "user", "user", "user", "user"]

    # The oldest iteration (step 0, turns 1-3): text + the net diff + the commit; its listing elided, no images, no calls.
    oldest = out[1]
    assert isinstance(oldest["content"], str) and oldest["content"].startswith("Fix the breaking test: step 0.")
    assert ENGINE_ELIDED in oldest["content"] and ENGINE_HEADER not in oldest["content"]
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
        assert ENGINE_ELIDED in text and ENGINE_HEADER not in text  # every next-step message lists engine.py: elided here
        assert f"[harness] Condensed record of the turns that fixed step {step} (turns {a}-{b}): reasoning and failed commands dropped" in text
        assert f"turn {a + 1}, python:\nprint('{probe}')\n->\n{probe}\n" in text
        assert f"turn {a + 1}: edit_file (replaced line 292 with 1 line) -> tested automatically: ALL STEPS MATCH." in text
        assert f"turn {b}, commit_engine" not in text  # the accepted commit is the block's last line, once
        assert failed not in text and "thinking at turn" not in text
        assert f"Net change to engine.py (version {step - 1 if step < 4 else 4} -> " in text or "Net change to engine.py (version" in text
        assert "-    moves = {" in text and old in text.split("-    moves = {")[1].split("\n")[0]
        assert new in text.split("+    moves = {")[1].split("\n")[0]
        assert f"commit_engine (turn {b}): {message}\n-> Committed: steps 0-" in text

    # The current iteration (step 7, turns 13-19), keeping the last 5 turns: its message as is, turns 13-14 stripped, 15-19 untouched.
    current = out[5]
    assert _texts(current).startswith("Commit accepted: steps 0-4 pass.") and "Step 7:" in _texts(current)
    assert ENGINE_HEADER in _texts(current) and ENGINE_ELIDED not in _texts(current)  # the latest listing stays
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
    as_live = condense(live, records, versions, keep_turns=5).messages
    assert [(m["role"], _without_placeholders(m)) for m in as_live] == [(m["role"], _without_placeholders(m)) for m in out]

    # The safety cap: the stripped results cut first, then the recent blocks reduced to the older form.
    capped = condense(full, records, versions, keep_turns=5, cap_tokens=1)
    assert capped.cap and capped.cap["results_cut"] == 1 and capped.cap["blocks_reduced"] == 3
    assert capped.cap["after"] < capped.cap["before"] == condensed.estimate
    assert all(isinstance(m["content"], str) and NO_COMMIT_MESSAGE not in m["content"] for m in capped.messages[1:5])
    assert "commit_engine (turn 12): up too" in capped.messages[4]["content"] and "probe4" not in capped.messages[4]["content"]
    cut = [m for m in capped.messages if m["role"] == "tool" and "cut to 600 characters by the context cap" in m["content"]]
    assert len(cut) == 1 and "tested automatically" in cut[0]["content"][:700]

    # Fewer iterations kept in full: the three recent blocks become old ones; more turns kept: nothing stripped.
    fewer = condense(full, records, versions, keep_iterations=1, keep_turns=5).messages
    assert [isinstance(m["content"], str) for m in fewer[1:5]] == [True, True, True, False]
    whole = condense(full, records, versions).messages  # the default keeps the last 10 turns whole
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
    # The same but for the kernel-names line of the next-step messages, which the older records do not keep.
    assert [(m["role"], _names_dropped(m), _images(m)) for m in run.messages] == [(m["role"], _names_dropped(m), _images(m)) for m in full]
    assert sum(ENGINE_HEADER in _texts(m) for m in run.messages) == 5
    condensed = new_scheme(run)
    assert [(m["role"], _texts(m)) for m in condensed[19].messages] == [
        (m["role"], _texts(m)) for m in condense(run.before(19), run.records, run.versions_dir).messages
    ]
    # (a) replays the live scheme: the prompt of the last turn is what the model was sent (images hidden the same way).
    prompts = current_scheme(run, ModelConfig())
    sent = [(m["role"], _names_dropped(m), _images(m)) for m in live[: len(prompts[19])]]
    assert [(m["role"], _names_dropped(m), _images(m)) for m in prompts[19]] == sent


def test_failed_commands() -> None:
    report = "TEST RESULT (full replay)\n  Acceptance test: step 3 is the first failure; 3 steps pass before it."
    assert not failed_command(report) and not failed_command("Committed: steps 0-3 pass.") and not failed_command("42\n")
    for output in (
        'Traceback (most recent call last):\n  File "<python>", line 1\nZeroDivisionError: division by zero',
        "Error: unknown tool 'read_file'. The tools are python, run_tests and commit_engine.",
        "Error: nothing was run. This code would replace the harness's built-in edit_file.",
        "Not committed: steps 0-4 do not all pass yet, so nothing moves on. The report:\n\n" + report,
        "[E_STALE_ANCHOR] 2 stale anchors: 12#MQ, 14#ZZ. engine.py changed since.",
        "engine.py was not changed.\n[E_NO_MATCH] Edit 1: replace_text found no exact match.",
        "[harness] edit_file is a python function, not a tool; this call ran as python: edit_file(edits=[])\n"
        "engine.py was not changed.\n[E_BAD_OP] edits must be a non-empty list.",
    ):
        assert failed_command(output), output
    # An edit that applied some of its edits is not a failed command (the lenient edit tool reports the rest).
    assert not failed_command("engine.py: applied 1 of 2 edits: replaced line 3 with 1 line. Syntax OK. (version 4)\n[E_NO_MATCH] Edit 2: no match.")
    # The harness's appended text does not count: an automatic test that shows a crash is not the call's error.
    appended = "ok\n\n\n[harness] engine.py changed, so it was tested automatically (run_tests with its defaults):\nTraceback (most recent call last): boom"
    assert not failed_command(appended)


def test_edit_one_liners_carry_the_automatic_tests_verdict() -> None:
    from engine_re.condense import Note, _edit_line

    note = Note(turn=14, iteration=4, kind="tool", edits=[{"op": "edit", "summary": "replaced line 292 with 1 line"}])
    same = "engine.py: replaced line 292 with 1 line. Syntax OK.\n\n[harness] engine.py changed, tested automatically: the same result as the last test (step 7 fails the same way: final frame: 2 px differ in 1 region(s))."
    assert _edit_line(note, same) == "turn 14: edit_file (replaced line 292 with 1 line) -> tested automatically: the same result as the last test (step 7 fails the same way: final frame: 2 px differ in 1 region(s))"
    report = "engine.py: replaced line 292 with 1 line. Syntax OK.\n\n[harness] engine.py changed, so it was tested automatically (run_tests with its defaults):\nTEST RESULT\n  Contract tests: 5/5 pass.\n  Acceptance test (the recording replayed in order): ALL STEPS MATCH.\n\nSteps 0-7 pass."
    assert _edit_line(note, report) == "turn 14: edit_file (replaced line 292 with 1 line) -> tested automatically: ALL STEPS MATCH."
    assert _edit_line(note, "engine.py: replaced line 292 with 1 line. Syntax OK.") == "turn 14: edit_file (replaced line 292 with 1 line)"


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


def _is_resume_note(message: dict) -> bool:
    return message["role"] == "user" and isinstance(message["content"], str) and message["content"].startswith(RESUME_HEAD)


class _OverModel(_RecordingModel):
    """The scripted model with a reasoning field on every turn, whose answers report a prompt over the compaction
    threshold at the given turns (counted from `first_turn` for a resumed run)."""

    def __init__(self, turns, over=(), first_turn: int = 1):
        super().__init__(turns)
        self.over, self.first_turn = set(over), first_turn

    def chat(self, messages, tools):
        turn = self.first_turn + len(getattr(self, "calls", []))
        response = super().chat(messages, tools)
        response["choices"][0]["message"]["reasoning"] = f"thinking at turn {turn}"
        response["usage"]["prompt_tokens"] = 10**6 if turn in self.over else 10
        return response


def _wire(messages: list[dict]) -> list[dict]:
    """Messages as a request sends them (what _RecordingModel.calls keeps)."""
    return json.loads(json.dumps(messages))


def _before_turn(full: list[dict], turn: int) -> list[dict]:
    """The full conversation as it stood when turn `turn` was asked."""
    return full[: [i for i, m in enumerate(full) if m["role"] == "assistant"][turn - 1]]


def _condense_agent(folder: Path, trace: Trace, script, over=(), first_turn: int = 1, max_turns: int = 19):
    trace.save(folder / "trace")
    model = _OverModel(copy.deepcopy(script), over, first_turn)
    agent = EngineAgent("tiny", folder, ModelConfig(context="condense"), Budget(max_turns=max_turns), client=model, stepwise=True)
    return agent, model, agent.run()


def test_before_the_condenser_fires_the_whole_conversation_is_sent(tmp_path: Path, tiny_trace: Trace) -> None:
    """context="condense" below the threshold: every request is the full conversation, only the latest images live."""
    from engine_re.condense import hide_but_latest

    agent, model, result = _condense_agent(tmp_path, tiny_trace, SCRIPT)
    assert result.status == "budget_turns" and result.step == 7 and result.context == "condense"
    assert json.loads((tmp_path / "result.json").read_text())["context"] == "condense"
    full = agent.messages
    assert all("reasoning" in m for m in full if m["role"] == "assistant") and sum(m["role"] == "assistant" for m in full) == 19
    assert not any(IMAGE_PLACEHOLDER in _texts(m) for m in full) and sum(_images(m) for m in full) >= 5  # every image live
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert not any("compact" in r or "hide_images" in r or "condense" in r for r in records)
    assert agent.records == records
    assert model.calls[0] == _wire(full[:2])  # the first request: the system prompt and the first message
    for turn, sent in enumerate(model.calls, 1):
        before = _before_turn(full, turn)
        assert sent == _wire(hide_but_latest(list(before)))
        # The conversation itself: unchanged but for the images of all but the latest message that has some.
        assert [(m["role"], _without_placeholders(m)) for m in sent] == [(m["role"], _without_placeholders(m)) for m in _wire(before)]
        assert sum(1 for m in sent if _images(m)) <= 1
    assert agent._context() == hide_but_latest(list(full)) and agent._condensed == [] and agent._condensed_from == 0


def test_the_condenser_fires_over_the_threshold_and_keeps_its_prefix(tmp_path: Path, tiny_trace: Trace) -> None:
    """Over the threshold the whole conversation is condensed once; the turns after it are appended as they are, the
    condensed prefix byte-identical, until the next firing recomputes it from the full conversation."""
    from engine_re.condense import hide_but_latest, measure

    agent, model, result = _condense_agent(tmp_path, tiny_trace, SCRIPT, over=(8, 14))
    assert result.status == "budget_turns" and result.step == 7
    full = agent.messages  # never shortened: every reasoning and image
    assert all("reasoning" in m for m in full if m["role"] == "assistant") and sum(m["role"] == "assistant" for m in full) == 19
    assert not any(IMAGE_PLACEHOLDER in _texts(m) for m in full)
    records = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert agent.records == records and not any("compact" in r or "hide_images" in r for r in records)
    fired = [i for i, r in enumerate(records) if "condense" in r]
    assert [records[i]["turn"] for i in fired] == [8, 14]  # one record per firing, after the turn that went over
    assert all(set(records[i]["condense"]) == {"tokens", "chars", "images", "messages"} for i in fired)

    versions = tmp_path / "engine_versions"
    views = []
    for i, turn in zip(fired, (8, 14)):
        covered = _before_turn(full, turn + 1)  # everything up to the end of the turn that went over
        view = condense(covered, records[:i], versions, keep_turns=10, chars_per_token=3)
        logged = records[i]["condense"]
        chars, images = measure(view.messages)
        assert logged == {"tokens": view.estimate, "chars": chars, "images": images, "messages": len(view.messages)}
        assert logged["tokens"] == chars // 3 + images * 1000
        views.append((len(covered), _wire(view.messages)))

    # Turns 1-8: the full conversation. Turns 9-14: the first view, then the messages since as they are.
    # Turns 15-19: the second view, recomputed from the full conversation (not from the first view), then the same.
    for turn, sent in enumerate(model.calls, 1):
        before = _before_turn(full, turn)
        covered, prefix = (0, []) if turn <= 8 else views[0] if turn <= 14 else views[1]
        assert sent[: len(prefix)] == prefix
        assert sent[len(prefix) :] == _wire(hide_but_latest(before[covered:]))
    first, second = views[0][1], views[1][1]
    assert first != second and len(views[0][1]) < len(_wire(_before_turn(full, 9)))
    assert json.dumps(model.calls[9][: len(first)]) == json.dumps(model.calls[13][: len(first)])  # byte-identical
    # The first firing (turn 8) was inside the iteration of step 3: it is the current one in the first view, with the
    # reasoning of all its turns (fewer than 10); the second view has it as a finished block, its probe call kept.
    assert any(m.get("reasoning") == "thinking at turn 8" for m in first)
    assert not any("thinking at turn 8" == m.get("reasoning") for m in second)
    assert "fixed step 3 (turns 7-9): reasoning and failed commands dropped" in "\n".join(_texts(m) for m in second if m["role"] == "user")
    assert "turn 8, python:\nprint('probe3')" in "\n".join(_texts(m) for m in second if m["role"] == "user")
    # The agent's own state: the second view and how many messages it covers; its next request is that view plus the rest.
    assert _wire(agent._condensed) == second and agent._condensed_from == views[1][0]
    assert _wire(agent._context()) == second + _wire(hide_but_latest(full[views[1][0] :]))


def test_a_resumed_run_sends_what_the_uninterrupted_one_did(tmp_path: Path, tiny_trace: Trace) -> None:
    """The condensed view is rebuilt from the transcript: at the last "condense" record, over the conversation and the
    records up to it. A resumed run sends the same requests as an uninterrupted one, but for the resume note."""
    whole, part = tmp_path / "whole", tmp_path / "part"
    for folder in (whole, part):
        folder.mkdir()
    uninterrupted, reference, _ = _condense_agent(whole, tiny_trace, SCRIPT, over=(8, 14))
    # Interrupted after turn 12 (the condenser fired after turn 8), resumed for the last seven turns (it fires after 14).
    first, _, result = _condense_agent(part, tiny_trace, SCRIPT[:12], over=(8,), max_turns=12)
    assert result.status == "budget_turns" and result.step == 7 and first._condensed_from > 0
    rebuilt = EngineAgent("tiny", part, ModelConfig(context="condense"), Budget(), client=_ScriptedModel([]), stepwise=True)
    state = rebuilt._rebuild_conversation()
    assert state["messages"] == first.messages
    assert rebuilt._condensed_from == first._condensed_from and rebuilt._condensed == first._condensed
    assert rebuilt._context() == first._context()  # the request the interrupted run would have sent next
    # Without context "condense" the record is ignored: the conversation is rebuilt as it is.
    plain = EngineAgent("tiny", part, ModelConfig(), Budget(), client=_ScriptedModel([]), stepwise=True)
    assert plain._rebuild_conversation()["messages"] == first.messages and plain._condensed == []

    second, model, result = _condense_agent(part, tiny_trace, SCRIPT[12:], over=(14,), first_turn=13)
    assert result.status == "budget_turns" and result.turns == 19 and result.resumes == 1 and result.context == "condense"
    assert [m for m in second.messages if not _is_resume_note(m)] == uninterrupted.messages
    records = [json.loads(line) for line in (part / "transcript.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in records if "condense" in r] == [8, 14]
    for sent, expected in zip(model.calls, reference.calls[12:]):
        assert [m for m in sent if not _is_resume_note(m)] == expected
        assert sum(_is_resume_note(m) for m in sent) == 1
    again = EngineAgent("tiny", part, ModelConfig(context="condense"), Budget(), client=_ScriptedModel([]), stepwise=True)
    again._rebuild_conversation()
    assert again._condensed == second._condensed and again._context() == second._context()


def test_an_older_transcript_resumed_with_the_condenser(tmp_path: Path, tiny_trace: Trace) -> None:
    """A transcript from before "message" records whose last request was over the threshold: written back in full, then
    the condenser fires once, so its record follows the messages it covers."""
    agent, _, _ = _condense_agent(tmp_path, tiny_trace, SCRIPT[:12], max_turns=12)
    log = tmp_path / "transcript.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    for r in records:
        if r.get("turn") == 12 and "finish_reason" in r:
            r["usage"]["prompt_tokens"] = 10**6
    log.write_text("".join(json.dumps(r) + "\n" for r in records if not any(k in r for k in ("message", "append"))))
    second, model, result = _condense_agent(tmp_path, tiny_trace, SCRIPT[12:13], first_turn=13, max_turns=13)
    assert result.turns == 13 and result.resumes == 1
    records = [json.loads(line) for line in log.read_text().splitlines()]
    rebased = next(i for i, r in enumerate(records) if "rebased" in r)
    fired = [i for i, r in enumerate(records) if "condense" in r]
    assert len(fired) == 1 and fired[0] > rebased and all("message" in r for r in records[rebased + 1 : fired[0]])
    covered = sum("message" in r for r in records[rebased + 1 : fired[0]])
    prefix = _wire(condense(second.messages[:covered], records[: fired[0]], tmp_path / "engine_versions", chars_per_token=3).messages)
    sent = model.calls[0]
    assert sent[: len(prefix)] == prefix and len(prefix) < covered and _is_resume_note(sent[len(prefix)]) and len(sent) == len(prefix) + 1



def test_the_report_compares_the_schemes_on_a_logged_run(tmp_path: Path, tiny_trace: Trace) -> None:
    """condense_report --schemes: a "compact" run's prompts replayed from its records, the compaction simulated the same,
    and the threshold condenser firing where the compaction did."""
    from engine_re.condense_report import compare_schemes, load_any

    tiny_trace.save(tmp_path / "trace")
    model = _OverModel(copy.deepcopy(SCRIPT), over=(8, 14))
    agent = EngineAgent("tiny", tmp_path, ModelConfig(), Budget(max_turns=19), client=model, stepwise=True)
    agent.run()
    run = load_any("tiny", tmp_path)
    assert run.turns == 19 and run.prompt_tokens[8] == 10**6
    assert [m["role"] for m in run.messages] == [m["role"] for m in agent.messages]
    res = compare_schemes(run, 140_000, 10)
    assert res["logged"] == "compaction" and res["check"] == {"compaction_simulation_matches_log": True}
    schemes = res["schemes"]
    assert schemes["compaction"]["fired_turns"] == [8, 14] == schemes["threshold, as compaction"]["fired_turns"]
    assert schemes["per-turn"]["fired"] == 19 and schemes["threshold, as compaction"]["rewritten"] <= 2
    assert schemes["compaction"]["real_total"] == sum(10 if t not in (8, 14) else 10**6 for t in range(1, 20))
