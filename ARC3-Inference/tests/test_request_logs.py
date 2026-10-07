"""Request logs: reply-only response lines, xz compression at game end, readers of both forms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
from inference.agent.tool_agent import _append_request_snapshot, _history_reasoning_details
from inference.framework.solver import HarnessSolver
from inference.utils.openai_compat import assemble_streamed_chat_response, build_chat_payload
from inference.utils.run_artifacts import compress_log, existing_log, open_log
from viewer import data as viewer_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import token_breakdown  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "rules"},
    {"role": "user", "content": "Current state: step 1, level 1."},
]
REPLY = {
    "role": "assistant",
    "content": "",
    "reasoning": "Read the win condition first.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": "read_game_code(1, 9)"})},
        }
    ],
}
USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "completion_tokens_details": {"reasoning_tokens": 8},
    "cost": 0.001,
}


def _write_exchange(path: Path) -> None:
    common = {"analysis_step": 1, "action": 1, "request_index_within_turn": 1}
    _append_request_snapshot(path, messages=MESSAGES, tools=[], event="request", **common)
    _append_request_snapshot(
        path, messages=None, tools=None, event="response", reply=REPLY, usage=USAGE, **common
    )


def test_response_line_holds_the_reply_not_the_request(tmp_path: Path) -> None:
    log = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    _write_exchange(log)

    request, response = (json.loads(line) for line in log.read_text().splitlines())

    assert request["messages"] == MESSAGES
    assert "messages" not in response and "tools" not in response
    assert response["reply"] == REPLY
    assert response["usage"] == USAGE


def test_compressed_log_reads_back_identically(tmp_path: Path) -> None:
    log = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    _write_exchange(log)
    text = log.read_text(encoding="utf-8")

    compressed = compress_log(log)

    assert compressed == tmp_path / "ls20-9607627b_p0_requests.jsonl.xz"
    assert not log.exists()
    assert existing_log(log) == compressed
    with open_log(compressed) as handle:
        assert handle.read() == text
    assert compress_log(log) is None


def test_solver_compresses_only_a_game_runs_own_log(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    state = artifacts / f"ls20-9607627b_p0_{RUNTIME_STATE_FILENAME}"
    own = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    _write_exchange(own)
    # a state file outside artifacts/ logs to the shared run-level file
    shared = tmp_path / "standalone" / "requests.jsonl"
    shared.parent.mkdir()
    _write_exchange(shared)

    HarnessSolver._compress_request_log(None, state)
    HarnessSolver._compress_request_log(None, shared.parent / RUNTIME_STATE_FILENAME)

    assert existing_log(own) == own.with_name(own.name + ".xz")
    assert existing_log(shared) == shared


def test_viewer_reads_a_compressed_log_and_shows_the_request(tmp_path: Path) -> None:
    log = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    _write_exchange(log)
    compress_log(log)

    found = viewer_data._resolve_request_log_path(
        run_dir=tmp_path,
        viewer_data_path=tmp_path / "artifacts" / "ls20-9607627b_p0_viewer_data.json",
        game_id="ls20-9607627b",
    )
    snapshots = viewer_data._load_request_snapshots(found)
    latest = viewer_data._latest_request_snapshot(snapshots, analysis_step=1)

    assert found == tmp_path / "ls20-9607627b_p0_requests.jsonl.xz"
    assert [s["event"] for s in snapshots] == ["request", "response"]
    assert latest is not None and latest["messages"] == MESSAGES
    assert viewer_data._analysis_step_usage(snapshots, 1)["completionTokens"] == 20


def test_breakdown_takes_the_reply_from_the_response_line(tmp_path: Path) -> None:
    log = tmp_path / "ls20-9607627b_p0_requests.jsonl"
    _write_exchange(log)
    compress_log(log)

    (response,) = token_breakdown.load_run(tmp_path)

    assert response.matched
    assert response.reasoning == "Read the win condition first."
    assert response.reasoning_tokens == 8 and response.tool_call_tokens == 12
    assert response.code_read_calls == 1


def test_streamed_reply_keeps_the_reasoning_summary_and_encrypted_block() -> None:
    def chunk(**delta: object) -> bytes:
        return ("data: " + json.dumps({"id": "gen-1", "choices": [{"delta": delta}]})).encode()

    summary = {"type": "reasoning.summary", "format": "openai-responses-v1", "index": 0}
    lines = [
        chunk(reasoning="Check ", reasoning_details=[{**summary, "summary": "Check "}]),
        chunk(reasoning="the rules.", reasoning_details=[{**summary, "summary": "the rules."}]),
        chunk(reasoning_details=[{"type": "reasoning.encrypted", "data": "gAAA", "index": 0}]),
        chunk(content="done"),
        b"data: [DONE]",
    ]
    reply = assemble_streamed_chat_response(lines)
    message = reply["choices"][0]["message"]
    assert message["reasoning"] == "Check the rules."
    assert message["reasoning_details"] == [
        {**summary, "summary": "Check the rules."},
        {"type": "reasoning.encrypted", "data": "gAAA", "index": 0},
    ]
    assert reply["id"] == "gen-1"


def test_openrouter_payload_sends_the_reasoning_effort(monkeypatch) -> None:
    def payload(thinking: bool) -> dict:
        return build_chat_payload(
            provider="openrouter", model="m", messages=MESSAGES, max_tokens=10,
            temperature=0.7, top_p=0.95, top_k=20, thinking=thinking,
        )

    assert payload(True)["reasoning"] == {"enabled": True}
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "xhigh")
    assert payload(True)["reasoning"] == {"enabled": True, "effort": "xhigh"}
    assert payload(False)["reasoning"] == {"enabled": False}


def test_response_line_records_the_request_settings(tmp_path: Path) -> None:
    log = tmp_path / "game_requests.jsonl"
    _append_request_snapshot(
        log, messages=None, tools=None, event="response", reply=REPLY, usage=USAGE,
        request_params={"model": "m", "reasoning": {"enabled": True, "effort": "xhigh"}},
        response_id="gen-1",
    )
    line = json.loads(log.read_text())
    assert line["request_params"]["reasoning"]["effort"] == "xhigh"
    assert line["response_id"] == "gen-1"


def test_reasoning_details_go_back_in_history_only_when_asked(monkeypatch) -> None:
    reply = {**REPLY, "reasoning_details": [{"type": "reasoning.encrypted", "data": "gAAA", "index": 0}]}
    assert _history_reasoning_details(reply) is None
    monkeypatch.setenv("ARC3_SEND_REASONING_DETAILS", "1")
    assert _history_reasoning_details(reply) == reply["reasoning_details"]
    assert _history_reasoning_details(REPLY) is None
    monkeypatch.setenv("OPENROUTER_REASONING_CONTEXT", "all_turns")
    payload = build_chat_payload(
        provider="openrouter", model="m", messages=MESSAGES, max_tokens=10,
        temperature=0.7, top_p=0.95, top_k=20, thinking=True,
    )
    assert payload["reasoning"]["context"] == "all_turns"
