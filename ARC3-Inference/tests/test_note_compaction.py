"""Note compaction: at a turn start past ARC3_NOTE_COMPACTION_TOKENS, ask for a
comment-only python note, then keep the last turns and the note exchange."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference.agent import tool_agent
from inference.agent.prompts import NOTE_COMPACTION_TOOL_RESULT
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult, _prune_control_messages

NOTE = "# RULES:\n# CONFIRMED: RIGHT extends the rail by one cell.\n" + "# TRIED: clicking (27,38) did nothing.\n" * 20


def _turn(step: int) -> list[dict]:
    return [
        {"role": "user", "content": f"Current state: step {step}, level 1."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": f"c{step}", "type": "function", "function": {"name": "python", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": f"c{step}", "content": "ok"},
    ]


def _agent(history: list[dict], reply: dict) -> ToolAgent:
    agent = object.__new__(ToolAgent)
    agent._system_prompt = "rules"
    agent._history_messages = list(history)
    agent._save_request_logs = False
    agent._session_generated_tokens = 0
    agent._turn_generated_tokens = 0
    agent._session_total_tokens = 0
    agent._context_was_trimmed = False
    agent._has_evicted = False
    agent.sent = []
    # 1,000 tokens a message: the threshold is crossed by message count
    agent._estimate_request_input_tokens = lambda messages, tools=None: 1000 * len(messages)

    def chat(messages, **kwargs):
        agent.sent.append(messages)
        return _ChatCompletionResult(message=reply, finish_reason="tool_calls", usage={"completion_tokens": 400})

    agent._chat_completion = chat
    return agent


def _compact(agent: ToolAgent, transcript: list) -> None:
    agent._maybe_compact_with_note(
        {"role": "user", "content": "Current state: step 31, level 1."},
        [],
        lambda label, text: transcript.append((label, text)),
        state_path=Path("unused"),
        analysis_step=31,
        display_action_num=31,
    )


def _note_reply(code: str = NOTE) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "note1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}
        ],
    }


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ARC3_NOTE_COMPACTION_TOKENS", "60000")
    monkeypatch.setenv("ARC3_NOTE_COMPACTION_KEEP_TURNS", "5")


def test_keeps_last_turns_then_the_note(env) -> None:
    history = [m for step in range(1, 31) for m in _turn(step)]  # 90 messages
    agent = _agent(history, _note_reply())
    transcript: list = []

    _compact(agent, transcript)

    (sent,) = agent.sent
    assert sent[:-1] == [{"role": "system", "content": "rules"}, *history]
    assert "last 5 turns (from game step 26 on)" in sent[-1]["content"]
    assert "about 60K tokens" in sent[-1]["content"]
    kept = agent._history_messages
    assert kept[:15] == history[-15:]
    request, reply, result = kept[15:]
    assert reply["tool_calls"][0]["function"]["arguments"] == json.dumps({"code": NOTE})
    assert result == {
        "role": "tool", "tool_call_id": "note1", "content": NOTE_COMPACTION_TOOL_RESULT,
        tool_agent._CONTROL_MESSAGE_KEY: "note",
    }
    assert agent._has_evicted and agent._context_was_trimmed
    assert agent._session_generated_tokens == 400 and agent._turn_generated_tokens == 0
    assert ("COMPACTION NOTE", NOTE) in transcript


def test_below_threshold_or_too_few_turns_does_nothing(env, monkeypatch) -> None:
    short = [m for step in range(1, 6) for m in _turn(step)]  # 5 turns, all kept anyway
    agent = _agent(short, _note_reply())
    agent._estimate_request_input_tokens = lambda messages, tools=None: 10**6
    _compact(agent, [])
    assert agent.sent == [] and agent._history_messages == short

    monkeypatch.setenv("ARC3_NOTE_COMPACTION_TOKENS", "0")
    long = [m for step in range(1, 31) for m in _turn(step)]
    agent = _agent(long, _note_reply())
    agent._estimate_request_input_tokens = lambda messages, tools=None: 10**6
    _compact(agent, [])
    assert agent.sent == [] and agent._history_messages == long


def test_keeps_fewer_turns_when_they_would_stay_near_the_threshold(env, monkeypatch) -> None:
    # 5 turns + system + opener = 17K, over 3/4 of 20K: one turn fewer (14K)
    monkeypatch.setenv("ARC3_NOTE_COMPACTION_TOKENS", "20000")
    history = [m for step in range(1, 31) for m in _turn(step)]
    agent = _agent(history, _note_reply())

    _compact(agent, [])

    assert "last 4 turns (from game step 27 on)" in agent.sent[0][-1]["content"]
    assert agent._history_messages[:12] == history[-12:]
    assert len(agent._history_messages) == 15


def test_unusable_note_leaves_history_to_the_trimmer(env) -> None:
    history = [m for step in range(1, 31) for m in _turn(step)]
    agent = _agent(history, _note_reply("# ok"))
    transcript: list = []

    _compact(agent, transcript)

    assert agent._history_messages == history
    assert not agent._has_evicted
    assert any("note_compaction_rejected" in text for _, text in transcript)


def test_earlier_note_is_replaced_and_never_pruned(env, monkeypatch) -> None:
    history = [m for step in range(1, 31) for m in _turn(step)]
    agent = _agent(history, _note_reply())
    _compact(agent, [])
    first = list(agent._history_messages)
    # more turns, then a second compaction
    agent._history_messages = [*first, *[m for step in range(31, 40) for m in _turn(step)]]
    _compact(agent, [])

    kinds = [m.get(tool_agent._CONTROL_MESSAGE_KEY) for m in agent._history_messages]
    assert kinds.count("note") == 3  # only the new exchange
    monkeypatch.setenv("ARC3_PRUNE_CONTROL_CONTEXT", "1")
    assert _prune_control_messages(agent._history_messages) == agent._history_messages
