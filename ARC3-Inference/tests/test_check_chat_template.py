"""scripts/check_chat_template.py: prefix check, pitfalls, and reading the
harness's request logs. Uses a cut-down template with the Qwen3.8-Flash-Next
behaviours that matter (reasoning_content, arguments|items, preserve_thinking,
a generation prompt that opens <think>) so no download is needed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from inference.agent.tool_agent import _append_request_snapshot

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_chat_template.py"
spec = importlib.util.spec_from_file_location("check_chat_template", SCRIPT)
cct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cct)

TEMPLATE = """\
{%- set ns = namespace(last_query_index=0) %}
{%- for message in messages %}
    {%- if message.role == 'user' %}{%- set ns.last_query_index = loop.index0 %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.role == 'system' or message.role == 'user' %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
    {%- elif message.role == 'assistant' %}
        {%- set reasoning = message.reasoning_content if message.reasoning_content is string else '' %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>assistant\\n<think>\\n' + reasoning + '\\n</think>\\n\\n' + (message.content or '') }}
        {%- else %}
            {{- '<|im_start|>assistant\\n' + (message.content or '') }}
        {%- endif %}
        {%- for call in message.tool_calls or [] %}
            {{- '<tool_call>\\n<function=' + call.function.name + '>\\n' }}
            {%- for name, value in call.function.arguments|items %}
                {{- '<parameter=' + name + '>\\n' + value + '\\n</parameter>\\n' }}
            {%- endfor %}
            {{- '</function>\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == 'tool' %}
        {{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}
"""


@pytest.fixture
def template():
    return cct.compile_template(TEMPLATE)


def test_sample_targets_with_preserve_thinking(template):
    targets = cct.assistant_targets(template, cct.SAMPLE_MESSAGES, cct.SAMPLE_TOOLS,
                                    {"preserve_thinking": True})
    assert [i for i, _ in targets] == [2, 4, 7]
    assert all(text is not None for _, text in targets)
    assert targets[0][1].startswith("The 2 is the player.")
    assert targets[0][1].endswith("</tool_call><|im_end|>\n")


def test_prefix_breaks_without_preserve_thinking(template):
    targets = dict(cct.assistant_targets(template, cct.SAMPLE_MESSAGES, cct.SAMPLE_TOOLS,
                                         {"preserve_thinking": False}))
    # Turns before the last user message lose their thinking in later renders.
    assert targets[2] is None and targets[4] is None
    assert targets[7] is not None


def test_pitfalls(template):
    found = cct.pitfalls(template, cct.SAMPLE_MESSAGES, cct.SAMPLE_TOOLS, {"preserve_thinking": True})
    assert found["reasoning under `reasoning` dropped"] is True
    assert found["arguments as a JSON string"].startswith("raises")
    assert found["generation prompt ends with"].endswith("<|im_start|>assistant\n<think>\n")


def test_normalize_message():
    message = {"role": "assistant", "reasoning": "think", "content": "",
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "python", "arguments": '{"code": "x"}'}}]}
    out = cct.normalize_message(message)
    assert out["reasoning_content"] == "think"
    assert out["tool_calls"][0]["function"]["arguments"] == {"code": "x"}
    assert message["tool_calls"][0]["function"]["arguments"] == '{"code": "x"}'
    with pytest.raises(ValueError):
        cct.normalize_message({**message, "tool_calls": [
            {"function": {"name": "python", "arguments": '{"code": "x'}}]})


def _reply(reasoning: str, code: str) -> dict:
    return {"role": "assistant", "content": "", "reasoning": reasoning,
            "tool_calls": [{"id": "c", "type": "function",
                            "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}


def test_request_log(template, tmp_path, capsys):
    log = tmp_path / "ls20_p0_requests.jsonl"
    history = [{"role": "system", "content": "Play."}, {"role": "user", "content": "Frame 1"}]
    kwargs = {"preserve_thinking": True}
    for step in range(3):
        reply = _reply(f"thinking {step}", f"act({step})")
        _append_request_snapshot(log, messages=history, tools=cct.SAMPLE_TOOLS,
                                 event="request", chat_template_kwargs=kwargs)
        _append_request_snapshot(log, messages=None, tools=None, event="response", reply=reply)
        history = history + [reply, {"role": "tool", "content": "ok"},
                             {"role": "user", "content": f"Frame {step + 2}"}]
    # A truncated reply: its arguments are not valid JSON.
    _append_request_snapshot(log, messages=history, tools=cct.SAMPLE_TOOLS,
                             event="request", chat_template_kwargs=kwargs)
    bad = _reply("cut", "x")
    bad["tool_calls"][0]["function"]["arguments"] = '{"code": "act('
    _append_request_snapshot(log, messages=None, tools=None, event="response", reply=bad)

    assert cct.check_request_log(template, log, None) is False
    out = capsys.readouterr().out
    assert "requests: 4" in out
    assert "ok: 3" in out
    assert "bad arguments: 1" in out
    assert "extends previous: 2" in out

    assert cct.check_request_log(template, log, {"preserve_thinking": False}) is False
    assert "prefix broken: 0" in capsys.readouterr().out  # one reply per request: nothing to drop
