"""The OpenAI Responses API adapter: chat payload out, chat-shaped reply back,
encrypted reasoning carried through history and kept out of the context estimate."""
from __future__ import annotations

import json

from inference.agent.tool_agent import _split_reasoning_details_for_estimate
from inference.utils.openai_compat import (
    assemble_streamed_responses,
    build_chat_payload,
    chat_response_from_responses,
    normalize_provider,
    responses_payload_from_chat,
)

IMAGE = "data:image/png;base64,iVBORw0KGgo="
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    }
]
RESPONSE = {
    "id": "resp_1",
    "status": "completed",
    "output": [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Try the left lever."}],
            "encrypted_content": "gAAAA-secret",
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "python",
            "arguments": '{"code": "step(1)"}',
        },
    ],
    "usage": {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100},
        "output_tokens": 300,
        "output_tokens_details": {"reasoning_tokens": 240},
        "total_tokens": 1300,
    },
}


def _chat_payload(messages, **kwargs):
    return build_chat_payload(
        provider="openai-responses", model="gpt-6.1-sol", messages=messages, max_tokens=12288,
        temperature=0.7, top_p=0.95, top_k=20, thinking=True, tools=TOOLS, tool_choice="auto",
        **kwargs,
    )


def test_provider_name() -> None:
    assert normalize_provider("openai-responses") == "openai-responses"
    assert normalize_provider("openai") == "vllm"


def test_reply_comes_back_in_chat_shape(monkeypatch) -> None:
    monkeypatch.setenv("ARC3_OPENAI_PRICING", "2,0.1,2.5,10")
    reply = chat_response_from_responses(RESPONSE)
    choice = reply["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["reasoning"] == "Try the left lever."
    assert message["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "python", "arguments": '{"code": "step(1)"}'}}
    ]
    [detail] = message["reasoning_details"]
    assert detail["data"] == "gAAAA-secret" and detail["tokens"] == 240
    usage = reply["usage"]
    assert usage["prompt_tokens"] == 1000 and usage["completion_tokens"] == 300
    assert usage["prompt_tokens_details"]["cached_tokens"] == 600
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 240
    # 300 uncached at $2, 600 cached at $0.10, 100 written at $2.50, 300 out at $10
    assert abs(usage["cost"] - (300 * 2 + 600 * 0.1 + 100 * 2.5 + 300 * 10) / 1e6) < 1e-12
    assert reply["id"] == "resp_1"


def test_cut_off_reply_reads_as_length() -> None:
    reply = chat_response_from_responses(
        {**RESPONSE, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
         "output": RESPONSE["output"][:1]}
    )
    assert reply["choices"][0]["finish_reason"] == "length"
    assert chat_response_from_responses({"status": "failed", "error": {"code": "x"}})["choices"] == []


def test_history_goes_out_as_responses_input(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    reply = chat_response_from_responses(RESPONSE)["choices"][0]["message"]
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "text", "text": "Board:"}, {"type": "image_url", "image_url": {"url": IMAGE}}]},
        {**reply, "content": None},
        {"role": "tool", "tool_call_id": "call_1", "content": "moved"},
        # cut off while reasoning: nothing for its reasoning to precede
        {"role": "assistant", "content": "", "reasoning_details": reply["reasoning_details"]},
        {"role": "user", "content": "go on"},
    ]
    payload = responses_payload_from_chat(_chat_payload(messages))
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert payload["max_output_tokens"] == 12288
    assert not {"temperature", "top_p", "top_k", "messages", "max_tokens"} & set(payload)
    assert payload["tools"] == [
        {"type": "function", "name": "python", "description": "Run code.",
         "parameters": TOOLS[0]["function"]["parameters"], "strict": False}
    ]
    assert payload["input"] == [
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Board:"},
            {"type": "input_image", "image_url": IMAGE, "detail": "auto"},
        ]},
        {"type": "reasoning", "encrypted_content": "gAAAA-secret",
         "summary": [{"type": "summary_text", "text": "Try the left lever."}]},
        {"type": "function_call", "call_id": "call_1", "name": "python", "arguments": '{"code": "step(1)"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "moved"},
        {"role": "user", "content": "go on"},
    ]
    # the effort ladder's override wins over the environment
    assert responses_payload_from_chat(_chat_payload(messages), reasoning_effort="low")["reasoning"]["effort"] == "low"


def test_stream_reads_the_terminal_event() -> None:
    lines = [
        b"event: response.created",
        b'data: {"type": "response.created", "response": {"status": "in_progress"}}',
        b'data: {"type": "response.reasoning_summary_text.delta", "delta": "Try"}',
        b"data: " + json.dumps({"type": "response.completed", "response": RESPONSE}).encode(),
    ]
    assert assemble_streamed_responses(lines) == chat_response_from_responses(RESPONSE)
    failed = assemble_streamed_responses([b'data: {"type": "error", "code": "server_error", "message": "boom"}'])
    assert failed["choices"] == [] and failed["error"]["message"] == "boom"


def test_encrypted_reasoning_counts_as_its_tokens_not_its_text() -> None:
    reply = chat_response_from_responses(RESPONSE)["choices"][0]["message"]
    messages = [{"role": "user", "content": "hi"}, reply]
    scrubbed, tokens = _split_reasoning_details_for_estimate(messages)
    assert tokens == 240
    assert scrubbed[1]["reasoning_details"][0]["data"] == "<encrypted>"
    assert reply["reasoning_details"][0]["data"] == "gAAAA-secret"
    assert scrubbed[0] is messages[0]
