"""Helpers for provider-specific OpenAI-compatible requests."""
from __future__ import annotations

import json

from typing import Any


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    if provider in {"openai-responses", "openai_responses", "responses"}:
        return "openai-responses"
    return provider


def build_headers(
    *,
    provider: str,
    api_key: str,
    referer: str = "",
    title: str = "",
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    seed: int | None = None,
    priority: int | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if priority is not None:
        # scheduling hint, ignored by servers that do not implement it
        payload["priority"] = int(priority)
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    normalized = normalize_provider(provider)
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if normalized == "openrouter":
        import os
        if not payload["stream"] and os.environ.get(
            "OPENROUTER_STREAM", ""
        ).strip().lower() not in ("0", "false", "no", "off"):
            payload["stream"] = True
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
        payload["reasoning"] = {"enabled": bool(thinking)}
        # e.g. "xhigh" for OpenAI models; unset leaves the provider default
        _effort = os.environ.get("OPENROUTER_REASONING_EFFORT", "").strip()
        if _effort and thinking:
            payload["reasoning"]["effort"] = _effort
        # which reasoning sent back in history the model reads (OpenAI):
        # auto, all_turns or current_turn
        _context = os.environ.get("OPENROUTER_REASONING_CONTEXT", "").strip()
        if _context and thinking:
            payload["reasoning"]["context"] = _context
        # A reply cut off at max_tokens while still reasoning is kept in history
        # as an assistant message with content=None and no tool calls. Some
        # upstreams (Alibaba) reject null content with HTTP 400 on every later
        # request, which ends the game; "" is accepted and renders the same.
        payload["messages"] = [
            {**message, "content": ""}
            if message.get("role") == "assistant"
            and message.get("content") is None
            and not message.get("tool_calls")
            else message
            for message in messages
        ]
        _order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
        if _order:
            payload["provider"] = {
                "order": [s.strip() for s in _order.split(",") if s.strip()],
                "allow_fallbacks": os.environ.get(
                    "OPENROUTER_ALLOW_FALLBACKS", ""
                ).strip().lower() in ("1", "true", "yes"),
            }
    if normalized == "vllm":
        if top_k > 0:
            payload["top_k"] = top_k
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if seed is not None and seed >= 0:
            payload["seed"] = seed

    return payload


def assemble_streamed_chat_response(lines) -> dict[str, Any]:
    """Consume an SSE chat-completion stream and rebuild the non-streaming
    response shape: {"choices": [{"message", "finish_reason"}], "usage"}.

    Tolerates keep-alive comments (": ..."), event: lines, usage-only chunks,
    and mid-stream error events. If an error arrives and no assistant content
    was produced, returns {"error": ..., "choices": []} so callers surface it.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    # Every reasoning_details entry, fragments of one entry joined: OpenAI
    # models stream a summary in pieces and then an encrypted block, which
    # the plain reasoning text leaves out.
    reasoning_details: list[dict[str, Any]] = []
    response_id = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason = ""
    usage: dict[str, Any] | None = None
    error: Any = None
    provider_name = [""]

    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error = chunk["error"]
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        if chunk.get("id") and not response_id:
            response_id = str(chunk["id"])
        if chunk.get("provider") and not provider_name[0]:
            provider_name[0] = str(chunk["provider"])
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            for detail in delta.get("reasoning_details") or []:
                if isinstance(detail, dict):
                    _merge_reasoning_detail(reasoning_details, detail)
            if isinstance(delta.get("reasoning"), str):
                reasoning_parts.append(delta["reasoning"])
            elif isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
            elif isinstance(delta.get("reasoning_details"), list):
                for detail in delta["reasoning_details"]:
                    if isinstance(detail, dict) and isinstance(detail.get("text"), str):
                        reasoning_parts.append(detail["text"])
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    continue
                index = int(fragment.get("index", 0) or 0)
                entry = tool_calls_by_index.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if fragment.get("id"):
                    entry["id"] = str(fragment["id"])
                if fragment.get("type"):
                    entry["type"] = str(fragment["type"])
                function = fragment.get("function") or {}
                if isinstance(function, dict):
                    if function.get("name"):
                        entry["function"]["name"] = str(function["name"])
                    if isinstance(function.get("arguments"), str):
                        entry["function"]["arguments"] += function["arguments"]

    produced = bool(content_parts or reasoning_parts or tool_calls_by_index)
    if error is not None and not produced:
        return {"error": error, "choices": []}

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    result: dict[str, Any] = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        result["usage"] = usage
    if error is not None:
        result["error"] = error
    if provider_name[0]:
        result["provider"] = provider_name[0]
    if response_id:
        result["id"] = response_id
    return result


_REASONING_DETAIL_TEXT_KEYS = ("text", "summary")


def _merge_reasoning_detail(merged: list[dict[str, Any]], detail: dict[str, Any]) -> None:
    """Append a streamed reasoning_details fragment, joining a text or summary
    fragment to the entry it continues (same type and index). Encrypted
    blocks arrive whole and are kept as they are."""
    last = merged[-1] if merged else None
    if (
        last is not None
        and detail.get("type") in ("reasoning.text", "reasoning.summary")
        and last.get("type") == detail.get("type")
        and last.get("index") == detail.get("index")
    ):
        for key, value in detail.items():
            if key in _REASONING_DETAIL_TEXT_KEYS and isinstance(value, str):
                last[key] = str(last.get(key) or "") + value
            elif value is not None:
                last.setdefault(key, value)
        return
    merged.append(dict(detail))



# OpenAI's Responses API, called directly (provider "openai-responses").
#
# Chat completions on api.openai.com never returns a reasoning model's
# reasoning, and OpenRouter's Azure route drops the encrypted block sent back.
# The Responses API with store=false and include=["reasoning.encrypted_content"]
# returns each reasoning item encrypted; sent back as an input item before the
# message or tool call it led to, the model reads it again. The harness keeps
# speaking chat completions: the request is translated on the way out and the
# reply translated back, with the reasoning items kept in reasoning_details
# (type "reasoning.encrypted", format "openai-responses-v1", as OpenRouter
# names them) so ARC3_SEND_REASONING_DETAILS carries them through history.

_RESPONSES_FORMAT = "openai-responses-v1"


def _responses_content_parts(content: Any) -> list[dict[str, Any]] | str:
    """Chat content (a string or text/image_url parts) as Responses input parts."""
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            detail = (image.get("detail") if isinstance(image, dict) else None) or "auto"
            parts.append({"type": "input_image", "image_url": str(url or ""), "detail": detail})
    return parts


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        )
    return "" if content is None else str(content)


def _responses_reasoning_items(details: Any) -> list[dict[str, Any]]:
    """reasoning_details kept by chat_response_from_responses, as the reasoning
    input items they came from. Ids are left out: with store=false nothing is
    looked up by id, and the encrypted content alone is read (measured)."""
    items: list[dict[str, Any]] = []
    if not isinstance(details, list):
        return items
    for detail in details:
        if (
            isinstance(detail, dict)
            and detail.get("type") == "reasoning.encrypted"
            and detail.get("format") == _RESPONSES_FORMAT
            and detail.get("data")
        ):
            items.append(
                {
                    "type": "reasoning",
                    "encrypted_content": detail["data"],
                    "summary": [
                        {"type": "summary_text", "text": str(text)}
                        for text in detail.get("summary") or []
                    ],
                }
            )
    return items


def responses_input_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-completions messages as Responses input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            items.append({"role": "developer", "content": _text_of(message.get("content"))})
        elif role == "user":
            items.append({"role": "user", "content": _responses_content_parts(message.get("content"))})
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": _responses_content_parts(message.get("content")),
                }
            )
        elif role == "assistant":
            following: list[dict[str, Any]] = []
            text = _text_of(message.get("content"))
            if text:
                following.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                following.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "arguments": str(function.get("arguments") or ""),
                    }
                )
            # a reasoning item must precede the output it led to; a reply that
            # produced nothing (cut off while reasoning) keeps no reasoning
            if following:
                items.extend(_responses_reasoning_items(message.get("reasoning_details")))
                items.extend(following)
    return items


def responses_payload_from_chat(
    chat_payload: dict[str, Any], *, reasoning_effort: str | None = None
) -> dict[str, Any]:
    """A chat-completions payload from build_chat_payload as a Responses
    request. Sampling settings (temperature, top_p, top_k, seed) are left out:
    OpenAI's reasoning models reject them."""
    import os

    payload: dict[str, Any] = {
        "model": chat_payload["model"],
        "input": responses_input_from_messages(chat_payload.get("messages") or []),
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "stream": os.environ.get("OPENAI_STREAM", "").strip().lower()
        not in ("0", "false", "no", "off"),
    }
    if chat_payload.get("max_tokens") is not None:
        payload["max_output_tokens"] = int(chat_payload["max_tokens"])
    reasoning: dict[str, Any] = {}
    effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if effort:
        reasoning["effort"] = effort
    # detailed: replayed on 8 logged requests that had none, a summary came back
    # for 4 against 2 with auto. OpenAI still skips it for some reasoning items.
    summary = os.environ.get("OPENAI_REASONING_SUMMARY", "detailed").strip()
    if summary:
        reasoning["summary"] = summary
    if reasoning:
        payload["reasoning"] = reasoning
    service_tier = os.environ.get("OPENAI_SERVICE_TIER", "").strip()
    if service_tier:
        payload["service_tier"] = service_tier
    tools = []
    for tool in chat_payload.get("tools") or []:
        function = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                # the harness's schemas are not written for strict mode, unless
                # a tool says so (ARC3_PYTHON_RATIONALE)
                "strict": bool(function.get("strict", False)),
            }
        )
    if tools:
        payload["tools"] = tools
        choice = chat_payload.get("tool_choice")
        if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            payload["tool_choice"] = {"type": "function", "name": choice["function"].get("name")}
        elif choice:
            payload["tool_choice"] = choice
    return payload


def _openai_pricing() -> tuple[float, float, float, float] | None:
    """ARC3_OPENAI_PRICING: dollars per million input, cached input, cache
    write and output tokens, e.g. "2,0.1,2.5,10". OpenAI returns no cost, so
    without it the request logs carry none."""
    import os

    raw = os.environ.get("ARC3_OPENAI_PRICING", "").strip()
    if not raw:
        return None
    try:
        values = [float(value) for value in raw.split(",")]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    return values[0], values[1], values[2], values[3]


def _chat_usage_from_responses(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    cached = int(input_details.get("cached_tokens") or 0)
    cache_write = int(input_details.get("cache_write_tokens") or 0)
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": cache_write},
        "completion_tokens_details": {
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0)
        },
    }
    pricing = _openai_pricing()
    if pricing is not None:
        price_in, price_cached, price_write, price_out = pricing
        uncached = max(0, prompt_tokens - cached - cache_write)
        result["cost"] = (
            uncached * price_in
            + cached * price_cached
            + cache_write * price_write
            + completion_tokens * price_out
        ) / 1e6
    return result


def chat_response_from_responses(response: dict[str, Any]) -> dict[str, Any]:
    """A Responses API response object in the chat-completions shape the
    harness reads: {"choices": [{"message", "finish_reason"}], "usage", "id"}."""
    if not isinstance(response, dict):
        return {"choices": [], "error": "empty response"}
    status = str(response.get("status") or "")
    if status == "failed" or (response.get("error") and not response.get("output")):
        return {"choices": [], "error": response.get("error") or status}
    content_parts: list[str] = []
    summaries: list[str] = []
    details: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            texts = [
                str(part.get("text") or "")
                for part in item.get("summary") or []
                if isinstance(part, dict)
            ]
            summaries.extend(text for text in texts if text)
            if item.get("encrypted_content"):
                details.append(
                    {
                        "type": "reasoning.encrypted",
                        "id": item.get("id"),
                        "format": _RESPONSES_FORMAT,
                        "index": len(details),
                        "data": item["encrypted_content"],
                        "summary": texts,
                    }
                )
        elif kind == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    content_parts.append(str(part.get("text") or ""))
                elif isinstance(part, dict) and part.get("type") == "refusal":
                    content_parts.append(str(part.get("refusal") or ""))
        elif kind == "function_call":
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or ""),
                    },
                }
            )
    usage = response.get("usage") or {}
    reasoning_tokens = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
    # what the blocks cost as input when sent back, for the harness's context
    # estimate (the encrypted text is far longer); split evenly when several
    for detail in details:
        detail["tokens"] = reasoning_tokens // len(details)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if summaries:
        message["reasoning"] = "\n\n".join(summaries)
    if details:
        message["reasoning_details"] = details
    if tool_calls:
        message["tool_calls"] = tool_calls
    reason = str((response.get("incomplete_details") or {}).get("reason") or "")
    if status == "incomplete":
        finish_reason = "length" if reason == "max_output_tokens" else (reason or "length")
    else:
        finish_reason = "tool_calls" if tool_calls else "stop"
    result: dict[str, Any] = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": _chat_usage_from_responses(response.get("usage")),
        "provider": "OpenAI",
    }
    if response.get("id"):
        result["id"] = str(response["id"])
    return result


def assemble_streamed_responses(lines) -> dict[str, Any]:
    """Consume a Responses API SSE stream and return the final response in the
    chat-completions shape (chat_response_from_responses). Only the terminal
    event is read: response.completed, response.incomplete or response.failed
    carry the whole response."""
    final: dict[str, Any] | None = None
    error: Any = None
    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            final = event.get("response")
        elif kind == "error":
            error = {key: value for key, value in event.items() if key != "type"}
    if final is None:
        return {"choices": [], "error": error or "stream ended without a final response"}
    return chat_response_from_responses(final)
