"""Minimal OpenRouter chat-completions client with retries.

Every call can be logged, request and response, by passing `log_path` (and a
`log_tag` naming the record and check). The log keeps the exact messages sent,
with image data URIs reduced to a sha reference and byte count so the file
stays small — the image bytes live in the source run. One JSON line per call
(including the final give-up), appended thread-safely, so a judge or generation
run has a complete, replayable record of what was asked and what came back.
"""
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.8-flash"
PROVIDER = "Alibaba"  # the only provider serving qwen3.8-flash (2026-10-07)

_log_lock = threading.Lock()


class CallError(RuntimeError):
    pass


def _slim(messages: list) -> list:
    """Messages with image data URIs replaced by a short sha ref + byte count."""
    out = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    parts.append({"type": "image_url", "sha": hashlib.sha1(url.encode()).hexdigest()[:12],
                                  "bytes": len(url)})
                else:
                    parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out


def _digest(messages: list) -> dict:
    """A compact, retraceable stand-in for the request messages: the appended
    instruction (the last message, kept verbatim with images slimmed) plus a
    verifiable reference to the context prefix (its message count, a sha and a
    char length). The prefix itself is the source record's context, which lives
    in the DVC-tracked source run, so it is not duplicated here; the sha lets a
    reconstruction be checked. `log_tag.key` names which source record it is."""
    msgs = messages or []
    prefix = msgs[:-1]
    ser = json.dumps(prefix, ensure_ascii=False, sort_keys=True)
    return {
        "n_messages": len(msgs),
        "prefix": {"n": len(prefix), "sha1": hashlib.sha1(ser.encode()).hexdigest(), "chars": len(ser)},
        "instruction": _slim(msgs[-1:])[0] if msgs else None,
    }


def log_call(log_path, request: dict, response: dict, tag: dict | None = None):
    """Append one request/response record. Never raises (logging must not break
    a call)."""
    if not log_path:
        return
    try:
        rec = {"ts": round(time.time(), 3), **(tag or {}), "request": request, "response": response}
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(p, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 - logging is best-effort
        print(f"[request-log] failed: {e!r}", flush=True)


def chat(messages: list, *, model: str = MODEL, provider: str | None = PROVIDER,
         tools: list | None = None, reasoning: bool = False, max_tokens: int = 8192,
         temperature: float = 0.7, retries: int = 6, timeout: int = 900,
         json_mode: bool = False, tool_choice: str | dict = "none",
         log_path=None, log_tag: dict | None = None) -> dict:
    """One completion. Returns {"content", "reasoning", "usage", "provider",
    "finish_reason", "secs"}. Retries 429s, 5xx, network errors and empty
    answers with exponential backoff."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"enabled": bool(reasoning)},
        "usage": {"include": True},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    last = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(120, 2 ** attempt) * (0.5 + random.random()))
        t = time.time()
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = repr(e)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            continue
        if r.status_code != 200:
            raise CallError(f"HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        if j.get("error") or not j.get("choices"):
            last = f"error in body: {str(j.get('error') or j)[:300]}"
            continue
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        if not content.strip() and not msg.get("tool_calls") and ch.get("finish_reason") != "length":
            last = f"empty answer (finish {ch.get('finish_reason')})"
            continue
        out = {
            "content": content,
            "reasoning": msg.get("reasoning") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "usage": j.get("usage") or {},
            "provider": j.get("provider"),
            "finish_reason": ch.get("finish_reason"),
            "secs": round(time.time() - t, 1),
        }
        log_call(log_path, {"api": "chat", "model": model, "provider": provider, "reasoning": reasoning,
                            "temperature": temperature, "max_tokens": max_tokens,
                            "tool_choice": tool_choice if tools else None, **_digest(messages)},
                 {k: out[k] for k in ("content", "reasoning", "tool_calls", "usage", "finish_reason", "secs")},
                 {**(log_tag or {}), "attempt": attempt + 1})
        return out
    log_call(log_path, {"api": "chat", "model": model, **_digest(messages)},
             {"error": last}, log_tag)
    raise CallError(f"gave up after {retries + 1} attempts: {last}")


OPENAI_URL = "https://api.openai.com/v1/responses"


def openai_responses(messages: list, *, model: str = "gpt-6.1-sol", effort: str = "high",
                     tools: list | None = None, json_mode: bool = False,
                     max_output_tokens: int = 32000, retries: int = 6, timeout: int = 900,
                     log_path=None, log_tag: dict | None = None) -> dict:
    """One OpenAI Responses call from chat-completions messages (images kept),
    through the harness's converter. Returns {"content", "usage", "secs"};
    usage carries OpenAI's token counts."""
    from inference.utils.openai_compat import chat_response_from_responses, responses_input_from_messages
    body = {"model": model, "input": responses_input_from_messages(messages), "store": False,
            "reasoning": {"effort": effort}, "max_output_tokens": max_output_tokens}
    if tools:
        body["tools"] = [{"type": "function", "name": t["function"]["name"],
                          "description": t["function"].get("description") or "",
                          "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}},
                          "strict": bool(t["function"].get("strict", False))} for t in tools]
        body["tool_choice"] = "none"
    if json_mode:
        body["text"] = {"format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    last = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(120, 2 ** attempt) * (0.5 + random.random()))
        t = time.time()
        try:
            r = requests.post(OPENAI_URL, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = repr(e)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            continue
        if r.status_code != 200:
            raise CallError(f"HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        chat = chat_response_from_responses(j)
        if not chat.get("choices"):
            last = f"no output: {str(chat.get('error'))[:300]}"
            continue
        content = chat["choices"][0]["message"].get("content") or ""
        if not content.strip():
            last = "empty answer"
            continue
        out = {"content": content, "usage": j.get("usage") or {}, "secs": round(time.time() - t, 1)}
        log_call(log_path, {"api": "responses", "model": model, "effort": effort,
                            "max_output_tokens": max_output_tokens, **_digest(messages)},
                 out, {**(log_tag or {}), "attempt": attempt + 1})
        return out
    log_call(log_path, {"api": "responses", "model": model, "effort": effort, **_digest(messages)},
             {"error": last}, log_tag)
    raise CallError(f"gave up after {retries + 1} attempts: {last}")
