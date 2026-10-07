"""Minimal OpenRouter chat-completions client with retries."""
import os
import random
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.8-flash"
PROVIDER = "Alibaba"  # the only provider serving qwen3.8-flash (2026-10-07)


class CallError(RuntimeError):
    pass


def chat(messages: list, *, model: str = MODEL, provider: str | None = PROVIDER,
         tools: list | None = None, reasoning: bool = False, max_tokens: int = 8192,
         temperature: float = 0.7, retries: int = 6, timeout: int = 900,
         json_mode: bool = False) -> dict:
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
        body["tool_choice"] = "none"
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
        if not content.strip() and ch.get("finish_reason") != "length":
            last = f"empty answer (finish {ch.get('finish_reason')})"
            continue
        return {
            "content": content,
            "reasoning": msg.get("reasoning") or "",
            "usage": j.get("usage") or {},
            "provider": j.get("provider"),
            "finish_reason": ch.get("finish_reason"),
            "secs": round(time.time() - t, 1),
        }
    raise CallError(f"gave up after {retries + 1} attempts: {last}")
