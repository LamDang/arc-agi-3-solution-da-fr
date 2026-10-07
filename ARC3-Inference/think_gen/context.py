"""Build what flash sees for one record: the agent's context with earlier
turns' thinking filled in, and the call to explain.

Earlier assistant turns carry the thinking generated for them, so each
request is generated in the context the student will have at deploy time
(`preserve_thinking`). OpenRouter drops `reasoning` / `reasoning_content` from
history messages for qwen3.8-flash, and Alibaba strips `<think>` blocks from
history content (tested 2026-10-07), so the thinking goes into the message
content between `[thinking]` and `[/thinking]` lines instead.
"""
import copy
import hashlib
import json

THINK_OPEN, THINK_CLOSE = "[thinking]", "[/thinking]"
REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details")


def message_ref(msg: dict) -> str:
    """Identifies an assistant turn across requests: its first tool call id,
    else a hash of its text."""
    calls = msg.get("tool_calls") or []
    if calls and calls[0].get("id"):
        return calls[0]["id"]
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    return "text:" + hashlib.sha1(content.strip().encode()).hexdigest()[:16]


def normalize(text: str) -> str:
    """Thinking as the harness puts it in history: blank lines removed
    (`_normalize_message_content` in inference/agent/tool_agent.py)."""
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def with_thinking(content, thinking: str):
    if not thinking:
        return content
    block = f"{THINK_OPEN}\n{normalize(thinking)}\n{THINK_CLOSE}"
    if not content:
        return block
    if isinstance(content, str):
        return f"{block}\n\n{content}"
    return [{"type": "text", "text": block}, *content]


def history(messages: list, thinking: dict[str, str]) -> list:
    """The request's messages with every assistant turn's reasoning removed
    and the generated thinking (by `message_ref`) inlined instead."""
    out = []
    for m in messages:
        # `_arc3_control` and other private keys never reach the server
        m = {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("_")}
        if m.get("role") == "assistant":
            ref = message_ref(m)
            for k in REASONING_KEYS:
                m.pop(k, None)
            m["content"] = with_thinking(m.get("content") or "", thinking.get(ref, ""))
        out.append(m)
    return out


def call_text(reply: dict) -> str:
    """The reply as the reconstruction prompt shows it: any visible text, then
    each tool call (python code verbatim, other tools as JSON arguments)."""
    parts = []
    content = reply.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(f"Visible message:\n{content.strip()}")
    for c in reply.get("tool_calls") or []:
        fn = c.get("function") or {}
        name, raw = fn.get("name", "?"), fn.get("arguments") or ""
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = raw
        if name == "python" and isinstance(args, dict) and isinstance(args.get("code"), str):
            parts.append(f"Tool call `python` with code:\n```python\n{args['code'].rstrip()}\n```")
        else:
            parts.append(f"Tool call `{name}` with arguments:\n{json.dumps(args, ensure_ascii=False)}")
    return "\n\n".join(parts) if parts else "(no visible output and no tool call)"
