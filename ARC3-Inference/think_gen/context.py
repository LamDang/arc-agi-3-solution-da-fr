"""Build what flash sees for one record: the agent's context with earlier
turns' thinking filled in, and the call to explain.

Earlier assistant turns carry the thinking generated for them, so each
request is generated in the context the student will have at deploy time
(`preserve_thinking`). By default (`native`) it goes in the history
message's `reasoning` field, which OpenRouter passes to qwen3.8-flash on
Alibaba: a probe with a 1,000-token reasoning block grew the prompt by that
much and the model could quote it, and in runs/base-max-dfranzen the prompt
grew by at least the previous reply's reasoning tokens on all 195
consecutive request pairs. `inline` puts it in the message content between
`[thinking]` and `[/thinking]` lines instead; the b1/b2 calibration runs
used it.
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


HISTORY_MODES = ("native", "inline")


def history(messages: list, thinking: dict[str, str], mode: str = "native") -> list:
    """The request's messages with every assistant turn's reasoning replaced
    by the generated thinking (by `message_ref`), in `reasoning` (`native`)
    or in the content (`inline`)."""
    out = []
    for m in messages:
        # `_arc3_control` and other private keys never reach the server
        m = {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("_")}
        if m.get("role") == "assistant":
            ref = message_ref(m)
            for k in REASONING_KEYS:
                m.pop(k, None)
            text = thinking.get(ref, "")
            if mode == "inline":
                m["content"] = with_thinking(m.get("content") or "", text)
            else:
                m["content"] = m.get("content") or ""
                if text:
                    m["reasoning"] = normalize(text)
        out.append(m)
    return out


def code_only_request(messages: list, tools: list) -> tuple[list, list]:
    """The messages and tools of a rationale request (python takes `reasoning`,
    `description`, `code`) turned into the code-only variant the teacher would
    have seen without ARC3_PYTHON_RATIONALE: the system prompt's python line and
    the python tool schema lose the reasoning and description. Used to regenerate
    a call from the generated thinking (the call-equivalence judge). The teacher's
    own earlier turns stay as logged; only the instructions change."""
    import re

    from inference.agent.tool_agent import _PYTHON_PROMPT_CODE_LINE, _PYTHON_TOOL_DESCRIPTION
    # Replace whatever "- The only tool is `python`; ... `code` string." bullet a
    # run logged (its wording and field order have changed over runs) with the
    # code-only one. The bullet is a single line.
    bullet = re.compile(r"- The only tool is `python`;[^\n]*`code` string\.\n?")
    out_msgs = []
    for m in messages:
        m = {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("_")}
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = bullet.sub(_PYTHON_PROMPT_CODE_LINE, m["content"], count=1)
        out_msgs.append(m)
    out_tools = []
    for t in tools or []:
        t = copy.deepcopy(t)
        fn = t.get("function") or {}
        if fn.get("name") == "python":
            # the logged description is the code-only base plus a rationale
            # sentence; reset it to the canonical base.
            fn["description"] = _PYTHON_TOOL_DESCRIPTION
            params = fn.setdefault("parameters", {})
            props = params.get("properties") or {}
            params["properties"] = {"code": props.get("code", {"type": "string",
                "description": "Python code to run. The snippet is ephemeral and is not saved across tool calls."})}
            params["required"] = ["code"]
            params.pop("additionalProperties", None)
            fn.pop("strict", None)
        out_tools.append(t)
    return out_msgs, out_tools


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
