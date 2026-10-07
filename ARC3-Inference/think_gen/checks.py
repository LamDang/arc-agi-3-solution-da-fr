"""Cheap checks on a generated thinking text, run before it is accepted."""
import json
import re

LEAK_PATTERNS = {
    "summary": r"\bsummar(y|ies|ised|ized|ize)\b",
    "the_agent": r"\bthe agent\b",
    "reconstruct": r"\breconstruct",
    "given_output": r"\b(given|provided|shown|specified) (output|call|code|tool call)\b",
    "next_output_was": r"\b(next|final) output (was|is)\b",
    "note": r"\bnote outside\b",
    "think_tags": r"\[/?thinking\]|</?think>",
}


def leaks(text: str) -> list[str]:
    return [name for name, pat in LEAK_PATTERNS.items() if re.search(pat, text, re.I)]


def pasted_code_fraction(text: str, reply: dict) -> float:
    """Share of the call's substantial code lines (20+ characters) that appear
    verbatim in the thinking: high means the code was pasted, not described."""
    lines = []
    for c in reply.get("tool_calls") or []:
        try:
            code = json.loads((c.get("function") or {}).get("arguments") or "{}").get("code") or ""
        except (json.JSONDecodeError, AttributeError):
            code = ""
        lines += [ln.strip() for ln in code.splitlines() if len(ln.strip()) >= 20]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln in text) / len(lines)


def clean(text: str) -> str:
    """Strip tags a model may wrap the answer in."""
    t = text.strip()
    t = re.sub(r"^\s*(\[thinking\]|<think>)\s*", "", t)
    t = re.sub(r"\s*(\[/thinking\]|</think>)\s*$", "", t)
    return t.strip()


def check(text: str, reply: dict) -> dict:
    out = {"leaks": leaks(text), "pasted_code": round(pasted_code_fraction(text, reply), 2),
           "chars": len(text)}
    out["ok"] = bool(text.strip()) and not out["leaks"] and out["pasted_code"] <= 0.5
    return out
