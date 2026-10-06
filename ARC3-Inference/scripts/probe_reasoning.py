"""Check which OpenRouter models return their full reasoning, for distillation.

    uv run --no-sync python scripts/probe_reasoning.py out.jsonl \
        moonshotai/kimi-k3 "moonshotai/kimi-k3@Amazon Bedrock" openai/gpt-6.1-sol

Each argument is an OpenRouter model id, optionally `@<provider>` to pin one
provider (no fallbacks). Every model gets two requests with reasoning enabled
(effort medium): a grid puzzle, and a prompt that must call a `python` tool.
For each, the reasoning tokens the provider bills (`usage.completion_tokens_
details.reasoning_tokens`) are compared with the reasoning text returned
(`reasoning`, and `reasoning_details` split by type: text, summary, encrypted).
`visible_over_billed` is the returned text's length / 3.6 over the billed
reasoning tokens: about 0.6-1.2 when the full chain of thought comes back
(Chinese-vocabulary tokenizers sit near 0.6), well under 0.4 for a summary,
0 when it is hidden or encrypted. A `reasoning.text` type does not mean full:
some providers put a summary there, so read `reasoning_head` too.

Appends one JSON line per model to the output file, with the raw responses.
Needs OPENROUTER_API_KEY.
"""
import json, os, sys, time, concurrent.futures as cf
import urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"

PUZZLE = (
    "A 6x6 grid has walls at (1,2),(2,2),(3,2),(3,3),(3,4),(1,4),(4,1),(5,4). "
    "Coordinates are (row,col), 0-indexed. Start at (0,0), goal at (5,5). Moves: "
    "up/down/left/right, one cell. Find the length of the shortest path and one "
    "such path. Then, if you may also remove exactly one wall, what is the "
    "shortest possible path length? Think carefully and verify."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"]},
    },
}]
TOOL_PROMPT = (
    "I need the 2000th prime number multiplied by the number of divisors of 720720. "
    "Plan how you will compute it, then call the python tool once with code that "
    "prints the answer. Do not answer from memory."
)


def call(model, messages, tools=None, provider=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 16000,
        "reasoning": {"enabled": True, "effort": "medium"},
    }
    if tools:
        body["tools"] = tools
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:400].decode('utf-8','replace')}"}
    except Exception as e:  # noqa
        return {"error": repr(e)[:400]}
    resp["_secs"] = round(time.time() - t, 1)
    return resp


def analyse(resp):
    if "error" in resp and not resp.get("choices"):
        return {"error": str(resp["error"])[:400]}
    ch = resp["choices"][0]
    msg = ch.get("message") or {}
    usage = resp.get("usage") or {}
    det = usage.get("completion_tokens_details") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    types = {}
    text_chars = summary_chars = enc_chars = 0
    for d in msg.get("reasoning_details") or []:
        ty = d.get("type", "?")
        types[ty] = types.get(ty, 0) + 1
        if ty == "reasoning.text":
            text_chars += len(d.get("text") or "")
        elif ty == "reasoning.summary":
            summary_chars += len(d.get("summary") or "")
        elif ty == "reasoning.encrypted":
            enc_chars += len(d.get("data") or "")
    billed = det.get("reasoning_tokens")
    visible_chars = max(len(reasoning), text_chars, summary_chars)
    est_tokens = visible_chars / 3.6
    ratio = (est_tokens / billed) if billed else None
    return {
        "provider": resp.get("provider"),
        "finish": ch.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens_billed": billed,
        "reasoning_chars": len(reasoning),
        "details_types": types,
        "detail_text_chars": text_chars,
        "detail_summary_chars": summary_chars,
        "detail_encrypted_chars": enc_chars,
        "visible_over_billed": round(ratio, 3) if ratio is not None else None,
        "content_chars": len(msg.get("content") or ""),
        "tool_calls": len(msg.get("tool_calls") or []),
        "cost": usage.get("cost"),
        "secs": resp.get("_secs"),
        "reasoning_head": reasoning[:300],
        "reasoning_tail": reasoning[-300:],
    }


def probe(model, provider=None):
    out = {"model": model, "provider_req": provider}
    r1 = call(model, [{"role": "user", "content": PUZZLE}], provider=provider)
    out["plain"] = analyse(r1)
    out["plain_raw"] = r1
    r2 = call(model, [{"role": "user", "content": TOOL_PROMPT}], tools=TOOLS, provider=provider)
    out["tool"] = analyse(r2)
    out["tool_raw"] = r2
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_path, specs = sys.argv[1], sys.argv[2:]
    jobs = [(s.split("@")[0], s.split("@")[1] if "@" in s else None) for s in specs]
    with cf.ThreadPoolExecutor(12) as ex, open(out_path, "a") as f:
        futs = {ex.submit(probe, m, p): (m, p) for m, p in jobs}
        for fu in cf.as_completed(futs):
            res = fu.result()
            f.write(json.dumps(res) + "\n"); f.flush()
            for k in ("plain", "tool"):
                a = res[k]
                print(f"{res['model']:45s} {k:5s}", {x: a.get(x) for x in (
                    "error", "provider", "reasoning_tokens_billed", "reasoning_chars",
                    "details_types", "visible_over_billed", "tool_calls", "cost")}, flush=True)
