"""Build v4-agent-transcripts.html: the ls20, lp85 and vc33 sessions of the v4 run, turn by turn.

Run from ARC3-Inference/ with the run directory present (runs/ is not committed):

    uv run --no-sync python engine_re/results/v4_transcripts/build.py
"""
import base64, gzip, json, sys
from pathlib import Path

sys.path.insert(0, ".")
from engine_re.agent import AUTO_TEST, NUDGE
from engine_re.prompts import first_user_message, system_prompt
from engine_re.skeleton import render_skeleton
from engine_re.trace import Trace

RUN = Path("runs/engine-re/qwen38flash-v4-simple-run20261004_135539")
GAMES = ["ls20", "lp85", "vc33"]
HERE = Path(__file__).resolve().parent

data = {"system": system_prompt("final", "simple"), "games": {}}
for g in GAMES:
    gd = RUN / g
    trace = Trace.load(gd / "trace")
    first = first_user_message(g, trace, render_skeleton(g, trace[0].available_actions, "simple"), "simple")
    result = json.loads((gd / "result.json").read_text())
    tests = [json.loads(l) for l in (gd / "tests.jsonl").read_text().splitlines()]
    turns = {}
    for line in (gd / "transcript.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = r.get("turn")
        if "finish_reason" in r:
            u = r.get("usage") or {}
            turns[t] = {
                "t": t, "min": r.get("elapsed_min"),
                "reasoning": r.get("reasoning") or "", "content": r.get("content") or "",
                "calls": [{"name": c["function"]["name"], "args": c["function"]["arguments"]} for c in (r.get("tool_calls") or [])],
                "outputs": [], "harness": [],
                "tok": [u.get("prompt_tokens"), u.get("completion_tokens"), (u.get("completion_tokens_details") or {}).get("reasoning_tokens")],
            }
        elif "tool" in r:
            turns[t]["outputs"].append(r["output"])
        elif "auto_test" in r:
            turns[t]["harness"].append({"kind": "auto", "text": AUTO_TEST.format(report=r["auto_test"]).strip()})
        elif "nudge" in r:
            turns[t]["harness"].append({"kind": "nudge", "text": NUDGE.format(n=r["nudge"]).strip()})
    for test in tests:
        if test["turn"] in turns:
            err = (test.get("error") or "").strip().splitlines()
            turns[test["turn"]].setdefault("tests", []).append(
                {"exact": test["exact"], "total": test["total"], "auto": test.get("auto"), "contract": test.get("contract_passed"), "error": err[-1] if err else None}
            )
    u = result["usage"]
    data["games"][g] = {
        "first": first,
        "meta": {"status": result["status"], "turns": result["turns"], "minutes": result["minutes"], "cost": round(u["cost_usd"], 2),
                 "out": u["completion_tokens"], "reasoning": u["reasoning_tokens"], "tests": len(tests),
                 "best": (result.get("best") or {}).get("exact", 0), "steps": result["trace_steps"], "finish": result.get("finish_summary")},
        "turns": [turns[k] for k in sorted(turns)],
    }

raw = json.dumps(data, separators=(",", ":")).encode()
packed = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
page = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", packed)
(HERE / "v4-agent-transcripts.html").write_text(page, encoding="utf-8")
print(f"raw {len(raw)/1e6:.1f} MB, packed {len(packed)/1e6:.2f} MB, page {len(page)/1e6:.2f} MB")
