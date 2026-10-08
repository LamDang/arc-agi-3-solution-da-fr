"""Can qwen3.8-flash regenerate an equivalent python call from the teacher (sol)
thinking already attached to each request in data/sol-nll-fold0-30?

For each request the final assistant reply carries sol's (converted) thinking in
`reasoning_content` and the teacher's code-only `python` call:
  - context  = messages[:-1]   (game state up to the decision point)
  - thinking = messages[-1]["reasoning_content"]
  - sol_code = messages[-1]  python tool-call code

We reuse think_gen's call-equivalence judge (judge_sol.check_call):
  1. flash regenerates a python call from the thinking (code-only request,
     REGEN_NOTE), and
  2. gpt-6.1-sol, with the game context in front, judges whether the two
     snippets are FUNCTIONALLY the same move on the game.

The dataset is already in the code-only deploy format the judge expects, so the
only adaptation is cleaning each message for the OpenRouter/OpenAI APIs:
strip private `_`-prefixed keys, JSON-encode tool-call arguments (stored here as
dicts), and move assistant `reasoning_content` into `reasoning` (native history,
as think_gen.context documents).

    # from ARC3-Inference/, with OPENROUTER_API_KEY and OPENAI_API_KEY set
    PYTHONPATH=. uv run --no-sync python \
        experiments/teacher-reasoning/sol-nll-regen/run.py \
        --out experiments/teacher-reasoning/sol-nll-regen --workers 3
"""
import argparse
import concurrent.futures as cf
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from think_gen import client, context, judge_sol, logs, prompts

DEFAULT_DATA = Path(__file__).resolve().parents[4] / "data/sol-nll-fold0-30/requests.jsonl"


def clean_message(m: dict, *, is_history_assistant: bool) -> dict:
    """One message ready for the APIs: no private keys, JSON-string tool-call
    arguments, and (for history assistant turns) thinking in `reasoning`."""
    out = {k: copy.deepcopy(v) for k, v in m.items() if not k.startswith("_")}
    rc = out.pop("reasoning_content", None)
    for tc in out.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if isinstance(fn.get("arguments"), (dict, list)):
            fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
    if out.get("role") == "assistant":
        out["content"] = out.get("content") or ""
        if is_history_assistant and isinstance(rc, str) and rc.strip():
            out["reasoning"] = context.normalize(rc)
    return out


def build_record(req: dict) -> tuple[logs.Record, str]:
    """A logs.Record whose context/tools/reply the judge can use, plus the
    teacher thinking string pulled off the final reply."""
    msgs = req["messages"]
    ctx = [clean_message(m, is_history_assistant=True) for m in msgs[:-1]]
    reply = clean_message(msgs[-1], is_history_assistant=False)  # drops its thinking
    thinking = (msgs[-1].get("reasoning_content") or "").strip()
    rec = logs.Record(
        game=req["game"], index=req.get("request_index", 0), analysis_step=0,
        request_in_turn=1, messages=ctx, tools=copy.deepcopy(req["tools"]),
        reply=reply, reasoning_tokens=0,
    )
    return rec, thinking


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N requests only (0=all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=client.MODEL)
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--sol-model", default="gpt-6.1-sol")
    ap.add_argument("--judge-effort", default="high")
    ap.add_argument("--sol-max-tokens", type=int, default=16000)
    ap.add_argument("--retries", type=int, default=10,
                    help="flash regen retry budget (upstream 429s need patience)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    # regen_code calls client.chat with the default retry budget; qwen3.8-flash
    # on Alibaba (its only provider) rate-limits large multimodal requests, so
    # give the regen call more retries/backoff without touching think_gen.
    _orig_chat = client.chat
    def _chat(*args, **kw):
        kw.setdefault("retries", a.retries)
        return _orig_chat(*args, **kw)
    client.chat = _chat

    reqs = [json.loads(l) for l in a.data.open() if l.strip()]
    if a.limit:
        reqs = reqs[: a.limit]
    print(f"{len(reqs)} requests", flush=True)

    jargs = SimpleNamespace(model=a.model, provider=a.provider, max_tokens=a.max_tokens,
                            sol_model=a.sol_model, judge_effort=a.judge_effort,
                            sol_max_tokens=a.sol_max_tokens)
    log_path = a.out / "calls.jsonl"

    def work(req):
        rec, thinking = build_record(req)
        sid = f"{req['game']}-r{req.get('request_index')}"
        if not thinking:
            return {"sample_id": sid, "skipped": "teacher reply has empty thinking"}
        t0 = time.time()
        try:
            v = judge_sol.check_call(rec, thinking, jargs, log_path=log_path)
        except Exception as e:  # noqa: BLE001
            return {"sample_id": sid, "error": repr(e)[:300]}
        sol_code = (prompts.python_args(rec.reply).get("code") or "").strip()
        return {
            "sample_id": sid, "game": req["game"],
            "request_index": req.get("request_index"),
            "functionally_same": v.get("functionally_same"),
            "differences": v.get("differences"),
            "notes": v.get("notes"),
            "regen_error": v.get("regen_error"),
            "secs": round(time.time() - t0, 1),
            "sol_code": sol_code, "regen_code": v.get("regen_code", ""),
            "thinking": thinking,
        }

    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(work, reqs):
            rows.append(r)
            tag = (r.get("functionally_same") if "functionally_same" in r
                   else r.get("regen_error") or r.get("error") or r.get("skipped"))
            print(f"  {r['sample_id']:<22} same={tag}", flush=True)

    (a.out / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    judged = [r for r in rows if isinstance(r.get("functionally_same"), bool)]
    same = [r for r in judged if r["functionally_same"]]
    regen_err = [r for r in rows if r.get("regen_error")]
    errs = [r for r in rows if r.get("error")]
    per_game = {}
    for g in sorted({r.get("game") for r in judged}):
        gj = [r for r in judged if r["game"] == g]
        per_game[g] = {"n": len(gj),
                       "same": round(statistics.mean([r["functionally_same"] for r in gj]), 3)}
    summary = {
        "n_requests": len(rows), "n_judged": len(judged),
        "n_functionally_same": len(same),
        "call_functionally_same": round(len(same) / len(judged), 3) if judged else None,
        "n_regen_error": len(regen_err), "n_error": len(errs),
        "per_game": per_game,
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
