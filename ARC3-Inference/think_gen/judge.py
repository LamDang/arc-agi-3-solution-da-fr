"""Calibration: compare generated thinking with the teacher's real thinking.

    uv run --no-sync python -m think_gen.judge runs/think-calib-b1/sum runs/think-calib-b1/nosum

For every record of an output directory that has both (teachers that return
their reasoning, such as qwen3.8-max), a judge model lists the real
thinking's key points and checks which the generated thinking covers, what it
contradicts, whether it leads to the output and whether it leaks. Results go
to `<dir>/judge/<game>.jsonl` (reruns skip judged records) and a summary
table per directory is printed and written to `<dir>/judge/summary.json`.
"""
import argparse
import concurrent.futures as cf
import json
import re
import statistics
from pathlib import Path

from . import client, prompts
from .generate import append_jsonl, log, read_jsonl


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_one(row: dict, args) -> dict:
    out = {"key": row["key"]}
    for attempt in range(3):
        try:
            res = client.chat([{"role": "user", "content": prompts.judge_prompt(
                row["call"], row["real_reasoning"], row["thinking"])}],
                model=args.model, provider=args.provider, reasoning=True,
                max_tokens=12000, temperature=0.2)
        except client.CallError as e:
            out["error"] = str(e)[:300]
            continue
        verdict = parse_json(res["content"])
        out["cost"] = (res["usage"] or {}).get("cost")
        if verdict and isinstance(verdict.get("covered"), list):
            out.update(verdict)
            out.pop("error", None)
            break
        out["error"] = "unparsable: " + res["content"][:200]
    return out


def summarize(rows: list[dict], verdicts: dict[str, dict]) -> dict:
    cov, contra, leads, leak, ratio = [], [], [], [], []
    for r in rows:
        v = verdicts.get(r["key"])
        if not v or "covered" not in v:
            continue
        c = v["covered"]
        cov.append(sum(1 for x in c if x) / len(c) if c else 0.0)
        contra.append(len(v.get("contradictions") or []))
        leads.append(bool(v.get("leads_to_output")))
        leak.append(bool(v.get("leak")))
        ratio.append(len(r["thinking"]) / max(1, len(r["real_reasoning"])))
    if not cov:
        return {"n": 0}
    return {
        "n": len(cov),
        "coverage_mean": round(statistics.mean(cov), 3),
        "contradictions_mean": round(statistics.mean(contra), 2),
        "with_contradiction": round(sum(1 for x in contra if x) / len(contra), 3),
        "leads_to_output": round(sum(leads) / len(leads), 3),
        "leak": round(sum(leak) / len(leak), 3),
        "length_ratio_median": round(statistics.median(ratio), 2),
        "regex_rejected": sum(1 for r in rows if r.get("status") == "rejected"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--model", default=client.MODEL)
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    for d in args.dirs:
        (d / "judge").mkdir(exist_ok=True)
        rows_all, verdicts = [], {}
        for path in sorted(d.glob("*.jsonl")):
            rows = [r for r in read_jsonl(path) if r.get("thinking") and r.get("real_reasoning")]
            rows_all += rows
            jpath = d / "judge" / path.name
            done = {v["key"]: v for v in read_jsonl(jpath) if "covered" in v}
            todo = [r for r in rows if r["key"] not in done]
            with cf.ThreadPoolExecutor(args.workers) as ex:
                for v in ex.map(lambda r: judge_one(r, args), todo):
                    append_jsonl(jpath, v)
                    if "covered" in v:
                        done[v["key"]] = v
                    else:
                        log(f"[{v['key']}] judge failed: {v.get('error')}")
            verdicts.update(done)
        per_game = {}
        for g in sorted({r["game"] for r in rows_all}):
            per_game[g] = summarize([r for r in rows_all if r["game"] == g], verdicts)
        summary = {"all": summarize(rows_all, verdicts), "games": per_game}
        (d / "judge" / "summary.json").write_text(json.dumps(summary, indent=1))
        log(f"{d}: {json.dumps(summary['all'])}")
        for g, s in per_game.items():
            log(f"  {g}: {json.dumps(s)}")


if __name__ == "__main__":
    main()
