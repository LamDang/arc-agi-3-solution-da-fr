"""Judge generated thinking where the teacher's real thinking is hidden
(gpt-6.1-sol): four checks per record, each against something that is known.

    uv run --no-sync python -m think_gen.judge_sol \
        --run runs/gpt61sol-features-25games \
        --gen runs/think-sol-b4/dev20 \
        --manifest experiments/teacher-reasoning/evalset/sol25_dev20.json

For every generated record (status ok, with thinking) it runs:

  1 code   judge model: does the thinking's plan match the python code the
           teacher ran? -> code_leads, code_disagreements;
  2 words  judge model: how much of the teacher's own stated reasoning,
           description and summary does the thinking cover? -> words_coverage,
           with a contradiction or not;
  3 fact   gpt-6.1-sol, given the real context and images, flags claims in the
           thinking that the state does not support -> fact_errors, grounded;
  4 call   regenerate a python call from the thinking (code-only request, the
           student model flash) and ask the judge model whether it makes the
           same next move as the teacher -> call_same_move.

The judge model (checks 1, 2 and 4's comparison) and the fact-check model are
OpenAI gpt-6 models reached through the Responses API (`--judge-model`,
default gpt-6-luna at high effort; `--sol-model`, default gpt-6.1-sol at
xhigh). Regeneration (check 4) uses the student model flash on OpenRouter, so
it tests whether the thinking leads the student back to the teacher's call.

The source `--run` provides each record's context, call, stated words and
tools; `--gen` is a think_gen.generate output directory. `--manifest`/`--split`
restrict to an evaluation set. Verdicts go to `<gen>/judge_sol/<game>.jsonl`
(reruns skip finished records, per check) and a summary table per check is
printed and written to `<gen>/judge_sol/summary.json`.
"""
import argparse
import concurrent.futures as cf
import json
import re
import statistics
from pathlib import Path

from . import client, context, logs, prompts
from .generate import append_jsonl, log, read_jsonl


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def responses_json(msgs: list, model: str, effort: str, want: str, max_tokens: int) -> dict:
    """One OpenAI Responses JSON call, retried until it parses and has `want`."""
    last = ""
    for _ in range(3):
        try:
            res = client.openai_responses(msgs, model=model, effort=effort,
                                          json_mode=True, max_output_tokens=max_tokens)
        except client.CallError as e:
            last = str(e)[:200]
            continue
        v = parse_json(res["content"])
        if v is not None and want in v:
            u = res["usage"] or {}
            v["_tokens"] = {"in": u.get("input_tokens"), "out": u.get("output_tokens")}
            return v
        last = "unparsable: " + res["content"][:150]
    return {"error": last}


def judge_json(prompt: str, args, want: str) -> dict:
    """A judge-model call on a single user prompt (checks 1, 2 and 4b)."""
    return responses_json([{"role": "user", "content": prompt}], args.judge_model,
                          args.judge_effort, want, args.judge_max_tokens)


def check_code(rec: logs.Record, thinking: str, args) -> dict:
    return judge_json(prompts.judge_code_prompt(context.call_text(rec.reply), thinking),
                      args, "leads_to_call")


def check_words(rec: logs.Record, thinking: str, args) -> dict:
    a = prompts.python_args(rec.reply)
    if not any((a.get("reasoning"), a.get("description"), rec.summary)):
        return {"skipped": "no stated words"}
    return judge_json(prompts.judge_words_prompt(
        a.get("reasoning") or "", a.get("description") or "", rec.summary, thinking), args, "covered")


def check_fact(rec: logs.Record, thinking: str, args) -> dict:
    """gpt-6.1-sol fact-checks the thinking against its own real context."""
    msgs = list(rec.messages) + [{"role": "user", "content": prompts.judge_fact_prompt(thinking)}]
    v = responses_json(msgs, args.sol_model, args.sol_effort, "grounded", args.sol_max_tokens)
    v["_sol_tokens"] = v.pop("_tokens", None)
    return v


def regen_code(rec: logs.Record, thinking: str, args) -> tuple[str, dict]:
    """Flash's python call, made from the thinking in the code-only request."""
    msgs, tools = context.code_only_request(rec.messages, rec.tools)
    msgs += [{"role": "user", "content": thinking},
             {"role": "user", "content": prompts.REGEN_NOTE}]
    try:
        res = client.chat(msgs, model=args.model, provider=args.provider, tools=tools,
                          tool_choice={"type": "function", "function": {"name": "python"}},
                          reasoning=True, max_tokens=args.max_tokens, temperature=0.4)
    except client.CallError as e:
        return "", {"error": str(e)[:200]}
    for c in res.get("tool_calls") or []:
        fn = c.get("function") or {}
        if fn.get("name") == "python":
            try:
                return json.loads(fn.get("arguments") or "{}").get("code") or "", {"cost": (res["usage"] or {}).get("cost") or 0}
            except json.JSONDecodeError:
                return "", {"error": "bad regenerated arguments"}
    return "", {"error": "no python call", "content": (res.get("content") or "")[:150]}


def check_call(rec: logs.Record, thinking: str, args) -> dict:
    sol_code = (prompts.python_args(rec.reply).get("code") or "").strip()
    if not sol_code:
        return {"skipped": "teacher call has no python code"}
    code, meta = regen_code(rec, thinking, args)
    if not code.strip():
        return {"regen_error": meta.get("error", "empty"), "regen_code": ""}
    v = judge_json(prompts.judge_call_prompt(sol_code, code), args, "same_next_move")
    v["regen_code"] = code
    v["_regen_cost"] = meta.get("cost", 0)
    return v


CHECKS = {"code": check_code, "words": check_words, "fact": check_fact, "call": check_call}


def judge_one(rec: logs.Record, thinking: str, done: dict, args) -> dict:
    out = {"key": rec.key, "game": rec.game}
    out.update(done)  # keep checks already finished on a rerun
    for name in args.checks:
        if name in out:
            continue
        out[name] = CHECKS[name](rec, thinking, args)
    return out


def _f(xs):
    return round(statistics.mean(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict:
    cov, codelead, codedis, grounded, facts, same = [], [], [], [], [], []
    for r in rows:
        w = r.get("words") or {}
        if isinstance(w.get("covered"), list) and w["covered"]:
            cov.append(sum(1 for x in w["covered"] if x) / len(w["covered"]))
        c = r.get("code") or {}
        if "leads_to_call" in c:
            codelead.append(bool(c["leads_to_call"]))
            codedis.append(len(c.get("disagreements") or []))
        f = r.get("fact") or {}
        if "grounded" in f:
            grounded.append(bool(f["grounded"]))
            facts.append(len(f.get("errors") or []))
        cl = r.get("call") or {}
        if "same_next_move" in cl:
            same.append(bool(cl["same_next_move"]))
    return {
        "n": len(rows),
        "words_coverage": _f(cov),
        "words_n": len(cov),
        "code_leads_to_call": _f(codelead),
        "code_disagreements_mean": _f(codedis),
        "fact_grounded": _f(grounded),
        "fact_errors_mean": _f(facts),
        "fact_n": len(grounded),
        "call_same_next_move": _f(same),
        "call_n": len(same),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="source run (context, calls, stated words)")
    ap.add_argument("--gen", type=Path, required=True, help="think_gen.generate output directory")
    ap.add_argument("--manifest", type=Path, help="restrict to this evaluation manifest's records")
    ap.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    ap.add_argument("--games", nargs="*", help="game id prefixes (default: all in --gen)")
    efforts = ["minimal", "low", "medium", "high", "xhigh"]
    ap.add_argument("--checks", nargs="+", choices=list(CHECKS), default=list(CHECKS))
    ap.add_argument("--judge-model", default="gpt-6-luna", help="checks 1, 2 and 4b (Responses API)")
    ap.add_argument("--judge-effort", default="high", choices=efforts)
    ap.add_argument("--judge-max-tokens", type=int, default=8000)
    ap.add_argument("--sol-model", default="gpt-6.1-sol", help="fact-check: sees the real context")
    ap.add_argument("--sol-effort", default="xhigh", choices=efforts)
    ap.add_argument("--sol-max-tokens", type=int, default=16000)
    ap.add_argument("--model", default=client.MODEL, help="call regenerator: the student model (flash)")
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    keys = None
    if args.manifest:
        man = json.loads(args.manifest.read_text())
        keys = {i["key"] for i in man["items"] if args.split in ("all", i["split"])}

    (args.gen / "judge_sol").mkdir(exist_ok=True)
    todo, rows_all = [], []
    for gpath in sorted(args.gen.glob("*.jsonl")):
        if args.games and not any(gpath.stem.rsplit("_p", 1)[0].startswith(g) for g in args.games):
            continue
        gen = {r["key"]: r for r in read_jsonl(gpath)
               if r.get("thinking") and r.get("status") == "ok"
               and (keys is None or r["key"] in keys)}
        if not gen:
            continue
        srcs = logs.request_logs(args.run, [gpath.stem.rsplit("_p", 1)[0]])
        recs = {r.key: r for p in srcs for r in logs.read_log(p)}
        jpath = args.gen / "judge_sol" / gpath.name
        done = {v["key"]: v for v in read_jsonl(jpath)}
        for key, grow in gen.items():
            rec = recs.get(key)
            if rec is None:
                log(f"[{key}] not in {args.run}; skipped")
                continue
            prior = done.get(key, {})
            if all(c in prior for c in args.checks):
                rows_all.append(prior)
                continue
            todo.append((rec, grow["thinking"], prior, jpath))

    log(f"{len(rows_all)} already judged, {len(todo)} to judge ({', '.join(args.checks)})")

    def work(item):
        rec, thinking, prior, jpath = item
        v = judge_one(rec, thinking, prior, args)
        append_jsonl(jpath, v)
        return v

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for v in ex.map(work, todo):
            rows_all.append(v)
            flags = []
            for name in args.checks:
                d = v.get(name) or {}
                if "error" in d or "regen_error" in d:
                    flags.append(f"{name}!={d.get('error') or d.get('regen_error')}"[:40])
            if flags:
                log(f"[{v['key']}] {' '.join(flags)}")

    # a judged file may hold several lines per key across reruns; keep the last
    for jpath in sorted((args.gen / "judge_sol").glob("*.jsonl")):
        latest = {v["key"]: v for v in read_jsonl(jpath)}
        jpath.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in latest.values()))

    by_key = {r["key"]: r for r in rows_all}
    rows = list(by_key.values())
    per_game = {g: summarize([r for r in rows if r["game"] == g]) for g in sorted({r["game"] for r in rows})}
    summary = {"all": summarize(rows), "games": per_game, "checks": args.checks}
    (args.gen / "judge_sol" / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"all: {json.dumps(summary['all'])}")
    for g, s in per_game.items():
        log(f"  {g}: {json.dumps(s)}")


if __name__ == "__main__":
    main()
