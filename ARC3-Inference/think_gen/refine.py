"""Second pass: the student revises its own thinking from the sol-judge feedback.

    uv run --no-sync python -m think_gen.refine \
        --run runs/gpt61sol-features-25games \
        --gen runs/think-sol-b4/dev20 \
        --out runs/think-sol-b4/refine/dev20 \
        --manifest experiments/teacher-reasoning/evalset/sol25_dev20.json

For each generated record whose sol-judge verdicts (`<gen>/judge_sol`) show an
issue -- coverage below 1, a code disagreement, or a fact the judge did not
find grounded -- the student (flash) is given its first draft and the feedback
(each check's note plus the specific points) and asked to keep what the draft
got right and fix what the feedback raises. Records the judge passed are copied
through unchanged. Output rows match think_gen.generate, so think_gen.judge_sol
and the annotation page read the refined run the same way; refined rows carry
`prompt_version` "<v>r" and `refined: true`.
"""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from . import checks, client, context, logs, prompts
from .generate import append_jsonl, log, read_jsonl


def has_issue(v: dict) -> bool:
    """The judge flagged coverage, code consistency or fact grounding."""
    w = v.get("words") or {}
    c = v.get("code") or {}
    f = v.get("fact") or {}
    cov = w.get("covered") or []
    if cov and not all(cov):
        return True
    if w.get("contradictions"):
        return True
    if c.get("disagreements"):
        return True
    if "grounded" in f and not f["grounded"]:
        return True
    return False


def refine_one(rec: logs.Record, row: dict, verdict: dict, args, out_path: Path) -> dict:
    draft = row.get("thinking") or ""
    feedback = prompts.refine_feedback(verdict)
    msgs = context.history(rec.messages, {}, args.history)
    msgs.append({"role": "user", "content": prompts.refine_prompt(rec.reply, draft, feedback)})
    context_text = json.dumps(rec.messages, ensure_ascii=False) + json.dumps(rec.reply, ensure_ascii=False)
    target = row.get("target_words") or 0
    attempts, usages, text, chk, extra = 0, [], "", {}, {}
    while attempts < args.max_attempts:
        attempts += 1
        try:
            out = client.chat(msgs, model=args.model, provider=args.provider, tools=rec.tools,
                              reasoning=args.reasoning, max_tokens=args.max_tokens,
                              log_path=args.out / "requests" / f"{rec.game}.jsonl",
                              log_tag={"key": rec.key, "check": "refine"})
        except client.CallError as e:
            log(f"[{rec.key}] refine call failed: {e}")
            usages.append({"error": str(e)[:300]})
            continue
        usages.append(out["usage"])
        text = checks.clean(out["content"])
        chk = checks.check(text, rec.reply, context_text, target // 2)
        extra = {"flash_thinking": out["reasoning"], "finish_reason": out["finish_reason"], "secs": out["secs"]}
        if chk["ok"]:
            break
        log(f"[{rec.key}] refine attempt {attempts} rejected: {chk}")
    cost = sum(u.get("cost") or 0 for u in usages)
    new = dict(row)
    new.update(thinking=text, status="ok" if chk.get("ok") else "rejected", attempts=attempts,
               checks=chk, usage=usages, cost=round(cost, 6), refined=True,
               draft=draft, feedback=feedback,
               prompt_version=(row.get("prompt_version") or "b4") + "r", **extra)
    append_jsonl(out_path, new)
    log(f"[{rec.key}] refined {chk.get('chars', 0)} chars ({chk.get('words')} words), ${cost:.4f}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="source run (context, call, stated words)")
    ap.add_argument("--gen", type=Path, required=True, help="first-pass think_gen.generate output")
    ap.add_argument("--out", type=Path, required=True, help="refined output directory")
    ap.add_argument("--judge", type=Path, help="sol-judge verdicts (default <gen>/judge_sol)")
    ap.add_argument("--manifest", type=Path, help="restrict to this manifest's records")
    ap.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--history", choices=context.HISTORY_MODES, default="native")
    ap.add_argument("--reasoning", action="store_true", help="let flash reason before writing")
    ap.add_argument("--model", default=client.MODEL)
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    judge_dir = args.judge or (args.gen / "judge_sol")
    args.out.mkdir(parents=True, exist_ok=True)

    keys = None
    if args.manifest:
        man = json.loads(args.manifest.read_text())
        keys = {i["key"] for i in man["items"] if args.split in ("all", i["split"])}

    todo, copied, passed = [], 0, 0
    for gpath in sorted(args.gen.glob("*.jsonl")):
        if args.games and not any(gpath.stem.rsplit("_p", 1)[0].startswith(g) for g in args.games):
            continue
        gen = {r["key"]: r for r in read_jsonl(gpath)
               if r.get("thinking") and r.get("status") == "ok" and (keys is None or r["key"] in keys)}
        if not gen:
            continue
        verdicts = {v["key"]: v for v in read_jsonl(judge_dir / gpath.name)}
        recs = {r.key: r for p in logs.request_logs(args.run, [gpath.stem.rsplit("_p", 1)[0]])
                for r in logs.read_log(p)}
        out_path = args.out / gpath.name
        done = {r["key"] for r in read_jsonl(out_path)}
        for key, row in gen.items():
            if key in done:
                continue
            v = verdicts.get(key) or {}
            if has_issue(v) and key in recs:
                todo.append((recs[key], row, v, out_path))
            else:  # judge passed it (or no verdict / record): keep the draft as-is
                append_jsonl(out_path, {**row, "refined": False})
                copied += 1
                passed += 1 if v else 0

    log(f"{len(todo)} records to refine, {copied} copied through unchanged")
    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda j: refine_one(j[0], j[1], j[2], args, j[3]), todo))
    log(f"done: refined {len(todo)}, kept {copied}")


if __name__ == "__main__":
    main()
