"""Generate the thinking before each teacher call of a base-agent run.

    uv run --no-sync python -m think_gen.generate \
        --run runs/base-gpt61sol-dfranzen --out runs/think-sol-b1 --games ft09

Games run in parallel; the requests of a game run in order, each in the
context of the thinking already generated for the earlier ones. One JSON line
per record goes to `<out>/<game>.jsonl` as soon as it is done, and a rerun
skips the records already there.

With `--manifest` (from think_gen.evalset) only the manifest's records of
`--split` are generated, independently and in parallel, with the teacher's
real thinking in the history: each request is then judged on its own,
without errors carried over from earlier generated turns.

Summaries (`--summary`):
  teacher  the teacher's reasoning summary when it has one (gpt-6.1-sol);
  synth    a summary written by flash from the teacher's real reasoning, in
           the teacher-summary style (calibration on qwen3.8-max runs);
           cached in `<out>/summaries/<game>.jsonl`;
  none     never.
A record with no summary gets the short-thinking instruction. A record whose
teacher spent 0 reasoning tokens gets empty thinking, as the teacher had.

`--prompt b3` (teachers whose python calls state `reasoning` and
`description`) uses no summary: the thinking works through the stated
reasoning and ends planning the described move, at a length set by the
teacher's reasoning tokens; 0-token records get a few sentences. `--prompt b4`
makes it the full working behind the stated reasoning, traced back through
earlier turns, at least 3 times the stated reasoning's length.
"""
import argparse
import concurrent.futures as cf
import json
import random
import sys
import threading
from pathlib import Path

from . import checks, client, context, logs, prompts

_lock = threading.Lock()


def log(msg: str):
    with _lock:
        print(msg, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append_jsonl(path: Path, row: dict):
    with _lock, open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def style_examples(run_dir: Path, n: int = 4, seed: int = 0) -> list[str]:
    """Real teacher summaries to imitate, from a run that has them."""
    pool = [r.summary for p in logs.request_logs(run_dir) for r in logs.read_log(p) if r.summary]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def synth_summary(rec: logs.Record, cache: Path, examples: list[str], args) -> str:
    done = {r["key"]: r["summary"] for r in read_jsonl(cache)}
    if rec.key in done:
        return done[rec.key]
    if not rec.real_reasoning.strip():
        s = ""
    else:
        out = client.chat([{"role": "user", "content": prompts.synth_summary_prompt(
            rec.real_reasoning, examples)}], model=args.model, provider=args.provider,
            reasoning=False, max_tokens=1024)
        s = out["content"].strip()
    append_jsonl(cache, {"key": rec.key, "summary": s})
    return s


def generate_one(rec: logs.Record, thinking: dict[str, str], args, examples: list[str],
                 out_path: Path) -> dict:
    """Generate, check and write the thinking of one record, with `thinking`
    (by assistant-turn ref) filling earlier turns. Returns the row written."""
    summary, source = "", "none"
    if args.summary == "teacher" and rec.summary:
        summary, source = rec.summary, "teacher"
    elif args.summary == "synth":
        summary = synth_summary(rec, args.out / "summaries" / f"{rec.game}.jsonl", examples, args)
        source = "synth" if summary else "none"
    row = {
        "key": rec.key, "game": rec.game, "index": rec.index,
        "analysis_step": rec.analysis_step, "request_in_turn": rec.request_in_turn,
        "ref": context.message_ref(rec.reply), "call": context.call_text(rec.reply),
        "teacher_reasoning_tokens": rec.reasoning_tokens,
        "summary": summary, "summary_source": source,
        "real_reasoning": rec.real_reasoning,
        "prompt_version": args.prompt, "history": args.history,
        "history_source": args.history_source, "model": args.model,
        "flash_reasoning": args.reasoning,
    }
    if args.prompt in ("b3", "b4"):
        row.update(summary="", summary_source="none")
    elif rec.reasoning_tokens == 0 and not rec.real_reasoning.strip():
        row.update(thinking="", status="teacher_empty", attempts=0, checks={}, usage={}, cost=0)
        append_jsonl(out_path, row)
        return row
    msgs = context.history(rec.messages, thinking, args.history)
    if args.prompt == "b4":
        prompt, row["target_words"] = prompts.reconstruct_prompt_b4(rec.reply, rec.reasoning_tokens)
    elif args.prompt == "b3":
        prompt = prompts.reconstruct_prompt_b3(rec.reply, rec.reasoning_tokens)
    else:
        prompt = prompts.reconstruct_prompt(row["call"], summary, args.prompt, args.history)
    msgs.append({"role": "user", "content": prompt})
    context_text = json.dumps(rec.messages, ensure_ascii=False) + json.dumps(rec.reply, ensure_ascii=False)
    attempts, usages, text, chk, extra = 0, [], "", {}, {}
    while attempts < args.max_attempts:
        attempts += 1
        try:
            out = client.chat(msgs, model=args.model, provider=args.provider, tools=rec.tools,
                              reasoning=args.reasoning, max_tokens=args.max_tokens)
        except client.CallError as e:
            log(f"[{rec.key}] call failed: {e}")
            usages.append({"error": str(e)[:300]})
            continue
        usages.append(out["usage"])
        text = checks.clean(out["content"])
        chk = checks.check(text, rec.reply, context_text)
        extra = {"flash_thinking": out["reasoning"], "finish_reason": out["finish_reason"],
                 "secs": out["secs"]}
        if chk["ok"]:
            break
        log(f"[{rec.key}] attempt {attempts} rejected: {chk}")
    c = sum(u.get("cost") or 0 for u in usages)
    row.update(thinking=text, status="ok" if chk.get("ok") else "rejected", attempts=attempts,
               checks=chk, usage=usages, cost=round(c, 6), **extra)
    append_jsonl(out_path, row)
    u = usages[-1] if usages else {}
    log(f"[{rec.key}] step {rec.analysis_step}.{rec.request_in_turn} {row['status']} "
        f"{chk.get('chars', 0)} chars, summary={source}, prompt {u.get('prompt_tokens')} "
        f"(cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')}), ${c:.4f}")
    return row


def run_game(path: Path, args, examples: list[str]) -> dict:
    """All records of a game in order, each in the context of the thinking
    generated for the earlier ones."""
    records = logs.read_log(path)
    if args.start:
        records = records[args.start:]
    if args.limit:
        records = records[:args.limit]
    game = records[0].game if records else path.name
    out_path = args.out / f"{game}.jsonl"
    done = {r["key"]: r for r in read_jsonl(out_path)}
    # thinking already generated, by assistant-turn ref, for the history
    thinking = {r["ref"]: r["thinking"] for r in done.values() if r.get("thinking")}
    cost = 0.0
    for rec in records:
        if rec.key in done:
            continue
        row = generate_one(rec, thinking, args, examples, out_path)
        cost += row.get("cost") or 0
        if row.get("thinking"):
            thinking[row["ref"]] = row["thinking"]
    return {"game": game, "records": len(records), "cost": round(cost, 4)}


def run_manifest(args, examples: list[str]) -> list[dict]:
    """The records of an evaluation manifest (think_gen.evalset), independently
    and in parallel: earlier turns carry the teacher's real thinking."""
    manifest = json.loads(args.manifest.read_text())
    keys = {i["key"] for i in manifest["items"] if args.split in ("all", i["split"])}
    jobs, results = [], {}
    for path in logs.request_logs(args.run, args.games):
        records = logs.read_log(path)
        mine = [r for r in records if r.key in keys]
        if not mine:
            continue
        out_path = args.out / f"{mine[0].game}.jsonl"
        done = {r["key"] for r in read_jsonl(out_path)}
        teacher = {context.message_ref(r.reply): r.real_reasoning for r in records if r.real_reasoning}
        jobs += [(r, teacher, out_path) for r in mine if r.key not in done]
        results[mine[0].game] = {"game": mine[0].game, "records": len(mine), "cost": 0.0}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(lambda j: generate_one(j[0], j[1], args, examples, j[2]), jobs):
            results[row["game"]]["cost"] += row.get("cost") or 0
    return list(results.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory with request logs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", nargs="*", help="game id prefixes (default: all)")
    ap.add_argument("--manifest", type=Path, help="evaluation manifest (think_gen.evalset): only its "
                    "records, independently, with the teacher's real thinking in the history")
    ap.add_argument("--split", choices=["dev", "eval", "all"], default="dev")
    ap.add_argument("--start", type=int, default=0, help="skip the first N records of each game")
    ap.add_argument("--limit", type=int, default=0, help="at most N records per game")
    ap.add_argument("--summary", choices=["teacher", "synth", "none"], default="teacher")
    ap.add_argument("--style-run", type=Path, default=Path("runs/base-gpt61sol-dfranzen"),
                    help="run whose teacher summaries set the style of synthetic ones")
    ap.add_argument("--prompt", choices=sorted(prompts.RECONSTRUCT_PROMPTS), default=prompts.PROMPT_VERSION,
                    help="reconstruction prompt version")
    ap.add_argument("--history", choices=context.HISTORY_MODES, default="native",
                    help="where earlier turns' generated thinking goes: the `reasoning` field, or the text")
    ap.add_argument("--reasoning", action="store_true", help="let flash reason before writing")
    ap.add_argument("--model", default=client.MODEL)
    ap.add_argument("--provider", default=client.PROVIDER)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args(argv)
    args.history_source = "teacher" if args.manifest else "generated"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summaries").mkdir(exist_ok=True)
    (args.out / "settings.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1))
    examples = style_examples(args.style_run) if args.summary == "synth" else []
    paths = logs.request_logs(args.run, args.games)
    if not paths:
        sys.exit(f"no request logs in {args.run}")
    if args.manifest:
        results = run_manifest(args, examples)
    else:
        with cf.ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda p: run_game(p, args, examples), paths))
    total = sum(r["cost"] for r in results)
    for r in results:
        log(f"{r['game']}: {r['records']} records, ${r['cost']:.4f}")
    log(f"total ${total:.4f}")


if __name__ == "__main__":
    main()
