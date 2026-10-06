"""Status of an engine_re play run (read-only): per-game table, phase split, last turns, errors, feature use.

    uv run --no-sync python ../.claude/skills/monitor-run/status.py runs/engine-play/<run> \
        [--compare runs/engine-play/<previous run>] [--last 8] [--since-turn sp80=175,ls20=222]

Run from ARC3-Inference (paths are relative to it). Reads <run>/<game>/result.json and transcript.jsonl;
writes nothing. Transcript record kinds it uses: model turns (finish_reason, usage, tool_calls,
reasoning), tool results (tool, output, called_as), move, batch, commit, engine_change, message,
append, auto_test, nudge, plan_nudge, compact, images, resumed, replay, refused_batch.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def games_of(run: Path) -> list[Path]:
    return sorted(p for p in run.iterdir() if (p / "result.json").exists())


def load_transcript(game: Path) -> list[dict]:
    path = game / "transcript.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written right now
            pass
    return out


def msg_text(record: dict) -> str:
    content = record["message"].get("content")
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def calls_of(record: dict) -> list[tuple[str, str]]:
    out = []
    for call in record.get("tool_calls") or []:
        fn = call.get("function", call)
        out.append((fn.get("name") or "?", str(fn.get("arguments") or "")))
    return out


ERROR_RE = re.compile(r"^(\w+(?:Error|Exception)|E_[A-Z_]+)\b", re.M)


def analyse(game: Path, since: int = 0) -> dict:
    r = json.loads((game / "result.json").read_text())
    recs = load_transcript(game)
    a = {"result": r, "recs": recs, "c": Counter(), "errors": Counter(), "phase_out": Counter(),
         "phase_turns": Counter(), "max_prompt": 0, "turns": {}, "level_changes": []}
    c, errors = a["c"], a["errors"]
    for d in recs:
        t = d.get("turn", 0)
        if "finish_reason" in d:
            u = d.get("usage") or {}
            a["max_prompt"] = max(a["max_prompt"], u.get("prompt_tokens", 0))
            a["phase_out"][d.get("phase")] += u.get("completion_tokens", 0)
            a["phase_turns"][d.get("phase")] += 1
            a["turns"].setdefault(t, {})["model"] = d
        if t < since:
            continue
        if "finish_reason" in d:
            for name, args in calls_of(d):
                c[f"call:{name}"] += 1
                for key in ("replica.step(", "replica.pour(", "make_level(", ".code()"):
                    c[key] += args.count(key)
                c["traced("] += len(re.findall(r"\btraced\(", args))
                c["support("] += len(re.findall(r"(?<![\w.])support\(", args))
                c["notes.md written"] += "notes.md" in args and ("write" in args or "edit" in args)
        elif "tool" in d:
            out = d.get("output") or ""
            a["turns"].setdefault(t, {}).setdefault("tools", []).append(d)
            if d.get("called_as"):
                c[f"{d['called_as']} called as a tool"] += 1
            if "Timed out after" in out:
                c["kernel timeouts"] += 1
            if "Traceback" in out:
                model = "<python>" in out or "engine.py" in out or "YOUR ENGINE RAISED" in out
                kind = "model or engine code" if model else "harness? read it"
                errors[f"traceback ({kind})"] += 1
            for m in ERROR_RE.finditer(out):
                errors[m.group(1)] += 1
            if "Support of your replica" in out:
                c["support blocks"] += 1
                c["support move lines"] += len(re.findall(r"^  moves? \d", out, re.M))
            if "animated over" in out:
                c["animation notes"] += 1
        elif "move" in d:
            m = d["move"]
            c["moves"] += 1
            w = m.get("warning")
            if w:
                c["move warnings (HUD)" if "HUD" in w else "move warnings (other)"] += 1
            if m.get("support"):
                c["move records with support"] += 1
            if m.get("level_solved"):
                # ok is False at a solve (the next level's frame is not drawn yet); a missed prediction
                # shows as "levels completed: the game says n, your replica m" in the verdict
                a["level_changes"].append({"turn": t, "step": m.get("index"),
                                           "predicted": "levels completed" not in (m.get("verdict") or "")})
        elif "message" in d:
            text = msg_text(d)
            if text.startswith("Plan the next moves"):
                c["PLAN messages"] += 1
                c["PLAN with notes.md"] += "notes.md" in text
            if "Unfamiliar elements" in text:
                c["unfamiliar-elements lists"] += 1
        elif "append" in d:
            if "Warning" in d["append"] or "WARNING" in d["append"]:
                c["warnings in harness appends"] += 1
        for kind in ("compact", "nudge", "plan_nudge", "refused_batch", "resumed", "commit", "batch",
                     "engine_change", "auto_test"):
            if kind in d:
                c[kind] += 1
    # level-change costs: from each level solve to the next commit and to the next batch
    for lc in a["level_changes"]:
        t0 = lc["turn"]
        nxt = {"commit": None, "batch": None}
        for d in recs:
            for kind in nxt:
                if kind in d and nxt[kind] is None and d["turn"] > t0:
                    nxt[kind] = d["turn"]
        for kind, t1 in nxt.items():
            lc[f"to_{kind}"] = t1
            lc[f"{kind}_out"] = sum((d.get("usage") or {}).get("completion_tokens", 0) for d in recs
                                    if "finish_reason" in d and t0 < d["turn"] <= (t1 or 10**9))
    return a


def one_line(t: int, turn: dict, width: int) -> str:
    m = turn.get("model", {})
    calls = calls_of(m)
    names = ",".join(n for n, _ in calls) or "-"
    text = ""
    for name, args in calls:
        hit = re.search(r'"(?:note|message)":\s*"((?:[^"\\]|\\.)*)', args)
        if hit:
            text = hit.group(1)
            break
    if not text:
        tools = turn.get("tools") or []
        text = (tools[-1].get("output") if tools else "") or (m.get("reasoning") or "")
    text = " ".join(text.replace("\\n", " ").split())
    return f"  T{t} {m.get('phase', '?'):4s} s{m.get('step', '?')} {names:14s} {text[:width]}"


def table(run: Path, since: dict) -> list[dict]:
    rows = []
    print(f"== {run}")
    hdr = "game  status        turns acts per-level              lvls score batch mism phase cost$  out_tok  maxprompt min"
    print(hdr)
    for g in games_of(run):
        a = analyse(g, since.get(g.name, 0))
        r, u = a["result"], a["result"].get("usage", {})
        per = " ".join(str(x) for x in r.get("actions_per_level") or [])
        print(f"{g.name:5s} {str(r.get('status')):13s} {r.get('turns', 0):5} {r.get('actions', 0):4} {per:24s} "
              f"{r.get('levels_completed', 0):4} {r.get('score') or 0:5.1f} {r.get('batches', 0):5} "
              f"{r.get('mismatches', 0):4} {str(r.get('phase')):5s} {u.get('cost_usd', 0):5.2f} "
              f"{u.get('completion_tokens', 0):8,} {max(u.get('max_prompt_tokens', 0), a['max_prompt']):9,} "
              f"{r.get('minutes', 0):5.0f}")
        rows.append((g, a))
    summary = run / "summary.md"
    print(f"summary.md: {'exists (run finished)' if summary.exists() else 'not yet'}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--compare", type=Path, help="a previous run on the same games (table only)")
    ap.add_argument("--last", type=int, default=8, help="one-line summaries of the last N model turns")
    ap.add_argument("--width", type=int, default=110, help="characters of note/output per line")
    ap.add_argument("--since-turn", default="", help="count features from this turn on, e.g. sp80=175,ls20=222")
    args = ap.parse_args()
    if not args.run.is_dir():
        sys.exit(f"no run directory {args.run}")
    since = {k: int(v) for k, v in (x.split("=") for x in args.since_turn.split(",") if x)}
    rows = table(args.run, since)
    if args.compare:
        table(args.compare, {})
    for g, a in rows:
        c = a["c"]
        print(f"\n--- {g.name}" + (f" (features from turn {since[g.name]})" if g.name in since else ""))
        split = ", ".join(f"{p} {a['phase_turns'][p]} turns / {a['phase_out'][p]:,} out" for p in a["phase_turns"])
        print(f"phase split: {split}")
        print("counts: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if v))
        print("errors: " + (", ".join(f"{k} {v}" for k, v in a["errors"].most_common()) or "none"))
        for lc in a["level_changes"]:
            print(f"level solve T{lc['turn']} step {lc['step']} (replica predicted the solve: {lc['predicted']}): "
                  f"next commit T{lc['to_commit']} (+{lc['commit_out']:,} out), "
                  f"next batch T{lc['to_batch']} (+{lc['batch_out']:,} out)")
        print(f"last {args.last} turns:")
        for t in sorted(a["turns"])[-args.last:]:
            if "model" in a["turns"][t]:
                print(one_line(t, a["turns"][t], args.width))


if __name__ == "__main__":
    main()
