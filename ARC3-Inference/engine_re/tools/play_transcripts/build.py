"""Build a transcript page for a play-and-model run (engine_re.play_agent), turn by turn, per game.

    uv run --no-sync python engine_re/tools/play_transcripts/build.py runs/engine-play/<run> <out.html> \
        [--compare runs/engine-play/<other run>] [--title "..."]

Reads <run>/config.json and, per game directory, result.json, transcript.jsonl and images/. Every
model turn becomes a card: the end of its reasoning, its message, the calls it made (python, edit_file,
commit_moves, commit_engine, run_tests), what came back, the moves the harness sent and whether each
matched, the support lines of the batch, the harness messages (PLAN and FIT) with their pictures, and
the automatic tests. The header compares the run with `--compare` (the previous run on the same games).
The page is built from template.html with the data gzipped and base64-embedded; it can be rebuilt while
the run goes on (an unfinished game shows its state so far).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line being written
    return rows


def _data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def game_data(gd: Path) -> dict | None:
    result_path = gd / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recs = _load_jsonl(gd / "transcript.jsonl")
    system = ""
    turns: dict[int, dict] = {}

    def turn(t: int) -> dict:
        if t not in turns:
            turns[t] = {
                "t": t, "min": None, "phase": None, "step": None, "reasoning": "", "content": "", "calls": [], "outputs": [],
                "moves": [], "batch": None, "commit": None, "edits": [], "harness": [], "images": [], "tok": [None, None, None],
                "tests": [], "nudges": 0,
            }
        return turns[t]

    for r in recs:
        t = r.get("turn")
        if t is None:
            continue
        if "message" in r:
            m = r["message"]
            if isinstance(m, dict) and m.get("role") == "system":
                system = m.get("content", "")
                continue
            text = m.get("content", "") if isinstance(m, dict) else str(m)
            # the message logged with turn t is what the model reads at turn t + 1
            turn(t)["harness"].append({"kind": "message", "phase": r.get("phase"), "text": text})
            continue
        tr = turn(t)
        if "finish_reason" in r:
            u = r.get("usage") or {}
            tr.update({
                "min": r.get("elapsed_min"), "phase": r.get("phase"), "step": r.get("step"),
                "reasoning": r.get("reasoning") or "", "content": r.get("content") or "",
                "calls": [{"name": c["function"]["name"], "args": c["function"]["arguments"]} for c in (r.get("tool_calls") or [])],
                "tok": [u.get("prompt_tokens"), u.get("completion_tokens"), (u.get("completion_tokens_details") or {}).get("reasoning_tokens")],
            })
        elif "tool" in r:
            tr["outputs"].append({"tool": r["tool"], "as": r.get("called_as"), "text": r.get("output") or "", "seconds": r.get("seconds")})
        elif "move" in r:
            tr["moves"].append(r["move"])
        elif "batch" in r:
            tr["batch"] = r["batch"]
        elif "commit" in r:
            tr["commit"] = r["commit"]
        elif "engine_change" in r:
            tr["edits"].append(r["engine_change"].get("summary", ""))
        elif "append" in r:
            tr["harness"].append({"kind": "append", "phase": r.get("phase"), "text": r["append"].strip()})
        elif "auto_test" in r:
            tr["tests"].append(r["auto_test"].splitlines()[-1] if r["auto_test"] else "")
        elif "step_start" in r:
            s = r["step_start"]
            tr["harness"].append({"kind": "fit", "phase": "fit", "text": f"Fit round opens on step {s.get('step')}", "step": s.get("step")})
        elif "nudge" in r or "plan_nudge" in r:
            tr["nudges"] += 1
        elif "images" in r:
            for path, cap in zip(r["images"], r.get("captions") or [""] * len(r["images"])):
                uri = _data_uri(gd / path)
                if uri:
                    # pictures logged with turn t are shown with the message the model reads at turn t
                    tr["images"].append({"src": uri, "caption": cap})
        elif "plan" in r:
            p = r["plan"]
            if p.get("step") is not None and tr["phase"] is None:
                tr["phase"] = "plan"
    # pictures were logged under the turn they are shown at; messages under the turn before: move the pictures
    # to the message they illustrate (the previous turn's harness message) so a card shows both together
    for t in sorted(turns):
        imgs = turns[t]["images"]
        if imgs and t - 1 in turns and any(h["kind"] == "message" for h in turns[t - 1]["harness"]):
            turns[t - 1].setdefault("after_images", []).extend(imgs)
            turns[t]["images"] = []
    u = result.get("usage") or {}
    meta = {
        "status": result.get("status"), "outcome": result.get("outcome"), "turns": result.get("turns"), "minutes": result.get("minutes"),
        "cost": round(u.get("cost_usd", 0.0), 3), "out": u.get("completion_tokens", 0), "reasoning": u.get("reasoning_tokens", 0),
        "prompt": u.get("prompt_tokens", 0), "actions": result.get("actions"), "levels": result.get("levels_completed"),
        "win_levels": result.get("win_levels"), "per_level": result.get("actions_per_level"), "baseline": result.get("baseline_actions"),
        "score": result.get("score"), "batches": result.get("batches"), "moves": result.get("moves_sent"),
        "mismatches": result.get("mismatches"), "fit_rounds": len(result.get("fit_rounds") or []),
        "phase_turns": result.get("phase_turns"), "plan_nudges": result.get("plan_nudges"), "phase": result.get("phase"),
        "steps": result.get("trace_steps"),
    }
    return {"meta": meta, "system": system, "turns": [turns[k] for k in sorted(turns) if turns[k]["min"] is not None or turns[k]["harness"]]}


def compare_data(run: Path, games: list[str]) -> dict:
    out = {}
    for g in games:
        p = run / g / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        u = r.get("usage") or {}
        out[g] = {
            "status": r.get("status"), "score": r.get("score"), "levels": r.get("levels_completed"), "win_levels": r.get("win_levels"),
            "actions": r.get("actions"), "turns": r.get("turns"), "cost": round(u.get("cost_usd", 0.0), 2), "mismatches": r.get("mismatches"),
            "batches": r.get("batches"), "per_level": r.get("actions_per_level"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lede", default=None)
    args = ap.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8")) if (args.run / "config.json").exists() else {}
    games = [g.strip() for g in str(config.get("games", "")).split(",") if g.strip()] or sorted(
        d.name for d in args.run.iterdir() if (d / "result.json").exists())
    data = {
        "title": args.title or args.run.name, "lede": args.lede or "", "run": args.run.name, "config": config,
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "compare": compare_data(args.compare, games) if args.compare else {},
        "compare_name": args.compare.name if args.compare else None, "games": {},
    }
    system = ""
    for g in games:
        gdata = game_data(args.run / g)
        if gdata is None:
            continue
        system = system or gdata.pop("system", "")
        gdata.pop("system", None)
        data["games"][g] = gdata
    data["system"] = system
    raw = json.dumps(data, separators=(",", ":")).encode()
    packed = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
    page = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", packed).replace("__TITLE__", data["title"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"{args.out}: raw {len(raw) / 1e6:.2f} MB, packed {len(packed) / 1e6:.2f} MB, page {len(page) / 1e6:.2f} MB; "
          f"games {', '.join(f'{g} ({len(d['turns'])} turns)' for g, d in data['games'].items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
