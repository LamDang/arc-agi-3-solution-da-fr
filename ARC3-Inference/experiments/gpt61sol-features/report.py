"""Status page for the gpt-6.1-sol feature run (exp/gpt61sol-features-25games.md).

    uv run --no-sync python experiments/gpt61sol-features/report.py <out.html> \
        [--transcripts-url URL]

Reads each attempt's benchmark.json, the live event logs (artifacts/<game>_p0_events.jsonl)
and the request logs (<game>_p0_requests.jsonl[.xz]), and the two baseline runs
(runs/base-gpt61sol-dfranzen, runs/base-gpt61sol-20games, packed is fine). Logs are read
incrementally: runs/<run>.report_cache.json keeps how far each file was read and the totals
so far, so a check-in during the run only parses the new lines. The check-in log comes from
checkins.json next to this script: [{"time": "...", "text": "..."}].
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from inference.utils.run_artifacts import open_log  # noqa: E402

ATTEMPTS = [
    ("Run 1", ROOT / "runs/gpt61sol-features-25games"),
    ("Rerun 1", ROOT / "runs/gpt61sol-features-25games-retry1"),
    ("Rerun 2", ROOT / "runs/gpt61sol-features-25games-retry2"),
]
BASELINES = [ROOT / "runs/base-gpt61sol-dfranzen", ROOT / "runs/base-gpt61sol-20games"]
GAMES = (
    "sk48 lf52 bp35 wa30 dc22 re86 s5i5 su15 g50t tr87 ls20 sb26 ka59 tu93 sc25 "
    "cd82 m0r0 vc33 sp80 lp85 ar25 cn04 r11l tn36 ft09"
).split()
NOTE_MARK = "Context notice: this conversation has reached"
HINT_MARK = "Repetition check:"
CACHE_VERSION = 2


# ---------------------------------------------------------------- reading


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _new_lines(path: Path, state: dict):
    """Complete lines of `path` after those already read; updates state in place.

    A live .jsonl is read from the byte offset; once the solver has replaced it with
    .jsonl.xz, the lines already counted are skipped by number.
    """
    if path.suffix == ".jsonl" and path.exists():
        with path.open("rb") as handle:
            handle.seek(state.get("offset", 0))
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                state["offset"] = state.get("offset", 0) + len(raw)
                state["lines"] = state.get("lines", 0) + 1
                yield raw.decode("utf-8", "replace")
        return
    skip = state.get("lines", 0)
    for index, line in enumerate(open_log(path)):
        if index < skip:
            continue
        state["lines"] = index + 1
        yield line
    state["done"] = True


def _log_path(run: Path, gid: str, kind: str) -> Path | None:
    for suffix in (f"_p0_{kind}.jsonl", f"_p0_{kind}.jsonl.xz"):
        base = run if kind == "requests" else run / "artifacts"
        path = base / f"{gid}{suffix}"
        if path.exists():
            return path
    return None


def _request_stats(path: Path, s: dict) -> None:
    s.setdefault("requests", 0)
    for key in ("prompt", "cached", "completion", "reasoning", "cost", "max_prompt", "notes",
                "notes_ok", "hints", "calls", "calls_fields", "reasoning_chars"):
        s.setdefault(key, 0)
    s.setdefault("recent_reasoning", [])
    s.setdefault("note", None)
    s.setdefault("hint_steps", [])
    s.setdefault("note_steps", [])
    for line in _new_lines(path, s.setdefault("read", {})):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "request":
            s["requests"] += 1
            messages = record.get("messages") or []
            last = _text(messages[-1].get("content")) if messages else ""
            s["pending_note"] = NOTE_MARK in last
            if s["pending_note"]:
                s["notes"] += 1
                s["note_steps"].append(record.get("analysis_step"))
            if HINT_MARK in last:
                s["hints"] += 1
                s["hint_steps"].append(record.get("analysis_step"))
            continue
        usage = record.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        s["prompt"] += prompt
        s["max_prompt"] = max(s["max_prompt"], prompt)
        s["cached"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        s["completion"] += int(usage.get("completion_tokens") or 0)
        s["reasoning"] += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        s["cost"] += float(usage.get("cost") or 0.0)
        reply = record.get("reply") or {}
        content = _text(reply.get("content"))
        if s.get("pending_note") and not reply.get("tool_calls") and len(content) >= 300:
            # the harness also keeps a note written as plain text
            s["notes_ok"] += 1
            s["note"] = {"step": record.get("analysis_step"), "text": content[:6000]}
        for call in reply.get("tool_calls") or []:
            try:
                args = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            code = str(args.get("code") or "")
            if s.get("pending_note"):
                lines = [ln for ln in code.splitlines() if ln.strip()]
                if lines and all(ln.lstrip().startswith("#") for ln in lines):
                    s["notes_ok"] += 1
                    s["note"] = {"step": record.get("analysis_step"), "text": code[:6000]}
                continue
            s["calls"] += 1
            reasoning = str(args.get("reasoning") or "")
            if args.get("description") and reasoning:
                s["calls_fields"] += 1
                s["reasoning_chars"] += len(reasoning)
                s["recent_reasoning"] = (s["recent_reasoning"] + [{
                    "step": record.get("analysis_step"),
                    "description": str(args.get("description"))[:400],
                    "reasoning": reasoning[:1500],
                }])[-3:]
        s["pending_note"] = False


def _event_stats(path: Path, s: dict) -> None:
    for key in ("actions", "levels", "game_overs", "level"):
        s.setdefault(key, 0)
    s.setdefault("per_level", [0])
    s.setdefault("overs_per_level", [0])
    for line in _new_lines(path, s.setdefault("read", {})):
        if '"type":"action"' not in line and '"type": "action"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "action":
            continue
        s["actions"] += 1
        s["per_level"][-1] += 1
        s["level"] = event.get("level")
        s["status"] = event.get("run_status")
        if str(event.get("game_over")) == "True":
            s["game_overs"] += 1
            s["overs_per_level"][-1] += 1
        if str(event.get("level_completed")) == "True":
            s["levels"] += 1
            s["per_level"].append(0)
            s["overs_per_level"].append(0)


def _benchmark(run: Path) -> dict[str, dict]:
    path = run / "benchmark.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {g["game_id"].split("-")[0]: g for g in data.get("game_runs") or []}


def _scores(run: Path) -> dict[str, float]:
    path = run / "score.json"
    if not path.exists():
        return {}
    return {gid.split("-")[0]: e["score"] for gid, e in json.loads(path.read_text())["games"].items()}


def _game_ids(run: Path) -> dict[str, str]:
    ids = {}
    for path in list(run.glob("*_p0_requests.jsonl*")) + list((run / "artifacts").glob("*_p0_events.jsonl*")):
        gid = path.name.split("_p0_")[0]
        ids[gid.split("-")[0]] = gid
    return ids


def collect(run: Path) -> dict:
    """Per-game stats of one run, read incrementally through its cache file."""
    if not run.exists():
        return {}
    cache_path = run.parent / f"{run.name}.report_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "games": {}}
    bench = _benchmark(run)
    ids = _game_ids(run)
    for short, entry in bench.items():
        ids.setdefault(short, entry["game_id"])
    games = {}
    for short, gid in ids.items():
        g = cache["games"].setdefault(short, {"req": {}, "ev": {}})
        req = _log_path(run, gid, "requests")
        if req is not None and not g["req"].get("read", {}).get("done"):
            _request_stats(req, g["req"])
        ev = _log_path(run, gid, "events")
        b = bench.get(short)
        if ev is not None and ev.suffix == ".jsonl" and not (b and b.get("state") != "playing"):
            _event_stats(ev, g["ev"])
        games[short] = {"gid": gid, "bench": b, "req": g["req"], "ev": g["ev"]}
    cache_path.write_text(json.dumps(cache))
    scores = _scores(run)
    for short, g in games.items():
        g["score"] = scores.get(short)
    return {"games": games, "scored": bool(scores), "run": run}


# ---------------------------------------------------------------- shaping


def _row(g: dict) -> dict:
    b = g.get("bench") or {}
    ev = g.get("ev") or {}
    req = g.get("req") or {}
    state = b.get("state") or ev.get("status") or "playing"
    finished = state not in ("playing", None)
    total_levels = b.get("number_of_levels")
    if finished:
        levels = b.get("levels_completed", 0)
        per_level = list(b.get("actions_per_level") or [])
        actions = len(b.get("history") or [])
        minutes = (b.get("final_wallclock_seconds") or 0) / 60
        score = g.get("score") if g.get("score") is not None else b.get("final_score")
    else:
        levels = ev.get("levels", b.get("levels_completed", 0))
        per_level = list(ev.get("per_level") or [])
        actions = ev.get("actions", 0)
        minutes = None
        if b.get("started_at"):
            started = datetime.fromisoformat(b["started_at"])
            minutes = (datetime.now() - started).total_seconds() / 60
        score = None
    note = (b.get("solver_note") or "") if finished else ""
    if state == "gave_up" and note.startswith("tokens="):
        state = "limit"
    return {
        "state": state, "finished": finished, "levels": levels, "total": total_levels,
        "per_level": per_level, "human": list(b.get("base_actions_per_level") or []),
        "actions": actions, "minutes": minutes, "score": score,
        "out": req.get("completion", 0), "reasoning": req.get("reasoning", 0),
        "prompt": req.get("prompt", 0), "cached": req.get("cached", 0),
        "cost": req.get("cost", 0.0), "max_prompt": req.get("max_prompt", 0),
        "notes": req.get("notes", 0), "notes_ok": req.get("notes_ok", 0),
        "hints": req.get("hints", 0), "game_overs": ev.get("game_overs"),
        "calls": req.get("calls", 0), "calls_fields": req.get("calls_fields", 0),
        "reasoning_chars": req.get("reasoning_chars", 0),
        "recent_reasoning": req.get("recent_reasoning") or [], "note": req.get("note"),
        "hint_steps": req.get("hint_steps") or [], "note_steps": req.get("note_steps") or [],
        "solver_note": note,
    }


# ---------------------------------------------------------------- page

E = html.escape

CSS = """
/* Layout: one column; a summary strip, the game table (scrolls sideways on its own), then notes and log. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --fg: #18202b; --muted: #5b6676; --line: #dde2ea;
  --accent: #2c5e8f; --accent-soft: #e4edf6;
  --good: #23704a; --good-soft: #e1f1e8; --warn: #9a6512; --warn-soft: #f7ecd9;
  --bad: #a83434; --bad-soft: #f6e1e1;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --cond: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161c; --surface: #1a2029; --fg: #e3e8ef; --muted: #98a3b3; --line: #2b3441;
  --accent: #7fb0e0; --accent-soft: #1f3247;
  --good: #6cc597; --good-soft: #193526; --warn: #e0b061; --warn-soft: #3a2d15;
  --bad: #ec8a8a; --bad-soft: #3e1d1d; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12161c; --surface: #1a2029; --fg: #e3e8ef; --muted: #98a3b3; --line: #2b3441;
  --accent: #7fb0e0; --accent-soft: #1f3247;
  --good: #6cc597; --good-soft: #193526; --warn: #e0b061; --warn-soft: #3a2d15;
  --bad: #ec8a8a; --bad-soft: #3e1d1d; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 var(--sans); }
.wrap { max-width: 1240px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px;
  display: grid; gap: 28px; }
h1 { font: 600 28px/1.15 var(--cond); margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font: 600 19px/1.2 var(--cond); margin: 0 0 10px; text-wrap: balance; }
.lede { color: var(--muted); max-width: 72ch; margin: 6px 0 0; }
.stamp { font: 12px/1.4 var(--mono); color: var(--muted); margin-top: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; }
.tile .k { font: 600 11px/1.3 var(--sans); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.tile .v { font: 600 24px/1.2 var(--cond); font-variant-numeric: tabular-nums; margin-top: 4px; }
.tile .s { font-size: 12.5px; color: var(--muted); }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; vertical-align: top; }
th { font: 600 11px/1.3 var(--sans); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
  background: var(--surface); position: sticky; top: 0; }
th.grp { text-align: center; border-bottom: 2px solid var(--line); }
td.l, th.l { text-align: left; }
td.game { font: 600 13px var(--mono); }
tr:last-child td { border-bottom: 0; }
tr.total td { font-weight: 600; border-top: 2px solid var(--line); }
.sep { border-left: 2px solid var(--line); }
.pill { display: inline-block; font: 600 11px/1 var(--sans); padding: 4px 7px; border-radius: 999px;
  letter-spacing: 0.03em; text-transform: uppercase; }
.won { background: var(--good-soft); color: var(--good); }
.playing { background: var(--accent-soft); color: var(--accent); }
.limit, .gave_up { background: var(--warn-soft); color: var(--warn); }
.crashed, .cancelled { background: var(--bad-soft); color: var(--bad); }
.queued { background: transparent; color: var(--muted); border: 1px solid var(--line); }
.hi { color: var(--good); font-weight: 600; }
.lo { color: var(--bad); font-weight: 600; }
.dim { color: var(--muted); }
.pl { font: 12px var(--mono); color: var(--muted); }
.pl b { color: var(--bad); font-weight: 600; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
.cols > * { min-width: 0; }
.box { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 0; font-size: 13.5px; }
dl.kv dt { font-family: var(--mono); color: var(--muted); font-size: 12.5px; }
dl.kv dd { margin: 0; }
.log { display: grid; gap: 12px; }
.log article { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; }
.log time { font: 600 12px var(--mono); color: var(--accent); }
.log p { margin: 6px 0 0; max-width: 90ch; white-space: pre-wrap; }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
details + details { margin-top: 8px; }
summary { cursor: pointer; padding: 9px 14px; font: 600 13px var(--mono); }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.det { padding: 0 14px 12px; display: grid; gap: 10px; }
.det h3 { font: 600 12px var(--sans); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 6px 0 0; }
pre { font: 12px/1.45 var(--mono); background: var(--bg); border: 1px solid var(--line); border-radius: 4px;
  padding: 10px; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 360px; overflow: auto; }
a { color: var(--accent); }
.note { font-size: 13px; color: var(--muted); max-width: 90ch; }
"""


def _pill(state: str) -> str:
    label = {"limit": "limit", "gave_up": "gave up", "queued": "queued"}.get(state, state)
    return f'<span class="pill {E(state)}">{E(label)}</span>'


def _num(value, fmt="{:,}") -> str:
    return "" if value is None else fmt.format(value)


def _k(n) -> str:
    if not n:
        return "0"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


def _per_level(per_level: list[int], human: list[int]) -> str:
    parts = []
    for i, n in enumerate(per_level):
        over = i < len(human) and n > human[i]
        parts.append(f"<b>{n}</b>" if over else str(n))
    return " ".join(parts)


def build(out: Path, transcripts_url: str | None) -> dict:
    now = datetime.now(timezone.utc)
    attempts = [(label, collect(run)) for label, run in ATTEMPTS]
    base_games = {}
    for run in BASELINES:
        data = collect(run)
        for short, g in (data.get("games") or {}).items():
            base_games[short] = _row(g)
    rows = {}
    for label, data in attempts:
        for short, g in (data.get("games") or {}).items():
            rows.setdefault(short, {})[label] = _row(g)

    main = attempts[0][1]
    main_rows = {s: r.get("Run 1") for s, r in rows.items()}
    best = {}
    for short in GAMES:
        scores = [r["score"] for r in rows.get(short, {}).values() if r and r["score"] is not None]
        best[short] = max(scores) if scores else None
    finished = [s for s in GAMES if main_rows.get(s) and main_rows[s]["finished"]]
    playing = [s for s in GAMES if main_rows.get(s) and not main_rows[s]["finished"]]
    at100 = sum(1 for s in GAMES if best[s] is not None and best[s] >= 100)
    total_cost = sum(r["cost"] for per in rows.values() for r in per.values() if r)
    total_out = sum(r["out"] for per in rows.values() for r in per.values() if r)
    base_cost = sum(r["cost"] for r in base_games.values())
    base_cost_done = sum(base_games[s]["cost"] for s in finished if s in base_games)
    run1_cost_done = sum(main_rows[s]["cost"] for s in finished)
    notes = sum(r["notes"] for per in rows.values() for r in per.values() if r)
    notes_ok = sum(r["notes_ok"] for per in rows.values() for r in per.values() if r)
    hints = sum(r["hints"] for per in rows.values() for r in per.values() if r)
    calls = sum(r["calls"] for per in rows.values() for r in per.values() if r)
    calls_fields = sum(r["calls_fields"] for per in rows.values() for r in per.values() if r)
    chars = sum(r["reasoning_chars"] for per in rows.values() for r in per.values() if r)
    mean_best = statistics.mean(best[s] for s in finished) if finished else None
    status = "finished" if (main.get("scored") and all(
        not d or d.get("scored") for _, d in attempts if d)) else "running"
    started = None
    for s in GAMES:
        b = ((main.get("games") or {}).get(s) or {}).get("bench") or {}
        if b.get("started_at"):
            t = datetime.fromisoformat(b["started_at"])
            started = t if started is None or t < started else started
    elapsed = (datetime.now() - started).total_seconds() / 3600 if started else 0

    retry_labels = [label for label, d in attempts[1:] if d]
    head = []
    head.append('<tr><th class="l" rowspan="2">game</th><th rowspan="2">best</th>'
                '<th class="grp sep" colspan="11">Run 1 (all four settings, 300K cap)</th>'
                '<th class="grp sep" colspan="5">baseline (no new settings)</th>'
                + "".join(f'<th class="grp sep" colspan="3">{E(l)}</th>' for l in retry_labels)
                + "</tr>")
    head.append('<tr><th class="l sep">state</th><th>levels</th><th>score</th><th>actions</th>'
                '<th class="l">per level (over human in red)</th><th>out tok</th><th>cost</th>'
                '<th>min</th><th>notes</th><th>hints</th><th>game overs</th>'
                '<th class="sep">score</th><th>levels</th><th>actions</th><th>cost</th><th>min</th>'
                + "".join('<th class="sep">state</th><th>levels</th><th>score</th>' for _ in retry_labels)
                + "</tr>")
    body = []
    for short in GAMES:
        r = main_rows.get(short)
        bl = base_games.get(short)
        cells = [f'<td class="l game">{E(short)}</td>']
        bs = best.get(short)
        cls = "hi" if bs is not None and bs >= 100 else ("lo" if bs is not None else "dim")
        cells.append(f'<td class="{cls}">{_num(bs, "{:.1f}")}</td>')
        if r:
            lv = f'{r["levels"]}/{r["total"]}' if r["total"] else str(r["levels"])
            cells += [
                f'<td class="l sep">{_pill(r["state"])}</td>', f"<td>{lv}</td>",
                f'<td>{_num(r["score"], "{:.1f}")}</td>', f'<td>{r["actions"]:,}</td>',
                f'<td class="l pl">{_per_level(r["per_level"], r["human"] or (bl or {}).get("human", []))}</td>',
                f'<td>{_k(r["out"])}</td>', f'<td>${r["cost"]:.2f}</td>',
                f'<td>{_num(r["minutes"], "{:.0f}")}</td>',
                f'<td>{r["notes_ok"]}{"" if r["notes_ok"] == r["notes"] else "/" + str(r["notes"])}</td>',
                f'<td>{r["hints"]}</td>', f'<td>{_num(r["game_overs"])}</td>',
            ]
        else:
            cells += [f'<td class="l sep">{_pill("queued")}</td>'] + ["<td></td>"] * 10
        if bl:
            cells += [
                f'<td class="sep">{_num(bl["score"], "{:.1f}")}</td>',
                f'<td>{bl["levels"]}/{bl["total"]}</td>', f'<td>{bl["actions"]:,}</td>',
                f'<td>${bl["cost"]:.2f}</td>', f'<td>{_num(bl["minutes"], "{:.0f}")}</td>',
            ]
        else:
            cells += ['<td class="sep"></td>'] + ["<td></td>"] * 4
        for label in retry_labels:
            rr = rows.get(short, {}).get(label)
            if rr:
                lv = f'{rr["levels"]}/{rr["total"]}' if rr["total"] else str(rr["levels"])
                cells += [f'<td class="sep">{_pill(rr["state"])}</td>', f"<td>{lv}</td>",
                          f'<td>{_num(rr["score"], "{:.1f}")}</td>']
            else:
                cells += ['<td class="sep dim">-</td>', "<td></td>", "<td></td>"]
        body.append("<tr>" + "".join(cells) + "</tr>")
    # totals over the games Run 1 finished, against the same games in the baseline
    if finished:
        fr = [main_rows[s] for s in finished]
        fb = [base_games[s] for s in finished if s in base_games]
        body.append(
            '<tr class="total">'
            f'<td class="l">finished</td><td>{len(finished)}</td><td class="sep l"></td>'
            f'<td>{sum(r["levels"] for r in fr)}/{sum(r["total"] or 0 for r in fr)}</td>'
            f'<td>{statistics.mean(r["score"] or 0 for r in fr):.1f}</td>'
            f'<td>{sum(r["actions"] for r in fr):,}</td><td></td>'
            f'<td>{_k(sum(r["out"] for r in fr))}</td><td>${sum(r["cost"] for r in fr):.2f}</td>'
            '<td></td>'
            f'<td>{sum(r["notes_ok"] for r in fr)}</td><td>{sum(r["hints"] for r in fr)}</td>'
            f'<td>{sum(r["game_overs"] or 0 for r in fr)}</td>'
            f'<td class="sep">{statistics.mean(b["score"] or 0 for b in fb):.1f}</td>'
            f'<td>{sum(b["levels"] for b in fb)}/{sum(b["total"] or 0 for b in fb)}</td>'
            f'<td>{sum(b["actions"] for b in fb):,}</td><td>${sum(b["cost"] for b in fb):.2f}</td><td></td>'
            + "".join('<td class="sep"></td><td></td><td></td>' for _ in retry_labels)
            + "</tr>")

    tiles = [
        ("Games at 100", f"{at100}/25", f"{len(finished)} finished, {len(playing)} playing in Run 1"),
        ("Mean score", _num(mean_best, "{:.1f}") or "-",
         "best attempt, finished games; baseline 99.1 over 25"),
        ("Cost so far", f"${total_cost:.2f}",
         f"finished games: ${run1_cost_done:.2f} vs ${base_cost_done:.2f} baseline; baseline total ${base_cost:.2f}"),
        ("Output tokens", _k(total_out), "all attempts; cap 300K per game"),
        ("Handover notes", f"{notes_ok}", f"{notes} requested; comments-only replies kept"),
        ("Step-back hints", f"{hints}", "3 positions seen 3 times on a level"),
        ("Reasoning fields", f"{calls_fields}/{calls}",
         f"calls with both fields; mean {chars / calls_fields:.0f} characters" if calls_fields else "calls with both fields"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{E(k)}</div><div class="v">{E(v)}</div><div class="s">{E(s)}</div></div>'
        for k, v, s in tiles)

    settings = [
        ("model", "gpt-6.1-sol, OpenAI Responses API, effort xhigh, encrypted reasoning sent back"),
        ("base settings", "params.yaml (dfranzen's), as base-gpt61sol-20games"),
        ("limits", "300K output tokens and 240 minutes per game; 10 games at a time"),
        ("ARC3_PYTHON_RATIONALE=1", "python(description, reasoning, code), strict schema"),
        ("OPENAI_REASONING_SUMMARY", "detailed (baseline: auto)"),
        ("ARC3_NOTE_COMPACTION_TOKENS", "120000, 10 turns kept after the note"),
        ("ARC3_NO_BUDGET_BURN=1", "system-prompt line against losing on purpose to reset a level"),
        ("ARC3_REPEAT_HINT=1", "step-back message at 3 positions x 3 visits, 10 turns apart"),
        ("reruns", "each game under 100 is played again, up to twice; every attempt is kept"),
    ]
    settings_html = "".join(f"<dt>{E(k)}</dt><dd>{E(v)}</dd>" for k, v in settings)

    checkins_path = HERE / "checkins.json"
    checkins = json.loads(checkins_path.read_text()) if checkins_path.exists() else []
    log_html = "".join(
        f'<article><time>{E(c["time"])}</time><p>{E(c["text"])}</p></article>'
        for c in reversed(checkins)) or '<p class="note">No check-in yet.</p>'

    details = []
    for short in GAMES:
        for label, per in rows.get(short, {}).items():
            if not per:
                continue
            parts = []
            if per["note_steps"] or per["hint_steps"]:
                parts.append(
                    f'<p class="note">Notes requested at steps {E(", ".join(map(str, per["note_steps"])) or "-")}; '
                    f'hints at steps {E(", ".join(map(str, per["hint_steps"])) or "-")}.</p>')
            if per["note"]:
                parts.append(f'<h3>Latest handover note (step {E(str(per["note"]["step"]))})</h3>'
                             f'<pre>{E(per["note"]["text"])}</pre>')
            for item in reversed(per["recent_reasoning"]):
                parts.append(f'<h3>Step {E(str(item["step"]))}: {E(item["description"])}</h3>'
                             f'<pre>{E(item["reasoning"])}</pre>')
            if per["solver_note"]:
                parts.append(f'<p class="note">Solver note: {E(per["solver_note"][:300])}</p>')
            if not parts:
                continue
            details.append(f'<details><summary>{E(short)} · {E(label)} · {E(per["state"])} '
                           f'{per["levels"]}/{per["total"] or "?"}</summary><div class="det">{"".join(parts)}</div></details>')

    link = (f'<a href="{E(transcripts_url)}">Step-by-step transcripts</a> of every game, '
            'with the reasoning fields and notes in place. ' if transcripts_url else "")
    page = f"""<title>gpt-6.1-sol Feature Run</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>gpt-6.1-sol Feature Run</h1>
  <p class="lede">The 25 official ARC-AGI-3 games, played by the base agent with stated reasoning in every
  python call, a handover note before the history cut, a step-back hint when board positions repeat and
  a line against budget burning. Each game under 100 is rerun up to twice. The baseline is the earlier
  gpt-6.1-sol pass with the same settings and none of the four. {link}</p>
  <div class="stamp">{E(status)} · updated {now:%Y-%m-%d %H:%M} UTC · {elapsed:.1f} h since Run 1 started</div>
</header>
<section class="tiles">{tiles_html}</section>
<section>
  <h2>Games</h2>
  <div class="scroll"><table><thead>{"".join(head)}</thead><tbody>{"".join(body)}</tbody></table></div>
  <p class="note">Levels, actions and per-level counts of a game still playing come from its event log;
  its score appears when it ends. "limit" means the game stopped at its token or time limit.
  Notes count comments-only notes kept; "a/b" means b were requested. Cost is computed at
  $2 / $0.10 / $2.50 / $10 per million uncached input, cached input, cache-write and output tokens.</p>
</section>
<section class="cols">
  <div><h2>Check-ins</h2><div class="log">{log_html}</div></div>
  <div><h2>Settings</h2><div class="box"><dl class="kv">{settings_html}</dl></div></div>
</section>
<section>
  <h2>Notes and reasoning, per game</h2>
  <p class="note">The latest handover note and the last three reasoning fields of each game.</p>
  {"".join(details) or '<p class="note">Nothing yet.</p>'}
</section>
</div>
"""
    out.write_text(page, encoding="utf-8")
    return {"finished": finished, "playing": playing, "at100": at100, "cost": total_cost,
            "status": status}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--transcripts-url", default=None)
    args = ap.parse_args()
    t0 = time.time()
    summary = build(args.out, args.transcripts_url)
    print(json.dumps(summary), f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
