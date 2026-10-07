"""Build a transcript page for a base-agent run (inference/agent/tool_agent.py), step by step, per game.

    uv run --no-sync python scripts/base_transcripts/build.py runs/<run>[=<label>] [runs/<run>[=<label>] ...] \
        <out.html> [--title "..."] [--lede "..."] [--compare-json <file>]

Reads the run's benchmark.json, the request logs (<game>_p0_requests.jsonl[.xz], needs
ANALYZER_SAVE_REQUEST_LOGS=true) and artifacts/<game>_p0_events.jsonl (unpack a packed run
first: scripts/pack_run.py unpack). Every analysis step becomes a card: the harness prompt
that opened it, each model request in it (reasoning summary, message, the python it ran and
what came back, tokens and cost), the actions it played and the board after them. The header
compares the run with the rows of --compare-json: [{"label", "scores": {game: score}, "cost",
"output_tokens"}]. The page is built from template.html with the data gzipped and
base64-embedded. Several runs (each with an optional tab-group label) go on one page, and a
run that is still playing can be built: a game benchmark.json still lists as playing takes its
levels and actions from its event log, which is written as it plays.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from inference.utils.run_artifacts import open_log  # noqa: E402

TEXT_LIMIT = 6000


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    """Keep both ends of a long text, as the harness's own tool-output cap does."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit:,} characters left out] ...\n{text[-half:]}"


def _text(content) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def _tool_output(content) -> str:
    """A python tool result: its stdout and stderr, then any other fields."""
    raw = _text(content)
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _clip(raw)
    if not isinstance(result, dict):
        return _clip(raw)
    parts = []
    if result.get("stdout"):
        parts.append(str(result["stdout"]).rstrip())
    if result.get("stderr"):
        parts.append("stderr:\n" + str(result["stderr"]).rstrip())
    rest = {k: v for k, v in result.items() if k not in ("tool", "stdout", "stderr", "function_retention")}
    if rest.get("returncode") == 0:
        rest.pop("returncode")
    if rest:
        parts.append(json.dumps(rest, indent=1, ensure_ascii=False))
    return _clip("\n\n".join(parts) or raw)


def _call(call: dict) -> dict:
    function = call.get("function") or {}
    arguments = str(function.get("arguments") or "")
    code = arguments
    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
            code = parsed["code"]
    except json.JSONDecodeError:
        pass
    return {"id": call.get("id"), "name": function.get("name") or "", "code": _clip(code)}


def _board(board) -> str:
    return "".join("".join(f"{v:x}" for v in row) for row in board)


def game_data(run: Path, game: dict) -> dict:
    gid = game["game_id"]
    logs = sorted(run.glob(f"{gid}_p0_requests.jsonl*"))
    records = []
    for path in logs:
        with open_log(path) as lines:
            records.extend(json.loads(line) for line in lines)
    tool_outputs: dict[str, str] = {}
    for record in records:
        for message in record.get("messages") or []:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                tool_outputs[message["tool_call_id"]] = _tool_output(message.get("content"))

    steps: dict[int, dict] = {}

    def step(s: int) -> dict:
        if s not in steps:
            steps[s] = {"s": s, "prompt": "", "reqs": [], "actions": [], "board": None, "level": None}
        return steps[s]

    system = ""
    pending_prompt: dict[int, str] = {}
    for record in records:
        s = int(record.get("analysis_step") or 0)
        if record.get("event") == "request":
            messages = record.get("messages") or []
            if not system and messages and messages[0].get("role") == "system":
                system = _text(messages[0].get("content"))
            users = [m for m in messages if m.get("role") == "user"]
            if users and s not in pending_prompt:
                pending_prompt[s] = _clip(_text(users[-1].get("content")), 4000)
            continue
        reply = record.get("reply") or {}
        usage = record.get("usage") or {}
        card = step(s)
        card["prompt"] = card["prompt"] or pending_prompt.get(s, "")
        calls = [_call(c) for c in reply.get("tool_calls") or []]
        for c in calls:
            c["out"] = tool_outputs.get(c.pop("id"), None)
        card["reqs"].append({
            "i": record.get("request_index_within_turn"),
            "reasoning": _clip(str(reply.get("reasoning") or "")),
            "content": _clip(str(reply.get("content") or "")),
            "calls": calls,
            "finish": record.get("finish_reason") or "",
            "tok": {
                "out": int(usage.get("completion_tokens") or 0),
                "rs": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                "in": int(usage.get("prompt_tokens") or 0),
                "cached": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "cost": float(usage.get("cost") or 0.0),
            },
        })

    events_path = run / "artifacts" / f"{gid}_p0_events.jsonl"
    live_apl: dict[int, int] = defaultdict(int)
    live_levels = 0
    live_won = False
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written
            if event.get("type") != "action":
                continue
            live_apl[int(event.get("level") or 0)] += 1
            live_levels += bool(event.get("level_completed"))
            live_won = live_won or event.get("state") == "WIN"
            card = step(int(event.get("analysis_step") or 0))
            card["actions"].append({
                "n": event.get("action_num"),
                "a": event.get("action_display") or event.get("action_name"),
                "lvl": event.get("level"),
                "done": bool(event.get("level_completed")),
                "over": bool(event.get("game_over")),
            })
            card["level"] = card["level"] or event.get("level")
            if event.get("board"):
                card["board"] = _board(event["board"])

    total = defaultdict(float)
    for card in steps.values():
        for req in card["reqs"]:
            for key, value in req["tok"].items():
                total[key] += value
    state = game.get("state")
    levels = game.get("levels_completed")
    apl = game.get("actions_per_level") or []
    if state == "playing":
        # benchmark.json is saved every 10 minutes; the event log is current
        total_levels = int(game.get("number_of_levels") or 0)
        state = "won" if live_won else ("playing" if live_apl else "not started")
        levels = total_levels if live_won else live_levels
        apl = [live_apl[k] for k in sorted(live_apl)]
    return {
        "id": gid,
        "state": state,
        "score": game.get("final_score"),
        "levels": levels,
        "total_levels": game.get("number_of_levels"),
        "apl": apl,
        "baseline": game.get("base_actions_per_level") or [],
        "minutes": round(float(game.get("final_wallclock_seconds") or 0) / 60, 1),
        "total": dict(total),
        "requests": sum(len(c["reqs"]) for c in steps.values()),
        "steps": [steps[s] for s in sorted(steps)],
        "system": system,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, each optionally =<tab-group label>")
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lede", default="")
    ap.add_argument("--compare-json", type=Path, default=None)
    args = ap.parse_args()
    runs = []
    for item in args.runs:
        path, _, label = item.partition("=")
        runs.append((Path(path), label or Path(path).name))
    first = runs[0][0]
    settings = {}
    if (first / "eval_settings.json").exists():
        settings = json.loads((first / "eval_settings.json").read_text(encoding="utf-8"))
    data = {
        "title": args.title or first.name,
        "lede": args.lede,
        "run": ", ".join(path.name for path, _ in runs),
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "settings": settings,
        "compare": json.loads(args.compare_json.read_text(encoding="utf-8")) if args.compare_json else [],
        "games": {},
    }
    system = ""
    for path, label in runs:
        benchmark = json.loads((path / "benchmark.json").read_text(encoding="utf-8"))
        for game in sorted(benchmark.get("game_runs") or [], key=lambda g: g["game_id"]):
            gdata = game_data(path, game)
            system = system or gdata["system"]
            gdata.pop("system")
            gdata["group"] = label
            data["games"][gdata["id"].split("-")[0]] = gdata
    data["system"] = system
    raw = json.dumps(data, separators=(",", ":")).encode()
    packed = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
    page = (HERE / "template.html").read_text(encoding="utf-8")
    page = page.replace("__DATA__", packed).replace("__TITLE__", data["title"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"{args.out}: raw {len(raw) / 1e6:.2f} MB, page {len(page) / 1e6:.2f} MB; games "
          + ", ".join(f"{g} ({len(d['steps'])} steps, {d['requests']} requests)" for g, d in data["games"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
