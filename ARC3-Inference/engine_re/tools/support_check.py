"""Offline check of the support measure on an archived play run (PLAY_DESIGN.md 3.11).

    uv run --no-sync python -m engine_re.tools.support_check runs/engine-play/<run> [--games sp80,ls20,ft09] [--json OUT]

For every batch a play run sent with its replica in step, the engine that predicted it (the committed engine at
that point of the transcript: the opening's engine, then each accepted commit, `commit` records with `next` null,
taken from engine_versions/ by its sha256) is replayed with the tracer over the trace up to the batch's last sent
move (tester.run_candidate, as tester.predict runs it). The steps before the batch give the support map
(support.fold); each sent move's path is read against it (support.path_support): its weakest-link support, whether
it ran untested code, whether it evaluated an and/or the earlier steps never separated. Whether the move mismatched
is the batch log's; the replay's own verdict (tester.check_step) is counted too, as a check.

Prints one table per game: the mismatch rate of the moves by weakest support (0, 1-2, 3+, and moves that ran no
step code, e.g. a RESET of a level already built) and with or without an unseparated condition. Nothing under the
run directory is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from engine_re import support as sup
from engine_re.game_api import sync_points
from engine_re.tester import check_step, run_candidate, trace_meta
from engine_re.trace import Trace

BUCKETS = ("0", "1-2", "3+", "no step code")


def _bucket(weakest: int | None) -> str:
    if weakest is None:
        return "no step code"
    return "0" if weakest == 0 else "1-2" if weakest < sup.THIN_SUPPORT else "3+"


def _versions_by_sha(game_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted((game_dir / "engine_versions").glob("v*.py")):
        out.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
    return out


def batches_with_engines(game_dir: Path) -> list[tuple[dict[str, Any], Path | None]]:
    """Each batch record of the transcript with the engine that predicted it (None: not found)."""
    by_sha = _versions_by_sha(game_dir)
    version: int | None = None  # engine.py's latest version
    committed: Path | None = None
    started = False
    out = []
    for line in (game_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        change = record.get("engine_change")
        if isinstance(change, dict) and change.get("version") is not None:
            version = int(change["version"])
        if "plan" in record and not started:  # the first PLAN message: the opening's engine is the committed one
            started = True
            if committed is None and version is not None:
                committed = game_dir / "engine_versions" / f"v{version:04d}.py"
        commit = record.get("commit")
        if isinstance(commit, dict) and commit.get("next") is None:
            committed = by_sha.get(commit.get("engine_sha", "")) or (
                game_dir / "engine_versions" / f"v{int(commit['version']):04d}.py" if commit.get("version") else committed)
        batch = record.get("batch")
        if isinstance(batch, dict) and not batch.get("blind"):
            out.append((batch, committed))
    return out


def check_game(game_dir: Path, scratch: Path | None = None) -> dict[str, Any]:
    trace = Trace.load(game_dir / "trace")
    meta = trace_meta(trace)
    ignore, resync = sync_points(trace.meta)
    moves: list[dict[str, Any]] = []
    missing = 0
    for batch, engine in batches_with_engines(game_dir):
        first, sent = int(batch["first_step"]), int(batch["sent"])
        if not sent:
            continue
        if engine is None or not engine.exists():
            missing += sent
            continue
        upto = first + sent
        actions = [s.action.to_json() for s in trace.steps[:upto]]
        result, frames = run_candidate(engine, actions, meta=meta, contract=False, scratch_root=scratch,
                                       ignore={i for i in ignore if i < upto}, resync={k: v for k, v in resync.items() if k < upto})
        source = engine.read_bytes()
        smap = sup.fold_result(result, {str(i): i for i in range(first) if i not in ignore},
                               source.decode("utf-8", "replace"), hashlib.sha256(source).hexdigest())
        if smap is None:
            missing += sent
            continue
        executed, evaluated = result.get("executed") or {}, result.get("evaluated") or {}
        steps = result.get("steps") or []
        for j in range(sent):
            pos = first + j
            ps = sup.path_support(smap, executed.get(str(pos), []), evaluated.get(str(pos))) if str(pos) in executed else None
            got = steps[pos] if pos < len(steps) else None
            replay_ok = got is not None and check_step(trace[pos], got, frames[pos] if pos < len(frames) else None).ok
            moves.append({
                "step": pos, "turn": batch["turn"], "engine": engine.name, "mismatch": batch.get("mismatch") == pos,
                "replay_mismatch": not replay_ok, "weakest": ps["weakest"] if ps else None,
                "untested": ps["untested"] if ps else [], "unseparated": [c["text"] for c in ps["unseparated"]] if ps else [],
            })
    return {"game": game_dir.name, "moves": moves, "missing": missing}


def table(check: dict[str, Any]) -> list[str]:
    moves = check["moves"]

    def row(label: str, chosen: list[dict[str, Any]]) -> str:
        n = len(chosen)
        bad = sum(m["mismatch"] for m in chosen)
        rate = f"{100 * bad / n:.0f}%" if n else "-"
        return f"| {check['game']} | {label} | {n} | {bad} | {rate} |"

    lines = ["| game | moves | n | mismatches | rate |", "|---|---|---|---|---|"]
    for b in BUCKETS:
        lines.append(row(f"weakest support {b}", [m for m in moves if _bucket(m["weakest"]) == b]))
    lines.append(row("unseparated and/or on the path", [m for m in moves if m["unseparated"]]))
    lines.append(row("no unseparated and/or", [m for m in moves if not m["unseparated"]]))
    lines.append(row("all", moves))
    agree = sum(m["mismatch"] == m["replay_mismatch"] for m in moves)
    lines.append(f"(replay agrees with the batch log on {agree}/{len(moves)} moves; {check['missing']} moves without an engine)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--games", default=None, help="comma-separated game directories (default: every one with a trace)")
    parser.add_argument("--json", type=Path, default=None, help="write the per-move data here")
    parser.add_argument("--scratch", type=Path, default=None, help="where the replays run (default: the system's temp dir)")
    args = parser.parse_args()
    games = args.games.split(",") if args.games else sorted(p.name for p in args.run.iterdir() if (p / "trace").is_dir())
    checks = []
    for game in games:
        check = check_game(args.run / game, args.scratch)
        checks.append(check)
        print("\n".join(table(check)) + "\n")
    if args.json:
        args.json.write_text(json.dumps(checks, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
