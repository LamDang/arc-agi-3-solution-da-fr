"""Run the reverse-engineering agent on several games of a harness run.

    uv run --no-sync python -m engine_re.run_experiment \\
        --run-dir runs/<harness run> --games ls20,ft09,vc33,sp80,lp85 --out runs/engine-re/<name>

For each game: rebuild the trace from the run's event log by replaying the
logged actions through the real engine (checking every final frame against the
logged board), then let one agent per game work in parallel. Writes
``<out>/summary.json`` and ``<out>/summary.md``.

Running the same command again skips finished games and continues interrupted
ones (see engine_re.agent).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine_re.agent import Budget, EngineAgent, ModelConfig
from engine_re.trace import trace_from_run


def prepare(run_dir: Path, game: str, out: Path, environments_dir: Path, max_steps: int | None) -> Path:
    game_dir = out / game
    if (game_dir / "trace" / "trace.json").exists():
        return game_dir
    trace, mismatches = trace_from_run(run_dir, game, environments_dir)
    if mismatches:
        raise RuntimeError(f"{game}: replay differs from the logged boards at steps {mismatches[:10]}")
    if max_steps is not None:
        trace.steps = trace.steps[:max_steps]
        trace.meta["truncated_to"] = max_steps
    trace.save(game_dir / "trace")
    print(f"[{game}] trace: {len(trace)} steps, levels {trace.level_starts()}, verified against the run log", flush=True)
    return game_dir


def summarize(out: Path, games: list[str]) -> None:
    rows = []
    for game in games:
        path = out / game / "result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| game | steps | status | final exact | final frames | first mismatch | best exact | turns | prompt tok | cached tok | output tok | reasoning tok | cost $ | min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        final, best, u = r.get("final") or {}, r.get("best") or {}, r["usage"]
        lines.append(
            f"| {r['game']} | {r['trace_steps']} | {r['status']} | {final.get('exact')}/{final.get('total')} | "
            f"{final.get('final_frame')}/{final.get('total')} | {final.get('first_fail')} | {best.get('exact')} | {r['turns']} | "
            f"{u['prompt_tokens']:,} | {u['cached_tokens']:,} | {u['completion_tokens']:,} | {u['reasoning_tokens']:,} | "
            f"{u['cost_usd']:.3f} | {r['minutes']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--games", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environments-dir", type=Path, default=Path("environment_files"))
    parser.add_argument("--model", default=ModelConfig.model)
    parser.add_argument("--max-turns", type=int, default=Budget.max_turns)
    parser.add_argument("--max-output-tokens", type=int, default=Budget.max_output_tokens)
    parser.add_argument("--max-cost", type=float, default=Budget.max_cost_usd)
    parser.add_argument("--max-minutes", type=float, default=Budget.max_minutes)
    parser.add_argument(
        "--python-quota", type=int, default=None, help="Pause the python tool after this many calls without an engine.py change."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Use only the first N steps of each trace.")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Text-only feedback: test reports and show() print hex digits instead of sending pictures.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    game_dirs = {g: prepare(args.run_dir, g, args.out, args.environments_dir, args.max_steps) for g in games}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        return 0

    model = ModelConfig(model=args.model)
    budget = Budget(args.max_turns, args.max_output_tokens, args.max_cost, args.max_minutes, args.python_quota)

    def work(game: str) -> None:
        previous = game_dirs[game] / "result.json"
        if previous.exists() and json.loads(previous.read_text(encoding="utf-8")).get("status") != "running":
            print(f"[{game}] already finished; skipping", flush=True)
            return
        agent = EngineAgent(
            game, game_dirs[game], model, budget, images=not args.no_images
        )
        result = agent.run()
        final = result.final or {}
        print(
            f"[{game}] {result.status}: final {final.get('exact')}/{final.get('total')} exact, turns {result.turns}, "
            f"output tokens {result.usage.completion_tokens:,}, cost ${result.usage.cost_usd:.3f}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=len(games)) as pool:
        for future in [pool.submit(work, g) for g in games]:
            future.result()
    summarize(args.out, games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
