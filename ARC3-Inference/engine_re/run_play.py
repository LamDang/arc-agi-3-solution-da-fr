"""Run the play-and-model agent (engine_re.play_agent) on live games, one agent per game in parallel.

    uv run --no-sync python -m engine_re.run_play --games sp80,ls20,ft09 --out runs/engine-play/<name> \\
        --model qwen/qwen3.8-flash --max-turns 300 --max-minutes 240 --max-cost 6 --max-actions 500

Writes <out>/<game>/ (the agent's files: trace/, workspace/engine.py, transcript.jsonl, result.json, ...),
<out>/config.json, <out>/summary.json and summary.md (score, levels, actions, batches, mismatches, fit
rounds, tokens, cost per game), and <out>/benchmark.json in TAAF's shape, so that
`make score_run SCORE_RUN_DIR=<out>` scores the run with the base harness's code and the viewer can
open <out>/<game>/artifacts/. Running the same command again skips finished games and resumes
interrupted ones (the real game is replayed from the saved trace).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine_re.agent import Budget, ModelConfig
from engine_re.live_game import benchmark_json
from engine_re.play_agent import PLAN_TURNS, PlayAgent


def summarize(out: Path, games: list[str]) -> None:
    rows = []
    for game in games:
        path = out / game / "result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| game | status | score | levels | actions | per level | batches | moves | mismatches | fit rounds | unexplained | turns (plan/fit) | final exact | prompt tok | cached | output tok | reasoning tok | cost $ | min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        u, final, pt = r["usage"], r.get("final") or {}, r.get("phase_turns") or {}
        score = r.get("score")
        lines.append(
            f"| {r['game']} | {r['status']} | {'-' if score is None else f'{score:.1f}'} | {r.get('levels_completed')}/{r.get('win_levels')} | "
            f"{r.get('actions')} | {' '.join(str(c) for c in r.get('actions_per_level') or [])} | {r.get('batches')} | {r.get('moves_sent')} | "
            f"{r.get('mismatches')} | {len(r.get('fit_rounds') or [])} | {len(r.get('unexplained') or [])} | {r['turns']} ({pt.get('plan', 0)}/{pt.get('fit', 0)}) | "
            f"{final.get('exact')}/{final.get('total')} | {u['prompt_tokens']:,} | "
            f"{(100 * u['cached_tokens'] // max(1, u['prompt_tokens']))}% | {u['completion_tokens']:,} | {u['reasoning_tokens']:,} | "
            f"{u['cost_usd']:.3f} | {r['minutes']} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", required=True, help="comma-separated game ids or prefixes (ft09, sp80, ...)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environments-dir", type=Path, default=Path("environment_files"))
    parser.add_argument("--model", default=ModelConfig.model)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=1_500_000)
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--max-minutes", type=float, default=240.0)
    parser.add_argument("--max-actions", type=int, default=500, help="actions per game, the opening RESET excluded")
    parser.add_argument("--batch-size", type=int, default=10, help="moves per commit_moves call")
    parser.add_argument("--no-auto-reset", action="store_true", help="after a game over, leave the RESET to the model")
    parser.add_argument("--fit-turns", type=int, default=None,
                        help="the escape hatch (PLAY_DESIGN.md 3.6): after this many turns in one fit round without an accepted "
                             "commit, the model may play on with its engine out of step (default: off)")
    parser.add_argument("--plan-turns", type=int, default=PLAN_TURNS,
                        help="turns of a plan round without commit_moves before the harness reminds the model to send a batch "
                             "(and again every as many turns; 0: never)")
    parser.add_argument("--cut-untested", action="store_true",
                        help="cut each batch after its first move whose predicted path runs engine code no recorded step has "
                             "run (PLAY_DESIGN.md 3.11; default: off, the model sees each move's support and decides)")
    parser.add_argument("--context", choices=("compact", "rebuilt"), default="compact",
                        help="how the conversation is bounded: 'compact' shortens it in place once a request passes 140K prompt "
                             "tokens (the default); 'rebuilt' keeps it in full and rebuilds every request from it (the system "
                             "prompt, one compacted-context message with the commit turns, the current PLAN or FIT message and the "
                             "older turns since it, then the last 10 turns as they are)")
    parser.add_argument("--no-images", action="store_true", help="text-only feedback (no pictures)")
    parser.add_argument("--providers", default=None, help="comma-separated OpenRouter providers, in order, no fallback")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=ModelConfig.temperature)
    parser.add_argument("--top-p", type=float, default=ModelConfig.top_p)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--label", default=None, help="the benchmark.json label (default: the out directory's name)")
    args = parser.parse_args()
    if args.thinking_budget is not None and args.reasoning_effort:
        parser.error("--thinking-budget and --reasoning-effort cannot be combined")

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    config["harness"] = "engine_re.play_agent (v10)"
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.time()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()] if args.providers else None
    model = ModelConfig(model=args.model, providers=providers, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, reasoning_effort=args.reasoning_effort, thinking_budget=args.thinking_budget,
                        context=args.context)
    budget = Budget(args.max_turns, args.max_output_tokens, args.max_cost, args.max_minutes)
    runs: dict[str, dict] = {}

    def agent_for(game: str) -> PlayAgent:
        return PlayAgent(game, args.out / game, model, budget, args.environments_dir, images=not args.no_images,
                         batch_size=args.batch_size, max_actions=args.max_actions, auto_reset=not args.no_auto_reset,
                         fit_turns=args.fit_turns, plan_turns=args.plan_turns, cut_untested=args.cut_untested)

    def work(game: str) -> None:
        previous = args.out / game / "result.json"
        if previous.exists():
            data = json.loads(previous.read_text(encoding="utf-8"))
            if data.get("status") not in ("running", "error"):
                print(f"[{game}] already finished ({data.get('status')}); skipping", flush=True)
                agent = agent_for(game)  # replays the real game from trace/
                agent._restore()
                runs[game] = agent.game_run()
                return
        agent = agent_for(game)
        result = agent.run()
        runs[game] = agent.game_run()
        print(
            f"[{game}] {result.status}: {result.levels_completed}/{result.win_levels} levels in {result.actions} actions, "
            f"score {result.score}, {result.batches} batches, {result.mismatches} mismatches, turns {result.turns}, "
            f"output tokens {result.usage.completion_tokens:,}, cost ${result.usage.cost_usd:.3f}",
            flush=True,
        )

    failed = []
    with ThreadPoolExecutor(max_workers=len(games)) as pool:
        futures = {g: pool.submit(work, g) for g in games}
        for game, future in futures.items():
            try:
                future.result()
            except Exception:  # noqa: BLE001  (one game failing must not lose the others' records)
                failed.append(game)
                print(f"[{game}] failed outside the agent loop:\n{traceback.format_exc()}", flush=True)
    summarize(args.out, games)
    label = args.label or args.out.name
    (args.out / "benchmark.json").write_text(
        json.dumps(benchmark_json(label, [runs[g] for g in games if g in runs], started), indent=2) + "\n", encoding="utf-8"
    )
    # inference/tools/eval.py reads run_config.json for optional metadata (model, pass_offset, seed)
    (args.out / "run_config.json").write_text(json.dumps({"label": label, "games": games, "model": args.model}, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
