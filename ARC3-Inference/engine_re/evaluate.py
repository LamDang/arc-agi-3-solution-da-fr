"""Does a reverse-engineered engine behave like the real one beyond the recording?

    uv run --no-sync python -m engine_re.evaluate runs/engine-re/<name> [--engine best|final]

For every game directory of an experiment, three checks:

1. Recorded replay: the agent's own test, the recorded actions from step 0.
2. Held-out rollouts: for each level the recording reached, both engines replay
   the recorded actions up to the step that entered that level, then play the
   same random action sequence (keys drawn from the advertised actions; clicks
   aimed mostly at object pixels; occasional RESET; RESET after a game over).
   Only the random part is compared, through the public interface alone. A
   rollout stops when the real engine leaves the levels the recording covers.
   Reported: steps that match exactly, those among steps that changed the
   screen, and the mean number of steps before the first mismatch.
3. A source scan for patterns that could read answers instead of computing them.

Writes ``<game>/evaluation.json`` and ``<experiment>/evaluation.md``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from engine_re.guard import scan_engine_source
from engine_re.tester import check_step, replay_test, run_candidate
from engine_re.trace import Action, Step, Trace, find_game_file, load_game_class, new_game, perform


def click_targets(frame: np.ndarray, recorded: list[Action]) -> list[tuple[int, int]]:
    """Screen pixels worth clicking: recorded click positions and every pixel not
    of the frame's most common colour (objects rather than background)."""
    values, counts = np.unique(frame, return_counts=True)
    background = values[np.argmax(counts)]
    ys, xs = np.nonzero(frame != background)
    targets = list(zip(xs.tolist(), ys.tolist()))
    targets += [(a.x, a.y) for a in recorded if a.id == 6]
    return targets or [(32, 32)]


def random_actions(trace: Trace, level: int, n: int, rng: random.Random) -> list[Action]:
    available = [a for a in trace[0].available_actions if a != 0]
    start_frame = trace[trace.level_starts()[level]].last
    targets = click_targets(start_frame, trace.actions)
    actions = []
    for _ in range(n):
        if rng.random() < 0.03:
            actions.append(Action(0))
            continue
        action_id = rng.choice(available)
        if action_id == 6:
            if rng.random() < 0.85:
                x, y = rng.choice(targets)
            else:
                x, y = rng.randrange(64), rng.randrange(64)
            actions.append(Action(6, x=x, y=y))
        else:
            actions.append(Action(action_id))
    return actions


def real_rollout(game_cls: type, prefix: list[Action], actions: list[Action], max_level: int) -> tuple[list[Action], list[Step]]:
    """Replay ``prefix`` on a fresh real engine, then play ``actions``, inserting
    a RESET after every game over and stopping before the engine leaves the
    recorded levels. Returns the actions played after the prefix and the real
    observations for them."""
    game = new_game(game_cls)
    for action in prefix:
        perform(game, action)
    played, steps = [], []
    queue = list(actions)
    while queue:
        if steps and steps[-1].state == "GAME_OVER":
            action = Action(0)
        else:
            action = queue.pop(0)
        obs = perform(game, action)
        played.append(action)
        steps.append(Step(index=len(steps), action=action, **obs))
        if obs["levels_completed"] > max_level or obs["state"] == "WIN":
            break
    return played, steps


def evaluate_game(game_dir: Path, engine_path: Path, environments_dir: Path, rollouts: int, length: int, seed: int) -> dict[str, Any]:
    import os

    os.environ["ONLY_RESET_LEVELS"] = "true"
    trace = Trace.load(game_dir / "trace")
    game_cls = load_game_class(find_game_file(trace.game_id, environments_dir))
    replay = replay_test(engine_path, trace, details=1, scratch_root=game_dir)
    out: dict[str, Any] = {"engine": engine_path.name, "replay": replay.summary(), "levels": {}}
    playable = [lvl for lvl in sorted(trace.level_starts()) if lvl < trace[0].win_levels]
    max_level = max(playable)
    rng = random.Random(seed)
    total = exact = 0
    changing_total = changing_exact = 0
    for level in playable:
        level_total = level_exact = level_changing = level_changing_exact = full = 0
        prefixes, examples = [], []
        entry = trace.level_starts()[level]
        recorded = trace.actions[: entry + 1]
        for _ in range(rollouts):
            actions, real_steps = real_rollout(game_cls, recorded, random_actions(trace, level, length, rng), max_level)
            result, frames = run_candidate(engine_path, [a.to_json() for a in recorded + actions], scratch_root=game_dir)
            got, frames = result.get("steps", [])[len(recorded) :], frames[len(recorded) :]
            checks = [
                check_step(step, got[k] if k < len(got) else None, frames[k] if k < len(frames) else None)
                for k, step in enumerate(real_steps)
            ]
            ok = [c.ok for c in checks]
            level_total += len(ok)
            level_exact += sum(ok)
            # Steps where the real screen changed or animated: a do-nothing engine cannot match these.
            for k, step in enumerate(real_steps):
                prev = real_steps[k - 1].last if k else trace[entry].last
                if step.n_frames > 1 or prev is None or step.last is None or not np.array_equal(prev, step.last):
                    level_changing += 1
                    level_changing_exact += ok[k]
            prefix = ok.index(False) if False in ok else len(ok)
            prefixes.append(prefix)
            full += prefix == len(ok)
            if prefix < len(ok) and len(examples) < 3:
                examples.append(
                    {"step": prefix, "action": str(actions[prefix]), "problems": checks[prefix].problems, "error": result.get("error")}
                )
        total += level_total
        exact += level_exact
        changing_total += level_changing
        changing_exact += level_changing_exact
        out["levels"][level] = {
            "steps": level_total,
            "exact": level_exact,
            "exact_rate": round(level_exact / level_total, 4) if level_total else None,
            "mean_steps_before_first_mismatch": round(float(np.mean(prefixes)), 1),
            "changing_steps": level_changing,
            "changing_exact": level_changing_exact,
            "rollouts": rollouts,
            "rollouts_fully_matching": full,
            "first_mismatch_examples": examples,
        }
    out["heldout_steps"] = total
    out["heldout_exact"] = exact
    out["heldout_exact_rate"] = round(exact / total, 4) if total else None
    out["heldout_changing_steps"] = changing_total
    out["heldout_changing_exact"] = changing_exact
    out["heldout_changing_exact_rate"] = round(changing_exact / changing_total, 4) if changing_total else None
    source = engine_path.read_text(encoding="utf-8")
    out["source_lines"] = len(source.splitlines())
    out["source_bytes"] = len(source.encode())
    out["suspicious_patterns"] = scan_engine_source(source)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--engine", choices=["best", "final"], default="best")
    parser.add_argument("--environments-dir", type=Path, default=Path("environment_files"))
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", default=None)
    args = parser.parse_args()

    games = args.games.split(",") if args.games else sorted(p.name for p in args.experiment.iterdir() if (p / "trace").is_dir())
    rows = []
    for game in games:
        game_dir = args.experiment / game
        engine = game_dir / ("engine_best.py" if args.engine == "best" else "workspace/engine.py")
        if not engine.exists():
            engine = game_dir / "workspace/engine.py"
        result = evaluate_game(game_dir, engine, args.environments_dir, args.rollouts, args.length, args.seed)
        (game_dir / f"evaluation_{args.engine}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append((game, result))
        print(game, json.dumps({k: result[k] for k in ("replay", "heldout_exact_rate", "suspicious_patterns")}), flush=True)

    lines = [
        f"Engine evaluated: `{args.engine}`. Held-out: {args.rollouts} random rollouts of {args.length} actions per recorded level.",
        "",
        "| game | recorded replay exact | held-out exact steps | held-out steps that changed the screen | per level (exact rate, mean steps to first mismatch, rollouts fully matching) | lines | flagged patterns |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for game, r in rows:
        rep = r["replay"]
        levels = "; ".join(
            f"L{lvl}: {v['exact_rate']:.0%}, {v['mean_steps_before_first_mismatch']}, {v['rollouts_fully_matching']}/{v['rollouts']}"
            for lvl, v in r["levels"].items()
            if v["steps"]
        )
        flags = ", ".join(r["suspicious_patterns"]) or "none"
        lines.append(
            f"| {game} | {rep['exact']}/{rep['total']} | {r['heldout_exact']}/{r['heldout_steps']} ({r['heldout_exact_rate']:.0%}) | "
            f"{r['heldout_changing_exact']}/{r['heldout_changing_steps']} ({r['heldout_changing_exact_rate']:.0%}) | {levels} | {r['source_lines']} | {flags} |"
        )
    (args.experiment / f"evaluation_{args.engine}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
