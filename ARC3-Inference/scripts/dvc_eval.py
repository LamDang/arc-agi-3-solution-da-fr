"""Run the DVC eval stage: play and score the games set in params.yaml.

dvc.yaml runs this from ARC3-Inference/ (`dvc exp run` or `dvc repro`). It
calls `make interactive` with the `eval.make` settings and the `eval.env`
environment, writing the run to runs/dvc-eval, then `make score_run` on it,
then writes the scores and the API token spend to metrics.json.

The options run the same settings outside DVC, for example two arms of an
experiment side by side:

    uv run --no-sync python scripts/dvc_eval.py --run-dir runs/engine-code \
        --metrics runs/engine-code.metrics.json --env ARC3_GAME_CODE_DIR=game_code

Each run directory gets eval_settings.json: the make variables and the
harness environment it ran with.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from inference.utils.run_artifacts import open_log

PARAMS_PATH = Path("params.yaml")
RUN_DIR = Path("runs/dvc-eval")
METRICS_PATH = Path("metrics.json")
# Harness settings read from the environment. Any not listed in eval.env are
# dropped so the shell cannot change a run without params.yaml showing it.
CONTROLLED_ENV_PREFIXES = ("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_")
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cost")


def _load_params() -> dict[str, Any]:
    # BaseLoader reads every value as its literal text. A typed load would
    # turn on/off/yes/no into booleans (YAML 1.1), and `dvc exp run -S`
    # rewrites params.yaml without the quotes that prevent it.
    return yaml.load(PARAMS_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["eval"]


def _run_env(settings: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CONTROLLED_ENV_PREFIXES) or key.endswith("_API_KEY")
    }
    dropped = sorted(set(os.environ) - set(env) - set(settings))
    if dropped:
        print(f"dvc_eval: not passing shell settings absent from params.yaml: {', '.join(dropped)}")
    env.update(settings)
    return env


def _usage_totals(run_dir: Path) -> dict[str, float]:
    # Request logs exist only with ANALYZER_SAVE_REQUEST_LOGS=true. Rolling
    # summary requests are not logged, so these totals leave them out.
    totals = {key: 0.0 for key in USAGE_KEYS}
    found = False
    logs = [*run_dir.glob("*requests.jsonl"), *run_dir.glob("*requests.jsonl.xz")]
    for path in sorted(logs):
        with open_log(path) as lines:
            for line in lines:
                # Request lines carry the full conversation; skip parsing them.
                if '"event": "response"' not in line:
                    continue
                usage = json.loads(line).get("usage") or {}
                found = True
                for key in USAGE_KEYS:
                    if isinstance(usage.get(key), (int, float)):
                        totals[key] += usage[key]
    return totals if found else {}


def _write_metrics(run_dir: Path, metrics_path: Path) -> None:
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {
        "score": evaluation["score"],
        "games": {
            game["game_id"]: {
                "score": game["score"],
                "levels_completed": game["levels_completed"],
                "total_levels": game["total_levels"],
            }
            for game in evaluation["games"]
        },
    }
    usage = _usage_totals(run_dir)
    if usage:
        metrics["api_usage"] = {
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "cost_usd": round(usage["cost"], 4),
        }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    for option, section in (("--make", "eval.make"), ("--env", "eval.env")):
        parser.add_argument(
            option,
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help=f"Setting to add to or override {section}. Repeatable.",
        )
    return parser.parse_args()


def _with_overrides(settings: dict[str, str], overrides: list[str]) -> dict[str, str]:
    settings = dict(settings)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"Expected KEY=VALUE, got {item!r}")
        settings[key] = value
    return settings


def main() -> int:
    args = _parse_args()
    run_dir: Path = args.run_dir
    params = _load_params()
    make_vars = _with_overrides(params.get("make") or {}, args.make)
    make_vars.update(ENVIRONMENTS_DIR="environment_files", EXPERIMENT_DIR=str(run_dir))
    settings = _with_overrides(params.get("env") or {}, args.env)
    env = _run_env(settings)

    subprocess.run(
        ["make", "interactive", *(f"{key}={value}" for key, value in make_vars.items())],
        env=env,
        check=True,
    )
    (run_dir / "eval_settings.json").write_text(
        json.dumps({"make": make_vars, "env": settings}, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(["make", "score_run", f"SCORE_RUN_DIR={run_dir}"], env=env, check=True)
    _write_metrics(run_dir, args.metrics)
    print(f"dvc_eval: wrote {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
