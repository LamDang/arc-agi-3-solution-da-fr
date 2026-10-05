"""Run a candidate engine on a list of actions, in its own sandboxed process.

    python -m engine_re.candidate_runner ENGINE ACTIONS_JSON OUT_DIR [--start-level L]
        [--win-levels N --available-actions JSON --levels JSON]

Two kinds of engine are accepted: a module with ``make_level`` and ``step``
(the simple interface, ``engine_re.game_api``; it needs --win-levels and
--available-actions, runs the contract tests first and reports them under
"contract"), or a module defining an ``arcengine.ARCBaseGame`` subclass.

The process never sees the expected observations: it gets only the actions,
and after loading the engine source it can read no files at all (only the
Python installation) and write only to OUT_DIR. It writes ``OUT_DIR/frames.npz``
and ``OUT_DIR/result.json``; an exception or a step timeout ends the run at
that step, with the traceback in the result.
"""

from __future__ import annotations

import argparse
import json
import linecache
import signal
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

import arcengine  # noqa: F401  (imported before the guard so the engine can use it)
from arcengine import ARCBaseGame

from engine_re import game_api, guard
from engine_re.trace import Action, new_game, perform

MODULE_NAME = "candidate_engine"


class StepTimeout(Exception):
    pass


def load_module(source: str, path: str) -> types.ModuleType:
    """Execute the engine source as a module."""
    # The guard blocks reading the file again, so seed linecache for tracebacks.
    linecache.cache[path] = (len(source), None, source.splitlines(True), path)
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = path
    sys.modules[MODULE_NAME] = module
    exec(compile(source, path, "exec", dont_inherit=True), module.__dict__)
    return module


def game_class(module: types.ModuleType) -> type:
    """The module's ARCBaseGame subclass (the most derived one)."""
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame and obj.__module__ == MODULE_NAME
    ]
    if not classes:
        raise TypeError("the engine module defines neither make_level and step nor a subclass of arcengine.ARCBaseGame")
    # Prefer the most derived class if the module defines a hierarchy.
    leaves = [c for c in classes if not any(o is not c and issubclass(o, c) for o in classes)]
    return leaves[-1]


def _short_traceback(exc: BaseException, path: str) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    # Keep the frames from the engine file and the arcengine library; drop this runner.
    kept = [f for f in frames if f.filename == path or "arcengine" in f.filename or "game_api" in f.filename] or frames[-3:]
    lines = traceback.format_list(kept[-8:])
    return "Traceback (most recent call last):\n" + "".join(lines) + f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("actions")
    parser.add_argument("out_dir")
    parser.add_argument("--start-level", type=int, default=None)
    parser.add_argument("--step-timeout", type=float, default=5.0)
    parser.add_argument("--win-levels", type=int, default=None)
    parser.add_argument("--available-actions", default=None, help="JSON list")
    parser.add_argument("--levels", default=None, help="JSON list of the levels the recording reaches (contract tests)")
    args = parser.parse_args()

    engine_path = str(Path(args.engine).resolve())
    source = Path(engine_path).read_text(encoding="utf-8")
    actions = [Action.from_json(a) for a in json.loads(Path(args.actions).read_text(encoding="utf-8"))]
    out_dir = Path(args.out_dir).resolve()
    guard.install(read_roots=[], write_roots=[str(out_dir)])

    result: dict = {"steps": [], "error": None, "error_step": None, "start_frame": None, "interface": None, "contract": None}
    frames: list[np.ndarray] = []
    offsets = [0]

    def on_alarm(signum, frame):  # noqa: ARG001
        raise StepTimeout(f"step took longer than {args.step_timeout:g}s")

    signal.signal(signal.SIGALRM, on_alarm)
    started = time.time()
    step_index = None
    try:
        signal.setitimer(signal.ITIMER_REAL, args.step_timeout * 4)
        module = load_module(source, engine_path)
        if game_api.is_simple_engine(module):
            result["interface"] = "simple"
            if args.win_levels is None or args.available_actions is None:
                raise ValueError("a make_level/step engine needs --win-levels and --available-actions")
            available = json.loads(args.available_actions)
            signal.setitimer(signal.ITIMER_REAL, args.step_timeout * 12)
            result["contract"] = game_api.contract_checks(
                module, source, levels=json.loads(args.levels or "[0]"), available_actions=available
            )
            # The contract tests modify states on purpose; replay on a freshly loaded module.
            signal.setitimer(signal.ITIMER_REAL, args.step_timeout * 4)
            module = load_module(source, engine_path)
            game = game_api.GameRunner(module, args.win_levels, available)
            play = game.perform
            if args.start_level is not None:
                game.set_level(args.start_level)
                game.score = args.start_level
                result["start_frame"] = game_api.render(game.state).tolist()
        else:
            result["interface"] = "arcengine"
            game = new_game(game_class(module))
            play = lambda action, game=game: perform(game, action)  # noqa: E731
            if args.start_level is not None:
                game.set_level(args.start_level)
                game._score = args.start_level
                result["start_frame"] = np.asarray(game.camera.render(game.current_level.get_sprites()), np.int8).tolist()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for step_index, action in enumerate(actions):
            signal.setitimer(signal.ITIMER_REAL, args.step_timeout)
            obs = play(action)
            signal.setitimer(signal.ITIMER_REAL, 0)
            frames.extend(obs.pop("frames"))
            offsets.append(len(frames))
            result["steps"].append(obs)
    except BaseException as exc:  # noqa: BLE001  (report every failure of model-written code)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["error"] = _short_traceback(exc, engine_path)
        result["error_step"] = step_index
    result["seconds"] = round(time.time() - started, 2)
    np.savez_compressed(
        out_dir / "frames.npz",
        frames=np.stack(frames).astype(np.int8) if frames else np.zeros((0, 64, 64), np.int8),
        offsets=np.asarray(offsets, np.int64),
    )
    (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
