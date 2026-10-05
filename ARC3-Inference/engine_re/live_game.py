"""A real ARC-AGI-3 game being played, and the trace of everything played so far.

The play-and-model agent (engine_re.play_agent) plays the game through this class: it steps the
arcengine game directly, as engine_re.trace does when it rebuilds a recording, and keeps every step in a
Trace, so the tests, the kernel's `recording`, the resume and the viewer record all come from one
object. Step 0 is the RESET that starts the game (``start``), as in every recording.

Scoring follows TAAF (tufa-arc-agi-framework ``taaf.game.GameRun``): every action after the opening
RESET counts, including automatic RESETs after a game over, against the level it was played in; a
completed level scores min(115, (baseline / actions)^2 * 100), weighted by its 1-based index, and the
game score is the weighted mean capped at the share of the levels that scored. ``benchmark_json``
writes the TAAF shape ``make score_run`` reads; ``events`` the base harness's viewer event sidecar
(``artifacts/<game_id>_p0_events.jsonl``, as inference.framework.solver names it) that
``engine_re.trace.trace_from_run`` reads.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine_re.trace import Action, Step, Trace, find_game_file, load_game_class, move_label, new_game, perform


class LiveGame:
    def __init__(self, game: str, environments_dir: Path, trace: Trace | None = None):
        self.game = game
        self.game_file = find_game_file(game, Path(environments_dir))
        self.game_id = f"{self.game_file.stem}-{self.game_file.parent.name}"
        meta_path = self.game_file.parent / "metadata.json"
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if self.metadata.get("game_id"):
            self.game_id = str(self.metadata["game_id"])
        os.environ["ONLY_RESET_LEVELS"] = "true"
        self.cls = load_game_class(self.game_file)
        self.engine = new_game(self.cls)
        self.started = time.time()
        self.trace = Trace(game[:4], [], {"source": "play", "game_id": self.game_id, "game_file": str(self.game_file)})
        if trace is not None:  # a resumed run: feed the real game the actions played, checking every frame
            self.replay(trace)

    # --- playing --------------------------------------------------------------------------------

    def start(self) -> Step:
        """Step 0: the RESET that starts the game."""
        if len(self.trace):
            raise RuntimeError("the game has started already")
        return self.perform(Action(0))

    def perform(self, action: Action) -> Step:
        obs = perform(self.engine, action)
        step = Step(index=len(self.trace), action=action, **obs)
        self.trace.steps.append(step)
        return step

    def replay(self, trace: Trace) -> None:
        """A fresh game fed `trace`'s actions; every observation must equal the saved one (the games are
        deterministic), so the live game is exactly where the saved trace left it. The saved meta (the
        play agent's unexplained steps and resync points) is kept."""
        self.trace.meta.update({k: v for k, v in trace.meta.items() if k not in self.trace.meta})
        for saved in trace.steps:
            step = self.perform(saved.action)
            same = (
                step.state == saved.state and step.levels_completed == saved.levels_completed
                and step.n_frames == saved.n_frames and (step.last is None or np.array_equal(step.last, saved.last))
            )
            if not same:
                raise RuntimeError(f"{self.game}: replaying the saved trace diverged at step {saved.index}")

    @property
    def last(self) -> Step:
        return self.trace.steps[-1]

    @property
    def level(self) -> int:
        """The level being played: levels completed, capped at the last level."""
        return min(self.last.levels_completed, max(0, self.last.win_levels - 1))

    @property
    def won(self) -> bool:
        return self.last.state == "WIN"

    @property
    def game_over(self) -> bool:
        return self.last.state == "GAME_OVER"

    def save(self, directory: Path) -> None:
        self.trace.save(Path(directory))

    # --- scoring --------------------------------------------------------------------------------

    @property
    def baseline_actions(self) -> list[int] | None:
        value = self.metadata.get("baseline_actions")
        return [int(v) for v in value] if value else None

    def actions_per_level(self) -> list[int]:
        """Actions played in each level: every step but the opening RESET, against the level it was
        played in (levels completed before it), as TAAF counts them."""
        counts = [0] * max(1, self.trace[0].win_levels if len(self.trace) else 1)
        for k in range(1, len(self.trace)):
            level = min(self.trace[k - 1].levels_completed, len(counts) - 1)
            counts[level] += 1
        return counts

    @property
    def actions(self) -> int:
        return max(0, len(self.trace) - 1)

    @property
    def levels_completed(self) -> int:
        """The most levels completed so far (TAAF keeps the maximum)."""
        return max((s.levels_completed for s in self.trace.steps), default=0)

    def score(self) -> float | None:
        """TAAF's formula (taaf.game.GameRun._compute_final_score); None without baselines."""
        baseline = self.baseline_actions
        if not baseline or not len(self.trace):
            return None
        counts = self.actions_per_level()
        total_score = 0.0
        total_weights = max_weights = 0
        for level in range(self.trace[0].win_levels):
            weight = level + 1
            total_weights += weight
            actions = counts[level] if level < len(counts) else 0
            base = baseline[level] if level < len(baseline) else 0
            if level < self.levels_completed and actions > 0:
                level_score = min(115.0, (base / actions) ** 2 * 100)
            else:
                level_score = 0.0
            if level_score > 0:
                max_weights += weight
            total_score += level_score * weight
        if not total_weights:
            return 0.0
        return min(total_score / total_weights, max_weights / total_weights * 100)

    # --- records for the base harness's tools ---------------------------------------------------

    def events(self) -> list[dict[str, Any]]:
        """The viewer event sidecar records (inference/framework/solver.py): one "initial" event for the
        opening RESET and one "action" event per step after it."""
        out = []
        for k, step in enumerate(self.trace.steps):
            frame = step.last if step.last is not None else (self.trace[k - 1].last if k else np.zeros((64, 64), np.int8))
            level_before = self.trace[k - 1].levels_completed if k else 0
            levels = max(1, int(step.win_levels))
            base = {  # as inference.framework.solver._base_viewer_event: "level" is the 1-based level shown after the step
                "board": [[int(v) for v in row] for row in np.asarray(frame)],
                "score": int(step.levels_completed),
                "state": step.state,
                "level": levels if step.state == "WIN" else max(1, min(levels, step.levels_completed + 1)),
                "run_status": "won" if step.state == "WIN" else "playing",
                "action_num": k,
                "analysis_step": None,
                "action_display": move_label(step.action),
                "action_name": step.action.name,
            }
            if k == 0:
                out.append({**base, "type": "initial", "title": "Initial State", "reward": 0.0})
            else:
                out.append({
                    **base, "type": "action", "title": f"Action {k}",
                    "reward": float(step.levels_completed - level_before),
                    "board_changed": not (self.trace[k - 1].last is not None and step.last is not None and np.array_equal(self.trace[k - 1].last, step.last)),
                    "level_completed": step.levels_completed > level_before and step.state != "WIN",
                    "game_over": step.state == "GAME_OVER",
                    "run_complete": step.state == "WIN",
                    "done": step.state == "WIN",
                })
        return out

    def events_path(self, run_dir: Path) -> Path:
        """artifacts/<game_id>_p0_events.jsonl: the base harness's name for the sidecar of
        <game_id>_p0_viewer_data.json (inference.utils.viewer_artifacts), which engine_re.trace.events_path finds."""
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.game_id)  # inference.utils.run_artifacts.artifact_stem
        return Path(run_dir) / "artifacts" / f"{stem}_p0_events.jsonl"

    def write_events(self, run_dir: Path) -> Path:
        path = self.events_path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for event in self.events():
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
        return path

    def game_run(
        self, tokens_per_step: list[int] | None = None, state: str | None = None, note: str | None = None, final_tokens: int = 0,
    ) -> dict[str, Any]:
        """A TAAF GameRun as JSON (taaf.game.GameRun.to_json_dict): the actions after the opening RESET
        with the output tokens spent on each (`tokens_per_step[k]` for step k), the per-level counts and the
        final score (TAAF's: 0 without baselines); `final_tokens`: output tokens spent after the last action."""
        tokens = list(tokens_per_step or [])
        history = []
        for k in range(1, len(self.trace)):
            action = self.trace[k].action
            history.append({
                "action": {"id": action.name, "data": action.data()},
                "generated_tokens": int(tokens[k]) if k < len(tokens) else 0,
                "uncached_input_tokens": 0,
                "wallclock_seconds": 0.0,
            })
        if state is None:
            state = "won" if self.won else "gave_up"
        return {
            "game_id": self.game_id,
            "number_of_levels": int(self.trace[0].win_levels) if len(self.trace) else 0,
            "base_actions_per_level": self.baseline_actions,
            "hint": None,
            "state": state,
            "history": history,
            "record_intermediate_states": False,
            "actions_per_level": self.actions_per_level(),
            "levels_completed": int(self.levels_completed),
            "final_score": self.score() or 0.0,
            "solver_note": note,
            "solver_analysis_html": None,
            "final_generated_tokens": int(final_tokens),
            "final_uncached_input_tokens": 0,
            "final_wallclock_seconds": round(time.time() - self.started, 1),
            "started_at": datetime.fromtimestamp(self.started).isoformat(),
        }


def benchmark_json(label: str, game_runs: list[dict[str, Any]], started: float, solver_label: str = "engine_re.play_agent") -> dict[str, Any]:
    """A TAAF benchmark.json (taaf.benchmark.Benchmark.to_json_dict) with one pass over `game_runs`."""
    return {
        "label": label,
        "n_passes": 1,
        "solver_label": solver_label,
        "start_time": datetime.fromtimestamp(started).isoformat(),
        "end_time": datetime.now().isoformat(),
        "game_weights": None,
        "periodic_save_interval_s": 600.0,
        "game_runs": game_runs,
    }
