"""Recorded game traces: every action of a run and everything the engine returned.

A trace is built by replaying a run's logged actions through the real engine.
The games are deterministic, so the replay reproduces the run's observations
exactly (`verify_against_events` checks the final board of every step against
the run's event log), and it also recovers what the logs do not keep: every
animation frame of every action.

On disk a trace is two files in one directory:

- ``trace.json``: game id, metadata and one record per step (action, engine
  state, levels completed, available actions, number of frames).
- ``frames.npz``: ``frames`` (all frames of all steps, int8, N x 64 x 64) and
  ``offsets`` (step i owns ``frames[offsets[i]:offsets[i + 1]]``).

Step 0 is always the RESET that starts the game.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7"}
# Labels the harness shows the model and writes to its event logs.
MODEL_LABELS = {"RESET": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "MOUSE": 6, "UNDO": 7}
_MOUSE_RE = re.compile(r"MOUSE\(row=(-?\d+),\s*col=(-?\d+)\)", re.IGNORECASE)


@dataclass(frozen=True)
class Action:
    id: int
    x: int | None = None
    y: int | None = None

    @property
    def name(self) -> str:
        return ACTION_NAMES[self.id]

    def data(self) -> dict[str, int]:
        return {"x": int(self.x), "y": int(self.y)} if self.id == 6 else {}

    def __str__(self) -> str:
        if self.id == 6:
            return f"ACTION6(x={self.x}, y={self.y})"
        return self.name

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id}
        if self.id == 6:
            out.update(x=self.x, y=self.y)
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Action:
        return cls(int(data["id"]), data.get("x"), data.get("y"))


@dataclass
class Step:
    """One action and everything the engine returned for it."""

    index: int
    action: Action
    frames: np.ndarray  # (n_frames, 64, 64) int8; n_frames is 0 when the engine refused the action
    state: str  # NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int
    win_levels: int
    available_actions: list[int]

    @property
    def outcome(self) -> str:
        """The game's status after the action, the same as ``state``: the name the agent's kernel uses
        (``state`` and ``status`` would read as the engine's State and State.status)."""
        return self.state

    @property
    def last(self) -> np.ndarray | None:
        """The final frame: the state the action left the game in."""
        return self.frames[-1] if len(self.frames) else None

    @property
    def n_frames(self) -> int:
        return int(len(self.frames))

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.to_json(),
            "n_frames": self.n_frames,
            "state": self.state,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "available_actions": list(self.available_actions),
        }

    def __repr__(self) -> str:
        return (
            f"Step({self.index}: {self.action}, frames={self.n_frames}, state={self.state}, "
            f"levels_completed={self.levels_completed}/{self.win_levels})"
        )


@dataclass
class Trace:
    game_id: str
    steps: list[Step]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]

    @property
    def actions(self) -> list[Action]:
        return [step.action for step in self.steps]

    def level_starts(self) -> dict[int, int]:
        """Level number (= levels completed) -> index of the step whose final frame
        first shows that level. Level 0 starts at step 0."""
        starts = {0: 0}
        for step in self.steps:
            if step.levels_completed not in starts:
                starts[step.levels_completed] = step.index
        return starts

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        offsets = np.zeros(len(self.steps) + 1, dtype=np.int64)
        for i, step in enumerate(self.steps):
            offsets[i + 1] = offsets[i] + step.n_frames
        frames = (
            np.concatenate([s.frames for s in self.steps if s.n_frames])
            if any(s.n_frames for s in self.steps)
            else np.zeros((0, 64, 64), np.int8)
        )
        np.savez_compressed(directory / "frames.npz", frames=frames.astype(np.int8), offsets=offsets)
        payload = {"game_id": self.game_id, "meta": self.meta, "steps": [s.record() for s in self.steps]}
        (directory / "trace.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Trace:
        directory = Path(directory)
        payload = json.loads((directory / "trace.json").read_text(encoding="utf-8"))
        with np.load(directory / "frames.npz") as data:
            frames, offsets = data["frames"], data["offsets"]
        steps = [
            Step(
                index=rec["index"],
                action=Action.from_json(rec["action"]),
                frames=frames[offsets[i] : offsets[i + 1]],
                state=rec["state"],
                levels_completed=rec["levels_completed"],
                win_levels=rec["win_levels"],
                available_actions=list(rec["available_actions"]),
            )
            for i, rec in enumerate(payload["steps"])
        ]
        return cls(payload["game_id"], steps, payload.get("meta", {}))


# --- Actions from a harness run ----------------------------------------------


def parse_action_label(label: str) -> Action:
    """Parse a harness action label: UP, DOWN, LEFT, RIGHT, SPACE, UNDO, RESET,
    ACTIONn or MOUSE(row=r, col=c) (row is the engine's y, col its x)."""
    text = label.strip().upper()
    match = _MOUSE_RE.fullmatch(text)
    if match:
        return Action(6, x=int(match.group(2)), y=int(match.group(1)))
    if text in MODEL_LABELS:
        return Action(MODEL_LABELS[text])
    if text.startswith("ACTION") and text[6:].isdigit():
        return Action(int(text[6:]))
    raise ValueError(f"unrecognised action label {label!r}")


def events_path(run_dir: Path, game: str, pass_index: int = 0) -> Path:
    """The run's event log for one game, matched by game id or its prefix (ls20)."""
    matches = sorted((Path(run_dir) / "artifacts").glob(f"{game}*_p{pass_index}_events.jsonl"))
    if not matches:
        raise FileNotFoundError(f"no events log for {game!r} pass {pass_index} in {run_dir}/artifacts")
    return matches[0]


def load_run_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return [e for e in events if e.get("type") in ("initial", "action")]


def actions_from_events(events: Iterable[dict[str, Any]]) -> list[Action]:
    """The game's action sequence: the initial RESET, then every executed action
    (including the harness's automatic RESETs, which it logs as actions)."""
    actions = []
    for event in events:
        if event["type"] == "initial":
            actions.append(Action(0))
        else:
            actions.append(parse_action_label(str(event.get("action_display") or event.get("action_name"))))
    return actions


# --- Replay through the real engine ------------------------------------------


def find_game_file(game: str, environments_dir: Path) -> Path:
    matches = sorted(Path(environments_dir).glob(f"{game[:4]}/*/{game[:4]}.py"))
    if not matches:
        raise FileNotFoundError(f"no game file for {game!r} under {environments_dir}")
    return matches[0]


def load_game_class(game_file: Path) -> type:
    """Load an ARCBaseGame subclass from a game file, the way arc_agi does."""
    import types

    from arcengine import ARCBaseGame

    module = types.ModuleType(f"engine_source_{game_file.stem}")
    module.__file__ = str(game_file)
    exec(compile(game_file.read_text(encoding="utf-8"), str(game_file), "exec"), module.__dict__)
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame and obj.__module__ == module.__name__
    ]
    named = [c for c in classes if c.__name__.lower() == game_file.stem.lower()]
    if named:
        return named[0]
    if len(classes) == 1:
        return classes[0]
    raise ValueError(f"cannot pick the game class in {game_file}: {[c.__name__ for c in classes]}")


def new_game(cls: type, seed: int = 0) -> Any:
    """Instantiate like arc_agi's LocalEnvironmentWrapper: pass seed if accepted."""
    return cls(seed=seed) if "seed" in inspect.signature(cls).parameters else cls()


def perform(game: Any, action: Action) -> dict[str, Any]:
    """Run one action on a live engine and return its observation."""
    from arcengine import ActionInput, GameAction

    result = game.perform_action(ActionInput(id=GameAction.from_id(action.id), data=action.data()), raw=True)
    frames = [np.asarray(f, dtype=np.int8) for f in result.frame]
    return {
        "frames": np.stack(frames) if frames else np.zeros((0, 64, 64), np.int8),
        "state": result.state.name if hasattr(result.state, "name") else str(result.state),
        "levels_completed": int(result.levels_completed),
        "win_levels": int(result.win_levels),
        "available_actions": [int(a) for a in result.available_actions],
    }


def record_trace(game_cls: type, game_id: str, actions: list[Action], meta: dict[str, Any] | None = None) -> Trace:
    """Replay ``actions`` on a fresh engine and record every observation.

    RESET only restarts the current level, as in the harness (ONLY_RESET_LEVELS)."""
    os.environ["ONLY_RESET_LEVELS"] = "true"
    game = new_game(game_cls)
    steps = []
    for index, action in enumerate(actions):
        obs = perform(game, action)
        steps.append(Step(index=index, action=action, **obs))
    return Trace(game_id, steps, dict(meta or {}))


def verify_against_events(trace: Trace, events: list[dict[str, Any]]) -> list[int]:
    """Steps whose replayed final frame differs from the board the run logged."""
    bad = []
    for step, event in zip(trace.steps, events, strict=True):
        board = np.asarray(event["board"], dtype=np.int8)
        if step.last is None or board.shape != step.last.shape or not np.array_equal(board, step.last):
            bad.append(step.index)
    return bad


def trace_from_run(run_dir: Path, game: str, environments_dir: Path, pass_index: int = 0) -> tuple[Trace, list[int]]:
    """Build a game's trace from a harness run directory and verify it."""
    path = events_path(run_dir, game, pass_index)
    events = load_run_events(path)
    actions = actions_from_events(events)
    game_file = find_game_file(game, environments_dir)
    trace = record_trace(
        load_game_class(game_file),
        game[:4],
        actions,
        meta={"source": "run", "run_dir": str(run_dir), "events": path.name, "pass": pass_index},
    )
    mismatches = verify_against_events(trace, events)
    trace.meta["verified_steps"] = len(events) - len(mismatches)
    trace.meta["replay_mismatches"] = mismatches
    return trace, mismatches
