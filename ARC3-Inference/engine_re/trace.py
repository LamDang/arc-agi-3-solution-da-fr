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

import ast
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


def move_label(action: Action) -> str:
    """The harness's label for an action: UP, DOWN, LEFT, RIGHT, SPACE, UNDO, RESET or MOUSE(row=r, col=c)
    (the event logs' form, which parse_action_label reads back)."""
    if action.id == 6:
        return f"MOUSE(row={action.y}, col={action.x})"
    return next((label for label, i in MODEL_LABELS.items() if i == action.id), ACTION_NAMES[action.id])


def action_code(action: Any) -> str:
    """An action as the code that builds the fixed block's Action: "Action(4)", "Action(6, x=12, y=40)",
    "Action(0)" for RESET. It is how the kernel's Action prints (game_api.canonical) and how the play mode's
    messages name moves, and commit_moves takes it back as it is (parse_move). The cell is left out: the harness
    computes it for a click."""
    aid = int(action.id)
    x, y = getattr(action, "x", None), getattr(action, "y", None)
    if aid == 6:
        return f"Action(6, x={x}, y={y})"
    if x or y:  # (a key with a position, which only code can make: shown, and ignored when sent)
        return f"Action({aid}, x={x}, y={y})"
    return f"Action({aid})"


_ACTION_FIELDS = ("id", "x", "y", "cell")


def _action_from_call(node: ast.Call, text: str) -> Action:
    """The action a call Action(id, x=0, y=0, cell=None) builds, its arguments being literals (the printed form of
    the fixed block's Action, its dataclass form Action(id=6, x=3, y=4, cell=(1, 2)) or the recorded action's
    Action(id=1, x=None, y=None)). The cell is ignored: the harness computes it."""
    if len(node.args) > len(_ACTION_FIELDS) or any(isinstance(a, ast.Starred) for a in node.args):
        raise ValueError(f"not an action: {text!r}")
    values: dict[str, Any] = {}
    try:
        for name, arg in zip(_ACTION_FIELDS, node.args):
            values[name] = ast.literal_eval(arg)
        for kw in node.keywords:
            if kw.arg not in _ACTION_FIELDS or kw.arg in values:
                raise ValueError(f"not an action: {text!r} (Action takes id, x, y and cell)")
            values[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(str(exc) if str(exc).startswith("not an action") else f"not an action: {text!r}") from None
    aid = values.get("id")
    if isinstance(aid, bool) or not isinstance(aid, int) or aid not in ACTION_NAMES:
        raise ValueError(f"not an action: {text!r} (the id must be 0 RESET, 1-5, 6 click or 7)")
    if aid != 6:
        return Action(aid)
    x, y = values.get("x"), values.get("y")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y)):
        raise ValueError(f"the click {text!r} needs its screen pixel: Action(6, x=12, y=40)")
    return Action(6, x=int(x), y=int(y))


def _call_node(text: str) -> ast.Call | None:
    """The call `text` is when it is Action(...) (or module.Action(...)), else None."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "Action":
            return node
    return None


def parse_move(item: Any) -> Action:
    """An action as the model gives it to commit_moves: the fixed block's Action as python prints it
    ("Action(4)", "Action(6, x=12, y=40)", "Action(0)" for RESET; the dataclass form "Action(id=6, x=12,
    y=40, cell=(1, 2))" too, its cell ignored) or an Action object, a label ("UP", "RESET", "ACTION3",
    "MOUSE(row=46, col=12)"), an action id, a click as {"click": [x, y]}, {"x": x, "y": y},
    {"action": "MOUSE", "row": r, "col": c}, (6, x, y) or "click 12 46" / "click(12, 46)" (x then y);
    {"action": "UP"} or {"id": 1} for the others. Raises ValueError with the accepted forms otherwise."""
    if isinstance(item, Action):
        return item
    if isinstance(item, bool):
        raise ValueError(f"not an action: {item!r}")
    if isinstance(item, int):
        if item in ACTION_NAMES and item != 6:
            return Action(item)
        raise ValueError(f"action id {item} needs a click position: give Action(6, x=x, y=y)" if item == 6 else f"unknown action id {item}")
    if not isinstance(item, (tuple, list, dict, str)) and isinstance(getattr(item, "id", None), int):  # the fixed block's Action
        return parse_move(action_code(item))
    if isinstance(item, (tuple, list)):
        if len(item) == 3 and int(item[0]) == 6:
            return Action(6, x=int(item[1]), y=int(item[2]))
        if len(item) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in item):
            return Action(6, x=int(item[0]), y=int(item[1]))
        raise ValueError(f"not an action: {item!r} (a click is Action(6, x=x, y=y))")
    if isinstance(item, dict):
        if "click" in item:
            x, y = item["click"]
            return Action(6, x=int(x), y=int(y))
        if "row" in item and "col" in item:
            return Action(6, x=int(item["col"]), y=int(item["row"]))
        if "x" in item and "y" in item and item.get("action", item.get("id", "MOUSE")) in ("MOUSE", "ACTION6", 6, "click"):
            return Action(6, x=int(item["x"]), y=int(item["y"]))
        if "action" in item:
            return parse_move(item["action"])
        if "id" in item:
            return parse_move(int(item["id"]))
        raise ValueError(f"not an action: {item!r}")
    if isinstance(item, str):
        text = item.strip()
        call = _call_node(text)
        if call is not None:
            return _action_from_call(call, text)
        match = re.fullmatch(r"(?:click|mouse)\s*\(?\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*\)?", text, re.IGNORECASE)
        if match:
            return Action(6, x=int(match.group(1)), y=int(match.group(2)))
        try:
            return parse_action_label(text)
        except ValueError:
            pass
        words = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "INTERACT": 5, "UNDO": 7, "RESET": 0}
        if text.upper() in words:
            return Action(words[text.upper()])
        raise ValueError(
            f"unrecognised action {item!r}: give an Action as python prints it, e.g. \"Action(4)\", \"Action(6, x=12, y=40)\" "
            "(x the column, y the row, 0-63) or \"Action(0)\" for RESET, or a label: UP, DOWN, LEFT, RIGHT, SPACE, UNDO, RESET"
        )
    raise ValueError(f"not an action: {item!r}")


def parse_moves(value: Any) -> list[Action]:
    """The moves of commit_moves' `actions`: a list of moves (each one as parse_move takes it), or one string
    holding a printed list of them, as python prints a list of Actions: "[Action(4), Action(6, x=12, y=40)]"
    (labels and dicts may be in it too). Raises ValueError otherwise."""
    if isinstance(value, (list, tuple)):
        return [parse_move(item) for item in value]
    if not isinstance(value, str):
        raise ValueError(f"not a list of actions: {value!r}")
    text = value.strip()
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        node = None
    if not isinstance(node, (ast.List, ast.Tuple)):
        return [parse_move(text)]
    moves = []
    for element in node.elts:
        source = ast.get_source_segment(text, element) or ""
        try:
            item = source if isinstance(element, (ast.Call, ast.Name)) else ast.literal_eval(element)  # (a bare name: UP)
        except ValueError:
            raise ValueError(f"not an action: {source!r}") from None
        moves.append(parse_move(item))
    return moves


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
