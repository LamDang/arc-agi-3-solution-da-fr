"""Continue an interrupted game run where it stopped (run.py --resume-mid-game).

A run killed mid-game leaves, per unfinished game run:

- ``artifacts/<stem>_tool_runtime_state.json``, rewritten after every action:
  every action taken, with the board after it. The most complete record.
- ``artifacts/<stem>_events.jsonl``, the viewer's event log: the same actions
  as display strings (``MOUSE(row=4, col=7)``). The fallback.
- ``<stem>_requests.jsonl`` (``.xz`` once compressed): every request sent to
  the model and every reply, with its token usage.
- its ``benchmark.json`` entry, saved every 10 minutes, so it can lag.

The game is deterministic, so replaying the recorded actions into a fresh
engine reproduces every saved frame; the replay checks each one and reports
the first that differs. The conversation is the last request sent to the model.

Everything here is a pure function of those files, so run.py can check a
candidate before the run starts and the solver can rebuild the same thing.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import arcengine

from inference.agent.action_names import to_model_action
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, load_runtime_state
from inference.utils.run_artifacts import existing_log, open_log

# A gap between the last saved action and the newest file this long or longer
# says the file times are not the run's (a copy, a checkout, a restore from
# DVC), so the elapsed time falls back to the last saved action.
MAX_TRUSTED_GAP_SECONDS = 3600.0

_MOUSE_DISPLAY_RE = re.compile(r"MOUSE\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)")


class MidGameResumeError(ValueError):
    """The interrupted game run cannot be continued; replay it from level 1."""


def grid_digest(grid: Iterable[Iterable[Any]]) -> str:
    """A short fingerprint of a board, the same for a list or tuple grid."""
    rows = [[int(cell) for cell in row] for row in grid]
    return hashlib.sha1(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def state_grid(state: Any) -> list[list[int]]:
    """The visible board of a taaf GameState, as the solver records it."""
    if state is None:
        return []
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return [[int(cell) for cell in row] for row in rows]


def engine_action(step: dict[str, Any]) -> arcengine.ActionInput:
    """The ActionInput a recorded step was issued as.

    ACTION6 is stored by the solver as ``{row, col}``; the engine takes
    ``x = col, y = row``. ``x``/``y`` already in engine form pass through.
    """
    action_id = arcengine.GameAction.from_name(str(step["id"]))
    data = dict(step.get("data") or {})
    if action_id == arcengine.GameAction.ACTION6:
        if "x" in data and "y" in data:
            data = {"x": int(data["x"]), "y": int(data["y"])}
        else:
            data = {"x": int(data.get("col", 0)), "y": int(data.get("row", 0))}
    else:
        data = {}
    return arcengine.ActionInput(id=action_id, data=data)


def _engine_data(action_name: str, data: dict[str, Any] | None) -> dict[str, int]:
    if action_name != "ACTION6":
        return {}
    data = dict(data or {})
    if "x" in data and "y" in data:
        return {"x": int(data["x"]), "y": int(data["y"])}
    return {"x": int(data.get("col", 0)), "y": int(data.get("row", 0))}


def _display(action_name: str, data: dict[str, int]) -> str:
    if action_name == "ACTION6":
        return f"MOUSE(row={data.get('y', 0)}, col={data.get('x', 0)})"
    return to_model_action(action_name)


def runtime_state_path(old_dir: Path, stem: str) -> Path:
    return Path(old_dir) / "artifacts" / f"{stem}_{RUNTIME_STATE_FILENAME}"


def events_path(old_dir: Path, stem: str) -> Path:
    return Path(old_dir) / "artifacts" / f"{stem}_events.jsonl"


def request_log_path(old_dir: Path, stem: str) -> Path | None:
    """The game run's request log, plain or compressed, or None."""
    return existing_log(Path(old_dir) / f"{stem}_requests.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete record; a torn last line (the process died mid-write) is skipped."""
    with open_log(path) as handle:
        lines = handle.read().split("\n")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if not any(rest.strip() for rest in lines[index + 1 :]):
                break
            raise MidGameResumeError(f"{path}: line {index + 1} is not JSON") from None
        if isinstance(record, dict):
            records.append(record)
    return records


def copy_log_for_continuation(source: Path, destination: Path) -> Path:
    """Copy a log the continued game will append to: plain, complete lines only.

    An ``.xz`` source is decompressed (the solver compresses the plain log at
    game end, which would otherwise replace the old copy). A torn last line
    is cut, or the next line appended would be glued to it. Other files are
    copied as they are.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name.endswith(".xz"):
        with lzma.open(source, "rb") as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
    else:
        shutil.copy2(source, destination)
    if not destination.name.endswith(".jsonl"):
        return destination
    with destination.open("r+b") as handle:
        size = handle.seek(0, 2)
        end = size
        while end > 0:
            start = max(0, end - (1 << 16))
            handle.seek(start)
            chunk = handle.read(end - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                end = start + newline + 1
                break
            end = start
        if end != size:
            handle.truncate(end)
    return destination


def recorded_actions(old_dir: Path, stem: str) -> dict[str, Any]:
    """The actions a game run took, with the board and levels after each.

    Returns ``{"source", "initial_grid_sha", "steps"}``. Each step holds the
    action (``id``, engine ``data``, ``display``, ``automatic``), the expected
    ``grid_sha`` and ``levels_completed`` after it, and the solver's per-action
    ``result`` record when the runtime state has one. Automatic RESETs (after a
    game over, or the warmup) are steps like any other: the solver issued them
    as a plain RESET, and so does the replay.
    """
    state_path = runtime_state_path(old_dir, stem)
    if state_path.is_file():
        try:
            _current, entries = load_runtime_state(state_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise MidGameResumeError(f"{state_path}: unreadable ({exc})") from exc
        if not entries:
            raise MidGameResumeError(f"{state_path}: no history")
        steps: list[dict[str, Any]] = []
        for number, entry in enumerate(entries[1:], start=1):
            result = dict(entry.result or {})
            name = str(result.get("action_name") or "").strip()
            if not name:
                raise MidGameResumeError(f"{state_path}: action {number} has no action_name")
            data = _engine_data(name, result.get("action_data"))
            steps.append(
                {
                    "id": name,
                    "data": data,
                    "display": str(result.get("action_display") or entry.action or _display(name, data)),
                    "automatic": bool(result.get("automatic")),
                    "grid_sha": grid_digest(entry.frame.grid),
                    "levels_completed": int(result.get("score") or 0),
                    "result": result,
                }
            )
        return {
            "source": "runtime_state",
            "initial_grid_sha": grid_digest(entries[0].frame.grid),
            "steps": steps,
        }

    path = events_path(old_dir, stem)
    if not path.is_file():
        raise MidGameResumeError(f"no {state_path.name} or {path.name} in {Path(old_dir) / 'artifacts'}")
    events = _read_jsonl(path)
    initial = next((e for e in events if e.get("type") == "initial"), None)
    if initial is None or not isinstance(initial.get("board"), list):
        raise MidGameResumeError(f"{path}: no initial board")
    steps = []
    for event in events:
        if event.get("type") != "action":
            continue
        name = str(event.get("action_name") or "").strip()
        display = str(event.get("action_display") or "")
        if name == "ACTION6":
            match = _MOUSE_DISPLAY_RE.search(display)
            if match is None:
                raise MidGameResumeError(f"{path}: cannot read the click in {display!r}")
            data = {"x": int(match.group(2)), "y": int(match.group(1))}
        elif name:
            data = {}
        else:
            raise MidGameResumeError(f"{path}: an action event has no action_name")
        steps.append(
            {
                "id": name,
                "data": data,
                "display": display or _display(name, data),
                "automatic": False,
                "grid_sha": grid_digest(event.get("board") or []),
                "levels_completed": int(event.get("score") or 0),
                "result": {},
            }
        )
    return {"source": "events", "initial_grid_sha": grid_digest(initial["board"]), "steps": steps}


def check_against_benchmark(steps: list[dict[str, Any]], prior_run: dict[str, Any]) -> None:
    """The benchmark.json history must be a prefix of the recorded actions."""
    history = prior_run.get("history") or []
    if len(history) > len(steps):
        raise MidGameResumeError(
            f"benchmark.json has {len(history)} actions, the recorded history only {len(steps)}"
        )
    for index, record in enumerate(history):
        action = record.get("action") or {}
        name = str(action.get("id") or "")
        data = _engine_data(name, action.get("data"))
        step = steps[index]
        if (name, data) != (step["id"], step["data"]):
            raise MidGameResumeError(
                f"action {index + 1}: benchmark.json has {name} {data}, "
                f"the recorded history {step['id']} {step['data']}"
            )


def restored_conversation(requests_log: Path) -> dict[str, Any]:
    """The conversation to continue, from the game run's request log.

    The last ``event=request`` line that is not a note-compaction request holds
    everything the model saw last. Dropped: ``messages[0]`` (the system prompt,
    rebuilt by the agent) and a trailing user message - the opener of that
    turn, superseded by the opener the resumed turn builds, or a nudge. A
    mid-turn tail ending in tool results is kept as it is.

    Token counts are summed over every response line, compaction notes
    included, the way the agent accumulates them: that sum is the analyzer's
    generated_tokens at the moment of the kill.
    """
    records = _read_jsonl(requests_log)
    last_request: dict[str, Any] | None = None
    generated = 0
    total = 0
    assistant_texts: list[str] = []
    tokens_by_action: Counter[int] = Counter()
    for record in records:
        event = record.get("event")
        if event == "request" and record.get("kind") != "note_compaction":
            if isinstance(record.get("messages"), list):
                last_request = record
        elif event == "response":
            usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
            completion = _usage_int(usage, ("completion_tokens", "output_tokens", "generated_tokens"))
            generated += completion
            total_value = usage.get("total_tokens")
            if total_value is not None:
                total += _as_int(total_value)
            else:
                total += sum(
                    _as_int(usage.get(key))
                    for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
                )
            try:
                tokens_by_action[int(record.get("action"))] += completion
            except (TypeError, ValueError):
                pass
            reply = record.get("reply") if isinstance(record.get("reply"), dict) else {}
            if record.get("kind") != "note_compaction":
                content = reply.get("content")
                if isinstance(content, str) and content.strip():
                    assistant_texts.append(content)
    if last_request is None:
        raise MidGameResumeError(f"{requests_log}: no request to continue from")
    messages = [dict(message) for message in last_request["messages"][1:] if isinstance(message, dict)]
    while messages and str(messages[-1].get("role", "")).strip() == "user":
        messages.pop()
    return {
        "messages": messages,
        "generated_tokens": generated,
        "total_tokens": total,
        "analysis_step": _as_int(last_request.get("analysis_step")),
        "request_action": _as_int(last_request.get("action")),
        "request_index_within_turn": _as_int(last_request.get("request_index_within_turn")),
        "assistant_texts": assistant_texts,
        "tokens_by_action": dict(tokens_by_action),
    }


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _usage_int(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        try:
            return max(0, int(usage.get(key)))
        except (TypeError, ValueError):
            continue
    return 0


def _file_elapsed(path: Path, started_at: datetime) -> float | None:
    try:
        return (datetime.fromtimestamp(path.stat().st_mtime) - started_at).total_seconds()
    except (OSError, ValueError, OverflowError):
        return None


def prior_elapsed_seconds(
    *,
    last_wallclock: float,
    started_at: datetime | None,
    files: Iterable[Path],
    max_gap_seconds: float = MAX_TRUSTED_GAP_SECONDS,
) -> float:
    """How long the game run had been playing when it was killed.

    The newest of ``files`` (the events log is appended to until the end)
    against the run's ``started_at``; at least the last saved action's
    wallclock. A file time before that or more than ``max_gap_seconds`` after
    it is not this run's and is ignored.
    """
    best = max(0.0, float(last_wallclock or 0.0))
    if started_at is None:
        return best
    candidates = [e for e in (_file_elapsed(Path(p), started_at) for p in files) if e is not None]
    trusted = [e for e in candidates if best <= e <= best + max_gap_seconds]
    return max([best, *trusted])


def replay(
    game: Any,
    steps: list[dict[str, Any]],
    *,
    initial_grid_sha: str | None = None,
    execute: Callable[[arcengine.ActionInput, dict[str, Any]], Any] | None = None,
    on_step: Callable[[int, dict[str, Any]], None] | None = None,
) -> int | None:
    """Replay ``steps`` into a started ``game``; the first divergence, or None.

    A divergence at index ``i`` means ``steps[:i]`` reproduced the recorded
    boards and levels and ``steps[i]`` did not (or the engine refused it); a
    mismatching initial board is a divergence at 0 before anything runs.
    ``execute`` runs one action (default ``game.execute_action``); ``on_step``
    is called after each executed one, matching or not.
    """
    if initial_grid_sha is not None and grid_digest(state_grid(game.current_state)) != initial_grid_sha:
        return 0
    run = execute or (lambda action, _step: game.execute_action(action))
    for index, step in enumerate(steps):
        try:
            run(engine_action(step), step)
        except Exception:
            return index
        if on_step is not None:
            on_step(index, step)
        state = game.current_state
        if (
            grid_digest(state_grid(state)) != step["grid_sha"]
            or int(state.levels_completed) != int(step["levels_completed"])
        ):
            return index
    return None


def build_plan(old_dir: Path, stem: str, prior_run: dict[str, Any]) -> dict[str, Any]:
    """Everything the solver needs to continue ``stem``, as a plain dict.

    ``prior_run`` is the game run's benchmark.json entry. Per step it adds the
    ``generated_tokens`` and ``wallclock_seconds`` its action record carries:
    benchmark.json's where it has them; beyond that, the tokens of the model's
    responses in the turn that issued the action (the request log's ``action``
    is the 1-based number of the next action) and a wallclock spread evenly up
    to the last write of the runtime state. Raises MidGameResumeError.
    """
    old_dir = Path(old_dir)
    recorded = recorded_actions(old_dir, stem)
    steps = recorded["steps"]
    check_against_benchmark(steps, prior_run)
    log_path = request_log_path(old_dir, stem)
    conversation = restored_conversation(log_path) if log_path is not None else None

    started_raw = prior_run.get("started_at")
    started_at = datetime.fromisoformat(started_raw) if started_raw else None
    history = prior_run.get("history") or []
    last_wallclock = float(history[-1]["wallclock_seconds"]) if history else 0.0
    prior_elapsed = prior_elapsed_seconds(
        last_wallclock=last_wallclock,
        started_at=started_at,
        files=[
            events_path(old_dir, stem),
            runtime_state_path(old_dir, stem),
            *([log_path] if log_path is not None else []),
        ],
    )
    last_action_elapsed = prior_elapsed_seconds(
        last_wallclock=last_wallclock,
        started_at=started_at,
        files=[runtime_state_path(old_dir, stem)] if recorded["source"] == "runtime_state" else [],
    )
    last_action_elapsed = min(last_action_elapsed, prior_elapsed)
    tokens_by_action = (conversation or {}).get("tokens_by_action") or {}
    lagging = len(steps) - len(history)
    for index, step in enumerate(steps):
        if index < len(history):
            step["generated_tokens"] = _as_int(history[index].get("generated_tokens"))
            step["wallclock_seconds"] = float(history[index].get("wallclock_seconds") or 0.0)
            continue
        position = index - len(history) + 1
        step["generated_tokens"] = 0 if step["automatic"] else _as_int(tokens_by_action.get(index + 1))
        step["wallclock_seconds"] = last_wallclock + (last_action_elapsed - last_wallclock) * position / lagging
    recorded_tokens = sum(step["generated_tokens"] for step in steps)
    plan: dict[str, Any] = {
        "source_dir": str(old_dir.resolve()),
        "stem": stem,
        "source": recorded["source"],
        "initial_grid_sha": recorded["initial_grid_sha"],
        "steps": steps,
        "prior_elapsed_seconds": prior_elapsed,
        "recorded_tokens": recorded_tokens,
        # what the game had generated when it was killed: the request log is
        # complete, benchmark.json and the action records may not be
        "prior_generated_tokens": max(
            recorded_tokens, (conversation or {}).get("generated_tokens", 0)
        ),
        "conversation": None,
    }
    if conversation is not None:
        plan["conversation"] = {
            "analysis_step": conversation["analysis_step"],
            "generated_tokens": conversation["generated_tokens"],
            "total_tokens": conversation["total_tokens"],
            "messages": len(conversation["messages"]),
        }
    return plan
