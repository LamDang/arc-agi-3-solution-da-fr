"""Test a candidate engine against a trace and report where it diverges.

The candidate runs in a separate sandboxed process (``candidate_runner``) that
gets only the actions. This module compares what it returned with the trace,
step by step. Two matching rules:

- ``final`` (default): a step passes when its last frame and the state fields
  (state, levels completed, win levels, available actions) match. Animation
  frames and the frame count are not compared.
- ``all``: every frame must match too, so the frame count and each animation
  frame count as well.

Three replay scopes:

- full replay (default): a fresh engine plays every action from step 0;
- one level (``level=L``): the engine starts at level L's start (``set_level(L)``,
  i.e. ``make_level(L)``), its drawing is compared with the recorded start of
  the level, and it plays that level's recorded steps only (up to and
  including the step that completes it). ``level=0`` is the full replay cut
  after level 0;
- ``from_level=L`` (kept for the command line and older callers): like
  ``level=L`` but playing every step to the end of the recording.

Two report styles, from the same full comparison (the counts in the summary
are always those of the whole scope):

- ``stop_on_fail=True``: the report stops at the first failing test. A failing
  contract test is reported and the replay is not; otherwise the report says
  which steps match and explains the first failing step (or, for one level,
  the level's start frame) and nothing after it;
- ``stop_on_fail=False``: everything: per-level counts, the first ``details``
  failing steps explained, and the list of all failing steps.

A failing step is explained with the differing regions of its final frame,
numbered, with the colours that differ and the engine's sprites that draw
there before and after the step (``engine_re.diff_report``; a second, short
candidate run collects them), the end of what the engine printed during the
step (the runner captures prints per step), and the ``try_step`` command that
reproduces it in the analysis kernel. With ``images=True`` the report also
carries a picture of both final frames with the regions boxed
(``TestReport.images``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine_re import diff_report
from engine_re.game_api import describe_contract, last_lines
from engine_re.guard import sandbox_env
from engine_re.trace import Step, Trace

HEX = "0123456789abcdef"
FIELDS = ("state", "levels_completed", "win_levels", "available_actions")
MATCH_MODES = ("final", "all")


@dataclass
class StepCheck:
    index: int
    ok: bool
    final_ok: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class TestImage:
    """A picture for the model: the engine's frame and the original's, regions boxed."""

    step: int  # the trace step whose final frame is compared (for a level start: the step that entered it)
    caption: str
    png: bytes


@dataclass
class TestReport:
    mode: str
    first_step: int
    total: int
    exact: int
    final_frame: int
    first_fail: int | None
    error: str | None
    error_step: int | None
    start_frame_diff: int | None
    seconds: float
    checks: list[StepCheck]
    text: str = ""
    match: str = "final"
    contract_passed: int | None = None  # simple-interface engines only
    contract_total: int | None = None
    level: int | None = None  # level=L: only level L was tested
    stop_on_fail: bool = False
    detail_steps: list[int] = field(default_factory=list)  # steps explained in the text (a level start: its entry step)
    images: list[TestImage] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        contract_ok = self.contract_total is None or self.contract_passed == self.contract_total
        return self.exact == self.total and self.error is None and contract_ok

    def summary(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k not in ("checks", "text", "images")}
        out["passed"] = self.passed
        return out


# --- Running the candidate ----------------------------------------------------


def trace_meta(trace: Trace) -> dict[str, Any]:
    """What a make_level/step engine needs from the recording: win levels, the advertised
    actions and the levels the recording reaches (for the contract tests)."""
    win = trace[0].win_levels
    return {
        "win_levels": win,
        "available_actions": list(trace[0].available_actions),
        "levels": sorted(level for level in trace.level_starts() if level < win),
    }


def run_candidate(
    engine_path: Path,
    actions: list[dict[str, Any]],
    *,
    start_level: int | None = None,
    step_timeout: float = 5.0,
    total_timeout: float = 900.0,
    scratch_root: Path | None = None,
    meta: dict[str, Any] | None = None,
    inspect: list[int | str] | None = None,
    contract: bool = True,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run the engine on ``actions`` in a sandboxed process.

    Returns the runner's result (per-step state fields, error) and the frames
    of each completed step. ``inspect``: action positions (and "start") whose
    states to describe (result["inspect"]); ``contract=False`` skips the
    contract tests."""
    engine_path = Path(engine_path).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="cand_", dir=Path(scratch_root).resolve() if scratch_root else None))
    try:
        (scratch / "actions.json").write_text(json.dumps(actions), encoding="utf-8")
        out_dir = scratch / "out"
        out_dir.mkdir()
        cmd = [
            sys.executable,
            "-m",
            "engine_re.candidate_runner",
            str(engine_path),
            str(scratch / "actions.json"),
            str(out_dir),
            "--step-timeout",
            str(step_timeout),
        ]
        if start_level is not None:
            cmd += ["--start-level", str(start_level)]
        if meta:
            cmd += [
                "--win-levels",
                str(meta["win_levels"]),
                "--available-actions",
                json.dumps(meta["available_actions"]),
                "--levels",
                json.dumps(meta.get("levels", [0])),
            ]
        if inspect:
            cmd += ["--inspect", json.dumps(list(inspect))]
        if not contract:
            cmd.append("--no-contract")
        try:
            proc = subprocess.run(
                cmd, cwd=scratch, env=sandbox_env(str(scratch)), capture_output=True, text=True, timeout=total_timeout
            )
        except subprocess.TimeoutExpired:
            return {"steps": [], "error": f"the engine run exceeded {total_timeout:g}s", "error_step": 0}, []
        result_path = out_dir / "result.json"
        if not result_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            return {"steps": [], "error": f"engine process died (exit {proc.returncode}):\n{tail}", "error_step": 0}, []
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["stdout"] = (proc.stdout or "")[-4000:]
        with np.load(out_dir / "frames.npz") as data:
            frames, offsets = data["frames"], data["offsets"]
        per_step = [frames[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        return result, per_step
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- Comparing ---------------------------------------------------------------


def _hex(row: np.ndarray) -> str:
    return "".join(HEX[v] if 0 <= v < 16 else "?" for v in row)


def frame_diff(expected: np.ndarray, got: np.ndarray) -> dict[str, Any] | None:
    """None when equal, else count, bounding box and colour transitions."""
    if expected.shape != got.shape:
        return {"count": -1, "shape": (expected.shape, got.shape)}
    mask = expected != got
    count = int(mask.sum())
    if not count:
        return None
    rows, cols = np.nonzero(mask)
    transitions = Counter(zip(expected[mask].tolist(), got[mask].tolist()))
    return {
        "count": count,
        "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
        "transitions": transitions.most_common(6),
    }


def crop_panels(expected: np.ndarray, got: np.ndarray, bbox: tuple[int, int, int, int], max_side: int = 24) -> str:
    """Expected and got side by side around ``bbox``, plus a mask of differing pixels."""
    r0, c0, r1, c1 = bbox
    r0, c0 = max(0, r0 - 1), max(0, c0 - 1)
    r1, c1 = min(63, r1 + 1, r0 + max_side - 1), min(63, c1 + 1, c0 + max_side - 1)
    width = c1 - c0 + 1
    lines = [f"    rows {r0}-{r1}, cols {c0}-{c1} (one hex digit per pixel = colour 0-15)"]
    lines.append(f"    {'row':>3}  {'expected':<{width}}  {'got':<{width}}  {'diff (x = differs)':<{width}}")
    for r in range(r0, r1 + 1):
        e, g = expected[r, c0 : c1 + 1], got[r, c0 : c1 + 1]
        mask = "".join("x" if a != b else "." for a, b in zip(e, g))
        lines.append(f"    {r:>3}  {_hex(e)}  {_hex(g)}  {mask}")
    return "\n".join(lines)


def _describe_diff(d: dict[str, Any]) -> str:
    if d["count"] == -1:
        return f"shape {d['shape'][1]} instead of {d['shape'][0]}"
    r0, c0, r1, c1 = d["bbox"]
    trans = ", ".join(f"{a}->{b} x{n}" for (a, b), n in d["transitions"])
    return f"{d['count']} pixels differ in rows {r0}-{r1}, cols {c0}-{c1} (expected->got colour: {trans})"


def check_step(step: Step, got: dict[str, Any] | None, got_frames: np.ndarray | None, match: str = "final") -> StepCheck:
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}")
    if got is None or got_frames is None:
        return StepCheck(step.index, False, False, ["not run"])
    problems = []
    if match == "all" and len(got_frames) != step.n_frames:
        problems.append("frame count")
    final_ok = step.n_frames == 0 and len(got_frames) == 0
    if step.n_frames and len(got_frames):
        final_ok = frame_diff(step.frames[-1], got_frames[-1]) is None
    if not final_ok:
        problems.append("final frame")
    if match == "all" and step.n_frames and len(got_frames):
        n = min(step.n_frames, len(got_frames))
        if any(frame_diff(step.frames[k], got_frames[k]) is not None for k in range(n - 1)):
            problems.append("animation frames")
    for name in FIELDS:
        if getattr(step, name) != got.get(name):
            problems.append(name)
    return StepCheck(step.index, not problems, final_ok, problems)


def describe_step(
    step: Step,
    got: dict[str, Any] | None,
    got_frames: np.ndarray | None,
    before: int,
    crashed_here: bool = False,
    match: str = "final",
    *,
    states: dict[str, Any] | None = None,
    crops: bool = True,
    images: bool = False,
    printed: str | None = None,
    show_vars: bool = True,
) -> tuple[str, list[diff_report.Region]]:
    """Detailed explanation of one failing step, and the numbered regions of its final frame.

    ``before`` is the level the step was played in (levels completed before it); ``states`` the
    candidate's state summaries {"before": ..., "after": ...} if available; ``printed`` what the
    engine printed during the step."""
    lines = [f"--- Step {step.index}: {step.action}   (played in level {before})"]
    if got is None or got_frames is None:
        lines.append("    your engine raised an error on this step (traceback above)" if crashed_here else "    not run (your engine stopped earlier)")
        lines += printed_lines(printed)
        return "\n".join(lines), []
    differ = [f"{name} expected {getattr(step, name)}, got {got.get(name)}" for name in FIELDS if getattr(step, name) != got.get(name)]
    if differ:
        lines.append("    state fields differ: " + "; ".join(differ))
    else:
        lines.append(f"    state fields match ({step.state}, levels_completed {step.levels_completed})")
    if match == "all":
        lines.append(f"    frames: expected {step.n_frames}, got {len(got_frames)}")
    elif step.n_frames > 1:
        lines.append(f"    (the original animated this action over {step.n_frames} frames; only the last is compared)")
    regions: list[diff_report.Region] = []
    if step.n_frames and len(got_frames):
        states = states or {}
        frame_lines, regions = diff_report.describe_frames(
            step.frames[-1], got_frames[-1], states.get("before"), states.get("after"), crops=crops, images=images, show_vars=show_vars
        )
        lines += frame_lines
    elif step.n_frames != len(got_frames):
        lines.append(f"    final frame: expected {'a frame' if step.n_frames else 'no frame'}, got {len(got_frames)} frame(s)")
    if match == "all" and step.n_frames and len(got_frames):
        n = min(step.n_frames, len(got_frames))
        anim = []
        for k in range(n - 1):
            dk = frame_diff(step.frames[k], got_frames[k])
            if dk is not None:
                anim.append(f"frame {k}: {_describe_diff(dk)}")
        if anim:
            lines.append("    animation frames (compared by position from the first frame):")
            lines.extend("      " + a for a in anim[:4])
            if len(anim) > 4:
                lines.append(f"      ... and {len(anim) - 4} more differing animation frames")
        elif step.n_frames > 1 and n > 1:
            lines.append(f"    animation frames 0-{n - 2}: match")
    lines += printed_lines(printed)
    return "\n".join(lines), regions


def printed_lines(printed: str | None, what: str = "this step") -> list[str]:
    """The end of what the engine printed, indented for a report."""
    if not printed or not printed.strip():
        return []
    kept, total = last_lines(printed)
    head = f"    your engine printed during {what}" + (f" (the last 20 of {total} lines):" if total > 20 else ":")
    return [head] + ["      " + line for line in kept.splitlines()]


def _ranges(indices: list[int]) -> str:
    if not indices:
        return "none"
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:] + [None]:  # type: ignore[list-item]
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(parts)


def levels_before(trace: Trace) -> list[int]:
    """For each step, the level it is played in: levels completed before it."""
    out, prev = [], 0
    for step in trace.steps:
        out.append(prev)
        prev = step.levels_completed
    return out


def level_span(trace: Trace, level: int) -> tuple[int, int, int]:
    """(entry, first, stop) for level L: step `entry`'s final frame first shows level L; the level's
    recorded steps are first..stop-1, up to and including the one that completes it. For level 0
    entry = first = 0 (step 0 is the RESET that draws it)."""
    starts = trace.level_starts()
    win = trace[0].win_levels
    if not 0 <= level < win:
        raise ValueError(f"level {level} is not a level of this game (levels 0-{win - 1})")
    if level not in starts:
        known = ", ".join(str(k) for k in sorted(starts) if k < win)
        raise ValueError(f"the recording never reaches level {level}; levels it reaches: {known}")
    entry = starts[level]
    first = 0 if level == 0 else entry + 1
    before = levels_before(trace)
    stop = first
    while stop < len(trace) and before[stop] == level:
        stop += 1
    return entry, first, stop


def repro_lines(step: int, level: int | None, action: str, start: bool = False) -> list[str]:
    """The python command that reproduces a failing step in the analysis kernel."""
    lv = f", level={level}" if level else ""
    if start:
        what = f"makes your level {level} start (make_level({level})) and compares it with step {step}'s final frame"
    elif level:
        what = f"starts your engine at level {level}, replays the level's steps before {step}, applies step {step}'s {action}"
    elif step:
        what = f"replays steps 0-{step - 1} on a fresh engine.py, applies step {step}'s {action}"
    else:
        what = "loads engine.py fresh and applies step 0's RESET (before is None)"
    return [
        "  Reproduce in python:",
        f"    before, after = try_step({step}{lv})",
        f"    # {what}; prints what your engine printed, what changed in your state and this comparison; "
        "returns copies of your State before and after",
    ]


def replay_test(
    engine_path: Path,
    trace: Trace,
    *,
    level: int | None = None,
    from_level: int | None = None,
    stop_on_fail: bool = False,
    details: int = 2,
    scratch_root: Path | None = None,
    match: str = "final",
    images: bool = False,
    crops: bool | None = None,
) -> TestReport:
    """Test an engine against the recording (see the module docstring).

    level: test only level L. from_level: start at level L and play to the end (older scope).
    stop_on_fail: report only up to the first failing test (the counts still cover the scope).
    details: with stop_on_fail=False, how many failing steps to explain.
    images: also draw each explained step's frames (TestReport.images).
    crops: include hex-digit crops of the regions (default: when there are no images)."""
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}")
    if level is not None and from_level is not None:
        raise ValueError("give level or from_level, not both")
    crops = (not images) if crops is None else crops
    n_steps = len(trace)
    entry: int | None = None
    start_level: int | None = None
    if level is not None:
        entry, first, stop = level_span(trace, level)
        steps = trace.steps[first:stop]
        span = f"steps {first}-{stop - 1}" if stop > first else "no steps (the recording ends there)"
        if level == 0:
            mode = f"level 0 only: a fresh engine plays {span}"
            entry = None
        else:
            start_level = level
            mode = f"level {level} only: your engine starts at make_level({level}) and plays the level's {span}"
    elif from_level is not None and from_level != 0:
        if from_level >= trace[0].win_levels:
            raise ValueError(f"level {from_level} is past the last level (win_levels={trace[0].win_levels})")
        starts = trace.level_starts()
        if from_level not in starts:
            known = ", ".join(str(k) for k in sorted(starts))
            raise ValueError(f"the trace never reaches level {from_level}; levels it reaches: {known}")
        entry = starts[from_level]
        steps = trace.steps[entry + 1 :]
        mode = f"level {from_level} onwards: your engine starts with set_level({from_level}) and plays steps {entry + 1}-{n_steps - 1}"
        start_level = from_level
    else:
        steps = trace.steps
        mode = f"full replay: a fresh engine plays steps 0-{n_steps - 1}"
    if stop_on_fail:
        mode += "; the report stops at the first failure"
    meta = trace_meta(trace)
    actions = [s.action.to_json() for s in steps]
    result, got_frames = run_candidate(engine_path, actions, start_level=start_level, scratch_root=scratch_root, meta=meta)
    contract = result.get("contract")
    got_steps = result.get("steps", [])

    checks = []
    for k, step in enumerate(steps):
        got = got_steps[k] if k < len(got_steps) else None
        frames = got_frames[k] if k < len(got_frames) else None
        checks.append(check_step(step, got, frames, match))
    exact = sum(c.ok for c in checks)
    final = sum(c.final_ok for c in checks)
    failing = [c.index for c in checks if not c.ok]
    first_fail = failing[0] if failing else None
    index_of = {s.index: k for k, s in enumerate(steps)}
    played_in = levels_before(trace)

    start_frame = None
    start_frame_diff = None
    if entry is not None and result.get("start_frame") is not None and trace[entry].last is not None:
        start_frame = np.asarray(result["start_frame"], np.int8)
        d = frame_diff(trace[entry].last, start_frame)
        start_frame_diff = 0 if d is None else d["count"]
    start_bad = bool(start_frame_diff)
    contract_failures = [c for c in contract or [] if not c["ok"]]
    error = result.get("error")
    error_pos = result.get("error_step")
    # An error before any step ran: loading engine.py, or make_level when the test starts at a level.
    startup_error = error is not None and not isinstance(error_pos, int)

    # What to explain in detail: "start" (a level's start frame) and/or step positions in `steps`.
    targets: list[int | str] = []
    if startup_error:
        pass
    elif stop_on_fail:
        if not contract_failures:
            if start_bad:
                targets = ["start"]
            elif first_fail is not None:
                targets = [index_of[first_fail]]
    else:
        ran = [index_of[i] for i in failing if not isinstance(error_pos, int) or index_of[i] <= error_pos]
        targets = ((["start"] if start_bad else []) + ran)[: max(1, details)]

    # A second, short run of the candidate describes its states at those steps (simple interface),
    # and keeps what it printed there whatever it printed before.
    inspected: dict[str, Any] = {}
    second_prints: dict[str, str] = {}

    def printed(key: str) -> str | None:
        return second_prints.get(key) or (result.get("prints") or {}).get(key)

    positions = [t for t in targets if t != "start"]
    if targets and result.get("interface") == "simple" and not (error is not None and error_pos is None):
        limit = max(positions) + 1 if positions else 0
        if isinstance(error_pos, int):
            limit = min(limit, error_pos + 1)
        second, second_frames = run_candidate(
            engine_path, actions[:limit], start_level=start_level, scratch_root=scratch_root, meta=meta,
            inspect=[t for t in targets if t == "start" or t < limit], contract=False,
        )
        inspected = second.get("inspect") or {}
        second_prints = second.get("prints") or {}
        for k in positions:
            if k < len(second_frames) and k < len(got_frames) and len(got_frames[k]) and len(second_frames[k]):
                if not np.array_equal(second_frames[k][-1], got_frames[k][-1]):
                    inspected.setdefault("nondeterministic", []).append(steps[k].index)

    lines = [f"TEST RESULT ({mode})"]
    if contract:
        lines.append(describe_contract(contract))
    acceptance_shown = not (stop_on_fail and contract_failures)
    if not acceptance_shown:
        lines.append(
            "  Acceptance test (replay of the recording): not reported until the contract tests pass. Fix them first, "
            "or call run_tests(stop_on_fail=false) to see the replay anyway."
        )
    elif stop_on_fail:
        head = "  Acceptance test (replay of the recording):" if contract else " "
        scope = f"level {level}'s" if level is not None else "the"
        if startup_error:
            pass  # the error is shown below
        elif start_bad:
            lines.append(f"{head} the level {level} start frame differs ({start_frame_diff} px), explained below; its steps are not checked in this report.")
        elif first_fail is None and error is None:
            what = f"ALL {len(steps)} STEPS OF LEVEL {level} MATCH" if level is not None else "ALL STEPS MATCH"
            extra = " (and its start frame)" if entry is not None and start_frame_diff == 0 else ""
            lines.append(f"{head} {what}{extra}." + ("" if not contract_failures else " The contract tests above must pass too."))
        elif first_fail is not None:
            k = index_of[first_fail]
            ok = "" if not k else f"step {first_fail - 1} matches; " if k == 1 else f"steps {steps[0].index}-{first_fail - 1} match; "
            start_ok = "the start frame matches; " if entry is not None and start_frame_diff == 0 else ""
            lines.append(
                f"{head} {start_ok}{ok}step {first_fail} is the first mismatch. Later steps are not reported "
                f"(run_tests(stop_on_fail=false) lists every failing step of {scope} replay)."
            )
    else:
        head = "  Acceptance test (replay of the recording):"
        if contract:
            lines.append(head)
        if match == "final":
            lines.append(f"  {exact}/{len(steps)} steps match (final frame and state; animation frames are not compared).")
        else:
            lines.append(f"  {exact}/{len(steps)} steps match exactly; {final}/{len(steps)} final frames match.")
        if first_fail is None and error is None:
            lines.append("  ALL STEPS MATCH." if not contract_failures else "  All steps match, but the contract tests above must pass too.")
        elif first_fail is not None:
            ok_prefix = first_fail - steps[0].index
            lines.append(f"  First mismatch at step {first_fail} (the {ok_prefix} step(s) before it match).")
        by_level: dict[int, list[StepCheck]] = {}
        for step, check in zip(steps, checks):
            by_level.setdefault(played_in[step.index], []).append(check)
        if len(by_level) > 1 or entry is None:
            parts = [
                f"level {lvl} (steps {cs[0].index}-{cs[-1].index}): {sum(c.ok for c in cs)}/{len(cs)}"
                for lvl, cs in sorted(by_level.items())
            ]
            lines.append("  Per level (by levels_completed before the step): " + "; ".join(parts))
        if entry is not None:
            if start_frame_diff is None:
                lines.append("  Level start frame: not compared.")
            else:
                verdict = "matches" if start_frame_diff == 0 else f"{start_frame_diff} pixels differ"
                lines.append(
                    f"  Level start frame (step {entry}'s final frame vs your engine's render right after set_level({start_level})): {verdict}."
                )
    if acceptance_shown and error:
        if startup_error:
            what = f"make_level({start_level}) or " if start_level is not None else ""
            lines.append(f"\n  YOUR ENGINE RAISED AN ERROR at start-up ({what}loading engine.py), so no step ran:")
            lines.append("\n".join("    " + line for line in error.splitlines()))
        else:
            at = f"step {steps[error_pos].index}" if error_pos < len(steps) else f"position {error_pos}"
            if not stop_on_fail or first_fail is None or error_pos >= len(steps) or steps[error_pos].index == first_fail:
                lines.append(f"\n  YOUR ENGINE RAISED AN ERROR at {at}:\n" + "\n".join("    " + line for line in error.splitlines()))
    if acceptance_shown and printed("load"):
        lines += [line[2:] for line in printed_lines(printed("load"), "loading engine.py")]
    if result.get("stdout", "").strip():  # written past the capture (to the file descriptor itself)
        tail = result["stdout"].splitlines()[-10:]
        lines.append("  Engine stdout (tail):\n" + "\n".join("    " + line for line in tail))
    if acceptance_shown and failing and not stop_on_fail:
        problem_counts = Counter(p for c in checks for p in c.problems)
        lines.append("  Mismatch kinds over all steps: " + ", ".join(f"{k} x{n}" for k, n in problem_counts.most_common()))

    report_images: list[TestImage] = []
    detail_steps: list[int] = []
    for target in targets if acceptance_shown else []:
        if target == "start":
            assert entry is not None and start_frame is not None
            detail_steps.append(entry)
            lines.append(f"--- Level {level if level is not None else from_level} start: your make_level as drawn vs the original (step {entry}'s final frame)")
            frame_lines, regions = diff_report.describe_frames(
                trace[entry].last, start_frame, None, inspected.get("start"), crops=crops, images=images
            )
            lines += frame_lines + printed_lines(printed("start"), "make_level")
            if images and regions:
                lvl = level if level is not None else from_level
                img = diff_report.comparison_image(
                    start_frame, trace[entry].last, regions,
                    left_title=f"YOUR ENGINE: start of level {lvl}", right_title=f"ORIGINAL GAME: start of level {lvl} (step {entry})",
                )
                report_images.append(TestImage(entry, f"Start of level {lvl}: left your make_level({lvl}) as drawn, right the original (step {entry})", diff_report.png_bytes(img)))
            continue
        k = int(target)
        step = steps[k]
        detail_steps.append(step.index)
        got = got_steps[k] if k < len(got_steps) else None
        frames = got_frames[k] if k < len(got_frames) else None
        crashed_here = error is not None and error_pos == k
        text, regions = describe_step(
            step, got, frames, played_in[step.index], crashed_here, match,
            states=inspected.get(str(k)), crops=crops, images=images, printed=printed(str(k)),
        )
        lines.append(text)
        if step.index in inspected.get("nondeterministic", []):
            lines.append("    (warning: a second run of your engine gave a different frame here; it is not deterministic)")
        if images and regions and frames is not None and len(frames):
            img = diff_report.comparison_image(
                frames[-1], step.last, regions,
                left_title=f"YOUR ENGINE: after step {step.index} ({step.action})", right_title=f"ORIGINAL GAME: after step {step.index}",
            )
            report_images.append(TestImage(step.index, f"Step {step.index} ({step.action}): left your engine's final frame, right the original game's", diff_report.png_bytes(img)))
    if acceptance_shown and failing and not stop_on_fail:
        lines.append(f"  All mismatching steps: {_ranges(failing)}")
    if acceptance_shown and targets and result.get("interface") == "simple":
        target = targets[0]
        lvl = level if level is not None else from_level
        if target == "start":
            lines += repro_lines(entry, lvl, "", start=True)
        else:
            step = steps[int(target)]
            lines += repro_lines(step.index, lvl if start_level is not None else None, str(step.action))
    elif acceptance_shown and error is not None and not targets:
        lines.append("  Reproduce in python: engine()   # loads engine.py in the kernel and shows the error")

    return TestReport(
        mode=mode,
        first_step=steps[0].index if steps else (entry + 1 if entry is not None else 0),
        total=len(steps),
        exact=exact,
        final_frame=final,
        first_fail=first_fail,
        error=error,
        error_step=error_pos,
        start_frame_diff=start_frame_diff,
        seconds=float(result.get("seconds", 0.0)),
        checks=checks,
        text="\n".join(lines),
        match=match,
        contract_passed=sum(c["ok"] for c in contract) if contract else None,
        contract_total=len(contract) if contract else None,
        level=level,
        stop_on_fail=stop_on_fail,
        detail_steps=detail_steps,
        images=report_images,
    )


def main() -> int:
    """python -m engine_re.tester ENGINE TRACE_DIR [--level L | --from-level L] [--stop-on-fail] [--details N]
    [--match final|all] [--images DIR]"""
    import argparse

    parser = argparse.ArgumentParser(description="Test an engine against a recorded trace.")
    parser.add_argument("engine", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--level", type=int, default=None, help="test only this level")
    parser.add_argument("--from-level", type=int, default=None, help="start at this level and play to the end")
    parser.add_argument("--stop-on-fail", action="store_true", help="report only up to the first failure")
    parser.add_argument("--details", type=int, default=2)
    parser.add_argument("--match", choices=MATCH_MODES, default="final")
    parser.add_argument("--images", type=Path, default=None, help="save the comparison images (PNG) in this directory")
    args = parser.parse_args()
    report = replay_test(
        args.engine, Trace.load(args.trace), level=args.level, from_level=args.from_level, stop_on_fail=args.stop_on_fail,
        details=args.details, match=args.match, images=args.images is not None,
    )
    print(report.text)
    if args.images is not None:
        args.images.mkdir(parents=True, exist_ok=True)
        for image in report.images:
            path = args.images / f"step{image.step}.png"
            path.write_bytes(image.png)
            print(f"[image] {path}: {image.caption}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
