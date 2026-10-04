"""Test a candidate engine against a trace and report where it diverges.

The candidate runs in a separate sandboxed process (``candidate_runner``) that
gets only the actions. This module compares what it returned with the trace,
step by step. A step passes when everything the real engine returned matches:
the number of frames, every frame's pixels, the game state, levels completed,
win levels and available actions.

Two modes:

- full replay: a fresh engine plays every action from step 0;
- one level (``from_level=L``): the engine starts with ``set_level(L)`` and
  plays the actions after the step that entered level L, so a later level can
  be worked on before the earlier ones pass.
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

from engine_re.guard import sandbox_env
from engine_re.trace import Step, Trace

HEX = "0123456789abcdef"
FIELDS = ("state", "levels_completed", "win_levels", "available_actions")


@dataclass
class StepCheck:
    index: int
    ok: bool
    final_ok: bool
    problems: list[str] = field(default_factory=list)


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

    @property
    def passed(self) -> bool:
        return self.exact == self.total and self.error is None

    def summary(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k not in ("checks", "text")}
        out["passed"] = self.passed
        return out


# --- Running the candidate ----------------------------------------------------


def run_candidate(
    engine_path: Path,
    actions: list[dict[str, Any]],
    *,
    start_level: int | None = None,
    step_timeout: float = 5.0,
    total_timeout: float = 900.0,
    scratch_root: Path | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run the engine on ``actions`` in a sandboxed process.

    Returns the runner's result (per-step state fields, error) and the frames
    of each completed step."""
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


def check_step(step: Step, got: dict[str, Any] | None, got_frames: np.ndarray | None) -> StepCheck:
    if got is None or got_frames is None:
        return StepCheck(step.index, False, False, ["not run"])
    problems = []
    if len(got_frames) != step.n_frames:
        problems.append("frame count")
    final_ok = step.n_frames == 0 and len(got_frames) == 0
    if step.n_frames and len(got_frames):
        final_ok = frame_diff(step.frames[-1], got_frames[-1]) is None
        if not final_ok:
            problems.append("final frame")
        n = min(step.n_frames, len(got_frames))
        if any(frame_diff(step.frames[k], got_frames[k]) is not None for k in range(n - 1)):
            problems.append("animation frames")
    for name in FIELDS:
        if getattr(step, name) != got.get(name):
            problems.append(name)
    return StepCheck(step.index, not problems, final_ok, problems)


def describe_step(
    step: Step, got: dict[str, Any] | None, got_frames: np.ndarray | None, before: int, crashed_here: bool = False
) -> str:
    """Detailed explanation of one failing step."""
    lines = [f"--- Step {step.index}: {step.action}   (levels_completed before the step: {before})"]
    if got is None or got_frames is None:
        lines.append("    your engine raised an error on this step (traceback above)" if crashed_here else "    not run (the engine stopped earlier)")
        return "\n".join(lines)
    lines.append(
        f"    expected: {step.n_frames} frame(s), state={step.state}, levels_completed={step.levels_completed}, "
        f"win_levels={step.win_levels}, available_actions={step.available_actions}"
    )
    lines.append(
        f"    got:      {len(got_frames)} frame(s), state={got.get('state')}, levels_completed={got.get('levels_completed')}, "
        f"win_levels={got.get('win_levels')}, available_actions={got.get('available_actions')}"
    )
    if step.n_frames and len(got_frames):
        d = frame_diff(step.frames[-1], got_frames[-1])
        if d is None:
            lines.append("    final frame: matches")
        else:
            lines.append(f"    final frame: {_describe_diff(d)}")
            if d["count"] > 0:
                lines.append(crop_panels(step.frames[-1], got_frames[-1], d["bbox"]))
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
    return "\n".join(lines)


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


def replay_test(
    engine_path: Path,
    trace: Trace,
    *,
    from_level: int | None = None,
    details: int = 2,
    scratch_root: Path | None = None,
) -> TestReport:
    starts = trace.level_starts()
    if from_level is not None and from_level != 0:
        if from_level >= trace[0].win_levels:
            raise ValueError(f"level {from_level} is past the last level (win_levels={trace[0].win_levels})")
        if from_level not in starts:
            known = ", ".join(str(k) for k in sorted(starts))
            raise ValueError(f"the trace never reaches level {from_level}; levels it reaches: {known}")
        entry = starts[from_level]
        steps = trace.steps[entry + 1 :]
        mode = f"level {from_level} only: your engine starts with set_level({from_level}) and plays steps {entry + 1}-{len(trace) - 1}"
        start_level: int | None = from_level
    else:
        entry = None
        steps = trace.steps
        mode = f"full replay: a fresh engine plays steps 0-{len(trace) - 1}"
        start_level = None
    actions = [s.action.to_json() for s in steps]
    result, got_frames = run_candidate(engine_path, actions, start_level=start_level, scratch_root=scratch_root)
    got_steps = result.get("steps", [])

    checks = []
    for k, step in enumerate(steps):
        got = got_steps[k] if k < len(got_steps) else None
        frames = got_frames[k] if k < len(got_frames) else None
        checks.append(check_step(step, got, frames))
    exact = sum(c.ok for c in checks)
    final = sum(c.final_ok for c in checks)
    failing = [c.index for c in checks if not c.ok]
    first_fail = failing[0] if failing else None

    start_frame_diff = None
    if entry is not None and result.get("start_frame") is not None and trace[entry].last is not None:
        d = frame_diff(trace[entry].last, np.asarray(result["start_frame"], np.int8))
        start_frame_diff = 0 if d is None else d["count"]

    lines = [f"TEST RESULT ({mode})"]
    lines.append(f"  {exact}/{len(steps)} steps match exactly; {final}/{len(steps)} final frames match.")
    if first_fail is None and result.get("error") is None:
        lines.append("  ALL STEPS MATCH.")
    elif first_fail is not None:
        ok_prefix = first_fail - steps[0].index
        lines.append(f"  First mismatch at step {first_fail} (the {ok_prefix} step(s) before it match).")
    by_level: dict[int, list[StepCheck]] = {}
    before_level = {}
    prev_level = steps[0].levels_completed if entry is None else trace[entry].levels_completed
    for step, check in zip(steps, checks):
        before_level[step.index] = prev_level
        by_level.setdefault(prev_level, []).append(check)
        prev_level = step.levels_completed
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
                f"  Level start frame (step {entry}'s final frame vs your engine's render right after set_level({from_level})): {verdict}."
            )
    if result.get("error"):
        where = result.get("error_step")
        at = f"step {steps[where].index}" if isinstance(where, int) and where < len(steps) else "start-up"
        lines.append(f"\n  YOUR ENGINE RAISED AN ERROR at {at}:\n" + "\n".join("    " + l for l in result["error"].splitlines()))
    if result.get("stdout"):
        lines.append("\n  Engine stdout (tail):\n" + "\n".join("    " + l for l in result["stdout"].splitlines()[-15:]))
    if failing:
        problem_counts = Counter(p for c in checks for p in c.problems)
        lines.append("  Mismatch kinds over all steps: " + ", ".join(f"{k} x{n}" for k, n in problem_counts.most_common()))
        index_of = {s.index: k for k, s in enumerate(steps)}
        for idx in failing[:details]:
            k = index_of[idx]
            got = got_steps[k] if k < len(got_steps) else None
            frames = got_frames[k] if k < len(got_frames) else None
            crashed_here = result.get("error") is not None and result.get("error_step") == k
            lines.append(describe_step(steps[k], got, frames, before_level[idx], crashed_here))
        lines.append(f"  All mismatching steps: {_ranges(failing)}")

    return TestReport(
        mode=mode,
        first_step=steps[0].index,
        total=len(steps),
        exact=exact,
        final_frame=final,
        first_fail=first_fail,
        error=result.get("error"),
        error_step=result.get("error_step"),
        start_frame_diff=start_frame_diff,
        seconds=float(result.get("seconds", 0.0)),
        checks=checks,
        text="\n".join(lines),
    )
