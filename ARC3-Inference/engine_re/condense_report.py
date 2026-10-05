"""Compare the agent's compaction (`_compact`) with `condense.condense` on finished stepwise runs.

    uv run --no-sync python -m engine_re.condense_report --out <folder> <name>=<run dir> ... \\
        [--samples <name>:<turn> ...] [--threshold 140000]
    uv run --no-sync python -m engine_re.condense_report --schemes --out <folder> <name>=<run dir> ... \\
        [--threshold 140000] [--keep-turns 10]

--schemes (`compare_schemes`, for transcripts with "message" records) compares the three context schemes
turn by turn: the compaction, the earlier per-turn condenser and the threshold-triggered one the agent runs
now (two variants of when it fires: where the compaction fired, and when its own prompt is over), the one
the run used replayed from its records and checked against them, the others simulated. Per scheme: the
total and maximum estimated prompt tokens (also calibrated to the run's real counts), the turns it fired
at, the turns whose prompt does not start with the previous one, and the share of prompt tokens that
repeat the previous prompt (what a prefix cache can serve). It writes `schemes.md` and `schemes.json`.

Without --schemes, for every run and every turn t it builds the prompt the model would receive under (a) the current
scheme and (b) the new one, measures characters, estimated tokens (4 per character, 1000 per live
image) and live images, and compares (a)'s estimate with the prompt_tokens the provider counted. The
output folder gets `measurements.json` (every turn), `report.md` (tables at turns 10, 25, 50, 75, 100
and the maximum, totals, the safety cap, calibration) and, for each sample, the two full prompts as text
(`<name>_t<turn>_current.txt`, `<name>_t<turn>_new.txt`; images shown as "[image: caption]").

The full conversation is rebuilt from transcript.jsonl in the older format (no "message" records), as
the agent's `_rebuild_legacy` does (the harness's messages as the current prompts write them: every
next-step message lists engine.py as it was then; a kernel-names line and the one-line "same result"
automatic test cannot be recovered from the old records), but across the whole run, resumes included,
and with nothing shortened; images are kept by file name, not loaded. (a) replays what happened live: `_compact` after
every turn whose prompt_tokens exceeded the threshold, images hidden when a newer image message comes,
and at a resume the conversation rebuilt afresh (shortened only if the last turn's prompt was over the
threshold, as the agent of those runs did) followed by the resume note.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine_re import hashline
from engine_re.agent import (
    AUTO_TEST,
    AUTO_TEST_CHARS,
    COMMIT_HINT,
    CONTINUE,
    IMAGE_NOTE,
    IMAGE_PLACEHOLDER,
    NUDGE,
    READ_CHARS_IN_MESSAGES,
    TEST_IMAGE_NOTE,
    EngineAgent,
    ModelConfig,
    _truncate,
    resume_note,
)
from engine_re.condense import CAP_TOKENS, KEEP_TURNS, RESUME_HEAD, Condensed, condense, hide_but_latest, measure
from engine_re.game_api import fixed_block_lines
from engine_re.prompts import advance_message, episode_message, system_prompt
from engine_re.trace import Trace

IMAGE_URL = "image:"  # an image part's url: the PNG's path in the run folder (the bytes are not needed)
# The resume note of runs from before the kernel replay (no "replay" record before the "resumed" one), as it was sent.
LEGACY_RESUME_NOTE = (
    "[harness] The run was interrupted here and has now resumed, in this same conversation. The python kernel restarted, "
    "so its variables and the functions you defined in it are gone: define again what you need. engine.py, its versions "
    "(undo_edit) and everything above are kept."
)


@dataclass
class Run:
    name: str
    dir: Path
    records: list[dict[str, Any]]
    messages: list[dict[str, Any]]  # the full conversation, nothing shortened, every image live
    turn_of: list[int]  # per message: the turn it follows (0 before the first assistant message)
    prompt_tokens: dict[int, int]  # per turn, what the provider counted
    resumed_after: set[int] = field(default_factory=set)  # turns after which the run was resumed

    @property
    def turns(self) -> int:
        return max(self.turn_of) if self.turn_of else 0

    @property
    def versions_dir(self) -> Path:
        return self.dir / "engine_versions"

    def group(self, turn: int) -> list[dict[str, Any]]:
        """The messages of one turn: its assistant message, tool results, images, and what the harness said after."""
        return [m for m, t in zip(self.messages, self.turn_of) if t == turn]

    def before(self, turn: int) -> list[dict[str, Any]]:
        """The full conversation as it stood when turn `turn` was asked."""
        return [m for m, t in zip(self.messages, self.turn_of) if t < turn]


def image_part(path: str, caption: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": IMAGE_URL + path}}]


def load_run(name: str, run_dir: Path) -> Run:
    """The full conversation of a stepwise run from its transcript (older format), nothing shortened."""
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    game = result.get("game") or run_dir.name
    trace = Trace.load(run_dir / "trace")
    starts = [i for i, r in enumerate(records) if "step_start" in r]
    if not starts:
        raise ValueError(f"{run_dir}: no step_start record")
    begin = starts[0]
    version = None
    for r in records[:begin]:
        if isinstance(r.get("engine_change"), dict) and r["engine_change"].get("version"):
            version = r["engine_change"]["version"]
    engine_file = run_dir / "engine_versions" / f"v{version or 1:04d}.py"
    engine = engine_file.read_text(encoding="utf-8")
    k = int(records[begin]["step_start"]["step"])
    visible = Trace(trace.game_id, trace.steps[: k + 1], {**trace.meta, "focus": k})
    read = hashline.render_read(engine, max_chars=READ_CHARS_IN_MESSAGES, fold=fixed_block_lines(engine))
    first = episode_message(game, visible, k, records[begin]["step_start"]["report"], read, True)
    system = system_prompt(result.get("match", "final"), result.get("interface", "simple"), bool(result.get("images", True)), "step", True)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    turn_of = [0, 0]
    auto_passed: dict[int, bool] = {}
    tests = run_dir / "tests.jsonl"
    if tests.exists():
        for line in tests.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("auto") is True:
                auto_passed[entry.get("turn")] = bool(entry.get("passed")) and entry.get("level") is None
    prompt_tokens: dict[int, int] = {}
    resumed_after: set[int] = set()
    focus, turn = k, 0
    last_tool: dict[str, Any] | None = None
    calls: Any = iter(())
    replay: dict[str, Any] | None = None

    def engine_listing() -> str:  # engine.py as it was at this point of the transcript
        file = run_dir / "engine_versions" / f"v{version or 1:04d}.py"
        text = (file if file.exists() else run_dir / "workspace" / "engine.py").read_text(encoding="utf-8")
        return hashline.render_read(text, max_chars=READ_CHARS_IN_MESSAGES, fold=fixed_block_lines(text))

    def with_text(message: dict[str, Any]) -> list[dict[str, Any]]:
        content = message["content"]
        return content if isinstance(content, list) else [{"type": "text", "text": content}]

    def pictures(r: dict[str, Any], note: str | None, tests_note: bool) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "text", "text": note}] if note else []
        for path, caption in zip(r["images"], r.get("captions") or [""] * len(r["images"])):
            text = (TEST_IMAGE_NOTE + " " + caption) if tests_note and "_show" not in path else caption
            parts += image_part(path, text)
        return parts

    def add(message: dict[str, Any]) -> None:
        messages.append(message)
        turn_of.append(turn)

    for r in records[begin + 1 :]:
        if "finish_reason" in r:
            turn = int(r["turn"])
            prompt_tokens[turn] = int((r.get("usage") or {}).get("prompt_tokens") or 0)
            assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
            if r.get("reasoning"):
                assistant["reasoning"] = r["reasoning"]
            if r.get("tool_calls"):
                assistant["tool_calls"] = r["tool_calls"]
                calls = iter(r["tool_calls"])
            add(assistant)
            if not r.get("tool_calls"):
                add({"role": "user", "content": CONTINUE})
        elif "tool" in r:
            call = next(calls, None)
            last_tool = {"role": "tool", "tool_call_id": r.get("id") or (call["id"] if call else ""), "content": r["output"]}
            add(last_tool)
        elif "auto_test" in r and last_tool is not None:
            hint = COMMIT_HINT.format(k=focus) if auto_passed.get(r["turn"]) else ""
            last_tool["content"] += AUTO_TEST.format(report=_truncate(r["auto_test"], AUTO_TEST_CHARS)) + hint
        elif "nudge" in r and last_tool is not None:
            last_tool["content"] += NUDGE.format(n=r["nudge"])
        elif "images" in r:
            if any("_episode" in path for path in r["images"]):
                messages[1]["content"] = with_text(messages[1]) + pictures(r, None, True)
            elif any("_advance" in path for path in r["images"]) and messages[-1]["role"] == "user":
                messages[-1]["content"] = with_text(messages[-1]) + pictures(r, None, True)
            else:
                add({"role": "user", "content": pictures(r, IMAGE_NOTE, True)})
        elif isinstance(r.get("engine_change"), dict) and r["engine_change"].get("version"):
            version = r["engine_change"]["version"]
        elif "advance" in r:
            a = r["advance"]
            focus = a["next"]
            add({"role": "user", "content": advance_message(trace, a["fixed"], a["next"], a["report"], True, engine_listing())})
        elif "replay" in r:
            replay = r["replay"]
        elif "resumed" in r:
            resumed_after.add(turn)
            add({"role": "user", "content": resume_note(replay, "") if replay is not None else LEGACY_RESUME_NOTE})
            replay = None
    return Run(name, run_dir, records, messages, turn_of, prompt_tokens, resumed_after)


# --- (a) the current scheme, replayed as it happened --------------------------------------------------


def hide_images(messages: list[dict[str, Any]]) -> None:
    EngineAgent._hide_images(None, messages)  # type: ignore[arg-type]  (a plain loop over the messages)


def compact(messages: list[dict[str, Any]], config: ModelConfig) -> None:
    shim = types.SimpleNamespace(messages=messages, model=config, _has_engine_listing=EngineAgent._has_engine_listing)
    EngineAgent._compact(shim)  # type: ignore[arg-type]


def _append_live(state: list[dict[str, Any]], group: list[dict[str, Any]]) -> None:
    """Add a turn's messages as the agent does: an image message hides the earlier images first."""
    for m in group:
        if m["role"] == "user" and isinstance(m.get("content"), list) and (m["content"][0].get("text") or "") == IMAGE_NOTE:
            hide_images(state)
        state.append(copy.deepcopy(m))


def current_scheme(run: Run, config: ModelConfig) -> dict[int, list[dict[str, Any]]]:
    """Per turn, the prompt under the agent's scheme, replaying the live loop (compaction, image hiding, resumes)."""
    prompts: dict[int, list[dict[str, Any]]] = {}
    state: list[dict[str, Any]] = []
    _append_live(state, run.group(0))
    for turn in range(1, run.turns + 1):
        prompts[turn] = copy.deepcopy(state)
        group = run.group(turn)
        resume = [m for m in group if m["role"] == "user" and str(m.get("content") or "").startswith(RESUME_HEAD)]
        _append_live(state, [m for m in group if m not in resume])
        if run.prompt_tokens.get(turn, 0) > config.compact_prompt_tokens:
            compact(state, config)
        if turn in run.resumed_after:  # rebuilt from the transcript: nothing shortened unless the last prompt was over
            state = []
            for t in range(0, turn + 1):
                _append_live(state, [m for m in run.group(t) if m not in resume])
            if run.prompt_tokens.get(turn, 0) > config.compact_prompt_tokens:
                compact(state, config)
            state += copy.deepcopy(resume)
    return prompts


def new_scheme(run: Run, **kwargs: Any) -> dict[int, Condensed]:
    return {turn: condense(run.before(turn), run.records, run.versions_dir, **kwargs) for turn in range(1, run.turns + 1)}


# --- measuring and rendering ----------------------------------------------------------------------


def stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    chars, images = measure(messages)
    return {"chars": chars, "images": images, "tokens": chars // 4 + images * 1000}


def render(messages: list[dict[str, Any]]) -> str:
    """The prompt as text: one block per message; images as "[image: caption]"."""
    out: list[str] = []
    for i, m in enumerate(messages):
        head = f"===== [{i}] {m['role'].upper()}"
        if m.get("tool_call_id"):
            head += f" (answers {m['tool_call_id']})"
        out.append(head)
        if m.get("reasoning"):
            out.append("--- reasoning ---\n" + m["reasoning"])
        content = m.get("content")
        if isinstance(content, list):
            caption = ""
            for p in content:
                if p.get("type") == "image_url":
                    out.append(f"[image: {caption or p['image_url']['url']}]")
                else:
                    caption = p.get("text") or ""
                    out.append(caption)
        elif content:
            out.append(content)
        for call in m.get("tool_calls") or []:
            f = call.get("function") or {}
            out.append(f"--- tool call {call.get('id')}: {f.get('name')} ---\n{f.get('arguments')}")
        out.append("")
    return "\n".join(out)


def _table_rows(turns: int, per_turn: dict[int, dict[str, Any]]) -> list[int]:
    wanted = [t for t in (10, 25, 50, 75, 100) if t <= turns]
    peak = max(per_turn, key=lambda t: per_turn[t]["a"]["tokens"])
    peak_b = max(per_turn, key=lambda t: per_turn[t]["b"]["tokens"])
    return sorted(set(wanted) | {peak, peak_b})


def analyse(run: Run, config: ModelConfig, **kwargs: Any) -> dict[str, Any]:
    a, b = current_scheme(run, config), new_scheme(run, **kwargs)
    per_turn: dict[int, dict[str, Any]] = {}
    for turn in range(1, run.turns + 1):
        per_turn[turn] = {
            "a": stats(a[turn]), "b": stats(b[turn].messages), "real": run.prompt_tokens.get(turn, 0),
            "cap": b[turn].cap, "compacted_before": run.prompt_tokens.get(turn - 1, 0) > config.compact_prompt_tokens,
        }
    ratios = [row["real"] / row["a"]["tokens"] for row in per_turn.values() if row["real"] and row["a"]["tokens"]]
    median = statistics.median(ratios)
    # The cap with the estimate calibrated to this run's provider: when would it have fired?
    calibrated = new_scheme(run, chars_per_token=round(4 / median, 2), **kwargs)
    for turn, c in calibrated.items():
        per_turn[turn]["cap_calibrated"] = c.cap
        per_turn[turn]["b_calibrated"] = c.estimate
    return {
        "name": run.name, "dir": str(run.dir), "turns": run.turns, "per_turn": per_turn,
        "total_a": sum(r["a"]["tokens"] for r in per_turn.values()),
        "total_b": sum(r["b"]["tokens"] for r in per_turn.values()),
        "total_real": sum(r["real"] for r in per_turn.values()),
        "cap_fired": sorted(t for t, r in per_turn.items() if r["cap"]),
        "cap_fired_calibrated": sorted(t for t, r in per_turn.items() if r["cap_calibrated"]),
        "chars_per_token_calibrated": round(4 / median, 2),
        "calibration": {"mean": statistics.fmean(ratios), "median": median, "min": min(ratios), "max": max(ratios)},
        "resumed_after": sorted(run.resumed_after),
        "iterations": _iteration_summary(run),
        "_prompts": (a, b),
    }


def _iteration_summary(run: Run) -> list[dict[str, Any]]:
    from engine_re.condense import annotate, iterations

    notes = annotate(run.messages, run.records)
    return [
        {"index": it.index, "step": it.step, "turns": [it.first_turn, it.last_turn], "finished": it.finished,
         "commit_message": bool((it.commit or {}).get("message"))}
        for it in iterations(notes, run.records)
    ]


def report(analyses: list[dict[str, Any]], threshold: int) -> str:
    out = ["# Current compaction vs. the iteration condenser", ""]
    out.append(f"Estimates: 4 characters per token, 1,000 tokens per live image. (a) = the agent's `_compact` as it ran "
               f"(threshold {threshold:,} prompt tokens); (b) = `condense.condense`. 'real' = the prompt_tokens the provider counted.")
    out.append("")
    for an in analyses:
        out.append(f"## {an['name']}  ({an['turns']} turns)")
        out.append("")
        its = an["iterations"]
        spans = ", ".join(f"step {it['step']}: turns {it['turns'][0]}-{it['turns'][1]}" + ("" if it["finished"] else " (open)") for it in its)
        out.append(f"Iterations: {len(its)} ({spans}). Resumed after turn(s): {an['resumed_after'] or 'none'}.")
        out.append("")
        out.append("| turn | real | (a) tokens | (b) tokens | (b)/(a) | (b) x ratio | (a) images | (b) images | compacted before | cap |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|")
        ratio = an["calibration"]["median"]
        peak = max(an["per_turn"], key=lambda t: an["per_turn"][t]["a"]["tokens"])
        for t in _table_rows(an["turns"], an["per_turn"]):
            r = an["per_turn"][t]
            cap = r["cap"]
            cap_text = f"cut {cap['results_cut']} results, reduced {cap['blocks_reduced']} blocks" if cap else ""
            label = f"{t}{' (max a)' if t == peak else ''}"
            out.append(f"| {label} | {r['real']:,} | {r['a']['tokens']:,} | {r['b']['tokens']:,} | {r['b']['tokens'] / r['a']['tokens']:.2f} | "
                       f"{int(r['b']['tokens'] * ratio):,} | {r['a']['images']} | {r['b']['images']} | {'yes' if r['compacted_before'] else ''} | {cap_text} |")
        out.append("")
        out.append(f"Total prompt tokens over the run: real {an['total_real']:,}; (a) estimated {an['total_a']:,}; "
                   f"(b) estimated {an['total_b']:,} ({an['total_b'] / an['total_a']:.2f} of (a)).")
        c = an["calibration"]
        out.append(f"Calibration of (a): real / estimate = mean {c['mean']:.3f}, median {c['median']:.3f}, "
                   f"range {c['min']:.3f}-{c['max']:.3f}.")
        out.append(f"Safety cap fired at {len(an['cap_fired'])} turn(s)" + (f": {an['cap_fired']}." if an["cap_fired"] else "."))
        fired = an["cap_fired_calibrated"]
        out.append(f"With the cap's estimate calibrated to this run ({an['chars_per_token_calibrated']} chars/token), it would have "
                   f"fired at {len(fired)} turn(s)" + (f": {fired}." if fired else "."))
        out.append(f"'(b) x ratio' = (b)'s estimate times this run's median real/estimate ratio ({ratio:.3f}): what the provider "
                   f"would likely count. Total (b) x ratio: {int(an['total_b'] * ratio):,} vs real {an['total_real']:,}.")
        out.append("")
    return "\n".join(out)


# --- the three context schemes, turn by turn, on a run logged with "message" records ----------------


def load_logged_run(name: str, run_dir: Path) -> Run:
    """The full conversation of a stepwise run from its "message" records (as the agent's _rebuild_conversation reads
    them, from the last system message), with nothing hidden or shortened; images kept by file name, not loaded."""
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [i for i, r in enumerate(records) if (r.get("message") or {}).get("role") == "system"]
    if not starts:
        raise ValueError(f"{run_dir}: no message records (an older transcript: use load_run)")
    messages: list[dict[str, Any]] = []
    turn_of: list[int] = []
    prompt_tokens: dict[int, int] = {}
    resumed_after: set[int] = set()
    calls: Any = iter(())
    turn = 0
    for r in records[starts[-1] :]:
        if "message" in r:
            message = dict(r["message"])
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL + p["path"]}} if p.get("type") == "image_file" else p
                    for p in message["content"]
                ]
            messages.append(message)
            turn_of.append(turn)
        elif "finish_reason" in r:
            turn = int(r["turn"])
            prompt_tokens[turn] = int((r.get("usage") or {}).get("prompt_tokens") or 0)
            assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
            if r.get("reasoning"):
                assistant["reasoning"] = r["reasoning"]
            if r.get("tool_calls"):
                assistant["tool_calls"] = r["tool_calls"]
            calls = iter(r.get("tool_calls") or [])
            messages.append(assistant)
            turn_of.append(turn)
        elif "tool" in r:
            call = next(calls, None)
            messages.append({"role": "tool", "tool_call_id": r.get("id") or (call["id"] if call else ""), "content": r["output"]})
            turn_of.append(turn)
        elif "append" in r:
            messages[-1]["content"] += r["append"]
        elif "resumed" in r:
            resumed_after.add(turn)
    return Run(name, run_dir, records, messages, turn_of, prompt_tokens, resumed_after)


def load_any(name: str, run_dir: Path) -> Run:
    """load_logged_run for a transcript with "message" records, else load_run (the older format)."""
    with (Path(run_dir) / "transcript.jsonl").open(encoding="utf-8") as f:
        logged = any('"message": {"role": "system"' in line for line in f)
    return load_logged_run(name, run_dir) if logged else load_run(name, run_dir)


def _with_placeholders(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {**message, "content": [{"type": "text", "text": IMAGE_PLACEHOLDER} if p.get("type") == "image_url" else p for p in content]}


class Tracker:
    """The prompts one scheme sends, turn by turn: their estimated tokens (stats()), how much of each one repeats the
    previous one message for message (what a prefix cache can reuse, estimated), and whether the previous prompt was
    rewritten other than by hiding images (the old images of the agent's _hide_images and of condense's latest-only rule)."""

    def __init__(self) -> None:
        self.rows: dict[int, dict[str, Any]] = {}
        self.fired: list[int] = []  # the turns after which the scheme rewrote the conversation
        self._previous: list[str] | None = None
        self._previous_plain: list[str] | None = None

    def add(self, turn: int, view: list[dict[str, Any]]) -> int:
        exact = [json.dumps(m, sort_keys=True) for m in view]
        plain = [json.dumps(_with_placeholders(m), sort_keys=True) for m in view]
        sizes = [stats([m])["tokens"] for m in view]
        row: dict[str, Any] = {"tokens": sum(sizes), "images": measure(view)[1], "messages": len(view), "reused": 0, "rewritten": False}
        if self._previous is not None:
            k = 0
            while k < min(len(exact), len(self._previous)) and exact[k] == self._previous[k]:
                k += 1
            row["reused"] = sum(sizes[:k])
            row["rewritten"] = plain[: len(self._previous_plain)] != self._previous_plain
        self._previous, self._previous_plain = exact, plain
        self.rows[turn] = row
        return row["tokens"]


def _has_tools(run: Run, turn: int) -> bool:
    """Whether the turn's answer called tools (the agent compacts, or condenses, only after such a turn)."""
    group = run.group(turn)
    return bool(group and group[0]["role"] == "assistant" and group[0].get("tool_calls"))


def _records_before_turn(run: Run, turn: int) -> list[dict[str, Any]]:
    """The records logged before turn `turn` was asked (all of them past the last turn)."""
    index = next((i for i, r in enumerate(run.records) if "finish_reason" in r and int(r["turn"]) == turn), len(run.records))
    return run.records[:index]


def logged_compaction(run: Run) -> Tracker:
    """The prompts of a "compact" run exactly as sent: its message records replayed with the hide_images and compact
    records where they were logged (the agent's _rebuild_conversation, snapshot before every turn)."""
    tracker = Tracker()
    starts = [i for i, r in enumerate(run.records) if (r.get("message") or {}).get("role") == "system"]
    state: list[dict[str, Any]] = []
    calls: Any = iter(())
    config = ModelConfig()
    fired: list[int] = []
    turn = 0
    for r in run.records[starts[-1] :]:
        if "message" in r:
            message = copy.deepcopy(r["message"])
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL + p["path"]}} if p.get("type") == "image_file" else p
                    for p in message["content"]
                ]
            state.append(message)
        elif "finish_reason" in r:
            turn = int(r["turn"])
            tracker.add(turn, state)
            assistant: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
            if r.get("reasoning"):
                assistant["reasoning"] = r["reasoning"]
            if r.get("tool_calls"):
                assistant["tool_calls"] = copy.deepcopy(r["tool_calls"])
            calls = iter(r.get("tool_calls") or [])
            state.append(assistant)
        elif "tool" in r:
            call = next(calls, None)
            state.append({"role": "tool", "tool_call_id": r.get("id") or (call["id"] if call else ""), "content": r["output"]})
        elif "append" in r:
            state[-1]["content"] += r["append"]
        elif "hide_images" in r:
            hide_images(state)
        elif "compact" in r:
            compact(state, config)
            fired.append(turn)
    tracker.fired = fired
    return tracker


def simulated_compaction(run: Run, config: ModelConfig, over: Any) -> Tracker:
    """`_compact` replayed on the full conversation: images hidden when a newer image message comes, and the conversation
    compacted in place after every turn with tools for which over(turn, estimated tokens of the prompt it was sent)."""
    tracker = Tracker()
    state: list[dict[str, Any]] = []
    fired: list[int] = []
    _append_live(state, run.group(0))
    for turn in range(1, run.turns + 1):
        tokens = tracker.add(turn, state)
        _append_live(state, run.group(turn))
        if _has_tools(run, turn) and over(turn, tokens):
            compact(state, config)
            fired.append(turn)
    tracker.fired = fired
    return tracker


def per_turn_condense(run: Run, keep_turns: int, chars_per_token: float) -> tuple[Tracker, list[dict[str, Any]]]:
    """The earlier per-turn mode: every request is condense() of the full conversation as it stands, with the records
    logged until then. Also returns, per turn, the record the agent would have logged (to check against a logged run)."""
    tracker = Tracker()
    logged: list[dict[str, Any]] = []
    for turn in range(1, run.turns + 1):
        result = condense(run.before(turn), _records_before_turn(run, turn), run.versions_dir, keep_turns=keep_turns,
                          chars_per_token=chars_per_token)
        tracker.add(turn, result.messages)
        chars, images = measure(result.messages)
        logged.append({"tokens": result.estimate, "chars": chars, "images": images, "messages": len(result.messages)})
    tracker.fired = list(range(1, run.turns + 1))
    return tracker, logged


def threshold_condense(run: Run, over: Any, keep_turns: int, chars_per_token: float) -> Tracker:
    """The agent's "condense" context: the full conversation until the condenser fires; after every turn with tools for
    which over(turn, estimated tokens of the prompt it was sent), condense() of the whole conversation so far becomes the
    prefix of every request, the messages since following it with only their latest images live (EngineAgent._context)."""
    tracker = Tracker()
    prefix: list[dict[str, Any]] = []
    covered = 0
    fired: list[int] = []
    for turn in range(1, run.turns + 1):
        tokens = tracker.add(turn, prefix + hide_but_latest(run.before(turn)[covered:]))
        if _has_tools(run, turn) and over(turn, tokens):
            conversation = run.before(turn + 1)
            prefix = condense(conversation, _records_before_turn(run, turn + 1), run.versions_dir, keep_turns=keep_turns,
                              chars_per_token=chars_per_token).messages
            covered = len(conversation)
            fired.append(turn)
    tracker.fired = fired
    return tracker


def _summary(tracker: Tracker, ratio: float, real: dict[int, int] | None = None) -> dict[str, Any]:
    rows = tracker.rows
    total = sum(r["tokens"] for r in rows.values())
    out = {
        "total": total, "max": max(r["tokens"] for r in rows.values()),
        "total_calibrated": int(total * ratio), "max_calibrated": int(max(r["tokens"] for r in rows.values()) * ratio),
        "fired": len(tracker.fired), "fired_turns": tracker.fired,
        "rewritten": sum(r["rewritten"] for r in rows.values()),
        "reused_share": sum(r["reused"] for r in rows.values()) / total if total else 0.0,
        "not_reused_calibrated": int(sum(r["tokens"] - r["reused"] for r in rows.values()) * ratio),
    }
    if real:
        out["real_total"], out["real_max"] = sum(real.values()), max(real.values())
    return out


def compare_schemes(run: Run, threshold: int, keep_turns: int, chars_per_token: float = 3.0, old_keep_turns: int = 5
                    ) -> dict[str, Any]:
    """The run under the three schemes, the one it ran with (logged) and the two others (simulated):

    compaction       `_compact` after every request over `threshold` prompt tokens: as logged for a "compact" run (its
                     records replayed exactly); for a "condense" run, simulated, firing when the estimate of its own
                     prompt (calibrated to the run) is over the threshold.
    per-turn         condense() before every request (keep_turns=`old_keep_turns`): what the earlier "condense" context
                     did; for such a run, checked against its logged "condense" records.
    threshold        the "condense" context now (keep_turns=`keep_turns`), in two variants of when it fires:
                     "as compaction": after the turns at which the compaction above fired (the same WHEN; for a
                     "compact" run that is where the logged prompt_tokens were over the threshold), and "on its own
                     prompt": when the estimate of the prompt it sent itself, calibrated to the run, is over it (what
                     the live agent would do, since it sends shorter prompts than the compaction).

    'calibrated' multiplies an estimate by the run's median ratio of the provider's prompt_tokens to the estimate of the
    prompts it was actually sent."""
    config = ModelConfig(compact_prompt_tokens=threshold)
    logged_condense = any("condense" in r for r in run.records)
    if logged_condense:
        per_turn, records = per_turn_condense(run, old_keep_turns, chars_per_token)
        reference = per_turn
        logged = [r["condense"] for r in run.records if "condense" in r]
        check = {"condense_records_match": records == logged, "condense_records": len(logged)}
    else:
        reference = logged_compaction(run)
        check = {}
    ratios = [run.prompt_tokens[t] / row["tokens"] for t, row in reference.rows.items() if run.prompt_tokens.get(t) and row["tokens"]]
    ratio = statistics.median(ratios)
    if logged_condense:
        compaction = simulated_compaction(run, config, lambda t, tokens: tokens * ratio > threshold)
    else:
        compaction = reference
        per_turn, _ = per_turn_condense(run, old_keep_turns, chars_per_token)
        simulated = simulated_compaction(run, config, lambda t, tokens: run.prompt_tokens.get(t, 0) > threshold)
        check = {"compaction_simulation_matches_log": simulated.rows == compaction.rows}
    compaction_fired = set(compaction.fired)
    same_when = threshold_condense(run, lambda t, tokens: t in compaction_fired, keep_turns, chars_per_token)
    own = threshold_condense(run, lambda t, tokens: tokens * ratio > threshold, keep_turns, chars_per_token)
    full = Tracker()
    for turn in range(1, run.turns + 1):
        full.add(turn, hide_but_latest(run.before(turn)))
    real = {t: v for t, v in run.prompt_tokens.items() if v}
    return {
        "name": run.name, "dir": str(run.dir), "turns": run.turns, "logged": "per-turn" if logged_condense else "compaction",
        "ratio": ratio, "threshold": threshold, "keep_turns": keep_turns, "old_keep_turns": old_keep_turns, "check": check,
        "real": {"total": sum(real.values()), "max": max(real.values()),
                 "cached_share": sum(int(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                                     for r in run.records if "finish_reason" in r) / sum(real.values())},
        "schemes": {
            "uncondensed": _summary(full, ratio),
            "compaction": _summary(compaction, ratio, None if logged_condense else real),
            "per-turn": _summary(per_turn, ratio, real if logged_condense else None),
            "threshold, as compaction": _summary(same_when, ratio),
            "threshold, on its own prompt": _summary(own, ratio),
        },
        "per_turn": {name: tracker.rows for name, tracker in
                     (("compaction", compaction), ("per-turn", per_turn), ("threshold, as compaction", same_when),
                      ("threshold, on its own prompt", own))},
    }


def schemes_report(results: list[dict[str, Any]]) -> str:
    out = ["# Context schemes: compaction, per-turn condense, threshold condense", ""]
    out.append(
        "Per run, the prompt every scheme would send at every turn, rebuilt from the transcript. Tokens are estimates: "
        "4 characters per token and 1,000 per live image ('est.'), and that estimate times the run's median ratio of the "
        "provider's prompt_tokens to the estimate of the prompts it was really sent ('cal.'). 'fired' counts the turns after "
        "which the scheme rewrote the conversation (the per-turn condenser: every turn). 'rewritten' counts the turns whose "
        "prompt does not start with the previous turn's prompt, image placeholders aside (hiding the previous turn's "
        "images is common to all schemes). 'reused' is the share of the estimated prompt tokens that repeat the previous "
        "prompt message for message: an estimate of what a prefix cache can serve; 'not reused' the rest (cal.), the "
        "tokens a prefix cache cannot serve."
    )
    out.append("")
    for res in results:
        thr = res["threshold"]
        out.append(f"## {res['name']} ({res['turns']} turns; logged with the {res['logged']} scheme)")
        out.append("")
        out.append(f"`{res['dir']}`. Calibration: {res['ratio']:.3f} real tokens per estimated token. The provider counted "
                   f"{res['real']['total']:,} prompt tokens in all (max {res['real']['max']:,}), "
                   f"{res['real']['cached_share']:.0%} of them cached.")
        out.append("")
        out.append("| scheme | total est. | total cal. | max est. | max cal. | fired | rewritten | reused | not reused cal. |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        labels = {
            "uncondensed": "nothing (the full conversation, latest images only)",
            "compaction": "old compaction (`_compact`)" + (", as logged" if res["logged"] == "compaction" else
                                                          ", simulated: fires when its own prompt is over (cal.)"),
            "per-turn": f"per-turn condense (keep {res['old_keep_turns']} turns)" + (", as logged" if res["logged"] == "per-turn" else ", simulated"),
            "threshold, as compaction": f"threshold condense (keep {res['keep_turns']}), fires where compaction fired"
                                        + (" (logged prompt_tokens > " + f"{thr:,})" if res["logged"] == "compaction" else
                                           " (est. of the uncondensed, compacted prompt, cal. > " + f"{thr:,})"),
            "threshold, on its own prompt": f"threshold condense (keep {res['keep_turns']}), fires when its own prompt is over (cal. > {thr:,})",
        }
        for key, label in labels.items():
            s = res["schemes"][key]
            fired = "-" if key == "uncondensed" else str(s["fired"])
            rewritten = "-" if key == "uncondensed" else str(s["rewritten"])
            out.append(f"| {label} | {s['total']:,} | {s['total_calibrated']:,} | {s['max']:,} | {s['max_calibrated']:,} | "
                       f"{fired} | {rewritten} | {s['reused_share']:.0%} | {s['not_reused_calibrated']:,} |")
        out.append("")
        for key in ("compaction", "threshold, as compaction", "threshold, on its own prompt"):
            turns = res["schemes"][key]["fired_turns"]
            out.append(f"- {key}: fired after turn(s) {turns or 'none'}.")
        if res["check"]:
            out.append(f"- checks: {res['check']}.")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="<name>=<run dir>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", nargs="*", default=[], help="<name>:<turn>: write both full prompts as text")
    parser.add_argument("--threshold", type=int, default=CAP_TOKENS)
    parser.add_argument("--schemes", action="store_true",
                        help="Compare the three context schemes (compaction, per-turn condense, threshold condense) instead: "
                             "schemes.md and schemes.json.")
    parser.add_argument("--keep-turns", type=int, default=KEEP_TURNS, help="--schemes: keep_turns of the threshold condenser.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.schemes:
        results = []
        for spec in args.runs:
            name, _, folder = spec.partition("=")
            res = compare_schemes(load_any(name, Path(folder)), args.threshold, args.keep_turns)
            results.append(res)
            print(f"{name}: " + "; ".join(f"{k} total {v['total']:,} fired {v['fired']} rewritten {v['rewritten']}"
                                          for k, v in res["schemes"].items()) + f"; checks {res['check']}", file=sys.stderr)
        (args.out / "schemes.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (args.out / "schemes.md").write_text(schemes_report(results), encoding="utf-8")
        return 0
    config = ModelConfig(compact_prompt_tokens=args.threshold)
    analyses = []
    samples = [s.split(":") for s in args.samples]
    for spec in args.runs:
        name, _, folder = spec.partition("=")
        run = load_run(name, Path(folder))
        an = analyse(run, config, cap_tokens=args.threshold)
        a, b = an.pop("_prompts")
        for sample_name, turn in samples:
            if sample_name == name:
                t = int(turn)
                (args.out / f"{name}_t{t}_current.txt").write_text(render(a[t]), encoding="utf-8")
                (args.out / f"{name}_t{t}_new.txt").write_text(render(b[t].messages), encoding="utf-8")
        analyses.append(an)
        print(f"{name}: {run.turns} turns, total (a) {an['total_a']:,} (b) {an['total_b']:,} real {an['total_real']:,}; "
              f"calibration median {an['calibration']['median']:.3f}; cap at {an['cap_fired']}", file=sys.stderr)
    (args.out / "measurements.json").write_text(json.dumps(analyses, indent=1), encoding="utf-8")
    (args.out / "report.md").write_text(report(analyses, args.threshold), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
