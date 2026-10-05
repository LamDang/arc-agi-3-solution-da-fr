"""Compare the agent's compaction (`_compact`) with `condense.condense` on finished stepwise runs.

    uv run --no-sync python -m engine_re.condense_report --out <folder> <name>=<run dir> ... \\
        [--samples <name>:<turn> ...] [--threshold 140000]

For every run and every turn t it builds the prompt the model would receive under (a) the current
scheme and (b) the new one, measures characters, estimated tokens (4 per character, 1000 per live
image) and live images, and compares (a)'s estimate with the prompt_tokens the provider counted. The
output folder gets `measurements.json` (every turn), `report.md` (tables at turns 10, 25, 50, 75, 100
and the maximum, totals, the safety cap, calibration) and, for each sample, the two full prompts as text
(`<name>_t<turn>_current.txt`, `<name>_t<turn>_new.txt`; images shown as "[image: caption]").

The full conversation is rebuilt from transcript.jsonl in the older format (no "message" records), as
the agent's `_rebuild_legacy` does, but across the whole run, resumes included, and with nothing
shortened; images are kept by file name, not loaded. (a) replays what happened live: `_compact` after
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
    NUDGE,
    READ_CHARS_IN_MESSAGES,
    RESUME_NOTE,
    TEST_IMAGE_NOTE,
    EngineAgent,
    ModelConfig,
    _truncate,
)
from engine_re.condense import CAP_TOKENS, Condensed, condense, measure
from engine_re.game_api import fixed_block_lines
from engine_re.prompts import advance_message, episode_message, system_prompt
from engine_re.trace import Trace

IMAGE_URL = "image:"  # an image part's url: the PNG's path in the run folder (the bytes are not needed)


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
        elif "advance" in r:
            a = r["advance"]
            focus = a["next"]
            add({"role": "user", "content": advance_message(trace, a["fixed"], a["next"], a["report"], True)})
        elif "resumed" in r:
            resumed_after.add(turn)
            add({"role": "user", "content": RESUME_NOTE})
    return Run(name, run_dir, records, messages, turn_of, prompt_tokens, resumed_after)


# --- (a) the current scheme, replayed as it happened --------------------------------------------------


def hide_images(messages: list[dict[str, Any]]) -> None:
    EngineAgent._hide_images(None, messages)  # type: ignore[arg-type]  (a plain loop over the messages)


def compact(messages: list[dict[str, Any]], config: ModelConfig) -> None:
    EngineAgent._compact(types.SimpleNamespace(messages=messages, model=config))  # type: ignore[arg-type]


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
        resume = [m for m in group if m["role"] == "user" and (m.get("content") or "") == RESUME_NOTE]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="<name>=<run dir>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", nargs="*", default=[], help="<name>:<turn>: write both full prompts as text")
    parser.add_argument("--threshold", type=int, default=CAP_TOKENS)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
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
