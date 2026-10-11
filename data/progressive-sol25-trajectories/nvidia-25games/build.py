"""Export the NVIDIA DreamTeam progressive-thinking run as loss-masked trajectories.

Same format and checks as ../build.py (the 25 official games), applied to the
22 games gpt-6.1-sol won in runs/gpt61sol-nvidia-25games-resume2. Counts are
derived from the source rather than fixed. Run from anywhere with
ARC3-Inference/.venv/bin/python; the source run, the generated run and the
pinned Qwen tokenizer/template must be present locally.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))
from think_gen import context, logs, progressive  # noqa: E402

_spec = importlib.util.spec_from_file_location("sol25_build", HERE.parent / "build.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

RUN = ROOT / "ARC3-Inference/runs/think-progressive-nvidia22"
SOURCE = ROOT / "ARC3-Inference/runs/gpt61sol-nvidia-25games-resume2"
QWEN = base.QWEN
LIMIT = base.LIMIT
KWARGS = base.KWARGS


def won_games(source):
    """Game runs that ended `won` in the scored benchmark.json."""
    return sorted(g for g, o in logs.game_outcomes(source).items() if o["state"] == "won")


def extends(previous, messages):
    """Does `messages` continue `previous`?

    The one allowed difference: a text-only reply ending `previous` is kept in
    later history with blank lines removed (the harness's normalization), so
    the trajectory carries that history copy. Such turns are masked targets.
    """
    head = messages[:len(previous)]
    if head == previous:
        return True
    last, copy = previous[-1], head[-1] if head else {}
    return (head[:-1] == previous[:-1] and last["role"] == copy.get("role") == "assistant"
            and not last.get("tool_calls") and not copy.get("tool_calls")
            and {k: v for k, v in last.items() if k != "content"} ==
            {k: v for k, v in copy.items() if k != "content"}
            and isinstance(last.get("content"), str)
            and context.normalize(last["content"]) == copy.get("content"))


def validate_boundary(previous, current):
    """Check where a new trajectory starts and describe the cut.

    note_compaction: ../build.py's check, with the retained turn count read
    from the notice. When ten turns would not fit, the harness keeps fewer and
    says so ("your last 9 turns"); the official games always kept ten.

    context_trim: the harness's budget trimmer dropped the oldest history
    mid-turn, with no notice (fw4821 only). The new trajectory is the system
    prompt followed by an exact suffix of the previous one, then new messages.
    """
    notices = [m["content"] for m in current["messages"]
               if m["role"] == "user" and isinstance(m.get("content"), str)
               and m["content"].startswith("Context notice:")]
    if notices and notices[-1] in [m.get("content") for m in previous["messages"][-2:]]:
        notice = notices[-1]
        match = re.search(r"last (\d+) turns", notice)
        if not match or not 1 <= int(match.group(1)) <= 10:
            raise ValueError(f"Unexpected compaction retention: {current['id']}")
        kept = int(match.group(1))
        stated = notice.replace(f"last {kept} turns", "last 10 turns", 1)
        def restate(sample):
            return {**sample, "messages": [{**m, "content": stated} if m.get("content") == notice else m
                                           for m in sample["messages"]]}
        boundary = base.validate_boundary(restate(previous), restate(current))
        boundary.update({"kind": "note_compaction", "retained_game_turns": kept})
        return boundary
    old, new = previous["messages"], current["messages"]
    if new[0] != old[0] or new[0]["role"] != "system":
        raise ValueError(f"Trim changed the system prompt: {current['id']}")
    for start in range(2, len(old)):
        if extends(old[start:], new[1:]):
            retained = len(old) - start
            return {"kind": "context_trim", "dropped_messages": start - 1,
                    "retained_messages": retained,
                    "retained_assistant_messages": sum(m["role"] == "assistant" for m in new[1:1 + retained]),
                    "first_new_message_index": 1 + retained}
    raise ValueError(f"New trajectory is neither a compaction nor a trimmed suffix: {current['id']}")


def finish(trajectory, student):
    """../build.py's checks and token counts, then mask text-only turns.

    A text-only reply (no tool call) is the teacher giving up on a board the
    NVIDIA adapter froze ("RESET is not available..."). It stays in the context
    exactly as the teacher saw it, but is not a loss target, and its tokens
    count as input rather than supervised output.
    """
    trajectory = base.finish(trajectory, student)
    targets = [t for t in trajectory["turns"] if not t["text_only"]]
    if not targets:
        raise ValueError(f"No supervised turn: {trajectory['trajectory_id']}")
    output_tokens = sum(t["output_tokens"] for t in targets)
    trajectory.update({
        "loss_target_message_indices": [t["assistant_message_index"] for t in targets],
        "masked_text_only_message_indices": [t["assistant_message_index"]
                                             for t in trajectory["turns"] if t["text_only"]],
        "output_tokens": output_tokens,
        "input_tokens": trajectory["total_tokens"] - output_tokens,
    })
    return trajectory


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    sha256 = progressive.file_hash
    if sha256(QWEN / "tokenizer.json") != manifest["tokenizer"] or \
            sha256(QWEN / "chat_template.jinja") != manifest["template"]:
        raise ValueError("Pinned tokenizer/template hash mismatch")
    student = progressive.StudentFormat(QWEN / "tokenizer.json", QWEN / "chat_template.jinja")
    won = won_games(SOURCE)
    source_paths = [p for p in logs.request_logs(SOURCE)
                    if logs.LOG_RE.match(p.name)["game"] + "_p0" in won]
    if len(source_paths) != len(won):
        raise ValueError(f"Expected {len(won)} won-game logs, got {len(source_paths)}")
    trajectories, all_ids = [], set()
    old_eligible_checked = boundary_count = text_only = 0
    for path in source_paths:
        if sha256(path) != manifest["source"][path.name]:
            raise ValueError(f"Source hash mismatch: {path}")
        records = logs.read_log(path)
        history, previous, active, system_message = {}, None, None, None
        exported = {}
        with (RUN / "sft" / f"{records[0].game}.jsonl").open() as stream:
            for line in stream:
                sample = json.loads(line)
                exported[sample["id"]] = sample
        for rec in records:
            final = RUN / "turns" / rec.game / f"{rec.index:05d}" / "final.json"
            if not final.exists():
                raise ValueError(f"Missing finalized turn: {rec.key}")
            row = json.loads(final.read_text())
            if row["key"] != rec.key or row["index"] != rec.index or \
                    row["ref"] != context.message_ref(rec.reply):
                raise ValueError(f"Checkpoint mismatch: {rec.key}")
            messages, tools = student.messages(rec, history, row["thinking"])
            if not messages or messages[0]["role"] != "system":
                raise ValueError(f"Missing initial system prompt: {rec.key}")
            if system_message is None:
                system_message = messages[0]
            elif messages[0] != system_message:
                raise ValueError(f"System prompt changed: {rec.key}")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"Target role mismatch: {rec.key}")
            if rec.key in exported:
                old = exported[rec.key]
                if old["messages"] != messages or old["tools"] != tools or \
                        old["student_input_tokens"] != row["student_input_tokens"]:
                    raise ValueError(f"Existing SFT sample mismatch: {rec.key}")
                old_eligible_checked += 1
            elif row["training_eligible"]:
                raise ValueError(f"Missing SFT sample: {rec.key}")
            output = base.target_tokens(student, messages, tools)
            if row["student_input_tokens"] + output > LIMIT:
                raise ValueError(f"Turn exceeds {LIMIT}: {rec.key}")
            sample = {"id": rec.key, "messages": messages, "tools": tools, "chunk": row["chunk"]}
            if active is None or row["chunk"] != active["chunk"]:
                retained = None
                if active is not None:
                    active["boundary_to_next"] = validate_boundary(previous, sample)
                    boundary_count += 1
                    trajectories.append(finish(active, student))
                    retained = active["boundary_to_next"]
                active = {
                    "trajectory_id": f"{rec.game}:chunk-{row['chunk']:02d}",
                    "game": rec.game.rsplit("_p", 1)[0],
                    "game_run": rec.game,
                    "pass": int(rec.game.rsplit("_p", 1)[1]),
                    "chunk": row["chunk"],
                    "max_total_tokens": LIMIT,
                    "tokenizer_sha256": manifest["tokenizer"],
                    "chat_template_sha256": manifest["template"],
                    "messages": messages,
                    "tools": tools,
                    "chat_template_kwargs": KWARGS,
                    "retained_from_previous": retained,
                    "turns": [],
                    "boundary_to_next": None,
                }
            else:
                if not extends(previous["messages"], messages):
                    raise ValueError(f"Non-prefix continuation: {rec.key}")
                active["messages"] = messages
                if active["tools"] != tools:
                    raise ValueError(f"Tool schema changed: {rec.key}")
            assistant_index = len(messages) - 1
            if active["turns"] and assistant_index <= active["turns"][-1]["assistant_message_index"]:
                raise ValueError(f"Non-increasing target positions: {rec.key}")
            is_text = not rec.reply.get("tool_calls")
            text_only += is_text
            active["turns"].append({
                "id": rec.key,
                "request_index": rec.index,
                "assistant_message_index": assistant_index,
                "input_tokens": row["student_input_tokens"],
                "output_tokens": output,
                "total_tokens": row["student_input_tokens"] + output,
                "previous_120k_input_eligible": row["training_eligible"],
                "text_only": is_text,
            })
            if rec.key in all_ids:
                raise ValueError(f"Duplicate target: {rec.key}")
            all_ids.add(rec.key)
            history[row["ref"]] = row["thinking"]
            previous = sample
        trajectories.append(finish(active, student))
    if Counter(t["game_run"] for t in trajectories).keys() != set(won):
        raise ValueError("Missing a game")

    data_path = HERE / "trajectories.jsonl"
    index = []
    with data_path.open("wb") as stream:
        for line_no, trajectory in enumerate(trajectories):
            raw = (json.dumps(trajectory, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            offset = stream.tell()
            stream.write(raw)
            index.append({k: trajectory[k] for k in (
                "trajectory_id", "game", "game_run", "pass", "chunk", "start_turn",
                "end_turn", "turn_count", "message_count", "image_count",
                "input_tokens", "output_tokens", "total_tokens", "terminal_prompt_tokens",
                "terminal_output_tokens", "ends_with_compaction_note")})
            index[-1].update({"text_only_turns": sum(t["text_only"] for t in trajectory["turns"]),
                              "line": line_no, "offset": offset, "length": len(raw)})
    (HERE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "ratios.csv").open("w", newline="") as stream:
        fields = ["trajectory_id", "game", "start_turn", "end_turn",
                  "input_tokens", "output_tokens", "output_input_ratio"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in index:
            writer.writerow({**{key: row[key] for key in fields[:-1]},
                             "output_input_ratio": f"{row['output_tokens'] / row['input_tokens']:.6f}"})
    summary = {
        "source_run": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": sha256(RUN / "manifest.json"),
        "trajectories_sha256": sha256(data_path),
        "max_total_tokens": LIMIT,
        "games": len(won),
        "trajectories": len(trajectories),
        "boundaries": boundary_count,
        "boundary_kinds": dict(Counter(t["boundary_to_next"]["kind"] for t in trajectories
                                       if t["boundary_to_next"])),
        "finalized_turns": len(all_ids),
        "supervised_targets": len(all_ids) - text_only,
        "masked_text_only_turns": text_only,
        "not_in_120k_input_sft_export": len(all_ids) - old_eligible_checked,
        "max_trajectory_total_tokens": max(t["total_tokens"] for t in trajectories),
        "max_turn_total_tokens": max(turn["total_tokens"] for t in trajectories for turn in t["turns"]),
        "role_counts": dict(Counter(m["role"] for t in trajectories for m in t["messages"])),
        "total_output_tokens": sum(t["output_tokens"] for t in trajectories),
        "total_sequence_tokens": sum(t["total_tokens"] for t in trajectories),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
