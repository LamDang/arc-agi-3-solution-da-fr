"""Export the finalized progressive-thinking run as loss-masked trajectories.

Run from anywhere with ARC3-Inference/.venv/bin/python. The source logs, run
snapshot, and pinned Qwen tokenizer/template must already be present locally.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))
from think_gen import context, logs, progressive  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = ROOT / "ARC3-Inference/runs/think-progressive-sol25"
SOURCE = ROOT / "ARC3-Inference/runs/gpt61sol-features-25games"
QWEN = ROOT / "data/sft-gpt61sol-features-25games/qwen"
LIMIT = 130_000
KWARGS = {"enable_thinking": True, "preserve_thinking": True}


def sha256(path: Path) -> str:
    return progressive.file_hash(path)


def target_tokens(student, messages, tools):
    """Count one assistant response with the same rendering as the SFT index."""
    prompt = student.index.render(student.template, messages[:-1], tools, KWARGS, True)
    full = student.index.render(student.template, messages, tools, KWARGS)
    if not full.startswith(prompt):
        raise ValueError("Assistant target is not a rendered prompt suffix")
    return len(student.tokenizer.encode(full[len(prompt):], add_special_tokens=False).ids)


def validate_boundary(previous, current):
    """The next chunk starts with the retained suffix and the previous note."""
    messages = current["messages"]
    positions = [i for i, message in enumerate(messages)
                 if message["role"] == "user" and
                 isinstance(message.get("content"), str) and
                 message["content"].startswith("Context notice:")]
    if len(positions) != 1:
        raise ValueError(f"Missing unique compaction notice: {current['id']}")
    at = positions[0]
    if messages[at:at + 2] != previous["messages"][-2:]:
        raise ValueError(f"Compaction prompt/answer mismatch: {current['id']}")
    if messages[at + 2]["role"] != "tool" or not \
            messages[at + 2]["content"].startswith("Note kept in your context"):
        raise ValueError(f"Missing compaction tool result: {current['id']}")
    match = re.search(r"last (\d+) turns", messages[at]["content"])
    if not match or int(match.group(1)) != 10:
        raise ValueError(f"Unexpected compaction retention: {current['id']}")
    return {"retained_game_turns": 10,
            "retained_assistant_messages": sum(m["role"] == "assistant"
                                               for m in messages[1:at]),
            "compaction_notice_message_index": at,
            "compaction_result_message_index": at + 2}


def finish(trajectory, student):
    messages, tools = trajectory["messages"], trajectory["tools"]
    turns = trajectory["turns"]
    if not turns or messages[-1]["role"] != "assistant":
        raise ValueError(f"Incomplete trajectory: {trajectory['trajectory_id']}")
    output_tokens = sum(t["output_tokens"] for t in turns)
    # Exact Qwen prompt count for the final request + its final assistant suffix.
    # The final request contains all preceding messages in this chunk.
    input_before_last = student.count(messages[:-1], tools)
    if input_before_last != turns[-1]["input_tokens"]:
        raise ValueError(f"Final prompt count changed: {trajectory['trajectory_id']}")
    total_tokens = input_before_last + turns[-1]["output_tokens"]
    if total_tokens > LIMIT:
        raise ValueError(f"Trajectory exceeds {LIMIT}: {trajectory['trajectory_id']}={total_tokens}")
    trajectory.update({
        "start_turn": turns[0]["request_index"],
        "end_turn": turns[-1]["request_index"],
        "turn_count": len(turns),
        "message_count": len(messages),
        "image_count": student.index.count_images(messages),
        "input_tokens": total_tokens - output_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "terminal_prompt_tokens": input_before_last,
        "terminal_output_tokens": turns[-1]["output_tokens"],
        "loss_target_message_indices": [t["assistant_message_index"] for t in turns],
        "ends_with_compaction_note": any(
            m["role"] == "user" and isinstance(m.get("content"), str) and
            m["content"].startswith("Context notice:") for m in messages[-2:]),
    })
    if output_tokens > total_tokens:
        raise ValueError(f"Bad loss-token accounting: {trajectory['trajectory_id']}")
    return trajectory


def main():
    snapshot = json.loads((RUN / "snapshot.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    if snapshot["finalized"] != 1334 or snapshot["games_complete"] != 25:
        raise ValueError("Expected complete 25-game finalized snapshot")
    if sha256(QWEN / "tokenizer.json") != manifest["tokenizer"] or \
            sha256(QWEN / "chat_template.jinja") != manifest["template"]:
        raise ValueError("Pinned tokenizer/template hash mismatch")
    student = progressive.StudentFormat(QWEN / "tokenizer.json",
                                        QWEN / "chat_template.jinja")
    source_paths = logs.request_logs(SOURCE)
    if len(source_paths) != 25:
        raise ValueError(f"Expected 25 source logs, got {len(source_paths)}")
    trajectories = []
    all_ids = set()
    old_eligible_checked = 0
    boundary_count = 0
    for path in source_paths:
        if sha256(path) != manifest["source"][path.name]:
            raise ValueError(f"Source hash mismatch: {path}")
        records = logs.read_log(path)
        history = {}
        previous = None
        active = None
        system_message = None
        exported = {}
        old_sft = RUN / "sft" / f"{records[0].game}.jsonl"
        with old_sft.open() as stream:
            for line in stream:
                sample = json.loads(line)
                exported[sample["id"]] = sample
        for rec in records:
            row = json.loads((RUN / "turns" / rec.game /
                              f"{rec.index:05d}" / "final.json").read_text())
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
                raise ValueError(f"Missing old SFT sample: {rec.key}")
            output = target_tokens(student, messages, tools)
            if row["student_input_tokens"] + output > LIMIT:
                raise ValueError(f"Turn exceeds {LIMIT}: {rec.key}")
            sample = {"id": rec.key, "messages": messages, "tools": tools,
                      "chunk": row["chunk"]}
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
                if messages[:len(previous["messages"])] != previous["messages"]:
                    raise ValueError(f"Non-prefix continuation: {rec.key}")
                active["messages"] = messages
                if active["tools"] != tools:
                    raise ValueError(f"Tool schema changed: {rec.key}")
            assistant_index = len(messages) - 1
            if active["turns"] and assistant_index <= \
                    active["turns"][-1]["assistant_message_index"]:
                raise ValueError(f"Non-increasing target positions: {rec.key}")
            active["turns"].append({
                "id": rec.key,
                "request_index": rec.index,
                "assistant_message_index": assistant_index,
                "input_tokens": row["student_input_tokens"],
                "output_tokens": output,
                "total_tokens": row["student_input_tokens"] + output,
                "previous_120k_input_eligible": row["training_eligible"],
            })
            if rec.key in all_ids:
                raise ValueError(f"Duplicate target: {rec.key}")
            all_ids.add(rec.key)
            history[row["ref"]] = row["thinking"]
            previous = sample
        trajectories.append(finish(active, student))
    if len(trajectories) != 58 or len(all_ids) != 1334 or \
            old_eligible_checked != 1324 or boundary_count != 33:
        raise ValueError("Unexpected game/trajectory/target counts")
    if Counter(t["game_run"] for t in trajectories).keys() != \
            {r["game"] for r in snapshot["games"]}:
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
            index[-1].update({"line": line_no, "offset": offset, "length": len(raw)})
    (HERE / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "ratios.csv").open("w", newline="") as stream:
        fields = ["trajectory_id", "game", "start_turn", "end_turn",
                  "input_tokens", "output_tokens", "output_input_ratio"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in index:
            writer.writerow({**{key: row[key] for key in fields[:-1]},
                             "output_input_ratio":
                             f"{row['output_tokens'] / row['input_tokens']:.6f}"})
    summary = {
        "source_snapshot_sha256": sha256(RUN / "snapshot.json"),
        "source_manifest_sha256": sha256(RUN / "manifest.json"),
        "trajectories_sha256": sha256(data_path),
        "max_total_tokens": LIMIT,
        "games": 25,
        "trajectories": len(trajectories),
        "compactions": boundary_count,
        "supervised_targets": len(all_ids),
        "newly_included_from_old_120k_input_filter": len(all_ids) - old_eligible_checked,
        "max_trajectory_total_tokens": max(t["total_tokens"] for t in trajectories),
        "max_turn_total_tokens": max(turn["total_tokens"] for t in trajectories
                                     for turn in t["turns"]),
        "role_counts": dict(Counter(m["role"] for t in trajectories
                                    for m in t["messages"])),
        "total_output_tokens": sum(t["output_tokens"] for t in trajectories),
        "total_sequence_tokens": sum(t["total_tokens"] for t in trajectories),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
