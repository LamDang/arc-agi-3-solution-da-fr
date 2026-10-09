"""Verify and measure the released 25-game generated-thinking SFT export.

Run from the repository root with ARC3-Inference/.venv/bin/python. Requires the
DVC checkout and local pinned Qwen tokenizer/template files.
"""

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "ARC3-Inference/runs/think-progressive-sol25"
QWEN = ROOT / "data/sft-gpt61sol-features-25games/qwen"
INDEX_CODE = ROOT / "data/sft-gpt61sol-features-25games/build_index.py"


def load_index_module():
    spec = importlib.util.spec_from_file_location("sft_index", INDEX_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text):
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def distribution(values):
    ordered = sorted(values)

    def percentile(p):
        at = (len(ordered) - 1) * p
        lo = int(at)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)

    return {
        "count": len(ordered), "sum": sum(ordered), "min": ordered[0],
        "p10": percentile(.10), "p25": percentile(.25),
        "median": percentile(.50), "mean": statistics.mean(ordered),
        "p75": percentile(.75), "p90": percentile(.90),
        "p95": percentile(.95), "max": ordered[-1],
    }


def audit_summary(finals):
    categories = Counter()
    flagged = 0
    available = 0
    for row in finals.values():
        verdict = row.get("final_judge")
        if verdict is None:
            continue
        available += 1
        issues = {
            "coverage": not all(verdict["words"]["covered"]) or bool(verdict["words"]["contradictions"]),
            "code_consistency": not verdict["code"]["leads_to_call"] or bool(verdict["code"]["disagreements"]),
            "fact_grounding": not verdict["fact"]["grounded"] or bool(verdict["fact"]["errors"]),
            "call_equivalence": "call" in verdict and not verdict["call"]["functionally_same"],
        }
        categories.update(name for name, issue in issues.items() if issue)
        flagged += any(issues.values())
    return {
        "requested": sum(bool(row["monitoring_sample"]) for row in finals.values()),
        "available": available,
        "unavailable": sum(row["monitoring_sample"] and row.get("final_judge") is None
                           for row in finals.values()),
        "clean": available - flagged,
        "flagged": flagged,
        "categories_overlap": dict(categories),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Write one length row per SFT sample")
    args = parser.parse_args()

    snapshot = json.loads((RUN / "snapshot.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert hashlib.sha256((QWEN / "tokenizer.json").read_bytes()).hexdigest() == manifest["tokenizer"]
    assert hashlib.sha256((QWEN / "chat_template.jinja").read_bytes()).hexdigest() == manifest["template"]
    finals = {}
    for path in sorted((RUN / "turns").glob("*/[0-9][0-9][0-9][0-9][0-9]/final.json")):
        row = json.loads(path.read_text())
        assert row["status"] == "ok" and row["key"] not in finals
        assert row["key"] == f"{row['game']}#{row['index']}"
        assert row["training_eligible"] == (row["student_input_tokens"] <= 120_000)
        finals[row["key"]] = row
    assert len(finals) == snapshot["finalized"] == 1334
    assert len({row["game"] for row in finals.values()}) == snapshot["games_complete"] == 25
    eligible = {key for key, row in finals.items() if row["training_eligible"]}
    excluded = set(finals) - eligible
    assert len(eligible) == snapshot["training_eligible"] == 1324
    assert excluded == set(snapshot["excluded_keys"])
    assert all(finals[key]["exclusion"] == "student_input_over_limit" for key in excluded)
    audits = audit_summary(finals)
    assert all(snapshot["audits"][key] == value for key, value in audits.items())

    index = load_index_module()
    template = index.compile_template((QWEN / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    rows = []
    seen = set()
    image_occurrences = 0
    text_only = 0
    for path in sorted((RUN / "sft").glob("*.jsonl")):
        for raw in path.open("rb"):
            sample = json.loads(raw)
            key = sample["id"]
            assert key in eligible and key not in seen
            seen.add(key)
            final = finals[key]
            game = sample["game"]
            assert path.stem == f"{game}_p0"
            assert key == f"{game}_p0#{sample['request_index']}"
            assert sample["loss_target_message_index"] == len(sample["messages"]) - 1
            assert sample["messages"][0]["role"] == "system"
            last = sample["messages"][-1]
            assert last["role"] == "assistant"
            thinking = last.get("reasoning_content", "")
            assert thinking and thinking == normalize(final["thinking"])
            assert sample["student_input_tokens"] == final["student_input_tokens"]
            assert sample["chat_template_kwargs"] == {
                "enable_thinking": True, "preserve_thinking": True,
            }
            assert len(sample["tools"]) == 1
            assert set(sample["tools"][0]["function"]["parameters"]["properties"]) == {"code"}
            calls = last.get("tool_calls") or []
            if calls:
                assert len(calls) == 1 and calls[0]["function"]["name"] == "python"
                assert set(calls[0]["function"]["arguments"]) == {"code"}
                code_chars = len(calls[0]["function"]["arguments"]["code"])
            else:
                text_only += 1
                code_chars = 0

            messages = sample["messages"]
            for message in messages[:-1]:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") != "image_url":
                        continue
                    url = part["image_url"]["url"]
                    assert url.startswith("data:image/png;base64,")
                    header = base64.b64decode(url.split(",", 1)[1][:64] + "===")
                    assert header[:8] == b"\x89PNG\r\n\x1a\n"
                    assert struct.unpack(">II", header[16:24]) == (640, 640)
                    image_occurrences += 1

            kwargs = sample["chat_template_kwargs"]
            prompt = index.render(template, messages[:-1], sample["tools"], kwargs, True)
            complete = index.render(template, messages, sample["tools"], kwargs)
            assert complete.startswith(prompt)
            images = index.count_images(messages[:-1])
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids) + images * 399
            assert input_tokens == sample["student_input_tokens"]
            output_tokens = len(tokenizer.encode(complete[len(prompt):], add_special_tokens=False).ids)
            rows.append({
                "id": key, "game": game, "request_index": sample["request_index"],
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "thinking_tokens_alone": len(tokenizer.encode(thinking, add_special_tokens=False).ids),
                "thinking_words": len(thinking.split()), "thinking_chars": len(thinking),
                "code_chars": code_chars, "images": images,
                "messages": len(messages), "json_bytes": len(raw),
                "level": index.parse_level(messages),
                "refinements": final["refinements"],
                "monitored": bool(final["monitoring_sample"]),
            })

    assert seen == eligible and len(rows) == 1324
    assert len(list((RUN / "sft").glob("*.jsonl"))) == 25
    assert image_occurrences == sum(row["images"] for row in rows)
    assert text_only == 1

    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game"]].append(row)
    game_stats = {}
    for game, group in sorted(by_game.items()):
        game_stats[game] = {
            "samples": len(group),
            "input_tokens": sum(row["input_tokens"] for row in group),
            "output_tokens": sum(row["output_tokens"] for row in group),
            "median_input": statistics.median(row["input_tokens"] for row in group),
            "median_output": statistics.median(row["output_tokens"] for row in group),
            "images": sum(row["images"] for row in group),
        }
    fields = [
        "input_tokens", "output_tokens", "total_tokens", "thinking_tokens_alone",
        "thinking_words", "thinking_chars", "code_chars", "images", "messages", "json_bytes",
    ]
    result = {
        "verified": True, "finalized": len(finals), "samples": len(rows),
        "games": len(by_game), "excluded": len(excluded),
        "excluded_input_tokens": distribution([finals[key]["student_input_tokens"] for key in excluded]),
        "all_finalized_input_tokens": distribution([row["student_input_tokens"] for row in finals.values()]),
        "sft_bytes": sum(path.stat().st_size for path in (RUN / "sft").glob("*.jsonl")),
        "images": image_occurrences, "text_only_targets": text_only,
        "level_counts": dict(sorted(Counter(row["level"] for row in rows).items())),
        "refinement_counts": dict(sorted(Counter(row["refinements"] for row in finals.values()).items())),
        "audits": snapshot["audits"],
        "distributions": {name: distribution([row[name] for row in rows]) for name in fields},
        "by_game": game_stats,
        "rows": rows,
    }
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
