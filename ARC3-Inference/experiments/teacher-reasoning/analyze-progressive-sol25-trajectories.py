"""Count mergeable context stretches and verify all note-compaction boundaries.

Run from the repository root with ARC3-Inference/.venv/bin/python. Requires the
DVC checkout and local pinned Qwen tokenizer/template files.
"""

import importlib.util
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "ARC3-Inference/runs/think-progressive-sol25"
QWEN = ROOT / "data/sft-gpt61sol-features-25games/qwen"


def render_tools():
    path = ROOT / "data/sft-gpt61sol-features-25games/build_index.py"
    spec = importlib.util.spec_from_file_location("sft_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.compile_template((QWEN / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    return module, template, tokenizer


def main():
    finals = defaultdict(dict)
    for path in (RUN / "turns").glob("*/[0-9][0-9][0-9][0-9][0-9]/final.json"):
        row = json.loads(path.read_text())
        finals[row["game"]][row["index"]] = row
    snapshot = json.loads((RUN / "snapshot.json").read_text())
    assert len(finals) == snapshot["games_complete"] == 25
    assert sum(map(len, finals.values())) == snapshot["finalized"] == 1334

    starts = {}
    sizes = {}
    for game, turns in finals.items():
        assert sorted(turns) == list(range(len(turns)))
        starts[game] = [i for i in sorted(turns)
                        if i == 0 or turns[i]["chunk"] != turns[i - 1]["chunk"]]
        sizes[game] = [sum(row["chunk"] == chunk for row in turns.values())
                       for chunk in sorted({row["chunk"] for row in turns.values()})]
        assert len(starts[game]) == len(sizes[game])

    first_after = {game: {start: None for start in boundaries[1:]}
                   for game, boundaries in starts.items()}
    prefix_checks = 0
    switches = 0
    training_counts = Counter()
    eligible_ids = set()
    for path in sorted((RUN / "sft").glob("*.jsonl")):
        game = path.stem
        previous = None
        for line in path.open():
            sample = json.loads(line)
            index = sample["request_index"]
            row = finals[game][index]
            assert row["training_eligible"] and sample["id"] == row["key"]
            eligible_ids.add(sample["id"])
            training_counts[(game, row["chunk"])] += 1
            if previous is not None:
                prev_chunk = finals[game][previous["request_index"]]["chunk"]
                if row["chunk"] == prev_chunk:
                    prefix_checks += 1
                    assert sample["messages"][:len(previous["messages"])] == previous["messages"]
                else:
                    switches += 1
            previous = sample
            for start in first_after[game]:
                if first_after[game][start] is None and index >= start \
                        and row["chunk"] == finals[game][start]["chunk"]:
                    first_after[game][start] = sample

    assert len(eligible_ids) == snapshot["training_eligible"] == 1324
    assert len(training_counts) == sum(map(len, starts.values())) == 58
    index, template, tokenizer = render_tools()
    with (Path(__file__).parent / "progressive-sol25-lengths.csv").open() as stream:
        exported_output_tokens = {row["id"]: int(row["output_tokens"])
                                  for row in csv.DictReader(stream)}
    boundaries = []
    for game, selected in first_after.items():
        for start, sample in selected.items():
            assert sample is not None
            messages = sample["messages"]
            positions = [i for i, message in enumerate(messages)
                         if message["role"] == "user"
                         and isinstance(message.get("content"), str)
                         and message["content"].startswith("Context notice:")]
            assert len(positions) == 1
            note_at = positions[0]
            prompt = messages[note_at]["content"]
            keep = int(re.search(r"last (\d+) turns", prompt).group(1))
            assert messages[note_at + 1]["role"] == "assistant"
            assert messages[note_at + 2]["role"] == "tool"
            assert messages[note_at + 2]["content"].startswith("Note kept in your context")
            prior = finals[game][start - 1]
            kwargs = sample["chat_template_kwargs"]
            prompt = index.render(template, messages[:note_at + 1], sample["tools"], kwargs, True)
            through_note = index.render(template, messages[:note_at + 2], sample["tools"], kwargs)
            through_tool = index.render(template, messages[:note_at + 3], sample["tools"], kwargs)
            assert through_note.startswith(prompt) and through_tool.startswith(prompt)
            output_tokens = len(tokenizer.encode(through_note[len(prompt):],
                                                 add_special_tokens=False).ids)
            output_and_tool_tokens = len(tokenizer.encode(through_tool[len(prompt):],
                                                          add_special_tokens=False).ids)
            if prior["key"] in eligible_ids:
                assert output_tokens == exported_output_tokens[prior["key"]]
            boundaries.append({
                "game": game, "next_request": start, "note_request": start - 1,
                "keep_game_turns": keep,
                "retained_assistant_responses": sum(
                    message["role"] == "assistant" for message in messages[1:note_at]),
                "note_target_eligible": prior["key"] in eligible_ids,
                "input_tokens": prior["student_input_tokens"],
                "note_output_tokens": output_tokens,
                "input_plus_note_tokens": prior["student_input_tokens"] + output_tokens,
                "input_plus_note_and_tool_tokens": prior["student_input_tokens"] + output_and_tool_tokens,
            })

    assert len(boundaries) == switches == 33
    assert prefix_checks == 1266
    assert all(boundary["keep_game_turns"] == 10 for boundary in boundaries)
    chunk_lengths = [length for game in sizes for length in sizes[game]]
    last_prompts = [max((row for row in turns.values() if row["chunk"] == chunk),
                        key=lambda row: row["index"])["student_input_tokens"]
                    for turns in finals.values()
                    for chunk in {row["chunk"] for row in turns.values()}]
    result = {
        "games": len(finals), "finalized_turns": sum(map(len, finals.values())),
        "training_samples": len(eligible_ids), "trajectories": len(chunk_lengths),
        "compactions": len(boundaries), "same_chunk_prefix_checks": prefix_checks,
        "training_targets_per_trajectory": {
            "min": min(training_counts.values()),
            "median": statistics.median(training_counts.values()),
            "max": max(training_counts.values()),
        },
        "finalized_responses_per_trajectory": {
            "min": min(chunk_lengths), "median": statistics.median(chunk_lengths),
            "max": max(chunk_lengths),
        },
        "retained_game_turns": dict(Counter(b["keep_game_turns"] for b in boundaries)),
        "retained_assistant_responses": {
            "min": min(b["retained_assistant_responses"] for b in boundaries),
            "median": statistics.median(b["retained_assistant_responses"] for b in boundaries),
            "max": max(b["retained_assistant_responses"] for b in boundaries),
        },
        "note_targets_eligible": sum(b["note_target_eligible"] for b in boundaries),
        "note_targets_excluded": sum(not b["note_target_eligible"] for b in boundaries),
        "note_trajectories_over_120k_total": sum(b["input_plus_note_tokens"] > 120_000
                                                 for b in boundaries),
        "max_note_trajectory_tokens": max(b["input_plus_note_tokens"] for b in boundaries),
        "max_note_trajectory_with_tool_tokens": max(
            b["input_plus_note_and_tool_tokens"] for b in boundaries),
        "terminal_prompt_tokens_sum": sum(last_prompts),
        "terminal_prompt_tokens_max": max(last_prompts),
        "by_game": {game: {"trajectories": len(sizes[game]), "lengths": sizes[game]}
                    for game in sorted(sizes)},
        "boundaries": sorted(boundaries, key=lambda b: (b["game"], b["next_request"])),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
