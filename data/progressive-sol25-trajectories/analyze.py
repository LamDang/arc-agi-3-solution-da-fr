"""Rebuild the trajectory exploration report with the pinned Qwen tokenizer.

Run from the repository root with ARC3-Inference/.venv/bin/python. The dataset
must have been generated with `dvc repro data/progressive-sol25-trajectories/dvc.yaml`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))
from think_gen.progressive import StudentFormat  # noqa: E402

HERE = Path(__file__).resolve().parent
QWEN = ROOT / "data/sft-gpt61sol-features-25games/qwen"
ROLE_ORDER = ("system_and_tools_schema", "user_text_and_format", "vision",
              "tool_results", "assistant_supervised", "assistant_masked")


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def number(value):
    return f"{value:,.0f}"


def counts_by_role(row, student):
    """Attribute each rendered token to its message span; count vision patches."""
    messages, tools = row["messages"], row["tools"]
    kwargs = row["chat_template_kwargs"]
    full = student.index.render(student.template, messages, tools, kwargs)
    encoded = student.tokenizer.encode(full, add_special_tokens=False)
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    first_user = full.index("<|im_start|>user\n")
    segments = [(0, first_user, "system_and_tools_schema", 0, 1)]
    cursor = first_user
    index = 1
    targets = set(row["loss_target_message_indices"])
    while index < len(messages):
        end_index = index + 1
        role = messages[index]["role"]
        # The chat template renders adjacent tool messages as one user block.
        if role == "tool":
            while end_index < len(messages) and messages[end_index]["role"] == "tool":
                end_index += 1
        prefix = student.index.render(student.template, messages[:end_index],
                                      tools, kwargs)
        if not full.startswith(prefix) or len(prefix) <= cursor:
            raise ValueError(f"Unstable render boundary: {row['trajectory_id']} at {index}")
        kind = {"user": "user_text_and_format", "tool": "tool_results"}.get(role)
        if role == "assistant":
            kind = "assistant_supervised" if index in targets else "assistant_masked"
        if kind is None:
            raise ValueError(f"Unexpected role {role}: {row['trajectory_id']}")
        segments.append((cursor, len(prefix), kind, index, end_index))
        cursor = len(prefix)
        index = end_index
    if cursor != len(full):
        raise ValueError(f"Unassigned rendered suffix: {row['trajectory_id']}")

    tokens = Counter()
    crossing = 0
    segment_index = 0
    for start, end in encoded.offsets:
        while segment_index + 1 < len(segments) and start >= segments[segment_index][1]:
            segment_index += 1
        boundary = segments[segment_index][1]
        if start < segments[segment_index][0] or end > boundary:
            crossing += 1
        tokens[segments[segment_index][2]] += 1
    images = student.index.count_images(messages)
    if images != row["image_count"]:
        raise ValueError(f"Image count mismatch: {row['trajectory_id']}")
    if any(student.index.count_images(messages[start:end]) for _, _, kind, start, end
           in segments if kind != "user_text_and_format"):
        raise ValueError(f"Image outside user message: {row['trajectory_id']}")
    tokens["user_text_and_format"] -= images  # replace each image pad placeholder
    tokens["vision"] += images * 400
    if sum(tokens.values()) != row["total_tokens"]:
        raise ValueError(f"Role tokens do not reconcile: {row['trajectory_id']}")
    return tokens, crossing


def distribution_table(rows, keys):
    lines = ["| Measure | Min | P25 | Median | P75 | P95 | Max |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, key in keys:
        values = [row[key] for row in rows]
        points = [min(values), quantile(values, .25), quantile(values, .5),
                  quantile(values, .75), quantile(values, .95), max(values)]
        lines.append(f"| {label} | " + " | ".join(number(value) for value in points) + " |")
    return lines


def bucket_table(values, bounds):
    result = []
    for lower, upper in zip(bounds, bounds[1:]):
        count = sum(lower <= value < upper for value in values)
        label = f"{number(lower)}–{number(upper - 1)}"
        result.append((label, count, 100 * count / len(values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "exploration.md")
    args = parser.parse_args()
    summary = json.loads((HERE / "summary.json").read_text())
    index = json.loads((HERE / "index.json").read_text())
    data_path = HERE / "trajectories.jsonl"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != summary["trajectories_sha256"]:
        raise ValueError("Dataset hash differs from summary")
    student = StudentFormat(QWEN / "tokenizer.json", QWEN / "chat_template.jinja")
    tokenizer_hash = hashlib.sha256((QWEN / "tokenizer.json").read_bytes()).hexdigest()
    template_hash = hashlib.sha256((QWEN / "chat_template.jinja").read_bytes()).hexdigest()
    trajectories = []
    with data_path.open("rb") as stream:
        for item in index:
            stream.seek(item["offset"])
            row = json.loads(stream.read(item["length"]))
            if row["trajectory_id"] != item["trajectory_id"]:
                raise ValueError("Index offset mismatch")
            trajectories.append(row)
    if len(trajectories) != summary["trajectories"] == 58:
        raise ValueError("Trajectory count mismatch")
    if any(row["tokenizer_sha256"] != tokenizer_hash or
           row["chat_template_sha256"] != template_hash
           for row in trajectories):
        raise ValueError("Pinned tokenizer or template changed")

    roles = Counter()
    crossings = 0
    for row in trajectories:
        counts, crossed = counts_by_role(row, student)
        roles.update(counts)
        crossings += crossed
    if crossings:
        raise ValueError(f"{crossings} token(s) cross message boundaries")
    if sum(roles.values()) != summary["total_sequence_tokens"]:
        raise ValueError("Aggregate token accounting mismatch")

    by_game = defaultdict(list)
    for row in trajectories:
        by_game[row["game"]].append(row)
    initial = [row for row in trajectories if row["chunk"] == 0]
    resumed = [row for row in trajectories if row["chunk"] > 0]
    all_turns = [turn for row in trajectories for turn in row["turns"]]
    if len(all_turns) != summary["supervised_targets"] == 1334:
        raise ValueError("Target count mismatch")
    generation_prefix_tokens = roles["assistant_supervised"] - summary["total_output_tokens"]
    if generation_prefix_tokens != 5 * len(all_turns):
        raise ValueError("Assistant span/output accounting changed")
    if sum(row["ends_with_compaction_note"] for row in trajectories) != 33:
        raise ValueError("Compaction count mismatch")
    first_prompt = lambda rows: quantile([row["turns"][0]["input_tokens"]
                                          for row in rows], .5)
    ratio = lambda row: row["output_tokens"] / row["input_tokens"]
    thinking = [len(student.tokenizer.encode(row["messages"][at].get(
                    "reasoning_content", ""), add_special_tokens=False).ids)
                for row in trajectories for at in row["loss_target_message_indices"]]
    tool_messages = [message for row in trajectories for message in row["messages"]
                     if message["role"] == "tool"]
    tool_content = [len(student.tokenizer.encode(str(message.get("content") or ""),
                                                 add_special_tokens=False).ids)
                    for message in tool_messages]
    max_row = max(trajectories, key=lambda row: row["total_tokens"])
    max_turns = max(trajectories, key=lambda row: row["turn_count"])
    game_response_counts = sorted((sum(row["turn_count"] for row in group)
                                   for group in by_game.values()), reverse=True)

    lines = [
        "# Sol25 trajectory dataset: exploratory analysis", "",
        "Generated by `analyze.py` from the DVC trajectory output using the pinned Qwen",
        "tokenizer and chat template. All 25 games are included. Counts are model",
        "processor tokens, with 400 tokens per 640×640 board image; they are not",
        "provider billing tokens.", "",
        "## Coverage and composition", "",
        f"- **{len(trajectories)} trajectories** across **{len(by_game)} games**: "
        f"{len(initial)} initial and {len(resumed)} after compaction.",
        f"- **{len(all_turns):,} supervised assistant responses**; "
        f"{sum(row['ends_with_compaction_note'] for row in trajectories)} compaction boundaries.",
        f"- **{sum(row['message_count'] for row in trajectories):,} messages** "
        f"(`system` {sum(1 for r in trajectories for m in r['messages'] if m['role']=='system'):,}, "
        f"`user` {sum(1 for r in trajectories for m in r['messages'] if m['role']=='user'):,}, "
        f"`assistant` {sum(1 for r in trajectories for m in r['messages'] if m['role']=='assistant'):,}, "
        f"`tool` {len(tool_messages):,}).",
        f"- **{sum(row['image_count'] for row in trajectories):,} image occurrences** "
        "in the grouped sequences; retained images recur after compaction.",
        f"- Largest sequence: `{max_row['trajectory_id']}` at "
        f"**{max_row['total_tokens']:,} / 130,000 tokens**. Most responses in one "
        f"trajectory: **{max_turns['turn_count']}** (`{max_turns['trajectory_id']}`).",
        "", "## Length distributions", "",
        "Each trajectory total is the final request prompt plus its final assistant",
        "response. `output_tokens` sums only newly supervised assistant responses;",
        "`input_tokens = total_tokens - output_tokens` includes masked inherited history.",
        f"Across all trajectories: **{sum(r['input_tokens'] for r in trajectories):,} input** + "
        f"**{sum(r['output_tokens'] for r in trajectories):,} supervised output** = "
        f"**{sum(r['total_tokens'] for r in trajectories):,} total tokens**.",
        "", *distribution_table(trajectories, [
            ("Total tokens", "total_tokens"), ("Input tokens", "input_tokens"),
            ("Supervised output tokens", "output_tokens"),
            ("New assistant responses", "turn_count"),
            ("Messages", "message_count"), ("Images", "image_count")]),
        "", "### Total-token bands", "",
        "| Total tokens | Trajectories | Share |", "| --- | ---: | ---: |",
    ]
    for label, count, share in bucket_table(
            [row["total_tokens"] for row in trajectories],
            [0, 40_000, 60_000, 80_000, 100_000, 110_000, 120_000, 130_001]):
        lines.append(f"| {label} | {count} | {share:.1f}% |")
    lines += ["", "### New-response bands", "",
              "| New responses | Trajectories | Share |", "| --- | ---: | ---: |"]
    for label, count, share in bucket_table(
            [row["turn_count"] for row in trajectories], [1, 6, 11, 21, 31, 41, 51]):
        lines.append(f"| {label} | {count} | {share:.1f}% |")
    lines += ["", "### Per-response target lengths", "",
              "These 1,334 assistant responses are the distinct loss targets, counted",
              "once each across the 58 trajectories.", "",
              *distribution_table(all_turns, [
                  ("Output tokens per response", "output_tokens"),
                  ("Prompt tokens before response", "input_tokens")])]

    total = sum(roles.values())
    lines += ["", "## Tokens by message type", "",
              "The pinned template is rendered once per trajectory. Text tokens are",
              "assigned to the message span that emitted them; adjacent tool results",
              "form one template block. Each image pad is replaced by its 400 vision",
              "tokens. Template markers are counted with their emitting message.",
              "The totals reconcile exactly with the 58 stored sequence lengths.",
              "", "| Type | Tokens | Share of sequence tokens |",
              "| --- | ---: | ---: |"]
    labels = {"system_and_tools_schema": "System prompt + tool schema",
              "user_text_and_format": "User text + formatting",
              "vision": "Board image vision tokens",
              "tool_results": "Tool results + formatting",
              "assistant_supervised": "Assistant, newly supervised",
              "assistant_masked": "Assistant, retained and masked"}
    for key in ROLE_ORDER:
        lines.append(f"| {labels[key]} | {roles[key]:,} | {100*roles[key]/total:.1f}% |")
    lines.append(f"| **Total** | **{total:,}** | **100.0%** |")
    lines += ["", f"The {len(thinking):,} supervised `reasoning_content` fields have a",
              f"median of **{number(quantile(thinking, .5))}** tokens and a P95 of",
              f"**{number(quantile(thinking, .95))}** when tokenized alone; their",
              "standalone counts are diagnostic and are not additive with the rendered",
              "role totals. Tool-result content alone has a median of",
              f"**{number(quantile(tool_content, .5))}** tokens and P95 of",
              f"**{number(quantile(tool_content, .95))}** tokens.",
              "", "## Initial versus post-compaction", "",
              "The after-compaction sequence starts with the same system prompt and",
              "the actual retained suffix of ten game turns, including earlier assistant",
              "responses. Those repeated responses are masked. Consequently, the first",
              "new response is conditioned on much more context, while fewer new",
              "responses are supervised in each later trajectory.", "",
              "| Group | Trajectories | Median first prompt | Median new responses | Median input | Median output | Median output/input |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, group in (("Initial", initial), ("After compaction", resumed)):
        lines.append(f"| {label} | {len(group)} | {number(first_prompt(group))} | "
                     f"{number(quantile([r['turn_count'] for r in group], .5))} | "
                     f"{number(quantile([r['input_tokens'] for r in group], .5))} | "
                     f"{number(quantile([r['output_tokens'] for r in group], .5))} | "
                     f"{quantile([ratio(r) for r in group], .5):.3f} |")
    lines += ["", f"Across all trajectories, the median output/input ratio is "
              f"**{quantile([ratio(r) for r in trajectories], .5):.3f}**. The "
              f"token-weighted ratio is **{sum(r['output_tokens'] for r in trajectories) / sum(r['input_tokens'] for r in trajectories):.3f}**. "
              "New output per response is actually higher after compaction:",
              f"**{sum(r['output_tokens'] for r in resumed) / sum(r['turn_count'] for r in resumed):,.0f}** "
              "versus "
              f"**{sum(r['output_tokens'] for r in initial) / sum(r['turn_count'] for r in initial):,.0f}** "
              "tokens. The ratio difference reflects the larger retained prompt and",
              "fewer new responses, not shorter responses.",
              "", "## Per-game coverage", "",
              "Sums below are over exported trajectories. Retained context is counted",
              "again when a game crosses a compaction boundary; each supervised",
              "assistant response is counted once. Games range from",
              f"**{min(game_response_counts)} to {max(game_response_counts)}** new responses; "
              f"the five longest supply **{sum(game_response_counts[:5])} / {len(all_turns)} "
              f"({100 * sum(game_response_counts[:5]) / len(all_turns):.1f}%)** of targets.", "",
              "| Game | Trajectories | Compactions | New responses | Median trajectory tokens | Max tokens | Input tokens | Supervised output tokens | Output/input |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for game, group in sorted(by_game.items()):
        input_tokens = sum(row["input_tokens"] for row in group)
        output_tokens = sum(row["output_tokens"] for row in group)
        lines.append(f"| `{game}` | {len(group)} | {len(group)-1} | "
                     f"{sum(row['turn_count'] for row in group)} | "
                     f"{number(quantile([row['total_tokens'] for row in group], .5))} | "
                     f"{max(row['total_tokens'] for row in group):,} | "
                     f"{input_tokens:,} | {output_tokens:,} | "
                     f"{output_tokens / input_tokens:.3f} |")
    lines += ["", "## Reading the counts", "",
              "- The token-by-type table describes actual rendered sequence tokens and",
              "  counts repeated post-compaction context each time it is present.",
              "- The dataset's `output_tokens` excludes each target's five-token",
              "  assistant generation prefix, which belongs to its prompt. The role table",
              f"  includes those {generation_prefix_tokens:,} prefix tokens in the assistant",
              "  spans. Thus rendered supervised-assistant tokens exceed the sum of",
              "  supervised output tokens by exactly 5 × 1,334; both methods give the",
              "  same trajectory total.",
              "- Per-game totals are useful for storage and training mix, but do not",
              "  represent independent observations; 25 games generated 58 overlapping",
              "  context windows. Audit flags remain as recorded in source checkpoints.",
              ""]
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}: {len(trajectories)} trajectories, {len(all_turns)} targets, "
          f"{total:,} sequence tokens; {crossings} boundary-crossing text tokens")


if __name__ == "__main__":
    main()
