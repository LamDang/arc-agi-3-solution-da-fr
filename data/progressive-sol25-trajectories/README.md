# Progressive thinking trajectories (25 games)

This dataset merges the finalized generated-thinking responses from all 25
games into **58 trajectories**. Each line of `trajectories.jsonl` is one
contiguous model-visible conversation. The old per-response filter used a
120,000-token **input** limit; this export uses a **130,000-token input + output
limit** for every complete trajectory and every constituent response. All
1,334 finalized assistant responses fit, including the ten omitted by the old
input-only filter. The largest trajectory is 129,169 tokens.

## Files

- `trajectories.jsonl` is the dataset (27.5 MB, a DVC pipeline output). One UTF-8 JSON
  object per trajectory, with embedded board images.
- `index.json` lists byte offsets and the main metadata for random access and
  length bucketing, without reading the image payloads.
- `summary.json` records counts and source/data hashes.
- `ratios.csv` lists input tokens, supervised output tokens, and their ratio
  for every trajectory.
- `dvc.yaml` and `dvc.lock` define and pin the reproducible export. `build.py`
  implements the stage.

## Reproduce

From the repository root, install the `ARC3-Inference` Python environment,
pull the two archived source runs, and run the stage:

```sh
dvc pull ARC3-Inference/runs/gpt61sol-features-25games.dvc ARC3-Inference/runs/think-progressive-sol25.dvc
dvc repro data/progressive-sol25-trajectories/dvc.yaml
```

The stage depends on the pinned Qwen tokenizer and chat template supplied by
the existing `data/sft-gpt61sol-features-25games` DVC pipeline. DVC fetches
them through that upstream stage if they are absent. The build checks their
SHA-256 hashes, the source log hashes, the 1,324 previously exported SFT
samples, all 33 compaction boundaries, and the 130,000-token cap. The four
outputs are DVC cached locally, and the lockfile pins their hashes. Until
their cache objects are pushed to the DVC remote, run `dvc repro` to generate
them from the archived source runs.

## Row schema

`messages` is an ordered list of chat messages. Every message has a `role`:
`system`, `user`, `assistant`, or `tool`. Assistant messages retain
`reasoning_content`, visible `content`, and `tool_calls`; tool results have
their own `tool` role and `tool_call_id`. The image parts remain embedded in
user messages. `tools` and `chat_template_kwargs` give the Qwen rendering
configuration.

The `turns` list maps each finalized response to its `request_index`, its
`assistant_message_index` in `messages`, and its input, output, and total token
counts. **Only** the assistant message indices in
`loss_target_message_indices` are supervised. All other content is context,
including retained assistant responses repeated after compaction and the
compaction answer repeated in the next trajectory. A loader must apply that
mask when rendering a row; training the whole rendered string without a mask
would train on repeated history.

The row metadata includes `game`, `game_run`, `pass`, `chunk`, `start_turn`,
`end_turn`, `turn_count`, `message_count`, `image_count`, `input_tokens`,
`output_tokens`, `total_tokens`, `terminal_prompt_tokens`,
`terminal_output_tokens`, `max_total_tokens`, and compaction boundary details.
`start_turn` and `end_turn` are inclusive, zero-based finalized response
indices in the source game run. `output_tokens` sums the separately rendered
supervised assistant targets. `input_tokens = total_tokens - output_tokens`;
it includes the system prompt, user/tool messages, and masked inherited
history. `total_tokens` is the final request's Qwen prompt count plus its
assistant output, including 400 tokens for each 640×640 board image. Token
counts use the pinned tokenizer/template hashes stored in each row.

At a context compaction, the earlier trajectory ends with the notice and
assistant note. The next starts with the same **system prompt**, followed by
the actual retained suffix of ten **game turns**, the same notice and note,
the note tool result, and then continues.
The retained suffix can contain more than ten assistant responses because a
game turn may involve several model requests. `retained_from_previous` and
`boundary_to_next` locate these pieces. No synthetic messages were added.

## Why output/input ratio falls after compaction

For this dataset, `output_tokens / input_tokens` counts only assistant
responses newly supervised in that trajectory as output. Input includes the
system prompt, user and tool messages, plus retained assistant responses and
the compaction note that are **masked** to avoid duplicate supervision.

| Group | Trajectories | Median first-response prompt | Median new responses | Median output/input |
| --- | ---: | ---: | ---: | ---: |
| Initial | 25 | 5,529 tokens | 35 | 0.402 |
| After compaction | 33 | 47,040 tokens | 16 | 0.270 |

The post-compaction first-response prompt is larger because it contains the
retained ten game turns. These trajectories also contain fewer new responses.
Consequently, their total input is similar to the initial group's while
their supervised output is smaller. The ratio difference does not mean each
response is shorter: aggregate output per new response is about 1,262 tokens
after compaction versus 918 tokens initially. Across all 58 trajectories,
the median ratio is 0.324 and the token-weighted ratio is 0.325.

## Example access

```python
import json
from pathlib import Path

folder = Path("data/progressive-sol25-trajectories")
row = json.loads((folder / "index.json").read_text())[0]
with (folder / "trajectories.jsonl").open("rb") as stream:
    stream.seek(row["offset"])
    trajectory = json.loads(stream.read(row["length"]))
assert trajectory["trajectory_id"] == row["trajectory_id"]
targets = set(trajectory["loss_target_message_indices"])
for index, message in enumerate(trajectory["messages"]):
    train_on_this_message = index in targets
```

The generated thinking may still contain reasoning errors; this export
changes grouping and eligibility, not the content or audit judgments.
