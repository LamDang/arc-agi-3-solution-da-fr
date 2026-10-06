"""Check the token counter against a finished run: the conversation rebuilt at given turns, counted, and compared with
the prompt_tokens the server reported for the request made at that point (the next reply's usage).

    uv run --no-sync python -m engine_re.tools.count_check <copy of a game dir> --turns 10,20,30 [--tokenizer <dir|id>]

The game directory is copied first (the transcript is rewritten per turn), so point it at a copy, never at runs/. For a
run in the compact mode the request equals the conversation only before its first compaction (the `compact` records);
for a rebuilt run every request is the rebuilt view, which `--rebuilt` recomputes (with the run's own context mode).
Prints, per turn: the count, the reported prompt_tokens, the error, and the estimate at the calibrated divisor when
`--calibrated` (the fallback: the divisor driven by the previous turn's prompt_tokens, seed 3).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from engine_re.agent import CHARS_PER_TOKEN_SEED, Budget, EngineAgent, ModelConfig, estimate_request_tokens
from engine_re.tokens import load_counter


class _NeverCalled:
    def chat(self, messages, tools):  # noqa: ARG002
        raise AssertionError("the model is never called")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game_dir", type=Path, help="a copy of a game directory (transcript.jsonl, images/, trace/)")
    parser.add_argument("--turns", default="10,20,30", help="the turns at which the request is rebuilt and counted ('all')")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--game", default=None, help="the game id (default: the directory's name)")
    parser.add_argument("--environments-dir", type=Path, default=Path("environment_files"))
    parser.add_argument("--rebuilt", action="store_true", help="count the rebuilt view (context 'rebuilt') instead of the conversation")
    parser.add_argument("--calibrated", action="store_true", help="also the calibrated estimate (the fallback)")
    args = parser.parse_args()
    folder = args.game_dir.resolve()
    game = args.game or folder.name
    counter = load_counter(args.tokenizer)
    if counter is None:
        print("no tokenizer found", file=sys.stderr)
        return 1
    print(counter.describe())
    lines = [line for line in (folder / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    usage = {r["turn"]: r.get("usage") or {} for r in records if "finish_reason" in r}
    turns = sorted(usage) if args.turns == "all" else [int(t) for t in args.turns.split(",")]
    full = folder / "transcript.full.jsonl"
    if not full.exists():
        full.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = ModelConfig(context="rebuilt" if args.rebuilt else "compact")
    if (folder / "trace" / "trace.json").exists() and (folder / "artifacts").exists():
        from engine_re.play_agent import PlayAgent

        agent = PlayAgent(game, folder, config, Budget(), args.environments_dir, client=_NeverCalled())
    else:
        agent = EngineAgent(game, folder, config, Budget(), client=_NeverCalled(), stepwise=True)
    agent.counter = counter
    tools = agent._tools()
    chars_per_token = CHARS_PER_TOKEN_SEED
    errors, errors_estimate, errors_seed = [], [], []
    for t in turns:
        real = int(usage.get(t + 1, {}).get("prompt_tokens") or 0)
        if not real:
            continue
        (folder / "transcript.jsonl").write_text("\n".join(line for line, r in zip(lines, records) if int(r.get("turn") or 0) <= t) + "\n",
                                                 encoding="utf-8")
        state = agent._rebuild_conversation()
        messages = state["messages"]
        if args.rebuilt:
            view, _ = agent._rebuilt_view(messages)
        else:
            view = messages
        counted = counter.count(view, tools)
        error = (counted["tokens"] - real) / real
        errors.append(error)
        line = (f"turn {t:3d}: counted {counted['tokens']:6,} (text {counted['text_tokens']:,}, images {counted['image_tokens']}) "
                f"reported {real:6,}  error {100 * error:+.2f}%")
        if args.calibrated:
            seed = estimate_request_tokens(view, tools, CHARS_PER_TOKEN_SEED)["tokens"]
            calibrated = estimate_request_tokens(view, tools, chars_per_token)["tokens"]
            errors_seed.append(real / seed)
            errors_estimate.append(real / calibrated)
            line += f"  | estimate seed {seed:,} (real/est {real / seed:.3f}), calibrated at {chars_per_token:.2f}: {calibrated:,} ({real / calibrated:.3f})"
            agent.chars_per_token = chars_per_token
            agent._calibrate_from_usage(view, tools, {"prompt_tokens": real})
            chars_per_token = agent.chars_per_token
        print(line, flush=True)
    (folder / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        print(f"count error: median {100 * statistics.median(errors):+.2f}%, max |{100 * max(errors, key=abs):+.2f}%| over {len(errors)} turns")
    if errors_estimate:
        print(f"real/estimate: seed median {statistics.median(errors_seed):.3f} max {max(errors_seed):.3f}; "
              f"calibrated median {statistics.median(errors_estimate):.3f} max {max(errors_estimate):.3f} min {min(errors_estimate):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
