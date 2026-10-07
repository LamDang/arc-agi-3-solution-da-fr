"""Training volume of harness request logs: requests, replay sequences (stretches),
tokens to process once, generated tokens, and the sequence-length distribution.

    python trace_volume.py results/kaggle-20261006/runA/*_requests.jsonl
"""
import argparse
from pathlib import Path

from traces import samples_from_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--long", type=int, default=104858, help="'long' threshold (default 80%% of 131072)")
    args = parser.parse_args()

    n_requests = generated = billed_prompt = 0
    lengths = []
    for path in args.logs:
        samples, requests = samples_from_log(path, "x", path.name, 0)
        n_requests += len(requests)
        generated += sum(int(r.usage.get("completion_tokens") or 0) for r in requests)
        billed_prompt += sum(int(r.usage.get("prompt_tokens") or 0) for r in requests)
        lengths += [s.logged_prompt_tokens + s.logged_completion_tokens for s in samples]

    lengths.sort()
    total = sum(lengths)
    print(f"runs {len(args.logs)}  requests {n_requests}  sequences {len(lengths)}")
    print(f"tokens to process {total / 1e6:.2f}M  generated {generated / 1e6:.2f}M "
          f"({generated / max(total, 1):.0%})  billed prompt {billed_prompt / 1e6:.1f}M")
    if lengths:
        q = lambda p: lengths[int(p * (len(lengths) - 1))]
        print(f"sequence length p10 {q(.1)}  p50 {q(.5)}  p90 {q(.9)}  max {lengths[-1]}")
        print(f"share of tokens in sequences >= {args.long}: "
              f"{sum(x for x in lengths if x >= args.long) / total:.2f}")


if __name__ == "__main__":
    main()
