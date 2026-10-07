"""Time the replay with each speed option, check they agree, profile one chunk.

    python bench.py --model-dir MODEL --traces kaggle_v3=DIR --sample kaggle_v3/ls20_p0:0 \\
        [--max-tokens 16384] [--variants base,compile,sdpa,both] [--profile]

Each variant replays the same truncated sample twice (the first run absorbs
compilation and warm-up) and reports tokens/s, next-token top-1 and NLL,
agreement of top-1 predictions with the first variant, and the expert
statistics' largest relative difference from it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import prune_eval
import reap_model
import replay as replay_mod
import run_reap
import traces
from run_reap import log

VARIANTS = {
    "base": {"compile_dequant": False, "attention": "einsum"},
    "compile": {"compile_dequant": True, "attention": "einsum"},
    "sdpa": {"compile_dequant": False, "attention": "sdpa"},
    "both": {"compile_dequant": True, "attention": "sdpa"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--traces", action="append", required=True)
    parser.add_argument("--sample", required=True, help="SOURCE/GAME_pPASS:STRETCH")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--variants", default="base,compile,sdpa,both")
    parser.add_argument("--profile", action="store_true", help="torch.profiler table for the first variant")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run, _, stretch = args.sample.partition(":")
    source, _, game_pass = run.partition("/")
    game, _, pass_ = game_pass.rpartition("_p")
    directory = next(d for s, _, d in (t.partition("=") for t in args.traces) if s == source)
    sample = next(s for s in traces.collect_samples(Path(directory), source, lambda g, p: g == game and p == int(pass_))
                  if s.stretch == int(stretch))
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model, recorder = reap_model.load_model(args.model_dir, device=args.device, log=log)
    reap_model.check_fast_linear_attention(args.device, log=log)
    enc, cats = prune_eval.encode(processor, sample, args.max_tokens)
    log(f"[bench] {args.sample}: {enc['input_ids'].shape[1]} tokens")

    reference = None
    for name in args.variants.split(","):
        reap_model.OPTIONS.update(VARIANTS[name])
        for attempt in range(2):
            recorder.reset()
            torch.cuda.synchronize()
            t = time.time()
            r = replay_mod.replay(model, recorder, enc, cats, chunk_tokens=args.chunk, predictions=True,
                                  device=args.device)
            torch.cuda.synchronize()
            seconds = time.time() - t
        stats = {k: v.clone() for k, v in recorder.data.items()}
        line = (f"[bench] {name:8s} {r['tokens'] / seconds:7.0f} tok/s ({seconds:.1f}s) top1 "
                f"{r['correct'] / r['scored']:.4f} nll {r['nll'] / r['scored']:.4f}")
        if reference is None:
            reference = (r, stats)
        else:
            agree = (r["argmax"] == reference[0]["argmax"]).float().mean().item()
            rel = max(((stats[k] - reference[1][k]).abs().max() / reference[1][k].abs().max().clamp_min(1e-12)).item()
                      for k in ("gate_norm", "norm", "gate"))
            line += f" | agree {agree:.4f}, stats max rel diff {rel:.1e}"
        log(line)

    if args.profile:
        reap_model.OPTIONS.update(VARIANTS[args.variants.split(",")[0]])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            replay_mod.replay(model, recorder, enc, cats, chunk_tokens=args.chunk, device=args.device)
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=60)
        print(table, flush=True)
        reap_rows = [e for e in prof.key_averages() if e.key.startswith("reap.")]
        for e in sorted(reap_rows, key=lambda e: -e.cpu_time_total):
            print(f"{e.key:20s} cpu {e.cpu_time_total / 1e6:7.2f}s  cuda {getattr(e, 'device_time_total', 0) / 1e6:7.2f}s"
                  f"  calls {e.count}", flush=True)


if __name__ == "__main__":
    main()
