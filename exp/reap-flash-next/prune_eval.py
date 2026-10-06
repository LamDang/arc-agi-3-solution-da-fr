"""How much does pruning to N experts change the model on games it was not
calibrated on? One model load: calibrate, then replay held-out games with
pruned routers.

    python prune_eval.py --model-dir MODEL --traces kaggle_v3=DIR --out OUT \\
        --calib-games ar25,bp35 --calib-passes 0 --eval-games ls20,vc33 --eval-passes 1 \\
        [--stats-dir PREVIOUS_OUT ...] [--keep 448,384,320,256,192]

1. Calibration: the first stretch of each selected game run is replayed with
   the recorder on (truncated to --calib-max-tokens) and saved like run_reap.py
   does, under OUT/stats. --stats-dir adds statistics from earlier runs.
2. Evaluation: for each held-out sample, experts are ranked by REAP score
   from every calibration run of other games. The sample is replayed with the
   full model, then with each pruned router (softmax over kept experts only,
   as if the others were deleted). Reported per N: next-token top-1 and NLL
   on the logged generated tokens, and agreement with the full model's top-1.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

import analyze
import render
import replay as replay_mod
import run_reap
import traces
from run_reap import log


def first_stretches(specs, games, passes, source_filter=None) -> list[traces.Sample]:
    games = set(games.split(",")) if games else None
    passes = {int(p) for p in passes.split(",")} if passes else None
    samples = []
    for spec in specs:
        source, _, directory = spec.partition("=")
        if source_filter and source not in source_filter:
            continue
        found = traces.collect_samples(
            Path(directory), source,
            lambda g, p: (games is None or g in games) and (passes is None or p in passes))
        by_run = {}
        for s in found:
            if s.game != "unknown" and (games is None or s.game in games):
                by_run.setdefault(s.run_key, s)  # the first stretch of each run
        samples += by_run.values()
    return sorted(samples, key=lambda s: s.run_key)


def encode(processor, sample, max_tokens):
    tokenizer = processor.tokenizer
    messages, tools, _ = traces.load_messages(sample)
    enc = render.render(processor, messages, tools, add_generation_prompt=False)
    enc = replay_mod.truncate(enc, max_tokens, tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    cats = render.token_categories(enc["input_ids"], tokenizer, tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    return enc, cats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--traces", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-dir", action="append", default=[], help="earlier run_reap/prune_eval output")
    parser.add_argument("--calib-games", default="")
    parser.add_argument("--calib-passes", default="0")
    parser.add_argument("--calib-max-tokens", type=int, default=65536)
    parser.add_argument("--eval-games", required=True)
    parser.add_argument("--eval-passes", default="1")
    parser.add_argument("--eval-max-tokens", type=int, default=32768)
    parser.add_argument("--keep", default="448,384,320,288,256,192")
    parser.add_argument("--categories", default="context,generated,image", help="tokens used to rank experts")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=["einsum", "sdpa"], default="einsum")
    parser.add_argument("--compile-dequant", action="store_true", help="fused int4 dequantization (GPU)")
    parser.add_argument("--precache", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    calib = first_stretches(args.traces, args.calib_games, args.calib_passes) if args.calib_games else []
    evals = first_stretches(args.traces, args.eval_games, args.eval_passes)
    log(f"[plan] calibration: {[s.run_key for s in calib]}")
    log(f"[plan] evaluation: {[s.run_key for s in evals]}")

    from transformers import AutoProcessor

    import reap_model

    processor = AutoProcessor.from_pretrained(model_dir)
    if args.precache:
        threading.Thread(target=run_reap.precache, args=(run_reap.model_files(model_dir),), daemon=True).start()
    t = time.time()
    reap_model.OPTIONS.update(attention=args.attention, compile_dequant=args.compile_dequant)
    model, recorder = reap_model.load_model(model_dir, device=args.device, log=log)
    log(f"[load] {time.time() - t:.0f}s")
    if torch.cuda.is_available():
        reap_model.check_fast_linear_attention(args.device, log=log)
    n_experts = model.config.text_config.num_experts

    # 1. calibration
    for sample in calib:
        target = out / "stats" / f"{sample.run_key}.npz"
        if target.exists():
            continue
        recorder.reset()
        recorder.enabled = True
        enc, cats = encode(processor, sample, args.calib_max_tokens)
        t = time.time()
        result = replay_mod.replay(model, recorder, enc, cats, chunk_tokens=args.chunk, device=args.device)
        log(f"[calib] {sample.run_key}:{sample.stretch} {result['tokens']} tok in {time.time() - t:.0f}s, "
            f"top1 {result['correct'] / max(result['scored'], 1):.3f}")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **recorder.numpy())
        result.update(stretch=sample.stretch, category_tokens={
            n: int((cats == i).sum()) for i, n in enumerate(render.CATEGORY_NAMES)})
        target.with_suffix(".json").write_text(json.dumps({"run": sample.run_key, "samples": [result]}, indent=1))

    runs = analyze.load(out)
    for extra in args.stats_dir:
        runs.update(analyze.load(Path(extra)))
    log(f"[stats] {len(runs)} calibration runs: {sorted(runs)}")
    categories = [analyze.CATEGORIES.index(c) for c in args.categories.split(",")]
    keep_list = [int(n) for n in args.keep.split(",")]

    # 2. evaluation
    recorder.enabled = False
    report = {"calibration_runs": sorted(runs), "categories": args.categories, "samples": []}
    for sample in evals:
        train = [k for k in runs if analyze.game_of(k) != sample.game]
        if not train:
            log(f"[eval] {sample.run_key}: no calibration runs from other games, skipped")
            continue
        scores = analyze.reap_scores(analyze.aggregate(runs, train, categories))
        enc, cats = encode(processor, sample, args.eval_max_tokens)
        rows = []
        full_argmax = None
        for n in [n_experts] + keep_list:
            keep = None if n >= n_experts else torch.from_numpy(analyze.keep_mask(scores, n))
            reap_model.set_pruning(model, keep)
            t = time.time()
            r = replay_mod.replay(model, None, enc, cats, chunk_tokens=args.chunk, predictions=True,
                                  device=args.device)
            if full_argmax is None:
                full_argmax, full_nll = r["argmax"], r["token_nll"]
            row = {"keep": n, "top1": r["correct"] / max(r["scored"], 1), "nll": r["nll"] / max(r["scored"], 1),
                   "agree_with_full": float((r["argmax"] == full_argmax).float().mean()),
                   "nll_increase": float((r["token_nll"] - full_nll).mean()), "scored": r["scored"],
                   "seconds": time.time() - t}
            rows.append(row)
            log(f"[eval] {sample.run_key}:{sample.stretch} keep {n}: top1 {row['top1']:.4f} nll {row['nll']:.4f} "
                f"(+{row['nll_increase']:.4f}) agree {row['agree_with_full']:.4f} on {row['scored']} "
                f"in {row['seconds']:.0f}s")
        reap_model.set_pruning(model, None)
        report["samples"].append({"run": sample.run_key, "stretch": sample.stretch, "tokens": int(enc["input_ids"].shape[1]),
                                  "calibrated_on": train, "rows": rows})
        (out / "prune_eval.json").write_text(json.dumps(report, indent=1))

    if not report["samples"]:
        return
    print(f"\n{'keep':>5} {'top1':>7} {'nll':>7} {'+nll':>7} {'agree':>7}   (mean over {len(report['samples'])} held-out samples)")
    for i, n in enumerate([n_experts] + keep_list):
        rows = [s["rows"][i] for s in report["samples"]]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("top1", "nll", "nll_increase", "agree_with_full")}
        print(f"{n:>5} {mean['top1']:>7.4f} {mean['nll']:>7.4f} {mean['nll_increase']:>7.4f} {mean['agree_with_full']:>7.4f}")


if __name__ == "__main__":
    main()
