"""Replay harness request logs through Flash-Next and save REAP statistics.

    python run_reap.py --model-dir MODEL --traces kaggle=DIR [--traces openrouter=DIR2] --out OUT

Statistics are saved per game run (source, game, pass) as
OUT/stats/<source>/<game>_p<pass>.npz, with a .json of per-sample results
next to it. Finished runs are skipped, so an interrupted job resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render  # noqa: E402
import replay as replay_mod  # noqa: E402
import traces  # noqa: E402


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def precache(paths: list[Path], threads: int = 16, block: int = 32 << 20):
    """Read files once, in parallel chunks, so later reads hit the page cache
    (Kaggle's model mount reads faster with many readers)."""
    from concurrent.futures import ThreadPoolExecutor

    start = time.time()
    jobs = [(p, o) for p in paths for o in range(0, max(p.stat().st_size, 1), block)]

    def read(job):
        path, offset = job
        with open(path, "rb", buffering=0) as fh:
            fh.seek(offset)
            return len(fh.read(block))

    with ThreadPoolExecutor(threads) as pool:
        total = sum(pool.map(read, jobs))
    log(f"[precache] {len(paths)} files, {total / 1e9:.1f} GB in {time.time() - start:.0f}s")


def model_files(model_dir: Path) -> list[Path]:
    """Weights first (read once by the loader), then the n-gram table."""
    main = sorted(p for p in model_dir.glob("model-*.safetensors") if not p.name.startswith("model-ple-"))
    return main + sorted(model_dir.glob("model-ple-*.safetensors"))


def plan(args) -> list[traces.Sample]:
    games = set(args.games.split(",")) if args.games else None
    passes = {int(p) for p in args.passes.split(",")} if args.passes else None
    wanted_runs = None
    if args.samples:
        wanted_runs = {item.split(":")[0] for item in args.samples.split(",")}
    samples = []
    for spec in args.traces:
        source, _, directory = spec.partition("=")

        def keep(game, pass_, source=source):
            return ((games is None or game in games) and (passes is None or pass_ in passes)
                    and (wanted_runs is None or f"{source}/{game}_p{pass_}" in wanted_runs))

        if wanted_runs is not None and not any(r.startswith(source + "/") for r in wanted_runs):
            continue
        found = traces.collect_samples(Path(directory), source, keep)
        log(f"[plan] {source}: {len(found)} samples from {directory}")
        samples += found
    if games is not None:
        samples = [s for s in samples if s.game in games]
    if passes is not None:
        samples = [s for s in samples if s.pass_ in passes]
    if args.samples:
        wanted = set(args.samples.split(","))
        samples = [s for s in samples if f"{s.run_key}:{s.stretch}" in wanted]
    samples.sort(key=lambda s: (s.source, s.game, s.pass_, str(s.path), s.stretch))
    if args.max_samples:
        samples = samples[: args.max_samples]
    return samples


def gpu_gb(fn) -> float:
    return fn() / 1e9 if torch.cuda.is_available() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--traces", action="append", required=True, help="SOURCE=DIR, repeatable")
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", help="comma-separated game prefixes, e.g. ft09,ls20")
    parser.add_argument("--passes", help="comma-separated pass numbers")
    parser.add_argument("--samples", help="comma-separated SOURCE/GAME_pPASS:STRETCH")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0, help="truncate each sample (0: whole)")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-measure", action="store_true", help="skip the next-token check")
    parser.add_argument("--precache", action="store_true", help="warm the page cache with the n-gram table")
    parser.add_argument("--deadline-minutes", type=float, default=0, help="stop starting new runs after this")
    parser.add_argument("--plan-only", action="store_true", help="write plan.json and stop")
    parser.add_argument("--dry-run", action="store_true", help="plan and render only")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = plan(args)
    by_run = defaultdict(list)
    for sample in samples:
        by_run[sample.run_key].append(sample)
    (out / "plan.json").write_text(json.dumps([
        {"run": s.run_key, "stretch": s.stretch, "requests": len(s.requests), "path": str(s.path),
         "logged_prompt_tokens": s.logged_prompt_tokens, "has_reply": bool(s.reply or s.reply_from)}
        for s in samples
    ], indent=1))
    log(f"[plan] {len(samples)} samples in {len(by_run)} game runs, "
        f"~{sum(s.logged_prompt_tokens for s in samples) / 1e6:.2f}M logged prompt tokens")
    if args.plan_only:
        return

    model_dir = Path(args.model_dir)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")

    if args.dry_run:
        for sample in samples:
            messages, tools, kwargs = traces.load_messages(sample)
            enc = render.render(processor, messages, tools, add_generation_prompt=False)
            log(f"[dry] {sample.run_key}:{sample.stretch} {enc['input_ids'].shape[1]} tokens "
                f"(logged prompt {sample.logged_prompt_tokens})")
        return

    if args.precache:
        threading.Thread(target=precache, args=(model_files(model_dir),), daemon=True).start()

    import reap_model

    t = time.time()
    model, recorder = reap_model.load_model(model_dir, device=args.device, log=log)
    log(f"[load] {time.time() - t:.0f}s, GPU {gpu_gb(torch.cuda.memory_allocated):.1f} GB allocated")
    if torch.cuda.is_available():
        reap_model.check_fast_linear_attention(args.device, log=log)
    meta_common = {"model_dir": str(model_dir), "chunk": args.chunk, "max_tokens": args.max_tokens,
                   "fields": list(recorder.FIELDS), "categories": list(render.CATEGORY_NAMES)}

    totals = defaultdict(float)
    for run_key, run_samples in by_run.items():
        target = out / "stats" / f"{run_key}.npz"
        if target.exists():
            log(f"[skip] {run_key} done")
            continue
        if args.deadline_minutes and (time.time() - started) / 60 > args.deadline_minutes:
            log(f"[stop] deadline reached before {run_key}")
            break
        recorder.reset()
        results = []
        for sample in run_samples:
            t = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            messages, tools, kwargs = traces.load_messages(sample)
            enc = render.render(processor, messages, tools, add_generation_prompt=False)
            enc = replay_mod.truncate(enc, args.max_tokens, vision_start)
            cats = render.token_categories(enc["input_ids"], tokenizer, image_token)
            render_s = time.time() - t
            result = replay_mod.replay(model, recorder, enc, cats, chunk_tokens=args.chunk,
                                       measure=not args.no_measure, device=args.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.time() - t
            result.update(
                stretch=sample.stretch, requests=len(sample.requests), seconds=seconds, render_seconds=render_s,
                logged_prompt_tokens=sample.logged_prompt_tokens,
                category_tokens={name: int((cats == i).sum()) for i, name in enumerate(render.CATEGORY_NAMES)},
                images=0 if enc.get("image_grid_thw") is None else int(enc["image_grid_thw"].shape[0]),
                peak_gpu_gb=gpu_gb(torch.cuda.max_memory_allocated),
            )
            results.append(result)
            acc = result["correct"] / max(result["scored"], 1)
            nll = result["nll"] / max(result["scored"], 1)
            log(f"[sample] {run_key}:{sample.stretch} {result['tokens']} tok ({result['images']} images) "
                f"{seconds:.0f}s = {result['tokens'] / seconds:.0f} tok/s | generated top1 {acc:.3f} "
                f"nll {nll:.3f} on {result['scored']} | peak GPU {result['peak_gpu_gb']:.1f} GB")
            for key in ("tokens", "scored", "nll", "correct", "seconds"):
                totals[key] += result[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix(".tmp.npz"), **recorder.numpy())
        os.replace(target.with_suffix(".tmp.npz"), target)
        target.with_suffix(".json").write_text(json.dumps(dict(meta_common, run=run_key, samples=results), indent=1))
        log(f"[run] {run_key} saved")

    summary = dict(meta_common, tokens=totals["tokens"], scored=totals["scored"], seconds=totals["seconds"],
                   top1=totals["correct"] / max(totals["scored"], 1),
                   nll=totals["nll"] / max(totals["scored"], 1),
                   tokens_per_second=totals["tokens"] / max(totals["seconds"], 1e-9),
                   wall_minutes=(time.time() - started) / 60)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"[done] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
