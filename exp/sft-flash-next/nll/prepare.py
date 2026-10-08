"""CPU-only: freeze 30 real requests, exact target annotations and clean maps."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from common import COUNTS, SCHEMA, digest, file_hash, read_json, write_json

LOCKED_MD5 = {
    "train.jsonl": "b1b7eaed4fe609748d1980751354b9f2",
    "index.json": "370ec02490f6567c91bc6ef5140ac172",
    "meta.json": "0170a4199d41eb05ad15ca1134afcba2",
}
PROCESSOR_MD5 = {"chat_template.jinja": "519239a4908bb1f805bbce5fa8c8a242",
                 "tokenizer.json": "085e10165f24499bb9d8fcee7e17f9ee"}


def sample_panel(index, folds, fold=0, seed=20261008):
    valid = next(x["game_ids"] for x in folds["folds"] if x["fold"] == fold)
    known = {g for f in folds["folds"] for g in f["game_ids"]}
    if len(known) != sum(len(f["game_ids"]) for f in folds["folds"]):
        raise ValueError("Overlapping game folds")
    if any(row["game"] not in known for row in index):
        raise ValueError("Unknown game in dataset index")
    keys = [(r["game"], r["request_index"]) for r in index]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate request IDs")
    chosen, strata = [], []
    for game in valid:
        rows = sorted((r for r in index if r["game"] == game),
                      key=lambda r: (r["context_tokens"], r["request_index"], r["line"]))
        for b, band in enumerate(np.array_split(np.arange(len(rows)), 3)):
            if len(band) < 2:
                raise ValueError(f"Need at least two rows per stratum: {game}, {b}")
            population = [rows[int(i)] for i in band]
            picked = random.Random(f"sol-nll-v1:{seed}:{game}:{b}").sample(population, 2)
            strata.append(dict(game=game, stratum=b, population=len(population), sampled=2,
                               min_context=population[0]["context_tokens"],
                               max_context=population[-1]["context_tokens"]))
            for draw, row in enumerate(sorted(picked, key=lambda r: r["request_index"])):
                chosen.append(dict(row, sample_id=f"{game}-r{row['request_index']}", stratum=b,
                                   draw=draw, population=len(population), inclusion_probability=2/len(population),
                                   weight=len(population)/2))
    # Matched five-game rounds, alternating context bands. No adaptive resampling.
    chosen.sort(key=lambda r: (r["draw"], r["stratum"], valid.index(r["game"])))
    return chosen, strata, valid, sorted(known - set(valid))


def clean_maps(stats_dir, folds, fold=0, counts=COUNTS):
    validation = set(next(x["games"] for x in folds["folds"] if x["fold"] == fold))
    known = {g for f in folds["folds"] for g in f["games"]}
    total, used, excluded = None, {}, []
    for path in sorted(Path(stats_dir).glob("stats/*/*.npz")):
        game = path.stem.rsplit("_p", 1)[0]
        if game not in known:
            raise ValueError(f"Unknown calibration game: {path.name}")
        if game in validation:
            excluded.append(str(path.relative_to(stats_dir)))
            continue
        with np.load(path, allow_pickle=False) as archive:
            a = archive["gate_norm"].astype(np.float64)
        if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"Invalid calibration statistics: {path}")
        a = a.sum(-1)
        if total is not None and total.shape != a.shape:
            raise ValueError("Calibration architecture mismatch")
        total = a if total is None else total + a
        used[str(path.relative_to(stats_dir))] = file_hash(path)
    if total is None or not (total.sum(-1) > 0).all():
        raise ValueError("No usable training-only calibration statistics")
    if max(counts) != total.shape[1]:
        raise ValueError("Statistics do not match the full expert count")
    order = np.argsort(-total, axis=1, kind="stable")
    maps = {}
    for count in counts:
        maps[str(count)] = {"num_experts": count, "source_num_experts": total.shape[1],
                           "kept": {str(i): sorted(row[:count].tolist()) for i, row in enumerate(order)},
                           "criterion": "gate_norm", "categories": ["context", "generated", "image"],
                           "calibration_runs": used, "excluded_validation_runs": excluded,
                           "validation_exposed": False}
    return maps


def prepare(args):
    from data import encode, load_processor

    data_dir, processor_dir, out = Path(args.data_dir), Path(args.processor), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation output must be empty; immutable panels are not overwritten")
    for name, expected in LOCKED_MD5.items():
        if file_hash(data_dir / name, "md5") != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    for name, expected in PROCESSOR_MD5.items():
        if file_hash(processor_dir / name, "md5") != expected:
            raise ValueError(f"Pinned processor hash mismatch: {name}")
    folds = read_json(args.folds)
    index = read_json(data_dir / "index.json")
    panel, strata, valid, train = sample_panel(index, folds, args.fold, args.seed)
    if len(panel) != 30 or len(valid) != 5:
        raise ValueError("This run requires exactly 30 requests / five validation games")
    maps = clean_maps(args.stats_dir, folds, args.fold)
    processor = load_processor(processor_dir)
    processor_files = {}
    for path in sorted(processor_dir.iterdir()):
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt", ".model"):
            dst = out / "processor" / path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
            processor_files[path.name] = file_hash(dst)
    records = []
    with open(data_dir / "train.jsonl", "rb") as stream:
        for row in panel:
            stream.seek(row["offset"])
            raw = stream.read(row["length"])
            sample = json.loads(raw)
            if (sample["game"], sample["request_index"]) != (row["game"], row["request_index"]):
                raise ValueError("Index byte range does not match request identity")
            _, annotation = encode(processor, sample)
            if annotation["total_tokens"] > 139264:
                raise ValueError("Selected full request exceeds model context; never truncate")
            relative = f"samples/{row['sample_id']}.json"
            write_json(out / relative, sample)
            record = dict(row, path=relative, sample_sha256=file_hash(out / relative), **annotation)
            write_json(out / f"annotations/{row['sample_id']}.json", annotation)
            records.append(record)
            print(f"prepared {row['sample_id']}: {annotation['prompt_tokens']} + "
                  f"{annotation['target_tokens']} tokens", flush=True)
    for count, value in maps.items():
        write_json(out / f"maps/keep-{count}.json", value)
    write_json(out / "folds.json", folds)
    token_budget = sum(r["total_tokens"] for r in records) * len(COUNTS)
    coverage = []
    for game in valid:
        levels = sorted({r.get("level") for r in index if r["game"] == game}, key=lambda x: -1 if x is None else x)
        for level in levels:
            population = [r for r in index if r["game"] == game and r.get("level") == level]
            selected = [r for r in records if r["game"] == game and r.get("level") == level]
            coverage.append(dict(game=game, level=level, population_requests=len(population),
                                 sampled_requests=len(selected), status="sampled" if selected else "not sampled"))
    manifest = dict(schema=SCHEMA, real_data=True, dataset_revision="37fadfeedfd54d2129db4525e4167596cc7337b6",
                    model_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
                    fold=args.fold, seed=args.seed, sampling="two-per-game-context-tertile-v1",
                    dataset_md5=LOCKED_MD5, fold_sha256=digest(folds), validation_games=valid,
                    training_games=train, strata=strata, coverage=coverage, counts=list(COUNTS), samples=records,
                    processor_files=processor_files, image_backend="pil",
                    image_processor_class=type(processor.image_processor).__name__,
                    maps={k: digest(v) for k, v in maps.items()},
                    jobs=len(records)*len(COUNTS), processed_tokens=token_budget,
                    estimated_minutes={str(rate): round(token_budget/rate/60 + 30, 1)
                                       for rate in (500, 850, 950)})
    manifest["manifest_sha256"] = digest(manifest)
    write_json(out / "manifest.json", manifest)  # completion marker, written last
    print(json.dumps({k: manifest[k] for k in ("jobs", "processed_tokens", "estimated_minutes")}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261008)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
