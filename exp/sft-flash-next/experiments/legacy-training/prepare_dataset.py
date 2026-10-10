"""CPU preparation; no model weights or teacher API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from dataset import digest, encode, file_hash, fold_games, inject_thinking, load_processor, read_json, validate_sample, write_json


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("Output already exists; choose a new immutable dataset directory")
    folds = read_json(args.folds)
    known, validation = fold_games(folds, args.validation_fold)
    generated, sources = {}, {}
    if args.generated_dir:
        for path in sorted(Path(args.generated_dir).glob("*.jsonl")):
            sources[str(path)] = file_hash(path)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["key"] in generated:
                    raise ValueError(f"Duplicate generated key: {row['key']}")
                generated[row["key"]] = row
        if not generated:
            raise ValueError("No generated thinking files found")
    processor = load_processor(args.processor)
    out.mkdir(parents=True)
    # Copy only processor assets, never model weights/cache directories.
    (out / "processor").mkdir()
    for p in Path(args.processor).iterdir():
        if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"} and "safetensors" not in p.name:
            shutil.copyfile(p, out / "processor" / p.name)
    write_json(out / "folds.json", folds)
    rows, seen, excluded_by_partition = [], set(), []
    with open(args.input, "rb") as stream, open(out / "requests.jsonl", "wb") as target:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("game") not in known:
                raise ValueError(f"Unknown game: {sample.get('game')}")
            split = "validation" if sample["game"] in validation else "train"
            if args.partition != "all" and split != args.partition:
                excluded_by_partition.append(dict(game=sample["game"], request_index=sample["request_index"]))
                continue
            if generated:
                sample = inject_thinking(sample, generated)
            validate_sample(sample, known)
            sid = f"{sample['game']}-r{sample['request_index']}"
            if sid in seen:
                raise ValueError(f"Duplicate request: {sid}")
            seen.add(sid)
            enc, annotation = encode(processor, sample)
            if annotation["total_tokens"] > args.max_tokens:
                raise ValueError(f"{sid} exceeds {args.max_tokens}; increase capacity, never truncate")
            if not annotation["category_counts"]["thinking"]:
                raise ValueError(f"{sid}: generated thinking vanished in rendering")
            raw = (json.dumps(sample, ensure_ascii=False) + "\n").encode()  # do not sort template-sensitive keys
            rows.append(dict(sample_id=sid, game=sample["game"],
                split=split,
                offset=target.tell(), length=len(raw), line_sha256=hashlib.sha256(raw).hexdigest(),
                annotation_sha256=digest(annotation), **{k: annotation[k] for k in
                    ("prompt_tokens", "target_tokens", "total_tokens", "images", "category_counts")}))
            target.write(raw)
            print(f"{sid}: {annotation['prompt_tokens']} + {annotation['target_tokens']} tokens", flush=True)
            del enc
    if not rows:
        raise ValueError("Empty input")
    files = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = dict(version=1, validation_fold=args.validation_fold, max_tokens=args.max_tokens,
        input_sha256=file_hash(args.input), generated_sources=sources, files=files, rows=rows,
        partition=args.partition, excluded_by_partition=excluded_by_partition,
        objective="mean per-request final-reply token NLL; generated thinking + code + format")
    manifest["sha256"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")}
    print(json.dumps(dict(counts=counts, manifest_sha256=manifest["sha256"])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Request-plus-reply JSONL, same schema as panel30")
    p.add_argument("--generated-dir", help="Optional think_gen refine2 directory, joined by game_p0#request_index")
    p.add_argument("--processor", required=True, help="Pinned local complete processor directory")
    p.add_argument("--folds", default=str(Path(__file__).resolve().parents[3] / "data/game_folds/folds.json"))
    p.add_argument("--validation-fold", type=int, default=0)
    p.add_argument("--partition", choices=("all", "train", "validation"), default="all",
                   help="Explicit fold filtering before joining thinking; exclusions recorded in manifest")
    p.add_argument("--max-tokens", type=int, default=130000)
    p.add_argument("--out", required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
