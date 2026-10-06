"""Write a checkpoint with only the kept routed experts, loadable by SGLang.

    python prune_checkpoint.py --model-dir MODEL --stats-dir REAP_OUT [--stats-dir ...] --keep 256 --out PRUNED \\
        [--criterion gate_norm] [--categories context,generated,image] [--calib-games ar25,bp35 | --exclude-games ls20]
    python prune_checkpoint.py --model-dir MODEL --keep-file keep.json --keep 256 --out PRUNED

Experts are ranked per layer from the saved statistics (analyze.expert_scores)
and the top N of every layer are kept, in their original order, renumbered
0..N-1. Each layer's router keeps only the rows of its kept experts, so
softmax, top-k and renormalization run over the kept experts as if the others
had never existed (what prune_eval.py measured). text_config.num_experts
becomes N. Shards holding routed experts or routers are rewritten; every other
file (n-gram table shards, tokenizer, chat template, processor configs) is
symlinked. MTP tensors are copied unchanged: SGLang skips them in the target
model and the draft model is a separate checkpoint. Tensor bytes are copied
without decoding, so only numpy is needed.

keep.json in the output records the kept expert ids per layer and the
statistics they came from.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import analyze

EXPERT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
ROUTER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")


def choose(stats_dirs, keep: int, criterion: str, categories: list[int], calib_games=None, exclude_games=None):
    runs = {}
    for directory in stats_dirs:
        runs.update(analyze.load(Path(directory)))
    keys = [k for k in runs
            if (not calib_games or analyze.game_of(k) in calib_games)
            and (not exclude_games or analyze.game_of(k) not in exclude_games)]
    if not keys:
        raise SystemExit(f"no statistics left after filtering ({len(runs)} runs found)")
    mask = analyze.keep_mask(analyze.expert_scores(analyze.aggregate(runs, keys, categories), criterion), keep)
    return mask, sorted(keys)


def read_header(path: Path) -> tuple[dict, int]:
    """safetensors header and the file offset where tensor data starts."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def write_safetensors(path: Path, entries: list, metadata: dict | None, read):
    """entries: (name, dtype, shape, nbytes, source). The header is padded to
    8 bytes like the safetensors library does; read(source) returns the bytes."""
    header, offset = {}, 0
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape, nbytes, _ in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for _, _, _, nbytes, source in entries:
            data = read(source)
            assert len(data) == nbytes
            f.write(data)
    return offset


def prune(model_dir: Path, out: Path, mask: np.ndarray, info: dict, log=print) -> dict:
    """Copies tensor bytes without decoding them (no torch needed): expert
    tensors are renamed, router rows sliced, everything else copied."""
    model_dir, out = Path(model_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_layers, n_experts = mask.shape
    n_keep = int(mask[0].sum())
    assert (mask.sum(1) == n_keep).all(), "every layer must keep the same number of experts"
    new_id = np.full(mask.shape, -1)
    for layer in range(n_layers):
        new_id[layer, mask[layer]] = np.arange(n_keep)

    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    assert text["num_experts"] == n_experts and text["num_hidden_layers"] == n_layers, "mask does not fit the model"
    assert n_keep >= text["num_experts_per_tok"]
    text["num_experts"] = n_keep

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_file: dict[str, list[str]] = {}
    for name, file in index["weight_map"].items():
        by_file.setdefault(file, []).append(name)
    weight_map, rewritten = {}, []
    for file, names in sorted(by_file.items()):
        if not any(EXPERT_RE.match(n) or ROUTER_RE.match(n) for n in names):
            _link(model_dir / file, out / file)
            weight_map.update({n: file for n in names})
            continue
        t = time.time()
        header, base = read_header(model_dir / file)
        metadata = header.pop("__metadata__", None)
        entries = []
        for name, spec in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            start, stop = spec["data_offsets"]
            source = (base + start, stop - start, None)
            if m := EXPERT_RE.match(name):
                layer, expert = int(m[1]), int(m[2])
                if not mask[layer, expert]:
                    continue
                name = f"model.language_model.layers.{layer}.mlp.experts.{new_id[layer, expert]}.{m[3]}"
            elif m := ROUTER_RE.match(name):
                assert spec["shape"][0] == n_experts, f"{name}: {spec['shape']}"
                rows = np.flatnonzero(mask[int(m[1])])
                row_bytes = (stop - start) // n_experts
                spec = dict(spec, shape=[n_keep, *spec["shape"][1:]])
                source = (base + start, stop - start, (rows, row_bytes))
            entries.append((name, spec["dtype"], spec["shape"], source[1] if source[2] is None
                            else len(source[2][0]) * source[2][1], source))
        with open(model_dir / file, "rb") as f:
            def read(source):
                offset, length, rows = source
                f.seek(offset)
                data = f.read(length)
                if rows is None:
                    return data
                kept, row_bytes = rows
                return np.frombuffer(data, dtype=np.uint8).reshape(-1, row_bytes)[kept].tobytes()

            target = out / file
            if target.is_symlink() or target.exists():
                target.unlink()
            size = write_safetensors(target, entries, metadata, read)
        weight_map.update({e[0]: file for e in entries})
        rewritten.append(file)
        log(f"[prune] {file}: {len(entries)}/{len(names)} tensors, {size / 1e9:.2f} GB in {time.time() - t:.0f}s")

    expected = n_layers * n_keep * 9  # 3 projections x (qweight, scales, qzeros)
    found = sum(1 for n in weight_map if EXPERT_RE.match(n))
    assert found == expected, f"{found} routed-expert tensors written, expected {expected}"
    index = dict(index, weight_map=weight_map)
    index["metadata"] = dict(index.get("metadata") or {}, pruned_from=str(model_dir), num_experts=n_keep)
    for name in ("config.json", "model.safetensors.index.json"):
        if (out / name).is_symlink():
            (out / name).unlink()
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    skip = {"config.json", "model.safetensors.index.json", *by_file}
    for path in model_dir.iterdir():
        if path.name not in skip and path.is_file():
            _link(path, out / path.name)
    keep = {"num_experts": n_keep, "source": str(model_dir),
            "kept": {str(layer): np.flatnonzero(mask[layer]).tolist() for layer in range(n_layers)}, **info}
    (out / "keep.json").write_text(json.dumps(keep) + "\n")
    log(f"[prune] {out}: {n_keep}/{n_experts} experts per layer, {len(rewritten)} shards rewritten, "
        f"{len(by_file) - len(rewritten)} linked")
    return keep


def _link(src: Path, dest: Path):
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == src.resolve():
            return
        dest.unlink()
    dest.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stats-dir", action="append", default=[], help="run_reap/prune_eval output")
    parser.add_argument("--keep-file", help="keep.json of an earlier pruning: the same experts, no statistics needed")
    parser.add_argument("--keep", type=int, required=True, help="experts kept per layer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--criterion", choices=analyze.CRITERIA, default="gate_norm")
    parser.add_argument("--categories", default=",".join(analyze.CATEGORIES))
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--calib-games", default="", help="rank with these games only")
    games.add_argument("--exclude-games", default="", help="rank without these games (e.g. the ones to play)")
    args = parser.parse_args()
    categories = [analyze.CATEGORIES.index(c) for c in args.categories.split(",")]
    calib = set(filter(None, args.calib_games.split(",")))
    exclude = set(filter(None, args.exclude_games.split(",")))
    if args.keep_file:
        info = json.loads(Path(args.keep_file).read_text())
        config = json.loads((Path(args.model_dir) / "config.json").read_text())
        text = config.get("text_config", config)
        mask = np.zeros((text["num_hidden_layers"], text["num_experts"]), dtype=bool)
        for layer, ids in info.pop("kept").items():
            mask[int(layer), ids] = True
        assert info["num_experts"] == args.keep == mask.sum(1).min() == mask.sum(1).max()
        info = {k: v for k, v in info.items() if k not in ("num_experts", "source")}
        print(f"[prune] experts from {args.keep_file}", flush=True)
    elif args.stats_dir:
        mask, runs = choose(args.stats_dir, args.keep, args.criterion, categories, calib, exclude)
        info = {"criterion": args.criterion, "categories": args.categories, "calibration_runs": runs}
        print(f"[prune] ranking from {len(runs)} runs: {runs}", flush=True)
    else:
        raise SystemExit("--stats-dir or --keep-file is required")
    prune(Path(args.model_dir), Path(args.out), mask, info, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
