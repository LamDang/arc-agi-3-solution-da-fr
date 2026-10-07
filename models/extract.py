"""Build a pruned Flash-Next checkpoint from the original 512-expert one.

    python extract.py --experts 384 --source MODEL_DIR --out /tmp/flash-next-reap-384 [--copy]

MODEL_DIR is dfranzen's Intel W4A16 AutoRound checkpoint
(/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1).
The kept experts come from flash-next-reap-<N>/keep.json in this repository, or
keep_<N>.json next to this script (the Kaggle dataset layout). Only numpy is
needed; tensor bytes are copied without decoding, about a minute from a warm
disk. Shards holding routed experts are rewritten; the other files are
symlinked to MODEL_DIR, or copied with --copy (a self-contained directory).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# prune_checkpoint.py sits next to this script in the Kaggle dataset, in exp/reap-flash-next in the repository
for code in (HERE, HERE.parent / "exp" / "reap-flash-next"):
    if (code / "prune_checkpoint.py").exists():
        sys.path.insert(0, str(code))
        break
from prune_checkpoint import prune  # noqa: E402

VARIANTS = (448, 384, 320, 256)


def keep_path(n: int) -> Path:
    for path in (HERE / f"flash-next-reap-{n}" / "keep.json", HERE / f"keep_{n}.json"):
        if path.exists():
            return path
    raise SystemExit(f"no keep file for {n} experts next to {HERE}")


def load_mask(keep_file: Path, model_dir: Path) -> tuple[np.ndarray, dict]:
    info = json.loads(keep_file.read_text())
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    n_layers, n_experts = text["num_hidden_layers"], text["num_experts"]
    if len(info["kept"]) != n_layers or max(max(ids) for ids in info["kept"].values()) >= n_experts:
        raise SystemExit(f"{keep_file} does not fit {model_dir} ({n_layers} layers, {n_experts} experts): "
                         "point --source at the original 512-expert checkpoint")
    mask = np.zeros((n_layers, n_experts), dtype=bool)
    for layer, ids in info["kept"].items():
        mask[int(layer), ids] = True
    assert (mask.sum(1) == info["num_experts"]).all()
    meta = {k: v for k, v in info.items() if k not in ("num_experts", "source", "kept")}
    return mask, dict(meta, keep_file=keep_file.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experts", type=int, choices=VARIANTS, help="experts kept per layer")
    parser.add_argument("--keep-file", type=Path, help="another keep.json instead of --experts")
    parser.add_argument("--source", type=Path, required=True, help="original 512-expert checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="copy unchanged files instead of symlinking them")
    args = parser.parse_args()
    if not args.experts and not args.keep_file:
        parser.error("--experts or --keep-file is required")
    keep_file = args.keep_file or keep_path(args.experts)
    mask, info = load_mask(keep_file, args.source)
    print(f"[extract] {int(mask[0].sum())}/{mask.shape[1]} experts per layer from {keep_file}", flush=True)
    prune(args.source, args.out, mask, info, log=lambda m: print(m, flush=True), copy=args.copy)


if __name__ == "__main__":
    main()
