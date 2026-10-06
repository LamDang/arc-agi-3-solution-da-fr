"""Play ARC-AGI-3 games with a pruned Flash-Next: dfranzen's submission notebook
with one step added before the server starts.

    python kaggle/push_games.py --fold all --keep 256 --passes 2 --maxreq 20 [--no-push]
    python kaggle/push_games.py --fold a --keep 256 [--passes 4] [--maxreq 10]
    python kaggle/push_games.py --fold a --keep 512        # same games, unpruned baseline

The added cell ranks experts from the REAP statistics of the calibration
statistics (output of the `lamdang/reap-flash-next` kernel run with
kaggle/calib.json; with folds, only of the games this run does not play), writes the pruned checkpoint to /tmp with
prune_checkpoint.py, and points MODEL_DIR at it. Everything else (harness,
SGLang build and flags, speculative decoding, per-game time budget scaled to
the competition's) is dfranzen's notebook as published, so scores compare
with its v3 run (25 games x 4 passes, mean 46.49).

`--fold all` plays all 25 games with experts ranked from every game's
statistics (the calibration traces and the games played are the same 25, a
small optimism we accept). Folds a (13 games) and b (12) instead rank
without the games played; running both plays every game held out. Wall time
is about 532 min x runs / 110 (the per-game budget is scaled to the
competition's GPU share): 25 games x 2 passes take about 4 hours. `--maxreq` sets the number of
concurrent decoding requests: SGLang --max-running-requests, the harness's
active streams, the linear-attention state cache (6 slots per request, as in
dfranzen's 60 for 10) and the CUDA graph batch sizes. 10 is dfranzen's
setting; 256 experts free about 31 GB, enough for about 20 at full context.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import push  # noqa: E402

GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 "
         "tn36 tr87 tu93 vc33 wa30").split()
FOLDS = {"a": GAMES[0::2], "b": GAMES[1::2], "all": GAMES}
SOURCE = "dfranzen/arc-agi-3-milestone-2-solution"
STATS_KERNEL = "lamdang/reap-flash-next"

CONFIG_CELL = '''# ---- pruned-model run (added by exp/reap-flash-next/kaggle/push_games.py) ----
PRUNED_RUN = __RUN__
PLAY_GAMES = PRUNED_RUN["play"]
EXCLUDED_GAMES = [g for g in PRUNED_RUN["all_games"] if g not in PLAY_GAMES]
print("pruned run:", json.dumps(PRUNED_RUN))
'''

PRUNE_CELL = '''# ---- build the pruned checkpoint (added by exp/reap-flash-next/kaggle/push_games.py) ----
import glob, threading
if PRUNED_RUN["keep"] < 512:
    _t = time.time()
    _code = Path(glob.glob("/kaggle/input/**/prune_checkpoint.py", recursive=True)[0]).parent
    _stats = sorted({str(Path(p).parent.parent.parent) for p in glob.glob("/kaggle/input/**/reap/calib/stats/*/*.npz", recursive=True)})
    assert len(_stats) == 1, f"expected the calibration statistics of {PRUNED_RUN['stats_kernel']} as an input, found {_stats}"
    _out = f"/tmp/flash-next-pruned-{PRUNED_RUN['keep']}"
    subprocess.run([sys.executable, str(_code / "prune_checkpoint.py"), "--model-dir", MODEL_DIR,
                    "--stats-dir", _stats[0], "--keep", str(PRUNED_RUN["keep"]), "--out", _out,
                    "--criterion", PRUNED_RUN["criterion"]]
                   + ([] if PRUNED_RUN["fold"] == "all" else ["--exclude-games", ",".join(PLAY_GAMES)]), check=True)
    import shutil as _shutil
    _shutil.copy(Path(_out) / "keep.json", WORKING_DIR / "keep.json")
    MODEL_DIR = _out
    # the launcher's precache and startup clock should start now, on the pruned checkpoint
    precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR),
                                             kwargs={"delay": 0, "threads": 3})
    NOTEBOOK_START_TIME = time.time()
    print(f"pruned checkpoint {MODEL_DIR} in {time.time() - _t:.0f}s")
else:
    print("keep >= 512: unpruned baseline")
'''


def _patch(source: str, old: str, new: str, count: int = 1) -> str:
    found = source.count(old)
    if found != count:
        raise SystemExit(f"expected {count} occurrence(s) of {old!r} in the source notebook, found {found}")
    return source.replace(old, new)


def build(nb: dict, run: dict) -> dict:
    cells = nb["cells"]
    text = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells]

    def find(marker: str) -> int:
        hits = [i for i, t in enumerate(text) if marker in t]
        if len(hits) != 1:
            raise SystemExit(f"expected one cell containing {marker!r}, found {len(hits)}")
        return hits[0]

    paths, launcher = find("MODEL_DIR         = "), find("def prepare_draft_view(")
    custom, serving = find("demo_excluded_games = "), find("## 5. Start serving")
    maxreq = run["maxreq"]
    text[paths] = _patch(text[paths], "'ARC3_MAX_ACTIVE_STREAMS': 10,", f"'ARC3_MAX_ACTIVE_STREAMS': {maxreq},")
    text[launcher] = _patch(text[launcher], "MAXREQ=10,", f"MAXREQ={maxreq},")
    text[launcher] = _patch(text[launcher], "CUDAGRAPH_MAXBS=10,", f"CUDAGRAPH_MAXBS={maxreq},")
    # SGLang also caps running requests at max_mamba_cache_size // (state slots per request), with only a
    # warning; dfranzen's 60 slots serve 10 requests, so scale them with the request count
    text[launcher] = _patch(text[launcher], "MAMBA_CACHE=60,", f"MAMBA_CACHE={6 * maxreq},")
    # CUDA graphs between 10 and the new maximum, so batches of 11-19 do not all pad to the maximum
    text[launcher] = _patch(text[launcher], "graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, ",
                            "graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, *range(12, CFG['MAXREQ'], 2), ")
    text[custom] = _patch(text[custom], "bm.n_passes = 4\n", "bm.n_passes = PRUNED_RUN['passes']\n")
    text[custom] = _patch(text[custom], "demo_excluded_games = [] if TRUE_SUBMISSION else []",
                          "demo_excluded_games = [] if TRUE_SUBMISSION else EXCLUDED_GAMES")
    config = CONFIG_CELL.replace("__RUN__", json.dumps(run))

    def code(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    out = []
    for i, cell in enumerate(cells):
        if i == serving:
            out.append(code(PRUNE_CELL))
        if cell["cell_type"] == "code":
            cell = dict(cell, source=text[i], outputs=[], execution_count=None)
        out.append(cell)
        if i == paths:
            out.append(code(config))
    return dict(nb, cells=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", choices=sorted(FOLDS), required=True)
    parser.add_argument("--keep", type=int, default=256, help="experts per layer; 512 = unpruned baseline")
    parser.add_argument("--criterion", default="gate_norm")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--maxreq", type=int, default=10)
    parser.add_argument("--user", default="lamdang")
    parser.add_argument("--kernel", help="default: flash-next-games-<keep>-<fold>")
    parser.add_argument("--no-push", action="store_true", help="write the notebook only")
    args = parser.parse_args()

    run = {"fold": args.fold, "play": FOLDS[args.fold], "all_games": GAMES, "keep": args.keep,
           "criterion": args.criterion, "passes": args.passes, "maxreq": args.maxreq, "stats_kernel": STATS_KERNEL}
    slug = args.kernel or f"flash-next-games-{args.keep}-{args.fold}"
    if args.maxreq != 10:
        slug += f"-req{args.maxreq}"
    build_dir = push.HERE / "build" / slug
    source_dir = push.HERE / "build" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    push.kaggle("kernels", "pull", SOURCE, "-p", str(source_dir), "-m")
    meta = json.loads((source_dir / "kernel-metadata.json").read_text())
    nb = json.loads((source_dir / meta["code_file"]).read_text())

    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "notebook.ipynb").write_text(json.dumps(build(nb, run), indent=1))
    pruned = args.keep < 512
    (build_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": meta["dataset_sources"] + ([f"{args.user}/reap-flash-next-code"] if pruned else []),
        "kernel_sources": [STATS_KERNEL] if pruned else [],
        "model_sources": meta["model_sources"],
        "competition_sources": meta["competition_sources"],
        **({"docker_image": meta["docker_image"]} if meta.get("docker_image") else {}),
    }, indent=1))
    print(f"notebook written to {build_dir}: {json.dumps(run)}")
    if not args.no_push:
        if pruned:
            push.upload_dataset(f"{args.user}/reap-flash-next-code", "reap-flash-next-code",
                                [push.HERE.parent / f for f in push.CODE_FILES], "code for the pruned game runs")
        push.kaggle("kernels", "push", "-p", str(build_dir), "--accelerator", "NvidiaRtxPro6000")


if __name__ == "__main__":
    main()
