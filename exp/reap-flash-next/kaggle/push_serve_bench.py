"""Does the pruned model fit more concurrent full-length requests, and how much
faster does it decode? Picks the serving config before the game runs.

    python kaggle/push_serve_bench.py [--keep-file kaggle/keep_256_smoke.json] [--concurrency 10,16,20,28] [--no-push]

The notebook takes dfranzen's cells for paths, precaching and the SGLang
launcher (same wheels, flags, speculative decoding), then:

1. writes the pruned checkpoint from --keep-file (expert choice barely
   matters for speed; the default mask comes from the smoke statistics);
2. serves the full model exactly as dfranzen does (10 requests, 60 state
   slots) and runs serve_bench.batch_test with 10 streams;
3. serves the pruned model sized for the largest concurrency (state slots
   and CUDA graphs scaled) and runs the batch test at each concurrency.

batch_test: n requests at once, each the longest logged prompt of a
different game run of dfranzen's v3 logs (about 100-110K tokens, near the
harness's context limit). They are prefilled first, then sent again to
decode 4096 tokens each from the cache. Reported: decode tokens/s with all n
running, cache hit, peak KV use and retractions (does n fit?), speculative
accept length. Results go to /kaggle/working/serve_bench.json and a table at
the end of the log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import push  # noqa: E402
import push_games  # noqa: E402

SETUP = '''import glob, json, os, signal, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 25 * 60
__PATHS__
WORKING_DIR = Path("/kaggle/working")
SERVED_MODEL_HOST, SERVED_MODEL_PORT = "127.0.0.1", 8001
BENCH = __BENCH__
CODE_DIR = Path(glob.glob("/kaggle/input/**/serve_bench.py", recursive=True)[0]).parent
LOG_DIR = Path(sorted(glob.glob("/kaggle/input/**/*arc-agi-3-milestone-2-solution*/**/*_p2_requests.jsonl*", recursive=True))[0]).parent
sys.path.insert(0, str(CODE_DIR))
print("bench:", json.dumps(BENCH), "\\ncode:", CODE_DIR, "\\nlogs:", LOG_DIR, flush=True)
'''

PRUNE = '''FULL_MODEL_DIR = MODEL_DIR
PRUNED_MODEL_DIR = f"/tmp/flash-next-pruned-{BENCH['keep']}"
_t = time.time()
subprocess.run([sys.executable, str(CODE_DIR / "prune_checkpoint.py"), "--model-dir", FULL_MODEL_DIR,
                "--keep-file", str(CODE_DIR / BENCH["keep_file"]), "--keep", str(BENCH["keep"]),
                "--out", PRUNED_MODEL_DIR], check=True)
print(f"pruned checkpoint in {time.time() - _t:.0f}s", flush=True)
'''

SERVER = '''LAUNCHER_SRC = __LAUNCHER__

def start_server(model_dir, maxreq, mamba):
    global MODEL_DIR, BENCH_MAXREQ, BENCH_MAMBA, NOTEBOOK_START_TIME
    MODEL_DIR, BENCH_MAXREQ, BENCH_MAMBA, NOTEBOOK_START_TIME = model_dir, maxreq, mamba, time.time()
    exec(LAUNCHER_SRC, globals())
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server not ready")

def stop_server():
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=120)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    for _ in range(60):  # until the port is free and the GPU memory released
        used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                              capture_output=True, text=True).stdout.split()
        with socket.socket() as probe:
            free_port = probe.connect_ex(("127.0.0.1", SERVED_MODEL_PORT)) != 0
        if free_port and used and int(used[0]) < 3000:
            return
        time.sleep(5)
    print("warning: GPU memory not released after stopping the server", flush=True)
'''

RUN = '''import serve_bench
sessions = serve_bench.load_sessions(LOG_DIR)
prompts = serve_bench.longest_prompts(sessions, max(max(s["concurrency"]) for s in BENCH["settings"]))
print(f"{len(sessions)} game runs; prompts of {prompts[-1][1]}-{prompts[0][1]} tokens", flush=True)
results = []
for setting in BENCH["settings"]:
    model_dir = FULL_MODEL_DIR if setting["model"] == "full" else PRUNED_MODEL_DIR
    t = time.time()
    try:
        start_server(model_dir, setting["maxreq"], setting["mamba"])
    except Exception as exc:
        print(f"[bench] {setting} failed to start: {exc}", flush=True)
        results.append({**setting, "error": str(exc)})
        stop_server()
        continue
    print(f"[bench] {setting['model']} ready in {time.time() - t:.0f}s", flush=True)
    for n in setting["concurrency"]:
        try:
            r = serve_bench.batch_test(f"http://127.0.0.1:{SERVED_MODEL_PORT}", SERVED_MODEL_NAME, prompts, n,
                                       max_tokens=BENCH["max_tokens"], log=lambda m: print(m, flush=True),
                                       server_log=str(WORKING_DIR / "serve.log"))
        except Exception as exc:
            r = {"streams": n, "error": str(exc)}
        results.append({"model": setting["model"], "maxreq": setting["maxreq"], **r})
        (WORKING_DIR / "serve_bench.json").write_text(json.dumps(results, indent=1))
    stop_server()

f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
print(f"\\n{'model':10} {'streams':>7} {'running':>7} {'decode tok/s':>12} {'per stream':>10} {'vs full@10':>10} "
      f"{'cache hit':>9} {'peak KV':>7} {'retracted':>9} {'accept':>6} {'prefill s':>9} {'errors':>6} {'131K fit':>8}")
base = next((r["decode_tok_s"] for r in results if r.get("model") == "full" and "decode_tok_s" in r), None)
for r in results:
    if "error" in r:
        print(f"{r['model']:10} {r.get('streams', '-'):>7} error: {r['error']}")
        continue
    ratio = f"{r['decode_tok_s'] / base:.2f}x" if base else "-"
    print(f"{r['model']:10} {r['streams']:>7} {r['max_running']:>7.0f} {r['decode_tok_s']:>12.0f} "
          f"{r['per_stream_tok_s']:>10.1f} {ratio:>10} {f(r['cache_hit']):>9} {f(r['peak_kv_usage']):>7} "
          f"{r['retracted_requests']:>9.0f} {f(r['spec_accept_length']):>6} {r['prefill_seconds']:>9.0f} {r['errors']:>6} "
          f"{r.get('streams_at_full_context', '-'):>8}")
'''


def build(nb: dict, bench: dict) -> dict:
    text = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]]

    def find(marker: str) -> str:
        hits = [t for t in text if marker in t]
        if len(hits) != 1:
            raise SystemExit(f"expected one cell containing {marker!r}, found {len(hits)}")
        return hits[0]

    paths_cell = find("MODEL_DIR         = ")
    paths = []
    for name in ("WHEELHOUSE_DIR", "MODEL_DIR", "DRAFT_MODEL_DIR", "SERVED_MODEL_NAME"):
        m = re.search(rf"^{name}\s*=.*$", paths_cell, re.M)
        if not m:
            raise SystemExit(f"{name} not found in the source notebook")
        paths.append(m[0])
    precache = find("def precache(")
    launcher = find("def prepare_draft_view(")
    p = push_games._patch
    launcher = p(launcher, "MAXREQ=10,", "MAXREQ=BENCH_MAXREQ,")
    launcher = p(launcher, "CUDAGRAPH_MAXBS=10,", "CUDAGRAPH_MAXBS=BENCH_MAXREQ,")
    launcher = p(launcher, "MAMBA_CACHE=60,", "MAMBA_CACHE=BENCH_MAMBA,")
    launcher = p(launcher, "graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, ",
                 "graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, *range(12, CFG['MAXREQ'], 2), ")
    # the launcher runs once per setting; the precache thread can only start once
    launcher = p(launcher, "precache_model_thread.start()\n",
                 "if precache_model_thread.ident is None: precache_model_thread.start()\n")

    def code(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source":
            "# Flash-Next serving throughput: full vs pruned\n\nGenerated by `exp/reap-flash-next/kaggle/"
            "push_serve_bench.py` from dfranzen's ARC-AGI-3 Milestone 2 notebook (paths, precaching and "
            "SGLang launcher cells)."},
        code(SETUP.replace("__PATHS__", "\n".join(paths)).replace("__BENCH__", json.dumps(bench))),
        code(precache),
        code(PRUNE),
        code(SERVER.replace("__LAUNCHER__", repr(launcher))),
        code(RUN),
    ]
    return dict(nb, cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep-file", default="kaggle/keep_256_smoke.json")
    parser.add_argument("--concurrency", default="10,16,20,28", help="streams to measure on the pruned model")
    parser.add_argument("--max-tokens", type=int, default=4096, help="tokens each stream generates")
    parser.add_argument("--user", default="lamdang")
    parser.add_argument("--kernel", default="flash-next-serve-bench")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    keep_file = push.HERE.parent / args.keep_file
    keep = json.loads(keep_file.read_text())["num_experts"]
    streams = [int(c) for c in args.concurrency.split(",")]
    bench = {"keep": keep, "keep_file": keep_file.name, "max_tokens": args.max_tokens,
             "settings": [{"model": "full", "maxreq": 10, "mamba": 60, "concurrency": [10]},
                          {"model": f"pruned{keep}", "maxreq": max(streams), "mamba": 6 * max(streams),
                           "concurrency": streams}]}
    source_dir = push.HERE / "build" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    push.kaggle("kernels", "pull", push_games.SOURCE, "-p", str(source_dir), "-m")
    meta = json.loads((source_dir / "kernel-metadata.json").read_text())
    nb = json.loads((source_dir / meta["code_file"]).read_text())
    build_dir = push.HERE / "build" / args.kernel
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "notebook.ipynb").write_text(json.dumps(build(nb, bench), indent=1))
    (build_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.kernel}", "title": args.kernel, "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": meta["dataset_sources"] + [f"{args.user}/reap-flash-next-code"],
        "kernel_sources": [push_games.SOURCE],
        "model_sources": meta["model_sources"],
        "competition_sources": meta["competition_sources"],
        **({"docker_image": meta["docker_image"]} if meta.get("docker_image") else {}),
    }, indent=1))
    print(f"notebook written to {build_dir}: {json.dumps(bench)}")
    if not args.no_push:
        push.upload_dataset(f"{args.user}/reap-flash-next-code", "reap-flash-next-code",
                            [push.HERE.parent / f for f in push.CODE_FILES] + [keep_file], "code for the serving benchmark")
        push.kaggle("kernels", "push", "-p", str(build_dir), "--accelerator", "NvidiaRtxPro6000")


if __name__ == "__main__":
    main()
