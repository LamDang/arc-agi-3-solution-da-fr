"""Upload the code (and optionally traces) as private Kaggle datasets, then push
and start the REAP kernel on an RTX PRO 6000.

    python kaggle/push.py --config kaggle/smoke.json [--traces-dir DIR]

Needs the kaggle CLI (KAGGLE_CLI, default `kaggle`) and ~/.kaggle/kaggle.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_FILES = ["traces.py", "render.py", "reap_model.py", "replay.py", "run_reap.py", "analyze.py", "prune_eval.py", "bench.py",
              "prune_checkpoint.py"]
KAGGLE = os.environ.get("KAGGLE_CLI", "kaggle")


def kaggle(*args, check=True) -> str:
    result = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    print(f"kaggle {' '.join(args)}\n  {output.splitlines()[-1] if output else ''}", flush=True)
    if check and result.returncode != 0:
        raise SystemExit(output)
    return output


def upload_dataset(slug: str, title: str, files: list[Path], message: str):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in files:
            shutil.copy(f, tmp / f.name)
        (tmp / "dataset-metadata.json").write_text(json.dumps(
            {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = "ready" in kaggle("datasets", "status", slug, check=False).lower()
        if exists:
            kaggle("datasets", "version", "-p", str(tmp), "-m", message, "--dir-mode", "skip")
        else:
            kaggle("datasets", "create", "-p", str(tmp), "--dir-mode", "skip")
    for _ in range(60):  # the kernel mounts the newest version only once it is processed
        if "ready" in kaggle("datasets", "status", slug, check=False).lower():
            return
        time.sleep(10)
    raise SystemExit(f"dataset {slug} not ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON with steps, sources, transformers version")
    parser.add_argument("--user", default="lamdang")
    parser.add_argument("--kernel", default="reap-flash-next")
    parser.add_argument("--traces-dir", help="upload these request logs as the openrouter traces dataset")
    parser.add_argument("--wheels-dir", help="upload these wheels (transformers and its dependencies) as a dataset")
    parser.add_argument("--no-push", action="store_true", help="write the kernel directory only")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    code_slug = f"{args.user}/reap-flash-next-code"
    traces_slug = f"{args.user}/arc3-openrouter-traces"
    wheels_slug = f"{args.user}/reap-flash-next-wheels"

    if not args.no_push:
        upload_dataset(code_slug, "reap-flash-next-code", [HERE.parent / f for f in CODE_FILES],
                       f"code for {Path(args.config).name}")
        if args.traces_dir:
            logs = sorted(Path(args.traces_dir).glob("*requests.jsonl*"))
            upload_dataset(traces_slug, "arc3-openrouter-traces", logs, "request logs")
        if args.wheels_dir:
            upload_dataset(wheels_slug, "reap-flash-next-wheels", sorted(Path(args.wheels_dir).glob("*.whl")),
                           "offline wheels")

    kernel_dir = HERE / "build" / args.kernel
    kernel_dir.mkdir(parents=True, exist_ok=True)
    template = (HERE / "kernel_template.py").read_text()
    (kernel_dir / "kernel.py").write_text(template.replace("__CONFIG__", json.dumps(config, indent=1)))
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.kernel}",
        "title": args.kernel,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # the RTX PRO 6000 is only offered to competition-attached notebooks, without internet
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [code_slug, traces_slug, wheels_slug],
        "kernel_sources": ["dfranzen/arc-agi-3-milestone-2-solution"],
        "model_sources": ["dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"],
        # the RTX PRO 6000 is offered to notebooks attached to the ARC-AGI-3 competition
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
    }, indent=1))
    print(f"kernel written to {kernel_dir}")
    if not args.no_push:
        kaggle("kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaRtxPro6000")


if __name__ == "__main__":
    main()
