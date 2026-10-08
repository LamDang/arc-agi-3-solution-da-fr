"""Freeze verified scoring outputs and their exact runtime into a private DVC ZIP."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from common import code_hash, digest, file_hash, read_json, verify_manifest, write_json
from report import generate
from results import load_result


def archive_results(root, bundle, setup, out, comparison=None):
    root, bundle, setup, out = map(Path, (root, bundle, setup, out))
    manifest, run = read_json(root / "manifest.json"), read_json(root / "run.json")
    verify_manifest(manifest)
    if digest(run["config"]) != run["identity"]:
        raise ValueError("Run configuration checksum mismatch")
    if read_json(bundle / "manifest.json") != manifest:
        raise ValueError("Prepared bundle differs from scoring manifest")
    if code_hash(setup / "nll", setup / "reap") != run["config"]["code_sha256"]:
        raise ValueError("Runtime code differs from scoring identity")
    decision = generate(root)
    if not decision["complete"] or read_json(root / "heartbeat.json")["phase"] != "complete":
        raise ValueError("Cannot archive an incomplete run")
    required = decision["required_counts"]
    for count in required:
        for row in manifest["samples"]:
            if load_result(root, run["identity"], count, row) is None:
                raise ValueError("Missing completed result")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sol-nll-archive-", dir=out) as temporary:
        stage = Path(temporary)
        names = ["manifest.json", "run.json", "heartbeat.json", "smoke.json", "selection.json",
                 "comparison.json", "comparison.csv", "comparison.html", "slices.json", "coverage.json",
                 "scope.json", "kernel-qualification.json", "launch.json", "worker.log", "preflight.log"]
        for name in names:
            if (root / name).exists():
                shutil.copyfile(root / name, stage / name)
        for count in required:
            shutil.copytree(root / "results" / str(count), stage / "results" / str(count))
        shutil.copytree(bundle / "maps", stage / "maps")
        shutil.copyfile(bundle / "folds.json", stage / "folds.json")
        for sub in ("nll", "reap"):
            target = stage / "runtime-code" / sub
            target.mkdir(parents=True)
            for p in (setup / sub).iterdir():
                if p.is_file() and (p.suffix == ".py" or p.name in
                                    ("README.md", "requirements-cpu.lock", "requirements-kaggle.lock")):
                    shutil.copyfile(p, target / p.name)
        if (setup / "kaggle-nll-30.ipynb").exists():
            shutil.copyfile(setup / "kaggle-nll-30.ipynb", stage / "runtime-code/kaggle-nll-30.ipynb")
        if comparison:
            shutil.copytree(comparison, stage / "panel-comparison")
        analysis = stage / "analysis-code"
        analysis.mkdir()
        for name in ("archive_results.py", "compare_panels.py"):
            shutil.copyfile(Path(__file__).parent / name, analysis / name)
        if code_hash(stage / "runtime-code/nll", stage / "runtime-code/reap") != run["config"]["code_sha256"]:
            raise ValueError("Archived runtime checksum mismatch")
        inventory = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        write_json(stage / "inventory.json", dict(sha256=inventory,
                   verified_results=len(required)*len(manifest["samples"]), verified_request_pairs=len(manifest["samples"])))
        target = out / "results.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage))
        provenance = dict(schema=1, run_identity=run["identity"], manifest_sha256=manifest["manifest_sha256"],
                          dataset_revision=manifest["dataset_revision"], dataset_md5=manifest["dataset_md5"],
                          variant=manifest.get("variant"), runtime_code_sha256=run["config"]["code_sha256"],
                          model=run["config"]["model"], versions=run["config"]["versions"], gpu=run["config"]["gpu"],
                          evaluated_counts=required, verified_results=len(required)*len(manifest["samples"]),
                          paired_requests=len(manifest["samples"]), scoring="teacher-forced final reply only; full context",
                          target_tokens_per_model=sum(r["target_tokens"] for r in manifest["samples"]),
                          archive_sha256=file_hash(target), archive_bytes=target.stat().st_size,
                          excluded="credentials, failed runs, collector/PID state, model/wheel/processor binaries")
        write_json(out / "provenance.json", provenance)
        for name in ("comparison.csv", "selection.json", "slices.json"):
            data = (root / name).read_bytes()
            if name.endswith(".csv"):
                data = data.replace(b"\r\n", b"\n")
            (out / name).write_bytes(data)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "bundle", "setup", "out"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--comparison")
    args = parser.parse_args()
    result = archive_results(args.root, args.bundle, args.setup, args.out, args.comparison)
    print({k: result[k] for k in ("verified_results", "archive_sha256", "archive_bytes")})
