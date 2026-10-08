"""Package local code, an interactive notebook and offline wheels; never launch."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from common import file_hash, write_json
from protocol import POLICY


def notebook():
    cells = []
    def markdown(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)))
    def code(text):
        cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True)))
    markdown("""# Sol NLL: 30 requests, 512 baseline then 256

Score all 30 requests at 512, then all 30 at 256 with teacher forcing.
Stop at 256 if weighted primary NLL is no more than 5% above the full
baseline. Otherwise scan 448, 384 and 320 on the same frozen requests.
The gate uses relative NLL, not perplexity or an absolute 0.05-nat tolerance.

Use an **interactive Kaggle Jupyter session** attached to the competition.
The notebook is saved with **GPU disabled**. Attach the setup archive, the
CPU-prepared bundle and Intel W4A16 model version 1 before requesting a GPU.
Run CPU checks first. Keep the external `collect.py` process connected for
verified durable results; the scoring worker pauses if its acknowledgment
is missing or stale. No training or game episodes run here.
""")
    code('''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

SETUP = Path("/kaggle/working/sol-nll-setup")
OUT = Path("/kaggle/working/sol-nll")
archives = list(Path("/kaggle/input").rglob("sol-nll-setup.zip"))
assert len(archives) == 1, "Attach exactly one sol-nll-setup.zip"
with zipfile.ZipFile(archives[0]) as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(SETUP)
inventory = json.loads((SETUP / "setup-manifest.json").read_text())
for name, checksum in inventory["sha256"].items():
    assert hashlib.sha256((SETUP/name).read_bytes()).hexdigest() == checksum, name
assert sys.version_info[:2] == (3, 13), "Wheelhouse targets Kaggle Python 3.13; build matching wheels for another image"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(SETUP/"wheels"),
                "-r", str(SETUP/"nll/requirements-kaggle.lock")], check=True)
print("Setup verified; torch/CUDA supplied by the Kaggle image were not replaced.")
''')
    code('''# Locate the frozen real-data panel. This cell must pass before GPU startup.
BUNDLE = SETUP / "bundle"
if not (BUNDLE / "manifest.json").exists():
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            value = json.loads(path.read_text())
            if value.get("real_data") is True and len(value.get("samples", [])) == 30:
                candidates.append(path.parent)
        except (ValueError, OSError):
            pass
    assert len(candidates) == 1, "Attach the CPU-prepared 30-request bundle; synthetic fixtures cannot start a GPU run"
    BUNDLE = candidates[0]
subprocess.run([sys.executable, str(SETUP/"nll/run.py"), "--bundle", str(BUNDLE),
                "--out", str(OUT), "--preflight-only"], check=True)
''')
    markdown("""If resuming, attach the downloaded durable result archive and restore it
into `/kaggle/working/sol-nll` before the next cell. Restore only this run's
verified manifest/run/results/smoke files. Never copy credentials into inputs.
Start the external collector against this session's Jupyter contents path
(`sol-nll` when the server root is `/kaggle/working`). It retries until run.json
exists and acknowledges verified downloads. The final cell is intentionally
disabled until you explicitly enable the GPU run.
""")
    code('''# Optional restore. Change this path only when resuming an existing run.
RESTORE_ARCHIVE = None
if RESTORE_ARCHIVE:
    assert not OUT.exists() or not any(OUT.iterdir()), "Restore into an empty output directory"
    with zipfile.ZipFile(RESTORE_ARCHIVE) as z:
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
        z.extractall(OUT)
    print("Restored results; the worker will validate identity/checksums before skipping jobs.")
''')
    code('''# Enable RTX PRO 6000 96 GB in the interactive notebook, then set True.
START_GPU_RUN = False
assert START_GPU_RUN, "GPU launch is disabled until the prepared data and external collector are ready"
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), "GiB")
model_indexes = [p for p in Path("/kaggle/input").rglob("model.safetensors.index.json")
                 if "intel" in str(p).lower() and "flash" in str(p).lower()]
assert len(model_indexes) == 1, "Attach the pinned full Intel W4A16 Flash-Next model version 1"
OUT.mkdir(exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
command = [sys.executable, "-u", str(SETUP/"nll/run.py"), "--bundle", str(BUNDLE),
           "--out", str(OUT), "--model-dir", str(model_indexes[0].parent),
           "--reap-dir", str(SETUP/"reap"), "--session-minutes", "120"]
with open(OUT/"worker.log", "ab", buffering=0) as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
print("Worker PID:", process.pid, "— monitor heartbeat/log below; the worker lock prevents duplicate evaluation.")
''')
    code('''# Re-run to monitor after a browser reconnect. It does not launch another worker.
from IPython.display import FileLink, display
if (OUT/"heartbeat.json").exists():
    print((OUT/"heartbeat.json").read_text())
if (OUT/"worker.log").exists():
    print("\\n".join((OUT/"worker.log").read_text(errors="replace").splitlines()[-12:]))
if (OUT/"comparison.html").exists():
    display(FileLink(str(OUT/"comparison.html")))
# To stop at the next request boundary: (OUT/"STOP").touch()
# Before resuming, remove STOP only after confirming you want to continue.
''')
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"},
                          "kaggle": {"accelerator": "none", "isGpuEnabled": False,
                                     "dockerImageVersionId": 28755,
                                     "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}})


def build(out, wheels=None, bundle=None):
    from run import validate_bundle
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "setup"
    if stage.exists():
        raise ValueError("Build destination already contains a setup; use a fresh output directory")
    for src in sorted(here.glob("*.py")) + sorted(here.glob("*.lock")) + [here / "README.md"]:
        dst = stage / "nll" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in ("reap_model.py", "render.py", "replay.py"):
        dst = stage / "reap" / name
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(repo / "exp/reap-flash-next" / name, dst)
    shutil.copyfile(repo / "data/game_folds/folds.json", stage / "folds.json")
    if wheels:
        shutil.copytree(wheels, stage / "wheels")
    if bundle:
        validate_bundle(bundle)
        shutil.copytree(bundle, stage / "bundle")
    nb = notebook()
    write_json(out / "kaggle-nll-30.ipynb", nb)
    write_json(stage / "kaggle-nll-30.ipynb", nb)
    files = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    write_json(stage / "setup-manifest.json", dict(sha256=files, requests=30, jobs=150,
                                                   protocol=POLICY, initial_jobs=60, maximum_jobs=150,
                                                   has_prepared_real_data=bool(bundle), has_offline_wheels=bool(wheels),
                                                   starts_gpu=False))
    archive = out / "sol-nll-setup.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage))
    write_json(out / "build.json", dict(archive_sha256=file_hash(archive), archive_bytes=archive.stat().st_size,
                                        has_real_panel=bool(bundle), gpu_started=False))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wheels")
    parser.add_argument("--bundle")
    args = parser.parse_args()
    print(build(args.out, args.wheels, args.bundle))
