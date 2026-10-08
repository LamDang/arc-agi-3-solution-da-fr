"""Download only CPU preparation inputs from the existing DVC remote and HF.

No model weights, Kaggle session or GPU allocation. Uses configured AWS SDK
credentials without reading or printing them. Domain access is still required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import file_hash
from prepare import LOCKED_MD5, PROCESSOR_MD5

BUCKET = "kaggle-arc-agi-3-dvc"
CALIB_MD5 = "8b2bb7ee1cb7db8f1aaa646caff29f29.dir"
MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def dvc_download(client, md5, destination):
    destination = Path(destination)
    expected = md5.removesuffix(".dir")
    if destination.exists() and file_hash(destination, "md5") == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download.partial")
    client.download_file(BUCKET, f"files/md5/{md5[:2]}/{md5[2:]}", str(temporary))
    if file_hash(temporary, "md5") != expected:
        raise ValueError(f"Downloaded DVC hash mismatch: {destination.name}")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import boto3
    from botocore.config import Config
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    client = boto3.client("s3", region_name="eu-west-3", config=Config(retries={"max_attempts": 1},
                                                                      connect_timeout=15, read_timeout=60))
    for name, checksum in LOCKED_MD5.items():
        dvc_download(client, checksum, out / "data" / name)
        print(f"verified data/{name}", flush=True)
    tree = out / "calib-tree.json"
    dvc_download(client, CALIB_MD5, tree)
    for entry in json.loads(tree.read_text()):
        rel = Path(entry["relpath"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe DVC directory entry")
        if rel.parts[0] == "stats" and rel.suffix == ".npz":
            dvc_download(client, entry["md5"], out / "calib" / rel)
    snapshot_download(MODEL, revision=REVISION, local_dir=out / "processor",
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
                      ignore_patterns=["*safetensors*", "*pytorch_model*"])
    for name, checksum in PROCESSOR_MD5.items():
        if file_hash(out / "processor" / name, "md5") != checksum:
            raise ValueError(f"Pinned tokenizer/template mismatch: {name}")
    print("CPU inputs downloaded and checksummed; run prepare.py next.")


if __name__ == "__main__":
    main()
