"""Externally mirror completed NLL artifacts through Jupyter's contents API.

Run on a durable machine, not inside the ephemeral Kaggle session. Set
JUPYTER_BASE_URL and JUPYTER_TOKEN without printing them. This downloads
results only; it never starts a kernel/session or allocates a GPU.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from common import atomic_bytes, read_json, write_json


class Jupyter:
    def __init__(self, base, token):
        parsed = urlsplit(base)
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote Jupyter must use HTTPS")
        if parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Pass tokens in JUPYTER_TOKEN, not in the server URL")
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = "token " + token

    def get(self, path, content=True):
        r = self.session.get(self.base + "/api/contents/" + quote(path.strip("/"), safe="/"),
                             params={"content": int(content)}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Jupyter content request failed: HTTP {r.status_code}")
        return r.json()

    def file(self, path):
        value = self.get(path)
        if value["type"] != "file":
            raise ValueError("Expected a Jupyter file")
        return base64.b64decode(value["content"]) if value["format"] == "base64" else value["content"].encode()

    def list(self, path):
        value = self.get(path)
        if value["type"] != "directory":
            raise ValueError("Expected a Jupyter directory")
        return value["content"]

    def put_json(self, path, value):
        r = self.session.put(self.base + "/api/contents/" + quote(path.strip("/"), safe="/"),
                             json={"type": "file", "format": "text", "content": json.dumps(value)}, timeout=60)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Jupyter acknowledgment failed: HTTP {r.status_code}")


def mirror_once(client, remote, local):
    local = Path(local)
    run = json.loads(client.file(f"{remote}/run.json"))
    if (local / "run.json").exists() and read_json(local / "run.json") != run:
        raise ValueError("Collector run identity changed")
    write_json(local / "run.json", run)
    atomic_bytes(local / "manifest.json", client.file(f"{remote}/manifest.json"))
    copied = 0
    # A result .json is the completion marker. Download and verify the array
    # before making the corresponding local marker visible.
    for directory in client.list(f"{remote}/results"):
        if directory["type"] != "directory" or not directory["name"].isdigit():
            continue
        for item in client.list(directory["path"]):
            name = item["name"]
            if item["type"] != "file" or not name.endswith(".json") or Path(name).name != name:
                continue
            rel = Path("results") / directory["name"] / name
            marker = json.loads(client.file(f"{remote}/{rel.as_posix()}"))
            if marker["identity"] != run["identity"]:
                raise ValueError("Remote result identity mismatch")
            if (local / rel).exists() and read_json(local / rel) == marker:
                array = local / rel.with_suffix(".npz")
                if array.exists() and hashlib.sha256(array.read_bytes()).hexdigest() == marker["npz_sha256"]:
                    continue
            array_bytes = client.file(f"{remote}/{rel.with_suffix('.npz').as_posix()}")
            if hashlib.sha256(array_bytes).hexdigest() != marker["npz_sha256"]:
                raise ValueError("Remote result checksum mismatch")
            atomic_bytes(local / rel.with_suffix(".npz"), array_bytes)
            write_json(local / rel, marker)
            copied += 1
    for name in ("smoke.json", "heartbeat.json"):
        try:
            atomic_bytes(local / name, client.file(f"{remote}/{name}"))
        except RuntimeError:
            pass  # status may not exist at early startup; completed results are mandatory
    acknowledgement = dict(identity=run["identity"], last_verified_unix=time.time())
    write_json(local / "mirror.json", acknowledgement)
    client.put_json(f"{remote}/mirror-ack.json", acknowledgement)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sol-nll")
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    client = Jupyter(os.environ["JUPYTER_BASE_URL"], os.environ["JUPYTER_TOKEN"])
    while True:
        try:
            print(f"Verified {mirror_once(client, args.remote, args.out)} new results", flush=True)
        except Exception as exc:
            # Avoid printing token-bearing URLs or request headers on errors.
            print(f"Mirror failed ({type(exc).__name__}); durable state unchanged", flush=True)
            if args.once:
                raise SystemExit(1) from None
        if args.once:
            break
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
