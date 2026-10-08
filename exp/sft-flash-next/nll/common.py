"""Portable, credential-free manifests and atomic evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = 1
COUNTS = (512, 448, 384, 320, 256)
LABELS = ("thinking", "tool_code", "tool_format", "assistant_text", "turn_format", "boundary")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def length_band(n):
    return ("<16K" if n < 16384 else "16K-<48K" if n < 49152 else
            "48K-<80K" if n < 81920 else ">=80K")


def verify_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(body) != manifest.get("manifest_sha256"):
        raise ValueError("Manifest checksum mismatch")


def code_hash(directory, reap_directory):
    paths = list(Path(directory).glob("*.py")) + [Path(reap_directory) / n for n in
                                                   ("replay.py", "reap_model.py", "render.py")]
    return digest({("reap/" if p.parent == Path(reap_directory) else "nll/") + p.name:
                   file_hash(p) for p in paths})
