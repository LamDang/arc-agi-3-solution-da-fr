"""Checksum-verified, restartable request/model results; no model state needed."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from common import atomic_bytes, digest, file_hash, read_json, write_json


def paths(root, count, sample_id):
    if any(c in sample_id for c in ("/", "\\", "..")):
        raise ValueError("Invalid sample ID")
    base = Path(root) / "results" / str(int(count)) / sample_id
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_result(root, identity, count, sample, nll, positions, **metrics):
    nll = np.asarray(nll, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.int64)
    if nll.shape != (sample["target_tokens"],) or not np.isfinite(nll).all() or (nll < 0).any():
        raise ValueError("Invalid token loss vector")
    if not np.array_equal(positions, sample["positions"]):
        raise ValueError("Scored positions differ from frozen target")
    array_path, marker_path = paths(root, count, sample["sample_id"])
    stream = io.BytesIO()
    np.savez_compressed(stream, nll=nll, positions=positions)
    atomic_bytes(array_path, stream.getvalue())
    metadata = dict(identity=identity, count=count, sample_id=sample["sample_id"],
                    target_sha256=sample["target_sha256"], scored=len(nll),
                    nll_sum=float(nll.astype(np.float64).sum()), npz_sha256=file_hash(array_path),
                    metrics=metrics)
    write_json(marker_path, metadata)  # completion is last, after the array is durable
    return metadata


def load_result(root, identity, count, sample):
    array_path, marker_path = paths(root, count, sample["sample_id"])
    if not marker_path.exists():
        return None  # an orphan .npz or .partial is not completion
    meta = read_json(marker_path)
    if (meta["identity"] != identity or meta["target_sha256"] != sample["target_sha256"] or
            meta["sample_id"] != sample["sample_id"] or meta["count"] != count):
        raise ValueError("Result identity mismatch; use a separate output directory")
    if file_hash(array_path) != meta["npz_sha256"]:
        raise ValueError(f"Corrupt result: {array_path}")
    with np.load(array_path, allow_pickle=False) as archive:
        nll, positions = archive["nll"], archive["positions"]
    if (nll.shape != (sample["target_tokens"],) or not np.isfinite(nll).all() or (nll < 0).any()
            or not np.array_equal(positions, sample["positions"])):
        raise ValueError("Invalid completed result arrays")
    if not np.isclose(nll.astype(np.float64).sum(), meta["nll_sum"], rtol=1e-12, atol=1e-12):
        raise ValueError("Result sum mismatch")
    return meta, nll


def bind_run(root, config):
    path = Path(root) / "run.json"
    identity = digest(config)
    value = dict(identity=identity, config=config)
    if path.exists() and read_json(path) != value:
        raise ValueError("Resume configuration changed; use a new output directory")
    write_json(path, value)
    return identity
