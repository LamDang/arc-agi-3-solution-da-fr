"""Reconcile definite failed HTTP attempts without changing their API journals."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))

from think_gen import logs  # noqa: E402
from think_gen.progressive import (  # noqa: E402
    HTTP_BILLING_ADJUSTMENTS, atomic_json, build_manifest,
    explicit_http_failure_status, file_hash, load_billing_adjustments, read_json,
)

OUT = ROOT / "ARC3-Inference/runs/think-progressive-sampled-pilot"
RUN = ROOT / "ARC3-Inference/runs/gpt61sol-features-25games"
TOKENIZER = ROOT / "data/sft-gpt61sol-features-25games/qwen/tokenizer.json"
TEMPLATE = ROOT / "data/sft-gpt61sol-features-25games/qwen/chat_template.jinja"
RECORD = OUT / "http-failure-billing-migration.json"
BACKUP = OUT / "manifest.before-http-failure-billing.json"
PILOT_BACKUP = OUT / "pilot-validated.before-http-failure-billing.json"
ADJUSTMENTS = OUT / HTTP_BILLING_ADJUSTMENTS


def checkpoint_fingerprint(pattern):
    paths = {str(p.relative_to(OUT)): file_hash(p) for p in sorted(OUT.glob(pattern))}
    sha = hashlib.sha256(json.dumps(paths, sort_keys=True).encode()).hexdigest()
    return {"count": len(paths), "sha256": sha}


def adjustment_rows():
    rows = []
    for path in sorted(OUT.glob("turns/*/*/calls/*/attempt-*.json")):
        row = read_json(path)
        if row["state"] in ("pending", "reserved", "rate_limited"):
            raise RuntimeError(f"An API request is unfinished: {path}")
        if row["state"] != "error_uncertain" or row.get("response") is not None:
            continue
        status = explicit_http_failure_status(row.get("error", ""))
        if status is None:
            continue
        rows.append({
            "path": str(path.relative_to(OUT)),
            "sha256": file_hash(path),
            "http_status": status,
            "original_reservation_usd": row["charged_usd"],
            "effective_charge_usd": 0.0,
        })
    return rows


def main():
    with (OUT / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The generation process holds the run lock") from exc
        if read_json(OUT / "driver_status.json").get("phase") != "stopped":
            raise RuntimeError("Billing migration requires a stopped generation driver")

        manifest_path = OUT / "manifest.json"
        current = read_json(manifest_path)
        old = read_json(BACKUP) if BACKUP.exists() else current
        old_hash = file_hash(BACKUP) if BACKUP.exists() else file_hash(manifest_path)
        expected_settings = {
            "model": "qwen/qwen3.8-flash", "provider": "Alibaba",
            "judge_model": "gpt-6.1-sol", "max_tokens": 8192,
            "sol_max_tokens": 16000, "input_limit": 120000,
            "cache_retention": "default", "sol_prices": [2.0, 0.1, 10.0],
            "monitor_per_game": 15,
        }
        if old["settings"] != expected_settings:
            raise RuntimeError("Unexpected production generation settings")
        args = SimpleNamespace(run=RUN, games=None, tokenizer=TOKENIZER,
                               template=TEMPLATE, **old["settings"])
        new = build_manifest(args, logs.request_logs(RUN))
        if any(old[key] != new[key] for key in old if key != "code"):
            raise RuntimeError("Source, tokenizer, template, or settings changed")
        changed = {key: {"old": old["code"].get(key), "new": new["code"].get(key)}
                   for key in set(old["code"]) | set(new["code"])
                   if old["code"].get(key) != new["code"].get(key)}
        if set(changed) != {"ARC3-Inference/think_gen/client.py",
                            "ARC3-Inference/think_gen/progressive.py"}:
            raise RuntimeError(f"Unexpected generation-code changes: {changed}")
        if current not in (old, new):
            raise RuntimeError("Current manifest does not match either migration endpoint")
        if RECORD.exists():
            recorded = read_json(RECORD)
            if (current == new and recorded.get("new_manifest_sha256") == file_hash(manifest_path)
                    and recorded.get("adjustments_sha256") == file_hash(ADJUSTMENTS)):
                load_billing_adjustments(OUT)
                print(json.dumps({"already_migrated": True, "manifest_sha256": file_hash(manifest_path)}))
                return
            raise RuntimeError("A different HTTP billing migration is already recorded")

        before = {kind: checkpoint_fingerprint(pattern) for kind, pattern in (
            ("attempts", "turns/*/*/calls/*/attempt-*.json"),
            ("stages", "turns/*/*/calls/*/complete.json"),
            ("finals", "turns/*/*/final.json"),
        )}
        if before["finals"]["count"] != 1004:
            raise RuntimeError(f"Unexpected finalized count: {before['finals']['count']}")
        rows = adjustment_rows()
        if len(rows) != 250:
            raise RuntimeError(f"Unexpected definite HTTP failure count: {len(rows)}")
        sidecar = {"version": 1, "reason": "Definite failed HTTP responses have no provider charge; preserve original journals.",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "adjustments": rows,
                   "original_reservations_usd": sum(row["original_reservation_usd"] for row in rows)}
        if not BACKUP.exists():
            atomic_json(BACKUP, old)
            old_hash = file_hash(BACKUP)
        if ADJUSTMENTS.exists():
            existing = read_json(ADJUSTMENTS)
            if existing.get("version") != 1 or existing.get("adjustments") != rows:
                raise RuntimeError("Existing billing adjustments differ from the original journals")
        else:
            atomic_json(ADJUSTMENTS, sidecar)
        if len(load_billing_adjustments(OUT)) != len(rows):
            raise RuntimeError("The billing adjustments did not validate")
        if current != new:
            atomic_json(manifest_path, new)
        new_hash = file_hash(manifest_path)
        after = {kind: checkpoint_fingerprint(pattern) for kind, pattern in (
            ("attempts", "turns/*/*/calls/*/attempt-*.json"),
            ("stages", "turns/*/*/calls/*/complete.json"),
            ("finals", "turns/*/*/final.json"),
        )}
        if before != after:
            raise RuntimeError("A saved API journal, stage, or final changed during migration")

        pilot_path = OUT / "pilot-validated.json"
        pilot = read_json(pilot_path)
        if pilot.get("manifest_hash") not in (old_hash, new_hash):
            raise RuntimeError("Pilot validation belongs to a different manifest")
        if not PILOT_BACKUP.exists():
            atomic_json(PILOT_BACKUP, pilot)
        if pilot["manifest_hash"] != new_hash:
            pilot["manifest_hash"] = new_hash
            pilot["billing_revalidation"] = {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "Only failed-HTTP billing and retry accounting changed; source and model settings match.",
                "preserved_final_count": before["finals"]["count"],
            }
            atomic_json(pilot_path, pilot)
        record = {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "old_manifest_sha256": old_hash,
            "new_manifest_sha256": new_hash,
            "changed_code": changed,
            "unchanged_manifest_sections": ["source", "tokenizer", "template", "settings", "cache_policy"],
            "preserved_checkpoints": before,
            "adjusted_http_failures": len(rows),
            "released_reservations_usd": sidecar["original_reservations_usd"],
            "adjustments_sha256": file_hash(ADJUSTMENTS),
        }
        atomic_json(RECORD, record)
        print(json.dumps({"migrated": True, **record}, indent=2))


if __name__ == "__main__":
    main()
