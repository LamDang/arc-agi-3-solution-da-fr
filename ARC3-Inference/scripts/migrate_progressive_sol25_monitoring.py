"""Record the monitor-only retry behavior change before resuming Sol25."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ARC3-Inference"))

from think_gen import logs  # noqa: E402
from think_gen.progressive import atomic_json, build_manifest, file_hash, read_json  # noqa: E402

OUT = ROOT / "ARC3-Inference/runs/think-progressive-sampled-pilot"
RUN = ROOT / "ARC3-Inference/runs/gpt61sol-features-25games"
TOKENIZER = ROOT / "data/sft-gpt61sol-features-25games/qwen/tokenizer.json"
TEMPLATE = ROOT / "data/sft-gpt61sol-features-25games/qwen/chat_template.jinja"
MIGRATION = OUT / "monitoring-optional-migration.json"
BACKUP = OUT / "manifest.before-monitoring-optional.json"


def checkpoint_hashes():
    return {str(path.relative_to(OUT)): file_hash(path)
            for path in sorted(OUT.glob("turns/*/*/calls/*/complete.json"))}


def main() -> None:
    lock = (OUT / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("A generation or recovery process holds the run lock") from exc
    if read_json(OUT / "driver_status.json").get("phase") != "stopped":
        raise RuntimeError("Manifest migration is only allowed while the driver is stopped")

    manifest_path = OUT / "manifest.json"
    current = read_json(manifest_path)
    current_hash = file_hash(manifest_path)
    if MIGRATION.exists():
        previous = read_json(MIGRATION)
        if previous.get("new_manifest_sha256") == current_hash:
            print(json.dumps({"already_migrated": True, "manifest_sha256": current_hash}))
            return
        raise RuntimeError("A different monitoring migration is already recorded")
    old = read_json(BACKUP) if BACKUP.exists() else current
    old_hash = file_hash(BACKUP) if BACKUP.exists() else current_hash

    settings = old["settings"]
    expected_settings = {
        "model": "qwen/qwen3.8-flash", "provider": "Alibaba",
        "judge_model": "gpt-6.1-sol", "max_tokens": 8192,
        "sol_max_tokens": 16000, "input_limit": 120000,
        "cache_retention": "default", "sol_prices": [2.0, 0.1, 10.0],
        "monitor_per_game": 15,
    }
    if settings != expected_settings:
        raise RuntimeError(f"Unexpected production settings: {settings}")
    args = SimpleNamespace(run=RUN, games=None, tokenizer=TOKENIZER, template=TEMPLATE, **settings)
    new = build_manifest(args, logs.request_logs(RUN))
    if any(old[key] != new[key] for key in old if key != "code"):
        raise RuntimeError("Source logs, model settings, tokenizer, or template changed")
    changed = {key: {"old": old["code"].get(key), "new": new["code"].get(key)}
               for key in set(old["code"]) | set(new["code"])
               if old["code"].get(key) != new["code"].get(key)}
    if set(changed) != {"ARC3-Inference/think_gen/progressive.py"}:
        raise RuntimeError(f"Unexpected inference code changes: {changed}")
    if current not in (old, new):
        raise RuntimeError("Current manifest is neither the saved pre-migration nor intended manifest")

    before = checkpoint_hashes()
    if len(before) < 1000:
        raise RuntimeError(f"Too few completed stage checkpoints to safely resume: {len(before)}")
    # These separate atomic writes are restartable: the backup is durable before
    # replacement, so a crash before the migration record can be repaired.
    if not BACKUP.exists():
        atomic_json(BACKUP, old)
        old_hash = file_hash(BACKUP)
    if current != new:
        atomic_json(manifest_path, new)
    after = checkpoint_hashes()
    if before != after:
        atomic_json(manifest_path, old)
        raise RuntimeError("A completed stage checkpoint changed during migration")
    migration = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Optional sampled regen/final-audit failures must not block final-thinking generation; core stages remain fatal.",
        "old_manifest_sha256": old_hash,
        "new_manifest_sha256": file_hash(manifest_path),
        "changed_code": changed,
        "unchanged_manifest_sections": ["source", "tokenizer", "template", "settings", "cache_policy"],
        "preserved_completed_stage_count": len(before),
        "preserved_completed_stages": before,
    }
    atomic_json(MIGRATION, migration)
    print(json.dumps({"migrated": True, "old_manifest_sha256": old_hash,
                      "new_manifest_sha256": migration["new_manifest_sha256"],
                      "changed_code": changed,
                      "preserved_completed_stage_count": len(before)}, indent=2))


if __name__ == "__main__":
    main()
