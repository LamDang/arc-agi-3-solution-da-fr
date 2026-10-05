"""Pack finished run directories for archiving, or unpack them.

    uv run --no-sync python scripts/pack_run.py pack runs/<run> [runs/<run> ...]
    uv run --no-sync python scripts/pack_run.py unpack runs/<run>
    uv run --no-sync python scripts/pack_run.py status runs/<run>

pack replaces what it can rebuild exactly: the boards in the viewer's event
logs are replayed from benchmark.json's actions, the transcript pages and the
transcript text in the event logs come from the transcripts, and every other
large file is compressed with xz. pack.json lists every replaced file with its
sha256; pack removes a file only after rebuilding it and comparing the bytes.
unpack rebuilds the files and checks the hashes. The viewer, `make traces` and
RESUME_FROM unpack a packed run by themselves. See inference/utils/run_pack.py
and LOCAL_EVAL.md, "Pack a run".
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from inference.utils.run_pack import is_packed, load_manifest, pack_run, unpack_run


def _status(run_dir: Path) -> str:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return f"{run_dir}: not packed"
    files = manifest["files"]
    missing = [relative for relative in files if not (run_dir / relative).exists()]
    state = "packed" if len(missing) == len(files) else "unpacked" if not missing else "partly unpacked"
    methods = dict(Counter(entry["method"] for entry in files.values()))
    return f"{run_dir}: {state}; {len(files)} files replaced ({methods}), {len(missing)} of them absent"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["pack", "unpack", "status"])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--environments-dir",
        type=Path,
        default=None,
        help="Game files for the replay. Default: $ENVIRONMENTS_DIR, else environment_files/.",
    )
    args = parser.parse_args()
    for run_dir in args.runs:
        if not (run_dir / "benchmark.json").exists():
            print(f"{run_dir}: no benchmark.json; not a run directory", file=sys.stderr)
            return 1
        if args.command == "status":
            print(_status(run_dir))
        elif args.command == "pack":
            report = pack_run(run_dir, environments_dir=args.environments_dir)
            print(
                f"{run_dir}: {report.bytes_before / 1e6:.1f} MB -> {report.bytes_after / 1e6:.1f} MB; "
                f"{len(report.packed)} files replaced ({dict(Counter(report.packed.values()))})"
            )
            for relative, reason in report.kept.items():
                print(f"  kept as is: {relative} ({reason})")
        else:
            if not is_packed(run_dir):
                print(f"{run_dir}: not packed")
                continue
            restored = unpack_run(run_dir, environments_dir=args.environments_dir)
            print(f"{run_dir}: rebuilt {len(restored)} files; all match pack.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
