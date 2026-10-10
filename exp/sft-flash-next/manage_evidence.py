"""Archive detailed experiment data in DVC; restore original reviewer paths.

Code, configurations and Markdown summaries stay in Git. The bundle contains
byte-for-byte copies and a SHA256 inventory, not shortened report substitutes.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "evidence-data"
MANIFEST = BUNDLE / "manifest.json"
DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".xml"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or path.parts[0] not in {"architecture", "train"}
            or path.suffix not in DATA_SUFFIXES or "configs" in path.parts):
        raise ValueError(f"Invalid evidence path: {value}")
    return Path(*path.parts)


def checked_path(base, relative):
    path = base / relative
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Evidence path escapes {base}: {relative}")
    return path


def inventory():
    data = json.loads(MANIFEST.read_text())
    if data["schema"] != 1:
        raise ValueError("Unsupported evidence inventory")
    return data["files"]


def archive(seed_paths=None):
    paths = {row["path"] for row in inventory()} if MANIFEST.exists() else set()
    if seed_paths:
        paths.update(json.loads(Path(seed_paths).read_text()))
    # Run archives under results/ and current gradient-results outputs already
    # have their own DVC pointers. Here we collect derived reports and metrics.
    for directory in ("architecture/reports", "train/reviews", "train/metrics"):
        paths.update(p.relative_to(ROOT).as_posix()
                     for p in (ROOT / directory).iterdir()
                     if p.is_file() and p.suffix in DATA_SUFFIXES)
    rows = []
    for value in sorted(paths):
        relative = relative_path(value)
        source = checked_path(ROOT, relative)
        target = checked_path(BUNDLE, relative)
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        elif not target.is_file():
            raise FileNotFoundError(source)
        rows.append(dict(path=value, bytes=target.stat().st_size,
                         sha256=digest(target)))
    BUNDLE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(schema=1, files=rows), indent=2) + "\n")
    return rows


def verify_or_restore(restore=False):
    rows = inventory()
    # Validate the whole bundle before materializing any original file.
    for row in rows:
        source = checked_path(BUNDLE, relative_path(row["path"]))
        if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
            raise ValueError(f"Corrupt evidence: {row['path']}")
        target = checked_path(ROOT, relative_path(row["path"]))
        if target.exists() and digest(target) != row["sha256"]:
            raise ValueError(f"Working file differs; archive or preserve it first: {target}")
    if restore:
        for row in rows:
            relative = relative_path(row["path"])
            target = checked_path(ROOT, relative)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checked_path(BUNDLE, relative), target)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "verify", "restore"))
    parser.add_argument("--paths-from", help="JSON list for a one-time migration")
    args = parser.parse_args()
    if args.action != "archive" and args.paths_from:
        parser.error("--paths-from is only valid with archive")
    rows = archive(args.paths_from) if args.action == "archive" else verify_or_restore(args.action == "restore")
    print(json.dumps(dict(action=args.action, files=len(rows),
                          bytes=sum(row["bytes"] for row in rows))))


if __name__ == "__main__":
    main()
