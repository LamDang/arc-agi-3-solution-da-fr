"""Desktop checkpoint publication; execute with a Python that has DVC installed.

Each run uses its own hardlink/reflink cache. Collection only removes obsolete
objects in that dedicated cache, never the repository's shared DVC cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def publish(repository, run, manifest_path):
    from dvc.repo import Repo
    repository,run,manifest_path = map(lambda p: Path(p).resolve(), (repository,run,manifest_path))
    if not run.is_relative_to(repository):
        raise ValueError('Training checkpoints must be inside the working repository')
    manifest = json.loads(manifest_path.read_text())
    expected = hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k != 'sha256'},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if manifest.get('version') != 1 or manifest['sha256'] != expected:
        raise ValueError('Checkpoint manifest checksum/version mismatch')
    name = manifest['checkpoint']
    if not name.startswith('update-') or not name[7:].isdigit():
        raise ValueError('Unsafe checkpoint name')
    for row in manifest['shards']:
        if Path(row['path']).name != row['path']:
            raise ValueError('Unsafe checkpoint shard')
        p = manifest_path.parent/row['path']
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Checkpoint shard not durably collected')
    destination = run/'checkpoints'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.parent != destination:
        if destination.exists():
            raise FileExistsError('Different snapshot already occupies checkpoint version')
        manifest_path.parent.rename(destination)
    cache = run/'dvc-cache'
    cache.mkdir(exist_ok=True)
    config = {'cache': {'dir': str(cache), 'type': ['reflink', 'hardlink']},
              'core': {'autostage': False}}
    latest = run/'latest.json'
    with Repo(str(repository), config=config) as repo:
        repo.add(str(destination))
        pointer = destination.with_suffix('.dvc')
        record = dict(checkpoint=str(destination.relative_to(run)),
                      manifest_sha256=manifest['sha256'], dvc_pointer=str(pointer.relative_to(repository)),
                      dvc_cache=str(cache.relative_to(repository)), updates=manifest['progress']['updates'])
        temporary = latest.with_suffix('.json.part')
        with temporary.open('w') as stream:
            json.dump(record, stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temporary.replace(latest)
        repo.add(str(latest))
        telemetry = run/'telemetry'
        if telemetry.exists() and any(telemetry.iterdir()):repo.add(str(telemetry))
        # DVC may replace workspace files with cache links. Flush those final
        # inodes and pointers before acknowledging or removing the old version.
        durable = [*destination.iterdir(), pointer, latest, latest.with_suffix('.json.dvc')]
        if telemetry.with_suffix('.dvc').exists():durable.append(telemetry.with_suffix('.dvc'))
        for item in durable:
            if item.is_file():
                with item.open('rb') as stream:os.fsync(stream.fileno())
        fd_checkpoint = os.open(destination, os.O_RDONLY)
        try:os.fsync(fd_checkpoint)
        finally:os.close(fd_checkpoint)
        fd = os.open(run, os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        # The new pointer and latest marker are durable before retiring the old
        # version. Never remove arbitrary paths supplied by a corrupt marker.
        for old in destination.parent.iterdir():
            if old.is_dir() and re.fullmatch(r'update-\d{8}', old.name) and old.name < name:
                old.with_suffix('.dvc').unlink(missing_ok=True)
                shutil.rmtree(old)
        # Repo is explicitly bound to this run's dedicated cache. GC doesn't
        # touch the shared cache or the remote and keeps current workspace refs.
        repo.gc(workspace=True, force=True, skip_failed=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.repository,args.run,args.manifest)))


if __name__ == '__main__':
    main()
