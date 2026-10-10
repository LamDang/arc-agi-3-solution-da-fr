"""Prune historical full-sample gradients; preserve inputs, sources and reports.

No remote DVC operations. The sole retained raw reference is pinned below.
Historical SHA inventories remain unchanged; retention records explain omissions.
Run from repository root with --apply after inspecting the JSON dry-run.
"""
import argparse
import hashlib
import json
from pathlib import Path

KEEP = '20261009221135477-25bf58fe'
ROOT = Path('exp/sft-flash-next')
CACHE = Path('.dvc/cache/files/md5')
REPORT = ROOT/'architecture/reports/gradient-retention.json'


def retired(path):
    path=Path(path)
    return path.suffix == '.pt' and (
        'gradient' in path.name.lower() or 'dhidden' in path.name.lower())


def digest(path, algorithm):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,algorithm).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if args.apply and REPORT.exists():
        raise SystemExit('Cleanup already recorded; refusing to overwrite the deletion inventory.')
    files=[p for p in ROOT.rglob('*.pt') if KEEP not in p.parts and retired(p)]
    # Never remove tracked code or fixture assets accidentally.
    import subprocess
    tracked=set(subprocess.check_output(['git','ls-files'],text=True).splitlines())
    assert not tracked.intersection(map(str,files))
    pointers={p.with_suffix(''):p for p in ROOT.rglob('*.dvc') if p.with_suffix('').is_dir()}
    import re
    pipeline=ROOT/'train/dvc.yaml'
    for relative in re.findall(r'^    - path: (gradient-results/[^\n]+)$',(ROOT/'train/dvc.lock').read_text(),re.M):
        base=ROOT/'train'/relative
        if base.is_dir():pointers[base]=pipeline
    retired_hashes=set();protected=set()
    for manifest in CACHE.glob('*/*.dir'):
        for row in json.loads(manifest.read_text()):
            h=row['md5']
            if retired(row['relpath']):retired_hashes.add(h)
            else:protected.add(h)
    # Sharded reference files are protected by the non-retired numeric filenames.
    cache_files=[CACHE/h[:2]/h[2:] for h in sorted(retired_hashes-protected)]
    cache_files=[p for p in cache_files if p.is_file()]
    summary=dict(retained_reference=KEEP,workspace_files=len(files),
        workspace_logical_bytes=sum(p.stat().st_size for p in files),
        cache_objects=len(cache_files),cache_logical_bytes=sum(p.stat().st_size for p in cache_files),
        physical_bytes_note='Logical bytes; APFS clones/dedup mean these cannot be added as physical freed space.',
        affected_dvc_paths=sorted(str(base) for base in pointers if any(p.is_relative_to(base) for p in files)))
    print(json.dumps(summary,indent=2),flush=True)
    if not args.apply:return
    rows=[];groups={}
    for p in files:
        row=dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p,'sha256'),md5=digest(p,'md5'))
        rows.append(row)
        owners=[base for base in pointers if p.is_relative_to(base)]
        if not owners:raise RuntimeError('No DVC evidence owner: '+str(p))
        owner=max(owners,key=lambda x:len(x.parts));groups.setdefault(owner,[]).append(row)
    summary.update(removed_files=rows,removed_cache_objects=[dict(md5=p.parent.name+p.name,bytes=p.stat().st_size) for p in cache_files],
        policy='Only current reference retains raw full-sample gradients; later experiments retain comparisons.',
        remote_dvc_modified=False,historical_manifests='Original file hashes/results preserved; listed raw files intentionally retired.')
    # Commit a deletion inventory before unlinking anything. Cache deletion is targeted,
    # never dvc gc; objects also referenced under non-gradient names are protected.
    REPORT.write_text(json.dumps(summary,indent=2)+'\n')
    for owner,items in groups.items():
        (owner/'gradient-retention.json').write_text(json.dumps(dict(policy=summary['policy'],retained_reference=KEEP,
            removed_files=items,historical_manifests=summary['historical_manifests']),indent=2)+'\n')
    for p in files:p.unlink()
    for p in cache_files:p.unlink()


if __name__ == '__main__':main()
