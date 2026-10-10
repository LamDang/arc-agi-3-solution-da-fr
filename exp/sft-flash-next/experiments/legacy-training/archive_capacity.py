"""Bundle capacity successes, failures and exact source snapshots for private DVC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def archive(root, target):
    root, target = Path(root), Path(target)
    files = [p for p in sorted(root.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc'
             and p.name != 'inventory.json']
    inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo('capacity-results/'+str(p.relative_to(root)))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
        z.writestr('capacity-results/inventory.json', json.dumps(inventory, indent=2)+'\n')
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for name, expected in inventory.items():
            if hashlib.sha256(z.read('capacity-results/'+name)).hexdigest() != expected:
                raise ValueError('Archived bytes changed: '+name)
    raw = target.read_bytes()
    provenance = dict(files=len(files), bytes=len(raw), md5=hashlib.md5(raw).hexdigest(),
                      sha256=hashlib.sha256(raw).hexdigest(),
                      contents='Capacity successes/failures, source snapshots, storage probes and fresh fold-0 validation; no trained adapter')
    target.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    target.with_suffix(target.suffix+'.dvc').write_text(
        f"outs:\n- md5: {provenance['md5']}\n  size: {len(raw)}\n  hash: md5\n  path: {target.name}\n")
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.root, args.out), indent=2))
