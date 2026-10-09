"""Verify and apply the 1,039-turn overlay after the 1,004-turn base restore."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RUN = Path('ARC3-Inference/runs/think-progressive-sampled-pilot')
STATUS = {'summary.json', 'driver_status.json', 'release-status.json'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def count(root):
    out = root / RUN
    return {'attempts':len(list(out.glob('turns/*/*/calls/*/attempt-*.json'))),
            'stages':len(list(out.glob('turns/*/*/calls/*/complete.json'))),
            'finals':len(list(out.glob('turns/*/*/final.json')))}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True, help='Active release checkout after base restoration')
    root = ap.parse_args().output.resolve()
    manifest = json.loads((HERE/'manifest.json').read_text())
    if sha((root/RUN/'manifest.json').read_bytes()) != manifest['generation_manifest_sha256']:
        raise RuntimeError('Base run manifest mismatch')
    before = count(root)
    allowed = ({'attempts':4591,'stages':4342,'finals':1004}, manifest['full_counts'])
    if before not in allowed:
        raise RuntimeError(f'Unexpected base checkpoint counts: {before}')
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp)/manifest['archive']
        with archive.open('wb') as f:
            for part in manifest['parts']:
                encoded = (HERE/part['name']).read_bytes()
                blob = hashlib.sha1(b'blob '+str(len(encoded)).encode()+b'\0'+encoded).hexdigest()
                if blob != part['git_blob_sha']:
                    raise RuntimeError('Part Git blob mismatch: '+part['name'])
                data = base64.b64decode(encoded, validate=False)
                if len(data) != part['bytes'] or sha(data) != part['sha256']:
                    raise RuntimeError('Part checksum mismatch: '+part['name'])
                f.write(data)
        if archive.stat().st_size != manifest['archive_bytes'] or sha(archive.read_bytes()) != manifest['archive_sha256']:
            raise RuntimeError('Full archive checksum mismatch')
        subprocess.run(['zstd','-t',str(archive)], check=True, stdout=subprocess.DEVNULL)
        names = subprocess.check_output(['tar','-I','zstd','-tf',str(archive)], text=True).splitlines()
        if len(names) != manifest['overlay_files'] or len(names) != len(set(names)):
            raise RuntimeError('Unexpected overlay file listing')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or path.parts[:3] != RUN.parts:
                raise RuntimeError('Unsafe overlay path: '+name)
            tail = path.parts[3:]
            if not (len(tail) == 1 and tail[0] in STATUS) and not (len(tail) >= 4 and tail[0] == 'turns'):
                raise RuntimeError('Unexpected overlay path: '+name)
            if before['finals'] == 1004 and (root/path).exists() and tail[0] == 'turns':
                raise RuntimeError('Overlay would overwrite a base turn file: '+name)
        if before['finals'] == 1004:
            subprocess.run(['tar','--no-same-owner','-I','zstd','-xf',str(archive),'-C',str(root)], check=True)
    after = count(root)
    if after != manifest['full_counts']:
        raise RuntimeError(f'Overlay counts mismatch: {after}')
    print('Verified Sol25 overlay:', after)

if __name__ == '__main__':
    main()
