"""Reassemble and SHA-256 verify the recovery files from a cloned branch."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
args.output.mkdir(parents=True, exist_ok=True)
for item in manifest['files']:
    output = args.output / item['name']
    temporary = output.with_name(output.name + '.partial')
    digest = hashlib.sha256()
    size = 0
    with temporary.open('wb') as target:
        for part in item['parts']:
            encoded = (root / part['path']).read_bytes()
            git_digest = hashlib.sha1(b'blob ' + str(len(encoded)).encode() + b'\0' + encoded).hexdigest()
            if git_digest != part['git_blob_sha']:
                raise RuntimeError('Changed encoded part: ' + part['path'])
            data = base64.b64decode(encoded, validate=True)
            if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                raise RuntimeError('Corrupt part: ' + part['path'])
            target.write(data)
            digest.update(data)
            size += len(data)
    if size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise RuntimeError('Full-file checksum mismatch: ' + item['name'])
    temporary.replace(output)
    print(str(output) + ' verified SHA-256 ' + digest.hexdigest())
