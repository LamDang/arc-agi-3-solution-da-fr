"""Serialize one intact prepared request as a safe, tensor-only audit input."""
import argparse
from pathlib import Path

import torch

from dataset import Requests, file_hash, write_json


def prepare(bundle_path, sample_id, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bundle = Requests(bundle_path)
    matches = [row for row in bundle.rows if row['sample_id'] == sample_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one request for {sample_id}; got {len(matches)}')
    row = matches[0]
    encoded, annotation = bundle.get(row)
    # BatchFeature is a Python class; a plain tensor dict supports weights_only=True.
    encoded = dict(encoded)
    if not all(isinstance(value, torch.Tensor) for value in encoded.values()):
        raise ValueError('Audit input must contain tensors only')
    torch.save(encoded, out / 'sample.pt')
    write_json(out / 'sample.json', dict(row=row, annotation=annotation,
        manifest_sha256=bundle.manifest['sha256'], sample_sha256=file_hash(out / 'sample.pt')))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(prepare(args.bundle, args.sample_id, args.out))
