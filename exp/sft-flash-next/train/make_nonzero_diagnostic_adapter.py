"""Derive a TEST-ONLY nonzero-A/nonzero-B PEFT state from the fixed zero-B anchor.

The saved adapter file, not the seed, is the authority for GPU replays. This
script never loads a model, sample, optimizer, or production checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


PURPOSE = 'diagnostic-gradient-qualification-only'
ALGORITHM = 'Keep anchor A exactly; fill each B in sorted name order with CPU torch.randn(float32, generator=manual_seed(seed)) * std'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha256(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_zero_b_anchor(state, *, expected_tensors=744):
    a = {name for name in state if 'lora_A' in name}
    b = {name for name in state if 'lora_B' in name}
    if len(state) != expected_tensors or len(a) != len(b) or a != {
        name.replace('lora_B', 'lora_A') for name in b
    }:
        raise ValueError('Anchor must contain paired A/B adapter tensors only')
    for name, value in state.items():
        if value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Anchor tensor must be finite CPU float32: {name}')
        if 'lora_A' in name and not torch.count_nonzero(value):
            raise ValueError(f'Anchor A is zero: {name}')
        if 'lora_B' in name and torch.count_nonzero(value):
            raise ValueError(f'Anchor B is not zero: {name}')


def derive_state(anchor, *, seed, std):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    if not 0 < std < 0.1:
        raise ValueError('Diagnostic B standard deviation must be in (0, 0.1)')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    result = {}
    for name in sorted(anchor):
        value = anchor[name].detach().cpu().contiguous().clone()
        if 'lora_B' in name:
            value = torch.randn(value.shape, generator=generator, dtype=torch.float32) * std
            if not torch.count_nonzero(value):
                raise ValueError(f'Derived B is zero: {name}')
        result[name] = value
    return result


def tensor_manifest(state):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'nonzero': int(torch.count_nonzero(value)),
                   'sha256': tensor_sha256(value)}
            for name, value in state.items()}


def tensor_state_sha256(state):
    """Canonical content hash, independent of torch.save container metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        for field in (name, str(value.dtype), json.dumps(list(value.shape))):
            encoded = field.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Exact saved zero-B diagnostic anchor .pt')
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--out', required=True, help='New directory for the test-only .pt and manifest')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--b-std', type=float, default=0.003)
    args = parser.parse_args()

    source = Path(args.source)
    source_hash = file_sha256(source)
    if source_hash != args.expected_source_sha256:
        raise ValueError('Diagnostic anchor file hash mismatch')
    anchor = torch.load(source, map_location='cpu', weights_only=True)
    validate_zero_b_anchor(anchor)
    first = derive_state(anchor, seed=args.seed, std=args.b_std)
    second = derive_state(anchor, seed=args.seed, std=args.b_std)
    if any(not torch.equal(first[name], second[name]) for name in first):
        raise RuntimeError('Same-process CPU regeneration was not bitwise equal')
    if any(not torch.equal(first[name], anchor[name]) for name in first if 'lora_A' in name):
        raise RuntimeError('Diagnostic construction changed an A tensor')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    adapter_path = out / 'adapter.pt'
    torch.save(first, adapter_path)
    saved = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if set(saved) != set(first) or any(not torch.equal(saved[name], first[name]) for name in first):
        raise RuntimeError('Saved adapter did not replay exactly')
    a_count = sum(bool(torch.count_nonzero(value)) for name, value in first.items() if 'lora_A' in name)
    b_count = sum(bool(torch.count_nonzero(value)) for name, value in first.items() if 'lora_B' in name)
    if (a_count, b_count) != (372, 372):
        raise RuntimeError('Diagnostic adapter needs all 372 A and all 372 B matrices nonzero')
    manifest = {
        'purpose': PURPOSE,
        'production_eligible': False,
        'validation_exposed': True,
        'optimizer_updates': 0,
        'source_zero_b_adapter_sha256': source_hash,
        'adapter_sha256': file_sha256(adapter_path),
        'tensor_state_sha256': tensor_state_sha256(first),
        'script_sha256': file_sha256(__file__),
        'seed': args.seed,
        'algorithm': ALGORITHM,
        'b_std': args.b_std,
        'torch_version': torch.__version__,
        'rng_device': 'cpu',
        'dtype': 'torch.float32',
        'tensor_count': len(first),
        'a_nonzero_tensors': a_count,
        'b_nonzero_tensors': b_count,
        'same_process_regeneration_bitwise_equal': True,
        'saved_replay_bitwise_equal': True,
        'tensors': tensor_manifest(first),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'tensors'}))


if __name__ == '__main__':
    main()
