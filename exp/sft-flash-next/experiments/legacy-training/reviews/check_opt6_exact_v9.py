"""Audit archived Opt6 operator pairs on CPU; never executes training or Torch.

Run from repository root with NumPy installed. Reports candidate rejection as
data (passed=false); exit success means the archived evidence was verified.
"""
import collections
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('exp/sft-flash-next/train')
FOLDER = ROOT / 'gradient-results/liger-opt6-overfit-v9-stopped'


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch' and name in ('FloatStorage', 'BFloat16Storage'):
            return {'FloatStorage': '<f4', 'BFloat16Storage': '<u2'}[name]
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda storage, offset, size, stride, *rest: (storage, offset, size, stride)
        raise ValueError((module, name))

    def persistent_load(self, item):
        kind, dtype, key, device, numel = item
        assert kind == 'storage' and device == 'cpu'
        return key, numel, dtype


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_pair(archive, prefix, reference, candidate):
    def tensor(meta):
        (key, stored, dtype), offset, size, stride = meta
        dt = np.dtype(dtype)
        raw = archive.read(prefix + 'data/' + key)
        assert len(raw) == stored * dt.itemsize
        return np.ndarray(size, dtype=dt, buffer=raw, offset=offset * dt.itemsize,
                          strides=tuple(x * dt.itemsize for x in stride))

    a, b = tensor(reference), tensor(candidate)
    assert a.dtype == b.dtype and a.shape == b.shape
    def values(x):
        return (x.astype(np.uint32) << 16).view(np.float32) if x.dtype == np.dtype('<u2') else x
    x, y = values(a).astype(np.float64), values(b).astype(np.float64)
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    bit_type = np.uint16 if a.dtype.itemsize == 2 else np.uint32
    mismatches = int(np.count_nonzero(a.view(bit_type) != b.view(bit_type)))
    return dict(bitwise_equal=mismatches == 0, mismatched_elements=mismatches,
                elements=a.size, dtype='bfloat16' if a.dtype.itemsize == 2 else 'float32',
                finite=finite, max_absolute_difference=float(np.max(np.abs(x-y))),
                relative_l2=float(np.linalg.norm(x-y) / max(float(np.linalg.norm(x)), 1e-30)))


cases = []
for path in sorted(FOLDER.glob('launch-opt6-check-*.pt')):
    with zipfile.ZipFile(path) as archive:
        prefix = next(n[:-len('data.pkl')] for n in archive.namelist() if n.endswith('/data.pkl'))
        assert archive.read(prefix + 'byteorder') == b'little'
        state = Reader(io.BytesIO(archive.read(prefix + 'data.pkl'))).load()
        output = audit_pair(archive, prefix, state['reference_output'], state['candidate_output'])
        assert len(state['reference_gradients']) == len(state['candidate_gradients'])
        gradients = [audit_pair(archive, prefix, a, b) for a, b in
                     zip(state['reference_gradients'], state['candidate_gradients'])]
        cases.append(dict(file=path.name, sha256=sha(path), output=output, gradients=gradients,
                          passed=all(x['bitwise_equal'] and x['finite'] for x in [output, *gradients])))
assert len(cases) == 11
rows = [json.loads(line) for line in (FOLDER / 'events.jsonl').read_text().splitlines()]
losses = [row for row in rows if row['event'] == 'loss']
updates = [row for row in rows if row['event'] == 'update']
monitor = json.loads((FOLDER / 'launch-monitor.json').read_text())
assert monitor['returncode'] == -15 and len(updates) == 1 and len(losses) == 2
assert len(list(FOLDER.glob('gradients-step-*.pt'))) == 1
with zipfile.ZipFile(FOLDER / 'gradients-step-000.pt') as archive:
    prefix = next(n[:-len('data.pkl')] for n in archive.namelist() if n.endswith('/data.pkl'))
    gradients = Reader(io.BytesIO(archive.read(prefix + 'data.pkl'))).load()
    assert len(gradients) == 744
report = dict(evidence_verified=True, passed=all(x['passed'] for x in cases),
    acceptance='bitwise output and every raw operator gradient; no tolerance or overfit substitution',
    cases=cases, passing_cases=sum(x['passed'] for x in cases), total_cases=len(cases),
    interrupted_attempt='20261009204949-5b997b5f', execution_commit='d8c0028',
    completed_optimizer_updates_before_user_stop=len(updates),
    raw_gradient_archives=1, raw_tensors=744,
    gradient_archive_sha256=sha(FOLDER / 'gradients-step-000.pt'),
    partial_losses=[row['loss'] for row in losses],
    baseline_v8_initial_loss=0.6250749230384827,
    full_model_initial_forward_difference_cause='not isolated; backward rounding cannot explain forward drift',
    partial_attempt_accepted=False, learning_test_complete=False,
    further_optimizer_updates_authorized=False, pushed_to_remote=False)
(ROOT / 'reviews/opt6-exact-v9.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
