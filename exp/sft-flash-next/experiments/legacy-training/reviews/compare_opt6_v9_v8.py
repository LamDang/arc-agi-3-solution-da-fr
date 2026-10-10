"""Compare existing pre-update v8/Opt6 loss and raw gradients on CPU.

Run from repository root with NumPy. No GPU job, optimizer, or network access.
Zero reference tensors are counted separately, never assigned a relative error.
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
BASE = ROOT / 'gradient-results/bf16-overfit-v8'
CANDIDATE = ROOT / 'gradient-results/liger-opt6-overfit-v9-stopped'


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch' and name == 'BFloat16Storage':
            return '<u2'
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda storage, offset, size, stride, *rest: (storage, offset, size, stride)
        raise ValueError((module, name))

    def persistent_load(self, item):
        kind, dtype, key, device, numel = item
        assert kind == 'storage' and device == 'cpu'
        return key, numel, dtype


class Tensors:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.prefix = next(n[:-len('data.pkl')] for n in self.zip.namelist() if n.endswith('/data.pkl'))
        assert self.zip.read(self.prefix + 'byteorder') == b'little'
        self.state = Reader(io.BytesIO(self.zip.read(self.prefix + 'data.pkl'))).load()

    def bits(self, name):
        (key, stored, dtype), offset, size, stride = self.state[name]
        assert dtype == '<u2'
        raw = self.zip.read(self.prefix + 'data/' + key)
        assert len(raw) == stored * 2
        return np.ndarray(size, dtype=dtype, buffer=raw, offset=offset * 2,
                          strides=tuple(x * 2 for x in stride))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def first_loss(folder):
    return next(json.loads(line)['loss'] for line in (folder / 'events.jsonl').read_text().splitlines()
                if json.loads(line)['event'] == 'loss')


configs = [json.loads((folder / 'resolved-config.json').read_text()) for folder in [BASE, CANDIDATE]]
for key in ('sample', 'model_config', 'reference_script'):
    assert configs[0]['expected_sha256'][key] == configs[1]['expected_sha256'][key]
initial = [Tensors(folder / 'initial-adapter.pt') for folder in [BASE, CANDIDATE]]
assert set(initial[0].state) == set(initial[1].state) and len(initial[0].state) == 744
for name in initial[0].state:
    a, b = [archive.bits(name) for archive in initial]
    assert a.shape == b.shape and a.tobytes() == b.tobytes(), name
archives = [Tensors(folder / 'gradients-step-000.pt') for folder in [BASE, CANDIDATE]]
assert set(archives[0].state) == set(archives[1].state) and len(archives[0].state) == 744
assert {name.replace('.default.weight', '.weight') for name in archives[0].state} == set(initial[0].state)
rows = []
ref_ss = cand_ss = diff_ss = dot = 0.0
for name in sorted(archives[0].state):
    a, b = [archive.bits(name) for archive in archives]
    assert a.shape == b.shape
    x, y = [(bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64) for bits in [a, b]]
    assert np.isfinite(x).all() and np.isfinite(y).all(), name
    xx, yy, dd, xy = [float(np.sum(t)) for t in [x*x, y*y, (x-y)**2, x*y]]
    ref_ss += xx; cand_ss += yy; diff_ss += dd; dot += xy
    rows.append(dict(name=name, elements=a.size, bitwise_equal=a.tobytes() == b.tobytes(),
        reference_norm=xx**.5, candidate_norm=yy**.5, difference_norm=dd**.5,
        relative_l2=(dd / xx)**.5 if xx else None,
        cosine=xy / (xx * yy)**.5 if xx and yy else None,
        max_absolute_difference=float(np.max(np.abs(x-y))),
        reference_zero=xx == 0, candidate_zero=yy == 0))
nonzero = [row for row in rows if not row['reference_zero']]
loss_a, loss_b = first_loss(BASE), first_loss(CANDIDATE)
report = dict(comparison='Opt6 v9 vs accepted BF16 v8, first loss and pre-clipping step-000 gradients',
    uses_existing_artifacts=True, new_forward_backward_runs=0, new_optimizer_updates=0,
    initialization_all_744_tensors_bitwise_equal=True, sample_model_reference_source_pins_equal=True,
    baseline_loss=loss_a, candidate_loss=loss_b, loss_difference=loss_b-loss_a,
    loss_relative_difference_percent=100*(loss_b-loss_a)/loss_a,
    raw_gradient_tensors=len(rows), all_gradients_finite=True,
    global_gradient_relative_l2= (diff_ss / ref_ss)**.5,
    global_gradient_relative_l2_percent=100*(diff_ss / ref_ss)**.5,
    global_gradient_cosine=dot/(ref_ss*cand_ss)**.5,
    gradient_norm_ratio=(cand_ss/ref_ss)**.5,
    exact_tensors=sum(row['bitwise_equal'] for row in rows),
    zero_reference_tensors=sum(row['reference_zero'] for row in rows),
    zero_reference_and_candidate_tensors=sum(row['reference_zero'] and row['candidate_zero'] for row in rows),
    nonzero_reference_tensors=len(nonzero),
    nonzero_tensor_relative_l2_percentiles_percent={str(p):float(100*np.percentile([r['relative_l2'] for r in nonzero],p)) for p in [50,90,95,99,100]},
    largest_relative_errors=sorted(nonzero,key=lambda r:r['relative_l2'],reverse=True)[:10],
    largest_absolute_contributions=sorted(rows,key=lambda r:r['difference_norm'],reverse=True)[:10],
    reference_gradient_archive_sha256=sha(BASE/'gradients-step-000.pt'),
    candidate_gradient_archive_sha256=sha(CANDIDATE/'gradients-step-000.pt'),
    scope='Fresh random-A/zero-B first step: all A gradients zero. Does not qualify trained nonzero-B gradients.',
    numerical_tolerance='User permits small variation; no numerical threshold assigned by this comparison.',
    pushed_to_remote=False, tensors=rows)
(ROOT/'reviews/opt6-v9-vs-v8.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['tensors','largest_relative_errors','largest_absolute_contributions']},indent=2))
