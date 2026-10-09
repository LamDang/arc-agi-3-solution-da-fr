"""Strict, per-tensor comparison; never silently ignore adapters or zero gradients."""
import hashlib
import math
import json
from pathlib import Path
import torch

OPERATOR_SOURCES = ('native_optimization_flags.py', 'native_checkpoint_blocks.py',
                    'native_mask_storage.py', 'native_head_loss.py',
                    'backend.py', 'offload.py', 'gdn_blocks.py')


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def assert_adapter_unchanged(current,state):
    if current.keys()!=state.keys():
        raise ValueError('Adapter state keys changed during comparison')
    for name,tensor in current.items():
        torch.testing.assert_close(tensor.cpu(),state[name],rtol=0,atol=0)


def compare(actual, reference, *, rtol=1e-5, atol=1e-8):
    if actual.keys() != reference.keys():
        raise ValueError('Adapter gradient keys differ')
    rows = {}
    total_error = total_ref = 0.0
    for name, ref in reference.items():
        got = actual[name]
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise ValueError('Gradient shape/dtype mismatch: ' + name)
        if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
            raise ValueError('Nonfinite gradient: ' + name)
        a, b = got.double(), ref.double()
        delta = a - b
        err, norm = delta.norm().item(), b.norm().item()
        denom = a.norm().item() * norm
        equal = torch.equal(got, ref)
        close = bool(torch.allclose(got, ref, rtol=rtol, atol=atol))
        # A zero reference gradient must stay exactly zero, regardless of atol.
        if norm == 0:
            close = equal
        rows[name] = dict(bitwise_equal=equal, within_tolerance=close,
            reference_norm=norm, actual_norm=a.norm().item(),
            relative_l2=err/norm if norm else (0.0 if err == 0 else None),
            max_absolute=delta.abs().max().item(),
            cosine=(a*b).sum().item()/denom if denom else None,
            different_elements=int(torch.count_nonzero(got != ref)))
        total_error += err*err
        total_ref += norm*norm
    return dict(bitwise_equal=all(x['bitwise_equal'] for x in rows.values()),
        within_tolerance=all(x['within_tolerance'] for x in rows.values()),
        rtol=rtol, atol=atol, relative_l2=math.sqrt(total_error/max(total_ref,1e-300)),
        failed_tensors=sum(not x['within_tolerance'] for x in rows.values()),
        total_tensors=len(rows), per_parameter=rows)


def qualified_flags(directory,adapter):
    def sha256(path):
        with open(path,"rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
    q=Path(directory)
    flags=json.loads((q/'accepted-flags.json').read_text())
    results=json.loads((q/'results.json').read_text())
    identity=json.loads((q/'identity.json').read_text())
    assert identity['arguments']['deterministic'], 'Use the stable deterministic baseline'
    assert identity['adapter_sha256']==sha256(adapter), 'Different adapter initialization'
    assert any(r['name']=='reference_repeat' and r['passed'] and r['bitwise_equal'] for r in results)
    assert flags, 'No optimization recipe was accepted'
    assert any(r['passed'] and {k:v for k,v in r['flags'].items() if k!='name'}==flags for r in results)
    for name in OPERATOR_SOURCES:
        assert identity['source_sha256'][name]==sha256(Path(__file__).parent/name), 'Changed operator source: '+name
    return flags,identity
