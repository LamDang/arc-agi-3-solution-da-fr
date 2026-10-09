"""Strict, per-tensor comparison; never silently ignore adapters or zero gradients."""
import hashlib
import math
import torch


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


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
