import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location('native_gradient_compare',Path(__file__).parents[1]/'native_gradient_compare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_zero_reference_never_passes_nonzero():
    result=m.compare({'a':torch.tensor([1e-12])},{'a':torch.zeros(1)})
    assert not result['within_tolerance']
    assert result['per_parameter']['a']['relative_l2'] is None


def test_missing_extra_and_nonfinite_fail():
    with pytest.raises(ValueError):m.compare({'a':torch.ones(1),'b':torch.ones(1)},{'a':torch.ones(1)})
    with pytest.raises(ValueError):m.compare({'a':torch.tensor([float('nan')])},{'a':torch.ones(1)})


def test_per_adapter_gate_cannot_be_hidden_by_global_norm():
    ref={'large':torch.ones(100),'small':torch.tensor([1e-4])}
    got={**ref,'small':torch.tensor([2e-4])}
    result=m.compare(got,ref)
    assert result['relative_l2'] < 1e-4
    assert result['failed_tensors']==1
    assert not result['within_tolerance']


def test_exact_clone_and_digest():
    ref={'a':torch.randn(5,dtype=torch.bfloat16)}
    result=m.compare({n:v.clone() for n,v in ref.items()},ref)
    assert result['bitwise_equal'] and result['within_tolerance']
    assert m.tensor_digest(ref['a'])==m.tensor_digest(ref['a'].clone())
