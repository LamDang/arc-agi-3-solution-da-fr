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


def test_capacity_requires_a_passed_recipe_and_unchanged_sources(tmp_path):
    import json,hashlib
    from pathlib import Path
    adapter=tmp_path/'adapter.pt';adapter.write_bytes(b'fixture')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=Path(m.__file__).parent
    names=['native_optimization_flags.py','native_checkpoint_blocks.py','native_mask_storage.py',
           'backend.py','offload.py','gdn_blocks.py']
    identity=dict(arguments=dict(deterministic=True),adapter_sha256=digest(adapter),
                  source_sha256={n:digest(source/n) for n in names})
    flags=dict(loss='selected')
    results=[dict(name='reference_repeat',passed=True,bitwise_equal=True,flags={'name':'reference_repeat'}),
             dict(name='selected_logits',passed=True,flags={'name':'selected_logits',**flags})]
    for name,value in [('accepted-flags.json',flags),('identity.json',identity),('results.json',results)]:
        (tmp_path/name).write_text(json.dumps(value))
    assert m.qualified_flags(tmp_path,adapter)[0]==flags
    results[1]['passed']=False;(tmp_path/'results.json').write_text(json.dumps(results))
    with pytest.raises(AssertionError):m.qualified_flags(tmp_path,adapter)
    results[1]['passed']=True;(tmp_path/'results.json').write_text(json.dumps(results))
    identity['source_sha256']['backend.py']='changed';(tmp_path/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(AssertionError,match='Changed operator source'):m.qualified_flags(tmp_path,adapter)


def test_adapter_state_check_rejects_updates_and_keeps_loss_scalar():
    loss=0.6247151494026184
    state={'a':torch.tensor([1.,2.])}
    m.assert_adapter_unchanged({'a':state['a'].clone()},state)
    assert isinstance(loss,float)
    with pytest.raises(AssertionError):m.assert_adapter_unchanged({'a':torch.tensor([1.,3.])},state)
    with pytest.raises(ValueError):m.assert_adapter_unchanged({},state)
