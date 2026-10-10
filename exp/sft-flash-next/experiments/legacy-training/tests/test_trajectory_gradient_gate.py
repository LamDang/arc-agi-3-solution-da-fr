import importlib.metadata
import json
import pytest
import torch
from dataset import file_hash
import trajectory_gradient_gate as gate


def setup_gate(root,monkeypatch):
    monkeypatch.setattr(gate,'source_hashes',lambda:{'test-source':'exact'})
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'test-gpu')
    state = {f'layer{i}.lora_{kind}.weight':torch.ones(2)
             for i in range(372) for kind in ['A','B']}
    for name in ['adapter.pt','native-1.pt','native-2.pt','candidate.pt']:
        torch.save(state,root/name)
    results = [dict(name=name,loss=10.) for name in ['native-1','native-2','candidate']]
    (root/'results.json').write_text(json.dumps(results))
    identity = dict(objective='native-all-assistant-summed-ce-v1',source_sha256=gate.source_hashes(),
        flags={},head_backward_rows=16384,deterministic=True,
        runtime={'torch':importlib.metadata.version('torch')},cuda=torch.version.cuda,gpu='test-gpu',
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        artifacts={name:file_hash(root/name) for name in
                   ['adapter.pt','native-1.pt','native-2.pt','candidate.pt','results.json']})
    (root/'identity.json').write_text(json.dumps(identity))
    return identity,state


def test_raw_gate_rejects_changed_evidence(tmp_path,monkeypatch):
    identity,state = setup_gate(tmp_path,monkeypatch)
    assert gate.verify_gradient_gate(tmp_path,{},16384) == identity
    state[next(iter(state))][0] += 1e-7
    torch.save(state,tmp_path/'candidate.pt')
    with pytest.raises(ValueError,match='Changed.*evidence'):
        gate.verify_gradient_gate(tmp_path,{},16384)


def test_raw_gate_rejects_nonbitwise_but_allclose(tmp_path,monkeypatch):
    identity,state = setup_gate(tmp_path,monkeypatch)
    state[next(iter(state))][0] += 1e-7
    torch.save(state,tmp_path/'candidate.pt')
    identity['artifacts']['candidate.pt'] = file_hash(tmp_path/'candidate.pt')
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='bitwise'):
        gate.verify_gradient_gate(tmp_path,{},16384)


def test_raw_gate_rejects_zero_b_and_legacy_objective(tmp_path,monkeypatch):
    identity,state = setup_gate(tmp_path,monkeypatch)
    for name,value in state.items():
        if 'lora_B' in name:
            value.zero_()
    torch.save(state,tmp_path/'adapter.pt')
    identity['artifacts']['adapter.pt'] = file_hash(tmp_path/'adapter.pt')
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='nonzero'):
        gate.verify_gradient_gate(tmp_path,{},16384)
    identity['objective'] = 'legacy-final-reply'
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='final-reply'):
        gate.verify_gradient_gate(tmp_path,{},16384)
