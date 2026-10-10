from types import SimpleNamespace

import pytest
import torch

import evaluate


def test_validation_resume_and_identity(tmp_path, monkeypatch):
    calls = []
    rows = [dict(sample_id=f'game-r{i}', game='game', target_tokens=2) for i in range(2)]
    bundle = SimpleNamespace(manifest={'sha256': 'frozen-data'}, get=lambda row: (
        {'index': int(row['sample_id'][-1])}, dict(prompt_tokens=3, positions=[3, 4], labels=[0, 1])))
    model = torch.nn.Linear(1, 1)
    fail = [True]
    def score(model, enc, *args):
        calls.append(enc['index'])
        if enc['index'] == 1 and fail[0]:
            raise RuntimeError('simulated interruption')
        return torch.tensor([1., 3.])
    monkeypatch.setattr(evaluate.backend, 'token_losses', score)
    out = tmp_path/'evaluation'
    with pytest.raises(RuntimeError):
        evaluate.evaluate_model(model, bundle, rows, {'loss_block': 128}, out, {'checkpoint': 'base'})
    fail[0] = False
    result = evaluate.evaluate_model(model, bundle, rows, {'loss_block': 128}, out, {'checkpoint': 'base'})
    assert calls == [0, 1, 1]
    assert result['token_nll']['all'] == 2.
    assert result['token_counts']['all'] == 4
    assert evaluate.evaluate_model(model, bundle, rows, {'loss_block': 128}, out, {'checkpoint': 'base'}) == result
    assert calls == [0, 1, 1]
    with pytest.raises(ValueError, match='different identity'):
        evaluate.evaluate_model(model, bundle, rows, {'loss_block': 128}, out, {'checkpoint': 'trained'})
