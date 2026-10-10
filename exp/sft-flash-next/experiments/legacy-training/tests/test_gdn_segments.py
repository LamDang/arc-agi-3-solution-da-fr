import gdn_segments


def test_state_carry_gradient_matches_full_sequence():
    result=gdn_segments.check('cpu')
    assert max(result['relative_errors']) < 1e-4
    assert result['initial_state_gradient_norm'] > 0


import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FLA runtime required')
def test_state_carry_cuda_at_real_head_dimensions(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', False)
    report = gdn_segments.check('cuda')
    assert max(report['relative_errors']) < .03
    assert report['initial_state_gradient_norm'] > 0
