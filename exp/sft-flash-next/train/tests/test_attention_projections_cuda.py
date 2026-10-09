import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA indexed-attention runtime required')
def test_attention_projection_blocks_preserve_global_context_gradients(monkeypatch):
    import attention_projection_checks
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', False)
    report = attention_projection_checks.check()
    assert report['input_gradient_relative_error'] < .01
    assert max(report['adapter_gradient_relative_errors'].values()) < .01
