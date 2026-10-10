import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from native_head_loss import selected_native_backward_loss


@pytest.mark.parametrize('scale', [1., .125, -2.])
@pytest.mark.parametrize('rows', [None, 32])
def test_native_shape_backward_matches_frozen_head_with_loss_scaling(scale, rows):
    torch.manual_seed(31)
    weight = torch.randn(113, 17)
    original = torch.randn(1, 29, 17)
    labels = torch.randint(0, 113, (1, 29))
    labels[:, :23] = -100
    # An ignored response token also follows native CE normalization.
    labels[:, 25] = -100
    native = original.clone().requires_grad_()
    candidate = original.clone().requires_grad_()
    expected = ForCausalLMLoss(torch.nn.functional.linear(native, weight),
                              labels, vocab_size=113)
    actual = selected_native_backward_loss(candidate, weight, labels, 23, rows)
    (expected * scale).backward()
    (actual * scale).backward()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(candidate.grad, native.grad, rtol=1e-6, atol=1e-7)
    assert not torch.count_nonzero(candidate.grad[:, :22])
    assert not torch.count_nonzero(candidate.grad[:, -1])


def test_rejects_trainable_head_or_supervised_omitted_labels():
    hidden = torch.randn(1, 7, 3, requires_grad=True)
    weight = torch.randn(11, 3)
    labels = torch.zeros(1, 7, dtype=torch.long)
    with pytest.raises(ValueError, match='prompt labels'):
        selected_native_backward_loss(hidden, weight, labels, 4)
    labels[:, :4] = -100
    with pytest.raises(ValueError, match='head must be frozen'):
        selected_native_backward_loss(hidden, weight.requires_grad_(), labels, 4)


@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.bfloat16])
def test_autocast_fp32_hidden_returns_native_fp32_gradient(weight_dtype):
    torch.manual_seed(91)
    weight = torch.randn(113, 17).to(weight_dtype)
    # FP32 values have information beyond BF16, exercising the native cast.
    original = torch.randn(1, 29, 17)
    labels = torch.randint(0, 113, (1, 29))
    labels[:, :23] = -100
    native = original.clone().requires_grad_()
    candidate = original.clone().requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = ForCausalLMLoss(torch.nn.functional.linear(native, weight),
                                  labels, vocab_size=113)
        actual = selected_native_backward_loss(candidate, weight, labels, 23)
    expected.backward()
    actual.backward()
    assert candidate.grad.dtype == native.grad.dtype == torch.float32
    # CPU CE changes the scalar summation order when ignored rows are omitted;
    # the cast-back and per-element hidden gradient must still be exact.
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=0)
    torch.testing.assert_close(candidate.grad, native.grad, rtol=0, atol=0)
