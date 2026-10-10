"""Interleaved supervision, native shifting, autocast and full-context gradients."""
from types import SimpleNamespace
import copy

import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from target_only_head import target_positions, target_logits, patch_model
from trajectory_head_loss import trajectory_loss


@pytest.mark.parametrize('targets', [[1, 2, 8, 9, 17, 28], [3, 15, 28], [1]])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_interleaved_masks_match_native_loss_and_gradients(targets, reduction):
    torch.manual_seed(921)
    original = torch.randn(1, 29, 17)
    weight = torch.randn(113, 17).to(torch.bfloat16)
    labels = torch.full((1, 29), -100, dtype=torch.long)
    labels[0, targets] = torch.randint(0, 113, (len(targets),))
    native, candidate = (original.clone().requires_grad_() for _ in range(2))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = ForCausalLMLoss(torch.nn.functional.linear(native, weight), labels,
                                 vocab_size=113,
                                 num_items_in_batch=1 if reduction == 'sum' else None)
        actual = trajectory_loss(candidate, weight, labels, reduction=reduction)
    (expected * .125).backward()
    (actual * .125).backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.grad, native.grad, rtol=0, atol=0)
    positions, shifted = target_positions(labels)
    assert positions.tolist() == [t - 1 for t in targets]
    assert shifted.tolist() == labels[:, targets].tolist()
    ignored = torch.ones(29, dtype=torch.bool)
    ignored[positions] = False
    assert not torch.count_nonzero(candidate.grad[:, ignored])
    assert candidate.grad.dtype == torch.float32


def test_rejects_unpredictable_first_token_empty_mask_and_trainable_head():
    labels = torch.full((1, 9), -100, dtype=torch.long)
    with pytest.raises(ValueError, match='No supervised'):
        target_positions(labels)
    labels[0, 0] = 1
    with pytest.raises(ValueError, match='preceding'):
        target_positions(labels)
    labels[0, 0], labels[0, 8] = -100, 1
    with pytest.raises(ValueError, match='frozen'):
        target_logits(torch.randn(1, 9, 3), torch.randn(7, 3, requires_grad=True), labels)


class NativeForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(17, 113, bias=False).to(torch.bfloat16)
        self.lm_head.requires_grad_(False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(output_router_logits=False))
        self.loss_function = ForCausalLMLoss
        self.decoder_rows = []

    def forward(self, *, inputs_embeds, labels=None, **kwargs):
        self.decoder_rows.append(inputs_embeds.shape[1])
        # Context mixing makes ignored input positions affect later targets.
        hidden = inputs_embeds.cumsum(1)
        logits = self.lm_head(hidden)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=113)
        return SimpleNamespace(loss=loss, logits=logits)


def test_native_forward_stays_full_context_and_cpu_logprobs_are_target_sized(tmp_path):
    torch.manual_seed(420)
    native = NativeForward()
    candidate = copy.deepcopy(native)
    labels = torch.full((1, 29), -100, dtype=torch.long)
    targets = [2, 3, 12, 13, 28]
    labels[0, targets] = torch.randint(0, 113, (len(targets),))
    original = torch.randn(1, 29, 17)
    a, b = (original.clone().requires_grad_() for _ in range(2))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = native(inputs_embeds=a, labels=labels)
    expected.loss.backward()
    report, restore = patch_model(candidate, tmp_path / 'target-head.json')
    try:
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = candidate(inputs_embeds=b, labels=labels)
            actual.loss.backward()
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        assert torch.equal(a.grad, b.grad)
        assert torch.count_nonzero(b.grad[:, 0])  # Context was not detached.
        assert candidate.decoder_rows == [29]
        assert actual.logits.shape == (1, len(targets), 113)
        assert report['head_calls'][0]['prediction_positions'] == [t - 1 for t in targets]
        saved = report['vocabulary_saved_tensor_calls']
        assert saved and all(r['shape'] == [len(targets), 113] for r in saved)
        assert all(r['saved_device'] == 'cpu' and r['dtype'] == 'torch.float32' for r in saved)
    finally:
        restore()
