"""Independent checks of the audit reference and comparison machinery."""
import math

import pytest
import torch
from torch.nn import functional as F

import backend as b
import gradient_audit as audit
import tiny


def test_reference_attention_matches_single_autograd_graph():
    torch.manual_seed(131)
    q = torch.randn(4, 31, 8, requires_grad=True)
    k = torch.randn(2, 31, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    selected = torch.arange(31).repeat(31, 1)
    selected[selected > torch.arange(31)[:, None]] = -1
    expected = b.attention_block(q, k, v, selected, .3)
    actual = audit.ReferenceAttention.apply(q, k, v, selected, .3, 7)
    dense = audit.DenseReferenceAttention.apply(q, k, v, selected, .3, 7)
    upstream = torch.randn_like(expected)
    reference_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    dense_grads = torch.autograd.grad(dense, (q, k, v), upstream)
    torch.testing.assert_close(dense, expected)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(actual_grads + dense_grads, reference_grads + reference_grads):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)


def test_metrics_measure_error_and_ignore_padded_route_choices():
    ref = {'x': torch.tensor([1., 2.])}
    identical = audit.gradient_metrics(ref, ref)
    assert identical['bitwise_equal'] and identical['relative_l2'] == 0
    changed = audit.gradient_metrics({'x': torch.tensor([1., 3.])}, ref)
    assert changed['relative_l2'] == pytest.approx(1 / math.sqrt(5))
    assert changed['per_parameter'][0]['max_absolute'] == 1
    assert changed['different_fraction'] == .5
    routes = audit.compare_routes({0: torch.tensor([[-1, 1, 3]])},
                                  {0: torch.tensor([[-1, 1, 2]])})
    assert routes['replaced_choice_fraction'] == .5


def test_hybrid_reference_and_recipe_reset(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    native = tiny.reference_model()
    tiny.write_checkpoint(native, tmp_path)
    model, _ = b.rm.load_model(tmp_path, device='cpu', dtype=torch.float32,
                               record=False, log=lambda *_: None)
    b.configure(model, rank=2, alpha=4, norm_block=0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_(0, .03)
    # apply_recipe mutates a global dispatcher; restore it after this test.
    monkeypatch.setattr(b, 'SparseAttention', b.SparseAttention)
    ids = torch.randint(0, 1000, (1, 25))
    enc = dict(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids))
    ids, embedded, positions = b.embeddings(model, enc, 'cpu')

    def gradients(flags):
        audit.apply_recipe(model, flags)
        model.zero_grad(set_to_none=True)
        h = b.language_hidden(model, ids, embedded, positions)
        loss = F.cross_entropy(F.linear(h[0, 19:-1], model.lm_head.weight), ids[0, 20:])
        loss.backward()
        return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    loss, reference = gradients(dict(layer_checkpoint=False))
    # Smaller boundaries exercise the same control flow on the tiny fixture.
    got_loss, actual = gradients(dict(expert_block=3, norm_block=3, hyper_block=3,
        gated_norm_block=7, ple_block=3, attention='reference_projected',
        attention_projection_block=8, index_block=3, query_block=3, checkpoint_group=2))
    torch.testing.assert_close(got_loss, loss, rtol=2e-5, atol=2e-5)
    assert audit.gradient_metrics(actual, reference)['relative_l2'] < 1e-4
    restored_loss, restored = gradients(dict(layer_checkpoint=False))
    torch.testing.assert_close(restored_loss, loss, rtol=0, atol=0)
    assert audit.gradient_metrics(restored, reference)['bitwise_equal']
