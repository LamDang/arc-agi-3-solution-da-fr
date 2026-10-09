import copy

import pytest
import torch
import numpy as np
from torch.nn import functional as F

import backend as b
import tiny


def test_unique_ple_lookup_and_request_cache():
    shards = [np.arange(24, dtype=np.float32).reshape(6, 4),
              np.arange(24, 44, dtype=np.float32).reshape(5, 4)]
    module = b.rm.MmapEmbedding(shards)
    ids = torch.tensor([[10, 0, 6, 6, 2, 10, 0]])
    expected = module(ids)
    torch.testing.assert_close(b.unique_mmap_lookup(module, ids), expected, rtol=0, atol=0)
    with b.request_ple_cache(module):
        first = module(ids)
        assert module(ids.clone()).data_ptr() == first.data_ptr()
        torch.testing.assert_close(module(ids.flip(-1)), expected.flip(1))
    torch.testing.assert_close(module(ids), expected)


@pytest.mark.parametrize("group", [None, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bounded_rms_norm_matches_reference(group, dtype):
    torch.manual_seed(123)
    norm = b.rm.mq.Qwen4ExpTextRMSNorm(32, group_size=group).to(dtype)
    norm.requires_grad_(False)
    with torch.no_grad():
        norm.weight.normal_(0, .2)
    # Deliberately noncontiguous, non-block-aligned, and with nontrivial scales.
    x = torch.randn(2, 32, 11, dtype=dtype).transpose(1, 2).requires_grad_(True)
    grad = torch.randn_like(x)
    expected = norm(x)
    expected_grad, = torch.autograd.grad(expected, x, grad)
    actual = b.FrozenRMSNorm.apply(x, norm.weight, norm.eps, group, 3)
    actual_grad, = torch.autograd.grad(actual, x, grad)
    # Making a noncontiguous reduction contiguous can change FP32 reduction
    # order by a few ulps; BF16 rounds this example to the same output.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad,
                               rtol=1e-5 if dtype == torch.float32 else .008,
                               atol=1e-6 if dtype == torch.float32 else .004)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.set_num_threads(2)
    ref = tiny.reference_model()
    root = tmp_path_factory.mktemp("tiny")
    tiny.write_checkpoint(ref, root)
    model, _ = b.rm.load_model(root, device="cpu", dtype=torch.float32, record=False, log=lambda *_: None)
    return ref, model


def test_frozen_experts_input_and_router_gradients(models):
    ref, ours = models
    e = ours.model.language_model.layers[0].mlp.experts
    x = torch.randn(17, 64, requires_grad=True)
    scores = torch.randn(17, 16, requires_grad=True)
    weights, indices = scores.softmax(-1).topk(4)
    expected = ref.model.language_model.layers[0].mlp.experts(x, indices, weights)
    grad = torch.randn_like(expected)
    dx, ds = torch.autograd.grad(expected, (x, scores), grad, retain_graph=True)
    got = b.FrozenExperts.apply(x, indices, weights, e, 3)
    gx, gs = torch.autograd.grad(got, (x, scores), grad)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(gx, dx)
    torch.testing.assert_close(gs, ds)


def test_sparse_attention_backward_matches_native():
    torch.manual_seed(1)
    q = torch.randn(4, 13, 8, requires_grad=True)
    k = torch.randn(2, 13, 8, requires_grad=True)
    v = torch.randn(2, 13, 8, requires_grad=True)
    selected = torch.arange(13).repeat(13, 1)
    selected = torch.where(selected <= torch.arange(13)[:, None], selected, -1)
    expected = b.attention_block(q, k, v, selected, .3)
    grad = torch.randn_like(expected)
    grads = torch.autograd.grad(expected, (q, k, v), grad)
    got = b.SparseAttention.apply(q, k, v, selected, .3, 3)
    actual = torch.autograd.grad(got, (q, k, v), grad)
    torch.testing.assert_close(got, expected)
    for a, e in zip(actual, grads):
        torch.testing.assert_close(a, e)


def test_selected_ce_exact_loss_and_gradient():
    h = torch.randn(11, 7, requires_grad=True)
    weight = torch.randn(31, 7)
    target = torch.randint(0, 31, (11,))
    ref = F.cross_entropy(F.linear(h, weight), target, reduction="sum")
    expected, = torch.autograd.grad(ref, h)
    actual = b.SelectedCrossEntropy.apply(h, target, weight, 3)
    actual.backward()
    torch.testing.assert_close(actual, ref)
    torch.testing.assert_close(h.grad, expected)


@pytest.mark.parametrize("checkpoint_group", [1, 2, 3])
@pytest.mark.parametrize('ple_block', [0, 3])
def test_full_hybrid_multimodal_backward_checkpoint_parity(models, checkpoint_group, ple_block):
    _, original = models
    native, checked = copy.deepcopy(original), copy.deepcopy(original)
    torch.manual_seed(9)
    targets = b.configure(native, rank=2, alpha=4, gradient_checkpointing=False, norm_block=0, hyper_block=0)
    torch.manual_seed(9)
    b.configure(checked, rank=2, alpha=4, expert_block=3, query_block=3, gradient_checkpointing=True, hyper_block=3, checkpoint_group=checkpoint_group, ple_checkpoint=True, gdn_checkpoint=True, ple_block=ple_block, gated_norm_block=7, gdn_chunk_tokens=8, rms_block_mib=.001, gdn_block_tokens=8, attention_projection_block=8)
    assert any("linear_attn" in n for n in targets)
    assert all("indexer" not in n and "visual" not in n for n in targets)
    ids = torch.randint(0, 1000, (1, 25))
    ids[0, 5:11] = torch.tensor([original.config.vision_start_token_id] + [original.config.image_token_id]*4 + [original.config.vision_end_token_id])
    vision = original.config.vision_config
    enc = dict(input_ids=ids, image_grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values=torch.randn(16, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2),
        mm_token_type_ids=(ids == original.config.image_token_id).int())
    # Same model, ordinary dequantized experts for a genuinely independent autograd reference.
    for layer in native.model.language_model.layers:
        e = layer.mlp.experts
        def reference_forward(hidden, indices, weights, e=e):
            out = torch.zeros_like(hidden)
            for i in range(e.num_experts):
                rows, slots = torch.where(indices == i)
                w1, w2 = b.expert_weights(e, i, hidden.dtype)
                y = b.expert_value(hidden[rows], w1, w2, e.act_fn)
                out = out.index_add(0, rows, y * weights[rows, slots, None])
            return out
        e.forward = reference_forward
    loss = b.loss_sum(native, enc, 20, "cpu", 2) / 5
    loss.backward()
    with torch.autograd.graph.save_on_cpu():
        other = b.loss_sum(checked, enc, 20, "cpu", 3) / 5
        other.backward()
    torch.testing.assert_close(loss, other, rtol=2e-5, atol=2e-5)
    expected = dict(native.named_parameters())
    for name, p in checked.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, expected[name].grad, rtol=4e-4, atol=4e-5, msg=name)
        else:
            assert p.grad is None
    optimizer = torch.optim.AdamW([p for p in checked.parameters() if p.requires_grad], lr=1e-3)
    before = b.adapter_state(checked)
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in b.adapter_state(checked).items())


@pytest.mark.parametrize('combine', [True, False])
def test_hyper_mix_bounded_gradient(combine):
    config = b.rm.mq.Qwen4ExpTextConfig(hidden_size=32, hc_count=4, hc_lowrank=8)
    module = b.rm.mq.Qwen4ExpTextGatedResidual(config, use_combine=combine)
    module.requires_grad_(False)
    x = torch.randn(2, 11, 128, requires_grad=True)
    ref = module(x)
    module.unblocked_forward = module.forward
    module.train_block = 3
    got = b.training_hyper_mix(module, x)
    ref = ref if isinstance(ref, tuple) else (ref,)
    got = got if isinstance(got, tuple) else (got,)
    gradients = [torch.randn_like(t) for t in ref]
    for a, e in zip(got, ref):
        torch.testing.assert_close(a, e)
    expected, = torch.autograd.grad(ref, x, gradients)
    actual, = torch.autograd.grad(got, x, gradients)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_ple_row_cache_eviction_parallel_lookup():
    module = b.rm.MmapEmbedding([np.arange(48, dtype=np.float32).reshape(12, 4),
                                np.arange(48, 96, dtype=np.float32).reshape(12, 4)])
    module.row_cache = b.PLERowCache(5, 4, np.dtype('float32'))
    module.read_workers = 3
    reference = torch.arange(96, dtype=torch.float32).reshape(24, 4)
    for keys in ([1, 4, 15, 23, 1], [1, 4, 15], [2, 3, 6, 8, 9, 10, 11], [23, 1, 10, 11], []):
        ids = torch.tensor(keys, dtype=torch.long)
        torch.testing.assert_close(b.unique_mmap_lookup(module, ids), reference[ids], rtol=0, atol=0)
        assert len(module.row_cache.index) <= 5
    assert module.row_cache.hits >= 3


def test_real_model_gradient_check_restores_optimized_paths(models, monkeypatch):
    from kernel_checks import check_model_backward
    _, original = models
    model = copy.deepcopy(original)
    b.configure(model, rank=2, alpha=4, hyper_block=7, checkpoint_group=2, ple_checkpoint=True, gdn_checkpoint=True, ple_block=3, gated_norm_block=7, gdn_chunk_tokens=64, rms_block_mib=.001, gdn_block_tokens=8, attention_projection_block=8)
    before = b.adapter_state(model)
    forwards = [(m,m.forward) for m in model.modules() if 'forward' in m.__dict__]
    real_loss = b.loss_sum
    monkeypatch.setattr(b, 'loss_sum', lambda model, enc, prompt, device, block:
                        real_loss(model, enc, prompt, 'cpu', block))
    report = check_model_backward(model, loss_block=31, tokens=160, tolerance=1e-4)
    assert report['adapter_gradient_relative_error'] < 1e-4
    assert model.train_checkpoint_group == 2
    assert all(m.forward == f for m,f in forwards)
    assert all(p.grad is None for p in model.parameters())
    for name, value in b.adapter_state(model).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize('block', [1, 3, 8, 64])
def test_ple_chunk_halo_preserves_boundary_gradients(models, block):
    _, model = models
    module = copy.deepcopy(next(l.ple for l in model.model.language_model.layers if l.ple is not None))
    module.requires_grad_(False)
    torch.manual_seed(81)
    x = torch.randn(1, 25, module.hidden_size*module.hc_count, requires_grad=True)
    ids = torch.randint(0, 1000, (1, 25))
    expected = module(x, ids, None)
    upstream = torch.randn_like(expected)
    ref, = torch.autograd.grad(expected, x, upstream)
    got = b.FrozenPLE.apply(x, ids, module, block)
    actual, = torch.autograd.grad(got, x, upstream)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, ref, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('activation', ['silu','sigmoid'])
def test_gated_rms_preserves_casts_and_both_gradients(dtype, activation):
    module = b.rm.mq.Qwen4ExpTextRMSNormGated(32,activation=activation).to(dtype=dtype)
    module.requires_grad_(False)
    torch.manual_seed(17)
    with torch.no_grad(): module.weight.normal_(1., .2)
    x = torch.randn(2,32,19,dtype=dtype).transpose(-1,-2).requires_grad_(True)
    gate = torch.randn_like(x).requires_grad_(True)
    expected = module(x,gate)
    grad = torch.randn_like(expected)
    ref = torch.autograd.grad(expected,(x,gate),grad)
    got = b.FrozenGatedRMSNorm.apply(x,gate,module.weight,module.variance_epsilon,7,activation)
    actual = torch.autograd.grad(got,(x,gate),grad)
    torch.testing.assert_close(got,expected,rtol=0 if dtype==torch.bfloat16 else 1e-6,atol=0 if dtype==torch.bfloat16 else 1e-6)
    for a,e in zip(actual,ref):
        assert float((a.float()-e.float()).norm()/e.float().norm()) < (0.006 if dtype==torch.bfloat16 else 1e-6)


def test_resident_ple_preserves_bytes_and_stops_reading_source(tmp_path):
    path = tmp_path/'table.bin'
    source = np.memmap(path,dtype=np.int16,mode='w+',shape=(31,8))
    source[:] = np.arange(source.size).reshape(source.shape)
    source.flush()
    module = b.rm.MmapEmbedding([source[:13],source[13:]])
    expected = source.copy()
    assert b.materialize_ple(module) == expected.nbytes
    source[:] = 0
    source.flush()
    assert all(not isinstance(s,np.memmap) and not s.flags.writeable for s in module.shards)
    np.testing.assert_array_equal(np.concatenate(module.shards), expected)
    assert b.materialize_ple(module) == expected.nbytes
