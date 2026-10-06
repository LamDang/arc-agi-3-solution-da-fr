"""The instrumented model computes what the reference transformers model does.

Run on CPU in float32 with a tiny random model: see tests/tiny.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as mq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reap_model  # noqa: E402
import replay  # noqa: E402
import tiny  # noqa: E402

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    ref = tiny.reference_model()
    ckpt = tmp_path_factory.mktemp("ckpt")
    tiny.write_checkpoint(ref, ckpt)
    ours, recorder = reap_model.load_model(ckpt, device="cpu", dtype=torch.float32, log=lambda *_: None)
    return ref, ours, recorder


def test_loader_matches_reference(models):
    ref, ours, _ = models
    ours_sd = dict(ours.named_parameters())
    ours_sd.update(dict(ours.named_buffers()))
    for name, tensor in ref.state_dict().items():
        if "mlp.experts." in name or "ngram_embedding" in name:
            continue
        assert torch.equal(ours_sd[name], tensor), name
    for name, buffer in ref.named_buffers():
        if "inv_freq" in name:
            assert torch.equal(ours_sd[name], buffer), name


def test_experts_and_reap_statistics(models):
    ref, ours, recorder = models
    torch.manual_seed(1)
    text = ref.config.text_config
    T, k, E = 37, text.num_experts_per_tok, text.num_experts
    hidden = torch.randn(T, text.hidden_size)
    logits = torch.randn(T, E)
    weights, index = torch.softmax(logits, -1).topk(k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    layer = 2
    ref_experts = ref.model.language_model.layers[layer].mlp.experts
    our_experts = ours.model.language_model.layers[layer].mlp.experts
    expected = ref_experts(hidden, index, weights)
    recorder.reset()
    cats = torch.randint(0, reap_model.N_CATEGORIES, (T,))
    recorder.categories = cats
    positions = torch.tensor([0, 40000, 70000, 100000, 32767, 32768])[torch.arange(T) % 6]
    bands = torch.tensor([0, 1, 2, 3, 0, 1])[torch.arange(T) % 6]
    recorder.positions = positions
    got = our_experts(hidden, index, weights)
    recorder.categories = recorder.positions = None
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    # brute-force REAP sums from the reference weights
    count = torch.zeros(E, reap_model.N_CATEGORIES, dtype=torch.float64)
    gate_norm = torch.zeros_like(count)
    norm = torch.zeros_like(count)
    for t in range(T):
        for slot in range(k):
            e = int(index[t, slot])
            gate, up = (hidden[t] @ ref_experts.gate_up_proj[e].T).chunk(2)
            y = (ref_experts.act_fn(gate) * up) @ ref_experts.down_proj[e].T
            count[e, cats[t]] += 1
            norm[e, cats[t]] += float(y.norm())
            gate_norm[e, cats[t]] += float(weights[t, slot]) * float(y.norm())
    data = recorder.data
    assert torch.equal(data["count"][layer], count)
    torch.testing.assert_close(data["norm"][layer], norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(data["gate_norm"][layer], gate_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(data["gate"][layer].sum(), torch.tensor(float(T), dtype=torch.float64))
    # position bands split the same sums
    for name in ("count", "gate", "gate_norm"):
        torch.testing.assert_close(data[f"{name}_pos"][layer].sum(-1), data[name][layer])
    count_pos = torch.zeros(E, reap_model.N_CATEGORIES, 4, dtype=torch.float64)
    for t in range(T):
        for slot in range(k):
            count_pos[int(index[t, slot]), cats[t], bands[t]] += 1
    assert torch.equal(data["count_pos"][layer], count_pos)


def _reference_selection_mask(attention, hidden, position_embeddings, kv_len):
    """Boolean [L, kv] mask from the reference QSA indexer (causal, no cache)."""
    L = hidden.shape[1]
    causal = torch.ones(L, kv_len, dtype=torch.bool).tril(diagonal=kv_len - L)[None, None]
    return attention.indexer(hidden, position_embeddings, causal, None)[0, 0] & causal[0, 0]


def test_token_selection_matches_reference_indexer(models):
    ref, _, _ = models
    torch.manual_seed(2)
    lm = ref.model.language_model
    attention = lm.layers[3].self_attn
    L = 61  # 15 complete blocks of 4, budget 16 tokens = 4 blocks: selection is sparse
    hidden = torch.randn(1, L, ref.config.text_config.hidden_size)
    positions = torch.arange(L).view(1, 1, -1).expand(3, 1, -1)
    position_embeddings = lm.rotary_emb(hidden, positions)
    expected = _reference_selection_mask(attention, hidden, position_embeddings, L)
    selected = reap_model.select_tokens(attention.indexer, hidden, *position_embeddings, None, query_block=16)
    got = torch.zeros(L, L + 1, dtype=torch.bool)
    got.scatter_(1, torch.where(selected >= 0, selected, L), True)
    got = got[:, :L]
    assert expected.sum(-1).max() < L  # the test does exercise sparsity

    # topk breaks ties in an unspecified order (relu makes exact zero ties
    # common here), so tied rows only need an equally scored selection
    scores = _block_scores(attention.indexer, hidden, position_embeddings)
    c = attention.indexer.compress_ratio
    k_top = attention.indexer.block_topk
    exact_rows = 0
    for p in range(L):
        visible = (p + 1) // c
        row = scores[p, :visible]
        exp_blocks = expected[p, : visible * c].view(visible, c).all(-1)
        got_blocks = got[p, : visible * c].view(visible, c).all(-1)
        assert torch.equal(expected[p, visible * c :], got[p, visible * c :]), p  # incomplete last block
        assert int(exp_blocks.sum()) == int(got_blocks.sum()) == min(k_top, visible), p
        torch.testing.assert_close(row[exp_blocks].sum(), row[got_blocks].sum())
        ranked = row.sort(descending=True).values
        if visible <= k_top or ranked[k_top - 1] - ranked[k_top] > 1e-4:
            assert torch.equal(exp_blocks, got_blocks), p
            exact_rows += 1
    assert exact_rows > L // 2


def _block_scores(indexer, hidden, position_embeddings):
    """Reference block scores [L, n_blocks], computed as Qwen4ExpTextQSAIndexer does."""
    cos, sin = position_embeddings
    L, d, c = hidden.shape[1], indexer.index_head_dim, indexer.compress_ratio
    q, k = torch.split(indexer.index_qk_proj(hidden), [indexer.index_n_heads * d, d], dim=-1)
    q = mq.apply_rotary_pos_emb(indexer.q_layernorm(q.reshape(1, L, -1, d)), cos=cos, sin=sin, unsqueeze_dim=2)
    nb = L // c
    pooled = indexer.k_layernorm(k[0, : nb * c].view(nb, c, d).float().mean(1).to(k.dtype))
    starts = torch.arange(nb) * c
    keys = mq.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0, starts], sin=sin[0, starts]).squeeze(1)
    return torch.stack([
        torch.relu(q[0, p].float() @ keys.float().T).sum(0) / d**0.5 for p in range(L)
    ])


def test_full_model_chunked_replay_matches_reference(models):
    ref, ours, recorder = models
    torch.manual_seed(3)
    config = ref.config
    vision = config.vision_config
    grid = torch.tensor([[1, 4, 4]])  # 16 patches -> 4 tokens after 2x2 merge
    n_patches = int(grid.prod())
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    pixel_values = torch.randn(n_patches, patch_dim)
    text_ids = torch.randint(0, 5000, (70,))
    image = torch.tensor([config.vision_start_token_id] + [config.image_token_id] * 4 + [config.vision_end_token_id])
    ids = torch.cat([text_ids[:20], image, text_ids[20:]])[None]
    mm_types = (ids == config.image_token_id).to(torch.int32)
    expected = ref(
        input_ids=ids, pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=mm_types,
        attention_mask=torch.ones_like(ids),
    ).logits
    enc = {"input_ids": ids, "pixel_values": pixel_values, "image_grid_thw": grid, "mm_token_type_ids": mm_types}
    cats = torch.zeros_like(ids)
    recorder.reset()
    for chunk in (ids.shape[1], 23):
        result = replay.replay(ours, recorder, enc, cats, chunk_tokens=chunk, measure=False,
                               collect_hidden=True, device="cpu")
        got = ours.lm_head(result["hidden"])
        torch.testing.assert_close(got, expected, rtol=2e-4, atol=2e-4, msg=f"chunk {chunk}")
    text = config.text_config
    counts = recorder.data["count"].sum(dim=(1, 2))
    assert torch.equal(counts, torch.full((text.num_hidden_layers,), 2.0 * ids.shape[1] * text.num_experts_per_tok,
                                          dtype=torch.float64))
    probs = recorder.data["prob"].sum(dim=(1, 2))
    torch.testing.assert_close(probs, torch.full_like(probs, 2.0 * ids.shape[1]))


def test_truncate_keeps_images_whole():
    ids = torch.tensor([[1, 2, 9, 5, 5, 5, 5, 8, 3, 9, 5, 5, 5, 5, 8, 4]])
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    enc = {"input_ids": ids, "image_grid_thw": grid, "pixel_values": torch.arange(32)[:, None],
           "mm_token_type_ids": ids == 5}
    cut = replay.truncate(enc, 12, vision_start_id=9)
    assert cut["input_ids"].shape[1] == 9
    assert cut["image_grid_thw"].shape[0] == 1 and cut["pixel_values"].shape[0] == 16
    assert replay.truncate(enc, 15, vision_start_id=9)["image_grid_thw"].shape[0] == 2


def test_grouped_matmul_path_matches_loop(models):
    """The GPU path (torch._grouped_mm over all experts) equals the per-expert loop."""
    _, ours, _ = models
    experts = ours.model.language_model.layers[1].mlp.experts
    torch.manual_seed(4)
    text = ours.config.text_config
    hidden = torch.randn(29, text.hidden_size, dtype=torch.bfloat16)
    index = torch.randint(0, text.num_experts - 3, (29, text.num_experts_per_tok))  # some experts get no tokens
    flat = index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    x = hidden[order // text.num_experts_per_tok]
    counts = torch.bincount(flat, minlength=text.num_experts)
    loop = experts._expert_outputs_loop(x, counts, torch.bfloat16)
    grouped = experts._expert_outputs_grouped(x, counts, torch.bfloat16)
    torch.testing.assert_close(grouped.float(), loop.float(), rtol=2e-2, atol=2e-2)


def test_pruned_routing_behaves_like_deleted_experts(models):
    _, ours, recorder = models
    torch.manual_seed(5)
    text = ours.config.text_config
    block = ours.model.language_model.layers[0].mlp
    hidden = torch.randn(1, 41, text.hidden_size)
    full = block(hidden)
    keep_all = torch.ones(text.num_hidden_layers, text.num_experts, dtype=torch.bool)
    reap_model.set_pruning(ours, keep_all)
    torch.testing.assert_close(block(hidden), full)

    keep = keep_all.clone()
    keep[:, ::3] = False
    reap_model.set_pruning(ours, keep)
    recorder.reset()
    pruned = block(hidden)
    reap_model.set_pruning(ours, None)
    counts = recorder.data["count"][0].sum(-1)
    assert counts[~keep[0]].sum() == 0 and counts[keep[0]].sum() == 41 * text.num_experts_per_tok
    # same as a router whose pruned rows were deleted
    flat = hidden.view(-1, text.hidden_size)
    logits = flat @ block.gate.weight[keep[0]].T
    top, idx = torch.softmax(logits.float(), -1).topk(text.num_experts_per_tok, -1)
    top = top / top.sum(-1, keepdim=True)
    expert_ids = keep[0].nonzero().flatten()[idx]
    expected = block.experts(flat, expert_ids, top) + torch.sigmoid(block.shared_expert_gate(flat)) * block.shared_expert(flat)
    torch.testing.assert_close(pruned.view(-1, text.hidden_size), expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(pruned, full)


def test_sdpa_attention_matches_einsum_attention(models):
    """Same model output with either attention kernel (the einsum path is the one
    checked against the reference above; this input has topk ties at score 0,
    so the reference itself may pick other, equally scored blocks)."""
    _, ours, _ = models
    torch.manual_seed(6)
    ids = torch.randint(0, 5000, (1, 75))
    enc = {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)}
    outputs = {}
    try:
        for attention in ("einsum", "sdpa"):
            reap_model.OPTIONS["attention"] = attention
            outputs[attention] = replay.replay(ours, None, enc, torch.zeros_like(ids), chunk_tokens=31,
                                               measure=False, collect_hidden=True, device="cpu")["hidden"]
    finally:
        reap_model.OPTIONS["attention"] = "einsum"
    torch.testing.assert_close(outputs["sdpa"], outputs["einsum"], rtol=1e-5, atol=1e-5)
