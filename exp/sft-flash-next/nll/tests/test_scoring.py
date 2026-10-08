import pytest
import torch
import torch.nn.functional as F

import reap_model
import replay
import tiny


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    ref = tiny.reference_model(seed=41)
    ckpt = tmp_path_factory.mktemp("nll-tiny")
    tiny.write_checkpoint(ref, ckpt)
    ours, recorder = reap_model.load_model(ckpt, device="cpu", dtype=torch.float32, record=False, log=lambda *_: None)
    assert recorder is None
    return ref, ours


def test_target_nll_matches_reference_across_chunks(models):
    ref, model = models
    torch.manual_seed(19)
    ids = torch.randint(0, 1000, (1, 53))
    enc = dict(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids))
    # First target is predicted by the previous chunk's final hidden state.
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 29:] = True
    cats = torch.ones_like(ids)  # all historical turns GENERATED: explicit mask must win
    cats[:, 0] = 0
    with torch.no_grad():
        expected_logits = ref(**enc).logits[0, 28:-1].float()
        expected = F.cross_entropy(expected_logits, ids[0, 29:], reduction="none")
    for chunk in (29, 11):
        got = replay.replay(model, None, enc, cats, target_mask=mask, token_losses=True,
                            chunk_tokens=chunk, lm_block=3, device="cpu")
        assert "correct" not in got and "argmax" not in got
        assert got["scored"] == 24
        assert got["token_positions"].tolist() == list(range(29, 53))
        torch.testing.assert_close(got["token_nll"], expected, rtol=2e-4, atol=2e-4)


def test_future_target_cannot_change_earlier_loss(models):
    _, model = models
    ids = torch.arange(45)[None] + 2
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 30:] = True
    def score(x):
        return replay.replay(model, None, {"input_ids": x}, torch.zeros_like(x),
                             target_mask=mask, token_losses=True, chunk_tokens=17,
                             device="cpu", lm_block=4)["token_nll"]
    original = score(ids)
    changed = ids.clone()
    changed[0, 40:] += 103
    torch.testing.assert_close(score(changed)[:10], original[:10], rtol=1e-5, atol=1e-5)


def test_invalid_target_zero_rejected(models):
    _, model = models
    ids = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="position zero"):
        replay.replay(model, None, {"input_ids": ids}, torch.zeros_like(ids),
                      target_mask=torch.ones_like(ids, dtype=torch.bool), device="cpu")


def test_multimodal_final_reply_nll_matches_official_model(models):
    ref, ours = models
    torch.manual_seed(3)
    config = ref.config
    grid = torch.tensor([[1, 4, 4]])
    dim = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    pixels = torch.randn(16, dim)
    text = torch.randint(0, 1000, (45,))
    image = torch.tensor([config.vision_start_token_id] + [config.image_token_id]*4 + [config.vision_end_token_id])
    ids = torch.cat([text[:10], image, text[10:]])[None]
    enc = dict(input_ids=ids, pixel_values=pixels, image_grid_thw=grid,
               mm_token_type_ids=(ids==config.image_token_id).to(torch.int32))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 40:] = True
    with torch.no_grad():
        logits = ref(**enc, attention_mask=torch.ones_like(ids)).logits[0, 39:-1].float()
        expected = F.cross_entropy(logits, ids[0, 40:], reduction="none")
    result = replay.replay(ours, None, enc, torch.zeros_like(ids), target_mask=mask,
                           token_losses=True, chunk_tokens=13, lm_block=3, device="cpu")
    torch.testing.assert_close(result["token_nll"], expected, atol=2e-4, rtol=2e-4)
