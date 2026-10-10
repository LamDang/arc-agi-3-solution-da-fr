import copy
import json
from types import SimpleNamespace

import pytest
import torch

import backend
import checkpoints
from dataset import fold_games, inject_thinking, validate_sample
from run import epoch_order, finish_step, learning_rate


def sample():
    return dict(game="train-123", request_index=2, thinking_source="think_gen-refine2",
        chat_template_kwargs={"preserve_thinking": True},
        tools=[{"function": {"name": "python", "parameters": {"properties": {"code": {"type": "string"}}, "required": ["code"]}}}],
        messages=[{"role": "user", "content": "board"},
                  {"role": "assistant", "reasoning_content": "Inspect the board.", "content": "",
                   "tool_calls": [{"function": {"name": "python", "arguments": {"code": "print(1)"}}}]}])


def test_thinking_join_preserves_context_code_and_order():
    before = sample()
    result = inject_thinking(before, {"train-123_p0#2": {"status": "ok", "thinking": "Generated."}})
    assert before["messages"][-1]["reasoning_content"] == "Inspect the board."
    expected = copy.deepcopy(before)
    expected["messages"][-1]["reasoning_content"] = "Generated."
    assert result == expected
    assert list(result["tools"][0]["function"]) == list(before["tools"][0]["function"])
    with pytest.raises(ValueError, match="Missing successful"):
        inject_thinking(before, {})


def test_validation_is_game_disjoint_and_format_is_strict():
    folds = {"folds": [{"fold": 0, "game_ids": ["held-123"]}, {"fold": 1, "game_ids": ["train-123"]}]}
    known, held = fold_games(folds, 0)
    assert held == {"held-123"} and known-held == {"train-123"}
    row = sample()
    validate_sample(row, known)
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = '{"code":"print(1)"}'
    with pytest.raises(ValueError, match="code-only mappings"):
        validate_sample(row, known)
    row = sample()
    row.pop("thinking_source")
    with pytest.raises(ValueError, match="thinking_source"):
        validate_sample(row, known)


def model_optimizer():
    torch.manual_seed(3)
    model = backend.LoRALinear(torch.nn.Linear(4, 3, bias=False), 2, 4)
    model.base.requires_grad_(False)
    return model, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0.)


def micro(model, x, y):
    (model(x)-y).square().mean().backward()


def test_partial_accumulation_resume_matches_uninterrupted(tmp_path):
    original, optimizer = model_optimizer()
    torch.manual_seed(11)
    inputs = [(torch.randn(n, 4), torch.randn(n, 3)) for n in (3, 7, 4, 6, 2)]
    identity = {"model": "frozen", "recipe": "same"}
    # One Adam step first so the resume exercises optimizer moments as well.
    micro(original, *inputs[0])
    finish_step(original, optimizer, 1, .01)
    micro(original, *inputs[1])
    checkpoints.save(tmp_path, original, optimizer, identity, cursor=2, step=1, pending=1)
    expected_rng = torch.randn(3)
    micro(original, *inputs[2])
    finish_step(original, optimizer, 2, .007)
    expected = backend.adapter_state(original)
    restored, restored_opt = model_optimizer()
    assert checkpoints.load(tmp_path, restored, restored_opt, identity) == (2, 1, 1)
    torch.testing.assert_close(torch.randn(3), expected_rng, rtol=0, atol=0)
    micro(restored, *inputs[2])
    finish_step(restored, restored_opt, 2, .007)
    for k, v in backend.adapter_state(restored).items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity changed"):
        checkpoints.load(tmp_path, restored, restored_opt, {"model": "different"})
    pointer = json.loads((tmp_path / "latest.json").read_text())
    (tmp_path / pointer["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        checkpoints.load(tmp_path, restored, restored_opt, identity)


def test_request_mean_accumulation_normalizes_partial_batch():
    a, opt = model_optimizer()
    b, opt_b = model_optimizer()
    inputs = [(torch.randn(n, 4), torch.randn(n, 3)) for n in (2, 9, 3)]
    for x, y in inputs:
        micro(a, x, y)
    finish_step(a, opt, 3, .01)
    sum((b(x)-y).square().mean() for x, y in inputs).div(3).backward()
    finish_step(b, opt_b, 1, .01)
    for k, v in backend.adapter_state(a).items():
        torch.testing.assert_close(v, backend.adapter_state(b)[k])


def test_order_and_schedule_do_not_depend_on_session_budget():
    order = epoch_order(list(range(7)), 2, 3)
    assert sorted(order[:7]) == sorted(order[7:]) == list(range(7))
    assert order == epoch_order(list(range(7)), 2, 3)
    assert learning_rate(0, 100, 1.) == .2
    assert learning_rate(99, 100, 1.) == pytest.approx(.1)


def test_partition_filters_before_thinking_join(tmp_path, monkeypatch):
    import prepare_dataset as prep
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": [
        {"fold": 0, "game_ids": ["held-123"]}, {"fold": 1, "game_ids": ["train-123"]}]}))
    held, training = sample(), sample()
    held["game"] = "held-123"
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(held) + "\n" + json.dumps(training) + "\n")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "one.jsonl").write_text(json.dumps({"key": "train-123_p0#2", "status": "ok", "thinking": "New."}))
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "config.json").write_text("{}")
    monkeypatch.setattr(prep, "load_processor", lambda _: object())
    monkeypatch.setattr(prep, "encode", lambda *args: ({}, dict(prompt_tokens=10, target_tokens=3,
        total_tokens=13, images=1, category_counts={"thinking": 1})))
    out = tmp_path / "output"
    prep.prepare(SimpleNamespace(out=out, folds=folds, input=source, processor=processor,
        generated_dir=generated, partition="train", validation_fold=0, max_tokens=20))
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["rows"]) == 1 and manifest["rows"][0]["split"] == "train"
    assert manifest["excluded_by_partition"] == [{"game": "held-123", "request_index": 2}]


def test_checkpoint_retention_keeps_latest_recoverable_payloads(tmp_path):
    model, optimizer = model_optimizer()
    for cursor in range(5):
        checkpoints.save(tmp_path, model, optimizer, {'run': 'retention'}, cursor, cursor, 0)
    payloads = sorted(tmp_path.glob('checkpoint-*.pt'))
    assert len(payloads) == 2
    assert [p.name for p in payloads] == ['checkpoint-00000003-00000003.pt', 'checkpoint-00000004-00000004.pt']
    assert checkpoints.load(tmp_path, model, optimizer, {'run': 'retention'}) == (4, 4, 0)


@pytest.mark.parametrize('reserved_gib,passed', [(91.5,True),(92.5,False)])
def test_qualification_covers_extremes_and_enforces_reserved_headroom(tmp_path,monkeypatch,reserved_gib,passed):
    import run
    model = torch.nn.Linear(1,1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    rows = [dict(sample_id='long',total_tokens=100,images=1,target_tokens=2),
            dict(sample_id='images',total_tokens=80,images=9,target_tokens=2),
            dict(sample_id='target',total_tokens=70,images=1,target_tokens=20)]
    calls = []
    def backward(model,bundle,row,profile,offload):
        calls.append(row['sample_id'])
        loss=model(torch.ones(1,1)).square().sum()
        loss.backward()
        return float(loss.detach())
    monkeypatch.setattr(run,'backward_request',backward)
    for name in ['empty_cache','reset_peak_memory_stats','synchronize']:
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:int(reserved_gib*2**30))
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:90*2**30)
    identity=dict(hardware={'total_memory':96*2**30},recipe={'gpu_headroom_gib':4.})
    if passed:
        run.qualify(model,optimizer,None,rows,{},identity,tmp_path,4,True)
    else:
        with pytest.raises(RuntimeError,match='headroom'):
            run.qualify(model,optimizer,None,rows,{},identity,tmp_path,4,True)
    report=json.loads((tmp_path/'qualification.json').read_text())
    assert report['passed'] is passed
    assert set(calls)=={'long','images','target'} and len(calls)==4
    assert not (tmp_path/'latest.json').exists()
