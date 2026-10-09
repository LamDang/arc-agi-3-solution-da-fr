import copy
import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from trajectory_head_loss import trajectory_loss, supervised_positions
from trajectory_training import labels_for_annotation, finish_token_update
import trajectory_checkpoints
import json
from types import SimpleNamespace
from dataset import file_hash
from trajectory_training import validate_clean_initial_adapter


@pytest.mark.parametrize('rows', [None, 32])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_disjoint_targets_match_native_full_ce(rows, reduction):
    torch.manual_seed(49)
    weight = torch.randn(73, 13)
    original = torch.randn(1, 29, 13)
    labels = torch.full((1, 29), -100, dtype=torch.long)
    # Thinking/tool-call/final spans separated by observation context.
    positions = torch.tensor([2, 3, 4, 12, 13, 22, 23, 24, 28])
    labels[0, positions] = torch.randint(0, 73, (len(positions),))
    native, candidate = (original.clone().requires_grad_() for _ in range(2))
    denominator = 1 if reduction == 'sum' else len(positions)
    expected = ForCausalLMLoss(torch.nn.functional.linear(native, weight), labels,
                               vocab_size=73, num_items_in_batch=denominator)
    actual = trajectory_loss(candidate, weight, labels, rows, reduction)
    (expected * .125).backward()
    (actual * .125).backward()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(candidate.grad, native.grad, rtol=1e-6, atol=1e-7)
    unowned = torch.ones(29, dtype=torch.bool)
    unowned[positions-1] = False
    assert torch.count_nonzero(candidate.grad[0, unowned]) == 0
    assert torch.count_nonzero(candidate.grad[0, -1]) == 0


def test_annotation_labels_and_native_shift():
    ids = torch.arange(20).view(1, -1)
    annotation = dict(positions=[2, 3, 10, 19], target_ids=[2, 3, 10, 19], target_tokens=4)
    labels = labels_for_annotation(ids, annotation)
    hidden, target = supervised_positions(labels)
    assert hidden.tolist() == [1, 2, 9, 18]
    assert target.tolist() == annotation['target_ids']
    with pytest.raises(ValueError, match='unique'):
        labels_for_annotation(ids, dict(positions=[2, 2], target_ids=[2, 2], target_tokens=2))
    with pytest.raises(ValueError, match='next-token'):
        labels_for_annotation(ids, dict(positions=[0], target_ids=[0], target_tokens=1))


def test_multispan_native_autocast_cast_back_and_sum_scaling():
    torch.manual_seed(511)
    original = torch.randn(1,29,17)
    weight = torch.randn(113,17).to(torch.bfloat16)
    labels = torch.full((1,29),-100,dtype=torch.long)
    labels[0,[2,3,12,13,23,28]] = torch.randint(0,113,(6,))
    native,candidate = (original.clone().requires_grad_() for _ in range(2))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        expected = ForCausalLMLoss(torch.nn.functional.linear(native,weight),labels,
                                   vocab_size=113,num_items_in_batch=1)
        actual = trajectory_loss(candidate,weight,labels,reduction='sum')
    (expected*.125).backward()
    (actual*.125).backward()
    assert candidate.grad.dtype == native.grad.dtype == torch.float32
    assert torch.equal(candidate.grad,native.grad)
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=0)


def unequal_examples():
    torch.manual_seed(191)
    return [(torch.randn(n, 3, dtype=torch.float64), torch.randint(0, 5, (n,)))
            for n in [2, 7, 3]]


def accumulate(model, examples):
    total = 0
    for inputs, targets in examples:
        torch.nn.functional.cross_entropy(model(inputs), targets, reduction='sum').backward()
        total += targets.numel()
    return total


def test_token_weighted_update_matches_concatenated_loss_and_clips_after_division():
    torch.manual_seed(12)
    model = torch.nn.Linear(3, 5, dtype=torch.float64)
    reference = copy.deepcopy(model)
    examples = unequal_examples()
    opt = torch.optim.SGD(model.parameters(), lr=.4)
    refopt = torch.optim.SGD(reference.parameters(), lr=.4)
    total = accumulate(model, examples)
    x = torch.cat([x for x,y in examples])
    y = torch.cat([y for x,y in examples])
    torch.nn.functional.cross_entropy(reference(x), y, reduction='mean').backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .03)
    refopt.step()
    norm = finish_token_update(model, opt, total, max_norm=.03)
    assert norm == pytest.approx(float(expected_norm), rel=1e-14)
    for a,b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-14)
        assert a.grad is None


def test_ignored_context_counts_do_not_change_update_denominator():
    torch.manual_seed(231)
    model = torch.nn.Linear(3, 5, dtype=torch.float64)
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(),lr=.1)
    refopt = torch.optim.SGD(reference.parameters(),lr=.1)
    supervised_logits, targets = [], []
    count = 0
    for length, positions in [(11,[2,8]),(39,[3,4,19,22,38])]:
        inputs = torch.randn(length,3,dtype=torch.float64)
        labels = torch.full((length,),-100,dtype=torch.long)
        labels[positions] = torch.randint(0,5,(len(positions),))
        torch.nn.functional.cross_entropy(model(inputs),labels,reduction='sum').backward()
        supervised_logits.append(reference(inputs)[positions])
        targets.append(labels[positions])
        count += int(torch.count_nonzero(labels != -100))
    assert count == 7  # 50 context rows and 2 trajectories never enter this denominator.
    torch.nn.functional.cross_entropy(torch.cat(supervised_logits),torch.cat(targets)).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    refopt.step()
    finish_token_update(model,opt,count)
    for actual,expected in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)


def test_resume_preserves_partial_raw_sum_and_token_denominator(tmp_path):
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 5, dtype=torch.float64)
    restored = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    examples = unequal_examples()
    pending = accumulate(model, examples[:1])
    identity = dict(objective='all-assistant', plan='immutable-test-plan', token_budget=10)
    trajectory_checkpoints.save(tmp_path, model, optimizer, identity, 1, 0, pending, 1, 4.5)
    assert trajectory_checkpoints.load(tmp_path, restored, new_optimizer, identity) == (1, 0, 2, 1, 4.5)
    for a,b in zip(model.parameters(),restored.parameters()):
        assert torch.equal(a.grad,b.grad)
    total = pending + accumulate(model,examples[1:])
    other_total = pending + accumulate(restored,examples[1:])
    finish_token_update(model,optimizer,total)
    finish_token_update(restored,new_optimizer,other_total)
    for a,b in zip(model.parameters(),restored.parameters()):
        assert torch.equal(a,b)
    with pytest.raises(ValueError, match='identity'):
        trajectory_checkpoints.load(tmp_path, restored, new_optimizer, {'objective':'legacy'})


def test_rejects_insufficient_backward_rows_and_empty_supervision():
    hidden = torch.randn(1, 9, 3, requires_grad=True)
    labels = torch.full((1, 9), -100, dtype=torch.long)
    with pytest.raises(ValueError, match='no supervised'):
        trajectory_loss(hidden,torch.randn(7,3),labels)
    labels[0, [2,4,7]] = 1
    with pytest.raises(ValueError, match='all assistant'):
        trajectory_loss(hidden,torch.randn(7,3),labels,2)


def test_runner_whole_plan_tail_and_midupdate_resume(tmp_path,monkeypatch):
    import trajectory_run as runner
    from trajectory_data import plan_token_updates
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for name in ['reset_peak_memory_stats','max_memory_allocated','max_memory_reserved']:
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(runner.signal,'signal',lambda *_:None)
    rows = [dict(sample_id=str(i),target_tokens=n,total_tokens=100+n,images=i%2)
            for i,n in enumerate([2,7,3,4])]
    plans = plan_token_updates(rows,10)
    def backward(model,bundle,row,flags,args):
        x = torch.arange(row['target_tokens']*3,dtype=torch.float64).view(-1,3)/10
        loss = model(x).square().sum()
        loss.backward()
        return float(loss.detach()),row['target_tokens']
    monkeypatch.setattr(runner,'backward_trajectory',backward)
    torch.manual_seed(88)
    model = torch.nn.Linear(3,1,dtype=torch.float64)
    reference = copy.deepcopy(model)
    resumed = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(),lr=.01)
    refopt = torch.optim.AdamW(reference.parameters(),lr=.01)
    resopt = torch.optim.AdamW(resumed.parameters(),lr=.01)
    identity = dict(plan='same-ordered-whole-trajectories',budget=10)
    args = SimpleNamespace(out=str(tmp_path/'resume'),resume=False,session_minutes=1e-12,
        max_updates=0,max_grad_norm=1.,lr=.01,token_budget=10)
    runner.train(model,opt,None,rows,plans,{},args,identity)
    state,_ = trajectory_checkpoints.read_state(args.out,identity)
    assert (state['cursor'],state['step'],state['pending_supervised_tokens'],state['pending_trajectories']) == (1,0,2,1)
    args.resume = True
    args.session_minutes = 0
    runner.train(resumed,resopt,None,rows,plans,{},args,identity)
    args.resume = False
    args.out = str(tmp_path/'complete')
    runner.train(reference,refopt,None,rows,plans,{},args,identity)
    for got,expected in zip(resumed.parameters(),reference.parameters()):
        assert torch.equal(got,expected)
    metrics = [json.loads(line) for line in (tmp_path/'resume/metrics.jsonl').read_text().splitlines()]
    updates = [m['update'] for m in metrics if m['update']]
    assert [(u['actual_supervised_tokens'],u['trajectories'],u['final_short_batch']) for u in updates] == [(12,3,False),(4,1,True)]
    assert updates[0]['overshoot_tokens']==2
    assert json.loads((tmp_path/'resume/status.json').read_text())['complete']


def test_evaluation_restores_trained_checkpoint_and_records_provenance(tmp_path):
    from trajectory_run import evaluation_provenance
    model = torch.nn.Linear(3,1)
    initial = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(),lr=.1)
    model(torch.ones(3,3)).square().sum().backward()
    finish_token_update(model,opt,3)
    assert any(not torch.equal(a,b) for a,b in zip(model.parameters(),initial.parameters()))
    identity = dict(objective='all-assistant',model='test-model',plan='test-plan')
    trajectory_checkpoints.save(tmp_path,model,opt,identity,2,1,0,0,0.)
    provenance = evaluation_provenance(initial,tmp_path,identity)
    assert provenance['step']==1 and provenance['cursor']==2 and provenance['sha256']
    for expected,got in zip(model.parameters(),initial.parameters()):
        assert torch.equal(expected,got)
        assert got.grad is None
    with pytest.raises(ValueError,match='explicit trained'):
        evaluation_provenance(initial,None,identity)


def test_production_initialization_rejects_nonzero_diagnostic_b(tmp_path):
    state = {f'layer{i}.lora_{kind}.weight':torch.ones(2) if kind=='A' else torch.zeros(2)
             for i in range(372) for kind in ['A','B']}
    path = tmp_path/'initial.pt'
    torch.save(state,path)
    provenance = dict(purpose='clean-production-initialization',validation_exposed=False,
                      optimizer_updates=0,adapter_sha256=file_hash(path))
    meta = tmp_path/'provenance.json'
    meta.write_text(json.dumps(provenance))
    assert validate_clean_initial_adapter(path,meta)==provenance
    for name,value in state.items():
        if 'lora_B' in name:
            value.fill_(.01)
    torch.save(state,path)
    provenance['adapter_sha256'] = file_hash(path)
    meta.write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match='never diagnostic'):
        validate_clean_initial_adapter(path,meta)
    provenance['optimizer_updates'] = 5
    meta.write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match='mismatch'):
        validate_clean_initial_adapter(path,meta)
