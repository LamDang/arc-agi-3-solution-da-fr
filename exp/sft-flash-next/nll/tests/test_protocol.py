import numpy as np
import pytest

from common import read_json
from protocol import budget, choice, execute_stages, gate
from report import generate
from results import save_result
from test_results import synthetic_run


def test_relative_primary_threshold_is_inclusive_and_not_perplexity():
    assert gate(2., 2.10)['status'] == 'stop_at_256'
    assert gate(2., 2.10001)['status'] == 'scan_required'
    assert gate(2., 1.8)['status'] == 'stop_at_256'
    assert gate(None, None)['status'] == 'pending'
    assert gate(0., 0.)['status'] == 'stop_at_256'
    assert gate(0., .01)['status'] == 'scan_required'
    with pytest.raises(ValueError):
        gate(2., float('nan'))
    assert choice({'512': {'primary': 2.}, '448': {'primary': 1.95},
                   '384': {'primary': 2.08}, '320': {'primary': 2.11},
                   '256': {'primary': 2.2}}) == 384


@pytest.mark.parametrize('candidate,scan', [(2.05, False), (2.2, True)])
def test_exact_stage_order_conditional_completion_and_resume(tmp_path, candidate, scan):
    manifest, identity = synthetic_run(tmp_path)
    samples = manifest['samples']
    called, completed = [], set()
    def evaluate(row, count):
        key = (count, row['sample_id'])
        if key in completed:
            return
        # A scan may never be admitted while either initial stage is partial.
        if count in (448, 384, 320):
            assert all((c, r['sample_id']) in completed for c in (512, 256) for r in samples)
        called.append(count)
        loss = candidate if count == 256 else 2.
        save_result(tmp_path, identity, count, row, np.full(3, loss), row['positions'])
        completed.add(key)
    selection = execute_stages(samples, evaluate, lambda: generate(tmp_path))
    assert called == [512]*30 + [256]*30 + ([448]*30 + [384]*30 + [320]*30 if scan else [])
    assert selection['complete'] is True
    assert selection['selected_experts'] == (320 if scan else 256)
    assert selection['required_jobs'] == (150 if scan else 60)
    assert selection['gate']['status'] == ('scan_required' if scan else 'stop_at_256')
    if not scan:
        assert not (tmp_path/'results'/'448').exists()
        assert read_json(tmp_path/'comparison.json')['summaries']['448']['primary'] is None
    before = list(called)
    assert execute_stages(samples, evaluate, lambda: generate(tmp_path)) == selection
    assert called == before


def test_partial_pair_cannot_decide_or_expand(tmp_path):
    manifest, identity = synthetic_run(tmp_path)
    for row in manifest['samples']:
        save_result(tmp_path, identity, 512, row, [1, 1, 1], row['positions'])
    for row in manifest['samples'][:-1]:
        save_result(tmp_path, identity, 256, row, [10, 10, 10], row['positions'])
    decision = generate(tmp_path)
    assert decision['gate']['status'] == 'pending'
    assert not decision['complete'] and decision['selected_experts'] is None
    assert decision['matched_requests'] == 29


def test_interrupted_baseline_does_not_admit_256():
    called = []
    def evaluate(row, count):
        called.append(count)
        if row == 2:
            raise InterruptedError('deadline')
    with pytest.raises(InterruptedError):
        execute_stages([0, 1, 2, 3], evaluate, lambda: {})
    assert called == [512, 512, 512]


def test_budget_does_not_charge_intermediates_on_early_stop():
    b = budget({'samples': [dict(total_tokens=100, target_tokens=10)]*30})
    assert b['initial_processed_tokens'] == 6000
    assert b['conditional_scan_processed_tokens'] == 9000
    assert b['maximum_processed_tokens'] == 15000


@pytest.mark.parametrize("candidate", [1.8, 2.2])
def test_manual_pair_waits_for_decision_then_explicit_expansion(tmp_path, candidate):
    from common import write_json
    manifest, identity = synthetic_run(tmp_path)
    run = read_json(tmp_path / "run.json")
    run["config"]["scan_control"] = "manual"
    write_json(tmp_path / "run.json", run)
    write_json(tmp_path / "scope.json", dict(identity=identity, counts=[512, 256]))
    called, completed = [], set()
    def evaluate(row, count):
        key = (count, row["sample_id"])
        if key in completed:
            return
        called.append(count)
        save_result(tmp_path, identity, count, row,
                    np.full(3, candidate if count == 256 else 2.), row["positions"])
        completed.add(key)
    result = execute_stages(manifest["samples"], evaluate, lambda: generate(tmp_path), scan_counts=[])
    assert called == [512]*30 + [256]*30
    assert result["complete"] and result["selected_experts"] is None
    assert result["status"] == "awaiting_expansion_decision"
    assert result["deferred_counts"] == [448, 384, 320]
    write_json(tmp_path / "scope.json", dict(identity=identity, counts=[512, 256, 320]))
    result = execute_stages(manifest["samples"], evaluate, lambda: generate(tmp_path), scan_counts=[320])
    assert called == [512]*30 + [256]*30 + [320]*30
    assert result["complete"] and result["required_jobs"] == 90
    execute_stages(manifest["samples"], evaluate, lambda: generate(tmp_path), scan_counts=[320])
    assert len(called) == 90
