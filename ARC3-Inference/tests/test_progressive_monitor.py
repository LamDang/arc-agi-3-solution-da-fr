"""Monitoring must not confuse rate limits, pending work, or stale summaries."""
import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/monitor_progressive_sol25.py'
spec = importlib.util.spec_from_file_location('progressive_monitor', SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_job(out):
    save(out / 'manifest.json', {'code': 'fixed'})
    save(out / 'driver_status.json', {'phase': 'full', 'updated': 100,
                                    'driver_pid': 42, 'child_pid': 43})


def test_cadence_requires_matching_validation(tmp_path):
    setup_job(tmp_path)
    assert monitor.snapshot(tmp_path, now=200, alive=lambda *_: True)['interval_seconds'] == 600
    save(tmp_path / 'pilot-validated.json', {'manifest_hash': 'wrong'})
    assert not monitor.pilot_passed(tmp_path)
    save(tmp_path / 'pilot-validated.json', {'manifest_hash': hashlib.sha256(
        (tmp_path / 'manifest.json').read_bytes()).hexdigest()})
    state = monitor.snapshot(tmp_path, now=200, alive=lambda *_: True)
    assert state['interval_seconds'] == 1800
    assert state['next_check_at'] == 2000


def test_live_pending_and_429_are_not_failed_calls(tmp_path):
    setup_job(tmp_path)
    calls = tmp_path / 'turns/game/00000/calls'
    save(calls / 'draft/attempt-00.json', {'state': 'rate_limited', 'charged_usd': .1,
         'rate_limit_retries': 12, 'api': 'flash'})
    save(calls / 'judge1/attempt-00.json', {'state': 'pending', 'charged_usd': .2, 'api': 'sol'})
    save(tmp_path / 'summary.json', {'completed': 1334, 'failures': ['stale pilot failure']})
    state = monitor.snapshot(tmp_path, now=200, alive=lambda *_: True)
    assert state['phase'] == 'full'
    assert len(state['active_429']) == len(state['pending_attempts']) == 1
    assert state['counts']['http_429_retries_current_policy'] == 12
    assert state['alerts'] == []
    assert abs(state['estimated_cost_including_reservations_usd'] - .3) < 1e-10
    dead = monitor.snapshot(tmp_path, now=200, alive=lambda *_: False)
    assert {a['kind'] for a in dead['alerts']} == {'process_missing', 'uncertain_billing'}


def test_completed_stage_clears_active_retry_and_reconciles_usage(tmp_path):
    setup_job(tmp_path)
    stage = tmp_path / 'turns/game/00000/calls/draft'
    save(stage / 'attempt-00.json', {'state': 'error_uncertain', 'charged_usd': .1,
                                   'api': 'flash', 'error': 'HTTP 429'})
    save(stage / 'attempt-01.json', {'state': 'response', 'charged_usd': .02, 'api': 'flash',
         'response': {'usage': {'prompt_tokens': 100, 'completion_tokens': 10,
                               'prompt_tokens_details': {'cached_tokens': 75}}}})
    save(stage / 'complete.json', {'value': 'finished'})
    state = monitor.snapshot(tmp_path, alive=lambda *_: True)
    assert state['active_429'] == []
    assert state['pending_attempts'] == []
    assert state['usage']['flash']['cache_fraction'] == .75
    assert state['response_estimated_cost_usd'] == .02
    assert state['unreconciled_reservations_usd'] == .1
    assert state['counts']['legacy_429_attempts'] == 1


def test_terminal_state_cannot_claim_success_from_stale_summary(tmp_path):
    setup_job(tmp_path)
    save(tmp_path / 'driver_status.json', {'phase': 'completed'})
    save(tmp_path / 'summary.json', {'completed': 1334})
    state = monitor.snapshot(tmp_path, alive=lambda *_: False)
    assert state['next_check_at'] is None
    assert state['alerts'][0]['kind'] == 'invalid_completion'


def test_four_worker_trial_checks_10m_then_returns_to_30m(tmp_path):
    setup_job(tmp_path)
    save(tmp_path / 'pilot-validated.json', {'manifest_hash': hashlib.sha256(
        (tmp_path / 'manifest.json').read_bytes()).hexdigest()})
    save(tmp_path / 'four-workers-transition.json', {
        'new_driver': {'driver_pid':42}, 'started_at':100, 'baseline_finalized':0})
    early = monitor.snapshot(tmp_path, now=200, alive=lambda *_:True)
    assert early['interval_seconds'] == 600
    assert not early['worker_trial']['initial_10m_health_check_complete']
    save(tmp_path / 'turns/game/00000/final.json', {'game':'game', 'key':'game#0',
        'training_eligible':True, 'student_input_tokens':10})
    late = monitor.snapshot(tmp_path, now=800, alive=lambda *_:True)
    assert late['interval_seconds'] == 1800
    assert late['worker_trial']['initial_10m_health_check_complete']


def test_pid_identity(tmp_path):
    import os
    assert not monitor.process_alive(os.getpid(), 'think_gen.progressive')
    assert not monitor.process_alive(-1, 'think_gen.progressive')
    assert not monitor.process_alive(None, 'think_gen.progressive')


def test_new_worker_trial_overrides_previous_and_requires_matching_driver(tmp_path):
    setup_job(tmp_path)
    save(tmp_path / 'pilot-validated.json', {'manifest_hash': hashlib.sha256(
        (tmp_path / 'manifest.json').read_bytes()).hexdigest()})
    save(tmp_path / 'four-workers-transition.json', {
        'new_driver': {'driver_pid': 42}, 'started_at': 1, 'baseline_finalized': 0})
    transition = {'to_workers': 8, 'new_driver': {'driver_pid': 42},
                  'started_at': 1000, 'baseline_finalized': 0}
    save(tmp_path / 'workers-transition.json', transition)
    early = monitor.snapshot(tmp_path, now=1100, alive=lambda *_: True)
    assert early['worker_trial']['workers'] == 8
    assert early['interval_seconds'] == 600
    save(tmp_path / 'turns/game/00000/final.json', {'game': 'game', 'key': 'game#0',
         'training_eligible': True, 'student_input_tokens': 10})
    late = monitor.snapshot(tmp_path, now=1700, alive=lambda *_: True)
    assert late['worker_trial']['initial_10m_health_check_complete']
    assert late['interval_seconds'] == 1800
    transition['new_driver']['driver_pid'] = 99
    save(tmp_path / 'workers-transition.json', transition)
    assert monitor.snapshot(tmp_path, now=1700, alive=lambda *_: True)['worker_trial'] is None


def test_pid_output_directory_identity(tmp_path):
    import subprocess
    import sys
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
                              '--out', str(tmp_path)])
    try:
        assert monitor.process_alive(child.pid, sys.executable, tmp_path)
        assert not monitor.process_alive(child.pid, sys.executable, tmp_path / 'another-run')
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_absolute_script_identity_accepts_relative_launch(tmp_path):
    import subprocess
    import sys
    script = 'scripts/run_progressive_sol25.py'
    child = subprocess.Popen([sys.executable, '-c',
                              'import time; print("ready", flush=True); time.sleep(30)', script],
                             cwd=tmp_path, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        assert monitor.process_alive(child.pid, str(tmp_path / script))
        assert not monitor.process_alive(child.pid, str(tmp_path / 'other' / script))
    finally:
        child.terminate()
        child.wait(timeout=5)
