"""Read-only job checks every 10 minutes before validation, then every 30.

Writes only monitor_* files. It never sends inference requests, changes the
runner, retries uncertain calls, or interprets live pending calls as failures.
This is a local watchdog, not a hosted chat notification or VM wake-up service.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'ARC3-Inference/runs/think-progressive-sampled-pilot'
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen.progressive import load_billing_adjustments  # noqa: E402


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def process_alive(pid, expected, out=None):
    """Check command identity and reject zombies or a reused PID."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = Path('/proc') / str(pid)
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        matches = state != 'Z' and any(expected == arg or arg.endswith('/' + expected) for arg in argv)
        if state != 'Z' and Path(expected).is_absolute():
            try:
                cwd = (proc / 'cwd').resolve()
            except PermissionError:
                # Managed exec sandboxes can deny another session's cwd link.
                # This launcher's documented working directory is the repo root.
                cwd = ROOT
            matches = any((Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() == Path(expected).resolve()
                          for arg in argv if arg and Path(arg).name == Path(expected).name)
        if out is not None:
            try:
                value = argv[argv.index('--out') + 1] if '--out' in argv else next(
                    arg.split('=', 1)[1] for arg in argv if arg.startswith('--out='))
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = (proc / 'cwd').resolve() / candidate
                matches = matches and candidate.resolve() == out.resolve()
            except (ValueError, IndexError, StopIteration):
                return False
        return matches
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def pilot_passed(out):
    validation = read(out / 'pilot-validated.json', {})
    manifest = out / 'manifest.json'
    return bool(validation and manifest.exists() and
                validation.get('manifest_hash') == hashlib.sha256(manifest.read_bytes()).hexdigest())


def audit_failed(verdict):
    return bool(verdict and (
        not all(verdict['words']['covered']) or verdict['words']['contradictions'] or
        not verdict['code']['leads_to_call'] or verdict['code']['disagreements'] or
        not verdict['fact']['grounded'] or verdict['fact']['errors'] or
        ('call' in verdict and not verdict['call']['functionally_same'])))


def snapshot(out, now=None, alive=process_alive):
    now = time.time() if now is None else now
    driver = read(out / 'driver_status.json', {})
    phase = driver.get('phase', 'unknown')
    passed = pilot_passed(out)
    interval = 1800 if passed else 600
    child_alive = alive(driver.get('child_pid'), 'think_gen.progressive', out)
    driver_alive = alive(driver.get('driver_pid'), str(ROOT / 'ARC3-Inference/scripts/run_progressive_sol25.py'))
    counts = Counter()
    games = Counter()
    audit_failures = []
    max_input = 0
    last_progress = driver.get('updated', 0)
    for p in out.glob('turns/*/*/final.json'):
        row = read(p)
        counts['finalized'] += 1
        counts['eligible'] += bool(row['training_eligible'])
        counts['excluded'] += not row['training_eligible']
        counts['audited'] += bool(row.get('monitoring_sample'))
        games[row['game']] += 1
        max_input = max(max_input, row['student_input_tokens'])
        if audit_failed(row.get('final_judge')):
            audit_failures.append(row['key'])
        last_progress = max(last_progress, p.stat().st_mtime)
    for p in out.glob('turns/*/*/calls/*/complete.json'):
        counts['completed_stages'] += 1
        last_progress = max(last_progress, p.stat().st_mtime)
    charged = actual = reserved = adjusted_usd = 0.
    billing_adjustments = load_billing_adjustments(out)
    tokens = {'flash': Counter(), 'sol': Counter()}
    sol_policies = {}
    states = Counter()
    active_429 = []
    pending = []
    for p in out.glob('turns/*/*/calls/*/attempt-*.json'):
        row = read(p)
        states[row['state']] += 1
        charge = billing_adjustments.get(str(p), row['charged_usd'])
        adjusted_usd += row['charged_usd'] - charge
        charged += charge
        counts['http_429_retries_current_policy'] += row.get('rate_limit_retries', 0)
        counts['legacy_429_attempts'] += bool(row['state'] == 'error_uncertain' and
                                           'HTTP 429' in row.get('error', ''))
        response = row.get('response')
        if response is not None:
            actual += charge
            usage = response.get('usage') or {}
            api = row['api']
            tokens[api]['input'] += usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
            tokens[api]['output'] += usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
            details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
            tokens[api]['cached_input'] += details.get('cached_tokens', 0) or 0
            tokens[api]['cache_write_input'] += details.get('cache_write_tokens', 0) or 0
            if api == 'sol':
                policy = sol_policies.setdefault(row.get('cache_policy') or 'legacy-implicit', Counter())
                policy['input'] += usage.get('input_tokens', 0) or 0
                policy['cached_input'] += details.get('cached_tokens', 0) or 0
                policy['cache_write_input'] += details.get('cache_write_tokens', 0) or 0
                policy['responses'] += 1
        else:
            reserved += charge
        # Older failed/429 attempts remain journaled: only the latest unfinished
        # slot describes current work. Completed stages have no active request.
        if not (p.parent / 'complete.json').exists() and p == sorted(p.parent.glob('attempt-*.json'))[-1]:
            rel = str(p.relative_to(out))
            if row['state'] == 'rate_limited':
                active_429.append({'path': rel, 'retries': row.get('rate_limit_retries', 0)})
            elif row['state'] == 'pending':
                pending.append(rel)
    for values in tokens.values():
        values['cache_fraction'] = values['cached_input'] / values['input'] if values['input'] else None
    for values in sol_policies.values():
        values['cache_fraction'] = values['cached_input'] / values['input'] if values['input'] else None
    worker_trial = None
    transition = read(out / 'workers-transition.json', None)
    if transition is None:
        transition = read(out / 'four-workers-transition.json', {})
    if transition.get('new_driver', {}).get('driver_pid') == driver.get('driver_pid') and transition.get('started_at'):
        elapsed = max(0, now - transition['started_at'])
        progressed = max(0, counts['finalized'] - transition['baseline_finalized'])
        observed = elapsed >= 600 and progressed > 0 and child_alive and driver_alive
        worker_trial = {'workers': transition.get('to_workers', 4), 'elapsed_seconds': elapsed,
                        'finalized_since_restart': progressed,
                        'turns_per_hour': progressed * 3600 / elapsed if elapsed else None,
                        'initial_10m_health_check_complete': observed}
        if not observed:
            interval = 600
    alerts = []
    if phase == 'completed':
        if counts['finalized'] == 1334:
            alerts.append({'kind': 'completed', 'detail': 'All 1334 responses finalized.'})
        else:
            alerts.append({'kind': 'invalid_completion', 'detail': 'Completed phase has fewer than 1334 finalized responses.'})
    elif phase == 'stopped':
        alerts.append({'kind': 'stopped', 'detail': driver.get('error', 'Runner stopped.')[:300]})
    elif not driver_alive or not child_alive:
        alerts.append({'kind': 'process_missing', 'detail': 'Driver or runner is absent; inspect before resuming.'})
    if not child_alive and pending:
        alerts.append({'kind': 'uncertain_billing', 'detail': f'{len(pending)} pending attempts after runner exit; do not automatically resend.'})
    if phase not in ('completed', 'stopped') and now - last_progress >= 1800:
        detail = ('Provider 429 retries continue; retry policy is unchanged.' if active_429 else
                  'No stage or turn completed for 30 minutes; inspect the runner log.')
        alerts.append({'kind': 'no_progress_30m', 'detail': detail})
    if audit_failures:
        alerts.append({'kind': 'quality_monitoring', 'detail': f'{len(audit_failures)} sampled final audits reported issues.'})
    return {'checked_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'checked_at': now, 'monitor_pid': os.getpid(), 'phase': phase,
            'pilot_passed': passed, 'interval_seconds': interval,
            'next_check_at': None if phase in ('completed', 'stopped') else now + interval,
            'driver_pid': driver.get('driver_pid'), 'child_pid': driver.get('child_pid'),
            'driver_alive': driver_alive, 'child_alive': child_alive,
            'counts': dict(counts), 'finalized_by_game': dict(games),
            'max_student_input_tokens': max_input,
            'estimated_cost_including_reservations_usd': charged,
            'response_estimated_cost_usd': actual, 'unreconciled_reservations_usd': reserved,
            'billing_adjustment_count': len(billing_adjustments),
            'billing_adjustment_usd': adjusted_usd,
            'usage': {k: dict(v) for k, v in tokens.items()}, 'attempt_states': dict(states),
            'sol_usage_by_cache_policy': {k: dict(v) for k, v in sol_policies.items()},
            'worker_trial': worker_trial,
            'active_429': active_429, 'pending_attempts': pending,
            'seconds_since_completed_stage_or_turn': max(0, now - last_progress),
            'audit_issue_keys': audit_failures, 'alerts': alerts}


def append_json(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--once', action='store_true', help='Read and print; write no monitor files.')
    args = parser.parse_args()
    if args.once:
        print(json.dumps(snapshot(args.out), indent=2))
        return
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with (args.out / '.monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(args.out / 'monitor_status.json', {})
        known = {json.dumps(a, sort_keys=True) for a in previous.get('alerts', [])}
        while not stop.is_set():
            try:
                state = snapshot(args.out)
                atomic_json(args.out / 'monitor_status.json', state)
                append_json(args.out / 'monitor_history.jsonl', state)
                current = {json.dumps(a, sort_keys=True) for a in state['alerts']}
                for key in sorted(current - known):
                    append_json(args.out / 'monitor_alerts.jsonl', {'checked_utc': state['checked_utc'], **json.loads(key)})
                known = current
                print(json.dumps(state), flush=True)
                if state['phase'] in ('completed', 'stopped'):
                    return
                stop.wait(max(0, state['next_check_at'] - time.time()))
            except (OSError, ValueError, KeyError, TypeError) as e:
                # Monitoring failures never affect generation. Retry the read
                # in a minute, preserving the last good snapshot.
                print(f'Monitor read failed: {type(e).__name__}: {e}', flush=True)
                stop.wait(60)


if __name__ == '__main__':
    main()
