"""Publish immutable partial/final DVC snapshots while generation continues.

The user authorized a checkpoint/PR after most games complete, then a final
commit. This watcher never sends inference requests or modifies live turn data.
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
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen import logs
from think_gen.progressive import StudentFormat, atomic_json, digest, export_game, file_hash, read_json

LIVE = ROOT / 'ARC3-Inference/runs/think-progressive-sampled-pilot'
RELEASE = ROOT / 'ARC3-Inference/runs/think-progressive-sol25'
SOURCE = ROOT / 'ARC3-Inference/runs/gpt61sol-features-25games'
DATA = ROOT / 'data/sft-gpt61sol-features-25games/qwen'
BRANCH = 'codex/progressive-sol25-20261008'
RESULTS = ROOT / 'ARC3-Inference/experiments/teacher-reasoning/progressive-sol25-results.json'
POINTER = 'ARC3-Inference/runs/think-progressive-sol25.dvc'
FILES = [
    'ARC3-Inference/think_gen/README.md', 'ARC3-Inference/think_gen/client.py',
    'ARC3-Inference/think_gen/progressive.py',
    *[f'ARC3-Inference/scripts/{name}.py' for name in (
        'check_sol_history_cache', 'migrate_progressive_sol25_cache',
        'monitor_progressive_sol25', 'run_progressive_sol25',
        'recover_progressive_sol25_filtered_regen',
        'wake_progressive_sol25', 'migrate_progressive_sol25_monitoring',
        'migrate_progressive_sol25_http_billing',
        'manage_progressive_sol25_release')],
    *[f'ARC3-Inference/tests/{name}.py' for name in (
        'test_progressive', 'test_progressive_monitor', 'test_sol_cache',
        'test_progressive_release')],
    *[f'ARC3-Inference/experiments/teacher-reasoning/progressive-sol25{name}' for name in (
        '.md', '-cache-validation.json', '-launch.json', '-monitoring.json',
        '-pilot.json', '-qwen-max-availability.json', '-results.json')],
    POINTER,
]


def completed_rows(live):
    return [(p, read_json(p)) for p in sorted(live.glob('turns/*/*/final.json'))]


def verify_history(rows):
    """Fail closed on a missing/changed retained final thinking dependency."""
    refs = {}
    for _, row in rows:
        ref = (row['game'], row['ref'])
        if ref in refs:
            raise ValueError(f'Duplicate finalized reference: {row["key"]}')
        refs[ref] = digest(row['thinking'])
    for _, row in rows:
        prior = row['history_refs']
        if row['history_hash'] != digest(prior):
            raise ValueError(f'Invalid history hash: {row["key"]}')
        for ref, expected in prior.items():
            if refs.get((row['game'], ref)) != expected:
                raise ValueError(f'Missing/changed retained thinking: {row["key"]}')


def copy_finalized(live, target, rows):
    """Only finalized folders are immutable; exclude in-flight turn journals."""
    verify_history(rows)
    for path, row in rows:
        relative = path.parent.relative_to(live)
        for attempt in (path.parent / 'calls').glob('*/attempt-*.json'):
            state = read_json(attempt)['state']
            if state in ('pending', 'rate_limited', 'reserved'):
                raise ValueError(f'Unfinished request inside finalized turn: {row["key"]}')
            if state == 'monitoring_uncertain' and attempt.parent.name not in ('regen', 'final_audit'):
                raise ValueError(f'Uncertain non-monitoring request inside finalized turn: {row["key"]}')
        shutil.copytree(path.parent, target / relative,
                        ignore=shutil.ignore_patterns('*.tmp'))
        if file_hash(path) != file_hash(target / relative / 'final.json'):
            raise ValueError(f'Final changed while copying: {row["key"]}')


def summarize_audits(rows):
    audits = [row['final_judge'] for _, row in rows if row.get('final_judge')]
    categories = Counter()
    flagged = 0
    for verdict in audits:
        issues = {
            'coverage': not all(verdict['words']['covered']) or bool(verdict['words']['contradictions']),
            'code_consistency': not verdict['code']['leads_to_call'] or bool(verdict['code']['disagreements']),
            'fact_grounding': not verdict['fact']['grounded'] or bool(verdict['fact']['errors']),
            'call_equivalence': 'call' in verdict and not verdict['call']['functionally_same'],
        }
        categories.update(key for key, issue in issues.items() if issue)
        flagged += any(issues.values())
    exceptions = [{'key': row['key'], 'detail': row['monitoring_exception']}
                  for _, row in rows if row.get('monitoring_exception')]

    def details(value):
        return value if isinstance(value, list) else [value]

    return {
        'requested': sum(bool(row.get('monitoring_sample')) for _, row in rows),
        'available': len(audits),
        'unavailable': sum(bool(row.get('monitoring_sample')) and row.get('final_judge') is None
                           for _, row in rows),
        'equivalence_unavailable': sum(
            bool(row.get('monitoring_sample')) and (
                (bool(row.get('monitoring_exception')) and
                 any('regen' in str(detail).lower() or 'equivalence' in str(detail).lower()
                     for detail in details(row['monitoring_exception']))) or
                (row.get('regenerated_code') is not None and row.get('final_judge') is None))
            for _, row in rows),
        'clean': len(audits) - flagged,
        'flagged': flagged,
        'categories_overlap': dict(categories),
        'exceptions': exceptions,
    }


def snapshot(kind):
    rows = completed_rows(LIVE)
    manifest = read_json(LIVE / 'manifest.json')
    for path, expected in manifest['code'].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f'Inference code differs from generation manifest: {path}')
    for name, expected in manifest['source'].items():
        if file_hash(SOURCE / name) != expected:
            raise ValueError(f'Source differs from generation manifest: {name}')
    if file_hash(DATA / 'tokenizer.json') != manifest['tokenizer'] or file_hash(DATA / 'chat_template.jinja') != manifest['template']:
        raise ValueError('Tokenizer/template differs from generation manifest')
    staging = RELEASE.with_name(RELEASE.name + '.snapshot-tmp')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    copy_finalized(LIVE, staging, rows)
    for name in ('manifest.json', 'pilot-validated.json', 'cache-migration.json',
                 'transport-migration.json', 'cache-validation.json',
                 'monitoring-optional-migration.json',
                 'manifest.before-monitoring-optional.json',
                 'http-failure-billing-adjustments.json',
                 'http-failure-billing-migration.json',
                 'manifest.before-http-failure-billing.json',
                 'pilot-validated.before-http-failure-billing.json',
                 'manifest.before-429-retries.json', 'manifest.before-explicit-cache.json',
                 'pilot-validated.before-explicit-cache.json'):
        path = LIVE / name
        if path.exists():
            shutil.copy2(path, staging / name)
    migration = LIVE / 'cache-migration.json'
    if migration.exists():
        evidence = read_json(migration)
        archive = evidence['archive']
        if Path(archive).name != archive or file_hash(LIVE / archive) != evidence['archive_sha256']:
            raise ValueError('Invalid referenced cache-migration archive')
        shutil.copy2(LIVE / archive, staging / archive)
    student = StudentFormat(DATA / 'tokenizer.json', DATA / 'chat_template.jinja')
    args = SimpleNamespace(out=staging, limit=0)
    exported = [row for path in logs.request_logs(SOURCE) for row in export_game(path, args, student)]
    if len(exported) != len(rows):
        raise ValueError('Snapshot does not contain complete consecutive game prefixes')
    if kind == 'final' and len(exported) != 1334:
        raise ValueError('Final snapshot must contain all 1334 turns')
    counts = Counter(row['game'] for _, row in rows)
    panel = read_json(ROOT / 'ARC3-Inference/experiments/teacher-reasoning/progressive-sol25-monitoring.json')
    games = [{'game': g['game'], 'finalized': counts[g['game']], 'total': g['responses'],
              'complete': counts[g['game']] == g['responses']} for g in panel['games']]
    audit_summary = summarize_audits(rows)
    summary = {'kind': kind, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'finalized': len(rows), 'training_eligible': sum(r['training_eligible'] for _, r in rows),
               'excluded_keys': [r['key'] for _, r in rows if not r['training_eligible']],
               'games_complete': sum(g['complete'] for g in games), 'games': games,
               'audits': {k: v for k, v in audit_summary.items() if k != 'exceptions'},
               'monitoring_exceptions': audit_summary['exceptions'],
               'source_dvc': 'ARC3-Inference/runs/gpt61sol-features-25games.dvc',
               'notes': ['Only fully finalized turns and their immutable API journals are archived.',
                         'In-flight turn journals remain in the live run and are excluded from this release.',
                         'Oversized targets are excluded from SFT but their thinking remains in retained history.',
                         'Audit flags are retained for review; they do not automatically drop training targets.',
                         'SFT is rebuilt from this snapshot with the pinned student template/tokenizer.']}
    atomic_json(staging / 'snapshot.json', summary)
    backup = RELEASE.with_name(RELEASE.name + '.previous')
    if backup.exists():
        shutil.rmtree(backup)
    if RELEASE.exists():
        RELEASE.rename(backup)
    staging.rename(RELEASE)
    if backup.exists():
        shutil.rmtree(backup)
    atomic_json(RESULTS, summary)
    return summary


def command(argv, *, capture=False):
    result = subprocess.run(argv, cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def verify_snapshot_pin(job):
    if file_hash(RELEASE / 'snapshot.json') != job['snapshot_hash']:
        raise RuntimeError('Release snapshot changed after checkpointing publication')
    if job.get('pointer_hash') and file_hash(ROOT / POINTER) != job['pointer_hash']:
        raise RuntimeError('DVC pointer changed after checkpointing publication')


def committed_pointer_hash(commit):
    result = subprocess.run(['git', 'show', f'{commit}:{POINTER}'], cwd=ROOT,
                            check=True, stdout=subprocess.PIPE)
    return hashlib.sha256(result.stdout).hexdigest()


def publish(kind, state, save):
    if command(['git', 'branch', '--show-current'], capture=True) != BRANCH:
        raise RuntimeError('Release branch changed; refuse to publish from another branch')
    job = state.setdefault(kind, {})
    if not job.get('snapshot'):
        if command(['git', 'rev-parse', 'HEAD'], capture=True) != state['expected_head']:
            raise RuntimeError('Branch HEAD changed before publication; inspect before resuming')
        job['base_commit'] = state['expected_head']
        job['snapshot'] = snapshot(kind)
        job['snapshot_hash'] = file_hash(RELEASE / 'snapshot.json')
        save()
    verify_snapshot_pin(job)
    summary = job['snapshot']
    if not job.get('dvc_pushed'):
        env = dict(os.environ, TMPDIR='/tmp', DVC_SITE_CACHE_DIR='/tmp/think-dvc-site',
                   UV_CACHE_DIR='/tmp/think-preflight-uv', UV_TOOL_DIR='/tmp/think-gen-tools',
                   UV_TOOL_BIN_DIR='/tmp/think-gen-tools/bin', DVC_NO_ANALYTICS='true')
        dvc = ['uv', 'tool', 'run', '--from', 'dvc[s3]==3.67.1', 'dvc']
        subprocess.run(dvc + ['add', str(RELEASE.relative_to(ROOT))], cwd=ROOT, env=env, check=True)
        pointer_hash = file_hash(ROOT / POINTER)
        if job.get('pointer_hash') and job['pointer_hash'] != pointer_hash:
            raise RuntimeError('DVC output differs from the saved snapshot pointer')
        job['pointer_hash'] = pointer_hash
        save()
        subprocess.run(dvc + ['push', POINTER], cwd=ROOT, env=env, check=True)
        job['dvc_pushed'] = True
        save()
    title = f'think_gen: {kind} Sol25 dataset ({summary["finalized"]} turns)'
    if not job.get('commit'):
        verify_snapshot_pin(job)
        head = command(['git', 'rev-parse', 'HEAD'], capture=True)
        if head != job['base_commit']:
            # Recover only our exact commit if it landed before its state save.
            if (command(['git', 'log', '-1', '--format=%s'], capture=True) != title or
                    command(['git', 'rev-parse', 'HEAD^'], capture=True) != job['base_commit'] or
                    committed_pointer_hash(head) != job['pointer_hash']):
                raise RuntimeError('Branch HEAD changed; refuse to publish another commit')
        else:
            command(['git', 'add', '--', *FILES])
            staged = command(['git', 'diff', '--cached', '--name-only'], capture=True).splitlines()
            unexpected = set(staged) - set(FILES) - {'ARC3-Inference/runs/.gitignore'}
            if unexpected:
                raise RuntimeError(f'Refuse to commit unrelated staged files: {sorted(unexpected)}')
            if not staged:
                raise RuntimeError('No staged release changes')
            command(['git', 'commit', '-m', title])
        job['commit'] = command(['git', 'rev-parse', 'HEAD'], capture=True)
        state['expected_head'] = job['commit']
        save()
    if command(['git', 'rev-parse', 'HEAD'], capture=True) != job['commit']:
        raise RuntimeError('Branch HEAD differs from the recorded release commit')
    verify_snapshot_pin(job)
    if not job.get('git_pushed'):
        command(['git', 'push', 'origin', f'{job["commit"]}:refs/heads/{BRANCH}'])
        job['git_pushed'] = True
        save()
    remote = command(['git', 'ls-remote', 'origin', f'refs/heads/{BRANCH}'], capture=True)
    if not remote or remote.split()[0] != job['commit']:
        raise RuntimeError('Remote branch differs from the recorded release commit')
    audit = summary['audits']
    requested_audits = audit.get('requested', audit.get('sampled', 0))
    available_audits = audit.get('available', audit.get('sampled', 0))
    body = LIVE / 'release-pr-body.md'
    body.write_text(
        'Generated thinking now progresses through all 25 Sol games, retaining each turn’s final '
        'thinking through later turns and compaction. A resumable two-loop pipeline runs games '
        'concurrently, uses explicit Sol history caching, and exports targets within the 120K input limit.\n\n'
        f'Dataset snapshot: **{summary["finalized"]}/1,334 turns**, '
        f'**{summary["games_complete"]}/25 complete games**, '
        f'{summary["training_eligible"]} eligible targets, {len(summary["excluded_keys"])} oversized targets. '
        f'The DVC pointer is `{POINTER}`; the partial commit remains available in Git history after final publication.\n\n'
        f'Quality monitoring: {requested_audits} sampled turns, {available_audits} audits available, '
        f'{audit["clean"]} clean and '
        f'{audit["flagged"]} flagged. Flags and raw verdicts are retained for review; '
        'judge equivalence checks do not execute code. See the results JSON and run document for details.\n\n'
        'Validation: independent setup, cache, concurrency and release reviews; pinned-tokenizer SFT export; '
        'all retained-final-thinking hashes checked while snapshotting; immutable checkpoints preserved during '
        'worker ramps; relevant offline tests passed. Generation continues independently of snapshot publication.\n')
    if not state.get('pr_url'):
        existing = json.loads(command(['gh', 'pr', 'list', '--head', BRANCH, '--state', 'open', '--json', 'url'], capture=True))
        state['pr_url'] = existing[0]['url'] if existing else command(
            ['gh', 'pr', 'create', '--draft', '--base', 'main', '--head', BRANCH,
             '--title', 'Generate progressive thinking for all 25 Sol games', '--body-file', str(body)], capture=True)
        save()
    else:
        command(['gh', 'pr', 'edit', state['pr_url'], '--body-file', str(body)])
    job['published'] = True
    job['published_utc'] = datetime.now(timezone.utc).isoformat()
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only', choices=['partial', 'final'])
    parser.add_argument('--threshold', type=int, default=13)
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.threshold <= 25 or args.interval < 1:
        parser.error('Invalid threshold or polling interval')
    with (LIVE / '.release.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.snapshot_only:
            print(json.dumps(snapshot(args.snapshot_only)))
            return
        path = LIVE / 'release-status.json'
        state = read_json(path) if path.exists() else {}
        state.setdefault('expected_head', command(['git', 'rev-parse', 'HEAD'], capture=True))
        state.update(watcher_pid=os.getpid(), threshold_games=args.threshold, branch=BRANCH)
        def save():
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, state)
        panel = read_json(ROOT / 'ARC3-Inference/experiments/teacher-reasoning/progressive-sol25-monitoring.json')
        while True:
            try:
                rows = completed_rows(LIVE)
                counts = Counter(r['game'] for _, r in rows)
                done = sum(counts[g['game']] == g['responses'] for g in panel['games'])
                state.update(finalized=len(rows), games_complete=done, error=None)
                save()
                if done >= args.threshold and not state.get('partial', {}).get('published'):
                    publish('partial', state, save)
                if done == 25 and read_json(LIVE / 'driver_status.json')['phase'] == 'completed':
                    publish('final', state, save)
                    state['phase'] = 'completed'
                    save()
                    return
                state['phase'] = 'waiting_for_final' if state.get('partial', {}).get('published') else 'waiting_for_majority'
                save()
            except Exception as e:
                state.update(phase='error', error=f'{type(e).__name__}: {e}')
                save()
                print(state['error'], flush=True)
                raise SystemExit(1)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
