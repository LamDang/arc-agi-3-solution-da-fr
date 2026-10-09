"""Live publication must exclude in-flight work and preserve history dependencies."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/manage_progressive_sol25_release.py'
spec = importlib.util.spec_from_file_location('progressive_release', SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def row(index, thinking, prior=None):
    prior = prior or {}
    return {'key': f'g#{index}', 'game': 'g', 'ref': f'ref{index}',
            'thinking': thinking, 'history_refs': prior,
            'history_hash': release.digest(prior)}


def test_snapshot_omits_inflight_turn_and_keeps_finalized_journal(tmp_path):
    live = tmp_path / 'live'
    final = live / 'turns/g/00000/final.json'
    save(final, row(0, 'final thinking'))
    save(final.parent / 'calls/draft/attempt-00.json', {'state': 'response'})
    save(live / 'turns/g/00001/calls/draft/attempt-00.json', {'state': 'pending'})
    target = tmp_path / 'snapshot'
    release.copy_finalized(live, target, release.completed_rows(live))
    assert (target / 'turns/g/00000/calls/draft/attempt-00.json').exists()
    assert not (target / 'turns/g/00001').exists()


def test_snapshot_requires_exact_final_thinking_in_retained_history(tmp_path):
    first = row(0, 'corrected final thinking')
    second = row(1, 'later thinking', {'ref0': release.digest('initial draft')})
    with pytest.raises(ValueError, match='Missing/changed retained thinking'):
        release.verify_history([(tmp_path / '0', first), (tmp_path / '1', second)])
    second['history_refs']['ref0'] = release.digest(first['thinking'])
    second['history_hash'] = release.digest(second['history_refs'])
    release.verify_history([(tmp_path / '0', first), (tmp_path / '1', second)])


def test_finalized_turn_cannot_contain_pending_api_attempt(tmp_path):
    live = tmp_path / 'live'
    final = live / 'turns/g/00000/final.json'
    save(final, row(0, 'thinking'))
    save(final.parent / 'calls/draft/attempt-00.json', {'state': 'pending'})
    with pytest.raises(ValueError, match='Unfinished request'):
        release.copy_finalized(live, tmp_path / 'snapshot', release.completed_rows(live))


def test_snapshot_allows_terminal_uncertain_monitoring_but_not_generation(tmp_path):
    live = tmp_path / 'live'
    final = live / 'turns/g/00000/final.json'
    save(final, row(0, 'thinking'))
    save(final.parent / 'calls/final_audit/attempt-00.json', {'state': 'monitoring_uncertain'})
    release.copy_finalized(live, tmp_path / 'snapshot', release.completed_rows(live))
    save(final.parent / 'calls/draft/attempt-00.json', {'state': 'monitoring_uncertain'})
    with pytest.raises(ValueError, match='Uncertain non-monitoring request'):
        release.copy_finalized(live, tmp_path / 'snapshot', release.completed_rows(live))


def test_audit_summary_counts_clean_and_unavailable_samples():
    clean_verdict = {
        'words': {'covered': [True], 'contradictions': []},
        'code': {'leads_to_call': True, 'disagreements': []},
        'fact': {'grounded': True, 'errors': []},
    }
    clean = {'key': 'g#0', 'monitoring_sample': True, 'final_judge': clean_verdict}
    unavailable = {'key': 'g#1', 'monitoring_sample': True, 'final_judge': None,
                   'monitoring_exception': ['regen unavailable after HTTP 400']}
    audit_missing = {'key': 'g#2', 'monitoring_sample': True, 'final_judge': None,
                     'regenerated_code': 'print(1)',
                     'monitoring_exception': ['final_audit unavailable after HTTP 503']}
    summary = release.summarize_audits([(Path('0'), clean), (Path('1'), unavailable),
                                        (Path('2'), audit_missing)])
    assert summary['requested'] == 3
    assert summary['available'] == 1 and summary['unavailable'] == 2
    assert summary['equivalence_unavailable'] == 2
    assert summary['clean'] == 1 and summary['flagged'] == 0


def test_publication_rejects_replaced_snapshot_or_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'RELEASE', tmp_path / 'snapshot')
    monkeypatch.setattr(release, 'POINTER', 'dataset.dvc')
    save(tmp_path / 'snapshot/snapshot.json', {'finalized': 20})
    pointer = tmp_path / 'dataset.dvc'
    pointer.write_text('outs: []\n')
    job = {'snapshot_hash': release.file_hash(tmp_path / 'snapshot/snapshot.json'),
           'pointer_hash': release.file_hash(pointer)}
    release.verify_snapshot_pin(job)
    pointer.write_text('outs: changed\n')
    with pytest.raises(RuntimeError, match='DVC pointer changed'):
        release.verify_snapshot_pin(job)
    pointer.write_text('outs: []\n')
    save(tmp_path / 'snapshot/snapshot.json', {'finalized': 21})
    with pytest.raises(RuntimeError, match='Release snapshot changed'):
        release.verify_snapshot_pin(job)


def publication_job(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'LIVE', tmp_path)
    monkeypatch.setattr(release, 'RELEASE', tmp_path / 'snapshot')
    monkeypatch.setattr(release, 'POINTER', 'dataset.dvc')
    summary = {'finalized': 20, 'games_complete': 13, 'training_eligible': 19,
               'excluded_keys': ['g#1'], 'audits': {'sampled': 5, 'clean': 4, 'flagged': 1}}
    save(tmp_path / 'snapshot/snapshot.json', summary)
    (tmp_path / 'dataset.dvc').write_text('outs: []\n')
    return {'snapshot': summary, 'base_commit': 'base', 'dvc_pushed': True,
            'snapshot_hash': release.file_hash(tmp_path / 'snapshot/snapshot.json'),
            'pointer_hash': release.file_hash(tmp_path / 'dataset.dvc')}


def test_publisher_recovers_its_commit_and_pushes_exact_recorded_sha(tmp_path, monkeypatch):
    job = publication_job(tmp_path, monkeypatch)
    state = {'expected_head': 'base', 'partial': job}
    calls = []
    def command(argv, capture=False):
        calls.append(argv)
        if argv == ['git', 'branch', '--show-current']:
            return release.BRANCH
        if argv == ['git', 'rev-parse', 'HEAD']:
            return 'recorded-sha'
        if argv == ['git', 'rev-parse', 'HEAD^']:
            return 'base'
        if argv == ['git', 'log', '-1', '--format=%s']:
            return 'think_gen: partial Sol25 dataset (20 turns)'
        if argv[:2] == ['git', 'ls-remote']:
            return f'recorded-sha\trefs/heads/{release.BRANCH}'
        if argv[:3] == ['gh', 'pr', 'list']:
            return '[{"url":"https://github.com/test/repo/pull/1"}]'
    monkeypatch.setattr(release, 'command', command)
    monkeypatch.setattr(release, 'committed_pointer_hash', lambda _: job['pointer_hash'])
    release.publish('partial', state, lambda: None)
    assert job['commit'] == 'recorded-sha'
    assert ['git', 'push', 'origin', f'recorded-sha:refs/heads/{release.BRANCH}'] in calls
    assert not any(argv[:2] in (['git', 'add'], ['git', 'commit']) for argv in calls)
    assert job['published']


def test_publisher_rejects_another_commit_after_saved_commit(tmp_path, monkeypatch):
    job = publication_job(tmp_path, monkeypatch)
    job['commit'] = 'recorded-sha'
    calls = []
    def command(argv, capture=False):
        calls.append(argv)
        return release.BRANCH if argv == ['git', 'branch', '--show-current'] else 'another-sha'
    monkeypatch.setattr(release, 'command', command)
    with pytest.raises(RuntimeError, match='HEAD differs'):
        release.publish('partial', {'partial': job}, lambda: None)
    assert not any(argv[:2] == ['git', 'push'] for argv in calls)
