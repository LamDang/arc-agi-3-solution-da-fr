"""Verify an incomplete capacity capture without claiming a numerical pass."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    job = Path(parser.parse_args().attempt).resolve()
    out = job / 'output'
    config, monitor = read(job / 'config.json'), read(job / 'monitor.json')
    provenance = read(out / 'provenance.json')
    assert monitor['returncode'] == -9 and not monitor['timed_out']
    assert provenance['mode'] == 'benchmark' and provenance['optimizer_updates'] == 0
    assert not provenance['pushed_to_remote']
    assert not (out / 'result.json').exists()
    assert not (out / 'gradients').exists() and not list(out.glob('gradients*.pt'))
    assert not (out / 'initial-adapter').exists()
    assert config['prompt_tokens'] is None and config['benchmark_repeats'] == 2
    for key in ['expert_chunking', 'qsa_chunking', 'hyperconnection_chunking', 'ple_chunking']:
        assert config['optimizations'][key]
    assert config['optimizations']['chunk_tokens'] == 8192
    initial = read(out / 'initial-comparison.json')
    assert initial['passed'] and initial['bitwise_equal_tensors'] == 74472
    for name, digest in read(job / 'source-hashes.json').items():
        assert sha(job / 'source' / name) == digest
    for name, digest in provenance['source_hashes'].items():
        assert sha(out / 'sources' / name) == digest
    for name, digest in config['expected_sha256'].items():
        actual = (provenance['sample_hashes'][config['samples'][0]] if name == 'sample'
                  else provenance['model_identity'][name])
        assert actual == digest
    for archived in provenance['archived_inputs'].values():
        assert sha(out / archived['path']) == archived['sha256']
    fixture = next(row for row in read(ROOT / 'results/20261010-benchmark-fixtures/manifest.json')['rows']
                   if row['tokens'] == config['benchmark_tokens'])
    assert fixture['sha256'] == config['expected_sha256']['sample']
    resources = read(out / 'resources.json')
    assert [row['phase'] for row in resources] == ['loading', 'initialization_verification']
    progress = read(out / 'resource-progress.json')
    assert progress['phase'] == 'forward-0' and not progress['completed']
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    assert [row['event'] for row in events] == ['load_start', 'load_complete', 'benchmark_repeat_start']
    assert events[-1]['tokens'] == fixture['tokens'] and events[-1]['targets'] == fixture['targets']
    preparations = [json.loads(line) for line in (out / 'ple-events.jsonl').read_text().splitlines()]
    assert len(preparations) == 1
    prep = preparations[0]
    assert prep['tokens'] == fixture['tokens'] and prep['lookup_bytes'] == fixture['tokens'] * 2560 * 2
    assert prep['disk_read_passes'] == 1 and 0 < prep['unique_rows'] <= prep['total_row_ids']
    prep['seconds'] = (prep['finished_monotonic_ns'] - prep['started_monotonic_ns']) / 1e9
    assert prep['seconds'] > 0
    storage = read(job / 'ple-storage-inspection.json')
    counters = dict(line.split(': ', 1) for line in storage['worker_io_snapshot'].splitlines())
    storage['worker_cumulative_read_bytes'] = int(counters['read_bytes'])
    cgroup = read(job / 'cgroup-after-failure.json')
    profile = read(out / 'numeric-profile.json')
    # The first backward never happened. Do not use the successful-run checker.
    assert not profile['backward_verified'] and profile['observed_configuration'] is None
    report = dict(
        failure_evidence_checks_passed=True, completed=False, numerical_gate_passed=None,
        attempt=job.name, tokens=config['benchmark_tokens'], fixture=fixture,
        execution_commit=config['dispatch_commit'], monitor=monitor,
        resources=resources, incomplete_forward=progress, events=events,
        loss=None, gradients=None, cuda_forward_backward_peak_bytes=None,
        optimizer_updates=0, raw_gradients_retained=False, pushed_to_remote=False,
        numerical_profile=profile, numerical_settings=provenance['numerical_settings'],
        ple_preparation=prep, ple_storage_inspection=storage,
        server_disk_inventory=read(job / 'server-disk-inventory.json'),
        server_block_details=read(job / 'server-block-details.json'),
        cgroup_after_failure=cgroup,
        failure_interpretation='SIGKILL with strong host-memory-exhaustion evidence.',
        attribution_limits='No pre-run cgroup counters; oom_kill=1 with max=0/oom=0 does not prove '
                           'this process incremented the counter or a memcg-limit-triggered kill. '
                           'Incomplete forward samples are lower observations, not completed phase peaks.',
    )
    (ROOT / 'reports' / f"benchmark-{config['benchmark_tokens']}-failure.json").write_text(
        json.dumps(report, indent=2) + '\n')
    inventory = {str(path.relative_to(job)): dict(bytes=path.stat().st_size, sha256=sha(path))
                 for path in sorted(job.rglob('*')) if path.is_file() and path.name != 'file-hashes.json'}
    (job / 'file-hashes.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(attempt=job.name, failure_evidence_checks_passed=True,
                          inventoried_files=len(inventory), completed=False)))


if __name__ == '__main__':
    main()
