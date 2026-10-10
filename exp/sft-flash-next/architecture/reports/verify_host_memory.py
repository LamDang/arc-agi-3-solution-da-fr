"""Verify the load-only RAM diagnostic and inventory its artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    job = Path(parser.parse_args().attempt).resolve()
    out = job / 'output'
    config, provenance = read(job / 'config.json'), read(out / 'provenance.json')
    result, monitor = read(out / 'result.json'), read(job / 'monitor.json')
    assert monitor['returncode'] == 0 and not monitor['timed_out']
    assert result['completed'] and result['mode'] == 'diagnostic-host-memory'
    assert not result['forward_executed'] and not result['backward_executed']
    assert not result['ple_prepared'] and not result['raw_gradients_retained']
    assert result['optimizer_updates'] == 0 and result['diagnostic_intervention_only']
    assert not (out / 'gradients').exists() and not list(out.glob('gradients*.pt'))
    initial = read(out / 'initial-comparison.json')
    assert initial['passed'] and initial['bitwise_equal_tensors'] == 74472
    for name, digest in read(job / 'source-hashes.json').items():
        assert sha(job / 'source' / name) == digest
    for name, digest in provenance['source_hashes'].items():
        assert sha(out / 'sources' / name) == digest
    for archived in provenance['archived_inputs'].values():
        assert sha(out / archived['path']) == archived['sha256']
    for name, digest in config['expected_sha256'].items():
        actual = provenance['sample_hashes'][config['samples'][0]] if name == 'sample' else provenance['model_identity'][name]
        assert actual == digest
    observations = read(out / 'host-memory.json')
    assert [row['label'] for row in observations] == [
        'before_loading', 'after_loading', 'after_python_gc',
        'after_initialization_verification', 'after_diagnostic_malloc_trim']
    report = dict(evidence_checks_passed=True, attempt=job.name,
        execution_commit=config['dispatch_commit'], result=result, monitor=monitor,
        resources=read(out / 'resources.json'), observations=observations,
        scope='Model loading and initialization only. No forward/backward or PLE lookup.',
        cgroup_peak_scope='Lifetime peak; not reset for this diagnostic.',
        malloc_trim_scope='Diagnostic-only return of unused glibc arena pages; not installed in training.',
        pushed_to_remote=False)
    (ROOT / 'reports/host-memory-diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    inventory = {str(path.relative_to(job)): dict(bytes=path.stat().st_size, sha256=sha(path))
                 for path in sorted(job.rglob('*')) if path.is_file() and path.name != 'file-hashes.json'}
    (job / 'file-hashes.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(attempt=job.name, evidence_checks_passed=True, inventoried_files=len(inventory))))


if __name__ == '__main__':
    main()
