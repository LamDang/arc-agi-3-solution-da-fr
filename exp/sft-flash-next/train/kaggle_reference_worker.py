"""Remote, observation-only launcher for one pinned native reference capture.

The native reference source and tensor arithmetic are never edited. This worker
wraps Tensor.backward only to mark its start/end and samples process resources.
The output directory is left absent until the native script creates it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def validate(config):
    required = ('run_id', 'remote_output', 'remote_launch', 'model', 'sample',
                'adapter', 'bootstrap', 'reference_script', 'prompt_tokens',
                'expected_sha256', 'environment', 'timeout_seconds')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Missing config fields: {missing}')
    if not config['run_id'].replace('-', '').replace('_', '').isalnum():
        raise ValueError('run_id must be alphanumeric with hyphens/underscores')
    if int(config['timeout_seconds']) != 1200:
        raise ValueError('Native capture timeout must be 1200 seconds')
    if config.get('reference_seed') != 20261009:
        raise ValueError('Native reference seed must match pinned source')
    if config['environment'] != {
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    }:
        raise ValueError('Deterministic environment differs from pinned reference')
    if Path(config['remote_output']) == Path(config['remote_launch']):
        raise ValueError('Output and launcher directories must differ')
    paths = {
        'sample': config['sample'], 'adapter': config['adapter'],
        'bootstrap': config['bootstrap'], 'reference_script': config['reference_script'],
        'model_config': str(Path(config['model']) / 'config.json'),
    }
    for key, path in paths.items():
        if sha256(path) != config['expected_sha256'][key]:
            raise ValueError(f'Pinned input SHA256 mismatch: {key}')
    return paths


def native_command(config, instrumented_bootstrap):
    return ['/usr/bin/python3', str(instrumented_bootstrap), '--model', config['model'],
            '--sample', config['sample'], '--prompt-tokens', str(config['prompt_tokens']),
            '--adapter-state', config['adapter'], '--checkpointing', '--save-on-cpu',
            '--first-pass-only', '--deterministic', '--gradient-repeats', '1',
            '--out', config['remote_output']]


def bootstrap_source(config):
    """Apply the original bootstrap's package view, then time backward only."""
    source = '''import builtins, hashlib, json, os, runpy, sys, time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path = [p for p in sys.path if p != '/usr/local/lib/python3.13/dist-packages']
sys.path[:0] = ['/tmp/peft-autoround-compat/package',
                '/tmp/peft-autoround-compat/site-without-torchao',
                '/kaggle/working/training-gradient-audit/exp/sft-flash-next/train']
import torch
timing_path = Path(%(timing_path)r)
allocator_path = Path(%(allocator_path)r)
native_events = {'load_start', 'load_complete', 'forward_start', 'loss',
                 'backward_start', 'gradients_saved'}
original_print = builtins.print
def allocator_peak(phase, reset=True):
    torch.cuda.synchronize()
    row = {'phase': phase, 'monotonic_ns': time.monotonic_ns(),
           'exact_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'exact_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    with allocator_path.open('a') as stream:
        stream.write(json.dumps(row)+'\\n')
    if reset:
        torch.cuda.reset_peak_memory_stats()
def observed_print(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str) and args[0].startswith('{'):
        try:
            event = json.loads(args[0]).get('event')
        except (ValueError, TypeError):
            event = None
        if event in native_events:
            if event == 'load_complete': allocator_peak('loading')
            elif event == 'forward_start': allocator_peak('forward_setup')
            elif event == 'loss': allocator_peak('forward')
            elif event == 'backward_start': allocator_peak('pre_backward_export')
            elif event == 'gradients_saved': allocator_peak('gradient_export', reset=False)
            with timing_path.open('a') as stream:
                stream.write(json.dumps({'event': event,
                                         'monotonic_ns': time.monotonic_ns()})+'\\n')
    return original_print(*args, **kwargs)
builtins.print = observed_print
original_backward = torch.Tensor.backward
def observed_backward(self, *args, **kwargs):
    torch.cuda.synchronize()
    with timing_path.open('a') as stream:
        stream.write(json.dumps({'event':'backward_call_start',
                                 'monotonic_ns':time.monotonic_ns()})+'\\n')
    try:
        return original_backward(self, *args, **kwargs)
    finally:
        allocator_peak('pure_backward')
        with timing_path.open('a') as stream:
            stream.write(json.dumps({'event':'backward_call_end',
                                     'monotonic_ns':time.monotonic_ns()})+'\\n')
torch.Tensor.backward = observed_backward
try:
    runpy.run_path(%(bootstrap)r, run_name='__main__')
finally:
    imported = {}
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if not value or not value.endswith('.py'):
            continue
        if not (value.startswith('/kaggle/working/training-gradient-audit/') or
                value.startswith('/tmp/peft-autoround-compat/')):
            continue
        p = Path(value)
        if p.is_file():
            with p.open('rb') as stream:
                imported[str(p)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    Path(%(source_hash_path)r).write_text(json.dumps(imported, indent=2)+'\\n')
'''
    return source % {'timing_path': str(Path(config['remote_launch']) / 'timing-events.jsonl'),
                     'allocator_path': str(Path(config['remote_launch']) / 'allocator-phase-peaks.jsonl'),
                     'bootstrap': config['bootstrap'],
                     'source_hash_path': str(Path(config['remote_launch']) / 'imported-source-hashes.json')}


def event_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def phase_for(timing, monotonic_ns):
    transitions = {'load_start': 'loading', 'load_complete': 'forward_setup',
                   'forward_start': 'forward',
                   'loss': 'pre_backward_export', 'backward_start': 'pure_backward',
                   'backward_call_end': 'gradient_export', 'gradients_saved': 'finished'}
    phase = 'startup'
    for row in timing:
        if row['monotonic_ns'] <= monotonic_ns and row['event'] in transitions:
            phase = transitions[row['event']]
    return phase


def summarize(timing, telemetry, allocator, total_seconds):
    by_timing = {row['event']: row['monotonic_ns'] for row in timing}
    def duration(first, last):
        return ((by_timing[last] - by_timing[first]) / 1e9
                if first in by_timing and last in by_timing else None)
    phases = {}
    for row in telemetry:
        phase = row['phase']
        item = phases.setdefault(phase, {'samples': 0, 'peak_gpu_used_mib': 0,
                                         'peak_process_rss_bytes': 0, 'peak_host_used_bytes': 0})
        item['samples'] += 1
        item['peak_gpu_used_mib'] = max(item['peak_gpu_used_mib'], row['gpu_used_mib'])
        item['peak_process_rss_bytes'] = max(item['peak_process_rss_bytes'], row['process_rss_bytes'])
        item['peak_host_used_bytes'] = max(item['peak_host_used_bytes'], row['host_used_bytes'])
    for row in allocator:
        phases.setdefault(row['phase'], {'samples': 0, 'peak_gpu_used_mib': 0,
                                         'peak_process_rss_bytes': 0,
                                         'peak_host_used_bytes': 0}).update({
            'exact_cuda_peak_allocated_bytes': row['exact_peak_allocated_bytes'],
            'exact_cuda_peak_reserved_bytes': row['exact_peak_reserved_bytes']})
    return {'loading_seconds': duration('load_start', 'load_complete'),
            'forward_setup_seconds': duration('load_complete', 'forward_start'),
            'forward_seconds': duration('forward_start', 'loss'),
            'pre_backward_export_seconds': duration('loss', 'backward_start'),
            'pure_backward_seconds': duration('backward_call_start', 'backward_call_end'),
            'gradient_export_seconds': duration('backward_call_end', 'gradients_saved'),
            'total_seconds': total_seconds, 'phase_resource_samples': phases,
            'timing_method': 'aligned monotonic event marks and observation-only Tensor.backward wrapper; synchronized CUDA at phase boundaries; allocator peaks reset per phase; RSS/device-used sampled every 2 seconds'}


def launch(config_path):
    config = json.loads(Path(config_path).read_text())
    paths = validate(config)
    output = Path(config['remote_output'])
    launch_dir = Path(config['remote_launch'])
    if output.exists() or launch_dir.exists():
        raise FileExistsError('Run output or launch directory already exists; refusing duplicate')
    active = subprocess.run(['pgrep', '-af', 'reference-one-run-bootstrap|instrumented-bootstrap'],
                            capture_output=True, text=True).stdout.strip()
    if active:
        raise RuntimeError('Native reference process already running')
    gpu_processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if gpu_processes:
        raise RuntimeError('GPU compute processes already running; refusing concurrent capture')
    launch_dir.mkdir(parents=True)
    instrumented = launch_dir / 'instrumented-bootstrap.py'
    instrumented.write_text(bootstrap_source(config))
    command = native_command(config, instrumented)
    launch_record = {'run_id': config['run_id'], 'command': command,
                     'environment': config['environment'],
                     'expected_sha256': config['expected_sha256'],
                     'validated_paths': paths, 'started_utc': timestamp(),
                     'instrumented_bootstrap_sha256': sha256(instrumented),
                     'worker_sha256': sha256(__file__),
                     'reference_script_commit': config['reference_script_commit'],
                     'no_optimizer_update': True, 'no_clipping': True}
    (launch_dir / 'launch.json').write_text(json.dumps(launch_record, indent=2) + '\n')
    (launch_dir / 'reference.py').write_bytes(Path(config['reference_script']).read_bytes())
    (launch_dir / 'worker.py').write_bytes(Path(__file__).read_bytes())
    for label, source in [('sample.pt', config['sample']), ('bootstrap.py', config['bootstrap']),
                          ('model-config.json', Path(config['model']) / 'config.json')]:
        (launch_dir / label).write_bytes(Path(source).read_bytes())
    with (launch_dir / 'launcher.log').open('w') as log:
        process = subprocess.Popen(['/usr/bin/python3', str(launch_dir / 'worker.py'),
                                    'monitor', str(config_path)], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (launch_dir / 'monitor-pid').write_text(f'{process.pid}\n')
    print(json.dumps({'run_id': config['run_id'], 'monitor_pid': process.pid,
                      'output_absent_at_launch': not output.exists()}))


def monitor(config_path):
    import psutil

    config = json.loads(Path(config_path).read_text())
    launch_dir = Path(config['remote_launch'])
    output = Path(config['remote_output'])
    command = json.loads((launch_dir / 'launch.json').read_text())['command']
    environment = os.environ.copy()
    environment.update(config['environment'])
    start = time.monotonic()
    with (launch_dir / 'process.log').open('w') as log, (launch_dir / 'telemetry.jsonl').open('w') as telemetry:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment)
        (launch_dir / 'child-pid').write_text(f'{process.pid}\n')
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed >= config['timeout_seconds']:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                rss = psutil.Process(process.pid).memory_info().rss
            except psutil.Error:
                rss = 0
            host = psutil.virtual_memory().used
            gpu = utilization = 0
            try:
                reply = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], capture_output=True,
                                       text=True, timeout=5)
                gpu, utilization = [int(x.strip()) for x in reply.stdout.splitlines()[0].split(',')]
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            timing = event_rows(launch_dir / 'timing-events.jsonl')
            phase = phase_for(timing, time.monotonic_ns())
            row = {'elapsed_seconds': elapsed, 'phase': phase,
                   'process_rss_bytes': rss, 'host_used_bytes': host,
                   'gpu_used_mib': gpu, 'gpu_utilization_percent': utilization}
            telemetry.write(json.dumps(row) + '\n')
            telemetry.flush()
            time.sleep(2)
    total = time.monotonic() - start
    result = {'run_id': config['run_id'], 'pid': process.pid,
              'returncode': process.wait(), 'timed_out': timed_out,
              'elapsed_seconds': total, 'finished_utc': timestamp()}
    timing = event_rows(launch_dir / 'timing-events.jsonl')
    telemetry_rows = event_rows(launch_dir / 'telemetry.jsonl')
    allocator = event_rows(launch_dir / 'allocator-phase-peaks.jsonl')
    (launch_dir / 'timing-summary.json').write_text(
        json.dumps(summarize(timing, telemetry_rows, allocator, total), indent=2) + '\n')
    (launch_dir / 'monitor.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'launch', 'monitor'])
    parser.add_argument('config')
    args = parser.parse_args()
    if args.action == 'validate':
        config = json.loads(Path(args.config).read_text())
        print(json.dumps({'preflight': 'ok', 'run_id': config['run_id'],
                          'validated_paths': validate(config)}))
    elif args.action == 'launch':
        launch(args.config)
    else:
        monitor(args.config)


if __name__ == '__main__':
    main()
