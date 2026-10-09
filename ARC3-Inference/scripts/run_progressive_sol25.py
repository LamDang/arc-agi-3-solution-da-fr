"""Launch the approved job: finish pilot, verify resume, then all 25 games.

Run from any directory using the repository venv. Output remains in the
sampled pilot's directory so its paid checkpoints are reused. No automatic
restart on non-429 failures; the runner itself retries429s indefinitely.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen.progressive import atomic_json, file_hash, read_json

OUT = ROOT / 'ARC3-Inference/runs/think-progressive-sampled-pilot'
BASE = [sys.executable, '-m', 'think_gen.progressive',
        '--run', str(ROOT/'ARC3-Inference/runs/gpt61sol-features-25games'),
        '--out', str(OUT),
        '--tokenizer', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/tokenizer.json'),
        '--template', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/chat_template.jinja'),
        '--monitor-per-game', '15']


def status(phase, **extra):
    atomic_json(OUT/'driver_status.json', {'phase': phase, 'updated': time.time(),
                'driver_pid': os.getpid(), **extra})
    print(phase, json.dumps(extra), flush=True)


def run(phase, options):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'ARC3-Inference')
    with (OUT/'runner.log').open('a', buffering=1) as log:
        child = subprocess.Popen(BASE+options, cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, options=options)
        code = child.wait()
    summary = read_json(OUT/'summary.json')
    atomic_json(OUT/f'{phase}-summary.json', summary)
    if code:
        raise RuntimeError(f'{phase} stopped with exit {code}; inspect runner.log and summary.json')
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.driver.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            ready = OUT/'pilot-validated.json'
            if not ready.exists():
                options = ['--games', 'sk48', '--limit', '2', '--workers', '1',
                           '--budget', '8', '--max-attempts', '11']
                summary = run('pilot', options)
                if summary['completed'] != 2 or summary['monitored'] != 1:
                    raise RuntimeError('Pilot did not finish sampled and unsampled turns')
                finals = sorted((OUT/'turns/sk48-d8078629_p0').glob('*/final.json'))[:2]
                rows = [read_json(p) for p in finals]
                if [r['monitoring_sample'] for r in rows] != [True, False]:
                    raise RuntimeError('Pilot monitoring selection mismatch')
                first, second = rows
                from think_gen.progressive import digest
                if second['history_refs'].get(first['ref']) != digest(first['thinking']):
                    raise RuntimeError('Pilot did not preserve finalized thinking')
                if first['final_judge'] is None or first['regenerated_code'] is None:
                    raise RuntimeError('Sampled turn missing final monitoring')
                if second['final_judge'] is not None or second['regenerated_code'] is not None:
                    raise RuntimeError('Unsampled turn unexpectedly monitored')
                second_folder = OUT/'turns/sk48-d8078629_p0/00001/calls'
                if (second_folder/'regen').exists() or (second_folder/'final_audit').exists():
                    raise RuntimeError('Unsampled turn performed monitoring calls')
                before = {str(p.relative_to(OUT)):file_hash(p) for p in OUT.glob('turns/*/*/calls/*/complete.json')}
                resumed = run('resume_check', options)
                if resumed['new_calls'] != 0:
                    raise RuntimeError('Resume repeated API calls')
                if any(file_hash(OUT/p) != sha for p,sha in before.items()):
                    raise RuntimeError('Resume modified a completed checkpoint')
                migration = read_json(OUT/'transport-migration.json')
                if any(file_hash(OUT/p) != sha for p,sha in migration['preserved_checkpoints'].items()):
                    raise RuntimeError('Transport migration altered a completed stage')
                if any(read_json(p)['state'] == 'pending' for p in OUT.glob('turns/*/*/calls/*/attempt-*.json')):
                    raise RuntimeError('Unresolved in-flight request in checkpoint journal')
                atomic_json(ready, {'validated':time.time(), 'manifest_hash':file_hash(OUT/'manifest.json'),
                            'pilot_summary':summary, 'zero_call_resume':resumed,
                            'source': 'sk48 first two consecutive turns; same final history; sampled + unsampled paths'})
            elif read_json(ready)['manifest_hash'] != file_hash(OUT/'manifest.json'):
                raise RuntimeError('Validation belongs to a different manifest')
            # Keep extra slots for resumable recovery from provider 5xx outages.
            # HTTP 429 still retries indefinitely inside the generation client.
            summary = run('full', ['--workers','16','--budget','300','--max-attempts','12'])
            if summary['completed'] != 1334:
                raise RuntimeError(f"Expected 1334 finalized responses, got {summary['completed']}")
            status('completed', summary=summary)
        except Exception as e:
            status('stopped', error=str(e))
            raise


if __name__ == '__main__':
    main()
