"""Generate thinking for the 22 won NVIDIA DreamTeam games with the Sol25 pipeline.

Same generator, models and settings as run_progressive_sol25.py (Flash draft +
up to two refinements, combined Sol xhigh judges, 15 monitored turns per game,
120K student-input limit, unlimited 429 backoff, $300 estimated-spend guard).
Source: the scored run directory runs/gpt61sol-nvidia-25games-resume2, which
holds the final request log of every game; the three unwon games (fl5273,
ma4173, ss6041) are left out.

First run: a two-turn one-worker pilot, a zero-call resume check, then all 22
games. Later runs skip the validated pilot and resume the full phase from the
durable turn checkpoints.

--max-minutes stops the runner gracefully (SIGTERM: in-flight responses are
journaled, no new calls start) so that a harness-tracked task with a 2-hour
limit never hard-kills an HTTP request. The driver then records `paused` and
exits 0; relaunch the same command to continue.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen.progressive import atomic_json, digest, file_hash, read_json

RUN = ROOT / 'ARC3-Inference/runs/gpt61sol-nvidia-25games-resume2'
OUT = ROOT / 'ARC3-Inference/runs/think-progressive-nvidia22'
WON = ['al7306', 'cc2048', 'cg1842', 'cl0426', 'df4821', 'dl4827', 'fw4821', 'gc4721',
       'hr2048', 'll4821', 'mb2741', 'ml2048', 'mt4926', 'ne4172', 'os1842', 'ps1842',
       'ps7413', 'rl2048', 'rs0427', 'sf2048', 'sl4821', 'td4826']
EXPECTED = 1297
PILOT_GAME = 'cc2048-f080c4af_p0'
BASE = [sys.executable, '-m', 'think_gen.progressive', '--run', str(RUN), '--out', str(OUT),
        '--tokenizer', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/tokenizer.json'),
        '--template', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/chat_template.jinja'),
        '--monitor-per-game', '15']


class Paused(Exception):
    pass


def status(phase, **extra):
    atomic_json(OUT/'driver_status.json', {'phase': phase, 'updated': time.time(),
                'driver_pid': os.getpid(), **extra})
    print(phase, json.dumps(extra), flush=True)


def run(phase, options, deadline):
    env = dict(os.environ, PYTHONPATH=str(ROOT/'ARC3-Inference'))
    stopped_by_us = False
    with (OUT/'runner.log').open('a', buffering=1) as log:
        child = subprocess.Popen(BASE+options, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status(phase, child_pid=child.pid, options=options)
        while True:
            try:
                code = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                if not stopped_by_us and time.time() > deadline:
                    child.send_signal(signal.SIGTERM)
                    stopped_by_us = True
    summary = read_json(OUT/'summary.json')
    atomic_json(OUT/f'{phase}-summary.json', summary)
    if stopped_by_us:
        raise Paused(f'{phase} paused at {summary["completed"]} turns')
    if code:
        raise RuntimeError(f'{phase} stopped with exit {code}; inspect runner.log and summary.json')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-minutes', type=float, default=110)
    ap.add_argument('--workers', default='16')
    args = ap.parse_args()
    deadline = time.time() + args.max_minutes * 60
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.driver.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            ready = OUT/'pilot-validated.json'
            if not ready.exists():
                options = ['--games', PILOT_GAME.split('-')[0], '--limit', '2', '--workers', '1',
                           '--budget', '8', '--max-attempts', '6']
                summary = run('pilot', options, deadline)
                if summary['completed'] != 2 or summary['monitored'] != 1:
                    raise RuntimeError('Pilot did not finish sampled and unsampled turns')
                rows = [read_json(p) for p in sorted((OUT/'turns'/PILOT_GAME).glob('*/final.json'))[:2]]
                if [r['monitoring_sample'] for r in rows] != [True, False]:
                    raise RuntimeError('Pilot monitoring selection mismatch')
                first, second = rows
                if second['history_refs'].get(first['ref']) != digest(first['thinking']):
                    raise RuntimeError('Pilot did not preserve finalized thinking')
                if first['final_judge'] is None or first['regenerated_code'] is None:
                    raise RuntimeError('Sampled turn missing final monitoring')
                if second['final_judge'] is not None or second['regenerated_code'] is not None:
                    raise RuntimeError('Unsampled turn unexpectedly monitored')
                before = {str(p.relative_to(OUT)): file_hash(p) for p in OUT.glob('turns/*/*/calls/*/complete.json')}
                resumed = run('resume_check', options, deadline)
                if resumed['new_calls'] != 0:
                    raise RuntimeError('Resume repeated API calls')
                if any(file_hash(OUT/p) != sha for p, sha in before.items()):
                    raise RuntimeError('Resume modified a completed checkpoint')
                if any(read_json(p)['state'] == 'pending' for p in OUT.glob('turns/*/*/calls/*/attempt-*.json')):
                    raise RuntimeError('Unresolved in-flight request in checkpoint journal')
                atomic_json(ready, {'validated': time.time(), 'manifest_hash': file_hash(OUT/'manifest.json'),
                            'pilot_summary': summary, 'zero_call_resume': resumed,
                            'source': f'{PILOT_GAME} first two consecutive turns; sampled + unsampled paths'})
            elif read_json(ready)['manifest_hash'] != file_hash(OUT/'manifest.json'):
                raise RuntimeError('Validation belongs to a different manifest')
            summary = run('full', ['--games', *WON, '--workers', args.workers,
                                   '--budget', '300', '--max-attempts', '12'], deadline)
            if summary['completed'] != EXPECTED:
                raise RuntimeError(f"Expected {EXPECTED} finalized responses, got {summary['completed']}")
            status('completed', summary=summary)
        except Paused as e:
            status('paused', note=str(e))
        except Exception as e:
            status('stopped', error=str(e))
            raise


if __name__ == '__main__':
    main()
