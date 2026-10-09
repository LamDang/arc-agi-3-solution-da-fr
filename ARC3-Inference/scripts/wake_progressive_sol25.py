"""One-shot, user-authorized Codex queue wake-up; records delivery failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread', required=True)
    parser.add_argument('--delay', type=float, default=600)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    due = time.time() + args.delay
    state = {'thread': args.thread, 'scheduled_at': time.time(), 'due_at': due,
             'due_utc': datetime.fromtimestamp(due, timezone.utc).isoformat(),
             'state': 'waiting'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(state) + '\n')
        tmp.replace(args.out)
    save()
    time.sleep(max(0, due - time.time()))
    message = (
        'User-authorized ten-minute wake-up test: continue the progressive Sol25 run '
        'in this same conversation. Read the run status and release watcher status. '
        'When at least 13 of 25 games finish, publish the immutable DVC checkpoint, '
        'commit/push the branch and create/attach the PR. After all 1334 turns finish, '
        'publish the final DVC snapshot and commit/push all completed results. '
        'Avoid duplicating operations already recorded by the release watcher.')
    try:
        result = subprocess.run(['codex', 'queue', '--thread', args.thread,
                                 '--message', message], capture_output=True, text=True,
                                timeout=90)
        state.update(state='delivered' if result.returncode == 0 else 'failed',
                     returncode=result.returncode, stdout=result.stdout[-3000:],
                     stderr=result.stderr[-3000:])
    except (OSError, subprocess.TimeoutExpired) as e:
        state.update(state='failed', error=str(e))
    state['finished_at'] = time.time()
    save()


if __name__ == '__main__':
    main()
