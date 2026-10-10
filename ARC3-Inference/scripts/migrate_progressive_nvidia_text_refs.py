"""One-time migration of runs/think-progressive-nvidia22 to normalized text refs.

think_gen.context.message_ref now hashes a text-only reply after removing
blank lines, as the harness stores it in history; before, the reply and its
history copy hashed differently and the next turn failed with a KeyError.
This updates the stored ref of finalized text-only turns, re-checks every
finalized turn's history hash under the new rule, and records the new
context.py hash in the manifest and pilot validation. Thinking, judge
verdicts and API journals are not touched.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen import context, logs
from think_gen.progressive import atomic_json, digest, file_hash, read_json

OUT = ROOT / 'ARC3-Inference/runs/think-progressive-nvidia22'
RUN = ROOT / 'ARC3-Inference/runs/gpt61sol-nvidia-25games-resume2'
CODE = 'ARC3-Inference/think_gen/context.py'


def main():
    if (OUT / 'text-ref-migration.json').exists():
        raise SystemExit('Already migrated')
    if any(read_json(p)['state'] == 'pending' for p in OUT.glob('turns/*/*/calls/*/attempt-*.json')):
        raise SystemExit('Pending attempt in journal; inspect before migrating')
    manifest = read_json(OUT / 'manifest.json')
    old_manifest_hash = file_hash(OUT / 'manifest.json')
    untouched = {str(p.relative_to(OUT)): file_hash(p) for p in OUT.glob('turns/**/*.json')}
    changed = []
    for path in logs.request_logs(RUN):
        records = logs.read_log(path)
        history = {}
        for rec in records:
            final = OUT / 'turns' / rec.game / f'{rec.index:05d}' / 'final.json'
            if not final.exists():
                break
            row = read_json(final)
            refs = [context.message_ref(m) for m in rec.messages if m['role'] == 'assistant']
            if digest({ref: digest(history[ref]) for ref in refs}) != row['history_hash']:
                raise SystemExit(f'History hash differs under normalized refs: {rec.key}')
            ref = context.message_ref(rec.reply)
            if ref != row['ref']:
                if not row['ref'].startswith('text:') or rec.reply.get('tool_calls'):
                    raise SystemExit(f'Unexpected ref change: {rec.key}')
                before, old_ref = file_hash(final), row['ref']
                row['ref'] = ref
                atomic_json(final, row)
                changed.append({'path': str(final.relative_to(OUT)), 'key': rec.key, 'old_ref': old_ref, 'new_ref': ref, 'old_sha256': before,
                    'new_sha256': file_hash(final)})
                untouched.pop(str(final.relative_to(OUT)))
            history[row['ref']] = row['thinking']
    if any(file_hash(OUT / p) != sha for p, sha in untouched.items()):
        raise SystemExit('A checkpoint other than the migrated refs changed')
    old_code = manifest['code'][CODE]
    manifest['code'][CODE] = file_hash(ROOT / CODE)
    atomic_json(OUT / 'manifest.json', manifest)
    ready = read_json(OUT / 'pilot-validated.json')
    if ready['manifest_hash'] != old_manifest_hash:
        raise SystemExit('Pilot validation belongs to a different manifest')
    ready['manifest_hash'] = file_hash(OUT / 'manifest.json')
    atomic_json(OUT / 'pilot-validated.json', ready)
    atomic_json(OUT / 'text-ref-migration.json', {
        'migrated': time.time(), 'code': CODE, 'old_code_sha256': old_code, 'new_code_sha256': manifest['code'][CODE],
        'old_manifest_sha256': old_manifest_hash, 'new_manifest_sha256': ready['manifest_hash'],
        'changed_refs': changed, 'unchanged_checkpoints': len(untouched)})
    print(json.dumps({'changed_refs': changed, 'unchanged_checkpoints': len(untouched)}, indent=2))


if __name__ == '__main__':
    main()
