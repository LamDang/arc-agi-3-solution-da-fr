"""One-time reviewed cache/pricing migration of the paused Sol25 job.

Archives original attempts before repricing; proves immutable completed stages,
finalized thinking, source, model settings and tokenizer stay unchanged.
Run only after the real long-context cache probe passes.
"""
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ARC3-Inference'))
from think_gen import logs, progressive as p
from run_progressive_sol25 import OUT, BASE

BACKUPS = {'ARC3-Inference/think_gen/client.py': Path('/tmp/think-gen-client-before-explicit-cache.py'),
           'ARC3-Inference/think_gen/progressive.py': Path('/tmp/think-gen-progressive-before-explicit-cache.py')}


def main():
    if (OUT/'cache-migration.json').exists():
        raise RuntimeError('Migration already recorded; inspect rather than apply twice')
    with (OUT/'.driver.lock').open('a') as driver_lock, (OUT/'.lock').open('a') as runner_lock:
        for lock in (driver_lock, runner_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pause = p.read_json(OUT/'cache-fix-pause.json')
        if pause['pending'] or not all(s == 'T' for s in pause['thread_states']):
            raise RuntimeError('Missing safe-pause evidence')
        if any(p.read_json(path)['state'] == 'pending' for path in OUT.glob('turns/*/*/calls/*/attempt-*.json')):
            raise RuntimeError('Unresolved in-flight production request')
        args = p.parser().parse_args(BASE[3:] + ['--budget', '300'])
        old = p.read_json(OUT/'manifest.json')
        new = p.build_manifest(args, logs.request_logs(args.run))
        for key in old:
            if key != 'code' and old[key] != new[key]:
                raise RuntimeError(f'Unexpected semantic change: {key}')
        if set(new) - set(old) != {'cache_policy'} or new['cache_policy'] != 'explicit-history-v1':
            raise RuntimeError('Unexpected manifest field changes')
        if set(new['code']) != set(old['code']):
            raise RuntimeError('Generation file set changed')
        for name, sha in old['code'].items():
            if name in BACKUPS:
                if p.file_hash(BACKUPS[name]) != sha:
                    raise RuntimeError(f'Original code backup mismatch: {name}')
            elif new['code'][name] != sha:
                raise RuntimeError(f'Unexpected source change: {name}')
        probe_out = OUT/'cache-validation-probe'
        probe = p.read_json(probe_out/'cache-validation.json')
        if not probe['passed'] or probe['new_calls'] != 0:
            raise RuntimeError('Require successful long probe and its zero-call resume')
        if p.read_json(probe_out/'manifest.json') != new:
            raise RuntimeError('Probe fingerprint does not match new production code')
        if any(p.read_json(path)['state'] == 'pending' for path in probe_out.glob('turns/*/*/calls/*/attempt-*.json')):
            raise RuntimeError('Unresolved in-flight cache probe request')
        attempts = list(OUT.glob('turns/*/*/calls/*/attempt-*.json'))
        immutable = {str(path.relative_to(OUT)): p.file_hash(path)
                     for pattern in ('turns/*/*/final.json', 'turns/*/*/calls/*/complete.json')
                     for path in OUT.glob(pattern)}
        ready = p.read_json(OUT/'pilot-validated.json')
        if ready['manifest_hash'] != p.file_hash(OUT/'manifest.json'):
            raise RuntimeError('Pilot validation fingerprint mismatch before migration')
        old_manifest_hash = p.file_hash(OUT/'manifest.json')
        archive = OUT/'before-explicit-cache.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in attempts + [OUT/'manifest.json', OUT/'pilot-validated.json']:
                z.write(path, str(path.relative_to(OUT)))
            for name in immutable:
                z.write(OUT/name, name)
            for name, backup in BACKUPS.items():
                z.write(backup, 'code/'+name)
        charges = []
        calculator = p.DurableCalls(OUT, args)
        for path in attempts:
            row = p.read_json(path)
            if row['api'] == 'sol' and row.get('response'):
                old_charge = row['charged_usd']
                new_charge = calculator.price(row['response'], 'sol')
                before = p.file_hash(path)
                response_hash = p.digest(row['response'])
                row.update(charged_usd=new_charge, pricing_version=2)
                p.atomic_json(path, row)
                if p.digest(p.read_json(path)['response']) != response_hash:
                    raise RuntimeError('Repricing modified a raw response')
                charges.append({'path':str(path.relative_to(OUT)), 'old_usd':old_charge,
                                'new_usd':new_charge, 'response_hash':response_hash,
                                'before_sha256':before, 'after_sha256':p.file_hash(path)})
        # Merge paid validation stages into production so the $300 guard includes them.
        merged = {}
        for path in probe_out.glob('turns/*/*/calls/*/*.json'):
            rel = path.relative_to(probe_out)
            target = OUT/rel
            if target.exists():
                raise RuntimeError(f'Probe destination already exists: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            merged[str(rel)] = p.file_hash(target)
        p.atomic_json(OUT/'cache-validation.json', probe)
        p.atomic_json(OUT/'manifest.before-explicit-cache.json', old)
        p.atomic_json(OUT/'pilot-validated.before-explicit-cache.json', ready)
        p.atomic_json(OUT/'manifest.json', new)
        new_manifest_hash = p.file_hash(OUT/'manifest.json')
        ready.update(manifest_hash=new_manifest_hash, operational_migration='cache-migration.json',
                     cache_validation_passed=True)
        p.atomic_json(OUT/'pilot-validated.json', ready)
        if any(p.file_hash(OUT/name) != sha for name,sha in immutable.items()):
            raise RuntimeError('Immutable stage or final turn changed')
        record = {'at':time.time(), 'reason':'User authorized explicit history cache and write-price correction',
                  'old_manifest_hash':old_manifest_hash, 'new_manifest_hash':new_manifest_hash,
                  'archive':archive.name, 'archive_sha256':p.file_hash(archive),
                  'old_code':{name:old['code'][name] for name in BACKUPS},
                  'new_code':{name:new['code'][name] for name in BACKUPS},
                  'preserved_checkpoints':immutable, 'repriced_attempts':charges,
                  'extra_write_cost_usd':sum(r['new_usd']-r['old_usd'] for r in charges),
                  'merged_probe_files':merged, 'probe_cost_usd':probe['cost_usd'],
                  'corrected_cost_including_reservations_usd':p.DurableCalls(OUT,args).cost()}
        p.atomic_json(OUT/'cache-migration.json', record)
        print(json.dumps({k:v for k,v in record.items() if k not in
                          ('preserved_checkpoints','repriced_attempts','merged_probe_files')}, indent=2))


if __name__ == '__main__':
    main()
