"""Paid, resumable cache probe: real long history, changed suffix, then next turn.

No source or finalized outputs are changed. Probe attempts live beside the
chosen turn's stages so they count toward the production ledger and budget.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ARC3-Inference'))
from think_gen import client, context, logs, progressive as p, prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-out', type=Path, help='Separate probe ledger while production drains; merge before resuming.')
    parser.add_argument('--game', default='ar25')
    parser.add_argument('--turn', type=int, default=17)
    opts = parser.parse_args()
    probe_out = opts.probe_out or opts.out
    probe_out.mkdir(parents=True, exist_ok=True)
    args = p.parser().parse_args([
        '--run', str(ROOT/'ARC3-Inference/runs/gpt61sol-features-25games'),
        '--out', str(probe_out), '--budget', '3' if opts.probe_out else '300', '--max-attempts', '3',
        '--tokenizer', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/tokenizer.json'),
        '--template', str(ROOT/'data/sft-gpt61sol-features-25games/qwen/chat_template.jinja')])
    manifest = p.build_manifest(args, logs.request_logs(args.run))
    source_manifest = p.read_json(opts.out/'manifest.json')
    for key in ('source', 'settings', 'tokenizer', 'template', 'version'):
        if source_manifest[key] != manifest[key]:
            raise ValueError(f'Probe source/settings differ from production: {key}')
    if opts.probe_out:
        manifest_path = probe_out/'manifest.json'
        if manifest_path.exists() and p.read_json(manifest_path) != manifest:
            raise ValueError('Probe manifest changed; inspect before resume')
        p.atomic_json(manifest_path, manifest)
    elif source_manifest != manifest:
        raise ValueError('Probe requires the reviewed production manifest migration')
    # Hold the same writer lock as production throughout the probe.
    import fcntl
    with (probe_out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.read_json(path)['state'] == 'pending' for path in probe_out.glob('turns/*/*/calls/*/attempt-*.json')):
            raise RuntimeError('Inspect pending requests before the cache probe')
        source = logs.request_logs(args.run, [opts.game])
        if len(source) != 1:
            raise ValueError('Select exactly one game')
        records = logs.read_log(source[0])
        student = p.StudentFormat(args.tokenizer, args.template)
        calls = p.DurableCalls(probe_out, args)
        history = {}
        for rec in records[:opts.turn + 2]:
            row = p.read_json(opts.out/'turns'/rec.game/f'{rec.index:05d}'/'final.json')
            history[row['ref']] = row['thinking']
        key = f'think-cache-validation-v1-{records[0].game}'
        evidence = []
        for label, index, with_code in [('warm', opts.turn, False),
                                        ('changed_suffix', opts.turn, True),
                                        ('next_turn', opts.turn + 1, False)]:
            rec = records[index]
            row = p.read_json(opts.out/'turns'/rec.game/f'{index:05d}'/'final.json')
            messages, tools = student.messages(rec, history)
            code = (row.get('regenerated_code') or prompts.python_args(rec.reply).get('code')) if with_code else None
            sol = p.api_history(messages, 'sol') + [
                {'role': 'user', 'content': p.combined_prompt(rec, row['thinking'], code)}]
            folder = probe_out/'turns'/rec.game/f'{opts.turn:05d}'
            stage = f'cache_probe_{label}'
            calls.call(folder, stage, 'sol', sol, [],
                       lambda r: p.validate_judge(r, code is not None, True), cache_key=key)
            completed = p.read_json(folder/'calls'/stage/'complete.json')
            attempt = p.read_json(folder/'calls'/stage/completed['attempt'])
            usage = attempt['response']['usage']
            details = usage['input_tokens_details']
            result = {'label': label, 'game': rec.game, 'turn': index,
                      'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
                      'cached_tokens': details.get('cached_tokens', 0),
                      'cache_write_tokens': details.get('cache_write_tokens', 0),
                      'ordinary_input_tokens': usage['input_tokens'] - details.get('cached_tokens', 0) - details.get('cache_write_tokens', 0),
                      'cache_fraction': details.get('cached_tokens', 0) / usage['input_tokens'],
                      'cost_usd': attempt['charged_usd'], 'seconds': attempt['response']['secs'],
                      'semantic_request_hash': attempt['request_hash'],
                      'transport_hash': attempt['transport_hash'], 'cache_key': key}
            evidence.append(result)
            p.atomic_json(probe_out/'cache-validation.json', {'updated':time.time(), 'passed':False, 'requests':evidence})
            print(json.dumps(result), flush=True)
        shared = evidence[1]
        growth = evidence[2]
        passed = (evidence[0]['input_tokens'] >= 50000 and
                  shared['cache_fraction'] >= .9 and shared['cache_write_tokens'] < 1024 and
                  growth['cache_fraction'] >= .8 and growth['cache_write_tokens'] < growth['input_tokens'] * .25)
        p.atomic_json(probe_out/'cache-validation.json', {'updated':time.time(), 'passed':passed, 'requests':evidence,
                      'cost_usd': sum(r['cost_usd'] for r in evidence),
                      'new_calls': calls.new_calls,
                      'criteria': '>=50K input; changed suffix >=90% reads and <1024 writes; growing history >=80% reads and <25% writes'})
        if not passed:
            raise RuntimeError('Cache probe did not meet reuse criteria; production remains paused')


if __name__ == '__main__':
    main()
