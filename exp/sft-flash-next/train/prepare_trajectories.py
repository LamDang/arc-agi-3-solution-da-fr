"""Prepare one complete untruncated sample per verified true trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from dataset import digest, file_hash, fold_games, load_processor, read_json, validate_sample, write_json
from trajectory_data import reconstruct, plan_token_updates
from trajectory_encode import encode, encode_input


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output exists; choose a new immutable directory')
    folds = read_json(args.folds)
    known, validation = fold_games(folds, args.validation_fold)
    generated, sources = {}, {}
    generated_root = Path(args.generated_dir)
    paths = sorted(generated_root.glob('*.jsonl'))
    paths += sorted(generated_root.glob('turns/*/*/final.json'))
    for metadata_name in ('manifest.json', 'snapshot.json'):
        metadata_path = generated_root/metadata_name
        if metadata_path.is_file():
            sources[str(metadata_path)] = file_hash(metadata_path)
    for path in paths:
        sources[str(path)] = file_hash(path)
        records = [json.loads(path.read_text())] if path.suffix == '.json' else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in records:
            if row['key'] in generated:
                raise ValueError(f"Duplicate generated key: {row['key']}")
            generated[row['key']] = row
    games = defaultdict(list)
    with open(args.input, 'rb') as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample['game'] not in known:
                raise ValueError(f"Unknown game: {sample['game']}")
            games[sample['game']].append(sample)
    if not games:
        raise ValueError('Empty input')
    source_run = None
    if getattr(args, 'source_run_dir', None):
        import importlib.util
        converter_path = Path(__file__).resolve().parents[3]/'data/sft-gpt61sol-features-25games/convert.py'
        spec = importlib.util.spec_from_file_location('trajectory_source_converter', converter_path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        run_root = Path(args.source_run_dir)
        evaluation = read_json(run_root/'evaluation.json')
        completed = {r['game_id']:r for r in evaluation['games']}
        source_run = dict(evaluation_sha256=file_hash(run_root/'evaluation.json'), logs={}, games={})
        for game in sorted(games):
            paths = list(run_root.glob(game+'_p0_requests.jsonl*'))
            if len(paths) != 1:
                raise ValueError(f'{game}: expected one original request log')
            original = []
            for request, reply in converter.read_log(paths[0]):
                if reply and (reply.get('tool_calls') or (reply.get('content') or '').strip()):
                    original.append(dict(game=game,request_index=len(original),**converter.request_sample(request,reply)))
            if [digest(r) for r in original] != [digest(r) for r in sorted(games[game],key=lambda r:r['request_index'])]:
                raise ValueError(f'{game}: input does not exactly cover original source log')
            status = completed.get(game)
            if not status or status.get('completion_rate') != 1.0:
                raise ValueError(f'{game}: original evaluation does not confirm completion')
            source_run['logs'][paths[0].name] = file_hash(paths[0])
            source_run['games'][game] = dict(requests=len(original),termination=status)
    processor = load_processor(args.processor)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Failed preparation never leaves a directory resembling a valid dataset.
    with tempfile.TemporaryDirectory(prefix=out.name+'.preparing-', dir=out.parent) as stage:
        root = Path(stage)
        (root/'processor').mkdir()
        for path in Path(args.processor).iterdir():
            if path.is_file() and path.suffix in {'.json','.jinja','.txt','.model'} and 'safetensors' not in path.name:
                shutil.copyfile(path, root/'processor'/path.name)
        write_json(root/'folds.json', folds)
        (root/'annotations').mkdir()
        rows, audit_games, errors = [], [], []
        with open(root/'trajectories.jsonl','wb') as target:
            for game in sorted(games):
                keys = [f"{game}_p0#{r['request_index']}" for r in games[game]]
                missing = [key for key in keys if key not in generated or generated[key].get('status') != 'ok' or not str(generated[key].get('thinking') or '').strip()]
                audit_row = dict(game=game, requests=len(keys), missing_thoughts=missing,
                    request_indices=[r.get('request_index') for r in games[game]])
                audit_games.append(audit_row)
                try:
                    sample = reconstruct(games[game], generated)
                    validate_sample(dict(sample, thinking_source='think_gen-refine2'), known)
                    preflight = encode_input(processor, sample)
                    total_tokens = int(preflight['input_ids'].shape[1])
                    audit_row['total_tokens'] = total_tokens
                    del preflight
                    if total_tokens > args.max_tokens:
                        raise ValueError(f'actual untrimmed {total_tokens} exceeds {args.max_tokens}; no splitting/truncation')
                    enc, annotation = encode(processor, sample)
                    audit_row.update(total_tokens=annotation['total_tokens'], target_tokens=annotation['target_tokens'])
                    if annotation['total_tokens'] > args.max_tokens:
                        raise ValueError(f"actual untrimmed {annotation['total_tokens']} exceeds {args.max_tokens}; no splitting/truncation")
                except ValueError as error:
                    audit_row['error'] = str(error)
                    errors.append(game)
                    continue
                annotation_file = 'annotations/'+sample['trajectory_id']+'.json'
                write_json(root/annotation_file, annotation)
                raw = (json.dumps(sample,ensure_ascii=False)+'\n').encode()
                row = dict(sample_id=sample['trajectory_id'], game=game,
                    split='validation' if game in validation else 'train',
                    offset=target.tell(), length=len(raw), line_sha256=hashlib.sha256(raw).hexdigest(),
                    annotation_sha256=digest(annotation), annotation_file=annotation_file, positions=annotation['positions'],
                    positions_sha256=annotation['positions_sha256'], source_turns=sample['source_turns'],
                    source_thought_coverage=dict(expected=len(games[game]), replaced=len(sample['source_turns']), missing=[]),
                    **{k:annotation[k] for k in ('total_tokens','target_tokens','images','category_counts','input_sha256','target_sha256')})
                rows.append(row)
                target.write(raw)
                print(f"{game}: {row['total_tokens']} context, {row['target_tokens']} assistant targets",flush=True)
                del enc
        write_json(Path(str(out)+'.audit.json'), dict(input_sha256=file_hash(args.input), games=audit_games, failed_games=errors))
        if errors:
            raise ValueError(f'Trajectory preparation blocked for {len(errors)} games; see {out}.audit.json')
        plans = {split:plan_token_updates([r for r in rows if r['split']==split],args.token_budget)
                 for split in ('train','validation')}
        plan = dict(policy='whole-trajectories-approximate-token-budget', token_budget=args.token_budget,
                    order='lexicographic trajectory_id', updates=plans)
        plan['sha256'] = digest(plan)
        write_json(root/'update_plan.json',plan)
        files = {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}
        manifest = dict(version=1,dataset_kind='true-trajectories',validation_fold=args.validation_fold,
            max_tokens=args.max_tokens,input_sha256=file_hash(args.input),generated_sources=sources,
            files=files,rows=rows,update_plan_sha256=plan['sha256'],
            objective='sum all assistant token CE divided by actual supervised tokens per optimizer update',
            reconstruction='contiguous verified history overlap; observed log prefix with indices starting at zero',
            source_completeness='complete original logs with verified game completion' if source_run else 'observed log prefix; termination unverified',
            source_run=source_run,
            trajectory_count=len(rows),request_count=sum(len(v) for v in games.values()))
        manifest['sha256'] = digest(manifest)
        write_json(root/'manifest.json',manifest)
        root.rename(out)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--generated-dir',required=True)
    p.add_argument('--source-run-dir',required=True,help='Original MD5-verified run logs and evaluation.json; verifies complete source coverage')
    p.add_argument('--processor',required=True)
    p.add_argument('--folds',default=str(Path(__file__).resolve().parents[3]/'data/game_folds/folds.json'))
    p.add_argument('--validation-fold',type=int,default=0)
    p.add_argument('--max-tokens',type=int,default=130000)
    p.add_argument('--token-budget',type=int,default=16384)
    p.add_argument('--out',required=True)
    prepare(p.parse_args())


if __name__=='__main__':
    main()
