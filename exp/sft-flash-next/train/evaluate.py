"""Matched full-context validation before/after training, with thinking/code NLL."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

import backend
import kernel_checks
from dataset import Requests, digest, file_hash, read_json, write_json

from common import LABELS


@torch.no_grad()
def evaluate_model(model, bundle, rows, profile, out, metadata):
    """Resume validation at atomic per-request records without re-scoring rows."""
    out = Path(out)
    identity = dict(metadata=metadata, dataset_sha256=bundle.manifest['sha256'],
                    sample_ids=[r['sample_id'] for r in rows], loss_block=profile['loss_block'])
    if out.exists():
        if not (out/'identity.json').exists() or read_json(out/'identity.json') != identity:
            raise ValueError('Validation output belongs to a different identity')
    else:
        out.mkdir(parents=True)
        write_json(out/'identity.json', identity)
    (out/'records').mkdir(exist_ok=True)
    model.eval()
    sums, counts, games = defaultdict(float), defaultdict(int), defaultdict(list)
    results = []
    for row in rows:
        path = out/'records'/f"{digest(row['sample_id'])}.json"
        if path.exists():
            record = read_json(path)
            result = record['result']
            if record['sha256'] != digest(result) or result['sample_id'] != row['sample_id'] or result['game'] != row['game']:
                raise ValueError('Validation record checksum/identity mismatch')
            losses = torch.tensor(result['token_losses'], dtype=torch.float32)
            categories = torch.tensor(result['labels'])
        else:
            enc, ann = bundle.get(row)
            losses = backend.token_losses(model, enc, ann['prompt_tokens'], 'cuda', profile['loss_block'])
            categories = torch.tensor(ann['labels'])
            if len(losses) != row['target_tokens'] or len(categories) != len(losses) or not torch.isfinite(losses).all():
                raise ValueError('Invalid validation loss vector')
            result = dict(sample_id=row['sample_id'], game=row['game'], nll=float(losses.mean()),
                positions=ann['positions'], labels=ann['labels'], token_losses=losses.tolist())
            temp = path.with_suffix('.partial')
            write_json(temp, dict(result=result, sha256=digest(result)))
            with temp.open('rb') as stream:
                os.fsync(stream.fileno())
            temp.replace(path)
        if len(losses) != row['target_tokens'] or len(categories) != len(losses) or not torch.isfinite(losses).all():
            raise ValueError('Invalid validation loss vector')
        results.append(result)
        games[row['game']].append(result['nll'])
        sums['all'] += float(losses.sum())
        counts['all'] += len(losses)
        for i, label in enumerate(LABELS):
            selected = losses[categories == i]
            sums[label] += float(selected.sum())
            counts[label] += len(selected)
        print(f"{row['sample_id']}: {result['nll']:.6f}", flush=True)
    with (out/'requests.jsonl').open('w') as stream:
        for result in results:
            stream.write(json.dumps(result)+'\n')
    per_game = {g: sum(v)/len(v) for g, v in games.items()}
    write_json(out/'summary.json', dict(**metadata, dataset_sha256=bundle.manifest['sha256'],
        macro_game_request_nll=sum(per_game.values())/len(per_game), per_game=per_game,
        token_nll={k: sums[k]/counts[k] if counts[k] else None for k in counts}, token_counts=dict(counts)))
    return read_json(out/'summary.json')


def main():
    from run import PROFILES, check_model, hardware, runtime
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--keep")
    p.add_argument("--data", required=True, help="Prepared held-out generated-thinking requests")
    p.add_argument("--checkpoint-dir", help="Omit for the matched untrained base")
    p.add_argument("--profile", choices=PROFILES, default="a100-80gb")
    p.add_argument("--out", required=True)
    p.add_argument("--ple-block", type=int, default=0)
    p.add_argument("--gated-norm-block", type=int, default=0)
    p.add_argument("--gdn-chunk-tokens", type=int, default=0)
    p.add_argument("--gdn-block-tokens", type=int, default=0)
    p.add_argument("--attention-projection-block", type=int, default=0)
    p.add_argument("--rms-block-mib", type=float, default=0)
    p.add_argument("--ple-resident", action="store_true")
    p.add_argument('--check-model-gradient', action='store_true',
                   help='Run the full-model matched-forward backward qualification before scoring')
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = Requests(args.data)
    rows = bundle.split("validation")
    if not rows:
        raise ValueError("No held-out games in the evaluation bundle")
    gpu = hardware(args.profile)
    model_hash = digest(check_model(args.model, bundle, args.keep))
    state = None
    if args.checkpoint_dir:
        root = Path(args.checkpoint_dir)
        pointer = read_json(root / "latest.json")
        name = pointer["file"]
        if Path(name).name != name or file_hash(root / name) != pointer["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(root / name, map_location="cpu", weights_only=True)
        if state["identity"]["model"] != model_hash:
            raise ValueError("Checkpoint uses a different model/expert map")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    kernel_report = kernel_checks.check(attention_backend="triton")
    write_json(out.with_suffix('.kernel-checks.json'), kernel_report)
    model, _ = backend.load_pruned(args.model, args.keep)
    recipe = state["identity"]["recipe"] if state else dict(rank=16, alpha=32, ple_block=args.ple_block, gated_norm_block=args.gated_norm_block, ple_resident=args.ple_resident, gdn_chunk_tokens=args.gdn_chunk_tokens, rms_block_mib=args.rms_block_mib, gdn_block_tokens=args.gdn_block_tokens, attention_projection_block=args.attention_projection_block)
    profile = dict(PROFILES[args.profile])
    if state:
        if recipe['profile'] != args.profile:
            raise ValueError('Use the training hardware profile for matched validation')
        profile['expert_block'] = recipe['expert_block']
    backend.configure(model, rank=recipe["rank"], alpha=recipe["alpha"],
        expert_block=profile["expert_block"], query_block=profile["query_block"],
        index_block=profile["index_block"], gradient_checkpointing=False,
        attention_backend=recipe.get('attention_backend', 'triton'),
        hyper_block=recipe.get('hyper_block', 1024), ple_cache_gib=1., ple_workers=8,
        ple_block=recipe.get('ple_block', 0), gated_norm_block=recipe.get('gated_norm_block',0), ple_resident=recipe.get('ple_resident',False), gdn_chunk_tokens=recipe.get('gdn_chunk_tokens',0), rms_block_mib=recipe.get('rms_block_mib',0), gdn_block_tokens=recipe.get('gdn_block_tokens',0), attention_projection_block=recipe.get('attention_projection_block',0))
    if state:
        backend.load_adapter(model, state["adapter"])
    if args.check_model_gradient:
        write_json(out.with_suffix('.model-gradient-check.json'),
                   kernel_checks.check_model_backward(model, profile['loss_block']))
    model.eval()
    code_root = Path(__file__).resolve().parents[1]
    code = {str(f.relative_to(code_root)): file_hash(f) for folder in ('train', 'nll')
            for f in sorted((code_root/folder).glob('*.py'))}
    code['reap_model.py'] = file_hash(Path(backend.rm.__file__))
    evaluate_model(model, bundle, rows, profile, out, dict(model_sha256=model_hash,
        hardware=gpu, checkpoint_sha256=pointer['sha256'] if state else None,
        runtime=runtime(), code=code,
        profile=profile, recipe=recipe))



if __name__ == "__main__":
    main()
