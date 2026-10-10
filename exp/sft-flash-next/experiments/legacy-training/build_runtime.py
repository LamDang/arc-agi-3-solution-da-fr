"""Package the runner, repository dependencies and optional prepared data for Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def build(out, prepared_data=None):
    train = Path(__file__).resolve().parent
    repo = train.parents[2]
    files = [*train.glob('*.py'), *train.glob('*.sh'), *train.glob('*.md'),
             train/'requirements-gpu.txt', *[p for p in train.glob('*.json') if 'provenance' not in p.name],
             *train.glob('artifacts/*.json'), *train.glob('tests/*.py'),
             *(repo/'exp/sft-flash-next/nll').glob('*.py'),
             repo/'exp/reap-flash-next/reap_model.py', repo/'exp/reap-flash-next/tests/tiny.py',
             *(repo/'exp/reap-flash-next/tests/fixtures').glob('*.json'),
             repo/'data/game_folds/folds.json']
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=str(path.relative_to(repo)))
        if prepared_data:
            data = Path(prepared_data)
            for name in ('manifest.json','folds.json','requests.jsonl'):
                if not (data/name).is_file():
                    raise ValueError('Not a prepared dataset: missing '+name)
            archive.add(data, arcname='prepared-data')
    with tarfile.open(target) as archive:
        assert 'exp/reap-flash-next/tests/fixtures/quantization_config.json' in archive.getnames()
        assert 'data/game_folds/folds.json' in archive.getnames()
    result = dict(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  source_files=len(set(files)),prepared_data_included=bool(prepared_data))
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--prepared-data')
    args = parser.parse_args()
    print(json.dumps(build(args.out,args.prepared_data),indent=2))
