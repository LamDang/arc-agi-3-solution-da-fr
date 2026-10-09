"""Check DVC cache objects directly against each reviewed attempt's SHA256 inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT.parents[2]/'.dvc/cache/files/md5'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempts',nargs='+');args=parser.parse_args();reports=[]
    for name in args.attempts:
        job=ROOT/'results'/name
        tree=re.search(r'md5: (\w+\.dir)',job.with_suffix('.dvc').read_text()).group(1)
        rows=json.loads((CACHE/tree[:2]/tree[2:]).read_text());inventory=json.loads((job/'file-hashes.json').read_text());checked=0
        for row in rows:
            path=CACHE/row['md5'][:2]/row['md5'][2:]
            assert path.is_file(),row['relpath']
            if row['relpath'] in inventory:
                with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                assert actual==inventory[row['relpath']]['sha256'],row['relpath']
                checked+=1
        assert checked==len(inventory)
        reports.append(dict(attempt=name,dvc_tree=tree,files=len(rows),sha256_verified=checked,passed=True))
    (ROOT/'reports/optimization-cache-verification.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
