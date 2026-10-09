"""Review saved component wiring evidence. This is not a numerical qualification."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from verify_iso import ROOT, Tensors, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');args=p.parse_args()
    job=Path(args.attempt).resolve();output=job/'output'
    read=lambda path:json.loads(path.read_text())
    config=read(job/'config.json');result=read(output/'result.json');monitor=read(job/'monitor.json')
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert result['optimizer_updates']==0 and not result['clipping_applied']
    assert result['tokens']==32 and result['targets']==16 and result['iso_verified'] is None
    assert config['baseline'] is None and config['comparison_mode']=='report'
    assert config['optimizations']['head']=='cce_exact'
    assert all(v is True for k,v in config['optimizations'].items() if k!='head')
    for name,digest in read(job/'source-hashes.json').items():assert sha(job/'source'/name)==digest,name
    for name,row in read(job/'archived-inputs.json').items():
        expected=config['expected_sha256']['adapter' if name==config['adapter'] else 'sample']
        assert sha(job/row['path'])==row['sha256']==expected
    counts=dict(Counter(row['implementation'] for row in read(output/'components.json')))
    expected={'BF16LigerGatedRMS':36,'BF16GDN':36,'BF16LigerExperts':48,
        'BF16LigerMLP':48,'BF16MoE':48,'BF16LigerRMS':148,'BF16Residual':97,
        'BF16Decoder':48,'BF16PLE':1,'DirectBiasIndexer':12,'BF16Attention':12,'DirectBiasTextModel':1}
    assert counts==expected,counts
    tensors=Tensors(output/'gradients.pt');assert len(tensors.state)==744
    nonzero=0
    for name in tensors.state:
        bits=tensors.tensor(name);assert bits.dtype==np.dtype('<u2')
        values=(bits.astype('<u4')<<16).view('<f4')
        assert np.isfinite(values).all(),name
        nonzero+=int(np.count_nonzero(values)>0)
    assert sha(output/'gradients.pt')==result['gradients_sha256']
    registry=read(job/'source/configs/dependencies.json')
    for name in ('liger','cce'):
        dep=registry[name];assert sha(job/'dependencies'/dep['filename'])==dep['sha256']
    manifest={str(path.relative_to(job)):dict(bytes=path.stat().st_size,sha256=sha(path))
        for path in sorted(job.rglob('*')) if path.is_file() and path.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(attempt=job.name,execution_commit=config['dispatch_commit'],process_success=True,
        scope='32-token text-only component assembly/forward/backward; no baseline or numerical qualification',
        numerically_qualified=False,loss=result['loss'],tokens=32,targets=16,
        raw_gradient_tensors=744,finite_bf16_gradients=744,nonzero_gradient_tensors=nonzero,
        optimizer_updates=0,components=counts,source_and_dependency_hashes_verified=True,
        input_archives_verified=True,resources=read(output/'resources.json'),pushed_to_remote=False)
    (ROOT/'reports/components-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','components')},indent=2))


if __name__=='__main__':main()
