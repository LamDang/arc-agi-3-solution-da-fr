"""Verify focused actual-value HC evidence; this is not architecture acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
read=lambda path:json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    job=Path(args.attempt).resolve();out=job/'output'
    config=read(job/'config.json');monitor=read(job/'monitor.json');result=read(out/'result.json')
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert result['completed'] and result['mode']=='diagnostic-hyperconnection'
    assert result['optimizer_updates']==0 and not result['raw_gradients_retained']
    assert not list(out.glob('gradients*')) and not list(out.glob('initial-adapter*'))
    initial=read(out/'initial-comparison.json')
    assert initial['passed'] and initial['raw_tensors']==initial['bitwise_equal_tensors']==74472
    for name,digest in read(job/'source-hashes.json').items():assert sha(job/'source'/name)==digest
    provenance=read(out/'provenance.json');assert provenance['script_commit']==config['dispatch_commit']
    for name,digest in provenance['source_hashes'].items():assert sha(out/'sources'/name)==sha(job/'source'/name)==digest
    for archived in provenance['archived_inputs'].values():assert sha(out/archived['path'])==archived['sha256']==config['expected_sha256']['sample']
    for key,digest in config['expected_sha256'].items():
        actual=provenance['sample_hashes'][config['samples'][0]] if key=='sample' else provenance['model_identity'][key]
        assert actual==digest
    shadows=read(out/'hyperconnection-shadows.json')
    assert result['modules_checked']==len(shadows)>0
    assert len({row['module'] for row in shadows})==len(shadows)
    for row in shadows:
        assert len(row['ports']) in (1,3)
        assert all(p['finite'] for p in row['ports'])
    mismatch=next((row['module'] for row in shadows if not all(p['bitwise_equal'] for p in row['ports'])),None)
    assert result['first_mismatching_module']==mismatch
    if mismatch:assert shadows[-1]['module']==mismatch==result['focused_module']
    else:assert len(shadows)==97
    variants=read(out/'hyperconnection-vjps.json')
    expected=['native','native-repeat','chunk-no-checkpoint','chunked','custom-unsplit','native-outer-offload','chunk-outer-offload']
    assert [row['variant'] for row in variants]==expected and result['variants']==len(expected)
    for row in variants:
        assert row['input_gradient']['finite']
        assert all(p['finite'] for p in row['outputs'])
        assert all(p['finite'] for p in row['projections'].values())
        assert len(row['outputs'])==len(variants[0]['outputs'])
    assert all(p['bitwise_equal'] for p in variants[1]['outputs']) and variants[1]['input_gradient']['bitwise_equal']
    cotangent=read(out/'cotangent.json')
    assert not cotangent['raw_values_retained'] and len(cotangent['ports'])==len(variants[0]['outputs'])
    for index,row in enumerate(cotangent['ports']):assert row['seed']==provenance['config']['seed']+10000+index
    staging=read(out/'expert-prefetch.json');assert staging['max_staged_layers']<=2
    resources=read(out/'resources.json')
    assert all(r['seconds']>0 and r['samples']>0 and r['tree_pss_bytes']>0 for r in resources)
    report=dict(evidence_checks_passed=True,attempt=job.name,execution_commit=config['dispatch_commit'],
        monitor=monitor,result=result,shadows=shadows,variants=variants,cotangent=cotangent,
        resources=resources,raw_gradients_retained=False,pushed_to_remote=False,
        scope='Actual-input HC port/projection shadows and fixed-cotangent VJP diagnosis; not full-model qualification.')
    (ROOT/'reports/hyperconnection-replay-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    inventory={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(result=result,attempt=job.name,evidence_checks_passed=True),indent=2))


if __name__=='__main__':main()
