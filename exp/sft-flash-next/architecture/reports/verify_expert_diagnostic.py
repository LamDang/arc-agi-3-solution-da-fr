"""Review the focused expert diagnostic; it does not qualify a full architecture."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
read=lambda path:json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check_gradients(report,expected=None):
    rows=report['tensors'];assert len(rows)==report['raw_tensors']
    if expected is not None:assert len(rows)==expected
    assert all(r['finite'] and r['candidate_dtype']==r['reference_dtype']=='torch.float32' for r in rows.values())
    assert sum(r['bitwise_equal'] for r in rows.values())==report['bitwise_equal_tensors']
    reference=sum(r['reference_norm']**2 for r in rows.values())
    error=sum(r['reference_norm']**2*r['relative_l2']**2 if r['reference_norm'] else r['candidate_norm']**2 for r in rows.values())
    assert math.isclose(report['global_relative_l2'],math.sqrt(error/reference),rel_tol=1e-9,abs_tol=1e-12)
    assert not any(k.startswith('loss') or k=='baseline_loss' for k in report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    job=Path(args.attempt).resolve();out=job/'output'
    config=read(job/'config.json');monitor=read(job/'monitor.json');result=read(out/'result.json')
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert result['completed'] and result['mode']=='diagnostic-expert'
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
    shadows=read(out/'expert-shadows.json')
    assert [r['layer'] for r in shadows]==list(range(len(shadows)))
    assert result['layers_checked']==len(shadows) and all(r['output']['finite'] for r in shadows)
    mismatch=next((r['layer'] for r in shadows if not r['output']['bitwise_equal']),None)
    if 'focused_layer' in result:
        assert result['first_mismatching_layer']==mismatch
        assert result['all_expert_outputs_bitwise']==(mismatch is None)
        variants=read(out/'expert-vjps.json')
        assert [r['variant'] for r in variants]==['native','native-repeat','chunked','custom-unsplit']
        assert result['variants']==4
        for row in variants:
            for key in ['output','input_gradient','routing_gradient']:assert row[key]['finite']
            check_gradients(row['adapter_gradients'],1536)
            partitions=row['adapter_partitions']
            assert sum(r['raw_tensors'] for r in partitions.values())==1536
            assert {n for r in partitions.values() for n in r['tensors']}==set(row['adapter_gradients']['tensors'])
            for partition in partitions.values():check_gradients(partition)
        repeated=variants[1]
        assert repeated['adapter_gradients']['passed']
        assert all(repeated[key]['bitwise_equal'] for key in ['output','input_gradient','routing_gradient'])
        cotangent=read(out/'cotangent.json')
        assert cotangent['seed']==provenance['config']['seed']+result['focused_layer'] and not cotangent['raw_values_retained']
    else:
        assert result['all_expert_outputs_bitwise'] and len(shadows)==48
        variants=[];cotangent=None
    staging=read(out/'expert-prefetch.json');assert staging['max_staged_layers']<=2
    resources=read(out/'resources.json')
    assert all(r['seconds']>0 and r['samples']>0 and r['tree_pss_bytes']>0 for r in resources)
    compact=[]
    for row in variants:
        compact.append({**row,'adapter_gradients':{k:v for k,v in row['adapter_gradients'].items() if k!='tensors'},
            'adapter_partitions':{name:{k:v for k,v in part.items() if k!='tensors'} for name,part in row['adapter_partitions'].items()}})
    report=dict(evidence_checks_passed=True,attempt=job.name,execution_commit=config['dispatch_commit'],
        monitor=monitor,result=result,shadows=shadows,variants=compact,cotangent=cotangent,
        resources=resources,raw_gradients_retained=False,pushed_to_remote=False,
        scope='Same-input expert forward and fixed-cotangent VJP diagnosis; not full-model qualification.')
    (ROOT/'reports/expert-replay-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    inventory={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(result=result,attempt=job.name,evidence_checks_passed=True),indent=2))


if __name__=='__main__':main()
