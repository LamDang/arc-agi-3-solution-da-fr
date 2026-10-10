"""Review capacity captures without retaining or reconstructing raw gradients."""
import argparse
import hashlib
import json
from pathlib import Path
from evidence_checks import numerical_profile

ROOT=Path(__file__).resolve().parents[1]
read=lambda path:json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--label',choices=['gpulora']);args=parser.parse_args()
    job=Path(args.attempt).resolve();out=job/'output'
    config=read(job/'config.json');monitor=read(job/'monitor.json')
    profile=numerical_profile(job,config)
    result=read(out/'result.json');provenance=read(out/'provenance.json')
    settings=provenance['numerical_settings']
    assert settings['deterministic_algorithms'] and settings['cublas_workspace_config']==':4096:8'
    assert settings['cuda_allocator_config']=='expandable_segments:True'
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert result['completed'] and result['mode']=='benchmark'
    assert result['optimizer_updates']==0 and not result['clipping_applied']
    assert not result['raw_gradients_retained'] and not result['unchunked_control_executed']
    assert not (out/'gradients').exists() and not list(out.glob('gradients*.pt'))
    assert not (out/'initial-adapter').exists()
    assert config['prompt_tokens'] is None and config['benchmark_repeats']==2
    for key in ['expert_chunking','qsa_chunking','hyperconnection_chunking','ple_chunking']:
        assert config['optimizations'][key]
    initial=read(out/'initial-comparison.json')
    assert initial['passed'] and initial['bitwise_equal_tensors']==74472
    for name,digest in read(job/'source-hashes.json').items():assert sha(job/'source'/name)==digest
    for name,digest in provenance['source_hashes'].items():assert sha(out/'sources'/name)==digest
    for name,digest in config['expected_sha256'].items():
        actual=provenance['sample_hashes'][config['samples'][0]] if name=='sample' else provenance['model_identity'][name]
        assert actual==digest
    for archived in provenance['archived_inputs'].values():
        assert sha(out/archived['path'])==archived['sha256']
    resources=read(out/'resources.json');phases={r['phase']:r for r in resources}
    repeats=result['repeats'];assert len(repeats)==2
    fixture=read(ROOT/'results/20261010-benchmark-fixtures/manifest.json')
    expected=next(r for r in fixture['rows'] if r['tokens']==config['benchmark_tokens'])
    assert expected['sha256']==config['expected_sha256']['sample']
    for i,row in enumerate(repeats):
        assert row['repeat']==i and row['all_finite'] and row['adapter_tensors']==74472
        assert row['tokens']==expected['tokens'] and row['targets']==expected['targets']
        for phase in [f'forward-{i}',f'backward-{i}',f'gradient_statistics-{i}']:
            assert phases[phase]['seconds']>0 and phases[phase]['samples']>0
    staging=read(out/'expert-prefetch.json');assert staging['max_staged_layers']<=2
    if args.label=='gpulora':
        assert staging['trainable_device']==staging['gradient_device']=='cuda'
        assert staging['parameter_devices']=={'cuda':73728}
        assert staging['gradient_devices'].get('cuda',0)>0 and set(staging['gradient_devices'])=={'cuda'}
    executes=[r for r in staging['records'] if r['event']=='execute']
    assert [r['layer'] for r in executes if r['direction']==1]==list(range(48))*2
    assert [r['layer'] for r in executes if r['direction']==-1]==list(range(47,-1,-1))*2
    assert all(r['input_dtype']=='torch.bfloat16' for r in executes)
    replay=[r for r in staging['records'] if r['event']=='chunk_backward']
    assert [r['layer'] for r in replay]==list(range(47,-1,-1))*2
    chunks=read(out/'chunking.json');size=config['optimizations']['chunk_tokens'];tokens=config['benchmark_tokens']
    attentions=[r for n,r in chunks.items() if n.endswith('.self_attn')]
    assert attentions and all(r['max_query_tokens']<=size and r['key_tokens']==tokens for r in attentions)
    names={r['name'] for r in read(out/'components.json') if r['implementation']=='ChunkedResidual'}
    mixers=[r for n,r in chunks.items() if any(n.endswith(name) for name in names)]
    assert len(mixers)==97 and all(r['max_tokens']<=size and r['projection_tokens']==tokens for r in mixers)
    decoders=[r for r in chunks.values() if 'max_injection_tokens' in r]
    assert len(decoders)==48 and all(r['max_injection_tokens']<=size for r in decoders)
    ple=[r for n,r in chunks.items() if n.endswith('.ple')]
    assert len(ple)==1 and ple[0]['halo_tokens']==9 and ple[0]['max_window_tokens']<=size+9
    preparations=[json.loads(line) for line in (out/'ple-events.jsonl').read_text().splitlines()]
    assert len(preparations)==1
    preparation=preparations[0]
    assert preparation['tokens']==tokens and preparation['lookup_bytes']==tokens*2560*2
    assert preparation['disk_read_passes']==1 and 0<preparation['unique_rows']<=preparation['total_row_ids']
    preparation['seconds']=(preparation['finished_monotonic_ns']-preparation['started_monotonic_ns'])/1e9
    assert preparation['seconds']>0
    report=dict(evidence_checks_passed=True,attempt=job.name,tokens=config['benchmark_tokens'],
        fixture=expected,execution_commit=config['dispatch_commit'],resources=resources,
        repeats=repeats,loss_repeat_bitwise_equal=repeats[0]['loss']==repeats[1]['loss'],
        optimizer_updates=0,raw_gradients_retained=False,unchunked_control_executed=False,
        measurement_scope=result['measurement_scope'],first_pass_cache_state=result['first_pass_cache_state'],
        monitor=monitor,pushed_to_remote=False,profile_label=args.label or 'historical-cpu-lora',
        expert_placement={k:v for k,v in staging.items() if k!='records'} if args.label else None,
        numerical_profile=profile,numerical_settings=settings,ple_preparation=preparation)
    inspection=job/'ple-storage-inspection.json'
    if inspection.exists():
        storage=read(inspection)
        counters=dict(line.split(': ',1) for line in storage['worker_io_snapshot'].splitlines())
        storage['worker_cumulative_read_bytes']=int(counters['read_bytes'])
        report['ple_storage_inspection']=storage
    for filename,key in [('server-disk-inventory.json','server_disk_inventory'),
                         ('server-block-details.json','server_block_details')]:
        if (job/filename).exists():report[key]=read(job/filename)
    suffix='-'+args.label if args.label else ''
    (ROOT/'reports'/f"benchmark-{config['benchmark_tokens']}{suffix}.json").write_text(json.dumps(report,indent=2)+'\n')
    inventory={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'resources'}},indent=2))


if __name__=='__main__':main()
