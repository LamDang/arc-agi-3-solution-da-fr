"""Review isolated identical-input CCE repeats and actual split expert VJPs."""
import argparse
import json
import math
from pathlib import Path
import struct
from verify_expert_diagnostic import check_gradients, sha

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(Path(path).read_text())


def tensor_metric(row):
    assert row['finite'] and row['dtype']=='torch.bfloat16'
    assert math.isfinite(row['relative_l2']) and row['relative_l2']>=0
    assert row['different_values']>=0 and row['max_absolute_difference']>=0
    assert row['bitwise_equal']==(row['different_values']==0)
    if row['bitwise_equal']:
        assert row['relative_l2']==row['max_absolute_difference']==0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    args = parser.parse_args()
    job = Path(args.attempt).resolve()
    out = job/'output'
    config = read(job/'config.json')
    monitor = read(job/'monitor.json')
    result = read(out/'result.json')
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert result['completed'] and result['mode']=='diagnostic-expert-head'
    assert result['optimizer_updates']==0 and not result['raw_gradients_retained']
    assert not result['full_model_backward']
    assert not list(out.glob('gradients*')) and not list(out.glob('initial-adapter*'))
    initial = read(out/'initial-comparison.json')
    assert initial['passed'] and initial['raw_tensors']==initial['bitwise_equal_tensors']==74472
    for name,digest in read(job/'source-hashes.json').items():
        assert sha(job/'source'/name)==digest
    provenance = read(out/'provenance.json')
    assert provenance['script_commit']==config['dispatch_commit']
    for name,digest in provenance['source_hashes'].items():
        assert sha(out/'sources'/name)==sha(job/'source'/name)==digest
    for archived in provenance['archived_inputs'].values():
        assert sha(out/archived['path'])==archived['sha256']==config['expected_sha256']['sample']
    for key,digest in config['expected_sha256'].items():
        actual = provenance['sample_hashes'][config['samples'][0]] if key=='sample' else provenance['model_identity'][key]
        assert actual==digest
    captured = read(out/'captured-inputs.json')
    assert not captured['raw_tensors_retained']
    assert captured['head_selected']['shape']==[651,2560]
    assert captured['head_selected']['dtype']=='torch.bfloat16' and captured['head_selected']['stride']==[2560,1]
    assert captured['targets']['shape']==[651] and captured['targets']['dtype']=='torch.int64'
    assert captured['hidden_dtype']==captured['head_weight']['dtype']=='torch.bfloat16'
    assert not captured['head_weight']['requires_grad']
    assert captured['hidden_shape']==[1,16249,2560]
    assert [int(i) for i in captured['experts']]==result['split_layers']==[15,44,45,46,47]
    settings = read(out/'cce-settings.json')
    assert settings['flags']==dict(impl='cce_exact',reduction='mean',filter_eps=None,filter_e_grad=False,
                                  filter_c_grad=False,accum_e_fp32=True,accum_c_fp32=True)
    assert settings['torch_deterministic_algorithms'] and not settings['autocast_enabled']
    assert settings['denominator']==651 and settings['selected_input']==captured['head_selected']
    head = read(out/'head-repeats.json')
    assert len(head)==result['head_repeats']==settings['repeats']==12
    for repeat,row in enumerate(head):
        assert row['repeat']==repeat and math.isfinite(row['loss'])
        assert struct.pack('<f',row['loss']).hex()==row['loss_fp32_bits']
        assert row['loss_equal_first']==(row['loss_fp32_bits']==head[0]['loss_fp32_bits'])
        assert row['loss_equal_previous']==(row['loss_fp32_bits']==head[max(0,repeat-1)]['loss_fp32_bits'])
        for key in ['hidden_gradient_first','hidden_gradient_previous']:
            tensor_metric(row[key])
            assert row[key]['shape']==[651,2560]
    assert result['head_loss_repeatable']==all(r['loss_equal_first'] for r in head)
    assert result['head_hidden_gradient_repeatable']==all(r['hidden_gradient_first']['bitwise_equal'] for r in head)
    expert = read(out/'expert-vjps.json')
    assert len(expert)==result['expert_vjps']==20
    for layer in result['split_layers']:
        rows = [r for r in expert if r['layer']==layer]
        assert [r['variant'] for r in rows]==['native','native-repeat','chunked','custom-unsplit']
        for row in rows:
            assert row['cotangent_seed']==provenance['config']['seed']+layer
            for key in ['output','input_gradient','routing_gradient']:
                tensor_metric(row[key])
            check_gradients(row['adapter_gradients'],1536)
            partitions = row['adapter_partitions']
            assert sum(r['raw_tensors'] for r in partitions.values())==1536
            assert {n for r in partitions.values() for n in r['tensors']}==set(row['adapter_gradients']['tensors'])
            for partition in partitions.values():
                check_gradients(partition)
            if row['variant']=='native-repeat':
                assert row['adapter_gradients']['passed']
                assert all(row[k]['bitwise_equal'] for k in ['output','input_gradient','routing_gradient'])
    staging = read(out/'expert-prefetch.json')
    assert staging['max_staged_layers']<=2
    resources = read(out/'resources.json')
    assert all(r['seconds']>0 and r['samples']>0 and r['tree_pss_bytes']>0 for r in resources)
    compact = [{**r,'adapter_gradients':{k:v for k,v in r['adapter_gradients'].items() if k!='tensors'},
        'adapter_partitions':{name:{k:v for k,v in part.items() if k!='tensors'} for name,part in r['adapter_partitions'].items()}}
        for r in expert]
    report = dict(evidence_checks_passed=True,attempt=job.name,execution_commit=config['dispatch_commit'],
        monitor=monitor,result=result,head_settings=settings,head_repeats=head,expert_vjps=compact,resources=resources,
        raw_gradients_retained=False,pushed_to_remote=False,
        scope='Same-input CCE repeats and fixed-cotangent expert VJPs; not full-model qualification.')
    (ROOT/'reports/expert-head-replay-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    inventory = {str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(attempt=job.name,result=result,evidence_checks_passed=True),indent=2))


if __name__=='__main__':
    main()
