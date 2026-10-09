"""Independent NumPy review of the all-expert reference's raw Torch shards."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from verify_iso import ROOT,Tensors,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    job=Path(args.attempt).resolve();output=job/'output'
    read=lambda p:json.loads(p.read_text())
    config=read(job/'config.json');result=read(output/'result.json');monitor=read(job/'monitor.json')
    assert monitor['returncode']==0 and not monitor['timed_out']
    assert config['architecture']=='reference' and config['diagnostic_initialization']
    assert not config['optimizations'].get('bf16_lora',False)
    for key in ['bf16_activations','disk_ple','offload_routed_experts','lora_routed_experts']:
        assert config['optimizations'][key] is True
    assert config['optimizations'].get('head','native')=='native'
    assert result['tokens']==16249 and result['targets']==651
    assert result['optimizer_updates']==0 and not result['clipping_applied']
    assert result['raw_gradient_tensors']==74472
    assert result['gradient_dtypes']==result['parameter_dtypes']=={'torch.float32':74472}
    assert result['trainable_parameter_devices']=={'cuda':744,'cpu':73728}
    for name,digest in read(job/'source-hashes.json').items():assert sha(job/'source'/name)==digest,name
    provenance=read(output/'provenance.json')
    assert provenance['model_config_sha256']==config['expected_sha256']['model_config']
    assert provenance['script_commit']==config['dispatch_commit']
    for original,row in provenance['archived_inputs'].items():
        assert sha(output/row['path'])==row['sha256']==provenance['sample_hashes'][original]==config['expected_sha256']['sample']
    for name,digest in provenance['source_hashes'].items():
        assert sha(output/'sources'/name)==sha(job/'source'/name)==digest,name
    fixture=read(job/'expert-offload-check.json');assert fixture['passed'] and fixture['loss_bitwise_equal']
    inventory=Counter(row['implementation'] for row in read(output/'components.json'))
    assert inventory['CPUBF16Experts']==48 and inventory['BF16Decoder']==48
    assert not any('Liger' in name or 'DirectBias' in name for name in inventory)
    loading=read(output/'loading.json')
    # Runtime checked native empty sets before JSON's default=str serialization.
    assert all(loading.get(key) in (None,[],'set()') for key in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    staged=read(output/'expert-prefetch.json')
    assert staged['canonical_device']=='cpu' and staged['max_staged_layers']<=2
    executes=[r for r in staged['records'] if r['event']=='execute']
    assert {r['layer'] for r in executes if r['direction']==1}==set(range(48))
    assert {r['layer'] for r in executes if r['direction']==-1}==set(range(48))
    assert all(r['input_dtype']=='torch.bfloat16' for r in executes)
    summary=read(output/'gradient-summary.json');metadata={};nonzero={}
    for stem in ['initial-adapter','gradients']:
        manifest=read(output/(stem+'-manifest.json'));assert manifest['tensors']==74472
        seen=set();nz=elements=0;shapes={}
        for row in manifest['shards']:
            path=output/row['path'];assert sha(path)==row['sha256']
            archive=Tensors(path);assert len(archive.state)==row['tensors']
            for name in archive.state:
                assert name not in seen;seen.add(name)
                value=archive.tensor(name);assert value.dtype==np.dtype('<f4') and np.isfinite(value).all(),name
                count=int(np.count_nonzero(value));nz+=int(count>0);elements+=value.size;shapes[name]=list(value.shape)
                if stem=='gradients':
                    assert summary[name]['nonzero']==count and summary[name]['dtype']=='torch.float32'
                    assert summary[name]['finite']
                else:assert count>0,name
            archive.zip.close()
        assert len(seen)==74472
        routed=[name for name in seen if '.mlp.experts.' in name]
        assert len(routed)==73728
        pairs={(int(m[1]),int(m[2]),m[3]) for name in routed
               if (m:=re.search(r'layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.',name))}
        assert len(pairs)==48*256*3
        nonzero[stem]=nz;metadata[stem]=dict(tensors=len(seen),elements=elements,bytes=elements*4,shards=len(manifest['shards']))
    resources=read(output/'resources.json')
    for phase in ['forward','backward']:
        row=next(r for r in resources if r['phase']==phase)
        assert row['seconds']>0 and row['cuda_peak_allocated_bytes']>0 and row['tree_pss_bytes']>0
    manifest={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(passed=True,attempt=job.name,execution_commit=config['dispatch_commit'],loss=result['loss'],
        tokens=16249,targets=651,raw_gradient_tensors=74472,raw_gradient_dtype='FP32',
        routed_expert_adapter_tensors=73728,nonzero_tensors=nonzero,archives=metadata,
        absent_unselected_gradient_tensors=len(result['absent_unrouted_gradients']),
        cpu_expert_prefetch_max_layers=staged['max_staged_layers'],bf16_expert_inputs_verified=True,
        source_input_and_archive_hashes_verified=True,optimizer_updates=0,clipping_applied=False,
        initialization='Test-only reproducible nonzero A/B; all exact values archived. Production uses PEFT random-A/zero-B.',
        scope='Measured 16K user-defined reference; not a 130K capacity or optimizer-memory result.',
        resources=resources,pushed_to_remote=False)
    (ROOT/'reports/reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='resources'},indent=2))


if __name__=='__main__':main()
