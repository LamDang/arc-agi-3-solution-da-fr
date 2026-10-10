"""Review preserved candidate metrics and identity; candidates retain no raw gradients.

Checks source/input hashes and internally consistent all-tensor comparison reports.
It cannot independently recompute candidate gradients after their requested deletion.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
from evidence_checks import numerical_profile

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/20261009221135477-25bf58fe'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--label');args=parser.parse_args()
    job=Path(args.attempt).resolve();out=job/'output'
    config=read(job/'config.json');result=read(out/'result.json');monitor=read(job/'monitor.json')
    profile=numerical_profile(job,config)
    comparison=read(out/'comparison.json');initial=read(out/'initial-comparison.json')
    summary=read(out/'gradient-summary.json');reference=read(REFERENCE/'output/gradient-summary.json')
    opt=config['optimizations'];head=opt['head'];bias=opt['direct_attention_bias']
    chunk_flags=['expert_chunking','qsa_chunking','hyperconnection_chunking','ple_chunking']
    enabled=[i for i,k in enumerate(chunk_flags,7) if opt.get(k)]
    label=f'Opt{max(enabled)}' if enabled else {('native',False):'RestoredReference',('target',False):'Opt1',('cce_exact',False):'Opt2',('cce_exact',True):'Opt3'}[(head,bias)]
    label=args.label or label
    assert not monitor['timed_out']
    if monitor['returncode']!=0:
        if enabled:
            assert result['chunking_gradient_gate_passed'] is False
            assert 'user gate' in read(out/'failure.json')['error']
        else:
            assert head=='native' and config['comparison_mode']=='exact' and not comparison['passed']
            assert 'equality gate failed' in read(out/'failure.json')['error']
    assert config['architecture']=='optimized' and config['diagnostic_initialization']
    for key in ['bf16_activations','disk_ple','offload_routed_experts','lora_routed_experts']:assert opt[key] is True
    for key in ['bf16_lora','liger_rmsnorm','liger_swiglu']:assert not opt.get(key,False)
    assert result['tokens']==16249 and result['targets']==651
    assert result['optimizer_updates']==0 and not result['clipping_applied']
    assert result['raw_gradient_tensors']==74472 and result['all_finite']
    assert result['gradient_dtypes']==result['parameter_dtypes']=={'torch.float32':74472}
    assert result['trainable_parameter_devices']=={'cuda':744,'cpu':73728}
    assert result['raw_gradients_retained'] is False and result['gradient_archive'] is None
    assert not (out/'gradients').exists() and not list(out.glob('gradients*.pt'))
    assert not (out/'initial-adapter').exists() and not (out/'initial-adapter.pt').exists()
    assert initial['passed'] and initial['raw_tensors']==initial['bitwise_equal_tensors']==74472
    assert initial['baseline_initial_sha256']==sha(REFERENCE/'output/initial-adapter-manifest.json')
    assert comparison['baseline_gradients_sha256']==sha(REFERENCE/'output/gradients-manifest.json')
    assert comparison['loss']==result['loss'] and comparison['baseline_loss']==read(REFERENCE/'output/result.json')['loss']
    assert comparison['raw_tensors']==len(summary)==len(reference)==74472
    rows=comparison['tensors'];assert set(rows)==set(summary)==set(reference)
    exact=sum(r['bitwise_equal'] for r in rows.values());nonzero=sum(not r['reference_zero'] for r in rows.values())
    assert exact==comparison['bitwise_equal_tensors']
    assert nonzero==comparison['nonzero_reference_tensors']==74394
    assert comparison['bitwise_equal_nonzero_reference_tensors']==sum(r['bitwise_equal'] and not r['reference_zero'] for r in rows.values())
    aa=bb=ee=0.
    for name,row in rows.items():
        a,b=reference[name],summary[name]
        assert a['shape']==b['shape'] and row['finite'] and b['finite']
        assert a['dtype']==b['dtype']=='torch.float32'
        assert row['reference_zero']==(a['nonzero']==0) and row['candidate_zero']==(b['nonzero']==0)
        norm=a['norm']**2;other=b['norm']**2
        aa+=norm;bb+=other;ee+=norm*row['relative_l2']**2 if norm else other
    assert math.isclose(comparison['global_relative_l2'],math.sqrt(ee/aa),rel_tol=1e-9,abs_tol=1e-12)
    assert math.isclose(comparison['cosine'],(aa+bb-ee)/(2*math.sqrt(aa*bb)),rel_tol=1e-9,abs_tol=1e-12)
    assert result['iso_verified']==comparison['passed']==(exact==74472 and comparison['loss_bitwise_equal'])
    for name,digest in read(job/'source-hashes.json').items():assert sha(job/'source'/name)==digest,name
    provenance=read(out/'provenance.json');assert provenance['script_commit']==config['dispatch_commit']
    assert provenance['model_config_sha256']==config['expected_sha256']['model_config']
    for name,digest in provenance['source_hashes'].items():assert sha(out/'sources'/name)==sha(job/'source'/name)==digest,name
    for original,row in provenance['archived_inputs'].items():assert sha(out/row['path'])==row['sha256']==config['expected_sha256']['sample']
    staged=read(out/'expert-prefetch.json');assert staged['max_staged_layers']<=2
    executes=[r for r in staged['records'] if r['event']=='execute']
    passes=2 if enabled else 1
    assert [r['layer'] for r in executes if r['direction']==1]==list(range(48))*passes
    assert [r['layer'] for r in executes if r['direction']==-1]==list(range(47,-1,-1))*passes
    if enabled:
        replay=[r for r in staged['records'] if r['event']=='chunk_backward']
        assert [r['layer'] for r in replay]==list(range(47,-1,-1))
        assert all(r['direction']==-1 and r['input_dtype']=='torch.bfloat16' for r in replay)
    assert all(r['input_dtype']=='torch.bfloat16' for r in executes)
    components=Counter(r['implementation'] for r in read(out/'components.json'))
    assert components['CPUBF16Experts']==48
    assert components['ChunkedDecoder' if opt.get('hyperconnection_chunking') else 'BF16Decoder']==48
    assert not any('Liger' in name for name in components)
    if bias:assert components['WindowIndexer' if opt.get('qsa_chunking') else 'DirectBiasIndexer']>0 and components['DirectBiasTextModel']==1
    else:assert not any('DirectBias' in name for name in components)
    operator_checks=bool(re.search(r'test_direct_bias_preserves_native_selection_and_sdpa[^\n]*\.\.\. ok',(job/'process.log').read_text()))
    if head=='cce_exact':assert operator_checks
    resources=read(out/'resources.json');phases={r['phase']:r for r in resources}
    for phase in ['forward','backward','gradient_export','gradient_comparison','initialization_verification']:
        row=phases[phase];assert row['seconds']>0 and row['samples']>0 and row['tree_pss_bytes']>0
    paired=None
    if enabled:
        paired=read(out/'chunking-comparison.json')
        assert paired['raw_tensors']==74472 and len(paired['tensors'])==74472
        assert paired['baseline_gradients_sha256'] is None and paired['raw_gradients_retained'] is False
        assert all(r['finite'] for r in paired['tensors'].values())
        assert paired['gradient_gate_passed']==(paired['global_relative_l2']<.01)
        assert result['chunking_gradient_gate_passed']==paired['gradient_gate_passed']
        assert sum(r['bitwise_equal'] for r in paired['tensors'].values())==paired['bitwise_equal_tensors']
        if 'reference_norm' in paired:
            norms=sum(r['reference_norm']**2 for r in paired['tensors'].values())
            errors=sum(r['reference_norm']**2*r['relative_l2']**2 if r['reference_norm'] else r['candidate_norm']**2 for r in paired['tensors'].values())
            assert math.isclose(paired['global_relative_l2'],math.sqrt(errors/norms),rel_tol=1e-9)
            assert math.isclose(paired['error_norm'],math.sqrt(errors),rel_tol=1e-9)
            assert all(math.isclose(r['candidate_norm'],summary[n]['norm'],rel_tol=1e-9,abs_tol=1e-12) for n,r in paired['tensors'].items())
        assert 'unchunked_control_forward' in phases and 'unchunked_control_backward' in phases
        assert 'test_chunked_quantized_experts_recompute_all_gradients' in (job/'process.log').read_text()
        if opt.get('qsa_chunking'):
            assert components['ChunkedAttention']>0
            chunks=read(out/'chunking.json')
            attentions=[r for n,r in chunks.items() if n.endswith('.self_attn')]
            assert attentions and all(r['max_query_tokens']<=8192 and r['key_tokens']==16249 for r in attentions)
        if opt.get('ple_chunking'):
            chunks=read(out/'chunking.json')
            ple=[r for n,r in chunks.items() if n.endswith('.ple')]
            assert len(ple)==1 and ple[0]['halo_tokens']==9
            assert ple[0]['max_window_tokens']<=8201
    report=dict(evidence_checks_passed=True,attempt=job.name,optimization=label,execution_commit=config['dispatch_commit'],
        loss=result['loss'],reference_loss=comparison['baseline_loss'],loss_relative_change=comparison['loss_relative_change'],
        loss_bitwise_equal=comparison['loss_bitwise_equal'],gradient_relative_l2=comparison['global_relative_l2'],gradient_cosine=comparison['cosine'],
        bitwise_equal_gradients=exact,gradient_tensors=74472,nonzero_reference_tensors=nonzero,
        bitwise_equal_nonzero_reference_gradients=comparison['bitwise_equal_nonzero_reference_tensors'],
        all_finite=True,initialization_bitwise_equal=True,raw_candidate_gradients_retained=False,optimizer_updates=0,
        numerical_bitwise_match=comparison['passed'],direct_bias_operator_checks_passed=operator_checks,acceptance_tolerance=None,resources=resources,
        scope='Metrics/source review; candidate raw gradients intentionally never archived. No numerical tolerance invented.',pushed_to_remote=False,
        numerical_profile=profile)
    if paired is not None:
        report.update(chunking_gradient_relative_l2=paired['global_relative_l2'],chunking_gradient_cosine=paired['cosine'],
            chunking_gradient_gate_passed=paired['gradient_gate_passed'],chunking_control_loss=paired['baseline_loss'],
            chunking_bitwise_equal_gradients=paired['bitwise_equal_tensors'],chunking_loss_bitwise_equal=paired['loss_bitwise_equal'],
            chunking_bitwise_equal_nonzero_gradients=paired['bitwise_equal_nonzero_reference_tensors'],chunking_gradient_limit=.01)
    (ROOT/'reports'/f'{label.lower()}-rerun.json').write_text(json.dumps(report,indent=2)+'\n')
    inventory={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(job.rglob('*')) if p.is_file() and p.name!='file-hashes.json'}
    (job/'file-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='resources'},indent=2))


if __name__=='__main__':main()
