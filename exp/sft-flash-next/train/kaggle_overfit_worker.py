"""Kaggle API worker for a bounded BF16 one-sample learning test."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle_reference_worker import sha256,timestamp,event_rows,summarize as summarize_capture


def validate(config):
    from kaggle_reference_worker import validate as validate_capture
    # Same immutable model, sample, bootstrap and source hashes. The legacy
    # gradient snapshot remains provenance, not a learning acceptance gate.
    validate_capture({**config,'timeout_seconds':1200})
    assert config['purpose']=='diagnostic-single-sample-overfit'
    assert config['timeout_seconds']==9000
    assert config['learning']==dict(initialization='seeded-random-A-zero-B',seed=20261009,
        optimizer='torch.optim.AdamW',learning_rate=.0002,weight_decay=0,clip_grad_norm=1,
        max_updates=20,target_loss_ratio=.05,
        acceptance='finite-learning-and-loss-reduction; gradient equality not required')
    return {k:config[k] for k in ('model','sample','bootstrap','reference_script')}


def native_command(config,bootstrap):
    learning=config['learning']
    return ['/usr/bin/python3',str(bootstrap),'--model',config['model'],'--sample',config['sample'],
        '--prompt-tokens',str(config['prompt_tokens']),'--checkpointing','--save-on-cpu','--deterministic',
        '--learning-rate',str(learning['learning_rate']),'--max-steps',str(learning['max_updates']),
        '--target-ratio',str(learning['target_loss_ratio']),'--out',config['remote_output']]


def bootstrap_source(config):
    # Spawned DataLoader imports cannot launch another model run.
    source="""import json,hashlib,runpy,sys
from pathlib import Path
sys.path=[p for p in sys.path if p!='/usr/local/lib/python3.13/dist-packages']
sys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao',
'/kaggle/working/training-gradient-audit/exp/sft-flash-next/train',%(runtime)r]
c=json.loads(Path(%(config)r).read_text())
from cce_target_loss import install_for_capture
install_for_capture(%(head)r)
from cce_operator_check import qualify
qualify(c['model'],c['sample'],c['remote_launch'])
from opt3_mask_operator_check import qualify as qualify_mask
qualify_mask(c['remote_launch'])
from opt3_mask_capture import install_for_capture as install_mask
install_mask(%(mask)r)
from opt4_ple_operator_check import qualify as qualify_ple
qualify_ple(c['model'],c['sample'],c['remote_launch'])
from opt4_ple_capture import install_for_capture as install_ple
finalize_ple=install_ple(c['model'],c['sample'],c['remote_launch'],%(ple)r,lookahead=2)
from bf16_model_activations import qualify_cpu,install_for_capture as install_bf16
qualify_cpu(c['remote_launch'])
finalize_bf16=install_bf16(None,c['remote_launch'],%(precision)r,training=True)
# Exercise fresh-mode loading before the expensive native model load.
import torch
probe=Path(c['remote_launch'])/'fresh-load-check.pt'
value=torch.tensor([1.0007,-0.31317],dtype=torch.float32)
torch.save(value,probe)
loaded=torch.load(probe,weights_only=True)
assert loaded.dtype==torch.float32 and torch.equal(loaded,value)
(Path(c['remote_launch'])/'fresh-initialization-precheck.json').write_text(json.dumps(dict(passed=True,no_adapter_load=True,fp32_input_preserved=True)))
from bf16_overfit_instrumentation import install
finalize_learning=install(c)
succeeded=False
try:
    runpy.run_path(c['bootstrap'],run_name='__main__')
    succeeded=True
finally:
    updates=finalize_learning()
    finalize_ple()
    pp=Path(%(ple)r);pr=json.loads(pp.read_text())
    pr['optimizer_updates']=updates
    pr['diagnostic_source']='one pre-encoded anchor reused for all overfit steps; two duplicate lookahead preparations'
    pr['current_retained_through_backward']=len({row['prepared_cpu_storage_pointer'] for row in pr['calls']})==1 and len(pr['calls'])==2*updates+1
    pp.write_text(json.dumps(pr,indent=2)+'\\n')
    if succeeded:finalize_bf16(optimizer_updates=updates)
    imported={}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and name.endswith('.py') and (name.startswith('/kaggle/working/training-gradient-audit/') or name.startswith('/tmp/peft-autoround-compat/') or name.startswith(%(runtime_prefix)r)):
            file=Path(name)
            if file.is_file():
                with file.open('rb') as stream:imported[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    Path(%(imports)r).write_text(json.dumps(imported,indent=2)+'\\n')
"""
    launch=Path(config['remote_launch']);out=Path(config['remote_output'])
    source=source%dict(runtime=str(launch/'cce-runtime'/config['cce_dependency']['archive_prefix']),
        config=config['remote_config'],head=str(out/'target-head.json'),mask=str(out/'attention-mask.json'),
        ple=str(out/'ple-preparation.json'),precision=str(out/'bf16-policy.json'),
        runtime_prefix=str(launch/'cce-runtime')+'/',imports=str(launch/'imported-source-hashes.json'))
    import textwrap
    return "if __name__ == '__main__':\n"+textwrap.indent(source,'    ')


def phase_for(timing,monotonic_ns):
    transitions={'load_start':'loading','load_complete':'forward_setup','forward_start':'forward',
        'loss':'pre_backward_export','backward_start':'pure_backward','backward_call_end':'gradient_export',
        'gradients_saved':'clip','clip_start':'clip','clip_end':'optimizer_setup',
        'optimizer_start':'optimizer','optimizer_end':'optimizer_audit','optimizer_audit_end':'step_end','finished':'finished'}
    phase='startup'
    for row in timing:
        if row['monotonic_ns']<=monotonic_ns and row['event'] in transitions:phase=transitions[row['event']]
    return phase


def summarize(timing,telemetry,allocator,total_seconds):
    steps=sorted({row['step'] for row in timing if row.get('step') is not None})
    result={'total_seconds':total_seconds,'steps':[]}
    for step in steps:
        marks=[r for r in timing if r.get('step')==step]
        samples=[r for r in telemetry if r.get('step')==step]
        peaks=[r for r in allocator if r.get('step')==step]
        row=summarize_capture(marks,samples,peaks,None);row['step']=step
        lookup={r['event']:r['monotonic_ns'] for r in marks}
        for label,start,end in [('optimizer_seconds','optimizer_start','optimizer_end'),('clip_seconds','clip_start','clip_end')]:
            row[label]=(lookup[end]-lookup[start])/1e9 if start in lookup and end in lookup else None
        result['steps'].append(row)
    result['startup_and_loading']=summarize_capture([r for r in timing if r.get('step') is None],
        [r for r in telemetry if r.get('step') is None],[r for r in allocator if r.get('step') is None],None)
    return result


def launch(config_path):
    config = json.loads(Path(config_path).read_text())
    paths = validate(config)
    output = Path(config['remote_output'])
    launch_dir = Path(config['remote_launch'])
    if output.exists() or launch_dir.exists():
        raise FileExistsError('Run output or launch directory already exists; refusing duplicate')
    active = subprocess.run(['pgrep', '-af', 'reference-one-run-bootstrap|instrumented-bootstrap'],
                            capture_output=True, text=True).stdout.strip()
    if active:
        raise RuntimeError('Native reference process already running')
    gpu_processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if gpu_processes:
        raise RuntimeError('GPU compute processes already running; refusing concurrent capture')
    launch_dir.mkdir(parents=True)
    # The archived worker is executed from this directory by the detached monitor.
    # Freeze its shared supervision helper beside it so imports stay self-contained.
    helper=Path(config['reference_script']).parent/'kaggle_reference_worker.py'
    (launch_dir/'kaggle_reference_worker.py').write_bytes(helper.read_bytes())
    if config.get('objective') == 'liger_target_flce':
        import zipfile
        wheel = Path(config['liger_dependency']['remote_wheel'])
        (launch_dir / wheel.name).write_bytes(wheel.read_bytes())
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(launch_dir / 'liger-runtime')
    if config.get('objective') in ('cce_target_exact', 'cce_opt3_mask', 'cce_opt4_ple', 'cce_opt5_bf16'):
        import tarfile
        source = Path(config['cce_dependency']['remote_archive'])
        (launch_dir / source.name).write_bytes(source.read_bytes())
        with tarfile.open(source) as archive:
            archive.extractall(launch_dir / 'cce-runtime', filter='data')
    instrumented = launch_dir / 'instrumented-bootstrap.py'
    instrumented.write_text(bootstrap_source(config))
    command = native_command(config, instrumented)
    launch_record = {'run_id': config['run_id'], 'command': command,
                     'environment': config['environment'],
                     'expected_sha256': config['expected_sha256'],
                     'validated_paths': paths, 'started_utc': timestamp(),
                     'instrumented_bootstrap_sha256': sha256(instrumented),
                     'worker_sha256': sha256(__file__),
                     'reference_script_commit': config['reference_script_commit'],
                     'no_optimizer_update': False, 'no_clipping': False,'learning_config': config['learning']}
    (launch_dir / 'launch.json').write_text(json.dumps(launch_record, indent=2) + '\n')
    (launch_dir / 'reference.py').write_bytes(Path(config['reference_script']).read_bytes())
    (launch_dir / 'worker.py').write_bytes(Path(__file__).read_bytes())
    for name in config.get('candidate_sources_sha256', {}):
        (launch_dir / ('candidate-' + name)).write_bytes(
            (Path(config['reference_script']).parent / name).read_bytes())
    for label, source in [('sample.pt', config['sample']), ('bootstrap.py', config['bootstrap']),
                          ('model-config.json', Path(config['model']) / 'config.json')]:
        (launch_dir / label).write_bytes(Path(source).read_bytes())
    with (launch_dir / 'launcher.log').open('w') as log:
        process = subprocess.Popen(['/usr/bin/python3', str(launch_dir / 'worker.py'),
                                    'monitor', str(config_path)], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (launch_dir / 'monitor-pid').write_text(f'{process.pid}\n')
    print(json.dumps({'run_id': config['run_id'], 'monitor_pid': process.pid,
                      'output_absent_at_launch': not output.exists()}))


def monitor(config_path):
    import psutil

    config = json.loads(Path(config_path).read_text())
    launch_dir = Path(config['remote_launch'])
    output = Path(config['remote_output'])
    command = json.loads((launch_dir / 'launch.json').read_text())['command']
    environment = os.environ.copy()
    environment.update(config['environment'])
    start = time.monotonic()
    with (launch_dir / 'process.log').open('w') as log, (launch_dir / 'telemetry.jsonl').open('w') as telemetry:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment)
        (launch_dir / 'child-pid').write_text(f'{process.pid}\n')
        timed_out = False
        while process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed >= config['timeout_seconds']:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            try:
                rss = psutil.Process(process.pid).memory_info().rss
            except psutil.Error:
                rss = 0
            host = psutil.virtual_memory().used
            gpu = utilization = 0
            try:
                reply = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], capture_output=True,
                                       text=True, timeout=5)
                gpu, utilization = [int(x.strip()) for x in reply.stdout.splitlines()[0].split(',')]
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            timing = event_rows(launch_dir / 'timing-events.jsonl')
            phase = phase_for(timing, time.monotonic_ns())
            row = {'elapsed_seconds': elapsed, 'phase': phase, 'step': timing[-1].get('step') if timing else None,
                   'process_rss_bytes': rss, 'host_used_bytes': host,
                   'gpu_used_mib': gpu, 'gpu_utilization_percent': utilization}
            if config.get('objective') in ('cce_opt4_ple', 'cce_opt5_bf16'):
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    row['children_rss_bytes'] = sum(child.memory_info().rss for child in children)
                    row['process_tree_pss_bytes'] = sum(child.memory_full_info().pss for child in [parent] + children)
                except (psutil.Error, AttributeError):
                    row['children_rss_bytes'] = row['process_tree_pss_bytes'] = None
                stat = Path('/sys/fs/cgroup/memory.stat')
                if stat.exists():
                    counts = dict(line.split() for line in stat.read_text().splitlines())
                    row['cgroup_anon_bytes'] = int(counts.get('anon', 0))
                    row['cgroup_file_bytes'] = int(counts.get('file', 0))
                    row['cgroup_current_bytes'] = int(Path('/sys/fs/cgroup/memory.current').read_text())
            telemetry.write(json.dumps(row) + '\n')
            telemetry.flush()
            time.sleep(2)
    total = time.monotonic() - start
    result = {'run_id': config['run_id'], 'pid': process.pid,
              'returncode': process.wait(), 'timed_out': timed_out,
              'elapsed_seconds': total, 'finished_utc': timestamp()}
    timing = event_rows(launch_dir / 'timing-events.jsonl')
    telemetry_rows = event_rows(launch_dir / 'telemetry.jsonl')
    allocator = event_rows(launch_dir / 'allocator-phase-peaks.jsonl')
    (launch_dir / 'timing-summary.json').write_text(
        json.dumps(summarize(timing, telemetry_rows, allocator, total), indent=2) + '\n')
    (launch_dir / 'monitor.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'launch', 'monitor'])
    parser.add_argument('config')
    args = parser.parse_args()
    if args.action == 'validate':
        config = json.loads(Path(args.config).read_text())
        print(json.dumps({'preflight': 'ok', 'run_id': config['run_id'],
                          'validated_paths': validate(config)}))
    elif args.action == 'launch':
        launch(args.config)
    else:
        monitor(args.config)


if __name__ == '__main__':
    main()
