"""Native HF trajectory training with whole-trajectory token-budget updates.

Production is gated independently from the legacy final-reply reference. Every
backward contributes summed assistant CE; accumulated gradients are divided by
the exact actual supervised count before clipping and updating. A trajectory is
never split, padded with duplicate targets, or truncated to meet a budget.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import resource
import signal
import time
import os
import torch
from dataset import digest, file_hash, read_json, write_json
from trajectory_data import Trajectories, plan_token_updates
from trajectory_training import labels_for_annotation, finish_token_update, validate_clean_initial_adapter
from trajectory_head_loss import trajectory_objective
from trajectory_gradient_gate import verify_gradient_gate, source_hashes
import trajectory_checkpoints


def load_model(path, adapter_state):
    from transformers import AutoModelForImageTextToText, AutoRoundConfig
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from overfit_hf_reference import TARGETS, resident_device_map
    base, loading = AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,
        device_map=resident_device_map(path),local_files_only=True,trust_remote_code=False,
        output_loading_info=True,quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
    if base.config.text_config.num_experts != 256 or any(loading.get(k) for k in
            ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Require exact native 256-expert checkpoint loading')
    for module in base.modules():
        for name,value in module._buffers.items():
            if value is not None and value.device.type != 'cuda':
                module._buffers[name] = value.to('cuda')
    for name,value in base.named_parameters():
        expected = 'cpu' if 'ple_embedding.ngram_embedding.weight' in name else 'cuda:0'
        if str(value.device) != expected:
            raise ValueError('Unexpected native parameter placement: '+name)
    base.requires_grad_(False)
    model = get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',
                                         target_modules=TARGETS,task_type='CAUSAL_LM'))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if adapter_state is not None:
        state = torch.load(adapter_state,map_location='cpu',weights_only=True)
        if state.keys() != get_peft_model_state_dict(model).keys():
            raise ValueError('Adapter keys differ')
        set_peft_model_state_dict(model,state)
        for name,value in get_peft_model_state_dict(model).items():
            torch.testing.assert_close(value.cpu(),state[name],rtol=0,atol=0)
    return model


def evaluation_provenance(model, checkpoint_dir, identity):
    if not checkpoint_dir:
        raise ValueError('Evaluation requires an explicit trained trajectory checkpoint')
    provenance = trajectory_checkpoints.load_for_evaluation(checkpoint_dir,model,identity)
    if provenance['step'] < 1:
        raise ValueError('Checkpoint has no completed training update')
    return provenance


def storage_for(model, flags, args):
    if flags.get('offload'):
        from offload import ActivationOffload
        return ActivationOffload(model,disk_dir=args.disk_dir if flags['offload']=='disk' else None,
            disk_budget_gib=args.disk_budget_gib if flags['offload']=='disk' else 0,prefetch=2)
    return torch.autograd.graph.save_on_cpu(pin_memory=False)


def backward_trajectory(model, bundle, row, flags, args):
    enc, annotation = bundle.get(row)
    batch = {k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
             for k,v in enc.items()}
    labels = labels_for_annotation(batch['input_ids'],annotation)
    with storage_for(model,flags,args):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss = trajectory_objective(model,batch,labels,args.head_backward_rows,'sum')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite trajectory summed CE')
        loss.backward()
    return float(loss.detach()), annotation['target_tokens']


def qualify_capacity(model, bundle, rows, flags, args, identity):
    """Disposable real-trajectory backwards/update, preserving full contexts."""
    ids = {max(rows,key=lambda row:row[field])['sample_id']
           for field in ['total_tokens','target_tokens','images']}
    selected = [row for row in rows if row['sample_id'] in ids]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr,weight_decay=0.)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    measured = []
    targets = 0
    for row in selected:
        loss,count = backward_trajectory(model,bundle,row,flags,args)
        targets += count
        measured.append(dict(sample_id=row['sample_id'],loss_sum=loss,supervised_tokens=count,
            total_tokens=row['total_tokens'],images=row['images'],
            host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print(json.dumps(measured[-1]),flush=True)
    norm = finish_token_update(model,optimizer,targets,max_norm=args.max_grad_norm,lr=args.lr)
    torch.cuda.synchronize()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    peak = torch.cuda.max_memory_reserved()
    report = dict(identity=identity,passed=peak+args.gpu_headroom_gib*2**30 < total_memory,
        measured=measured,actual_supervised_tokens=targets,trajectories=len(selected),gradient_norm=norm,
        peak_reserved_bytes=peak,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        observed_gpu_headroom_gib=(total_memory-peak)/2**30,required_gpu_headroom_gib=args.gpu_headroom_gib,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seconds=time.monotonic()-started,optimizer_updates=1,updates_discarded=True,
        full_length_gradient_equivalence_proven=False)
    write_json(Path(args.out)/'capacity.json',report)
    if not report['passed']:
        raise RuntimeError('Measured trajectory capacity did not retain required GPU headroom')


def train(model, optimizer, bundle, rows, plans, flags, args, identity):
    out = Path(args.out)
    by_id = {row['sample_id']:row for row in rows}
    ordered = [sid for plan in plans for sid in plan['sample_ids']]
    boundaries = {}
    count = 0
    for plan in plans:
        count += len(plan['sample_ids'])
        boundaries[count] = plan
    cursor = step = pending_tokens = pending_trajectories = 0
    pending_loss = 0.
    if args.resume:
        cursor,step,pending_tokens,pending_trajectories,pending_loss = trajectory_checkpoints.load(
            out,model,optimizer,identity)
    elif (out/'latest.json').exists():
        raise ValueError('Output has a checkpoint; use resume')
    if cursor < 0 or cursor > len(ordered):
        raise ValueError('Checkpoint cursor exceeds immutable update plan')
    completed = [end for end in boundaries if end <= cursor]
    previous = max(completed,default=0)
    expected_tokens = sum(by_id[sid]['target_tokens'] for sid in ordered[previous:cursor])
    if step != len(completed) or pending_trajectories != cursor-previous or pending_tokens != expected_tokens:
        raise ValueError('Checkpoint partial accumulation differs from immutable trajectory plan')
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    started = time.monotonic()
    initial_step = step
    while cursor < len(ordered):
        row = by_id[ordered[cursor]]
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        loss,targets = backward_trajectory(model,bundle,row,flags,args)
        pending_tokens += targets
        pending_trajectories += 1
        pending_loss += loss
        cursor += 1
        update = None
        if cursor in boundaries:
            plan = boundaries[cursor]
            if pending_tokens != plan['target_tokens'] or pending_trajectories != len(plan['sample_ids']):
                raise ValueError('Actual update counts differ from immutable plan')
            norm = finish_token_update(model,optimizer,pending_tokens,max_norm=args.max_grad_norm,lr=args.lr)
            step += 1
            update = dict(step=step,actual_supervised_tokens=pending_tokens,
                trajectories=pending_trajectories,token_budget=args.token_budget,
                overshoot_tokens=plan['overshoot_tokens'],final_short_batch=plan['tail'],
                token_mean_loss=pending_loss/pending_tokens,gradient_norm=norm)
            pending_tokens = pending_trajectories = 0
            pending_loss = 0.
        trajectory_checkpoints.save(out,model,optimizer,identity,cursor,step,pending_tokens,
                                    pending_trajectories,pending_loss)
        metrics = dict(cursor=cursor,step=step,sample_id=row['sample_id'],loss_sum=loss,
            supervised_tokens=targets,total_tokens=row['total_tokens'],images=row['images'],
            pending_supervised_tokens=pending_tokens,pending_trajectories=pending_trajectories,
            update=update,seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with (out/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if stop[0] or (args.max_updates and step-initial_step >= args.max_updates) or (
                args.session_minutes and time.monotonic()-started >= args.session_minutes*60):
            break
    write_json(out/'status.json',dict(complete=cursor==len(ordered),cursor=cursor,step=step,
        pending_supervised_tokens=pending_tokens,pending_trajectories=pending_trajectories))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','qualify','train','evaluate'],required=True)
    for name in ['model','data','adapter-state','gradient-gate','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--capacity-report',help='Measured actual-trajectory capacity report required for training')
    p.add_argument('--adapter-provenance',required=True,help='Clean zero-update production initialization metadata')
    p.add_argument('--checkpoint-dir',help='Required trained v2 trajectory checkpoint for evaluation')
    p.add_argument('--flags',default='{}',help='JSON object of individually qualified native flags')
    p.add_argument('--head-backward-rows',type=int,default=None)
    p.add_argument('--token-budget',type=int,default=16384)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--disk-dir',default='/tmp/trajectory-activations')
    p.add_argument('--disk-budget-gib',type=float,default=32.)
    p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--session-minutes',type=float,default=0.)
    p.add_argument('--gpu-headroom-gib',type=float,default=4.)
    args = p.parse_args()
    if args.epochs < 1:
        p.error('epochs must be positive')
    if args.mode=='evaluate' and not args.checkpoint_dir:
        p.error('evaluate requires --checkpoint-dir; initial-adapter evaluation is not implicit')
    flags = json.loads(args.flags)
    if 'loss' in flags or 'name' in flags or 'head_backward_rows' in flags:
        raise ValueError('Trajectory loss and backward rows are separate identity fields')
    bundle = Trajectories(args.data)
    rows = bundle.split('train')
    plans = []
    for epoch in range(args.epochs):
        # Lexicographic, reproducible whole trajectories; flush final short update each epoch.
        for plan in plan_token_updates(rows,args.token_budget):
            plans.append(dict(plan,epoch=epoch))
    if not rows:
        raise ValueError('No training trajectories')
    if args.head_backward_rows is not None and any(row['target_tokens'] > args.head_backward_rows
            for row in bundle.rows):
        raise ValueError('Explicit backward rows cannot cover every actual assistant target count')
    gate = verify_gradient_gate(args.gradient_gate,flags,args.head_backward_rows)
    initial = validate_clean_initial_adapter(args.adapter_state,args.adapter_provenance)
    from run import check_model
    hashes = check_model(args.model,bundle)
    if hashes['config.json'] != gate['model_config_sha256']:
        raise ValueError('Gradient gate used a different model')
    if initial['model_config_sha256'] != hashes['config.json']:
        raise ValueError('Production initial adapter belongs to a different model')
    if {k:v for k,v in hashes.items() if k!='expert_map'} != gate['model_sha256']:
        raise ValueError('Gradient gate model weights/index identity differs')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python startup')
    import importlib.metadata
    identity = dict(objective='all-assistant-token-weighted-ce-v1',dataset=bundle.manifest['sha256'],
        model=hashes,adapter_sha256=file_hash(args.adapter_state),flags=flags,
        clean_initial_provenance_sha256=file_hash(args.adapter_provenance),
        head_backward_rows=args.head_backward_rows,update_plan=digest(plans),epochs=args.epochs,
        token_budget=args.token_budget,lr=args.lr,max_grad_norm=args.max_grad_norm,
        source_sha256=source_hashes(),gradient_gate_sha256=file_hash(Path(args.gradient_gate)/'identity.json'),
        runtime={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','auto-round']},
        deterministic=True,disk_budget_gib=args.disk_budget_gib,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,required_gpu_headroom_gib=args.gpu_headroom_gib)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'identity.json',identity)
    write_json(out/'update_plan.json',plans)
    if args.mode == 'preflight':
        return
    torch.manual_seed(20261009)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    model = load_model(args.model,args.adapter_state)
    from native_optimization_flags import apply_flags
    with apply_flags(model.get_base_model(),flags):
        if args.mode == 'qualify':
            qualify_capacity(model,bundle,rows,flags,args,identity)
        elif args.mode == 'train':
            if not args.capacity_report:
                raise ValueError('Training requires actual-trajectory memory qualification')
            capacity = read_json(args.capacity_report)
            if capacity.get('identity') != identity or capacity.get('passed') is not True:
                raise ValueError('Trajectory capacity identity differs')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=args.lr,weight_decay=0.)
            train(model,optimizer,bundle,rows,plans,flags,args,identity)
        else:
            # Same independently qualified all-span CE; no optimizer.
            checkpoint = evaluation_provenance(model,args.checkpoint_dir,identity)
            model.eval()
            total_loss = 0.
            total_targets = 0
            results = []
            with torch.no_grad():
                for row in bundle.split('validation'):
                    enc,ann = bundle.get(row)
                    batch = {k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                             for k,v in enc.items()}
                    labels = labels_for_annotation(batch['input_ids'],ann)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss = trajectory_objective(model,batch,labels,args.head_backward_rows,'sum')
                    total_loss += loss.item()
                    total_targets += ann['target_tokens']
                    results.append(dict(sample_id=row['sample_id'],loss_sum=loss.item(),target_tokens=ann['target_tokens']))
            if not total_targets:
                raise ValueError('No held-out assistant targets')
            write_json(out/'evaluation.json',dict(identity=identity,fold=bundle.manifest['validation_fold'],
                token_mean_loss=total_loss/total_targets,supervised_tokens=total_targets,rows=results,
                trained_checkpoint=checkpoint))


if __name__ == '__main__':
    main()
