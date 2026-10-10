"""Three no-update all-adapter passes for an independent multi-span native gate.

The sample/labels must be explicit artifacts from the separate trajectory
encoder. This never modifies the preserved legacy final-reply reference.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import os
from concurrent.futures import ThreadPoolExecutor
import torch
from dataset import file_hash, write_json
from native_gradient_compare import compare
from trajectory_gradient_gate import source_hashes
from trajectory_head_loss import supervised_positions, trajectory_objective
from trajectory_run import load_model, storage_for
from native_optimization_flags import apply_flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model','sample','labels','adapter-state','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--flags',default='{}')
    p.add_argument('--head-backward-rows',type=int,default=None)
    p.add_argument('--disk-dir',default='/tmp/trajectory-audit-activations')
    p.add_argument('--disk-budget-gib',type=float,default=32.)
    args = p.parse_args()
    flags = json.loads(args.flags)
    if any(k in flags for k in ['loss','name','head_backward_rows']):
        raise ValueError('Keep the multi-span objective separate from legacy flags')
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    batch = torch.load(args.sample,map_location='cpu',weights_only=True)
    labels = torch.load(args.labels,map_location='cpu',weights_only=True)
    if labels.shape != batch['input_ids'].shape:
        raise ValueError('Labels/sample shape differs')
    positions,targets = supervised_positions(labels)
    if not torch.equal(batch['input_ids'][0,positions+1],targets):
        raise ValueError('Trajectory labels differ from actual next-token IDs')
    if positions.numel() < 2 or not bool((positions[1:]-positions[:-1] > 1).any()):
        raise ValueError('Qualification sample must include disjoint assistant spans')
    if not bool((labels[0] == -100).any()):
        raise ValueError('Qualification sample must include ignored observations')
    shutil.copyfile(args.sample,out/'sample.pt')
    shutil.copyfile(args.labels,out/'labels.pt')
    shutil.copyfile(args.adapter_state,out/'adapter.pt')
    identity = dict(objective='native-all-assistant-summed-ce-v1',flags=flags,
        head_backward_rows=args.head_backward_rows,deterministic=True,
        model_config_sha256=file_hash(Path(args.model)/'config.json'),
        source_sha256=source_hashes(),artifacts={},tokens=labels.shape[1],targets=targets.numel())
    model_root = Path(args.model)
    index = json.loads((model_root/'model.safetensors.index.json').read_text())
    names = sorted({'config.json','model.safetensors.index.json',*index['weight_map'].values()})
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise ValueError('Unsafe checkpoint shard path')
    with ThreadPoolExecutor(max_workers=4) as pool:
        identity['model_sha256'] = dict(zip(names,pool.map(lambda n:file_hash(model_root/n),names)))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Require native deterministic CUBLAS workspace setting')
    import importlib.metadata
    identity['runtime'] = {n:importlib.metadata.version(n) for n in ['torch','transformers','peft','auto-round']}
    identity.update(cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    write_json(out/'identity.json',identity)
    torch.manual_seed(20261009)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    model = load_model(args.model,args.adapter_state)
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(parameters) != 744:
        raise ValueError('Expected every one of 744 LoRA tensors')
    batch = {k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
             for k,v in batch.items()}
    labels = labels.to('cuda')
    cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state_all()
    results = []
    reference = None
    for name in ['native-1','native-2','candidate']:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        started = time.monotonic()
        choice = flags if name == 'candidate' else {}
        with apply_flags(model.get_base_model(),choice),storage_for(model,choice,args):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if name == 'candidate':
                    loss = trajectory_objective(model,batch,labels,args.head_backward_rows,'sum')
                else:
                    loss = model(**batch,labels=labels,use_cache=False,num_items_in_batch=1).loss
            value = loss.item()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite native trajectory loss')
            loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
            raise ValueError('Missing or nonfinite trajectory adapter gradient')
        gradients = {n:p.grad.detach().cpu().clone() for n,p in parameters.items()}
        torch.save(gradients,out/(name+'.pt'))
        report = dict(name=name,loss=value,seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        if reference is None:
            reference = gradients
        else:
            comparison = compare(gradients,reference)
            write_json(out/(name+'-comparison.json'),comparison)
            report.update({k:v for k,v in comparison.items() if k != 'per_parameter'})
        results.append(report)
        write_json(out/'results.json',results)
        print(json.dumps(report),flush=True)
        if name == 'native-2' and not report['bitwise_equal']:
            raise RuntimeError('Multi-span native reference is not deterministic')
    for name in ['sample.pt','labels.pt','adapter.pt','native-1.pt','native-2.pt','candidate.pt','results.json']:
        identity['artifacts'][name] = file_hash(out/name)
    write_json(out/'identity.json',identity)
    passed = all(row.get('bitwise_equal',True) and row['loss']==results[0]['loss'] for row in results)
    write_json(out/'qualification.json',dict(completed=True,bitwise_equal=passed,
        strict_tolerance=results[-1]['within_tolerance'],optimizer_updates=0))
    if not passed:
        raise RuntimeError('Multi-span native gate failed; preserve evidence and do not promote')


if __name__ == '__main__':
    main()
