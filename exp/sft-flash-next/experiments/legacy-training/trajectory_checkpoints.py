"""Versioned atomic trajectory checkpoints with raw summed-CE gradients."""
import os
import random
from pathlib import Path
import torch
from dataset import file_hash, read_json, write_json


def save(root, model, optimizer, identity, cursor, step, pending_supervised_tokens,
         pending_trajectories, pending_loss_sum):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f'trajectory-{cursor:08d}-{step:08d}.pt'
    temp = root / (name + '.partial')
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    state = dict(version=2, objective='all-assistant-token-weighted-ce-v1', identity=identity,
        cursor=cursor, step=step, pending_supervised_tokens=pending_supervised_tokens,
        pending_trajectories=pending_trajectories, pending_loss_sum=pending_loss_sum,
        parameters={n:p.detach().cpu().clone() for n,p in params.items()},
        gradients={n:p.grad.detach().cpu().clone() for n,p in params.items() if p.grad is not None},
        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate())
    with temp.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = file_hash(temp)
    os.replace(temp, root/name)
    pointer = root/'latest.json.partial'
    write_json(pointer, dict(file=name, sha256=checksum))
    with pointer.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(pointer, root/'latest.json')
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    for old in sorted(root.glob('trajectory-????????-????????.pt'))[:-2]:
        old.unlink()


def read_state(root, identity):
    root = Path(root)
    pointer = read_json(root/'latest.json')
    name = pointer['file']
    if Path(name).name != name or file_hash(root/name) != pointer['sha256']:
        raise ValueError('Incomplete or corrupt trajectory checkpoint')
    state = torch.load(root/name, map_location='cpu', weights_only=True)
    if state.get('version') != 2 or state.get('objective') != 'all-assistant-token-weighted-ce-v1':
        raise ValueError('Legacy request-mean checkpoints cannot resume trajectory training')
    if state['identity'] != identity:
        raise ValueError('Trajectory resume identity changed')
    provenance = dict(file=name,sha256=pointer['sha256'],cursor=state['cursor'],step=state['step'],
        pending_supervised_tokens=state['pending_supervised_tokens'],pending_trajectories=state['pending_trajectories'])
    return state,provenance


def restore_parameters(model, state):
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if params.keys() != state['parameters'].keys():
        raise ValueError('Trainable parameter keys changed')
    if not state['gradients'].keys() <= params.keys():
        raise ValueError('Unknown accumulated gradient')
    tokens, count = state['pending_supervised_tokens'], state['pending_trajectories']
    if tokens < 0 or count < 0 or bool(tokens) != bool(count):
        raise ValueError('Invalid pending token/trajectory counts')
    if not tokens and state['gradients']:
        raise ValueError('Unexpected gradients at optimizer boundary')
    with torch.no_grad():
        for n,p in params.items():
            value = state['parameters'][n]
            if value.shape != p.shape or value.dtype != p.dtype:
                raise ValueError('Trainable parameter shape/dtype changed: '+n)
            p.copy_(value)
            p.grad = state['gradients'][n].to(p.device) if n in state['gradients'] else None


def load(root, model, optimizer, identity):
    state,_ = read_state(root,identity)
    restore_parameters(model,state)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return (state['cursor'], state['step'], state['pending_supervised_tokens'],
            state['pending_trajectories'], state['pending_loss_sum'])


def load_for_evaluation(root, model, identity):
    state,provenance = read_state(root,identity)
    restore_parameters(model,state)
    # Evaluation uses the current trained weights, never applies a partial sum.
    model.zero_grad(set_to_none=True)
    return provenance
