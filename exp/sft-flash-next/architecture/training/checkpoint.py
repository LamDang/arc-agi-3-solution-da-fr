"""Bounded, acknowledged adapter/AdamW snapshots; no frozen model weights."""
import json
import os
from pathlib import Path
import random
import time

import torch

from .data import digest, file_hash


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Mailbox:
    """Pause at each shard until the desktop has durably acknowledged it."""
    def __init__(self, output, timeout=86400):
        self.root = Path(output)/'transfer'
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def __call__(self, checkpoint, path, kind):
        request = dict(checkpoint=checkpoint, path=path.name, kind=kind,
                       bytes=path.stat().st_size, sha256=file_hash(path))
        request['id'] = digest(request)
        atomic_json(self.root/'request.json', request)
        deadline = time.monotonic()+self.timeout
        while True:
            try:
                ack = json.loads((self.root/'ack.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                ack = {}
            if ack.get('id') == request['id'] and ack.get('sha256') == request['sha256']:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Checkpoint copy-back not acknowledged; last complete local checkpoint remains resumable')
            time.sleep(1)
        path.unlink()  # Bounded spool: only a verified local copy permits removal.


class RestoreMailbox:
    """Request one local checkpoint shard at a time through the same API."""
    def __init__(self, output, timeout=86400):
        self.root = Path(output)/'transfer'
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'restore-spool').mkdir(exist_ok=True)
        self.timeout = timeout

    def __call__(self, row):
        request = {k:row[k] for k in ('path','sha256','bytes')}
        request['id'] = digest(request)
        atomic_json(self.root/'restore-request.json', request)
        deadline = time.monotonic()+self.timeout
        while True:
            try:ready = json.loads((self.root/'restore-ready.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):ready = {}
            if ready.get('id') == request['id']:
                path = self.root/'restore-spool'/row['path']
                if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
                    raise ValueError('Restored checkpoint shard checksum differs')
                return path
            if time.monotonic() >= deadline:
                raise TimeoutError('Local resume shard not supplied')
            time.sleep(1)


def trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def rng_state():
    import numpy as np
    state = np.random.get_state()
    metadata = dict(python=random.getstate(), numpy=[state[0],state[1].tolist(),*state[2:]])
    tensors = {'rng/cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        tensors.update({f'rng/cuda-{i}': s for i,s in enumerate(torch.cuda.get_rng_state_all())})
    return metadata, tensors


def restore_rng(metadata, tensors):
    import numpy as np
    def tuples(x):
        return tuple(tuples(v) for v in x) if isinstance(x, list) else x
    random.setstate(tuples(metadata['python']))
    state = metadata['numpy']
    np.random.set_state((state[0],np.asarray(state[1], dtype=np.uint32),*state[2:]))
    torch.set_rng_state(tensors['rng/cpu'])
    cuda = [tensors[f'rng/cuda-{i}'] for i in range(len(tensors)-1)]
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError('CUDA RNG topology differs from checkpoint')
        torch.cuda.set_rng_state_all(cuda)


def save(model, optimizer, root, progress, identity, shard_bytes=64 << 20, transfer=None):
    """Weights stay fixed until the entire update snapshot is acknowledged."""
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    name = f"update-{progress['updates']:08d}"
    parameters = trainable(model)
    names = {id(p): n for n,p in parameters.items()}
    groups = [{**{k:v for k,v in group.items() if k != 'params'},
               'params': [names[id(p)] for p in group['params']]} for group in optimizer.param_groups]
    rng, rng_tensors = rng_state()
    states = {}
    def tensors():
        for n,p in parameters.items():
            yield 'adapter/'+n, p
            states[n] = []
            for key,value in optimizer.state.get(p, {}).items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError('Expected tensor AdamW state: '+key)
                states[n].append(key)
                yield 'optimizer/'+n+'/'+key, value
        yield from rng_tensors.items()
    rows, buffer, count, specs = [], {}, 0, {}
    def flush():
        nonlocal buffer, count
        if not buffer:
            return
        path = root/f'{len(rows):05d}.pt'
        torch.save(buffer, path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        row = dict(path=path.name, sha256=file_hash(path), bytes=path.stat().st_size, tensors=specs.copy())
        rows.append(row)
        if transfer:
            transfer(name, path, 'shard')
        buffer, count = {}, 0
        specs.clear()
    for key,value in tensors():
        size = value.numel()*value.element_size()
        if size > shard_bytes:
            raise ValueError('A parameter exceeds checkpoint shard budget: '+key)
        if count+size > shard_bytes:
            flush()
        buffer[key] = value.detach().to('cpu', copy=True).contiguous()
        specs[key] = dict(shape=list(value.shape), dtype=str(value.dtype))
        count += size
    flush()
    manifest = dict(version=1, checkpoint=name, shards=rows, progress=progress, identity=identity,
                    parameters=list(parameters), optimizer_groups=groups, optimizer_states=states,
                    rng=rng, rng_keys=list(rng_tensors))
    manifest['sha256'] = digest(manifest)
    path = root/'manifest.json'
    atomic_json(path, manifest)
    if transfer:
        transfer(name, path, 'manifest')
    return manifest


def validate(root, identity=None, verify_files=True):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('version') != 1 or digest({k:v for k,v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('Invalid checkpoint manifest checksum/version')
    if identity is not None and manifest['identity'] != identity:
        raise ValueError('Checkpoint dataset/model/config/source identity differs')
    seen = set()
    for row in manifest['shards']:
        if Path(row['path']).name != row['path']:
            raise ValueError('Unsafe checkpoint shard path')
        path = root/row['path']
        if verify_files and (path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']):
            raise ValueError('Incomplete or corrupt checkpoint shard')
        if seen.intersection(row['tensors']):
            raise ValueError('Duplicate checkpoint tensor')
        seen.update(row['tensors'])
    expected = {'adapter/'+n for n in manifest['parameters']} | set(manifest['rng_keys'])
    expected.update('optimizer/'+n+'/'+key for n,keys in manifest['optimizer_states'].items() for key in keys)
    if seen != expected:
        raise ValueError('Checkpoint tensor inventory differs')
    return manifest


def load(model, optimizer, root, identity, receive=None):
    """Validate every file first; restore one bounded shard at a time."""
    root = Path(root)
    manifest = validate(root, identity, verify_files=receive is None)
    parameters = trainable(model)
    if set(parameters) != set(manifest['parameters']):
        raise ValueError('Checkpoint adapter names differ')
    groups = manifest['optimizer_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError('Optimizer group count differs')
    names = {id(p): n for n,p in parameters.items()}
    for current,saved in zip(optimizer.param_groups, groups):
        if [names[id(p)] for p in current['params']] != saved['params']:
            raise ValueError('Optimizer parameter order differs')
        current.update({k:v for k,v in saved.items() if k != 'params'})
        current['betas'] = tuple(current['betas'])
    optimizer.state.clear()
    rng = {}
    with torch.no_grad():
        for row in manifest['shards']:
            path = receive(row) if receive else root/row['path']
            state = torch.load(path, map_location='cpu', weights_only=True)
            if set(state) != set(row['tensors']):
                raise ValueError('Shard tensor keys differ')
            for key,value in state.items():
                if dict(shape=list(value.shape), dtype=str(value.dtype)) != row['tensors'][key]:
                    raise ValueError('Shard tensor shape/dtype differs')
                if key.startswith('adapter/'):
                    p = parameters[key[len('adapter/'):]]
                    if p.shape != value.shape or p.dtype != value.dtype:
                        raise ValueError('Adapter shape/dtype differs')
                    p.copy_(value)
                elif key.startswith('optimizer/'):
                    n,field = key[len('optimizer/'):].rsplit('/', 1)
                    p = parameters[n]
                    # Non-capturable AdamW keeps its scalar step on CPU.
                    optimizer.state[p][field] = value if field == 'step' else value.to(p.device)
                else:
                    rng[key] = value
            del state
            if receive:path.unlink()
    restore_rng(manifest['rng'], rng)
    return manifest['progress']
