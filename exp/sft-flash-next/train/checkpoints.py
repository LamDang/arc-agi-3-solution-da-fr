"""Atomic local checkpoints, including in-flight accumulation and exact cursor."""
from __future__ import annotations

import os
import random
from pathlib import Path

import torch

from backend import adapter_state, load_adapter
from dataset import file_hash, read_json, write_json


def save(root, model, optimizer, identity, cursor, step, pending, keep_last=2):
    if keep_last < 2:
        raise ValueError('Retain at least two complete checkpoints')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-{cursor:08d}-{step:08d}.pt"
    target = root / name
    temp = root / (name + ".partial")
    state = dict(version=1, identity=identity, cursor=cursor, step=step, pending=pending,
        adapter=adapter_state(model), optimizer=optimizer.state_dict(),
        gradients={n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is not None},
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate())
    with open(temp, "wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = file_hash(temp)
    os.replace(temp, target)
    pointer = root / "latest.json.partial"
    write_json(pointer, dict(file=name, sha256=checksum))
    with open(pointer, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pointer, root / "latest.json")
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # Delete only old checkpoints after the new payload and pointer are durable.
    for old in sorted(root.glob('checkpoint-????????-????????.pt'))[:-keep_last]:
        old.unlink()
    return target


def load(root, model, optimizer, identity):
    root = Path(root)
    pointer = read_json(root / "latest.json")
    name = pointer["file"]
    if Path(name).name != name or file_hash(root / name) != pointer["sha256"]:
        raise ValueError("Incomplete/corrupt checkpoint")
    state = torch.load(root / name, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Resume identity changed (data, model, map, code, runtime, GPU or recipe)")
    load_adapter(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    params = dict(model.named_parameters())
    for name, gradient in state["gradients"].items():
        params[name].grad = gradient.to(params[name].device)
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state["cursor"], state["step"], state["pending"]
