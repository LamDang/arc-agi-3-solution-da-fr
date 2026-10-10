"""Load-only RAM accounting. No forward, backward, PLE lookup or updates."""
import argparse
from collections import defaultdict
import ctypes
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from peft import get_peft_model_state_dict
from config import load_config
from model import build
from runtime.evidence import compare_initial, model_identity, sha, snapshot, snapshot_imports, write
from runtime.resources import Resources


def memory_snapshot(label, architecture=None):
    torch.cuda.synchronize()
    cgroup = Path('/sys/fs/cgroup')
    files = ['memory.current', 'memory.peak', 'memory.max', 'memory.events',
             'memory.events.local', 'memory.stat', 'memory.swap.max']
    usage = {name: (cgroup / name).read_text() for name in files if (cgroup / name).exists()}
    groups = defaultdict(lambda: defaultdict(int))
    group = None
    for line in Path('/proc/self/smaps').read_text().splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
            parts = line.split(maxsplit=5)
            group = parts[5] if len(parts) == 6 else '[anonymous]'
        else:
            match = re.match(r'^(Rss|Pss|Anonymous|Private_Dirty|Private_Clean|Shared_Dirty|Shared_Clean|Locked):\s+(\d+) kB$', line)
            if match:
                groups[group][match[1] + '_bytes'] += int(match[2]) * 1024
    host_stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    host = dict(host_stats()) if host_stats is not None else {'available': False}
    storages = {}
    if architecture is not None:
        tensors = list(architecture.model.named_parameters()) + list(architecture.model.named_buffers())
        tensors += [(f'expert-slab-{i}', slab) for i, slab in enumerate(architecture.expert_stager.slabs)]
        for name, tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            record = storages.setdefault(key, dict(device=str(tensor.device), bytes=storage.nbytes(),
                pinned=tensor.is_pinned() if tensor.device.type == 'cpu' else False,
                categories=set(), references=0))
            category = ('expert_slab' if name.startswith('expert-slab-') else
                        'adapter' if '.lora_' in name else 'frozen_model')
            record['categories'].add(category)
            record['references'] += 1
    totals = defaultdict(int)
    for record in storages.values():
        categories = '+'.join(sorted(record['categories']))
        totals[f"{record['device']}:{categories}:pinned={record['pinned']}"] += record['bytes']
    return dict(label=label, monotonic=time.monotonic(), pid=os.getpid(),
        cgroup=usage, smaps_rollup=Path('/proc/self/smaps_rollup').read_text(),
        smaps_by_path=dict(sorted(groups.items(), key=lambda row: row[1].get('Pss_bytes', 0), reverse=True)),
        host_allocator_stats=host, model_unique_storage_bytes=dict(totals),
        cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config = load_config(parser.parse_args().config, 'test')
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=False)
    assert config.optimizations.offload_routed_experts and config.optimizations.disk_ple
    torch.manual_seed(config.seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    snapshot(out, config, 'diagnostic-host-memory')
    actual = dict(sample=sha(config.samples[0]), **model_identity(config.model))
    assert all(actual.get(key) == digest for key, digest in config.expected_sha256.items())
    resources = Resources(out)
    architecture = None
    observations = []
    def observe(label):
        observations.append(memory_snapshot(label, architecture))
        write(out / 'host-memory.json', observations)
        print(json.dumps(dict(event=label, pid=os.getpid(),
            cgroup_current=observations[-1]['cgroup'].get('memory.current'))), flush=True)
    try:
        observe('before_loading')
        with resources.phase('loading'):
            architecture = build(config, out)
        observe('after_loading')
        gc.collect()
        observe('after_python_gc')
        initial = {name: parameter.detach().cpu()
                   for name, parameter in get_peft_model_state_dict(architecture.model).items()}
        with resources.phase('initialization_verification'):
            identity = compare_initial(initial, config.baseline)
        write(out / 'initial-comparison.json', identity)
        assert identity['passed']
        del initial
        gc.collect()
        observe('after_initialization_verification')
        # Diagnostic intervention only: return unused glibc arena pages, never
        # release live tensor storage. This is not installed in build/training.
        libc = ctypes.CDLL(None)
        trim = getattr(libc, 'malloc_trim', None)
        trimmed = None
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trimmed = trim(0)
        observe('after_diagnostic_malloc_trim')
        write(out / 'result.json', dict(mode='diagnostic-host-memory', completed=True,
            forward_executed=False, backward_executed=False, ple_prepared=False,
            optimizer_updates=0, raw_gradients_retained=False,
            malloc_trim_return=trimmed, diagnostic_intervention_only=True))
    finally:
        write(out / 'resources.json', resources.rows)
        snapshot_imports(out)
        if architecture is not None:
            architecture.expert_stager.close()
            write(out / 'expert-prefetch.json', architecture.expert_stager.report())


if __name__ == '__main__':
    main()
