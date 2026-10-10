"""Training memory telemetry and a last-checkpoint-preserving host-RAM guard."""
import json
import os
from pathlib import Path
import threading
import time

import torch


def host_memory():
    status = Path('/proc/self/status').read_text().splitlines()
    result = {}
    for name in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem'):
        result[name + '_gib'] = next((float(s.split()[1])/2**20 for s in status if s.startswith(name+':')), 0.)
    path = Path('/sys/fs/cgroup/memory.stat')
    if path.exists():
        values = dict(line.split() for line in path.read_text().splitlines())
        result.update(cgroup_anon_gib=int(values['anon'])/2**30, cgroup_file_gib=int(values['file'])/2**30,
                      cgroup_shmem_gib=int(values.get('shmem',0))/2**30,
                      cgroup_unevictable_gib=int(values.get('unevictable',0))/2**30)
        result['cgroup_noncache_gib'] = sum(int(values.get(k,0)) for k in
            ('anon','shmem','slab_unreclaimable','kernel_stack'))/2**30
    return result


class MemoryMonitor:
    def __init__(self, out, anon_limit_gib=160.):
        self.out = Path(out)
        self.limit = anon_limit_gib
        path = Path('/sys/fs/cgroup/memory.max')
        if path.exists() and path.read_text().strip() != 'max':
            self.limit = min(self.limit, .93*int(path.read_text())/2**30)
        self.stop = threading.Event()
        self.started = time.monotonic()

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        while not self.stop.wait(5):
            value = dict(seconds=time.monotonic()-self.started, **host_memory(),
                         gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                         gpu_reserved_gib=torch.cuda.memory_reserved()/2**30)
            with (self.out/'memory.jsonl').open('a') as stream:
                stream.write(json.dumps(value)+'\n')
            if value.get('cgroup_noncache_gib', value['RssAnon_gib']+value['RssShmem_gib']) > self.limit:
                report = dict(reason='host non-file-cache-memory guard', limit_gib=self.limit, **value,
                              recovery='Resume latest complete checkpoint; current request was not committed.')
                (self.out/'memory-guard.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(report), flush=True)
                os._exit(3)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)
