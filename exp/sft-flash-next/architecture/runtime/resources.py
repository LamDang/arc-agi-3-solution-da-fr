"""Per-phase synchronized CUDA peaks and sampled host RAM; no arithmetic hooks."""
from contextlib import contextmanager
import os
import json
from pathlib import Path
import threading
import time

import psutil
import torch


class Resources:
    def __init__(self, output=None):
        self.rows = []
        self.output=Path(output) if output is not None else None

    def persist(self,name,value):
        if self.output is None:return
        path=self.output/name;temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stopped = threading.Event()
        peak = dict(rss_bytes=0,tree_pss_bytes=0,children_rss_bytes=0,host_used_bytes=0,samples=0,
            cgroup_memory_current_bytes=0,cgroup_anon_bytes=0,cgroup_file_bytes=0,cgroup_shmem_bytes=0,
            host_pinned_allocated_bytes=0)
        process = psutil.Process(os.getpid())
        started=None
        def sample():
            peak['samples'] += 1
            peak['rss_bytes'] = max(peak['rss_bytes'],process.memory_info().rss)
            children = process.children(recursive=True)
            peak['children_rss_bytes'] = max(peak['children_rss_bytes'],sum(p.memory_info().rss for p in children if p.is_running()))
            pss = sum(getattr(p.memory_full_info(),'pss',0) for p in [process,*children] if p.is_running())
            peak['tree_pss_bytes'] = max(peak['tree_pss_bytes'],pss)
            peak['host_used_bytes'] = max(peak['host_used_bytes'],psutil.virtual_memory().used)
            cgroup=Path('/sys/fs/cgroup')
            if (cgroup/'memory.current').exists():
                peak['cgroup_memory_current_bytes']=max(peak['cgroup_memory_current_bytes'],int((cgroup/'memory.current').read_text()))
                stats=dict(line.split() for line in (cgroup/'memory.stat').read_text().splitlines())
                for key in ['anon','file','shmem']:
                    peak['cgroup_'+key+'_bytes']=max(peak['cgroup_'+key+'_bytes'],int(stats.get(key,0)))
            host_stats=getattr(torch.cuda.memory,'host_memory_stats',None)
            if host_stats is not None:
                peak['host_pinned_allocated_bytes']=max(peak['host_pinned_allocated_bytes'],int(host_stats().get('allocated_bytes.current',0)))
            if started is not None and peak['samples']%5==0:
                self.persist('resource-progress.json',dict(phase=name,completed=False,
                    seconds=time.monotonic()-started,**peak))
        def poll():
            while not stopped.wait(.5):
                try:sample()
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        sample();thread = threading.Thread(target=poll,daemon=True);thread.start()
        started = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            stopped.set();thread.join(timeout=2);sample()
            self.rows.append(dict(phase=name,seconds=elapsed,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),**peak))
            self.persist('resources.json',self.rows)
            self.persist('resource-progress.json',dict(completed=True,**self.rows[-1]))
