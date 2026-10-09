"""Per-phase synchronized CUDA peaks and sampled host RAM; no arithmetic hooks."""
from contextlib import contextmanager
import os
import threading
import time

import psutil
import torch


class Resources:
    def __init__(self):
        self.rows = []
    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stopped = threading.Event()
        peak = dict(rss_bytes=0,tree_pss_bytes=0,children_rss_bytes=0,host_used_bytes=0,samples=0)
        process = psutil.Process(os.getpid())
        def sample():
            peak['samples'] += 1
            peak['rss_bytes'] = max(peak['rss_bytes'],process.memory_info().rss)
            children = process.children(recursive=True)
            peak['children_rss_bytes'] = max(peak['children_rss_bytes'],sum(p.memory_info().rss for p in children if p.is_running()))
            pss = sum(getattr(p.memory_full_info(),'pss',0) for p in [process,*children] if p.is_running())
            peak['tree_pss_bytes'] = max(peak['tree_pss_bytes'],pss)
            peak['host_used_bytes'] = max(peak['host_used_bytes'],psutil.virtual_memory().used)
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
