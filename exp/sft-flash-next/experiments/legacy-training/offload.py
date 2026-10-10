"""Bounded activation offload to RAM and optional disk with backward prefetch.

Stores exact tensor bytes, never a quantized activation or detached context.
Disk files are per-request scratch, removed on success and on exceptions.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import shutil
import threading
import time
import weakref

import torch


@dataclass
class SavedActivation:
    index: int
    device: torch.device
    value: torch.Tensor | None = None
    path: Path | None = None
    shape: tuple = ()
    dtype: torch.dtype | None = None
    written: object = None
    read: object = None


class ActivationOffload:
    def __init__(self, model, *, disk_dir=None, disk_budget_gib=0, min_bytes=1 << 20,
                 disk_min_bytes=64 << 20, prefetch=2, max_pending_writes=2,
                 pin_reads=True, offload_cpu=False):
        if disk_budget_gib < 0 or prefetch < 0 or max_pending_writes < 1:
            raise ValueError('Invalid activation offload budget')
        if disk_budget_gib and disk_dir is None:
            raise ValueError('Disk activation budget requires a scratch directory')
        self.resident = {(str(t.device), t.untyped_storage().data_ptr())
                         for t in [*[p for p in model.parameters() if not p.requires_grad], *model.buffers()]}
        self.disk_dir = Path(disk_dir) if disk_dir is not None else None
        self.disk_budget = int(disk_budget_gib * 2**30)
        self.min_bytes, self.disk_min_bytes = min_bytes, disk_min_bytes
        self.prefetch, self.max_pending_writes = prefetch, max_pending_writes
        self.pin_reads = pin_reads and torch.cuda.is_available()
        self.offload_cpu = offload_cpu
        self.nodes, self.active_reads, self.pending_writes, self.dedup = [], {}, deque(), {}
        self.stats = dict(cpu_bytes=0, disk_bytes=0, deduplicated_bytes=0,
                          prefetch_hits=0, synchronous_reads=0, read_wait_seconds=0.,
                          write_seconds=0., read_seconds=0.)
        self.closed = False
        self.stats_lock = threading.Lock()

    def __enter__(self):
        self.scratch = None
        if self.disk_budget:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.disk_dir).free < self.disk_budget + (2 << 30):
                raise OSError('Insufficient scratch space for activation budget plus 2 GiB reserve')
            self.scratch = tempfile.TemporaryDirectory(prefix='activations-', dir=self.disk_dir)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='activation-write')
        self.reader = ThreadPoolExecutor(max_workers=max(1, self.prefetch), thread_name_prefix='activation-read')
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.hooks.__enter__()
        return self

    @staticmethod
    def _bytes(tensor):
        return memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')

    def _write(self, path, tensor):
        start = time.monotonic()
        data = self._bytes(tensor)
        with path.open('xb', buffering=0) as stream:
            offset = 0
            while offset < len(data):
                wrote = stream.write(data[offset:offset + (16 << 20)])
                if not wrote:
                    raise OSError('Short activation write')
                offset += wrote
            os.fdatasync(stream.fileno())
            # This drops this file's clean cache, but a loop backing file can
            # have its own reclaimable cache. Never drop host-wide caches.
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.stats['write_seconds'] += time.monotonic() - start

    def _read(self, node):
        node.written.result()
        start = time.monotonic()
        value = torch.empty(node.shape, dtype=node.dtype, device='cpu', pin_memory=self.pin_reads)
        data = self._bytes(value)
        with node.path.open('rb', buffering=0) as stream:
            offset = 0
            while offset < len(data):
                got = stream.readinto(data[offset:offset + (16 << 20)])
                if not got:
                    raise OSError('Truncated activation scratch file')
                offset += got
        with self.stats_lock:
            self.stats['read_seconds'] += time.monotonic() - start
        return value

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        storage = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if (tensor.device.type != 'cuda' and not self.offload_cpu) or storage in self.resident or size < self.min_bytes:
            return tensor.detach()
        key = (*storage, tensor._version, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
        prior = self.dedup.get(key)
        if prior is not None and prior[0]() is not None and prior[1]() is not None:
            self.stats['deduplicated_bytes'] += size
            return prior[1]()
        spill = self.scratch is not None and size >= self.disk_min_bytes and self.stats['disk_bytes'] + size <= self.disk_budget
        if spill:
            while len(self.pending_writes) >= self.max_pending_writes:
                self.pending_writes.popleft().result()
        # Blocking D2H gives the writer an immutable completed copy. Writes
        # then overlap later GPU layers, with the queue bounding host buffers.
        value = tensor.detach().to('cpu', copy=True).contiguous()
        node = SavedActivation(len(self.nodes), tensor.device, shape=tuple(tensor.shape), dtype=tensor.dtype)
        if spill:
            node.path = Path(self.scratch.name) / f'{node.index:06d}.bin'
            node.written = self.writer.submit(self._write, node.path, value)
            self.pending_writes.append(node.written)
            self.stats['disk_bytes'] += size
        else:
            node.value = value
            self.stats['cpu_bytes'] += size
        self.nodes.append(weakref.ref(node))
        self.dedup[key] = (weakref.ref(tensor), weakref.ref(node))
        return node

    def _prefetch_before(self, index):
        # Look a few saved tensors ahead in reverse autograd order, rather
        # than retaining disk tensors for the entire backward pass.
        for i in range(index - 1, max(-1, index - 9), -1):
            if len(self.active_reads) >= self.prefetch:
                break
            node = self.nodes[i]()
            if node is not None and node.path is not None and node.read is None:
                node.read = self.reader.submit(self._read, node)
                self.active_reads[i] = node

    def unpack(self, node):
        if isinstance(node, torch.Tensor):
            return node
        if self.closed:
            raise RuntimeError('Activation offload context must enclose backward')
        if node.path is None:
            value = node.value
        else:
            start = time.monotonic()
            if node.read is not None:
                self.stats['prefetch_hits'] += int(node.read.done())
                value = node.read.result()
                node.read = None
                self.active_reads.pop(node.index, None)
            else:
                self.stats['synchronous_reads'] += 1
                value = self._read(node)
            self.stats['read_wait_seconds'] += time.monotonic() - start
        self._prefetch_before(node.index)
        # CUDA operations using the result execute on the same stream after
        # this copy. PyTorch's pinned allocator tracks its transfer lifetime.
        return value.to(node.device, non_blocking=self.pin_reads)

    def __exit__(self, exc_type, exc, tb):
        self.hooks.__exit__(exc_type, exc, tb)
        try:
            self.writer.shutdown(wait=True)
            self.reader.shutdown(wait=True)
            if exc_type is None:
                for future in self.pending_writes:
                    future.result()
        finally:
            self.closed = True
            self.active_reads.clear()
            self.nodes.clear()
            self.dedup.clear()
            if self.scratch is not None:
                self.scratch.cleanup()
        return False
