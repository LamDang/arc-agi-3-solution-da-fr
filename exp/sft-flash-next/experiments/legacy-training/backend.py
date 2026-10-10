"""Autograd-safe W4A16 adapter backend; never use the REAP workspace in backward.

Full sequences traverse the native hybrid model without an inference cache.
Frozen expert and sparse-attention intermediates are recomputed in bounded
blocks. First-order gradients pass through frozen weights and router scores.
"""
from __future__ import annotations

import math
import re
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reap-flash-next"))
import reap_model as rm

NO_MASKS = {"indexed_attention": None, "linear_attention": None}
GDN_SEGMENT_TOKENS = ContextVar('gdn_segment_tokens', default=0)


def install_segmented_gdn():
    from gdn_segments import segmented_delta
    current = rm.mq.torch_chunk_gated_delta_rule
    native = getattr(current, '_unsegmented', current)
    def dispatch(q,k,v,**kwargs):
        size = GDN_SEGMENT_TOKENS.get()
        if size:
            return segmented_delta(native,q,k,v,chunk_tokens=size,**kwargs)
        return native(q,k,v,**kwargs)
    dispatch._unsegmented = native
    rm.mq.torch_chunk_gated_delta_rule = dispatch


def training_segmented_gdn(self, hidden_states, **kwargs):
    token = GDN_SEGMENT_TOKENS.set(self.train_gdn_chunk_tokens)
    try:
        return self.unsegmented_forward(hidden_states, **kwargs)
    finally:
        GDN_SEGMENT_TOKENS.reset(token)


def training_blocked_gdn(self, hidden_states, **kwargs):
    size = self.train_gdn_block_tokens
    if not size or hidden_states.shape[1] <= size:
        return self.unblocked_gdn_forward(hidden_states, **kwargs)
    from gdn_blocks import blocked_forward
    return blocked_forward(self, hidden_states, block_tokens=size, kernels=rm.mq, **kwargs)


def retain_local_tensor(tensor):
    """Bounded backward recomputation needs no additional CPU offloading."""
    return tensor


class FrozenRMSNorm(torch.autograd.Function):
    """Exact first-order RMSNorm with bounded FP32 temporaries and frozen scale.

    Qwen uses (1 + weight), FP32 normalization/scaling, then casts to the input
    dtype. Groups normalize independently; no token/context gradient is cut.
    """
    @staticmethod
    def forward(ctx, x, weight, eps, group_size, block):
        if weight.requires_grad:
            raise ValueError("FrozenRMSNorm requires a frozen scale")
        ctx.weight, ctx.eps, ctx.group_size, ctx.block = weight, eps, group_size, block
        ctx.save_for_backward(x)
        width = x.shape[-1]
        group = group_size or width
        flat = x.reshape(-1, width)
        out = torch.empty_like(flat)
        scale = 1.0 + weight.float()
        for s in range(0, len(flat), block):
            local = flat[s:s+block].float().reshape(-1, width // group, group)
            normalized = local * torch.rsqrt(local.square().mean(-1, keepdim=True) + eps)
            out[s:s+block] = (normalized.flatten(-2) * scale).to(x.dtype)
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        width = x.shape[-1]
        group = ctx.group_size or width
        flat, upstream = x.reshape(-1, width), grad.reshape(-1, width)
        dx = torch.empty_like(flat)
        scale = 1.0 + ctx.weight.float()
        for s in range(0, len(flat), ctx.block):
            local = flat[s:s+ctx.block].float().reshape(-1, width // group, group)
            reciprocal = torch.rsqrt(local.square().mean(-1, keepdim=True) + ctx.eps)
            normalized = local * reciprocal
            g = (upstream[s:s+ctx.block].float() * scale).reshape_as(local)
            dx[s:s+ctx.block] = ((g - normalized * (g * normalized).mean(-1, keepdim=True))
                                     * reciprocal).flatten(-2).to(x.dtype)
        return dx.reshape(x.shape), None, None, None, None


def training_rms_norm(self, x):
    return FrozenRMSNorm.apply(x, self.weight, self.eps, self.group_size, self.train_block)


class FrozenGatedRMSNorm(torch.autograd.Function):
    """Bound FP32 GDN gated-normalization temporaries; retain native BF16 casts."""
    @staticmethod
    def forward(ctx, x, gate, weight, eps, block, activation):
        ctx.weight, ctx.eps, ctx.block, ctx.activation = weight, eps, block, activation
        ctx.save_for_backward(x, gate)
        flat, gates = x.reshape(-1,x.shape[-1]), gate.reshape(-1,x.shape[-1])
        output = torch.empty_like(flat)
        for s in range(0,len(flat),block):
            local = flat[s:s+block].float()
            normed = (local * torch.rsqrt(local.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
            value = weight * normed
            g = gates[s:s+block].float()
            factor = F.silu(g) if activation == 'silu' else g.sigmoid()
            output[s:s+block] = (value * factor).to(x.dtype)
        return output.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x, gate = ctx.saved_tensors
        flat, gates, upstream = [t.reshape(-1,x.shape[-1]) for t in (x,gate,grad)]
        dx, dg = torch.empty_like(flat), torch.empty_like(gates)
        for s in range(0,len(flat),ctx.block):
            local, g, dy = [t[s:s+ctx.block].float() for t in (flat,gates,upstream)]
            inv = torch.rsqrt(local.square().mean(-1,keepdim=True)+ctx.eps)
            normed = local * inv
            value = ctx.weight * normed.to(x.dtype)
            sigmoid = g.sigmoid()
            # Cast at the same boundaries as the native autograd graph.
            factor = F.silu(g) if ctx.activation == 'silu' else sigmoid
            dn = ((dy * factor).to(x.dtype) * ctx.weight).to(x.dtype).float()
            dx[s:s+ctx.block] = (inv * (dn - normed*(dn*normed).mean(-1,keepdim=True))).to(x.dtype)
            derivative = sigmoid*(1+g*(1-sigmoid)) if ctx.activation == 'silu' else sigmoid*(1-sigmoid)
            dg[s:s+ctx.block] = (dy * value.float() * derivative).to(gate.dtype)
        return dx.reshape(x.shape), dg.reshape(gate.shape), None, None, None, None


def training_gated_rms_norm(self, hidden_states, gate):
    return FrozenGatedRMSNorm.apply(hidden_states, gate, self.weight, self.variance_epsilon, self.train_gated_block, self.activation)


class PLERowCache:
    """Bounded frozen-row FIFO cache; raw-row budget excludes Python index overhead."""
    def __init__(self, capacity, width, dtype):
        self.capacity = capacity
        self.values = np.empty((capacity, width), dtype=dtype)
        self.keys = np.full(capacity, -1, dtype=np.int64)
        self.index, self.cursor = {}, 0
        self.hits = self.misses = 0

    def lookup(self, keys, output):
        slots = np.fromiter((self.index.get(int(k), -1) for k in keys), dtype=np.int64, count=len(keys))
        hit = slots >= 0
        output[hit] = self.values[slots[hit]]
        self.hits += int(hit.sum())
        self.misses += int((~hit).sum())
        return ~hit

    def insert(self, keys, values):
        # Cache only the last capacity rows if a request exceeds the budget.
        keys, values = keys[-self.capacity:], values[-self.capacity:]
        slots = (np.arange(len(keys)) + self.cursor) % self.capacity
        for old in self.keys[slots]:
            self.index.pop(int(old), None)
        self.values[slots] = values
        self.keys[slots] = keys
        self.index.update((int(k), int(i)) for k, i in zip(keys, slots))
        self.cursor = (self.cursor + len(keys)) % self.capacity


def materialize_ple(module):
    """Keep exact frozen table bytes in anonymous RAM, immune to file-cache eviction."""
    if getattr(module, 'ple_resident_bytes', 0):
        return module.ple_resident_bytes
    size = sum(s.nbytes for s in module.shards)
    limit_path, stat_path = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.stat')
    if limit_path.exists() and stat_path.exists() and limit_path.read_text().strip() != 'max':
        values = dict(line.split() for line in stat_path.read_text().splitlines())
        occupied = int(values.get('anon',0)) + int(values.get('shmem',0))
        if size + occupied > .8 * int(limit_path.read_text()):
            raise MemoryError('Resident PLE table exceeds 80% of host limit before activations; use mmap/row cache')
    from concurrent.futures import ThreadPoolExecutor
    def copy_shard(item):
        i, shard = item
        if isinstance(shard, np.memmap) and shard.flags.c_contiguous:
            # Large file reads avoid a stream of small NFS mmap page faults.
            root = shard
            while isinstance(root.base, np.memmap):
                root = root.base
            offset = root.offset + shard.ctypes.data - root.ctypes.data
            resident = np.empty(shard.shape, dtype=shard.dtype)
            data = memoryview(resident).cast('B')
            with open(root.filename, 'rb', buffering=0) as stream:
                stream.seek(offset)
                pos = 0
                while pos < len(data):
                    count = stream.readinto(data[pos:pos+(16 << 20)])
                    if not count:
                        raise OSError('Truncated PLE source shard')
                    pos += count
        else:
            resident = np.array(shard, copy=True)
        resident.setflags(write=False)
        print(f'PLE resident shard {i+1}/{len(module.shards)}: {resident.nbytes/2**30:.3f} GiB', flush=True)
        return i, resident
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='ple-resident') as pool:
        for i, resident in pool.map(copy_shard, enumerate(module.shards)):
            module.shards[i] = resident
    module.ple_resident_bytes = size
    return size


def unique_mmap_lookup(self, ids):
    """Read each needed frozen PLE row once, with parallel sorted shard reads."""
    from concurrent.futures import ThreadPoolExecutor
    flat = ids.reshape(-1).cpu().numpy()
    unique, inverse = np.unique(flat, return_inverse=True)
    if len(unique) and (unique[0] < 0 or unique[-1] >= self.num_embeddings):
        raise IndexError("PLE row outside the embedding table")
    rows = np.empty((len(unique), self.embedding_dim), dtype=self.shards[0].dtype)
    cache = getattr(self, 'row_cache', None)
    missing = cache.lookup(unique, rows) if cache is not None else np.ones(len(unique), dtype=bool)
    positions = np.flatnonzero(missing)
    required = unique[positions]
    shard = np.searchsorted(self.offsets, required, side="right") - 1
    def read_shard(s):
        mask = shard == s
        rows[positions[mask]] = self.shards[s][required[mask] - self.offsets[s]]
    shards = np.unique(shard)
    workers = getattr(self, 'read_workers', 1)
    if workers > 1 and len(shards) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(read_shard, shards))
    else:
        for shard_id in shards:
            read_shard(shard_id)
    if cache is not None:
        cache.insert(required, rows[positions])
    tensor = torch.from_numpy(rows[inverse])
    if tensor.dtype == torch.int16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.reshape(*ids.shape, self.embedding_dim)


@contextmanager
def request_ple_cache(model):
    """Retain only this request's frozen PLE lookups through recomputation.

    The table remains memory mapped; the 130K request cache is approximately
    0.62 GiB, avoiding a second random pass over the 95 GiB NFS table.
    """
    originals = []
    for module in model.modules():
        if isinstance(module, rm.MmapEmbedding):
            original = module.forward
            cache = {}
            def cached(ids, original=original, cache=cache):
                key = ids.detach().cpu()
                if "ids" not in cache or not torch.equal(cache["ids"], key):
                    cache["ids"] = key.clone()
                    cache["value"] = original(ids)
                return cache["value"]
            originals.append((module, original))
            module.forward = cached
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original



def load_pruned(model_dir, keep_path=None, **kwargs):
    """Load either a 256-expert checkpoint or select clean-map experts in RAM.

    The source-512 path temporarily requires the full W4A16 model on the GPU.
    No duplicate 35+ GiB checkpoint is written to Kaggle's small writable volume.
    """
    import gc
    import json
    model, recorder = rm.load_model(model_dir, record=False, **kwargs)
    count = model.config.text_config.num_experts
    if count == 256:
        return model, recorder
    if count != 512 or keep_path is None:
        raise ValueError('A 512-expert source requires an explicit validated 256-expert map')
    kept = json.loads(Path(keep_path).read_text())['kept']
    for i, layer in enumerate(model.model.language_model.layers):
        e = layer.mlp.experts
        ids = torch.tensor(kept[str(i)], device=e.qweight_down.device)
        if len(ids) != 256 or len(set(ids.tolist())) != 256:
            raise ValueError('Invalid expert map')
        for name in ('qweight_gate_up', 'scales_gate_up', 'qweight_down', 'scales_down'):
            setattr(e, name, getattr(e, name).index_select(0, ids))
        e.num_experts = 256
        e.keep = e.filled = None
        gate = layer.mlp.gate
        gate.weight = nn.Parameter(gate.weight.index_select(0, ids), requires_grad=False)
        gate.num_experts = 256
    model.config.text_config.num_experts = 256
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, recorder


class FrozenHyperMix(torch.autograd.Function):
    """Token-local hyperconnection recomputation with a bounded working set."""
    @staticmethod
    def forward(ctx, x, module, block):
        ctx.module, ctx.block = module, block
        ctx.save_for_backward(x)
        flat = x.reshape(-1, x.shape[-1])
        mixed = x.new_empty((len(flat), module.hidden_size))
        inject = x.new_empty((len(flat), module.hc_count)) if module.block_inject_weight is not None else x.new_empty((0,))
        for i in range(0, len(flat), block):
            result = module.unblocked_forward(flat[i:i+block])
            if isinstance(result, tuple):
                mixed[i:i+block], _, inject[i:i+block] = result
            else:
                mixed[i:i+block] = result
        return mixed.reshape(*x.shape[:-1], module.hidden_size), inject

    @staticmethod
    def backward(ctx, grad_mixed, grad_inject):
        x, = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        upstream = grad_mixed.reshape(-1, ctx.module.hidden_size)
        dx = torch.empty_like(flat)
        for i in range(0, len(flat), ctx.block):
            with torch.autograd.graph.saved_tensors_hooks(retain_local_tensor, retain_local_tensor), torch.enable_grad():
                local = flat[i:i+ctx.block].detach().requires_grad_(True)
                result = ctx.module.unblocked_forward(local)
                if isinstance(result, tuple):
                    outputs = (result[0], result[2])
                    grads = (upstream[i:i+ctx.block], grad_inject[i:i+ctx.block])
                else:
                    outputs, grads = (result,), (upstream[i:i+ctx.block],)
                dx[i:i+ctx.block], = torch.autograd.grad(outputs, local, grads)
        return dx.reshape_as(x), None, None


def training_hyper_mix(self, x):
    mixed, injection = FrozenHyperMix.apply(x, self, self.train_block)
    if self.block_inject_weight is None:
        return mixed
    return mixed, x, injection.reshape(*x.shape[:-1], self.hc_count)

def expert_weights(module, expert, dtype):
    return (rm.dequantize_gptq(module.qweight_gate_up[expert], module.scales_gate_up[expert], dtype),
            rm.dequantize_gptq(module.qweight_down[expert], module.scales_down[expert], dtype))


def expert_value(x, w1, w2, act):
    gate, up = (x @ w1).chunk(2, dim=-1)
    return (act(gate) * up) @ w2


class FrozenExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, weights, module, block):
        ctx.module, ctx.block = module, block
        ctx.save_for_backward(x, indices, weights)
        out = torch.zeros_like(x)
        for e in range(module.num_experts):
            rows, slots = torch.where(indices == e)
            if not len(rows):
                continue
            w1, w2 = expert_weights(module, e, x.dtype)
            for start in range(0, len(rows), block):
                r, s = rows[start:start+block], slots[start:start+block]
                y = expert_value(x[r], w1, w2, module.act_fn)
                out.index_add_(0, r, y * weights[r, s, None])
        return out

    @staticmethod
    def backward(ctx, grad):
        x, indices, weights = ctx.saved_tensors
        module, block = ctx.module, ctx.block
        dx, dw = torch.zeros_like(x), torch.zeros_like(weights)
        for e in range(module.num_experts):
            rows, slots = torch.where(indices == e)
            if not len(rows):
                continue
            # Local, immutable weights: no shared buffer can be overwritten.
            w1, w2 = expert_weights(module, e, x.dtype)
            for start in range(0, len(rows), block):
                r, s = rows[start:start+block], slots[start:start+block]
                with torch.autograd.graph.saved_tensors_hooks(retain_local_tensor, retain_local_tensor), torch.enable_grad():
                    local = x[r].detach().requires_grad_(True)
                    y = expert_value(local, w1, w2, module.act_fn)
                    g = torch.autograd.grad(y, local, grad[r] * weights[r, s, None])[0]
                dx.index_add_(0, r, g)
                dw[r, s] = (grad[r].float() * y.detach().float()).sum(-1).to(dw.dtype)
        return dx, None, dw, None, None


def training_experts(self, hidden_states, top_k_index, top_k_weights):
    return FrozenExperts.apply(hidden_states, top_k_index, top_k_weights, self, self.train_block)


def attention_block(q, k, v, selected, scale):
    idx = selected.clamp_min(0)
    kg, vg = k[:, idx].transpose(0, 1), v[:, idx].transpose(0, 1)
    return F.scaled_dot_product_attention(
        q.float().transpose(0, 1).unsqueeze(2), kg.float(), vg.float(),
        attn_mask=(selected >= 0)[:, None, None, :], scale=scale, enable_gqa=True,
    ).squeeze(2).transpose(0, 1).to(q.dtype)


class SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected, scale, block):
        ctx.scale, ctx.block = scale, block
        ctx.save_for_backward(q, k, v, selected)
        out = torch.empty_like(q)
        for s in range(0, q.shape[1], block):
            out[:, s:s+block] = attention_block(q[:, s:s+block], k, v, selected[s:s+block], scale)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, selected = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk, dv = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
        # Gather only this query block; scatter its key/value cotangents back.
        for s in range(0, q.shape[1], ctx.block):
            sel = selected[s:s+ctx.block]
            idx = sel.clamp_min(0)
            with torch.autograd.graph.saved_tensors_hooks(retain_local_tensor, retain_local_tensor), torch.enable_grad():
                qs = q[:, s:s+ctx.block].detach().float().transpose(0, 1).unsqueeze(2).requires_grad_(True)
                ks = k[:, idx].detach().float().transpose(0, 1).requires_grad_(True)
                vs = v[:, idx].detach().float().transpose(0, 1).requires_grad_(True)
                out = F.scaled_dot_product_attention(qs, ks, vs,
                    attn_mask=(sel >= 0)[:, None, None, :], scale=ctx.scale, enable_gqa=True)
                gq, gk, gv = torch.autograd.grad(out, (qs, ks, vs),
                    grad[:, s:s+ctx.block].float().transpose(0, 1).unsqueeze(2))
            dq[:, s:s+ctx.block] = gq.squeeze(2).transpose(0, 1)
            # Masked slots were gathered from key zero only as a placeholder.
            # Their gradients are zero; scattering thousands of them onto the
            # same BF16 address causes severe atomic contention on CUDA.
            valid = sel.flatten() >= 0
            destinations = idx.flatten()[valid]
            dk.index_add_(1, destinations, gk.transpose(0, 1).flatten(1, 2)[:, valid].float())
            dv.index_add_(1, destinations, gv.transpose(0, 1).flatten(1, 2)[:, valid].float())
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


def training_attention(self, hidden_states, position_embeddings, attention_mask=None,
                       past_key_values=None, **kwargs):
    if past_key_values is not None or hidden_states.shape[0] != 1:
        raise ValueError("Training requires batch 1 and no inference cache")
    cos, sin = position_embeddings
    with torch.no_grad():
        selected = rm.select_tokens(self.indexer, hidden_states, cos, sin, None,
                                    query_block=self.index_block)
    def project(hidden, c, s):
        shape = hidden.shape[:-1]
        hs = (*shape, -1, self.head_dim)
        q, gate = self.q_proj(hidden).view(*shape, -1, self.head_dim*2).chunk(2,-1)
        q = self.q_norm(q.reshape(hs)).transpose(1,2)
        k = self.k_norm(self.k_proj(hidden).view(hs)).transpose(1,2)
        v = self.v_proj(hidden).view(hs).transpose(1,2)
        q,k = rm.mq.apply_rotary_pos_emb(q,k,c,s)
        return q,k,v,gate.reshape(*shape,-1)
    size = getattr(self, 'train_projection_block', 0)
    blocked = size and hidden_states.shape[1] > size
    if blocked:
        from torch.utils.checkpoint import checkpoint
        pieces = []
        for h,c,s in zip(hidden_states.split(size,dim=1), cos.split(size,dim=1), sin.split(size,dim=1)):
            pieces.append(checkpoint(project,h,c,s,use_reentrant=False) if torch.is_grad_enabled() else project(h,c,s))
        q,k,v = [torch.cat([part[i] for part in pieces],dim=2) for i in range(3)]
        gates = [part[3] for part in pieces]
        del pieces
    else:
        q,k,v,gate = project(hidden_states,cos,sin)
    if self.train_attention_backend == 'triton':
        from sparse_kernels import IndexedAttention
        out = IndexedAttention.apply(q[0], k[0], v[0], selected, self.scaling, self.key_block)
    else:
        out = SparseAttention.apply(q[0], k[0], v[0], selected, self.scaling, self.query_block)
    if not blocked:
        out = out.transpose(0,1).reshape(*hidden_states.shape[:-1],-1) * gate.sigmoid()
        return self.o_proj(out), None
    def finish(attended, gate):
        local = attended.transpose(0,1).reshape(1,gate.shape[1],-1)
        return self.o_proj(local * gate.sigmoid())
    # Shared split nodes accumulate gradients once; selected keys and K/V still
    # span the entire request. These are token-local projection blocks only.
    outputs = [checkpoint(finish,part,gate,use_reentrant=False) if torch.is_grad_enabled() else finish(part,gate)
               for part,gate in zip(out.split(size,dim=1),gates)]
    return torch.cat(outputs,dim=1), None


class LoRALinear(nn.Module):
    """FP32 adapter state, base-dtype matmuls; zero-init leaves base unchanged."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)),
                                     self.lora_B.to(x.dtype)) * self.scale


TARGET = re.compile(r"^model\.language_model\.layers\.\d+\.(?:"
                    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
                    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
                    r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))$")



def ple_local(module, hidden, embeddings):
    """Native PLE arithmetic on a token window including its left conv halo."""
    key = module.norm_key(module.key_proj(embeddings)).unflatten(-1, (module.hc_count, module.hidden_size))
    value = module.value_proj(embeddings)
    query = module.norm_query(hidden).unflatten(-1, (module.hc_count, module.hidden_size))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(module.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    return gated + module._short_conv(module.norm_conv(gated), None)


class FrozenPLE(torch.autograd.Function):
    """Recompute frozen PLE windows; overlapping halo gradients are summed.

    N-gram embeddings are computed on the FULL input before windowing, so both
    the hashing history and dilated convolution cross every window boundary.
    """
    @staticmethod
    def forward(ctx, hidden, ids, module, block):
        ctx.module, ctx.block = module, block
        ctx.save_for_backward(hidden, ids)
        embeddings = module.ple_embedding(ids, None)
        output = torch.empty_like(hidden)
        for start in range(0, hidden.shape[1], block):
            end, left = min(start+block, hidden.shape[1]), max(0, start-module.short_conv_state_len)
            output[:,start:end] = ple_local(module, hidden[:,left:end], embeddings[:,left:end])[:,start-left:]
        return output

    @staticmethod
    def backward(ctx, grad):
        hidden, ids = ctx.saved_tensors
        module = ctx.module
        with torch.no_grad():
            embeddings = module.ple_embedding(ids, None)
        result = torch.zeros_like(hidden)
        for start in range(0, hidden.shape[1], ctx.block):
            end, left = min(start+ctx.block, hidden.shape[1]), max(0, start-module.short_conv_state_len)
            with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(retain_local_tensor, retain_local_tensor):
                x = hidden[:,left:end].detach().requires_grad_(True)
                y = ple_local(module, x, embeddings[:,left:end])[:,start-left:]
                dx, = torch.autograd.grad(y, x, grad[:,start:end])
            result[:,left:end] += dx
        return result, None, None, None


def training_ple(self, hidden_states, input_ids, past_key_values, conv_mask=None):
    if past_key_values is not None or conv_mask is not None:
        raise ValueError('Windowed training PLE requires an unpadded full sequence without cache')
    return FrozenPLE.apply(hidden_states, input_ids, self, self.train_ple_block)


def checkpointed_ple(self, hidden_states, input_ids, past_key_values, conv_mask=None):
    if past_key_values is not None:
        raise ValueError('Training PLE checkpoint does not support an inference cache')
    from torch.utils.checkpoint import checkpoint
    if torch.is_grad_enabled() and self.training:
        return checkpoint(self.uncheckpointed_forward, hidden_states, input_ids, None,
                          conv_mask=conv_mask, use_reentrant=False)
    return self.uncheckpointed_forward(hidden_states, input_ids, None, conv_mask=conv_mask)



def checkpointed_gdn(self, hidden_states, **kwargs):
    from torch.utils.checkpoint import checkpoint
    if kwargs.get('cache_params') is not None:
        raise ValueError('GDN training checkpoint requires no inference cache')
    if torch.is_grad_enabled() and self.training:
        return checkpoint(self.uncheckpointed_gdn_forward, hidden_states, use_reentrant=False, **kwargs)
    return self.uncheckpointed_gdn_forward(hidden_states, **kwargs)


def configure(model, rank=16, alpha=32, expert_block=1024, query_block=16,
              index_block=128, gradient_checkpointing=True, norm_block=1024,
              attention_backend='sdpa', key_block=32, hyper_block=0,
              ple_cache_gib=0., ple_workers=1, checkpoint_group=1, ple_checkpoint=False, gdn_checkpoint=False,
              ple_block=0, gated_norm_block=0, ple_resident=False, gdn_chunk_tokens=0, rms_block_mib=0, gdn_block_tokens=0, attention_projection_block=0):
    if attention_backend not in ('sdpa', 'triton') or key_block not in (32, 64):
        raise ValueError('Unsupported attention backend or key block')
    if checkpoint_group < 1:
        raise ValueError("Checkpoint group must be positive")
    model.train_checkpoint_group = checkpoint_group
    if gdn_chunk_tokens:
        install_segmented_gdn()
    model.requires_grad_(False)
    targets = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and TARGET.fullmatch(name):
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr, LoRALinear(module, rank, alpha))
            targets.append(name)
    if not targets:
        raise ValueError("No supported language adapter targets resolved")
    if norm_block:
        for module in model.model.language_model.modules():
            if isinstance(module, rm.mq.Qwen4ExpTextRMSNorm):
                module.train_block = max(1,int(rms_block_mib*2**20)//(4*module.weight.numel())) if rms_block_mib else norm_block
                module.forward = types.MethodType(training_rms_norm, module)
    if hyper_block:
        for module in model.model.language_model.modules():
            if isinstance(module, rm.mq.Qwen4ExpTextGatedResidual):
                if any(p.requires_grad for p in module.parameters()):
                    raise ValueError('Bounded hyperconnection requires frozen weights')
                module.train_block = hyper_block
                module.unblocked_forward = module.forward
                module.forward = types.MethodType(training_hyper_mix, module)
    if gated_norm_block:
        for module in model.model.language_model.modules():
            if isinstance(module, rm.mq.Qwen4ExpTextRMSNormGated):
                if module.activation not in ('silu','sigmoid') or module.weight.requires_grad:
                    raise ValueError('Bounded gated RMS requires frozen SiLU/sigmoid normalization')
                module.train_gated_block = gated_norm_block
                module.forward = types.MethodType(training_gated_rms_norm, module)
    for module in model.modules():
        if isinstance(module, rm.MmapEmbedding):
            if ple_resident:
                materialize_ple(module)
            module._lookup = types.MethodType(unique_mmap_lookup, module)
            module.read_workers = ple_workers
            if ple_cache_gib:
                capacity = int(ple_cache_gib * 2**30) // (module.embedding_dim * module.shards[0].dtype.itemsize)
                if capacity < 1:
                    raise ValueError('PLE cache must fit at least one row')
                module.row_cache = PLERowCache(capacity, module.embedding_dim, module.shards[0].dtype)
    for layer in model.model.language_model.layers:
        if gdn_chunk_tokens and hasattr(layer, 'linear_attn'):
            layer.linear_attn.train_gdn_chunk_tokens = gdn_chunk_tokens
            layer.linear_attn.unsegmented_forward = layer.linear_attn.forward
            layer.linear_attn.forward = types.MethodType(training_segmented_gdn, layer.linear_attn)
        if gdn_block_tokens and hasattr(layer, 'linear_attn'):
            layer.linear_attn.train_gdn_block_tokens = gdn_block_tokens
            layer.linear_attn.unblocked_gdn_forward = layer.linear_attn.forward
            layer.linear_attn.forward = types.MethodType(training_blocked_gdn, layer.linear_attn)
        if ple_block and layer.ple is not None:
            if any(p.requires_grad for p in layer.ple.parameters()):
                raise ValueError('Windowed PLE requires frozen weights')
            layer.ple.train_ple_block = ple_block
            layer.ple.forward = types.MethodType(training_ple, layer.ple)
        if gdn_checkpoint and hasattr(layer, 'linear_attn'):
            layer.linear_attn.uncheckpointed_gdn_forward = layer.linear_attn.forward
            layer.linear_attn.forward = types.MethodType(checkpointed_gdn, layer.linear_attn)
        if ple_checkpoint and layer.ple is not None:
            layer.ple.uncheckpointed_forward = layer.ple.forward
            layer.ple.forward = types.MethodType(checkpointed_ple, layer.ple)
        experts = layer.mlp.experts
        experts.train_block = expert_block
        experts.forward = types.MethodType(training_experts, experts)
        experts.recorder = None
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            attn.query_block, attn.index_block = query_block, index_block
            attn.train_projection_block = attention_projection_block
            attn.train_attention_backend, attn.key_block = attention_backend, key_block
            attn.forward = types.MethodType(training_attention, attn)
    model.config.text_config.use_cache = False
    model.train()
    model.model.visual.eval()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return targets


@torch.no_grad()
def embeddings(model, enc, device):
    """Frozen vision/PLE inputs, no image cropping or sequence truncation."""
    base = model.model
    ids = enc["input_ids"].to(device)
    embeds = base.get_input_embeddings()(ids)
    grid = enc.get("image_grid_thw")
    if grid is not None and len(grid):
        grid = grid.to(device)
        # Process individual images to bound frozen vision activation memory.
        features, offset = [], 0
        for g in grid:
            patches = int(g.prod())
            pixels = enc["pixel_values"][offset:offset+patches].to(device)
            features.extend(base.get_image_features(pixels, g[None], return_dict=True).pooler_output)
            offset += patches
        if offset != len(enc["pixel_values"]):
            raise ValueError("Image grid/patch count mismatch")
        embeds = embeds.masked_scatter((ids == model.config.image_token_id).unsqueeze(-1),
                                      torch.cat(features).to(embeds.dtype))
        positions, _ = base.get_rope_index(ids, mm_token_type_ids=enc["mm_token_type_ids"].to(device),
                                         image_grid_thw=grid)
    else:
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
    return ids, embeds, positions


class SelectedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, weight, block):
        ctx.weight, ctx.block = weight, block  # frozen resident head: do not CPU-copy it per call
        ctx.save_for_backward(hidden, targets)
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for s in range(0, len(targets), block):
            logits = F.linear(hidden[s:s+block], weight).float()
            loss += F.cross_entropy(logits, targets[s:s+block], reduction="sum")
        return loss

    @staticmethod
    def backward(ctx, grad):
        hidden, targets = ctx.saved_tensors
        dh = torch.empty_like(hidden)
        for s in range(0, len(targets), ctx.block):
            h, t = hidden[s:s+ctx.block], targets[s:s+ctx.block]
            probabilities = F.linear(h, ctx.weight).float().softmax(-1)
            probabilities[torch.arange(len(t), device=t.device), t] -= 1
            dh[s:s+ctx.block] = (probabilities.to(h.dtype) @ ctx.weight) * grad
        return dh, None, None, None



def language_hidden(model, ids, inputs, positions):
    group = getattr(model, 'train_checkpoint_group', 1)
    lm = model.model.language_model
    if group == 1 or not torch.is_grad_enabled():
        return lm(inputs_embeds=inputs, position_ids=positions, ple_input_ids=ids,
                  use_cache=False, past_key_values=None, attention_mask=NO_MASKS).last_hidden_state
    # This runner only supplies unpadded batch-one full requests, three mRoPE
    # coordinate planes, no cache, and no output-attention/router collection.
    if positions.shape[0] != 3 or inputs.shape[0] != 1:
        raise ValueError('Grouped checkpoint path expects one unpadded multimodal request')
    from torch.utils.checkpoint import checkpoint
    position_embeddings = lm.rotary_emb(inputs, positions)
    hidden = inputs.repeat(1, 1, lm.config.hc_count)
    for start in range(0, len(lm.layers), group):
        layers = tuple(lm.layers[start:start+group])
        def run_group(x, layers=layers):
            for layer in layers:
                x = layer(x, position_embeddings=position_embeddings, attention_mask=None,
                          conv_mask=None, past_key_values=None, ple_input_ids=ids)
            return x
        # Keep inner per-layer checkpoints: outer checkpoints reduce persistent
        # CPU boundaries; backward temporarily reconstructs one group's boundaries.
        hidden = checkpoint(run_group, hidden, use_reentrant=False)
    return lm.hyper_connection_mixer(hidden)

def loss_sum(model, enc, prompt_tokens, device, loss_block=128):
    ids, inputs, positions = embeddings(model, enc, device)
    if not 0 < prompt_tokens < ids.shape[1]:
        raise ValueError("Invalid final-reply boundary")
    hidden = language_hidden(model, ids, inputs, positions)
    # Causal shift: last prompt hidden state predicts first completion token.
    selected = hidden[0, prompt_tokens-1:-1]
    return SelectedCrossEntropy.apply(selected, ids[0, prompt_tokens:], model.lm_head.weight, loss_block)


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if current.keys() != state.keys():
        raise ValueError("Adapter names do not match this model/recipe")
    with torch.no_grad():
        for name, value in state.items():
            if current[name].shape != value.shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            current[name].copy_(value)


def activation_offload(model):
    """Offload activations, retaining already-resident frozen weights by reference."""
    resident = {(str(t.device), t.untyped_storage().data_ptr())
                for t in [*[p for p in model.parameters() if not p.requires_grad], *model.buffers()]
                if t.device.type == "cuda"}

    def pack(tensor):
        if tensor.device.type != "cuda" or (str(tensor.device), tensor.untyped_storage().data_ptr()) in resident:
            return None, tensor
        # Pageable CPU storage avoids pinning the entire ~100 GB long-context graph.
        return tensor.device, tensor.to("cpu")

    def unpack(value):
        device, tensor = value
        return tensor if device is None else tensor.to(device)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


@torch.no_grad()
def token_losses(model, enc, prompt_tokens, device, loss_block=128):
    ids, inputs, positions = embeddings(model, enc, device)
    hidden = language_hidden(model, ids, inputs, positions)
    selected, targets = hidden[0, prompt_tokens-1:-1], ids[0, prompt_tokens:]
    losses = []
    for s in range(0, len(targets), loss_block):
        logits = model.lm_head(selected[s:s+loss_block]).float()
        losses.append(F.cross_entropy(logits, targets[s:s+loss_block], reduction="none").cpu())
    return torch.cat(losses)
