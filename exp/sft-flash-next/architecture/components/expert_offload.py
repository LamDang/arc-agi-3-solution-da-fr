"""CPU frozen expert buffers; GPU FP32 LoRA masters and gradients.

Frozen quantized buffers use layer views into one shared pinned allocation. FP32 trainable
parameters retain their original CUDA Parameter objects and CUDA FP32 gradients.
Only current and next layer frozen GPU states exist.
Native expert routing, quantized projections and activation arithmetic are used.
"""
from contextlib import contextmanager
import time
import torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from .common import adopt
from .precision import BF16Experts, BF16LigerExperts
from .mlp import LigerExperts


class ExpertStage:
    def forward(self, hidden_states, top_k_index, top_k_weights, *, _prefetched=False):
        if _prefetched:
            return super().forward(hidden_states,top_k_index,top_k_weights)
        if getattr(self,'chunk_tokens',0):
            from .expert_chunks import expert_chunks
            return expert_chunks(self,hidden_states,top_k_index,top_k_weights)
        return self.expert_stager.call(self,hidden_states,top_k_index,top_k_weights)


class CPUExperts(ExpertStage,native.Qwen4ExpTextExperts): pass
class CPUBF16Experts(ExpertStage,BF16Experts): pass
class CPULigerExperts(ExpertStage,LigerExperts): pass
class CPUBF16LigerExperts(ExpertStage,BF16LigerExperts): pass

CLASSES={native.Qwen4ExpTextExperts:CPUExperts,BF16Experts:CPUBF16Experts,
         LigerExperts:CPULigerExperts,BF16LigerExperts:CPUBF16LigerExperts}


class LayerPrefetch:
    def __init__(self, modules, pin=True):
        self.modules=modules;self.stream=torch.cuda.Stream();self.cache={};self.direction=1
        self.records=[];self.slabs=[];self.layouts=[];self.max_staged_layers=0
        for index,module in enumerate(modules):
            module.expert_stager=self;module.expert_layer=index
            # The native quantized projection otherwise creates an unregistered
            # CUDA g_idx on first use. Make its unchanged values explicit state.
            for child in module.modules():
                if hasattr(child,'qweight') and hasattr(child,'group_size') and not hasattr(child,'g_idx'):
                    child.register_buffer('g_idx',torch.arange(child.infeatures,dtype=torch.int32)//child.group_size,persistent=False)
            layout=[];size=0
            for name,value in module.named_buffers():
                size=(size+7)//8*8;count=value.numel()*value.element_size()
                layout.append((name,size,count,value.shape,value.dtype));size+=count
            self.layouts.append(layout)
            self.slabs.append((size+7)//8*8)
        # CUDA's pinned allocator rounds allocation sizes. One ~0.614 GiB
        # allocation per layer reserved ~48 GiB for ~29.49 GiB of data. A single
        # backing allocation avoids 48 independent rounding gaps; layer views
        # still support exactly the same two-layer GPU prefetch window.
        self.backing=torch.empty(sum(self.slabs),dtype=torch.uint8,device='cpu',pin_memory=pin)
        sizes=self.slabs;self.slabs=[];offset=0
        for module,layout,size in zip(modules,self.layouts,sizes):
            slab=self.backing[offset:offset+size];offset+=size
            buffers=dict(module.named_buffers())
            for name,start,count,shape,dtype in layout:
                view=slab[start:start+count].view(dtype).reshape(shape)
                # Copy GPU -> pinned CPU directly: do not allocate a pageable
                # CPU copy of every frozen weight before populating the slab.
                view.copy_(buffers[name])
                parent,_,key=name.rpartition('.')
                module.get_submodule(parent)._buffers[key]=view
            self.slabs.append(slab)
            assert all(p.device.type=='cuda' for p in module.parameters())
            assert all(p.dtype==torch.float32 for p in module.parameters() if p.requires_grad)

    @contextmanager
    def phase(self,direction):
        old=self.direction;self.direction=direction
        try:yield
        finally:self.direction=old

    def checkpoint_contexts(self):
        return self.phase(1),self.phase(-1)

    def prefetch(self,index):
        if not 0<=index<len(self.modules) or index in self.cache:return
        if len(self.cache)>=2:raise RuntimeError('Expert staging exceeded two layers')
        started=time.monotonic();module=self.modules[index]
        with torch.cuda.stream(self.stream):
            gpu=self.slabs[index].to('cuda',non_blocking=True)
            state={name:gpu[start:start+count].view(dtype).reshape(shape)
                   for name,start,count,shape,dtype in self.layouts[index]}
            ready=torch.cuda.Event();ready.record(self.stream)
        self.cache[index]=(state,ready)
        self.max_staged_layers=max(self.max_staged_layers,len(self.cache))
        self.records.append(dict(event='prefetch',layer=index,direction=self.direction,
            frozen_bytes=self.slabs[index].numel(),parameter_bytes=0,
            resident_parameter_bytes=sum(p.numel()*p.element_size() for p in module.parameters()),
            host_dispatch_seconds=time.monotonic()-started,staged_layers=len(self.cache)))

    def acquire(self,module,input_dtype,parameter_flat=None,event='execute'):
        index=module.expert_layer
        # Checkpoint recomputation reverses the layer order. Remove a stale
        # lookahead before acquiring a new window (e.g. early-stop recompute).
        keep={index,index+self.direction}
        for stale in list(self.cache):
            if stale not in keep:self.release(stale)
        self.prefetch(index)
        state,ready=self.cache[index];current=torch.cuda.current_stream()
        current.wait_event(ready)
        for value in state.values():value.record_stream(current)
        self.prefetch(index+self.direction)
        state=dict(state)
        if parameter_flat is not None:
            offset=0
            for name,p in module.named_parameters():
                state[name]=parameter_flat[offset:offset+p.numel()].view_as(p).to(p.dtype)
                offset+=p.numel()
            assert offset==parameter_flat.numel()
        else:
            state.update(module.named_parameters())
        self.records.append(dict(event=event,layer=index,direction=self.direction,input_dtype=str(input_dtype)))
        return state

    def call(self,module,*args):
        state=self.acquire(module,args[0].dtype)
        try:
            return torch.func.functional_call(module,state,args,{'_prefetched':True},strict=True)
        finally:self.release(module.expert_layer)

    def release(self,index):
        entry=self.cache.pop(index,None)
        if entry:
            state,ready=entry
            # A lookahead may be cancelled before it is consumed; the caching
            # allocator must still respect its copy stream.
            for value in state.values():value.record_stream(self.stream)

    def close(self):
        for index in list(self.cache):self.release(index)
        self.stream.synchronize()

    def report(self):
        from collections import Counter
        parameters=[p for module in self.modules for p in module.parameters() if p.requires_grad]
        return dict(canonical_device='cpu',frozen_canonical_device='cpu',
            trainable_device='cuda',gradient_device='cuda',trainable_dtype='torch.float32',
            parameter_devices=dict(Counter(p.device.type for p in parameters)),
            gradient_devices=dict(Counter(p.grad.device.type for p in parameters if p.grad is not None)),
            frozen_cpu_bytes=sum(s.numel() for s in self.slabs),max_staged_layers=self.max_staged_layers,
            records=self.records)


def install(model,inventory,chunk_tokens=0):
    modules=[]
    for index,layer in enumerate(model.get_base_model().model.language_model.layers):
        old=layer.mlp.experts
        if type(old) not in CLASSES:raise TypeError('Unsupported expert component '+type(old).__name__)
        module=adopt(old,CLASSES[type(old)])
        layer.mlp.experts=module;module.chunk_tokens=chunk_tokens;modules.append(module)
        inventory.append(dict(name=f'model.language_model.layers.{index}.mlp.experts',
            original=type(old).__name__,implementation=type(module).__name__))
    return LayerPrefetch(modules)
