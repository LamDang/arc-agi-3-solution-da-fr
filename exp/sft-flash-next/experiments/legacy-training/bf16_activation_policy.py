"""Opt5: BF16 LoRA, module-boundary activations and saved CPU activations.

FP32 arithmetic is permitted inside operators/modules. Non-scalar floating
inputs/outputs crossing nn.Module boundaries are BF16. Scalar loss remains
FP32. Saved floating tensors (including FP32 statistics) are packed as BF16
and restored to their original dtype for backward; this is lossy storage.
The original FP32 diagnostic adapter is deterministically rounded to BF16 at
load time. No production initialization or optimizer update is performed.
"""
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_map, tree_flatten

MOE_SOURCE_SHA256 = '2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c'


def cast_boundary(value, observe=None):
    def cast(tensor):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.ndim == 0:
            return tensor
        if observe is not None:
            observe(tensor)
        return tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
    return tree_map(cast, value)


def saved_bf16_hooks(pack, unpack, observe=None):
    """Preserve original dtype metadata while storing only BF16 floating data."""
    def packed(tensor):
        original_dtype = tensor.dtype
        stored = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        if observe is not None:
            observe(tensor, stored)
        return original_dtype, pack(stored)
    def unpacked(item):
        original_dtype, payload = item
        tensor = unpack(payload)
        return tensor.to(original_dtype) if tensor.dtype != original_dtype else tensor
    return packed, unpacked


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def qualify_cpu(output_dir):
    """Check boundary rounding, gradients and CPU storage dtype contracts."""
    out = Path(output_dir)
    original = torch.tensor([1.0007, -0.31317, 0.000123456, 19.037], requires_grad=True)
    records = []
    def observe(a,b):records.append((str(a.dtype),str(b.dtype)))
    pack, unpack = saved_bf16_hooks(lambda x:x.detach().clone(), lambda x:x, observe)
    scalar = torch.tensor(0.123456, dtype=torch.float32)
    indices = torch.tensor([1,3],dtype=torch.int64)
    cast = cast_boundary({'activation':original, 'loss':scalar, 'indices':indices})
    assert cast['activation'].dtype == torch.bfloat16
    assert cast['loss'] is scalar and cast['indices'] is indices
    assert torch.equal(cast['activation'],original.to(torch.bfloat16))
    cast['activation'].float().sum().backward()
    assert torch.equal(original.grad,torch.ones_like(original))
    for tensor in (original, original.to(torch.bfloat16), scalar, indices):
        item=pack(tensor);restored=unpack(item)
        assert restored.dtype==tensor.dtype and restored.shape==tensor.shape
        expected=tensor.to(torch.bfloat16).to(tensor.dtype) if tensor.is_floating_point() else tensor
        assert torch.equal(restored,expected)
        assert item[1].dtype==(torch.bfloat16 if tensor.is_floating_point() else tensor.dtype)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        x=original.detach().clone().requires_grad_();loss=x.square().sum()
        loss.backward()
    assert torch.equal(x.grad,2*original.detach().to(torch.bfloat16).float())
    report=dict(passed=True,device='cpu',boundary_bf16=True,scalar_loss_preserved_fp32=True,
                integer_indices_preserved=True,pack_only_bf16_floating=True,
                unpack_original_dtype=True,lossy_backward_checked=True,
                optimizer_updates=0,records=records)
    (out/'bf16-policy-operator-check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def install_for_capture(adapter_path, launch_dir, report_path):
    import peft
    import auto_round.modeling.fused_moe.moe_experts_interface as moe
    source_path=Path(inspect.getfile(moe));source=source_path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=MOE_SOURCE_SHA256:
        raise RuntimeError('Unreviewed AutoRound MoE source')
    launch=Path(launch_dir)
    (launch/'bf16-policy-moe-source.py').write_bytes(source)
    path=Path(report_path)
    report=dict(implementation='opt5_bf16_lora_activations',diagnostic_only=True,
        module_boundary_dtype='torch.bfloat16',saved_floating_storage_dtype='torch.bfloat16',
        scalar_loss_dtype='torch.float32',fp32_intermediate_computation_allowed=True,
        saved_fp32_statistics_compressed=True,unpack_restores_original_dtype=True,
        autocast_dtype='torch.bfloat16',optimizer_updates=0,moe_source_sha256=MOE_SOURCE_SHA256,
        boundary_input_casts={},boundary_output_casts={},boundary_output_dtype_counts={},
        saved_original_dtype_counts={},saved_storage_dtype_counts={},saved_original_bytes={},
        saved_storage_bytes={},layer_calls=[],finalized=False)
    counts={key:Counter() for key in ('boundary_input_casts','boundary_output_casts',
        'boundary_output_dtype_counts','saved_original_dtype_counts','saved_storage_dtype_counts',
        'saved_original_bytes','saved_storage_bytes')}
    def save():
        report.update({key:dict(value) for key,value in counts.items()})
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    original_load=torch.load
    def load(file,*args,**kwargs):
        value=original_load(file,*args,**kwargs)
        if isinstance(file,(str,Path)) and Path(file)==Path(adapter_path):
            if not isinstance(value,dict) or len(value)!=744:raise ValueError('Unexpected adapter state')
            rounded={name:t.to(torch.bfloat16) for name,t in value.items()}
            repeated={name:t.to(torch.bfloat16) for name,t in value.items()}
            before=state_digest(value);after=state_digest(rounded)
            assert after==state_digest(repeated)
            assert all(torch.count_nonzero(v)>0 for v in rounded.values())
            torch.save(rounded,launch/'bf16-policy-rounded-adapter.pt')
            e2=r2=0.
            for name,a in value.items():
                e2+=(rounded[name].double()-a.double()).square().sum().item()
                r2+=a.double().square().sum().item()
            report['adapter_rounding']=dict(source_state_digest=before,bf16_state_digest=after,
                repeated_conversion_digest_exact=True,global_relative_l2=(e2/r2)**.5,
                source_dtype_counts=dict(Counter(str(v.dtype) for v in value.values())),
                rounded_dtype_counts=dict(Counter(str(v.dtype) for v in rounded.values())))
            save();return rounded
        return value
    torch.load=load
    original_factory=peft.get_peft_model
    def factory(*args,**kwargs):
        model=original_factory(*args,**kwargs)
        trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
        if len(trainable)!=744 or any('.lora_' not in name for name in trainable):
            raise ValueError('Unexpected trainable parameter set')
        for parameter in trainable.values():parameter.data=parameter.data.to(torch.bfloat16)
        report['adapter_parameter_dtype_counts']=dict(Counter(str(p.dtype) for p in trainable.values()))
        report['adapter_parameter_bytes']=sum(p.numel()*p.element_size() for p in trainable.values())
        base=model.get_base_model()
        handles=[]
        def record_cast(key,label,tensor):
            if tensor.dtype!=torch.bfloat16:
                counts[key][label+' '+str(tensor.dtype)]+=1
        for name,module in base.named_modules():
            def before(mod,args,kwargs,label=name):
                if label.startswith('model.language_model.layers.') and label.count('.')==3:
                    hidden=args[0] if args else kwargs.get('hidden_states')
                    report['layer_calls'].append(dict(event='enter',module=label,dtype=str(hidden.dtype)))
                return cast_boundary(args,lambda t:record_cast('boundary_input_casts',label,t)),cast_boundary(
                    kwargs,lambda t:record_cast('boundary_input_casts',label,t))
            def after(mod,args,output,label=name):
                result=cast_boundary(output,lambda t:record_cast('boundary_output_casts',label,t))
                for t in tree_flatten(result)[0]:
                    if isinstance(t,torch.Tensor) and t.is_floating_point() and t.ndim>0:
                        counts['boundary_output_dtype_counts'][str(t.dtype)]+=1
                if label.startswith('model.language_model.layers.') and label.count('.')==3:
                    report['layer_calls'].append(dict(event='exit',module=label,dtype=str(result.dtype)))
                return result
            handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
            handles.append(module.register_forward_hook(after))
        report['hooked_modules']=len(handles)//2
        original_init=torch.autograd.graph.save_on_cpu.__init__
        def init(storage,*args,**kwargs):
            original_init(storage,*args,**kwargs)
            def observed(original,stored):
                counts['saved_original_dtype_counts'][str(original.dtype)]+=1
                counts['saved_storage_dtype_counts'][str(stored.dtype)]+=1
                counts['saved_original_bytes'][str(original.dtype)]+=original.numel()*original.element_size()
                counts['saved_storage_bytes'][str(stored.dtype)]+=stored.numel()*stored.element_size()
            storage.pack_hook,storage.unpack_hook=saved_bf16_hooks(storage.pack_hook,storage.unpack_hook,observed)
        torch.autograd.graph.save_on_cpu.__init__=init
        save();return model
    peft.get_peft_model=factory
    def finalize():
        floating={key:count for key,count in counts['saved_storage_dtype_counts'].items()
                  if 'float' in key}
        assert set(floating)=={'torch.bfloat16'}
        assert set(counts['boundary_output_dtype_counts'])=={'torch.bfloat16'}
        assert report['adapter_parameter_dtype_counts']=={'torch.bfloat16':744}
        assert report['adapter_rounding']['rounded_dtype_counts']=={'torch.bfloat16':744}
        assert report['layer_calls'] and all(row['dtype']=='torch.bfloat16' for row in report['layer_calls'])
        report['finalized']=True;save()
    return finalize


def compare_gradients(candidate_path, reference_path, output_path):
    """Compare raw BF16 gradients with FP32 baseline in float64; preserve both."""
    candidate=torch.load(candidate_path,map_location='cpu',weights_only=True)
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    assert set(candidate)==set(reference) and len(candidate)==744
    rows={};e2=r2=c2=dot=0.
    for name,a in reference.items():
        b=candidate[name]
        assert a.shape==b.shape and a.dtype==torch.float32 and b.dtype==torch.bfloat16
        aa,bb=a.double(),b.double();error=bb-aa
        en,rn,cn=error.square().sum().item(),aa.square().sum().item(),bb.square().sum().item()
        rows[name]=dict(bytes_exact=False,values_exact_after_promotion=bool(torch.equal(a,b.float())),
            reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),max_abs_error=error.abs().max().item(),
            relative_l2_error=(en/rn)**.5 if rn else None,finite=bool(torch.isfinite(b).all()))
        e2+=en;r2+=rn;c2+=cn;dot+=(aa*bb).sum().item()
    with Path(reference_path).open('rb') as f:reference_sha=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(tensors=rows,raw_tensors=744,bitwise_exact_tensors=0,
        values_exact_tensors=sum(row['values_exact_after_promotion'] for row in rows.values()),
        global_relative_l2_error=(e2/r2)**.5,cosine_similarity=dot/(r2*c2)**.5,
        max_abs_error=max(row['max_abs_error'] for row in rows.values()),
        max_tensor_relative_l2_error=max(row['relative_l2_error'] or 0 for row in rows.values()),
        all_finite=all(row['finite'] for row in rows.values()),bitwise_qualified=False,
        reference_sha256=reference_sha,reference_dtype='torch.float32',candidate_dtype='torch.bfloat16',
        comparison_dtype='torch.float64',precision_change_expected=True,tolerance_accepted=False)
    Path(output_path).write_text(json.dumps(report,indent=2)+'\n')
    return report
