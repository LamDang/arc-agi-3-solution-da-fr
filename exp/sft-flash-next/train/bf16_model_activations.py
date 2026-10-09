"""Corrected Opt5: BF16 LoRA and declared model activations only.

Keep native FP32 numerical statistics and all autograd saved tensors unchanged.
FP32 operator intermediates are permitted. Never blanket-cast module arguments,
auxiliary outputs, masks, position tensors, loss or saved autograd state.
The original v6 implementation is retained solely as rejected-run provenance.
"""
from collections import Counter
import hashlib
import inspect
import traceback
import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_map
from bf16_activation_policy import state_digest, compare_gradients

MOE_SOURCE_SHA256 = '2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c'


def cast_boundary(value, observe=None):
    def cast(tensor):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.ndim == 0:
            return tensor
        if observe is not None:
            observe(tensor)
        return tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
    return tree_map(cast, value)


def cast_hidden_input(args, kwargs, observe=None):
    """Only the declared hidden-state argument; never masks, positions or stats."""
    if args:
        return (cast_boundary(args[0], observe), *args[1:]), kwargs
    kwargs = dict(kwargs)
    key = next((k for k in ('hidden_states', 'x', 'hyper_input') if k in kwargs), None)
    if key is None:
        raise ValueError('Missing declared model activation input')
    kwargs[key] = cast_boundary(kwargs[key], observe)
    return args, kwargs


def cast_hidden_output(output, role, observe=None):
    if role == 'attention':
        # The second return value is attention weights, not hidden states.
        return (cast_boundary(output[0], observe), *output[1:])
    if role == 'residual' and isinstance(output, tuple):
        # Mixed hidden state and hyper residual; keep injection coefficients native.
        return (cast_boundary(output[0], observe), cast_boundary(output[1], observe), *output[2:])
    return cast_boundary(output, observe)


def activation_role(name, module):
    kind = type(module).__name__
    if kind == 'Qwen4ExpTextAttention':
        return 'attention'
    if kind == 'Qwen4ExpTextGatedResidual':
        return 'residual'
    if kind in ('Qwen4ExpTextDecoderLayer', 'Qwen4ExpTextGatedDeltaNet',
                'Qwen4ExpTextMLP', 'Qwen4ExpTextSparseMoeBlock',
                'Qwen4ExpTextRMSNorm', 'Qwen4ExpTextRMSNormGated',
                'Qwen4ExpTextPLELayer') or name.endswith('.mlp.experts'):
        return 'hidden'
    return None


def observe_saved_hooks(pack, unpack, observe=None):
    """Observation only: no value conversion, quantization or dtype metadata."""
    def packed(tensor):
        payload = pack(tensor)
        if observe is not None:
            observe(tensor, payload, unpack)
        return payload
    return packed, unpack


def qualify_cpu(output_dir):
    out = Path(output_dir)
    x = torch.tensor([1.0007, -0.31317, 0.000123456, 19.037], requires_grad=True)
    scalar = torch.tensor(0.123456, dtype=torch.float32)
    lse = torch.tensor([12.345678, 21.234567], dtype=torch.float32)
    mask = torch.tensor([[0., -torch.inf]], dtype=torch.float32)
    indices = torch.tensor([1, 3], dtype=torch.int64)
    args, kwargs = cast_hidden_input((x, mask), {'lse': lse, 'indices': indices})
    assert args[0].dtype == torch.bfloat16 and args[1] is mask
    assert kwargs['lse'] is lse and kwargs['indices'] is indices
    output = cast_hidden_output((x, lse), 'attention')
    assert output[0].dtype == torch.bfloat16 and output[1] is lse
    assert cast_boundary(scalar) is scalar
    storage = torch.autograd.graph.save_on_cpu(pin_memory=False)
    records = []
    def observed(tensor, payload, unpack):
        restored = unpack(payload)
        assert restored.dtype == tensor.dtype and torch.equal(restored, tensor)
        records.append((str(tensor.dtype), str(restored.dtype)))
    pack, unpack = observe_saved_hooks(storage.pack_hook, storage.unpack_hook, observed)
    for tensor in (x, x.to(torch.bfloat16), lse, scalar, indices):
        assert torch.equal(unpack(pack(tensor)), tensor)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = x.square().sum()
        loss.backward()
    assert torch.equal(x.grad, 2*x.detach())
    # CCE-style exp(logit - LSE) must consume the exact original FP32 LSE.
    exact = torch.exp(torch.tensor([12., 21.]) - lse)
    assert torch.equal(torch.exp(torch.tensor([12., 21.]) - unpack(pack(lse))), exact)
    assert not torch.equal(torch.exp(torch.tensor([12., 21.]) - lse.bfloat16().float()), exact)
    report = dict(passed=True, device='cpu', named_hidden_activation_bf16=True,
        auxiliary_arguments_unchanged=True, scalar_loss_preserved_fp32=True,
        saved_tensors_original_dtype_and_values=True, fp32_lse_exact=True,
        fp32_saved_backward_exact=True, optimizer_updates=0, records=records)
    (out/'bf16-policy-operator-check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def install_for_capture(adapter_path, launch_dir, report_path, training=False):
    import peft
    import auto_round.modeling.fused_moe.moe_experts_interface as moe
    source_path=Path(inspect.getfile(moe));source=source_path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=MOE_SOURCE_SHA256:
        raise RuntimeError('Unreviewed AutoRound MoE source')
    launch=Path(launch_dir)
    (launch/'bf16-policy-moe-source.py').write_bytes(source)
    path=Path(report_path)
    report=dict(implementation='opt5_bf16_model_activations_native_statistics',diagnostic_only=True,
        model_activation_dtype='torch.bfloat16',saved_tensor_storage='native dtype and values',
        scalar_loss_dtype='torch.float32',fp32_intermediate_computation_allowed=True,
        saved_fp32_statistics_compressed=False,saved_tensor_conversion=False,
        fp32_saved_tensors=[],cce_fp32_lse=[],activation_roles={},
        autocast_dtype='torch.bfloat16',optimizer_updates=0,training=training,moe_source_sha256=MOE_SOURCE_SHA256,
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
        if training:
            assert adapter_path is None
            source={name:value.detach().cpu().clone() for name,value in peft.get_peft_model_state_dict(model).items()}
            assert len(source)==744
            assert all(torch.count_nonzero(value)==0 for name,value in source.items() if 'lora_B' in name)
            assert all(torch.count_nonzero(value)>0 for name,value in source.items() if 'lora_A' in name)
            rounded={name:value.to(torch.bfloat16) for name,value in source.items()}
            repeated={name:value.to(torch.bfloat16) for name,value in source.items()}
            assert state_digest(rounded)==state_digest(repeated)
            torch.save(source,launch/'bf16-policy-clean-fp32-source-adapter.pt')
            torch.save(rounded,launch/'bf16-policy-rounded-adapter.pt')
            report['adapter_rounding']=dict(source_state_digest=state_digest(source),
                bf16_state_digest=state_digest(rounded),repeated_conversion_digest_exact=True,
                source_dtype_counts=dict(Counter(str(v.dtype) for v in source.values())),
                rounded_dtype_counts=dict(Counter(str(v.dtype) for v in rounded.values())),
                initialization='seeded PEFT random A / zero B; no supplied trained adapter')
        for parameter in trainable.values():parameter.data=parameter.data.to(torch.bfloat16)
        report['adapter_parameter_dtype_counts']=dict(Counter(str(p.dtype) for p in trainable.values()))
        report['adapter_parameter_bytes']=sum(p.numel()*p.element_size() for p in trainable.values())
        base=model.get_base_model()
        handles=[]
        def record_cast(key,label,tensor):
            if tensor.dtype!=torch.bfloat16:
                counts[key][label+' '+str(tensor.dtype)]+=1
        for name,module in base.named_modules():
            role = activation_role(name, module)
            if role is None:
                continue
            report['activation_roles'][name] = dict(role=role, module_type=type(module).__name__)
            def before(mod,args,kwargs,label=name):
                args, kwargs = cast_hidden_input(args, kwargs,
                    lambda t:record_cast('boundary_input_casts',label,t))
                if label.startswith('model.language_model.layers.') and label.count('.')==3:
                    hidden=args[0] if args else kwargs['hidden_states']
                    report['layer_calls'].append(dict(event='enter',module=label,dtype=str(hidden.dtype)))
                return args, kwargs
            def after(mod,args,output,label=name,output_role=role):
                result=cast_hidden_output(output,output_role,
                    lambda t:record_cast('boundary_output_casts',label,t))
                hidden = result[0] if isinstance(result, tuple) else result
                assert isinstance(hidden, torch.Tensor) and hidden.dtype==torch.bfloat16
                counts['boundary_output_dtype_counts'][str(hidden.dtype)]+=1
                if label.startswith('model.language_model.layers.') and label.count('.')==3:
                    report['layer_calls'].append(dict(event='exit',module=label,dtype=str(hidden.dtype)))
                return result
            handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
            handles.append(module.register_forward_hook(after))
        report['hooked_modules']=len(handles)//2
        original_init=torch.autograd.graph.save_on_cpu.__init__
        def init(storage,*args,**kwargs):
            original_init(storage,*args,**kwargs)
            def observed(original,payload,unpack):
                stored=payload[1]  # stock save_on_cpu: (original_device, CPU_tensor)
                assert stored.dtype==original.dtype
                counts['saved_original_dtype_counts'][str(original.dtype)]+=1
                counts['saved_storage_dtype_counts'][str(stored.dtype)]+=1
                counts['saved_original_bytes'][str(original.dtype)]+=original.numel()*original.element_size()
                counts['saved_storage_bytes'][str(stored.dtype)]+=stored.numel()*stored.element_size()
                if original.dtype==torch.float32:
                    stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name)
                           for f in traceback.extract_stack()[:-1]][-12:]
                    row=dict(shape=list(original.shape),dtype=str(original.dtype),
                             bytes=original.numel()*original.element_size(),stack=stack)
                    report['fp32_saved_tensors'].append(row)
                    if original.ndim==1 and any(f['file']=='cce.py' for f in stack):
                        restored=unpack(payload)
                        assert restored.dtype==torch.float32 and torch.equal(restored,original)
                        row['roundtrip_values_exact']=True
                        row['semantic']='CCE log-sum-exp'
                        report['cce_fp32_lse'].append(row)
            storage.pack_hook,storage.unpack_hook=observe_saved_hooks(storage.pack_hook,storage.unpack_hook,observed)
        torch.autograd.graph.save_on_cpu.__init__=init
        save();return model
    peft.get_peft_model=factory
    def finalize(optimizer_updates=0):
        report['optimizer_updates']=optimizer_updates
        assert counts['saved_original_dtype_counts']==counts['saved_storage_dtype_counts']
        assert counts['saved_original_bytes']==counts['saved_storage_bytes']
        assert report['cce_fp32_lse'] and all(row['roundtrip_values_exact'] for row in report['cce_fp32_lse'])
        assert set(counts['boundary_output_dtype_counts'])=={'torch.bfloat16'}
        assert report['adapter_parameter_dtype_counts']=={'torch.bfloat16':744}
        assert report['adapter_rounding']['rounded_dtype_counts']=={'torch.bfloat16':744}
        assert report['layer_calls'] and all(row['dtype']=='torch.bfloat16' for row in report['layer_calls'])
        report['finalized']=True;save()
    return finalize


