"""Opt3 only: native causal selection, direct dense additive bias, unchanged CCE.

Diagnostic installation is limited to batch-one, unpadded, cache-free SDPA.
The source guard pins the installed native functions before patching. No model
weights, selected-token arithmetic, loss implementation, or optimizer is changed.
"""
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
from native_mask_storage import indexer_with_direct_bias, language_with_lazy_mask

NATIVE_HASHES = {
    'Qwen4ExpTextQSAIndexer': 'cc3b0347832752a530a86a4b69e5147daea9b58d29f756e8f60a580bf519f093',
    'Qwen4ExpTextAttention': '4b4c94843d05bb342787497c124f9405f62cc84edc36c3011e4ee457c253a77a',
    'Qwen4ExpTextModel': '90dc3a00f32e784417cebf8dd5a10657b18d196e22e0d374a3b049c2527553cc',
}


def check_native_sources():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as native
    for name, expected in NATIVE_HASHES.items():
        observed = hashlib.sha256(inspect.getsource(getattr(native, name).forward).encode()).hexdigest()
        if observed != expected:
            raise RuntimeError('Unsupported native attention source: ' + name)
    return native


def patch_model(base, report_path):
    check_native_sources()
    lm = base.model.language_model
    if lm.config._attn_implementation != 'sdpa':
        raise ValueError('Opt3 requires native SDPA')
    if hasattr(lm, 'native_mask_forward'):
        raise ValueError('Opt3 already installed')
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'implementation': 'opt3_direct_dense_bias', 'native_source_sha256': NATIVE_HASHES,
              'selection_arithmetic_unchanged': True, 'bias_dtype_source': 'hidden_states.dtype',
              'dense_causal_mask': False, 'dense_selected_boolean_mask': False,
              'dense_combined_boolean_mask': False, 'sparse_attention_kernel': False,
              'batch_one_unpadded_cache_free': True, 'bias_calls': []}
    def observe(module, inputs, output):
        hidden = inputs[0]
        if output.dtype != hidden.dtype or tuple(output.shape) != (1, 1, hidden.shape[1], hidden.shape[1]):
            raise RuntimeError('Unexpected Opt3 bias metadata')
        report['bias_calls'].append({'layer': module.layer_idx, 'shape': list(output.shape),
                                    'dtype': str(output.dtype), 'logical_bytes': output.numel() * output.element_size(),
                                    'storage_bytes': output.untyped_storage().nbytes(),
                                    'stride': list(output.stride()), 'requires_grad': output.requires_grad})
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    count = 0
    for module in lm.modules():
        if module.__class__.__name__ == 'Qwen4ExpTextQSAIndexer':
            module.forward = MethodType(indexer_with_direct_bias(module.forward), module)
            module.register_forward_hook(observe)
            count += 1
    if count != 12:
        raise RuntimeError(f'Expected 12 native indexed attention layers, got {count}')
    report['indexed_layers'] = count
    lm.native_mask_forward = lm.forward
    lm.forward = MethodType(language_with_lazy_mask, lm)
    report_path.write_text(json.dumps(report, indent=2) + '\n')


def install_for_capture(report_path):
    import peft
    original = peft.get_peft_model
    def factory(base, *args, **kwargs):
        model = original(base, *args, **kwargs)  # CCE installer remains inside this wrapper.
        patch_model(base, report_path)
        return model
    peft.get_peft_model = factory
