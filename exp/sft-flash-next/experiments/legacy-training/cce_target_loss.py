"""Diagnostic frozen-head CCE exact candidate; production remains on the qualified v1 head.

Uses the official cce_exact preset, explicitly disabling both gradient filters.
No dense context/target-by-vocabulary output is retained. FP32 accumulation and
on-chip vocabulary tiles remain; a frozen head receives no weight gradient.
"""
import functools
import hashlib
import json
from pathlib import Path
import types

import torch
from target_only_head import target_positions


def cce_target_loss(hidden, weight, labels, reduction='mean', denominator=None):
    if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
        raise ValueError('Expected batch-one hidden states matching labels')
    if weight.requires_grad or weight.ndim != 2 or weight.shape[1] != hidden.shape[2]:
        raise ValueError('CCE candidate requires a frozen vocabulary head')
    if reduction not in ('sum', 'mean'):
        raise ValueError('Expected sum or mean reduction')
    if weight.dtype != torch.bfloat16 or hidden.device.type != 'cuda':
        raise ValueError('This diagnostic is qualified only for CUDA/BF16 head weights')
    from cut_cross_entropy import linear_cross_entropy
    positions, targets = target_positions(labels)
    # Match the native head's input rounding at the BF16 autocast boundary.
    selected = hidden[0].index_select(0, positions).to(weight.dtype).contiguous()
    with torch.autocast('cuda', enabled=False):
        loss = linear_cross_entropy(selected, weight, targets.flatten(),
            impl='cce_exact', reduction=reduction, filter_eps=None,
            filter_e_grad=False, filter_c_grad=False,
            accum_e_fp32=True, accum_c_fp32=True)
    if denominator is not None:
        if isinstance(denominator, torch.Tensor):
            denominator = denominator.to(loss.device)
        loss = loss / denominator
    return loss


def patch_model(base, report_path):
    if base.lm_head.bias is not None or base.lm_head.weight.requires_grad:
        raise ValueError('Expected frozen bias-free head')
    if getattr(base.config.text_config, 'output_router_logits', False):
        raise ValueError('Router auxiliary loss requires separate qualification')
    import cut_cross_entropy
    from cut_cross_entropy.cce_utils import CCEPresets
    options = CCEPresets.build_for_impl('cce_exact', dict(filter_eps=None,
        accum_e_fp32=True, accum_c_fp32=True, filter_e_grad=False, filter_c_grad=False))
    assert options['filter_eps'] is None and not options['filter_e_grad'] and not options['filter_c_grad']
    assert options['accum_e_fp32'] and options['accum_c_fp32']
    original_forward, original_head, original_loss = base.forward, base.lm_head.forward, base.loss_function
    state = {'labels': None}
    report = {'objective': 'cce_target_exact', 'diagnostic_only': True,
              'cce_version': cut_cross_entropy.__version__, 'implementation': 'cce_exact',
              'effective_options': options,
              'frozen_head': True, 'head_calls': [], 'vocabulary_saved_tensor_calls': []}

    def save():
        Path(report_path).write_text(json.dumps(report, indent=2) + '\n')

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get('labels')
        if labels is None:
            return original_forward(*args, **kwargs)
        if args or kwargs.get('logits_to_keep', 0) != 0 or kwargs.get('return_dict') is False:
            raise ValueError('CCE diagnostic expects keyword inputs, all decoder rows and ModelOutput')
        if state['labels'] is not None:
            raise RuntimeError('Nested CCE forward')
        target_positions(labels)
        state['labels'] = labels
        try:
            output = original_forward(*args, **kwargs)
            output.logits = None
            output['logits'] = None  # Clear both ModelOutput attribute and mapping.
            return output
        finally:
            state['labels'] = None

    def head(self, hidden):
        if state['labels'] is None:
            return original_head(hidden)
        positions, _ = target_positions(state['labels'])
        targets, width, vocab = positions.numel(), hidden.shape[-1], self.weight.shape[0]
        report['head_calls'].append({'context_tokens': hidden.shape[1],
            'target_tokens': targets, 'vocabulary': vocab, 'hidden_width': width,
            'hidden_dtype': str(hidden.dtype), 'cce_input_dtype': str(self.weight.dtype),
            'prediction_positions': positions.cpu().tolist(),
            'selected_hidden_bytes': targets * width * 2,
            'dense_logits_materialized': False, 'native_backward_scratch_rows': 0})
        save()
        return hidden

    def loss(*, logits, labels, **kwargs):
        if state['labels'] is None:
            return original_loss(logits=logits, labels=labels, **kwargs)
        if 'shift_labels' in kwargs:
            raise ValueError('Explicit shifted labels conflict with the target mask')
        denominator = kwargs.get('num_items_in_batch')
        return cce_target_loss(logits, base.lm_head.weight, labels,
                                 'mean' if denominator is None else 'sum', denominator)

    original_init = torch.autograd.graph.save_on_cpu.__init__
    def observed_init(storage, *args, **kwargs):
        original_init(storage, *args, **kwargs)
        original_pack = storage.pack_hook
        def pack(tensor):
            packed = original_pack(tensor)
            if tensor.ndim >= 2 and tensor.shape[-1] == base.lm_head.weight.shape[0]:
                report['vocabulary_saved_tensor_calls'].append({
                    'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'bytes': tensor.numel() * tensor.element_size(),
                    'saved_device': str(packed[1].device)})
                save()
            return packed
        storage.pack_hook = pack
    torch.autograd.graph.save_on_cpu.__init__ = observed_init
    base.forward, base.lm_head.forward = types.MethodType(forward, base), types.MethodType(head, base.lm_head)
    base.loss_function = loss
    save()
    def restore():
        base.forward, base.lm_head.forward, base.loss_function = original_forward, original_head, original_loss
        torch.autograd.graph.save_on_cpu.__init__ = original_init
    return report, restore


def install_for_capture(report_path):
    import peft
    original = peft.get_peft_model
    def factory(*args, **kwargs):
        model = original(*args, **kwargs)
        patch_model(model.get_base_model(), report_path)
        return model
    peft.get_peft_model = factory


def compare_gradients(candidate_path, reference_path, output_path):
    """Save numerical differences; never silently turn a tolerance into acceptance."""
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    if set(candidate) != set(reference) or len(reference) != 744:
        raise ValueError('Adapter gradient names/count differ')
    rows, squared_error, squared_reference, squared_candidate, dot = {}, 0., 0., 0., 0.
    for name, expected in reference.items():
        observed = candidate[name]
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            raise ValueError('Adapter gradient metadata differs: ' + name)
        a, b = expected.double(), observed.double()
        error = b - a
        e2, r2, c2 = error.square().sum().item(), a.square().sum().item(), b.square().sum().item()
        rows[name] = {'bytes_exact': torch.equal(expected.view(torch.uint8), observed.view(torch.uint8)),
                      'max_abs_error': error.abs().max().item(),
                      'relative_l2_error': (e2 / r2) ** .5 if r2 else None,
                      'finite': bool(torch.isfinite(observed).all())}
        squared_error += e2; squared_reference += r2; squared_candidate += c2
        dot += (a * b).sum().item()
    with open(reference_path, 'rb') as stream:
        reference_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'tensors': rows, 'raw_tensors': len(rows),
              'bitwise_exact_tensors': sum(row['bytes_exact'] for row in rows.values()),
              'global_relative_l2_error': (squared_error / squared_reference) ** .5,
              'cosine_similarity': dot / (squared_reference * squared_candidate) ** .5,
              'max_abs_error': max(row['max_abs_error'] for row in rows.values()),
              'max_tensor_relative_l2_error': max(row['relative_l2_error'] or 0 for row in rows.values()),
              'all_finite': all(row['finite'] for row in rows.values()),
              'bitwise_qualified': all(row['bytes_exact'] for row in rows.values()),
              'reference_sha256': reference_sha}
    Path(output_path).write_text(json.dumps(report, indent=2) + '\n')
    return report
