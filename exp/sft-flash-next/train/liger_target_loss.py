"""Diagnostic Liger FLCE candidate; production remains on the qualified v1 head.

No dense context/target-by-vocabulary output is retained. Liger still computes
small transient vocabulary chunks and FP32 reductions inside its CE kernel.
"""
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
import types

import torch
from target_only_head import target_positions


def liger_target_loss(hidden, weight, labels, reduction='mean', denominator=None):
    if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
        raise ValueError('Expected batch-one hidden states matching labels')
    if weight.requires_grad or weight.ndim != 2 or weight.shape[1] != hidden.shape[2]:
        raise ValueError('Liger candidate requires a frozen vocabulary head')
    if reduction not in ('sum', 'mean'):
        raise ValueError('Expected sum or mean reduction')
    if weight.dtype != torch.bfloat16 or hidden.device.type != 'cuda':
        raise ValueError('This diagnostic is qualified only for CUDA/BF16 head weights')
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    positions, targets = target_positions(labels)
    # Native CUDA autocast rounds FP32 hidden states to BF16 at the head.
    # Keep this explicit so FLCE cannot retain an FP32 gradient-input buffer.
    selected = hidden[0].index_select(0, positions).to(weight.dtype).contiguous()
    with torch.autocast('cuda', enabled=False):
        loss, _, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            selected, weight, targets.flatten(), None, None, -100, 0., 0.,
            reduction, None, False, None, False, False, False,
            'triton', None, 1)
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
    original_forward, original_head, original_loss = base.forward, base.lm_head.forward, base.loss_function
    state = {'labels': None}
    report = {'objective': 'liger_target_flce', 'diagnostic_only': True,
              'liger_version': importlib.metadata.version('liger-kernel'),
              'frozen_head': True, 'head_calls': [], 'vocabulary_saved_tensor_calls': []}

    def save():
        Path(report_path).write_text(json.dumps(report, indent=2) + '\n')

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get('labels')
        if labels is None:
            return original_forward(*args, **kwargs)
        if args or kwargs.get('logits_to_keep', 0) != 0 or kwargs.get('return_dict') is False:
            raise ValueError('FLCE diagnostic expects keyword inputs, all decoder rows and ModelOutput')
        if state['labels'] is not None:
            raise RuntimeError('Nested FLCE forward')
        target_positions(labels)
        state['labels'] = labels
        try:
            output = original_forward(*args, **kwargs)
            output.logits = None  # Training returns a loss, never hidden states mislabeled as logits.
            return output
        finally:
            state['labels'] = None

    def head(self, hidden):
        if state['labels'] is None:
            return original_head(hidden)
        positions, _ = target_positions(state['labels'])
        targets, width, vocab = positions.numel(), hidden.shape[-1], self.weight.shape[0]
        expansion = (vocab + width - 1) // width
        rows = (targets + expansion - 1) // expansion
        chunk = min(targets, 1 << (rows - 1).bit_length())
        report['head_calls'].append({'context_tokens': hidden.shape[1],
            'target_tokens': targets, 'vocabulary': vocab, 'hidden_width': width,
            'hidden_dtype': str(hidden.dtype), 'flce_input_dtype': str(self.weight.dtype),
            'prediction_positions': positions.cpu().tolist(),
            'chunk_rows': chunk, 'chunk_logits_bytes': chunk * vocab * 2,
            'selected_hidden_gradient_bytes': targets * width * 2,
            'dense_logits_materialized': False, 'native_backward_scratch_rows': 0})
        save()
        return hidden

    def loss(*, logits, labels, **kwargs):
        if state['labels'] is None:
            return original_loss(logits=logits, labels=labels, **kwargs)
        if 'shift_labels' in kwargs:
            raise ValueError('Explicit shifted labels conflict with the target mask')
        denominator = kwargs.get('num_items_in_batch')
        return liger_target_loss(logits, base.lm_head.weight, labels,
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
