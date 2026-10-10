"""Target-mask vocabulary projection with native-shaped frozen-head backward.

Supports batch-one trajectories with any interleaving of ignored context and
supervised spans. Decoder context is unchanged. Only vocabulary-head rows are
selected. The native T-by-V BF16 backward scratch remains intentionally intact
until a smaller backward shape earns its own numerical qualification.
"""
import functools
import json
from pathlib import Path
import types

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def target_positions(labels):
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape[1] < 2:
        raise ValueError('Expected batch-one labels with at least two tokens')
    if labels[0, 0].item() != -100:
        raise ValueError('The first token has no preceding prediction position')
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('No supervised next-token targets')
    return positions, labels.index_select(1, positions + 1).contiguous()


class _MaskedFrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, positions):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError('Expected batch-one hidden states')
        if weight.requires_grad or weight.ndim != 2 or weight.shape[1] != hidden.shape[2]:
            raise ValueError('Expected a frozen vocabulary weight with matching width')
        if positions.ndim != 1 or not positions.numel():
            raise ValueError('Expected at least one prediction position')
        # CUDA autocast follows the native head's dtype conversion.
        logits = F.linear(hidden.index_select(1, positions), weight)
        ctx.save_for_backward(weight.to(logits.dtype), positions)
        ctx.tokens, ctx.hidden_dtype = hidden.shape[1], hidden.dtype
        return logits

    @staticmethod
    @once_differentiable
    def backward(ctx, dlogits):
        weight, positions = ctx.saved_tensors
        # Preserve native GEMM row count, positions, dtype and stride. Changing
        # the row count previously changed BF16 results in real-model tests.
        padded = torch.zeros((ctx.tokens, weight.shape[0]),
                             dtype=dlogits.dtype, device=dlogits.device)
        padded.index_copy_(0, positions, dlogits[0])
        with torch.autocast(weight.device.type, enabled=False):
            gradient = torch.mm(padded, weight).unsqueeze(0)
        return gradient.to(ctx.hidden_dtype), None, None


def target_logits(hidden, weight, labels):
    positions, targets = target_positions(labels)
    if hidden.shape[:2] != labels.shape:
        raise ValueError('Labels must match hidden token dimensions')
    return _MaskedFrozenHead.apply(hidden, weight, positions), targets


def target_cross_entropy(logits, labels, positions, targets, num_items_in_batch=None):
    """Native row-wise log-softmax plus NLL at original reduction positions.

    Omitting ignored rows from the scalar reduction changes its summation order.
    A T-by-1 NLL input preserves the original row positions/ignore mask without
    a T-by-vocabulary log-probability matrix. Vocabulary dimension does not
    participate in NLL's scalar sum: it only selects each target log-probability.
    Whole-model CUDA equivalence remains an empirical qualification gate.
    """
    values = F.log_softmax(logits.float().reshape(-1, logits.shape[-1]), dim=-1)
    selected = values.gather(1, targets.reshape(-1, 1)).flatten()
    full = torch.zeros(labels.shape[1], device=values.device, dtype=values.dtype)
    full = full.index_copy(0, positions, selected)
    native_mask = torch.full((labels.shape[1],), -100, device=values.device, dtype=torch.long)
    native_mask.index_fill_(0, positions, 0)
    loss = F.nll_loss(full.unsqueeze(1), native_mask, ignore_index=-100,
                      reduction='mean' if num_items_in_batch is None else 'sum')
    if num_items_in_batch is not None:
        denominator = num_items_in_batch.to(loss.device) if isinstance(num_items_in_batch, torch.Tensor) else num_items_in_batch
        loss = loss / denominator
    return loss


def patch_model(base, report_path=None):
    """Keep native HF/PEFT forward; replace only head rows and loss alignment."""
    if base.lm_head.bias is not None or base.lm_head.weight.requires_grad:
        raise ValueError('This candidate requires the frozen bias-free head')
    if getattr(base.config.text_config, 'output_router_logits', False):
        raise ValueError('Router auxiliary-loss qualification is separate')
    original_forward, original_head = base.forward, base.lm_head.forward
    original_loss = base.loss_function
    state = {'selection': None}
    report = {'objective': 'target_only_mask_native_backward',
              'native_backward_rows': True, 'head_calls': [],
              'vocabulary_saved_tensor_calls': []}

    def save_report():
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2) + '\n')

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get('labels')
        if labels is None:
            return original_forward(*args, **kwargs)
        if args or kwargs.get('logits_to_keep', 0) != 0:
            raise ValueError('Target-mask training expects keyword inputs and full decoder rows')
        if state['selection'] is not None:
            raise RuntimeError('Nested target-mask model forward')
        positions, targets = target_positions(labels)
        state['selection'] = (positions, targets, labels.shape[1])
        try:
            return original_forward(*args, **kwargs)
        finally:
            state['selection'] = None

    def head(self, hidden):
        if state['selection'] is None:
            return original_head(hidden)
        positions, targets, tokens = state['selection']
        if hidden.shape[:2] != (1, tokens):
            raise ValueError('Native decoder rows differ from mask labels')
        logits = _MaskedFrozenHead.apply(hidden, self.weight, positions)
        report['head_calls'].append({
            'context_tokens': tokens, 'target_tokens': positions.numel(),
            'vocabulary': self.weight.shape[0], 'hidden_dtype': str(hidden.dtype),
            'logits_shape': list(logits.shape), 'logits_dtype': str(logits.dtype),
            'prediction_positions': positions.detach().cpu().tolist(),
            'logits_bytes': logits.numel() * logits.element_size(),
            'full_logits_bytes_at_same_dtype': tokens * self.weight.shape[0] * logits.element_size(),
            'backward_scratch_rows': tokens})
        save_report()
        return logits

    def loss(*, logits, labels, **kwargs):
        if state['selection'] is None:
            return original_loss(logits=logits, labels=labels, **kwargs)
        if 'shift_labels' in kwargs:
            raise ValueError('Explicit shifted labels conflict with target mask')
        positions, targets, _ = state['selection']
        return target_cross_entropy(logits, labels, positions, targets,
                                    kwargs.get('num_items_in_batch'))

    # Observe existing CPU pack operations; do not replace their copy policy.
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
                    'source_device': str(tensor.device),
                    'saved_device': str(packed[1].device)})
                save_report()
            return packed
        storage.pack_hook = pack
    torch.autograd.graph.save_on_cpu.__init__ = observed_init
    base.forward = types.MethodType(forward, base)
    base.lm_head.forward = types.MethodType(head, base.lm_head)
    base.loss_function = loss
    save_report()
    def restore():
        base.forward, base.lm_head.forward = original_forward, original_head
        base.loss_function = original_loss
        torch.autograd.graph.save_on_cpu.__init__ = original_init
    return report, restore


def install_for_capture(report_path):
    """Install before the unchanged native capture imports PEFT's factory."""
    import peft
    original_factory = peft.get_peft_model
    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        patch_model(model.get_base_model(), report_path)
        return model
    peft.get_peft_model = factory
