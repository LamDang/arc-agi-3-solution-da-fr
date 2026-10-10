"""Explicit frozen vocabulary objectives; supports interleaved supervised spans.

Arithmetic ported from the archived target_only_head/CCE/FLCE components.
No head replacement, factory mutation or process-global saved-tensor hooks.
"""
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

def objective(model, batch, labels, implementation):
    if implementation == 'native':
        return model(**batch, labels=labels, use_cache=False).loss
    base = model.get_base_model()
    hidden = base.model(**batch, use_cache=False).last_hidden_state
    if implementation == 'target':
        positions, targets = target_positions(labels)
        logits, _ = target_logits(hidden, base.lm_head.weight, labels)
        return target_cross_entropy(logits, labels, positions, targets)
    if implementation == 'cce_exact':
        return cce_target_loss(hidden, base.lm_head.weight, labels)
    if implementation == 'liger_flce':
        return liger_target_loss(hidden, base.lm_head.weight, labels)
    raise ValueError('Unknown objective: ' + implementation)
