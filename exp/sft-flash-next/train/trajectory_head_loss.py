"""Experimental native CE for arbitrary assistant spans.

This has its own qualification identity. It does not alter the fixed-reference
head operator. Row-count changes require measured native gradient comparisons.
"""
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from transformers.loss.loss_utils import ForCausalLMLoss


def supervised_positions(labels):
    """Return hidden indices and labels in the native next-token convention."""
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape[1] < 2:
        raise ValueError('Expected batch-one labels with at least two tokens')
    if labels[0, 0].item() != -100:
        raise ValueError('The first token cannot have a next-token prediction')
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError('Trajectory has no supervised targets')
    return positions, labels[0, positions + 1]


class _TrajectoryNativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, backward_rows, denominator):
        positions, targets = supervised_positions(labels)
        if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
            raise ValueError('Hidden states must match batch-one labels')
        if weight.ndim != 2 or weight.shape[1] != hidden.shape[2] or weight.requires_grad:
            raise ValueError('Expected a frozen vocabulary head with matching width')
        if denominator <= 0 or not isinstance(denominator, int):
            raise ValueError('Loss denominator must be a positive integer')
        if backward_rows is not None and (not isinstance(backward_rows, int)
                                          or backward_rows < targets.numel()):
            raise ValueError('Backward rows must cover all assistant targets')
        logits = F.linear(hidden[:, positions], weight)
        loss = ForCausalLMLoss(logits, labels, vocab_size=weight.shape[0],
                              shift_labels=targets, num_items_in_batch=denominator)
        ctx.save_for_backward(logits, weight.to(logits.dtype), positions, targets)
        ctx.tokens, ctx.hidden_dtype = hidden.shape[1], hidden.dtype
        ctx.rows, ctx.denominator = backward_rows, denominator
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        logits, weight, positions, targets = ctx.saved_tensors
        with torch.enable_grad():
            values = logits.detach().requires_grad_(True)
            local = ForCausalLMLoss(values, targets, vocab_size=weight.shape[0],
                                    shift_labels=targets, num_items_in_batch=ctx.denominator)
            dlogits, = torch.autograd.grad(local, values, upstream)
        rows = ctx.tokens if ctx.rows is None else ctx.rows
        padded = torch.zeros((rows, weight.shape[0]), dtype=dlogits.dtype, device=weight.device)
        if ctx.rows is None:
            padded.index_copy_(0, positions, dlogits[0])
        else:
            padded[:targets.numel()].copy_(dlogits[0])
        del values, local, logits, dlogits
        with torch.autocast(weight.device.type, enabled=False):
            gradient = torch.mm(padded, weight)
        if ctx.rows is None:
            hidden_gradient = gradient.unsqueeze(0)
        else:
            hidden_gradient = torch.zeros((1, ctx.tokens, weight.shape[1]),
                                           dtype=gradient.dtype, device=weight.device)
            hidden_gradient[0].index_copy_(0, positions, gradient[:targets.numel()])
        return hidden_gradient.to(ctx.hidden_dtype), None, None, None, None


def trajectory_loss(hidden, weight, labels, backward_rows=None, reduction='sum'):
    """Compute summed CE for accumulation, or a trajectory target-token mean.

    The final ignored row and every user/tool/image position remain in decoder
    context. Only their unsupervised vocabulary-head rows are omitted.
    """
    if reduction not in ('sum', 'mean'):
        raise ValueError('Expected sum or mean reduction')
    denominator = 1 if reduction == 'sum' else supervised_positions(labels)[0].numel()
    return _TrajectoryNativeBackward.apply(hidden, weight, labels, backward_rows, denominator)


def trajectory_objective(model, batch, labels, backward_rows=None, reduction='sum'):
    base = model.get_base_model()
    hidden = base.model(**batch, use_cache=False).last_hidden_state
    return trajectory_loss(hidden, base.lm_head.weight, labels, backward_rows, reduction)
