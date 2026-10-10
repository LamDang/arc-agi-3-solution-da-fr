"""Selected native CE with an explicitly controlled LM-head backward shape.

Candidate only: whole-model gradient qualification is required. The frozen
head is evaluated at supervised positions. By default its backward pads ignored
rows with zeros before the original full-shape matrix multiplication. Explicit
row counts are separate empirical candidates: changing the row count can change
BF16 rounding. Both modes avoid full-context FP32 cross-entropy tensors.
"""
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from transformers.loss.loss_utils import ForCausalLMLoss


class _SelectedNativeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, prompt, backward_rows):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError('This candidate supports batch-one hidden states')
        if labels.shape != hidden.shape[:2] or not 0 < prompt < hidden.shape[1]:
            raise ValueError('Invalid labels or prompt boundary')
        if weight.requires_grad:
            raise ValueError('The vocabulary head must be frozen')
        if weight.ndim != 2 or weight.shape[1] != hidden.shape[2]:
            raise ValueError(f'Head/hidden width mismatch: {tuple(weight.shape)}, {tuple(hidden.shape)}')
        if not bool((labels[:, :prompt] == -100).all()):
            raise ValueError('All omitted prompt labels must be ignored')
        targets = labels[:, prompt:].contiguous()
        if backward_rows is not None and (not isinstance(backward_rows, int)
                                         or backward_rows < targets.numel()):
            raise ValueError('Explicit backward rows must cover every target')
        logits = F.linear(hidden[:, prompt-1:-1], weight)
        loss = ForCausalLMLoss(logits, labels, vocab_size=weight.shape[0],
                               shift_labels=targets)
        # F.linear performs native autocast above. Its ordinary backward uses
        # the cast weight, then casts the hidden gradient back to input dtype.
        # PEFT may supply FP32 decoder states to a frozen BF16 vocabulary head.
        ctx.save_for_backward(logits, weight.to(dtype=logits.dtype), targets)
        ctx.tokens, ctx.prompt = hidden.shape[1], prompt
        ctx.hidden_dtype = hidden.dtype
        ctx.backward_rows = backward_rows
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        logits, weight, targets = ctx.saved_tensors
        # Ordinary native CE autograd determines dLogits, including upstream
        # loss scaling. Do not scale after the BF16 head multiplication.
        with torch.enable_grad():
            values = logits.detach().requires_grad_(True)
            local_loss = ForCausalLMLoss(values, targets,
                vocab_size=weight.shape[0], shift_labels=targets)
            dlogits, = torch.autograd.grad(local_loss, values, upstream)
        del values, local_loss, logits
        rows = ctx.tokens if ctx.backward_rows is None else ctx.backward_rows
        offset = ctx.prompt-1 if ctx.backward_rows is None else 0
        count = dlogits.shape[1]
        full = torch.zeros((rows, weight.shape[0]),
                           dtype=dlogits.dtype, device=dlogits.device)
        full[offset:offset+count].copy_(dlogits[0])
        del dlogits
        # Default: native row count/dtype/layout. Explicit row counts are
        # separate candidates and require their own empirical gradient gate.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            padded_gradient = torch.mm(full, weight)
        if ctx.backward_rows is None:
            dhidden = padded_gradient.unsqueeze(0)
        else:
            dhidden = torch.zeros((1, ctx.tokens, weight.shape[1]),
                                  dtype=padded_gradient.dtype, device=weight.device)
            dhidden[:, ctx.prompt-1:-1].copy_(padded_gradient[:count])
        return dhidden.to(dtype=ctx.hidden_dtype), None, None, None, None


def selected_native_backward_loss(hidden, weight, labels, prompt, backward_rows=None):
    return _SelectedNativeBackward.apply(hidden, weight, labels, prompt, backward_rows)
