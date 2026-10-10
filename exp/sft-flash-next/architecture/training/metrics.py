"""Teacher-forced metrics with bounded vocabulary logits."""
import torch
from torch.nn import functional as F

from components.head import target_positions


@torch.no_grad()
def correct_tokens(hidden, weight, labels, token_chunk=128, vocab_chunk=8192):
    positions, targets = target_positions(labels)
    correct = 0
    for start in range(0, len(positions), token_chunk):
        selected = hidden[0].index_select(0, positions[start:start+token_chunk]).to(weight.dtype)
        best = torch.full((selected.shape[0],), -float('inf'), device=hidden.device,
                          dtype=torch.float64 if selected.dtype == torch.float64 else torch.float32)
        indices = torch.zeros(selected.shape[0], dtype=torch.long, device=hidden.device)
        for offset in range(0, weight.shape[0], vocab_chunk):
            logits = F.linear(selected, weight[offset:offset+vocab_chunk])
            value, index = logits.max(-1)
            replace = value > best  # Strict comparison preserves first-index argmax ties.
            indices = torch.where(replace, index+offset, indices)
            best = torch.maximum(best, value.to(best.dtype))
            del logits
        correct += int((indices == targets[0, start:start+token_chunk]).sum())
    return correct


class Metrics:
    def __init__(self):
        self.nll_sum = 0.;self.correct = 0;self.targets = 0

    def add(self, nll_sum, correct, targets):
        self.nll_sum += float(nll_sum);self.correct += int(correct);self.targets += int(targets)

    def result(self):
        if not self.targets:
            raise ValueError('Empty validation target set')
        return dict(nll=self.nll_sum/self.targets, accuracy=self.correct/self.targets,
                    correct=self.correct, targets=self.targets)
