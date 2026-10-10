"""Token-weighted accumulation primitives, independent of model optimizations."""
import torch
import json
from pathlib import Path
from dataset import file_hash


def validate_clean_initial_adapter(path, provenance_path):
    """Reject overfit/nonzero-B states as production initializations."""
    provenance = json.loads(Path(provenance_path).read_text())
    if provenance.get('purpose') != 'clean-production-initialization' or provenance.get('validation_exposed') is not False:
        raise ValueError('Production initialization provenance is not clean/train-only')
    if provenance.get('optimizer_updates') != 0 or provenance.get('adapter_sha256') != file_hash(path):
        raise ValueError('Production initial adapter/provenance mismatch')
    state = torch.load(path,map_location='cpu',weights_only=True)
    a = {n:v for n,v in state.items() if 'lora_A' in n}
    b = {n:v for n,v in state.items() if 'lora_B' in n}
    if len(state)!=744 or len(a)!=372 or len(b)!=372 or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Incomplete/nonfinite production initial state')
    if any(torch.count_nonzero(v) for v in b.values()) or not all(torch.count_nonzero(v) for v in a.values()):
        raise ValueError('Production must start from random nonzero A and zero B, never diagnostic-trained adapters')
    return provenance


def labels_for_annotation(input_ids, annotation):
    positions = torch.as_tensor(annotation['positions'], device=input_ids.device, dtype=torch.long)
    targets = torch.as_tensor(annotation['target_ids'], device=input_ids.device, dtype=input_ids.dtype)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('Expected batch-one input IDs')
    if positions.numel() != annotation['target_tokens'] or positions.numel() != targets.numel():
        raise ValueError('Supervised token count differs from annotation')
    if positions.numel() == 0 or positions.min() < 1 or positions.max() >= input_ids.shape[1]:
        raise ValueError('Invalid next-token target position')
    if not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError('Target positions must be unique and ordered')
    if not torch.equal(input_ids[0, positions], targets):
        raise ValueError('Annotated targets differ from encoded tokens')
    labels = torch.full_like(input_ids, -100)
    labels[0, positions] = targets
    return labels


def finish_token_update(model, optimizer, supervised_tokens, *, max_norm=1.0, lr=None):
    """Normalize raw sums exactly once, then clip, update, and clear gradients."""
    if not isinstance(supervised_tokens, int) or isinstance(supervised_tokens, bool) or supervised_tokens <= 0:
        raise ValueError('An update needs a positive actual supervised token count')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters')
    for p in parameters:
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError('Missing or nonfinite accumulated gradient')
    for p in parameters:
        p.grad.div_(supervised_tokens)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    if lr is not None:
        for group in optimizer.param_groups:
            group['lr'] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)
