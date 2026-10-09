"""Reversible optimization flags applied to the HF-loaded model with PEFT intact.

Importing this module changes no operator. No custom checkpoint loader, adapter
implementation, quantization kernel or precomputed embedding path is enabled.
Each requested replacement is restored on exit, including after exceptions.
"""
from contextlib import contextmanager
import types
import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def apply_flags(base, flags):
    import backend as b
    undo = []
    def set_attr(obj, name, value):
        existed = name in obj.__dict__
        old = obj.__dict__.get(name)
        undo.append((obj, name, existed, old))
        setattr(obj, name, value)
    def bind(obj, fn):
        set_attr(obj, 'forward', types.MethodType(fn, obj))
    try:
        for m in base.model.language_model.modules():
            cls = type(m).__name__
            native_choice = {
                'Qwen4ExpTextRMSNorm': ('native_norm_block', 'rms_rows'),
                'Qwen4ExpTextRMSNormGated': ('native_gated_block', 'gated_rows'),
                'Qwen4ExpTextGatedResidual': ('native_hyper_block', 'hyper_rows'),
            }.get(cls)
            if native_choice and flags.get(native_choice[0]):
                import native_checkpoint_blocks as native_blocks
                set_attr(m, 'native_forward', m.forward)
                set_attr(m, 'native_block', flags[native_choice[0]])
                bind(m, getattr(native_blocks, native_choice[1]))
            if cls == 'Qwen4ExpTextRMSNorm' and flags.get('norm_block'):
                set_attr(m, 'train_block', flags['norm_block']); bind(m, b.training_rms_norm)
            if cls == 'Qwen4ExpTextGatedResidual' and flags.get('hyper_block'):
                set_attr(m, 'unblocked_forward', m.forward)
                set_attr(m, 'train_block', flags['hyper_block']); bind(m, b.training_hyper_mix)
            if cls == 'Qwen4ExpTextRMSNormGated' and flags.get('gated_norm_block'):
                set_attr(m, 'train_gated_block', flags['gated_norm_block']); bind(m, b.training_gated_rms_norm)
            if cls == 'Qwen4ExpTextPLELayer' and flags.get('ple_block'):
                set_attr(m, 'train_ple_block', flags['ple_block']); bind(m, b.training_ple)
            if cls == 'Qwen4ExpTextGatedDeltaNet' and flags.get('gdn_block_tokens'):
                set_attr(m, 'unblocked_gdn_forward', m.forward)
                set_attr(m, 'train_gdn_block_tokens', flags['gdn_block_tokens']); bind(m, b.training_blocked_gdn)
            if cls == 'Qwen4ExpTextAttention' and flags.get('attention'):
                set_attr(m, 'index_block', flags.get('index_block',256))
                set_attr(m, 'query_block', flags.get('query_block',16))
                set_attr(m, 'key_block',32)
                set_attr(m, 'train_projection_block',flags.get('attention_projection_block',0))
                set_attr(m, 'train_attention_backend',flags['attention']); bind(m,b.training_attention)
        yield
    finally:
        for obj,name,existed,old in reversed(undo):
            if existed:setattr(obj,name,old)
            else:delattr(obj,name)


def objective(model, batch, labels, prompt, flags):
    """Native loss, native selected logits, or checkpointed native CE blocks."""
    if flags.get('loss') == 'selected':
        positions = torch.arange(prompt-1, labels.shape[1]-1, device=labels.device)
        return model(**batch, labels=labels, use_cache=False, logits_to_keep=positions,
                     shift_labels=labels[:,prompt:]).loss
    if flags.get('loss') == 'chunked':
        # Call the same native multimodal decoder; every prompt token remains in
        # its autograd graph. Only ignored head logits are omitted.
        base = model.get_base_model()
        hidden = base.model(**batch, use_cache=False).last_hidden_state[:,prompt-1:-1]
        targets = labels[:,prompt:]
        total = targets.numel()
        block = flags.get('loss_block',128)
        def local(h,t):
            logits = base.lm_head(h).float()
            return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                t.reshape(-1),reduction='sum') / total
        return sum(checkpoint(local,h,t,use_reentrant=False)
                   for h,t in zip(hidden.split(block,1),targets.split(block,1)))
    return model(**batch, labels=labels, use_cache=False).loss
