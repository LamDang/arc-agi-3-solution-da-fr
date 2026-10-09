"""Preserve native QSA selection/SDPA while removing redundant dense masks.

Only for an unpadded, cache-free, batch-one request. The indexer still executes
HF's original query loop and floating-point arithmetic. Its final storage step
writes one additive bias directly, instead of boolean scatter plus mask merges.
"""
import hashlib
import inspect
import textwrap
import torch


class LazyCausalMask:
    def __init__(self,tokens,device):
        self.shape=(1,1,tokens,tokens)
        self.device=device
        self.dtype=torch.bool
        self.keys=torch.arange(tokens,device=device)

    def __getitem__(self,index):
        batch,head,query=index
        if batch!=0 or head!=0 or not isinstance(query,int):
            raise ValueError('Unexpected native causal-mask indexing')
        return self.keys<=query

    def is_floating_point(self):return False

    def __and__(self,selected_bias):
        # Native selection used only visible causal keys. Its additive bias
        # already contains exactly the final allowed set; no mask merge remains.
        if not isinstance(selected_bias,torch.Tensor) or not selected_bias.is_floating_point():
            raise ValueError('Expected the patched indexer additive bias')
        if tuple(selected_bias.shape)!=self.shape:
            raise ValueError('Unexpected native indexer bias shape')
        return selected_bias


def indexer_with_direct_bias(native_forward):
    """Version-guarded replacement of the native indexer's final mask storage."""
    source=textwrap.dedent(inspect.getsource(native_forward))
    marker='    # Create the additive mask to be added to the main causal mask\n'
    if source.count(marker)!=1:
        raise RuntimeError('Unsupported HF indexer source; storage patch was not applied')
    prefix,tail=source.split(marker)
    if 'selected_token_mask.scatter(' not in tail or not tail.rstrip().endswith('return selected_token_mask'):
        raise RuntimeError('Unexpected native indexer mask construction')
    storage='''    kv_length = attention_mask.shape[-1]
    # One padded float bias: -1 indices scatter into an invisible dummy column.
    width = ((kv_length + 1 + 7) // 8) * 8
    bias = torch.full((*selected_token_indices.shape[:-1], width), float('-inf'),
                      device=hidden_states.device, dtype=hidden_states.dtype)
    indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
    bias.scatter_(-1, indices, 0.)
    return bias[..., :kv_length].unsqueeze(1)
'''
    namespace=dict(native_forward.__func__.__globals__ if hasattr(native_forward,'__func__') else native_forward.__globals__)
    exec(compile(prefix+storage,'<native-qsa-mask-storage>','exec'),namespace)
    function=namespace[native_forward.__name__]
    function.native_source_sha256=hashlib.sha256(source.encode()).hexdigest()
    return function


def language_with_lazy_mask(self,*args,**kwargs):
    if args:
        raise ValueError('Lazy mask expects native multimodal keyword arguments')
    if kwargs.get('past_key_values') is not None or kwargs.get('use_cache'):
        raise ValueError('Lazy mask requires cache-free training')
    existing=kwargs.get('attention_mask')
    if existing is not None:
        if not isinstance(existing,torch.Tensor) or existing.ndim!=2 or not bool(existing.bool().all()):
            raise ValueError('Lazy mask only supports unpadded requests')
    inputs=kwargs.get('inputs_embeds')
    if inputs is None:inputs=kwargs.get('input_ids')
    if inputs is None or inputs.shape[0]!=1:
        raise ValueError('Lazy mask requires batch one')
    kwargs['attention_mask']={'indexed_attention':LazyCausalMask(inputs.shape[1],inputs.device),
                              'linear_attention':None}
    return self.native_mask_forward(**kwargs)
