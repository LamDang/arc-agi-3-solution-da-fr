"""Check causal target alignment and patch restoration against native HF."""
import pytest
import torch
from torch import nn
import tiny
from native_optimization_flags import objective, apply_flags


class Wrapper(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def get_base_model(self):return self.base
    def forward(self,**kw):return self.base(**kw)


def setup():
    torch.set_num_threads(2);torch.manual_seed(52)
    config=tiny.tiny_config();config.text_config.vocab_size=1024
    base=tiny.mq.Qwen4ExpForConditionalGeneration(config).train()
    base.requires_grad_(False)
    # Exercise loss gradients into the decoder, across the prompt/target boundary.
    parameter=base.model.language_model.layers[0].linear_attn.in_proj_qkv.weight
    parameter.requires_grad_(True)
    ids=torch.randint(20,1000,(1,25))
    batch=dict(input_ids=ids,mm_token_type_ids=torch.zeros_like(ids))
    labels=ids.clone();labels[:,:17]=-100
    return base,Wrapper(base),parameter,batch,labels


def test_selected_and_chunked_native_objectives_keep_causal_alignment():
    base,model,param,batch,labels=setup()
    expected=None
    for flags in ({},{'loss':'selected'},{'loss':'chunked','loss_block':3}):
        model.zero_grad(set_to_none=True)
        loss=objective(model,batch,labels,17,flags);loss.backward()
        assert param.grad is not None and torch.count_nonzero(param.grad)
        got=(loss.detach(),param.grad.clone())
        if expected is None:expected=got
        else:
            for a,b in zip(got,expected):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-8)


def test_flags_restore_native_methods_after_exception():
    base,model,param,batch,labels=setup()
    before={id(m):(m,'forward' in m.__dict__,m.forward) for m in base.modules()}
    with pytest.raises(RuntimeError,match='intentional'):
        with apply_flags(base,{'norm_block':3,'hyper_block':3,'ple_block':3}):
            raise RuntimeError('intentional')
    for module,owned,method in before.values():
        assert ('forward' in module.__dict__)==owned
        assert module.forward==method
