import types
import pytest
import torch
import tiny
from native_checkpoint_blocks import rms_rows,gated_rows,hyper_rows


@pytest.mark.parametrize('kind',['rms','gated','hyper'])
def test_native_blocks_outputs_and_input_gradients(kind):
    torch.manual_seed(141);torch.set_num_threads(2)
    mq=tiny.mq
    if kind=='rms':
        m=mq.Qwen4ExpTextRMSNorm(64,group_size=16);fn=rms_rows;width=64
    elif kind=='gated':
        m=mq.Qwen4ExpTextRMSNormGated(64,activation='sigmoid');fn=gated_rows;width=64
    else:
        cfg=tiny.tiny_config().text_config
        m=mq.Qwen4ExpTextGatedResidual(cfg);fn=hyper_rows;width=cfg.hidden_size*cfg.hc_count
    m.requires_grad_(False);m.native_forward=m.forward;m.native_block=7
    args=[torch.randn(1,19,width)]
    if kind=='gated':args.append(torch.randn_like(args[0]))
    def run(blocked):
        values=[v.clone().requires_grad_(True) for v in args]
        outputs=fn(m,*values) if blocked else m(*values)
        if not isinstance(outputs,tuple):outputs=(outputs,)
        loss=sum(v.square().sum() for v in outputs)
        return outputs,torch.autograd.grad(loss,values)
    native,actual=run(False),run(True)
    for aa,bb in zip(native,actual):
        for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
