import copy
from types import SimpleNamespace

import pytest
import torch

import backend
from gdn_blocks import blocked_forward
from kernel_checks import REFERENCE_GDN, REFERENCE_CONV


@pytest.mark.parametrize('block', [1,3,8,32])
def test_whole_gdn_blocks_preserve_halo_state_and_adapter_gradients(block):
    torch.manual_seed(17)
    config = backend.rm.mq.Qwen4ExpTextConfig(hidden_size=32,linear_num_key_heads=2,
        linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,num_hidden_layers=1,layer_types=['linear_attention'],
        output_gate_type='sigmoid')
    reference = backend.rm.mq.Qwen4ExpTextGatedDeltaNet(config,0).float()
    reference.requires_grad_(False)
    for name in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'):
        setattr(reference,name,backend.LoRALinear(getattr(reference,name),rank=2,alpha=4))
    with torch.no_grad():
        for name,p in reference.named_parameters():
            if p.requires_grad:p.normal_(0,.03)
    checked = copy.deepcopy(reference)
    kernels = SimpleNamespace(torch_chunk_gated_delta_rule=REFERENCE_GDN,causal_conv1d_fn=REFERENCE_CONV)
    x = torch.randn(1,35,32,requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = reference(x)
    actual = blocked_forward(checked,y,block_tokens=block,kernels=kernels)
    # Late-only loss exercises cross-block state and convolution derivatives.
    weights = torch.randn_like(expected[:,-3:])
    (expected[:,-3:]*weights).sum().backward()
    (actual[:,-3:]*weights).sum().backward()
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(y.grad,x.grad,rtol=4e-4,atol=5e-6)
    for (name,p),(_,q) in zip(reference.named_parameters(),checked.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(q.grad,p.grad,rtol=4e-4,atol=5e-6,msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GDN/convolution runtime required')
def test_whole_gdn_cuda_native_layout_and_adapter_gradients(monkeypatch):
    import gdn_blocks
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction', False)
    report = gdn_blocks.check()
    assert report['input_gradient_relative_error'] < .01
    assert max(report['adapter_gradient_relative_errors'].values()) < .01
