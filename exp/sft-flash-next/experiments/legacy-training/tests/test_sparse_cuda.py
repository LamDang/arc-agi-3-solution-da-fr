"""CUDA regressions for masked-prefix indexed attention; CPU CI skips these."""
import pytest
import torch

import backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('head_dim', [32, 256])
def test_empty_leading_tiles_and_very_negative_logits(head_dim):
    from sparse_kernels import IndexedAttention
    torch.manual_seed(17)
    n, h, kv = 5, 24, 2
    q = torch.full((h,n,head_dim), -10., device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.full((kv,n,head_dim), 10., device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(kv,n,head_dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    selected = torch.full((n,67), -1, device='cuda', dtype=torch.long)
    selected[0,-1] = 0
    selected[1,-2:] = torch.tensor([0,1],device='cuda')
    # Row 2 is entirely masked. Row 3 also checks duplicate selected keys.
    selected[3,-2:] = 2
    selected[4,-3:] = torch.tensor([0,2,4],device='cuda')
    scale = head_dim**-.5
    expected = backend.attention_block(q.float(),k.float(),v.float(),selected,scale).to(q.dtype)
    actual = IndexedAttention.apply(q,k,v,selected,scale,32)
    upstream = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(expected,(q,k,v),upstream)
    actual_grad = torch.autograd.grad(actual,(q,k,v),upstream)
    torch.testing.assert_close(actual,expected,rtol=.008,atol=.008)
    assert torch.count_nonzero(actual[:,2]) == 0
    for got, ref in zip(actual_grad,expected_grad):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got,ref,rtol=.015,atol=.008)
