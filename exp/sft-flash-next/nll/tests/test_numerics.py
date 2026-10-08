from types import SimpleNamespace

import torch

import numerics


def test_fixed_projection_preserves_chunked_rows_and_bias():
    torch.manual_seed(1)
    x = torch.randn(1, 601, 13)
    weight, bias = torch.randn(7, 13), torch.randn(7)
    whole = numerics.linear(x, weight, bias)
    split = torch.cat([numerics.linear(x[:, :512], weight, bias),
                       numerics.linear(x[:, 512:], weight, bias)], dim=1)
    torch.testing.assert_close(whole, split, rtol=0, atol=0)
    torch.testing.assert_close(whole, torch.nn.functional.linear(x, weight, bias))


def test_expert_sum_restores_router_order_before_weighting():
    y = torch.tensor([[2., 3.], [5., 7.], [11., 13.], [17., 19.]])
    order = torch.tensor([2, 0, 3, 1])
    weights = torch.tensor([[.25, .75], [.6, .4]])
    module = SimpleNamespace(hidden_dim=2, _expert_outputs_loop=lambda *args: y)
    result = numerics.expert_sum(module, None, None, None, order, None, None,
                                 weights, 2, torch.float32)
    unsorted = torch.empty_like(y); unsorted[order] = y
    expected = torch.stack([unsorted[0]*.25 + unsorted[1]*.75,
                            unsorted[2]*.6 + unsorted[3]*.4])
    torch.testing.assert_close(result, expected)


def test_expert_fixed_rows_cover_empty_experts_and_tail():
    torch.manual_seed(2)
    x = torch.randn(267, 3)
    w1, w2 = torch.randn(3, 3, 8), torch.randn(3, 4, 3)
    module = SimpleNamespace(qweight_gate_up=w1, qweight_down=w2,
                             scales_gate_up=torch.ones(3), scales_down=torch.ones(3),
                             _nll_dequantize=lambda w, scales, dtype: w,
                             act_fn=torch.nn.functional.silu)
    counts = torch.tensor([260, 0, 7])
    got = numerics.expert_outputs(module, x, counts, torch.float32)
    expected = []
    for e, rows in [(0, x[:260]), (2, x[260:])]:
        gate, up = (rows @ w1[e]).chunk(2, -1)
        expected.append((torch.nn.functional.silu(gate)*up) @ w2[e])
    torch.testing.assert_close(got, torch.cat(expected))
