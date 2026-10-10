import pytest
import torch

from offload import ActivationOffload


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('prefetch', [0, 2])
def test_disk_offload_preserves_gradients_and_cleans_files(tmp_path, dtype, prefetch):
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(7, 9), torch.nn.SiLU(), torch.nn.Linear(9, 3)).to(dtype)
    x = torch.randn(13, 7, dtype=dtype, requires_grad=True)
    model(x).float().square().mean().backward()
    expected = [x.grad.clone(), *[p.grad.clone() for p in model.parameters()]]
    model.zero_grad(set_to_none=True);x.grad = None
    with ActivationOffload(model, disk_dir=tmp_path, disk_budget_gib=.001,
                           min_bytes=1, disk_min_bytes=1, prefetch=prefetch,
                           offload_cpu=True, pin_reads=False) as offload:
        # Offload preserves values exactly even for a noncontiguous view.
        probe = x.detach().t()
        torch.testing.assert_close(offload.unpack(offload.pack(probe)), probe, rtol=0, atol=0)
        model(x).float().square().mean().backward()
    for actual, ref in zip([x.grad, *[p.grad for p in model.parameters()]], expected):
        # Contiguous restores can select a different CPU BLAS reduction order.
        # The saved values above remain bitwise exact; FP32 gradients may round.
        torch.testing.assert_close(actual, ref, rtol=2e-6 if dtype==torch.float32 else 0,
                                   atol=1e-9 if dtype==torch.float32 else 0)
    assert offload.stats['disk_bytes'] > 0
    assert not list(tmp_path.iterdir())


def test_disk_budget_falls_back_to_ram_and_exception_cleans(tmp_path):
    model = torch.nn.Linear(7, 9)
    with pytest.raises(RuntimeError, match='abort'):
        with ActivationOffload(model, disk_dir=tmp_path, disk_budget_gib=100/2**30,
                               min_bytes=1, disk_min_bytes=1, offload_cpu=True,
                               pin_reads=False) as offload:
            x = torch.randn(13, 7, requires_grad=True)
            model(x).sum()
            assert offload.stats['cpu_bytes'] > 0
            assert offload.stats['disk_bytes'] <= 100
            raise RuntimeError('abort')
    assert not list(tmp_path.iterdir())


def test_dedup_respects_in_place_versions(tmp_path):
    offload = ActivationOffload(torch.nn.Identity(), min_bytes=1, offload_cpu=True)
    x = torch.arange(10.)
    with offload:
        a = offload.pack(x)
        assert offload.pack(x) is a
        x.add_(1)
        b = offload.pack(x)
        assert a is not b
        torch.testing.assert_close(offload.unpack(a), torch.arange(10.))
        torch.testing.assert_close(offload.unpack(b), torch.arange(10.) + 1)
