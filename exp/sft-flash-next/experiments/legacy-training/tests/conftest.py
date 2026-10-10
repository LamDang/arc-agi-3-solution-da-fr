import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "exp/reap-flash-next/tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest


@pytest.fixture(autouse=True)
def explicit_cpu_reference_kernels(monkeypatch):
    # These hybrid/reference tests use CPU tensors even on a CUDA host.
    # Transformers' installed-kernel dispatcher can otherwise choose the
    # CUDA-only causal-conv1d implementation merely because it is installed.
    # Real CUDA convolution/GDN parity is checked separately before GPU runs.
    import backend
    from kernel_checks import REFERENCE_CONV, REFERENCE_GDN
    monkeypatch.setattr(backend.rm.mq, 'causal_conv1d_fn', REFERENCE_CONV)
    monkeypatch.setattr(backend.rm.mq, 'torch_chunk_gated_delta_rule', REFERENCE_GDN)
