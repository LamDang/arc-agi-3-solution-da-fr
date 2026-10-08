# Exact causal-conv1d build used by both Sol NLL panels

The native `causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl` is preserved in
the existing private DVC remote. This is the actual wheel compiled on the
Kaggle image used for the 2026-10-08 original and generated-thinking panels,
not a rebuild. Its bytes match both panels' qualified wheelhouses and the
committed GPU qualification record.

- Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0.
- Linux x86_64, CXX11 ABI=true, RTX PRO 6000 Blackwell (compute capability 12.0).
- Wheel size: 173,329,986 bytes.
- SHA256: `f928f1aa1de1306f26f58a3f2c7a6d9ac2d696090fd1bd26430951d82be9928b`.

From the repository root:

```bash
dvc pull data/sol-nll-causal-conv1d-20261008/causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl.dvc
sha256sum data/sol-nll-causal-conv1d-20261008/causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl
```

Copy the retrieved wheel into the offline wheelhouse before running
`build_setup.py`. The [NLL packaging instructions](../../exp/sft-flash-next/nll/README.md#qualified-cuda-kernels-and-chunk-stable-scoring-2026-10-08)
describe the other two pinned kernel packages and the exact installation/build
procedure. The result archives preserve scoring code and dependency locks;
this separate artifact preserves the native convolution binary they exclude.

`provenance.json` records the build environment, native libraries and successful
ZIP CRC check. The [existing GPU qualification record](../../exp/sft-flash-next/verification/gpu-qualification.json)
records its checksum and numerical qualification. This archival step runs no
GPU code. If the Python/torch/CUDA/ABI image changes, rebuild and requalify.
