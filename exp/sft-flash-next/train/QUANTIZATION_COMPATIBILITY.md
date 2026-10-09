# PEFT with this AutoRound checkpoint

QLoRA freezes base weights. It still requires an input gradient through every
frozen quantized layer on the path to earlier adapters. HF's AutoRound
`is_trainable=False` metadata is therefore not sufficient evidence that PEFT
cannot work with the packed weights. The earlier blanket inference was wrong.

The checkpoint uses AutoRound 0.15.0, symmetric INT4, group size 128, and
`auto_round:auto_gptq` packing. AutoRound's official `auto_round:tritonv2_zp`
backend accepts that format and implements an input backward by dequantizing
the frozen weight and multiplying the upstream gradient by its transpose.
No gradient for the packed codes, scales or zero points is required.

## Measured operator compatibility

On the existing RTX PRO 6000, torch 2.11.0+cu128 and PEFT 0.20.0, an isolated
probe used the actual checkpoint's
`model.language_model.layers.0.mlp.experts.0.gate_proj` tensors (2560 inputs,
640 outputs). It placed ordinary PEFT rank-16/alpha-32 adapters on regular
linear projections before and after the frozen expert, initialized both LoRA
matrices nonzero, and ran 257 tokens with BF16 autocast. The expert used the
unmodified official AutoRound kernel. The comparison used independently
unpacked copies of the same codes/scales/zero points with a frozen BF16 linear.

| Compared quantity | Relative L2 difference |
|---|---:|
| Output | 0.3484% |
| Input gradient | 0.4277% |
| Before-expert LoRA A gradient | 0.4866% |
| Before-expert LoRA B gradient | 0.3591% |
| After-expert LoRA A gradient | 0.3417% |
| After-expert LoRA B gradient | 0.3830% |

All measured gradients were finite. They were not bitwise equal. These results
establish gradient flow in this operator chain, not exact gradient equivalence,
full-model loading support, MoE routing parity, or 16K memory fit. That operator probe performed no optimizer update or full-model training;
the later full-model result is described below.

Directly targeting the AutoRound `QuantLinear` with ordinary PEFT LoRA failed
with an unsupported-module error. That is separate from our selected recipe:
its adapters target regular language projections while routed experts stay
frozen. A successful adapter attachment alone would not prove upstream
backward support; the probe explicitly exercised that path.

## Environment and remaining work

The first probe failed before adapter attachment because the environment has
optional torchao 0.10.0, while PEFT 0.20.0 requires at least 0.16.0 when that
package is present. The successful probe used a process-local package view
excluding torchao; it changed no installed packages or PEFT source. AutoRound's
wheel was extracted into a temporary isolated path, not installed globally.

The later [full-model HF + PEFT overfit](OVERFIT_REFERENCE.md) passed on the
same RTX PRO 6000. It used the byte-preserving 256-expert export, CPU-resident
PLE table, GPU decoder, standard HF non-reentrant checkpointing and PyTorch
CPU activation storage. With 16,249 input tokens and 651 supervised tokens,
loss fell from 0.624715 to 0.010218 after 5 updates
(98.36% reduction). This establishes functional learning in this manual
HF/PEFT configuration. It does not establish exact gradient parity with the
custom trainer or support in HF Trainer, which may still reject the
quantizer's `is_trainable=False` integration metadata.

The backend is selected through HF's standard AutoRound loading option;
existing packed weights are not requantized. The generic
`prepare_model_for_kbit_training` helper is deliberately avoided: for this
AutoRound model it can bulk-cast BF16/FP16 base parameters to FP32, including
the 95.37 GiB PLE table. The base is frozen directly before PEFT attaches LoRA.

## Evidence

Raw logs, the exact probe scripts, package provenance and result JSON are under
`capacity-results/20261009-gradient-audit/peft-autoround-compat/` (operator evidence; the full overfit evidence is in
`overfit-reference.zip.dvc`). AutoRound wheel SHA256:
`f0c6c3023d38d75b0779939946b48713c5e79b9c26391fcc8e70ee5a81f75bdc`.

The official implementation is
[`QuantLinearFunction.backward`](https://github.com/intel/auto-round/blob/main/auto_round_extension/triton/triton_utils_zp/dequant.py)
and backend format selection is in
[`inference/backend.py`](https://github.com/intel/auto-round/blob/main/auto_round/inference/backend.py).
The inspected code was the pinned 0.15.0 PyPI wheel, not an assumption about the
current contents of the moving `main` links.
