# QLoRA dtype review — 2026-10-09

The user requested a separate framework review while the primary agent traces
the actual native model. This review does not change adapter precision or
qualify a new training policy.

## Standard framework behavior

| Implementation | Adapter storage | Other precision choices |
| --- | --- | --- |
| Original `artidoro/qlora` | With `--bf16`, explicitly casts `LoraLayer` modules to BF16 | Keeps norms FP32; casts floating embedding/head weights to BF16 |
| Modern PEFT default | Promotes FP16/BF16 adapters to FP32 for stability | `autocast_adapter_dtype=False` can disable promotion; backend details matter |
| Current TRL `SFTTrainer` | Explicitly casts trainable quantized-model adapters to BF16, excluding DoRA magnitude vectors | Includes backend-specific handling because disabling PEFT promotion is not sufficient for every quantized model |

Primary sources:

- [Original QLoRA implementation](https://github.com/artidoro/qlora/blob/main/qlora.py#L371-L380).
- [PEFT adapter dtype documentation](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter).
- [TRL SFTTrainer implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L826-L835).
- [PEFT standard LoRA forward](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L1037-L1074).
- [PEFT bitsandbytes forward](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py#L463-L510).
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/main/amp.html).

These upstream links refer to `main` as reviewed on this date, not a claim
that our installed PEFT 0.20.0/Transformers 5.18.0 have exactly those trainer
implementations. The diagnostic freezes its actual imported native sources.

## Meaning for this reference

`overfit_hf_reference.py` loads floating model weights with BF16, calls default
`get_peft_model`, and executes under BF16 CUDA autocast. It does not call
`prepare_model_for_kbit_training`. Its 744 adapter tensors are FP32 because of
PEFT's adapter policy, not a Qwen4Exp requirement.

Parameter storage and operation dtype are separate. FP32 adapter parameters
can participate in BF16 linear operations under autocast. Standard LoRA
computes the frozen base branch and the low-rank branch separately; it does
not materialize a full FP32 merged weight during ordinary training. The
standard PEFT linear wrapper returns the base output dtype after addition.

FP32 adapter storage preserves small updates and FP32 parameter gradients.
BF16 adapter storage is nevertheless a standard QLoRA choice. Casting an
adapter to BF16 does not itself create FP32 master parameters, and optimizer
state policy must be checked separately. GPU preference for BF16 computation
does not establish the performance effect of changing storage alone.

Generic [PEFT k-bit preparation](https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L149-L241)
can upcast ordinary floating parameters beyond norms, with backend exceptions.
Do not apply it indiscriminately to this model's 95 GiB frozen PLE table.

## Closest public model comparison

The review did not find a public official QLoRA recipe for this exact
Qwen4Exp/Qwen3.8 export. Qwen3-Next is a useful hybrid Gated DeltaNet + MoE
comparison, not an exact substitute. Its normalization/decay implementation
contains explicit FP32 arithmetic but restores several outputs to the input
dtype. FP32 internal arithmetic alone does not require FP32 residual storage.

See the [native Qwen3-Next implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L114-L127).

The local short-forward diagnostic is recorded separately in
`LAYER_DTYPE_TRACE.md`. Its purpose is to identify the first persistent
promotion in our actual native model/AutoRound/PEFT path, rather than infer
that cause from generic framework defaults.
