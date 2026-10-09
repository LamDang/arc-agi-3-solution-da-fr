# Single-sample HF learning qualification

Status: **passed the requested single-sample learning test** after
**5 optimizer updates**. Loss fell from **0.6247151494** to
**0.0102175446**: **1.636% of the initial loss**, or a
**98.364% reduction**. The required threshold was
**0.0312357575**.

| Completed updates | Measured loss | Fraction of initial loss |
|---|---:|---:|
| 0 | 0.62471515 | 100.000% |
| 1 | 0.50304472 | 80.524% |
| 2 | 0.33412135 | 53.484% |
| 3 | 0.17196685 | 27.527% |
| 4 | 0.05930277 | 9.493% |
| 5 | 0.01021754 | 1.636% |

Measurements use the same fixed input and target mask. The loss in row N is
measured after N updates; row 0 has zero LoRA contribution. All adapter
gradients were present and finite before each update. Peak GPU memory over
the completed training steps was **85.33 GiB allocated / 85.89 GiB
reserved** on RTX PRO 6000 Blackwell 96 GB. The run used the 175 GiB host-RAM
environment; no A100 execution is claimed. Loading took 252.24 seconds; the
first forward took 119.48 seconds. Steady full steps took approximately
261.5–265.5 seconds, including diagnostic
adapter writes. Total elapsed time through the passing measurement was
28.30 minutes, including loading.

The final measured loss, result JSON, best adapter and final adapter were saved
successfully. A terminal logging call then raised `TypeError` because it
supplied `elapsed_seconds` twice. Thus the learning criterion passed, but the
original process exited with a reporting error. The original script and
failure trace are retained. The repository script fixes that single logging
line; a regression check exercised the same colliding-field call successfully.
The full training experiment was not rerun for this reporting-only fix.
Downloaded files match server-side SHA256 hashes. All 744 adapter tensors
are finite; all 372 A and 372 B tensors changed from initialization, and
best/final checkpoint tensors are identical. The process exited and a final
GPU check reported 0 MiB / 0% utilization; the Kaggle session itself remains
attached.

This qualifies the reference for the requested learning criterion. It does
**not** establish exact gradient equivalence for the optimized 120K trainer.

## Fixed diagnostic input

The source is panel request `sp80-589a99af-r11`: 31,839 tokens, 13 images,
and 651 supervised target tokens. For the native full-vocabulary loss, the
16K test uses a suffix with 16,249 tokens, 7 complete images, and all 651
original target tokens. It removes 15,590 context tokens and the corresponding
6 images; no image is split and no target is removed. The native HF model
recomputes positions for this window. This is a cropped diagnostic input,
not a claim about full-context training.

Serialized window SHA256:
`48f6f88b7bff3b2be5b823c7394388d6b530e26febd25e4c21d3c0b13fdd49ff`.
The original manifest SHA256 is
`64789772729c72d8cdd988901d7104acd8ca2b4177aaace7cabc82eeac59274f`.
This request belongs to fold 0. Its diagnostic adapters must never be used as
a production checkpoint or included in an unbiased fold-0 validation result.

## Model and operators

`prune_checkpoint.py` exports the existing keep-256 selection to an ordinary
checkpoint directory. It copies the retained expert tensor bytes, slices the
corresponding router rows, and symlinks unchanged shards. It does not
requantize weights. The keep-map SHA256 is
`c76c2734c30fabcde2a44fa44467f93ae50ade945068a11e4befb5246bd5fca6`.

`overfit_hf_reference.py` loads that checkpoint through stock Transformers
`AutoModelForImageTextToText.from_pretrained` with the official AutoRound
`auto_round:tritonv2_zp` backend. The base is frozen. PEFT attaches rank-16,
alpha-32 LoRA to the same ordinary language projections as the intended
trainer. Initial B matrices are zero, so the initial adapter contribution is
zero. Routed experts and the PLE table stay frozen. The generic PEFT k-bit
preparation helper is deliberately avoided because it would upcast this
checkpoint's large BF16 base tensors.

The training recipe enables stock HF non-reentrant layer checkpointing and
PyTorch `torch.autograd.graph.save_on_cpu(pin_memory=False)`. These are declared
memory controls. No repository attention, router, GDN, quantized linear,
activation-offload implementation, or loss replacement is imported. The
model computes logits for every input position using its native causal loss;
prompt labels are `-100`. The full vocabulary has 248,320 entries.

Two placement attempts failed before training: `device_map="auto"` put the
decoder on CPU, and an explicit mixed CPU/GPU map caused Accelerate's
inference-hook setup to copy the entire 95.37 GiB table onto CUDA. The second
attempt loaded the checkpoint but failed during hook installation.

The successful third loading attempt maps every checkpoint branch except the large PLE table to
GPU. With no root map entry, HF's native placement default leaves the omitted
table on CPU; because all explicitly mapped devices are the same GPU, HF
skips mixed-device inference dispatch. Nonpersistent rotary/hash buffers are
moved to GPU explicitly. Every parameter's resident device is checked.
Native PLE forward already moves lookup IDs to the table device and returns
results to the decoder. No model implementation is patched for this placement. Loading and adapter
setup took 252.24 seconds; GPU allocations after setup were 39.35 GiB, with
33,478,656 trainable adapter parameters.

Kaggle's optional torchao 0.10.0 conflicts with PEFT 0.20.0. The subprocess
uses an isolated package view without torchao and the extracted official
AutoRound 0.15.0 wheel; no installed package or model implementation is patched.
See [QUANTIZATION_COMPATIBILITY.md](QUANTIZATION_COMPATIBILITY.md).

## Run and acceptance

```bash
python overfit_hf_reference.py \
  --model /tmp/reference-256-hf \
  --sample /path/to/overfit-sample-16k/sample.pt \
  --prompt-tokens 15598 \
  --out /path/to/new-result-directory \
  --checkpointing --save-on-cpu
```

The runner uses AdamW with learning rate 0.0002, no weight decay, and gradient
norm clipping at 1.0, for up to 200 updates. The loss logged at step N is
measured after exactly N updates; step 0 is the untrained baseline. The input,
loss mask and model operators remain fixed. Every adapter gradient must be
present and finite before an update. It saves the initial, best and final
adapter states, the per-step loss and timing, memory peaks, load diagnostics,
package versions, hashes and failures. A pass requires measured loss / initial
loss <= 0.05. Hitting the update limit is not a pass.

The result JSON and complete per-update history are preserved in
`overfit-reference.zip.dvc`, together with initial/best/final diagnostic
adapters and the exact tensor input. `overfit-reference.provenance.json`
records checksums and the private DVC location. The user authorized the
archive upload, and an S3 download round trip matched both MD5 and SHA256. Gradient comparisons for each custom
optimization remain a separate outstanding test.

## Preparing the checkpoint and input

Run from this directory with the existing GPU environment (torch 2.11.0+cu128,
Transformers 5.18.0, PEFT 0.20.0, AutoRound 0.15.0). Preserve the CUDA-compatible
torch installation. PEFT must either have no optional torchao installed or a
compatible version; the measured run isolated the existing incompatible
package instead of changing the shared environment.

```bash
python ../../reap-flash-next/prune_checkpoint.py \
  --model-dir /path/to/original-512-expert-checkpoint \
  --keep-file artifacts/keep-256-fold0.json --keep 256 \
  --out /tmp/reference-256-hf
python prepare_audit_sample.py \
  --bundle /path/to/prepared-data --sample-id sp80-589a99af-r11 \
  --out /path/to/overfit-sample-32k
python prepare_overfit_window.py \
  --source /path/to/overfit-sample-32k --out /path/to/overfit-sample-16k \
  --config /tmp/reference-256-hf/config.json --max-tokens 16384
```

The exported checkpoint needs roughly 40 GiB of writable scratch space, plus
access to the unchanged source shards through symlinks. The native loader
materializes the 95.37 GiB embedding table in RAM. The exported checkpoint,
CPU table, saved activations, and trainable adapter states are distinct
allocations; the successful load alone is not a training-memory estimate.
