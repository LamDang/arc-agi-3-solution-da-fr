# Fine-tune the 256-expert Flash-Next on generated-thinking requests

This is **PyTorch + Hugging Face Transformers, with a custom LoRA loop** for
the repository's **Qwen3.8-Flash-Next / `qwen4_exp` Intel AutoRound W4A16**
checkpoint, physically pruned to 256 experts. It is not Qwen3-Next-80B-A3B.

A **120,000-token context plus 10,000 target tokens has completed a full
forward/backward/Adam step** on the Kaggle RTX PRO 6000 Blackwell 96 GB, using
59 real images and a disposable composite of the verified 30-request panel.
The selected recipe measured **124.1 s per sample at 120K context**
and **87.9 s at 80K context**, with 10K target tokens in both cases. Peak allocated
GPU memory was **74.28 / 62.65 GiB**. A lower-memory variant reduced
120K to **71.42 GiB allocated / 75.04 GiB reserved** at 131.8 s.
These are disposable forward/backward/Adam steps, excluding startup, encoding
and durable checkpoint writes. Exact configurations, disk-offload results and
numerical checks are in [LONG_CONTEXT.md](LONG_CONTEXT.md). Earlier OOMs remain
in [CAPACITY.md](CAPACITY.md) and the private DVC evidence archive. **No actual
A100 was tested.**

The training dataset PR is a separate dependency: capacity probes discard all
updates and never turn the fold-0 panel into training data.

## Why this framework

The earlier [design](../PLAN.md) proposed Axolotl's exact-model **NF4** recipe
first and this **W4A16 + LoRA** route as a fallback. This implementation chooses
the latter to retain the exact quantized base already evaluated, reuse the
repository's PLE memory mapping and sparse attention, and control final-reply
loss, complete multimodal contexts and partial-accumulation recovery directly.
Axolotl's researched NF4 route requires the unquantized source and a new
quantization/baseline; it cannot simply load this AutoRound checkpoint as NF4.

This trades framework convenience for custom code that needs qualification.
It does not use TRL or PEFT. Adapters are simple standard alpha/r LoRA on
resolved language projections, saved in this runner's checkpoint format.
They are **not directly loadable as a PEFT/SGLang adapter**. `evaluate.py`
loads them through the same backend; a serving export is separate work.

## Training contract

- Batch 1, accumulation 4, one epoch, rank 16, alpha 32, dropout 0, AdamW
  at 1e-4, weight decay 0, clip norm 1. Warm up for 5% of planned optimizer
  updates, then cosine decay to 10% of peak LR. A session stop does not
  compress the schedule. Final partial batches use their actual size.
- Train attention Q/K/V/O, GDN input/output and shared-expert projections.
  Keep routed experts, routers, indexers, vision, PLE and output head frozen.
  Gradients still flow **through** frozen experts and router scores.
  “256 experts” specifies the student architecture; it does not mean all
  expert weights are trainable. `run-identity.json` lists resolved targets.
- Each request contributes its mean final-reply NLL. Final generated
  thinking, code and template delimiters have equal token weight. Historical
  turns, images and the generation prompt are context only.
- Preserve the full request, image resolution, tools and template kwargs.
  Check processor-expanded prompt-prefix equality. No packing, truncation,
  inference cache or detached-prefix approximation. Default admission ceiling
  is 130,000 tokens; preparation fails if exceeded.
- Native full-sequence hybrid computation with non-reentrant layer
  checkpointing. CPU offload retains frozen resident weights by reference.
  Expert intermediates, sparse-attention gathers and output logits have
  bounded recomputation. The inference-only shared dequantization workspace
  is never used by the training backward pass. These reference-oriented
  expert loops use bounded recomputation; measured throughput is in LONG_CONTEXT.md.

## Hardware and installation

| Profile | Device checked by runner | Expert rows | Attention query rows | CE rows |
|---|---|---:|---:|---:|
| `a100-80gb` | Ampere, at least 79 GiB | 512 | 8 | 64 |
| `rtx-pro-6000-96gb` | Blackwell, at least 94 GiB | 8192 | 16 | 128 |

RTX 6000 Ada 48 GB and older RTX 6000 24 GB do not match these profiles.
Expose one GPU with `CUDA_VISIBLE_DEVICES`. The tested Kaggle runtime uses Linux and Python 3.13.15.
The tested Blackwell runtime uses torch
2.11.0+cu128, CUDA 12.8, FLA 0.5.2 and causal-conv1d 1.7.0. Use a torch/torchvision build supporting the
allocated GPU and native wheels built for that exact torch/CUDA ABI.

From the repository root, in the GPU environment:

```bash
# On the qualified Python 3.13 / torch 2.11.0+cu128 image, reuse PR #23's exact wheel:
dvc pull data/sol-nll-causal-conv1d-20261008/causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl.dvc
python -m pip install --no-deps data/sol-nll-causal-conv1d-20261008/causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl
python -m pip install --no-build-isolation -r exp/sft-flash-next/train/requirements-gpu.txt
python -m pytest exp/sft-flash-next/train/tests -q
```

Do not install the CPU torch lock into the GPU environment. Keep the chosen
GPU image and freeze its packages after qualification; resume checks versions.
The runner compares native GDN/convolution outputs **and gradients** against
Torch references before loading production weights.

The 256-expert serving model previously occupied about **40.4 GB of GPU
weights**; that is not its training footprint. The frozen PLE table also
maps approximately **102.4 GB of disk** into host page cache. At 114,688
tokens, 48 checkpoint boundaries at width `2560 × 4`, in BF16, alone are
roughly **105 GiB of activation storage** before other host use. Plan for a
high-memory host and space for source weights and checkpoints. The measured
server has a 175 GiB RAM limit; grouped checkpoints substantially reduce the
activation working set. See LONG_CONTEXT.md for per-case measurements.
Memory mapping does not require the entire PLE table to remain resident,
but paging can dominate runtime. A small-RAM Colab instance may be unsuitable
even with an 80 GB GPU. Qualification records peak host RSS and GPU memory.

## Prepare the model and data on CPU

Use the **training-only expert map from the completed NLL bundle**. The
checked-in `models/flash-next-reap-256/keep.json` used all 25 games and is
rejected for this held-out training experiment. On the measured Kaggle server,
use the attached source plus `--keep` in the launcher below. The following CPU
export is optional and requires ample output disk space; it is unsuitable for
Kaggle's 20 GiB working volume.

```bash
python models/extract.py \
  --keep-file /path/to/nll-bundle/maps/keep-256.json \
  --source /path/to/intel-qwen3.8-flash-next-w4a16-autoround \
  --out /path/to/flash-next-256-trainonly
```

Unchanged shards are symlinks, so retain the source, or add `--copy`. If the
clean map is absent, reconstruct it from the saved per-game calibration
statistics using the existing [NLL preparation](../nll/README.md). No new GPU
calibration is needed. Model preflight checks the map's provenance and hashes
every referenced weight shard, including symlink targets. Hashing is a
streaming CPU/disk operation with four bounded readers and can take several minutes.

Training requests use the same structured JSONL schema as
[`sol-nll-fold0-30-genthink`](../../../data/sol-nll-fold0-30-genthink/README.md):
`game`, `request_index`, `messages`, `tools`, `chat_template_kwargs`, and
`thinking_source: "think_gen-refine2"`. The last assistant message carries
`reasoning_content`; Python arguments are a mapping containing only `code`.
Text-only terminal assistant replies are supported too.

Use the processor assets **from the attached model directory**. The historical
upstream revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`, but its saved
processor files differ from the attached AutoRound package. Re-prepare against
the exact attached assets rather than bypassing the compatibility check. Include
the tokenizer, chat template, image/video processor configs and model config.
Nothing is downloaded by the training entry points.

For raw request-plus-reply data, join the existing `think_gen` refinement
output by `game_p0#request_index`. This preserves all context/code and replaces
only final thinking. Missing, failed or duplicate generated records stop
preparation. This code does not call a teacher API or fabricate thinking.

```bash
python exp/sft-flash-next/train/prepare_dataset.py \
  --input /path/to/incoming-request-dataset/requests.jsonl \
  --generated-dir /path/to/training-games/refine2 \
  --partition train \
  --processor /path/to/pinned-processor \
  --out /path/to/prepared-train
```

If the input already has generated thinking and the source marker, omit
`--generated-dir`. `--partition train` excludes fold-0 games **before** joining
thinking and records every exclusion. No missing training sample is silently
dropped. The full training-games refinement output is a required input; the
30 held-out generated records do not supply it.

Prepare the existing 30 samples for validation separately:

```bash
dvc pull data/sol-nll-fold0-30-genthink/requests.jsonl.dvc
python exp/sft-flash-next/train/prepare_dataset.py \
  --input data/sol-nll-fold0-30-genthink/requests.jsonl \
  --processor /path/to/pinned-processor \
  --out /path/to/prepared-validation
```

All 30 belong to fold 0 and cannot become training samples with the default
fold. Tool dictionary order is preserved as supplied; the panel export sorts
keys, unlike the earlier frozen NLL bundle. For comparisons against those
exact historical tokens, use its source-order-restored requests. For training
comparisons, evaluate both the untrained and trained model on the **same new
prepared bundle**. Do not substitute old NLL results as the matched baseline.

## Preflight, qualify, train and resume

These explicit phase examples illustrate the unmeasured A100 profile. For the
measured Blackwell setup, use the single-process launcher below; its optimized
flags must remain identical across qualification and training.

```bash
python exp/sft-flash-next/train/run.py --mode preflight \
  --model /path/to/flash-next-256-trainonly --data /path/to/prepared-train \
  --profile a100-80gb --out /path/to/preflight

CUDA_VISIBLE_DEVICES=0 python exp/sft-flash-next/train/run.py --mode qualify \
  --model /path/to/flash-next-256-trainonly --data /path/to/prepared-train \
  --profile a100-80gb --out /durable/qualification-a100

CUDA_VISIBLE_DEVICES=0 python exp/sft-flash-next/train/run.py --mode train \
  --model /path/to/flash-next-256-trainonly --data /path/to/prepared-train \
  --profile a100-80gb --qualification /durable/qualification-a100/qualification.json \
  --out /durable/sft-256 --session-minutes 110
```

Qualification runs complete training requests with the longest context, most
images and longest target, fills at least one accumulation window, and takes
an Adam step to allocate optimizer state. It checks finite loss/gradients and
requires at least 4 GiB above peak reserved VRAM (`--gpu-headroom-gib`).
The fixed margin replaces the initial 10% rule. The selected three-layer
recipe uses segmented GDN and blocked attention projections; consult the
measured reserved-memory values in LONG_CONTEXT.md.
The chosen margin is recorded in the qualification and resume identity. Its updates are
discarded when the process exits. It can be expensive; no automatic GPU job
is launched by preparing these files. Passing these extremes is a capacity
gate, not a proof that every request shape fits.

Training requires a matching successful qualification for the same data,
model, map, code, runtime, GPU and hyperparameters. An OOM fails visibly;
there is no automatic shortening, dropping of images or switch to truncated
backpropagation. If full-context qualification fails, further backend work or
more memory is needed. `--no-offload` is available only as a separately
qualified recipe.

Repeat the train command with `--resume` to continue. `--max-updates 1` limits
a session for a pilot without changing the epoch schedule. Each completed
request atomically saves adapters, Adam state, any partial gradients, RNG,
cursor and optimizer step; `latest.json` points to a checksummed complete
file. SIGINT/SIGTERM stop after the current request is checkpointed. SIGKILL
can lose that request, but not the earlier accumulated gradients. Time limits
are also checked between requests and do not shut down GPU billing.

Use an actually durable mounted filesystem for `/durable`, or arrange external
checkpoint synchronization. The code does not upload files or make ephemeral
Kaggle storage durable. The latest two complete checkpoints are retained; old payloads are removed only
after the new payload and pointer are durable. Resume is numerical-state complete; bitwise GPU replay is
not guaranteed by nondeterministic CUDA kernels.

## Compare held-out NLL before and after

Run the baseline before training and then repeat with the trained adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python exp/sft-flash-next/train/evaluate.py \
  --model /path/to/flash-next-256-trainonly --data /path/to/prepared-validation \
  --profile a100-80gb --out /durable/validation-base

CUDA_VISIBLE_DEVICES=0 python exp/sft-flash-next/train/evaluate.py \
  --model /path/to/flash-next-256-trainonly --data /path/to/prepared-validation \
  --profile a100-80gb --checkpoint-dir /durable/sft-256 \
  --out /durable/validation-trained
```

Outputs include per-target-token losses, separate thinking/code/format NLL,
token-weighted NLL, per-game mean-request NLL and its macro game mean. Evaluate
on the same GPU profile for a controlled comparison. The separate `train` mode does not automatically evaluate. The recommended
`fit` mode performs matched before/after validation; neither mode runs LR
sweeps or gameplay. For custom attention or expert-block settings, use `fit`
so both baselines inherit the exact same numerical recipe.

## Local verification

In a separate CPU environment install `../nll/requirements-cpu.lock`, then:

```bash
python -m pytest exp/sft-flash-next/train/tests -q
```

Tests cover expert input/router gradients, sparse attention gradients, selected
CE versus full CE, checkpointed multimodal hybrid backward, adapter updates,
thinking joins, data validation, partial-batch normalization, corruption and
resume-identity rejection, and exact CPU recovery during accumulation with
nonempty Adam moments. See `verification.json` for this implementation's
actual CPU and real-data checks. [CAPACITY.md](CAPACITY.md) records the subsequent
GPU smoke-test success and the original backward OOMs, including runtime and input
provenance; LONG_CONTEXT.md covers the subsequent successful optimizations. No updated adapter from these disposable probes is retained.

## Single-process fit on the attached 512-expert source

`--keep` can select 256 experts in GPU memory from the attached 512-expert
source, avoiding a second large checkpoint on Kaggle's 20 GiB working volume.
The temporary source-model load uses about 68.1 GiB of GPU memory. The map in
`artifacts/keep-256-fold0.json` is the verified train-only map from the completed
NLL archive; the map and every source shard enter the resume identity.

Prepare the incoming request dataset with `--validation-fold 0 --partition all`
and `--max-tokens 130000`. If it contains only training games, combine it with
the held-out 30-request JSONL before preparation (duplicate requests are errors).
Both inputs must already contain the verified generated thinking, or use the
matching `--generated-dir` join. Preserve the attached model's processor assets.

```bash
# SFT_REQUESTS is the incoming all-folds generated-thinking JSONL.
export SFT_MODEL=/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1
SFT_REQUESTS=/path/to/incoming-generated-thinking/requests.jsonl
python exp/sft-flash-next/train/prepare_dataset.py \
  --input "$SFT_REQUESTS" --processor "$SFT_MODEL" \
  --validation-fold 0 --partition all --max-tokens 130000 \
  --out /kaggle/working/prepared-sft
SFT_DATA=/kaggle/working/prepared-sft \
  bash exp/sft-flash-next/train/launch_fit.sh
```

The launcher defaults to the measured **175 GiB host / 96 GiB Blackwell**
configuration, including the 95.37 GiB PLE table resident in host RAM. Set
`SFT_PLE_MODE=mmap` to use NFS plus a bounded row cache on a smaller host.
The default `SFT_GDN_CHUNK_TOKENS=8192` and
`SFT_ATTENTION_PROJECTION_BLOCK=32768` balance memory and speed. For the measured
lower-memory whole-GDN variant, set `SFT_GDN_CHUNK_TOKENS=0` and
`SFT_GDN_BLOCK_TOKENS=32768`; retain the attention projection blocks.
`SFT_PROFILE=a100-80gb` selects the unmeasured Ampere profile and still requires
native-kernel and actual-data qualification on that GPU.
Set `SFT_MINUTES` to the remaining training budget, reserving time for the
other phases. `SFT_OUT` must be durable or exported before Kaggle stops.

An equivalent
explicit invocation is:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python exp/sft-flash-next/train/run.py \
  --mode fit --model /path/to/intel-source-512 \
  --keep exp/sft-flash-next/train/artifacts/keep-256-fold0.json \
  --data /path/to/prepared-all-folds --profile rtx-pro-6000-96gb \
  --out /durable/sft-run --session-minutes 900 \
  --checkpoint-group 3 --ple-block 8192 --no-ple-checkpoint \
  --gated-norm-block 262144 --rms-block-mib 40 \
  --gdn-chunk-tokens 8192 --attention-projection-block 32768 \
  --ple-resident --ple-cache-gib 0
```

`fit` loads the model once, qualifies training extremes, restores the untouched
adapters/optimizer/RNG, evaluates all fold-0 requests, trains, and evaluates
again. Add `--resume` to recover the latest request boundary. The session budget
covers the training phase; leave time for preparation, qualification and validation.

The default optimized backend uses gather-free Triton indexed attention,
bounded RMS normalization and hyperconnection recomputation, halo-aware PLE
window recomputation, 8,192-token differentiable GDN segments, 32,768-token
attention projection blocks, and resident host PLE. The mmap alternative uses parallel
PLE reads and a 1 GiB frozen-row cache. The row cache's
Python index has additional host-memory overhead. The same CLI options must be
used for qualification and training because they enter the run identity.

Optional activation disk tier:

```bash
# Append to qualify/train/fit; this changes the qualified recipe.
--disk-dir /tmp/flash-activation-scratch --disk-budget-gib 32 --prefetch 2
```

This saves exact activation bytes and prefetches upcoming backward reads into
pinned CPU buffers. It does not shorten the context or detach its gradients.
Disk writes overlap GPU work but can stall when the bounded queue fills.
Read prefetch is best-effort, not a guarantee that storage keeps up. Temporary
files are deleted after each request, including on Python exceptions.
Kaggle's reported overlay free space is misleading: the writable thin pool is
only 96 GiB. Do not copy the 95.37 GiB PLE table there. See the measured storage
path and real-device benchmarks in CAPACITY.md.

`--checkpoint-group 2` adds outer checkpoints around pairs of already
checkpointed layers; this reduces persistent host boundaries but costs extra
forward work and temporary GPU memory. `--expert-block` controls frozen-expert
microbatches. Both alternatives require their own capacity qualification.
`memory.jsonl` records host anonymous, file-backed and shared memory separately;
shared/pinned buffers are not treated as reclaimable file cache. A host-memory guard exits before exhausting the cgroup and leaves the
last atomic checkpoint recoverable.

## Processor compatibility and baseline identity

Prepare using the **processor assets attached to the selected model**. The
runner checks those assets, including `processor_config.json`, before loading
weights. The original upstream processor export has different bytes; all 30
current validation requests were re-prepared with the attached processor and
checked for identical input tokens, pixel tensors and annotations. This is not
a general permission to mix processor revisions.

The archived generated-thinking NLL run has matching target hashes but different
full-input hashes for every panel request. Use a fresh matched baseline. The
standalone evaluator records processor/data identity, source hashes, runtime,
expert microbatch and numerical recipe; diagnostic kernel error measurements
are kept separately so nondeterministic reduction noise cannot invalidate a
resume identity.

## Choosing 120K versus 80K

The measured workloads are 120K context + 10K targets and 80K context + the
**same 10K targets**. Their image counts differ (59 versus 34), so the speed
comparison includes that difference. Full details and cold/warm timings are
in LONG_CONTEXT.md. A 90,000-token preparation ceiling rejects longer real
requests; it does not truncate them or silently filter them out. Train the
incoming dataset at its full request lengths after actual-data qualification.

`--gdn-chunk-tokens` bounds the native delta-rule operation with differentiable
state carry. It can be combined with layer checkpointing and offload. The
initial/final recurrent state is never detached; full-context indexed attention
continues across the entire request. See the numerical checks and memory/speed
measurements before selecting this option for a new GPU.

## Reproduce a disposable capacity step

Prepare the verified held-out panel as above, then build the composite. Repeat
with `--context-tokens 80000` for the shorter comparison. This builder preserves
whole source requests/replies and checks the image-token/patch accounting.

```bash
python exp/sft-flash-next/train/build_capacity_sample.py \
  --panel /path/to/prepared-validation --context-tokens 120000 \
  --out /kaggle/working/capacity-130k.pt.gz
gzip -dk /kaggle/working/capacity-130k.pt.gz
PYTORCH_ALLOC_CONF=expandable_segments:True python exp/sft-flash-next/train/capacity_probe.py \
  --model "$SFT_MODEL" \
  --keep exp/sft-flash-next/train/artifacts/keep-256-fold0.json \
  --sample /kaggle/working/capacity-130k.pt --tokens 130000 --targets 10000 \
  --out /kaggle/working/capacity-new --repeat 2 \
  --attention-backend triton --query-block 16 --index-block 256 \
  --expert-block 8192 --hyper-block 1024 --checkpoint-group 3 \
  --ple-block 8192 --gated-norm-block 262144 --rms-block-mib 40 \
  --gdn-chunk-tokens 8192 --attention-projection-block 32768 \
  --ple-resident --ple-workers 8 --model-gradient-check --prefetch-weights
```

For 80K, change `--sample` and `--tokens` to the 90K-total composite. Use `--gdn-chunk-tokens 0 --gdn-block-tokens 32768` for whole-GDN blocks;
`--gpu-budget-gib 75` adds an allocator cap on the same GPU. This cap does not emulate A100 hardware.
Do not reuse an output directory. Inspect `result.json` and
`completed-cases.json`: a later suite OOM does not invalidate earlier completed
cases, but it does make the suite fail. Diagnostic updates are discarded,
including between repeated trials. Download the full original evidence using the
[capacity archive instructions](capacity-results/README.md).

## Portable Kaggle package

A full Git checkout already has the dependencies. For uploading a small runtime
bundle to a notebook instead, the packager includes the fold map and **all**
tiny-model test fixtures as well as training/NLL/model code:

```bash
python exp/sft-flash-next/train/build_runtime.py \
  --out /path/to/training-runtime.tar.gz \
  --prepared-data /path/to/prepared-all-folds
# On the GPU host, extract into a fresh directory:
mkdir -p /kaggle/working/training
tar -xzf /path/to/training-runtime.tar.gz -C /kaggle/working/training
cd /kaggle/working/training
python -m pytest exp/sft-flash-next/train/tests -q
SFT_DATA=/kaggle/working/training/prepared-data \
  bash exp/sft-flash-next/train/launch_fit.sh
```

Omit `--prepared-data` to package code only. Model weights and native wheels
are not bundled. The packager writes the archive's SHA256 beside the tarball.
The CPU reference tests explicitly select Torch reference convolution/GDN on
CUDA hosts. Separate CUDA tests exercise indexed attention, differentiable
GDN state/history carry and checkpointed attention projections. The runner
also executes independent native CUDA convolution/GDN and model-gradient gates
before qualification; selecting CPU references in unit tests does not bypass
those production checks.
