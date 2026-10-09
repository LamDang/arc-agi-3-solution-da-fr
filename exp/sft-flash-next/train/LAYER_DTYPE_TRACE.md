# Native layer dtype trace — 2026-10-09

## Result

**The first persistent FP32 hidden-state promotion is the AutoRound MoE
expert-output sum in decoder layer 0, not the LoRA matrix multiplies.**

The executed source is `auto_round/modeling/fused_moe/moe_experts_interface.py`,
`linear_loop_experts_forward`, line 286:

```python
final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
```

With BF16 CUDA autocast enabled, recorded operation 18957 is:

```text
aten.sum.dim_IntList
BF16 [32, 10, 2560] -> FP32 [32, 2560]
```

PyTorch 2.11 lists `sum` among CUDA operations that autocast to FP32.
See the [versioned AMP operation reference](https://docs.pytorch.org/docs/2.11/amp.html#cuda-ops-that-can-autocast-to-float32).
This sum's output is returned without restoring the expert input dtype.
The FP32 routed-expert output is added to the BF16 shared-expert output,
then injected into the residual stream. Operation 18968 records the residual
addition as BF16 + FP32 -> FP32 at native decoder line 1308.

This is a consequence of the installed AutoRound expert adaptation combined
with PyTorch autocast policy; it is not evidence that Qwen requires FP32
residual storage. It explains the promotion in this bounded diagnostic.

## Observed path

| Boundary | Input | Output |
| --- | --- | --- |
| Token embedding | integer token IDs; BF16 weight | BF16 |
| Layer 0 attention hyper-connection | BF16 | BF16 |
| Layer 0 linear attention | BF16 | BF16 |
| Layer 0 MLP hyper-connection | BF16 | BF16 |
| Layer 0 selected-expert sum | BF16 | **FP32** |
| Layer 0 MLP/residual output | BF16 residual + FP32 injection | **FP32** |
| Layer 1, including prepared PLE | FP32 | FP32 |
| Layer 2 | FP32 | FP32 |
| Layer 3, including indexed attention | FP32 | FP32 |

All **62 observed LoRA A/B linear calls returned BF16**. All 744 adapter
parameters remained FP32. PEFT casts some adapter inputs to FP32 to match
parameter storage, then eligible linear calls execute under BF16 autocast.
Thus FP32 adapter storage does introduce temporary FP32 input copies, but
does not cause the persistent residual promotion found here.

The first temporary conversion occurs earlier, in native RMSNorm's
`x.float()` at line 167. Its output returns to BF16. Linear-attention decay,
gated normalization and router softmax also perform FP32 work; these must
not be confused with the later FP32 expert-sum return.

## Scope and measurements

- Execution commit: `3a5a58fb853d58acc9bce132302feddbde000b7f`.
- Attempt: `20261009184603-203da5fa`.
- Same native HF/AutoRound/PEFT loader, checkpoint config, and exact nonzero
  test-only adapter as v5. Native Opt3 mask and Opt4 PLE lookup retained.
- **32 token IDs from the anchor prefix; only layers 0–3 execute.** Text-only
  language-model input: no image injection or vision execution. This is a
  deliberately shortened diagnostic, not a training trajectory.
- Forward-only under `no_grad`, model training mode, BF16 autocast. No loss,
  backward, gradient archive, clipping, optimizer construction or update.
- 37,176 recorded ATen operations, module entry/exit metadata, source stacks,
  input/embedding/PLE tensors and four raw hidden outputs saved.
- Loading/setup: 105.454 s; traced short forward: 10.109 s. Trace overhead and
  first-use kernel tuning make this unsuitable for training-speed comparison.
- Forward GPU allocated peak: 39.657 GiB; this includes the full resident
  frozen model. Forward GPU reserved peak is retained in `report.json`.
- Forward sampled process RSS peak: 3.136 GiB; whole-host used peak: 5.067 GiB.
  RAM sampling interval 20 ms; these are sampled peaks, not exact allocation
  maxima. No backward measurement exists because none was run.

TorchDispatch records ATen boundaries and module hooks expose custom kernel
outputs. It does not inspect arithmetic or accumulation inside Triton/CUDA
kernels. Small-shape dispatch can differ from full-context kernels. This
diagnostic does not establish full-sample gradient equivalence or a 130K
memory estimate.

## Evidence and reproduction

- Config: `configs/layer-dtype-trace-v1.json`.
- DVC stage: `layer_dtype_trace_v1`.
- Artifacts: `gradient-results/layer-dtype-trace-v1/`.
- Metrics: `metrics/layer-dtype-trace-v1.json`.
- Independent review: `reviews/layer-dtype-trace-v1.json`.
- Framework comparison: `QLORA_DTYPE_REVIEW.md`.

The remote manifest's 50 files were independently hash-verified. The narrow
initial source collector omitted AutoRound's **fused** MoE module. That exact
source was collected afterward as a supplemental source artifact; SHA256
`2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c`
also matches v5's previously recorded imported-source hash. The original
remote manifest remains intact and the supplement is labeled separately.

All **63 final artifact files** were restored identically from DVC. The
artifact tree is `ae2294f3a1dd56b543600f6e0ca68284.dir`; it includes the frozen
runner/config/source snapshots and supplemental MoE source. DVC push completed.

An unchanged DVC reproduction restores/skips cached execution; that is not a
fresh numerical replay. The Jupyter runner resumes an existing attempt when
its local resolved config exists, so collector recovery does not duplicate
the GPU forward. A fresh diagnostic requires a new run ID/output/config.

## Candidate change to discuss next

A targeted future candidate can keep the selected-expert reduction in FP32
and cast its returned result back to the original activation dtype before
residual mixing. That may keep the residual stream BF16 while preserving
FP32 reduction arithmetic. It still changes numerical results and needs a
separate measured loss/gradient comparison. No such change was made here.
