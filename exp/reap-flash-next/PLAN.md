# Fine-tuning and deployment plan (2026-10-07)

What comes after run A (256 experts lost 23 points against the full model,
README "Run A"). Measured numbers are marked as such; everything else is an
estimate to be checked by the GPU benchmark at the end. Nothing here has
been run as training yet.

## Serving under KV pressure

What SGLang does when the running requests' contexts outgrow the KV pool:

- **Admission control**: a new request waits in the queue until its prefill
  fits.
- **Retraction**: if a running request cannot grow by its next decode
  tokens, the scheduler evicts the request with the longest output
  (`retraction_policy 'length'`), frees its KV and requeues it to be
  recomputed later. The work is not lost, but its prefill is paid twice.

dfranzen's v3 run (full model, mem fraction 0.96, 10 requests, 1.01M-token
pool, about 1.3x oversubscribed against 131K contexts) had 0 retractions; in
6% of the log lines requests were queued while 5-9 were running, i.e. the
pool was full. Run A (256 experts, 20 streams, 2.59M pool) peaked at 72% KV
and never queued for memory.

**dfranzen does not restart the server.** `--watchdog-timeout 1800` only
detects hangs. The harness retries HTTP with backoff and gives up a game
after `ARC3_MAX_ANALYZER_FAILURES` (default 10) consecutive failures
(`-1` retries forever); nothing restarts SGLang. `_PriorityGate` with
`ARC3_MAX_ACTIVE_STREAMS` caps concurrent LLM calls. Chunked prefill 8192,
`max_prefill_tokens` 16384.

**Our crash was activation memory, not KV**: at 28 streams the 256-expert
server ran out of memory in the linear-attention short convolution during
prefill, with 0.7 GB free after startup. Safety settings for a submission:

1. leave about 8-9 GB free after startup and CUDA graphs (mem fraction
   ~0.91 instead of 0.93-0.96);
2. optionally `--max-prefill-tokens 8192`;
3. validate with the worst-case batch test (all streams at ~120K contexts);
4. a restart supervisor plus `ARC3_MAX_ANALYZER_FAILURES=-1` only as a last
   resort.

## Throughput in a competition run, per expert count

Measured in real games (both runs 7-25 games x 4 passes, same harness):

| run | streams | running (half-hour means) | decode tok/s |
|---|---:|---|---|
| v3, full model | 10 | ~9.5 throughout | ~690 throughout, 594 over the job |
| A, 256 experts | 20 | 19.2, 18.5, 16.3, 12.0, 8.3, 1.5 | 989, 932, 874, 758, 599, 222; 717 over the job |

Run A's taper is the end of the job: only 28 runs were queued, so fewer
games were left than streams.

Decode throughput depends almost only on the number of running requests,
not on the expert count: every token still goes through 10 experts, and
v3's point lies within 6% of run A's curve, which fits
`tok/s ~ 169 * running^0.6` within 3%. Pruning therefore pays by freeing
memory for more concurrent streams, not by making a token cheaper.

Estimates at mem fraction 0.91 (the safe setting), KV pool sized for 1.3x
the streams' full 131K contexts (about dfranzen's oversubscription), 96% of
streams in an LLM call at a time (as in run A). Weights
`W(N) ~ 10.9 + 0.1151 N` GB, KV ~12.4 KB/token with the draft model,
6 state slots of 56 MB per request.

| experts | streams | running | tok/s | vs dfranzen | tok/s per game | basis |
|---:|---:|---:|---:|---:|---:|---|
| 512, dfranzen's settings (0.96, ~4 GB free) | 10 | 9.5 | 690 | 1.00x | 73 | measured (v3) |
| 512, safe | 8 | ~7 | ~540 | 0.78x | 77 | estimate |
| 448 | 11 | ~10.6 | ~690 | 1.0x | 65 | estimate |
| **384** | 15 | ~14.4 | **~830** | **1.2x** | 58 | estimate |
| 320 | 20 | ~19.2 | ~990 | 1.4x | 51 | run A's 20-stream point (256 experts) |
| 256 | 25 | ~24 | ~1,130 | 1.6x | 47 | extrapolated past 20 streams; 28 crashed |

Caveats:

1. The gain needs at least as many active game runs as streams. Over the
   whole of run A the gain was 717 vs 594 tok/s, 1.21x rather than 1.4x,
   because of the tail.
2. Each game decodes more slowly (73 down to 47-58 tok/s). Under a per-game
   wall-clock budget a game gets fewer tokens.
3. Contexts average about 90-100K, so the pool could be oversubscribed
   ~1.45x before queueing; beyond that, queueing and retraction cost
   throughput. Not measured.
4. The 20-25 stream rows assume the extra ~2 GB free at 0.91 and the 8192
   prefill cap prevent the 28-stream crash. The worst-case batch test has
   to confirm it.
5. Throughput is not score: at equal tokens run A lost 23 points. The
   speedup only pays if fine-tuning brings quality back.

Expected under real conditions: ~1.2x at 384, ~1.4x at 320, less after the
end-of-job tail.

## Fine-tuning data

`kaggle_v3` is the output of dfranzen's public notebook
[arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
v3: the full 512-expert int4 model on SGLang, one RTX PRO 6000, their
harness, all 25 public games x 4 passes (100 runs, 8 h 11 min on
2026-10-03, mean score 46.49, 33/100 won), with request logging on. Every
LLM call is a line of `<game>_p<pass>_requests.jsonl` holding the full
message list (system prompt, frames, tool calls and results, assistant turns
with thinking preserved) and SGLang's token usage.

One training sequence per stretch (`traces.py`): the last request of a
stretch plus its reply holds every token of the stretch once. Loss only on
assistant turns. `trace_volume.py` counts it:

| | run A, 28 runs (measured) | v3, 100 runs (scaled from A) |
|---|---:|---:|
| requests | 4,155 | ~13-15K |
| training sequences | 167 | ~500-600 |
| tokens to process once | 18.2M | ~45-55M |
| generated tokens (with loss) | 5.9M (33%) | 17.5M (measured) |
| prompt tokens billed, repeats included | 311M | ~1B |

Sequence lengths in run A: p10 89K, median 115K, p90 120K, max 130K. 84% of
the tokens are in sequences of at least 105K tokens (80% of the context), so
long context is nearly all of the data, not a subset.

Limits: the traces are the full model's own outputs (training the full model
on them is close to a no-op; training a pruned model on them is
self-distillation back to the full model), they are not filtered (67 runs
lost), and they cover only the 25 public games, so evaluation needs held-out
games. More data costs about 8 GPU hours per 100 runs.

## Exact long-context gradient on one GPU

Truncating the gradient at chunk boundaries (detached caches, as in
Transformer-XL or RNN truncated BPTT) would stop later tokens from teaching
how earlier tokens are encoded. Most tokens are observations without loss,
and pruning damages their encoding too, so that path matters. Long-context
LLM training does not truncate: it computes the full gradient with
FlashAttention, activation checkpointing and context parallelism (Ring
Attention, Ulysses, Megatron CP) across GPUs; single-GPU fine-tunes use
offloaded checkpoints (Unsloth) and chunked loss over the vocabulary
(Liger, MsT). LongLoRA is the known approximation (sparse attention in
training only).

Chunk-wise exact gradient (backprop through time with checkpointing at
chunk boundaries):

1. forward all chunks without gradients, keeping only what crosses a chunk
   boundary: K/V of the 12 full-attention layers, and the linear-attention
   and short-conv state at each boundary;
2. go back from the last chunk: recompute chunk k with gradients, with the
   stored K/V and incoming state as leaf inputs that require gradients;
   backpropagate its loss plus the gradients that later chunks sent to its
   own K/V and final state; add the gradients of its inputs into running
   buffers for the earlier chunks.

The result is the gradient of one 131K pass. The sparse indexer's top-2048
selection is not differentiable either way. Check on CPU: chunked gradients
must match a single pass on the tiny config to rounding.

## Training memory and time per expert count

Throughput: forward-only replay runs at 905-950 tok/s (measured on 512). The
base is frozen, so a backward costs about one forward; an exact step is
about 4 forward-equivalents (forward without gradients, forward,
checkpoint recompute, backward), ~240 tok/s; truncated is about 3, ~315
tok/s. Every token uses 10 experts at any expert count, so time barely
depends on it. The backward cost of the linear-attention and indexer kernels
is unmeasured: +-30%.

| experts | weights (int4) | exact gradient, chunks 8K / 4K | fits 96 GB | tok/s | 1 pass (~50M tokens) |
|---:|---:|---:|---|---:|---:|
| 512 | 70 GB | ~101 / ~96 GB | only with offload and 2K chunks (below) | ~220-250 | ~2.3-2.6 days |
| 448 | 62 GB | ~93 / ~88 GB | 4K chunks | ~225-255 | ~2.3-2.6 days |
| 384 | 55 GB | ~85 / ~80 GB | yes | ~230-260 | ~2.2-2.5 days |
| 320 | 48 GB | ~77 / ~72 GB | yes | ~235-265 | ~2.2-2.5 days |
| 256 | 40 GB | ~70 / ~65 GB | yes | ~240-270 | ~2.1-2.4 days |

Truncated gradients take 5-10 GB less and ~1.8 days. The 512 smoke test
already peaked at 88 GB doing forward-only replay.

Smaller chunks barely help at 512, because most of the memory does not
depend on the chunk:

| part | depends on chunk | estimate |
|---|---|---:|
| weights | no | ~70 GB |
| one layer's experts dequantized to bf16 | no | ~5 GB |
| cross-chunk K/V, their gradients, boundary states | no (on total context) | 5-10 GB |
| CUDA context, allocator, LoRA and optimizer | no | ~3 GB |
| activations of the current chunk | yes | 8K ~11, 4K ~6, 2K ~3, 1K ~1.5 GB |

And small chunks cost speed: each pass dequantizes every expert of all 48
layers (~0.3 s of memory traffic), about 1.2 s per chunk over 4 passes.
That is +3.5% at 8K, +7% at 4K, +14% at 2K, +28% at 1K (plus ~20 tokens per
expert GEMM, far from peak) and +55% at 512 tokens. 2K is the floor.

What does fit 512: keep the cross-chunk buffers in pinned CPU memory and
stream them per chunk (saves 5-10 GB; ~3 GB of K/V per chunk over PCIe 5 is
~60 ms), and dequantize half a layer's experts at a time (5 GB -> 2.5 GB).
At 2K chunks that is about 77-81 GB. The script takes chunk size and offload
as options so the benchmark can sweep them.

Kaggle: a pass is longer than one GPU session, so training checkpoints and
resumes (~5 sessions per pass) and may hit the weekly quota. A logit
teacher pass (512's top-k over the 50M tokens) is ~15 h once and a few GB.

Plan: 384 (or 320). The pass costs the same as at 512, memory is
comfortable at 4-8K chunks, and serving gets 12-15 streams at 384 (about 20
at 320) instead of 6-10.

## Precedent for a pruned student learning a stronger teacher

From memory up to mid-2026; citations to be checked before quoting.

Smaller students on a stronger teacher's outputs:

- DeepSeek-R1 distillation (2025): ~800K R1 traces, SFT into dense 1.5-70B
  models; the 32B distill beat o1-mini on maths and code, and SFT on traces
  beat RL on the same small model.
- Qwen3 strong-to-weak distillation (2025): off-policy SFT, then on-policy
  logit distillation from the large models; better than RL at a fraction of
  the GPU hours, on-policy being the step that mattered.
- SWE-smith / SWE-Gym (2025): frontier-model agent trajectories, SFT into
  Qwen2.5-32B; multi-turn tool-using coding agents distil from a few
  thousand trajectories. Closest to our harness.
- OpenThoughts, s1, LIMO (2025): small curated reasoning sets; much of the
  gain draws out what the base can already do.

Prune, then distil to recover:

- Minitron (NVIDIA, 2024), Nemotron-Nano-2 (12B -> 9B): width/depth pruning
  then logit distillation from the unpruned model on ~2-3% of the original
  tokens recovers most of the loss.
- REAP (Cerebras, 2025), our ranking: up to ~50% of experts removed from
  Qwen3-Coder and GLM-4.5-Air near-losslessly on coding and agentic tasks
  without retraining; pruning beat merging for generation.
- Task-specific expert pruning (Chen et al. 2022), SEER-MoE, MoE-Pruner
  (2024): keeping a task's experts and then fine-tuning or distilling keeps
  most of the performance. A narrow task is the favourable case.

Warnings that match what we saw:

- Compression costs little perplexity but much on multi-step and
  knowledge-heavy tasks (LLM-KICK, 2023). Ours: +0.067 NLL, -23 points.
  Evaluate in games.
- Imitation copies style more than capability on broad tasks (The False
  Promise of Imitating Proprietary LLMs, 2023); narrow tasks transfer
  better.
- A teacher too far ahead can teach less ("Small Models Struggle to Learn
  from Strong Reasoners", 2025). Our student keeps the same active compute,
  so the gap should be small; 512 can serve as an intermediate teacher.
- Multi-turn drift: off-policy SFT trains on the teacher's states, the
  student plays its own. GKD, MiniLLM (2024) and on-policy distillation
  score the student's own rollouts with the teacher. Run A's 20-60% more
  actions on the same tokens is that drift.

Not much precedent for exactly our case: an expert-pruned MoE distilled on
long multi-turn agentic games with images at 100K+ context. A different
teacher model with another tokenizer allows only sequence-level SFT on its
text, in this harness's exact format.

## Order of work

Each step checked in games, not on loss:

1. Recovery KD from 512 into 384 (or 320) on the v3 traces (the Minitron
   recipe). Shows whether pruning plus recovery closes the -23 gap; if not
   at 384, a stronger teacher is unlikely to help.
2. SFT on a stronger teacher's traces in the same harness, ideally filtered
   to level-clearing runs or segments.
3. On-policy: the student plays, the teacher scores (512: logits) or
   relabels (API teacher) its turns.

Before step 1:

- training script (exact chunk-wise gradient, LoRA, per-layer
  checkpointing, loss on generated tokens in 1K sub-chunks, chunk size and
  offload options), tested on CPU against a single pass on the tiny config;
- ~1.5 h GPU benchmark: peak memory and tok/s at 512 and 384 experts, 131K
  context, 2K/4K/8K chunks, with and without offload; and the worst-case
  serving test (384 experts, 0.91, 15 streams: no crash, few retractions).
