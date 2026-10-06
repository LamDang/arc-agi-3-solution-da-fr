# REAP statistics for Qwen3.8-Flash-Next on ARC-AGI-3 traces

Step 1 of the pruning plan: replay logged games through Flash-Next and record,
for every layer and expert, what REAP needs to rank experts. Step 2
(`analyze.py`) turns that into kept-expert sets and checks them on games the
selection never saw.

## Data

| Source | Games | Passes | Served by | Where |
|---|---|---|---|---|
| `kaggle_v3` | all 25 public games | 4 (100 game runs) | SGLang, Intel W4A16, RTX PRO 6000 | output of [dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) v3 (mean score 46.49) |
| `openrouter` | ft09, lp85, ls20, sp80, vc33 | 1 | OpenRouter, Alibaba | `ARC3-Inference/runs/20261004_135539` (DVC), Kaggle dataset `lamdang/arc3-openrouter-traces` |

Both used this repo's harness with the submission settings and request logs
on. The `kaggle_v3` traces come from the exact model and server the
submission runs, so they are the main calibration set.

## How a replay works

1. **Stretches** (`traces.py`). Between history trims every request extends
   the previous one, so the last request of a stretch plus its reply holds
   every token of the stretch once. The 5-game OpenRouter run has 37
   stretches and about 3.9M tokens to replay, against 55.5M prompt tokens
   billed. Older logs repeat the request on response lines; replies are then
   recovered from the next request. A run-level `requests.jsonl` (a since-fixed
   naming bug) is attributed to games by shared messages.
2. **Rendering** (`render.py`). Messages go through the model's chat template
   with `preserve_thinking`, tool-call arguments parsed to mappings, and tools
   serialized as SGLang's `Tool.model_dump()` does. Every request of a Kaggle
   log renders to exactly the `prompt_tokens` SGLang logged. OpenRouter logs
   come out a constant 11 tokens longer per request (the provider serializes
   the tool definition differently), so the messages render identically.
   Tokens are tagged context, generated (assistant turns) or image.
3. **Model** (`reap_model.py`). transformers 5.18 `qwen4_exp`, loaded from
   the Intel checkpoint with three replacements:
   - the routed experts stay GPTQ int4 on the GPU (62 GB) and are dequantized
     per layer into a shared 5 GB buffer; one `torch._grouped_mm` call per
     projection computes every expert's output, so REAP's ||f_j(x)|| is exact
     (falls back to a per-expert loop if grouped matmul fails a self-check);
   - full attention keeps the QSA indexer's selection rule (top 512 blocks of
     4 tokens plus the incomplete last block) but selects per query and gathers,
     instead of the reference's Python loop over queries and dense masks;
   - the 51B-parameter n-gram table is memory-mapped from the safetensors files
     in host RAM.
   Everything else is BF16 on the GPU (10.8 GB).
4. **Replay** (`replay.py`). The vision tower runs once per sample, then the
   sequence goes through in 8K-token chunks sharing one cache. Besides the
   statistics it scores the model on its own logged generations (next-token
   top-1 and NLL): those tokens were sampled from this model, so a low score
   means the replay is broken.

## Output

`OUT/stats/<source>/<game>_p<pass>.npz` per game run, arrays
`[layer, expert, category]` (48 x 512 x 3), categories context / generated /
image:

| Array | Sum over tokens routed to the expert |
|---|---|
| `count` | 1 |
| `gate` | router weight g_j(x), after top-10 renormalization |
| `norm` | expert output norm ||f_j(x)|| |
| `gate_norm` | g_j(x) ||f_j(x)||; REAP score = `gate_norm / count` |
| `prob` | full softmax probability, routed or not (over all tokens) |

Sums, not averages, so any set of games or categories combines by addition.
A `.json` next to each file has per-sample tokens, timing, peak GPU memory and
the next-token check. `analyze.py OUT` writes `analysis.json`: routing
concentration, in-sample coverage, and leave-one-game-out coverage per
kept-expert count.

## Validation done

Local, CPU (`pytest tests`, 22 tests):

- **Tiny model** with the same architecture and a fake Intel-format checkpoint
  (`tests/tiny.py`):
  - the loader reproduces every reference weight;
  - experts and REAP sums match a brute-force computation;
  - token selection matches the reference indexer (ties at score 0 may pick
    different but equally scored blocks; `topk` tie order is unspecified);
  - the whole model, replayed in 23-token chunks with an image, matches the
    reference logits to within 2e-4;
  - grouped matmul matches the loop.
- **Real Intel checkpoint**: the int4 format was checked with HTTP range
  reads. Layer 0, expert 0 dequantizes to correlation 0.993 with the original
  BF16 weights; reversed nibble order gives 0.
- **Real log** (`REAP_TEST_LOG`, `REAP_TEST_PROCESSOR`): every request of a
  Kaggle log renders to the logged length, and `run_reap.py` runs end to end
  on CPU with consistent sums.

Kaggle (`kaggle/smoke.json`): see "Smoke test" below.

## Running

```bash
# local tests (CPU): transformers 5.18, torch, torchvision, pillow, safetensors, pytest
pytest tests
REAP_TEST_LOG=path/to/sk48-..._p0_requests.jsonl REAP_TEST_PROCESSOR=path/to/processor_files pytest tests

# on a GPU with the checkpoint
python run_reap.py --model-dir MODEL --traces kaggle_v3=DIR --out out/ --games ft09,ls20 --precache
python analyze.py out/

# on Kaggle: uploads code (+ traces) as private datasets, pushes and starts the kernel
KAGGLE_CLI=kaggle python kaggle/push.py --config kaggle/smoke.json --traces-dir ../../ARC3-Inference/runs/20261004_135539
```

`--samples SOURCE/GAME_pPASS:STRETCH,...` picks samples, `--max-tokens`
truncates them, `--deadline-minutes` stops before Kaggle's 12-hour limit, and
finished game runs are skipped on restart.

`prune_eval.py` calibrates and then replays held-out games with pruned routers
(softmax over kept experts only, as if the others were deleted).
`bench.py` times the speed options (`--compile-dequant`, `--attention sdpa`)
against each other; not yet run on a GPU.

### Kaggle constraints

- The RTX PRO 6000 (97.9 GB, 176 GB RAM) is only offered to notebooks attached
  to the `arc-prize-2026-arc-agi-3` competition, and those must run with the
  internet off. A kernel pushed without both silently gets T4 x2.
  `kaggle/push.py` sets both and passes `--accelerator NvidiaRtxPro6000`.
- With no internet, dependencies come from the private dataset
  `lamdang/reap-flash-next-wheels` (transformers 5.18 and its dependencies,
  flash-linear-attention 0.5.2). `causal_conv1d` has no sm_120 wheel; its
  PyTorch fallback is cheap and used on purpose.
- `/kaggle/working` is lost when an interactive session stops and cannot be
  uploaded from it; zip it and download it from the Output panel.
  `kaggle/interactive_session.ipynb` is the notebook used for the runs below.

## Results (interactive RTX PRO 6000 session, 2026-10-06)

### Smoke test

Three samples from two game runs, full length:

| Sample | Tokens | Images | tok/s | Top-1 on own generations | Peak GPU |
|---|---:|---:|---:|---:|---:|
| `kaggle_v3/sk48_p0:0` | 117,582 | 40 | 845 | 0.879 | 88.1 GB |
| `kaggle_v3/sk48_p0:1` | 51,796 | 13 | 848 | 0.835 | 86.0 GB |
| `openrouter/ft09_p0:0` | 116,610 | 27 | 863 | 0.884 | 88.0 GB |

Overall top-1 0.874, NLL 0.363. Loading took 730 s from a cold disk and
12-21 s once the files were in the page cache. With flash-linear-attention
the replay runs at about 950 tok/s, so all 100 `kaggle_v3` runs (45-50M
tokens) would take about 15 hours: two sessions, or the speed options.

Routing is spread out: per layer, the median number of experts carrying
50 / 90 / 99% of the router weight is 68 / 260 / 416.

### Pruned routers on held-out games

Calibration: the first stretch (up to 64K tokens) of 8 `kaggle_v3` pass-0
runs plus the two smoke runs, 10 runs and 9 games. Evaluation: the first
stretch (up to 32K tokens) of pass 1 of ls20, sb26 and vc33, none of them in
calibration; 35,514 generated tokens scored. Mean over the three games of the
next-token NLL increase over the full model (lower is better):

| Experts kept | REAP (mean g\|\|f\|\|) | REAP, generated tokens only | `gate` (sum g) | **`gate_norm` (sum g\|\|f\|\|)** |
|---:|---:|---:|---:|---:|
| 448 | +0.010 | | | |
| 384 | +0.034 | +0.022 | +0.015 | **+0.012** |
| 320 | +0.065 | +0.036 | +0.039 | **+0.035** |
| 288 | +0.090 | | | |
| 256 | +0.114 | +0.080 | +0.085 | **+0.067** |
| 192 | +0.204 | | | |
| Top-1 at 256 (full model 0.856) | 0.821 | 0.830 | 0.830 | **0.835** |
| Agreement with the full model at 256 | 0.878 | 0.894 | 0.895 | **0.901** |

- `gate_norm` wins at every size and on each game (at 256: ls20 0.084 vs
  0.106 for `gate`, sb26 0.054 vs 0.068, vc33 0.062 vs 0.081). It is REAP's
  quantity summed instead of averaged: the average lets rarely routed experts
  with large outputs outrank experts the router uses constantly. It is now
  the default in `analyze.py` and `prune_eval.py`.
- With `gate_norm`, 256 experts cost what 320 cost with REAP; 384 is nearly
  free (+0.012 NLL, about 3%). Mean full-model NLL is 0.417.
- Ranking on generated tokens only helped REAP (0.080 vs 0.114 at 256);
  `gate_norm` with generated or context+generated tokens is the next test.
- Noise floor: replaying the full model twice gives 96-100% top-1 agreement
  (GPU kernels are not bitwise deterministic) and NLL within 0.0007. Compare
  agreement figures with that, not with 1.0; NLL is the reliable measure.
- The metric is held-out NLL: rankings and expert counts are chosen by it,
  never by the coverage `analyze.py` prints. Coverage (share of router weight
  on kept experts) needs no GPU, so it is a quick screen, but it is not what
  pruning costs: `gate` maximizes it by construction and still lost to
  `gate_norm` on NLL. The ranking was chosen on the same three games it is
  reported on, so the final choice should be confirmed on games not used for
  any decision, and in the end on game scores.

Statistics from these runs (`kaggle_v3` ar25, bp35, cd82, dc22, g50t, re86,
tn36, tu93, plus the smoke runs) are in `reap_results.zip` from the session,
not in git.

## Serving throughput, full vs pruned

Picks the serving config before the game runs: do N full-length requests
fit on the pruned model, and how much faster does it decode than the full
model at dfranzen's 10? `kaggle/push_serve_bench.py` builds a notebook from
dfranzen's paths, precaching and SGLang launcher cells (same wheels, flags
and speculative decoding). It writes a 256-expert checkpoint from
`kaggle/keep_256_smoke.json` (ranked on the smoke statistics; which experts
are kept barely matters for speed). It then serves:

- the full model exactly as dfranzen does (10 requests, 60 state slots),
  tested with 10 streams;
- the pruned model sized for 28 requests, tested with 10, 16, 20 and 28 streams.

The test is `serve_bench.batch_test`: N requests at once, each the longest
logged prompt of a different game run (about 100-110K tokens, near the
harness's context limit). They are prefilled first, then sent again to
decode 4096 tokens each from the cache. It reports:

- decode tokens/s with all N running, and the ratio to the full model at 10;
- whether N fits: all N running, phase-2 cache hit about 1, peak KV use, no
  retractions;
- speculative accept length.

About 50 minutes, mostly installing and loading.

```bash
KAGGLE_CLI=kaggle python kaggle/push_serve_bench.py
```

## Real-game test of a pruned model

Next-token NLL on logged games is a proxy; the test that counts is playing.
Two Kaggle runs, both on the RTX PRO 6000:

1. **Calibration statistics** (`kaggle/calib.json`, kernel `lamdang/reap-flash-next`,
   about 50 minutes): the whole first stretch of pass 0 of all 25 games, up to
   the harness's first history trim at about 116K tokens (at most 3M tokens).
   Full length rather than truncated, because the games run near the context
   limit and only the late tokens show how experts are routed at long context.

   ```bash
   KAGGLE_CLI=kaggle python kaggle/push.py --config kaggle/calib.json
   ```

2. **Games** (`kaggle/push_games.py`, kernel `lamdang/flash-next-games-<keep>-<fold>`):
   dfranzen's submission notebook as published, plus one cell before the server
   starts. That cell ranks experts (`gate_norm`) from the statistics of the games
   this run does not play, writes the pruned checkpoint with
   `prune_checkpoint.py` and points `MODEL_DIR` at it. The harness, SGLang
   build and flags, speculative decoding and the per-game time budget (scaled
   to the competition's) are unchanged, so scores compare with dfranzen's v3
   run (25 games x 4 passes, mean 46.49).

   ```bash
   KAGGLE_CLI=kaggle python kaggle/push_games.py --fold a --keep 256 --maxreq 20   # 13 games x 4 passes, ~4.6 h
   KAGGLE_CLI=kaggle python kaggle/push_games.py --fold b --keep 256 --maxreq 20   # the other 12 games, ~4.3 h
   ```

   Running both folds plays every game with experts chosen without it.
   `--maxreq 20` spends the ~31 GB the pruning frees on twice dfranzen's 10
   concurrent requests, which is the setup a submission would use. It also
   raises the linear-attention state cache (SGLang otherwise caps running
   requests at 60 slots / 6 per request, with only a warning) and adds CUDA
   graphs for 12-18. If the score drops, `--maxreq 10` separates the pruning's
   quality loss from contention, and `--keep 512` plays the same games
   unpruned.

`prune_checkpoint.py` keeps the top N experts of every layer, renumbered
0..N-1, slices each router to their rows and sets `num_experts` to N. It
copies tensor bytes without decoding them, so it needs only numpy. Shards
without routed experts, the n-gram table, tokenizer and chat template are
symlinked. A test checks that the pruned checkpoint computes what the full
model computes with the same experts masked out of the router. SGLang's
loader reads `num_experts` from the config and maps expert tensors by index.
256 experts takes its power-of-two top-k fast path; 320 and 384 use the
generic one. Only a GPU run shows whether its kernels accept the pruned shapes.
