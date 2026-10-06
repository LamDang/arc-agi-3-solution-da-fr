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

Local, CPU (`pytest tests`, 14 tests):

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

## Smoke test

Pending.
