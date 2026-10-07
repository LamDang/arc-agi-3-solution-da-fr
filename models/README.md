# Pruned Flash-Next models

Recipes for Qwen3.8-Flash-Next (dfranzen's Intel W4A16 AutoRound
checkpoint, 512 routed experts per layer, top-10) with fewer experts per
layer, chosen from REAP statistics on dfranzen's v3 harness traces. Each
directory holds `keep.json` (kept expert ids per layer) and a README on how
it was made, what it costs and how to serve it. `extract.py` builds the
checkpoint from the original model; the weights are not stored.

| model | experts | router weight kept | held-out NLL increase | weights in SGLang | rewritten on disk | games |
|---|---:|---:|---:|---:|---:|---|
| [flash-next-reap-448](flash-next-reap-448/README.md) | 448 | 0.995 | n/a | ~62.5 GB | ~62.9 GB | not played |
| [flash-next-reap-384](flash-next-reap-384/README.md) | 384 | 0.979 | +0.012 | ~55.1 GB | ~55.5 GB | not played |
| [flash-next-reap-320](flash-next-reap-320/README.md) | 320 | 0.949 | +0.035 | ~47.7 GB | ~48.1 GB | not played |
| [flash-next-reap-256](flash-next-reap-256/README.md) | 256 | 0.899 | +0.067 | 40.4 GB (measured) | 40.7 GB (measured) | 45.95 vs 68.97 (run A) |
| full model | 512 | 1 | 0 | 69.9 GB | | 68.97 (v3, same 7 games) |

The sets are nested (256 in 320 in 384 in 448); the 256 set is the one
served in run A. Selection: top N per layer by `gate_norm` (sum of router
weight times expert output norm) over 2.93M tokens of the 25 public games.
Details and measurements: `exp/reap-flash-next/README.md`; plans and
serving estimates: `exp/reap-flash-next/PLAN.md`.

## Build

```bash
python models/extract.py --experts 384 \
    --source /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1 \
    --out /tmp/flash-next-reap-384 [--copy]
```

Only numpy is needed. Tensor bytes are copied without decoding: about a
minute from a warm disk (the 256 build took 35 s on Kaggle). Without
`--copy` the unchanged files are symlinks to the source.

On Kaggle the private dataset **`lamdang/flash-next-reap`** has the same
code and keep files (`extract.py`, `prune_checkpoint.py`, `analyze.py`,
`keep_<N>.json`, this README):

```bash
python /kaggle/input/datasets/lamdang/flash-next-reap/extract.py --experts 384 \
    --source /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1 \
    --out /tmp/flash-next-reap-384
```

## Serve

The server command of run A (256 experts; server sized for 28 requests,
the harness ran 20 streams), as dfranzen's launcher built it; the draft view is his MTP drafter
(`dfranzen/albucino-qwen3-8-flash-next-drafter`) linked by the launcher:

```bash
sglang serve \
    --model-path /tmp/flash-next-pruned-256-calib \
    --load-format safetensors \
    --model-loader-extra-config '{"enable_multithread_load":false}' \
    --served-model-name flashnext \
    --host 127.0.0.1 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization auto-round \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.93 \
    --context-length 139264 \
    --page-size 64 \
    --max-running-requests 28 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 16384 \
    --cuda-graph-max-bs-decode 28 \
    --cuda-graph-bs-decode 1 2 4 7 8 9 10 12 14 16 18 20 22 24 26 28 \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 168 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --mamba-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-prefill-backend flashinfer \
    --moe-runner-backend auto \
    --ple-offload-embedding \
    --trust-remote-code \
    --mm-feature-transport cpu \
    --image-processor-backend pil \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"preserve_thinking":true}' \
    --watchdog-timeout 1800 \
    --schedule-policy lpm \
    --warmups structured_output \
    --enable-cache-report \
    --enable-metrics \
    --enable-request-time-stats-logging \
    --weight-loader-prefetch-checkpoints \
    --weight-loader-drop-cache-after-load \
    --chat-template /tmp/flash-next-pruned-256-calib/chat_template.jinja \
    --gdn-mtp-cache-mode none \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-path /tmp/sgl-intel/draft-view-72201e1b66f4cf7a \
    --speculative-draft-model-quantization compressed-tensors \
    --speculative-moe-runner-backend auto \
    --speculative-draft-kv-cache-dtype fp8_e4m3 \
    --speculative-accept-threshold-single 1.0 \
    --speculative-accept-threshold-acc 1.0 \
    --speculative-token-map /kaggle/input/datasets/dfranzen/pennyroyal-v253/hot_tokens_64k.pt
```

For another expert count change the model and chat-template paths and the
request limits given in that model's README.

## Tests

`exp/reap-flash-next/tests/test_prune_checkpoint.py` builds a tiny
checkpoint, prunes it and checks that the pruned model computes what the
full model computes with the same experts masked out of the router, and
that `extract.py` writes the same files as `prune_checkpoint.py`.
