# Flash-Next REAP, 320 of 512 experts

dfranzen's Qwen3.8-Flash-Next Intel W4A16 AutoRound checkpoint with the
routed experts of every MoE layer cut from 512 to **320**. Nothing is
retrained: the kept experts' int4 weights are copied byte for byte, each
router keeps only the rows of its kept experts (softmax, top-10 and
renormalization then run over those 320), and `text_config.num_experts`
becomes 320. Attention, the shared expert, the PLE n-gram table, the vision
tower and the MTP tensors are unchanged.

This directory holds the recipe, not the weights: `keep.json` lists the
kept expert ids per layer, and `../extract.py` builds the checkpoint from
the original model in about a minute.

## How it was made

1. **Calibration data**: dfranzen's v3 Kaggle run (full model, his harness,
   25 public games x 4 passes). The first stretch (up to the 131K context)
   of pass 0 of each of the 25 games, replayed through the full model:
   2.93M tokens, 1.60M of them generated, 8K-token chunks, RTX PRO 6000,
   2026-10-06.
2. **Statistics** (`exp/reap-flash-next/run_reap.py`): for every layer and
   expert, the sum over tokens of router weight times output norm,
   `g * ||f(x)||` (`gate_norm`), over all tokens (context, generated,
   image). Saved under DVC at
   `exp/reap-flash-next/results/kaggle-20261006/calib` and in the private
   Kaggle dataset `lamdang/reap-flash-next-stats`.
3. **Selection** (`prune_checkpoint.choose`): the top 320 experts of every
   layer by that score, kept in their original order and renumbered
   0..319. The sets are nested: 256 in 320 in 384 in 448.
4. **Checkpoint** (`prune_checkpoint.prune`, via `../extract.py`): shards
   holding routed experts or routers are rewritten; every other file is
   linked or copied.

## Expected quality

| measure | value |
|---|---|
| router weight kept (`gate`, calibration games; held-out games agree within 0.001) | 0.949 |
| next-token NLL increase over the full model, 3 held-out games (ls20, sb26, vc33) | +0.035 |
| agreement with the full model's next token, same games | 0.924 (two full-model replays agree 0.96-1.00) |

The NLL and agreement rows were measured with `gate_norm` sets from an
earlier, smaller calibration (10 runs, 9 games), not with this exact set.

Not played yet. Run A showed that a small NLL cost can still lose many game points (256 experts: +0.067 NLL, -23 points), so check in games before relying on it.

## Size

About 48.1 GB of rewritten shards are written (14 of the 17 weight
shards); the other files are symlinks to the source, or copies with
`--copy`. SGLang loads about 47.7 GB of weights (the full model:
69.9 GB), plus 3.8 GB for dfranzen's separate MTP drafter.

## Use

Build it from the original checkpoint (only numpy is needed):

```bash
# from this repository
python models/extract.py --experts 320 --source /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1 --out /tmp/flash-next-reap-320
# on Kaggle, with the private dataset lamdang/flash-next-reap attached
python /kaggle/input/datasets/lamdang/flash-next-reap/extract.py --experts 320 \
    --source /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1 --out /tmp/flash-next-reap-320
```

Add `--copy` for a self-contained directory (to move it to another disk or
machine). The output holds `keep.json` with the source and the selection.

Serve it like the full model, with dfranzen's launcher and drafter, the
model path pointed at the output directory, and the request limits raised
to use the freed memory. Suggested start (estimate):

```
--model-path /tmp/flash-next-reap-320 --chat-template /tmp/flash-next-reap-320/chat_template.jinja
--mem-fraction-static 0.91 --max-running-requests 20 --max-mamba-cache-size 120
--cuda-graph-max-bs-decode 20
```

with `ARC3_MAX_ACTIVE_STREAMS=20` in the harness and the
`--cuda-graph-bs-decode` list cut at 20. Every other flag
stays as dfranzen sets it (`../README.md` has run A's full command line;
`exp/reap-flash-next/kaggle/session_scripts.py` builds such a server).
Before a submission, run the worst-case batch test (`serve_bench.py`) at
that stream count: at 28 streams the 256-expert server ran out of
activation memory in prefill.

## License

Derived from Qwen3.8-Flash-Next under the Qwen Community License 1.0; the
license and copyright notice of the source checkpoint apply.
