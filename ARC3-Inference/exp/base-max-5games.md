# The base agent with qwen3.8-max-0902 on its five games

## Question

The base agent (the Duck harness with dfranzen's changes, `inference/agent/tool_agent.py`)
scored 66.9 on ft09, lp85, ls20, sp80 and vc33 with qwen/qwen3.8-flash
(`runs/20261004_135539`), and 61.5 over the four passes of the dfranzen notebook
([v12-5games.md](v12-5games.md#against-the-dfranzen-notebooks-four-passes)).
On the v12 play agent, qwen3.8-max-0902 scored below flash (53.9 against 62.2).
Does the larger model help the base agent?

## Two runs

| | default settings | dfranzen settings |
| --- | --- | --- |
| run | `runs/base-max-default` | `runs/base-max-dfranzen` |
| launched with | `make interactive` (config defaults) | `scripts/dvc_eval.py --make MODEL=qwen/qwen3.8-max-0902` (`params.yaml`) |
| started | 2026-10-07 02:35 UTC | 2026-10-07 06:37 UTC |
| status | finished | running |

Both use the same code (`fb89742`), the same games, 1 pass, 5 games at once,
500K output tokens and 240 minutes per game, and HTTP 429s retried without
limit. They differ only in the harness settings.

**The dfranzen settings** are the ~40 environment variables of the dfranzen
notebook (v3, cell 4), copied in `params.yaml` without the Kaggle-only items.
The flash baseline `runs/20261004_135539` ran with them. Its system prompt
carries the text each one adds, its prompts reach 119K tokens and drain to
about half, and its images are 640 pixels. `scripts/dvc_eval.py` applies
them, so a run through it differs from the baseline only in `MODEL`.

**The default settings** are what plain `make interactive` exports: the
values in `configs/inference.openrouter.json` and the harness defaults. The
first run used them by mistake. It is kept as a measure of how much the
dfranzen settings are worth on max.

| setting | dfranzen settings (flash baseline, `base-max-dfranzen`) | default settings (`base-max-default`) |
| --- | --- | --- |
| context window / max output | 128K / 12,288 | 64K / provider default |
| history drain | at 58K tokens, 150 assistant turns, 30 at a time | none (defaults) |
| temperature / top_p / top_k | 0.7 / 0.95 / 20 | 0.6 / 0.95 / 20 |
| image upscale | 10x (640 pixels) | 16x (1,024 pixels) |
| UNDO | exposed and executable | advertised as ACTION7, refused |
| animation frames, diff image, frame_diff hint | on | off |
| functions kept between python calls | on (whole game) | off |
| level-transfer guidance, action info, gameplay_changed prompts | on | off |
| world-model memory sections | off | on |
| guards against wasted actions | from level 2 | from level 1 |
| tool output cap / yield | 3,072 tokens / every 2,048 generated tokens | 1,024 tokens / every 60 s |
| analyzer timeout | 900 s | 120 s |

System prompt: 18,619 characters with the dfranzen settings, 13,750 without.

## Results: default settings

| game | score | levels | actions per level (human) | output tokens | prompt tokens (cached) | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | **100.0** | 6/6 won | 4 11 15 21 **53** 19 (43 12 23 28 65 37) | 91K | 1.69M (54%) | $2.34 | 31 |
| lp85 | **100.0** | 8/8 won | 8 9 26 14 17 20 17 9 (17 38 31 16 41 60 26 159) | 101K | 1.68M (43%) | $2.69 | 32 |
| ls20 | 3.6 | 1/7 | 15, then 1,531 on level 1 unsolved (22 123) | 265K | 10.23M (51%) | $12.92 | 240 |
| sp80 | 0.3 | 1/6 | **164**, then 937 on level 1 unsolved (39 58) | 327K | 12.86M (31%) | $20.65 | 240 |
| vc33 | 21.6 | 4/7 | **11** 12 42 **596**, then 116 on level 4 unsolved (7 18 44 61 131) | 398K | 13.59M (33%) | $21.07 | 241 |
| **total** | **mean 45.1** | 20/34 | 3,667 actions | 1.18M (0.99M reasoning) | 40.1M (38%) | **$59.68** | 4 h 01 |

Numbers above the human baseline are in bold. Tokens and cost are summed from
the `usage` of the response records in the request logs.

- ft09 and lp85 were won in about 31 minutes each, in 123 and 120 actions
  (flash baseline: 100 and 119).
- The three other games ran to the 240-minute limit, stuck on one level each,
  and played hundreds of actions there: 1,531 on ls20's level 1, 937 on
  sp80's level 1 after 164 on level 0, and 596 on vc33's level 3. None
  reached the 500K-token cap (265-398K).

## Comparison

| game | base max, default settings | base flash, dfranzen settings (`20261004_135539`) | dfranzen notebook, 4 passes (mean) | v12 flash | v12 max |
| --- | --- | --- | --- | --- | --- |
| ft09 | 100.0 | 100.0 | 86.9 | 100.0 | 100.0 |
| lp85 | 100.0 | 100.0 | 100.0 | 77.8 | 100.0 |
| ls20 | 3.6 | 31.0 | 26.9 | 10.7 | 10.8 |
| sp80 | 0.3 | 3.7 | 13.1 | 47.6 | 23.0 |
| vc33 | 21.6 | 100.0 | 80.4 | 75.0 | 35.7 |
| **mean** | **45.1** | **66.9** | **61.5** | **62.2** | **53.9** |
| cost | $59.68 | $2.10 | - | $9.90 | $112.79 |
| output tokens | 1.18M | 1.52M | - | 2.84M | 2.18M |

Without the dfranzen settings, max scores 21.8 points below flash with them,
at 28 times the cost. The gap is on the three games that need many levels of
play: the flash baseline won vc33 and reached 5/7 on ls20, where the default
run stalled at one level for hours. Undo, the animation and diff images, the
128K context with its drain and the functions kept between calls are what
the default run lacked, so this run measures those settings as much as the
model. It does not say whether max helps the base agent; the dfranzen-settings
run below does.

## Results: dfranzen settings

Running (launched 06:37 UTC, ends by 10:37 UTC). The harness process was
checked to carry the settings: 128K context, temperature 0.7, 10x images,
UNDO on, animation, kept functions, the 58K drain and the 900 s timeout. This
section will hold its per-game results and the comparison with the flash
baseline.
