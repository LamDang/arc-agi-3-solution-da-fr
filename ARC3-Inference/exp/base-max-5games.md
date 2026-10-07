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
| status | finished | finished (10:39 UTC) |

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

## Results: dfranzen settings

The main result. Same settings as the flash baseline, only the model differs.

| game | score | levels | actions per level (human) | output tokens | prompt tokens (cached) | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | **100.0** | 6/6 won | 4 7 16 21 29 19 (43 12 23 28 65 37) | 77K | 1.90M (93%) | $1.16 | 28 |
| lp85 | 78.7 | 8/8 won | 9 9 19 **25** **110** **108** 12 5 (17 38 31 16 41 60 26 159) | 187K | 8.67M (94%) | $4.27 | 81 |
| ls20 | 7.8 | 3/7 | 17 **271** **162**, then 706 on level 4 unsolved (22 123 73 84) | 350K | 19.62M (92%) | $9.18 | 240 |
| sp80 | 4.8 | 1/6 | 8, then 662 on level 2 unsolved (39 58) | 431K | 26.25M (93%) | $12.07 | 240 |
| vc33 | **100.0** | 7/7 won | **18** 11 26 28 71 24 64 (7 18 44 61 131 34 152) | 122K | 4.39M (92%) | $2.44 | 50 |
| **total** | **mean 58.2** | 25/34 | 2,461 actions | 1.17M | 60.8M (93%) | **$29.12** | 4 h 02 |

Numbers above the human baseline are in bold. Tokens and cost are summed from
the `usage` of the response records in the request logs; the flash baseline
below is summed the same way (1.41M output, 47.1M prompt, $1.86). These are
lower than the 1.52M, 55.5M and $2.10 quoted for it in
[v12-5games.md](v12-5games.md), which counted the harness's own token tally
and a different prompt total. No HTTP 429 or 402 in the run.

- ft09 and vc33 were won in fewer actions and output tokens than flash: vc33 in
  242 actions and 122K tokens against 331 and 241K.
- lp85 was won, but levels 4-6 took 25, 110 and 108 actions against human 16,
  41 and 60 (flash: 17, 12, 31), which costs 21.3 points.
- ls20 reached level 4 at 131K tokens, then played 706 actions on it until the
  240-minute limit. Flash solved levels 4 and 5 and stopped at 5/7.
- sp80 stalled on level 2 for both models (662 actions for max, 67 for flash
  before it gave up at the token cap).

### Against the default settings

The dfranzen settings take max from 45.1 to 58.2 (vc33 21.6 to 100, ls20 3.6
to 7.8, sp80 0.3 to 4.8) at half the cost ($29.12 against $59.68), mostly from
the prompt cache (93% cached against 38%). lp85 went the other way (100 to
78.7), on its levels 4-6.

### Score at equal output tokens

Each game's score if it had stopped after the given number of output tokens
(cumulative per action from `benchmark.json`, levels counted once finished,
scored with the TAAF formula of `taaf/game.py`).

| output tokens per game | 25K | 50K | 100K | 150K | 200K | 300K | 500K (end) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base max, dfranzen settings | 10.8 | 18.7 | **37.1** | 47.6 | 58.2 | 58.2 | 58.2 |
| base flash (`20261004_135539`) | 6.2 | 18.5 | 32.7 | **49.8** | 58.5 | 66.9 | 66.9 |

With each game cut at the smaller of its two totals, max scores 47.6 and flash
43.6: max leads on ft09 (100 against 47.6 at 77K) and vc33 (100 against 35.7 at
122K), flash on lp85 (100 against 25.6 at 173K) and ls20 (31.0 against 7.8 at
350K). Max gets more per token up to about 100K per game, while ft09 and vc33
are being won. Flash draws level near 150K-200K as it finishes lp85 and vc33,
and pulls ahead after that on ls20 alone, where its levels 4 and 5 came at 205K
and 290K tokens.

## Comparison

| game | base max, dfranzen settings | base max, default settings | base flash, dfranzen settings (`20261004_135539`) | dfranzen notebook, 4 passes (mean) | v12 flash | v12 max |
| --- | --- | --- | --- | --- | --- | --- |
| ft09 | 100.0 | 100.0 | 100.0 | 86.9 | 100.0 | 100.0 |
| lp85 | 78.7 | 100.0 | 100.0 | 100.0 | 77.8 | 100.0 |
| ls20 | 7.8 | 3.6 | 31.0 | 26.9 | 10.7 | 10.8 |
| sp80 | 4.8 | 0.3 | 3.7 | 13.1 | 47.6 | 23.0 |
| vc33 | 100.0 | 21.6 | 100.0 | 80.4 | 75.0 | 35.7 |
| **mean** | **58.2** | **45.1** | **66.9** | **61.5** | **62.2** | **53.9** |
| cost | $29.12 | $59.68 | $1.86 | - | $9.90 | $112.79 |
| output tokens | 1.17M | 1.18M | 1.41M | - | 2.84M | 2.18M |

## Conclusions

- With the same settings, max does not beat flash on these five games: 58.2
  against 66.9, at 15.6 times the cost. One pass each; the dfranzen notebook's
  four flash passes spread with an SD of 9.9 (mean 61.5), so the 8.7-point gap
  is within one pass's noise, and the two runs differ on two games (lp85, ls20).
- Max is more token-efficient early: it leads at 100K output tokens per game
  (37.1 against 32.7) and wins ft09 and vc33 with fewer tokens. It does not
  convert the extra budget on the long games, where it spends hundreds of
  actions on one level (ls20 level 4, sp80 level 2).
- The settings matter more than the model: dfranzen's settings are worth 13.1
  points on max (45.1 to 58.2) and halve the cost.
- Reasoning goes back in the context for both models (checked on this run's
  logs: a replayed request costs 105,125 prompt tokens with the earlier
  reasoning and 42,612 without), so neither run lacked its earlier thinking.

Run archived with DVC: `runs/base-max-dfranzen.dvc`; metrics in
`runs/base-max-dfranzen.metrics.json`.
