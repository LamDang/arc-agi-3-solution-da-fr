# The base agent with gpt-6.1-sol on its five games

## Question

The base agent (the Duck harness with dfranzen's changes, `inference/agent/tool_agent.py`)
scored 66.9 on ft09, lp85, ls20, sp80 and vc33 with qwen/qwen3.8-flash
(`runs/20261004_135539`) and 58.2 with qwen3.8-max-0902
([base-max-5games.md](base-max-5games.md)), both with dfranzen's settings. How far does
OpenAI's gpt-6.1-sol get with the same harness and settings, when its encrypted
reasoning is sent back to it between requests?

## Setup

| | |
| --- | --- |
| run | `runs/base-gpt61sol-dfranzen` (DVC) |
| code | `6bd8eef` (`4204e35`: the Responses API adapter) |
| launched with | `scripts/dvc_eval.py --make CONFIG_PATH=configs/inference.openai.json --make MODEL=gpt-6.1-sol --env LOCAL_ANALYZER_MAX_OUTPUT=0 --env OPENAI_REASONING_EFFORT=xhigh --env ARC3_SEND_REASONING_DETAILS=1 --env ARC3_OPENAI_PRICING=2,0.1,2.5,10` |
| started | 2026-10-07 11:09 UTC, finished 11:32 |
| games, limits | the same five, 1 pass, 5 at once, 500K output tokens and 240 minutes per game |

Everything else is `params.yaml`: the ~40 dfranzen settings the qwen flash baseline and
`base-max-dfranzen` ran with (128K context and its 58K drain, 10x images, UNDO,
animation and diff images, functions kept for the game, 900 s timeout). Game-code
access is off (`ARC3_GAME_CODE_DIR` empty); the only tool the model saw was `python`.
Four things differ from the qwen runs:

- **API.** Requests go to `api.openai.com/v1/responses`, not OpenRouter
  (`configs/inference.openai.json`, provider `openai-responses`). OpenRouter's Azure
  route drops the encrypted reasoning sent back, and chat completions on OpenAI
  returns none. The adapter in `inference/utils/openai_compat.py` translates the
  harness's chat messages; see [LOCAL_EVAL.md](../LOCAL_EVAL.md#run-on-openai-directly).
- **Reasoning sent back.** Each reply's encrypted reasoning item goes back before the
  tool call it led to (`ARC3_SEND_REASONING_DETAILS=1`, `store: false`).
- **No sampling settings.** Temperature 0.7, top_p 0.95 and top_k 20 are not sent:
  OpenAI's reasoning models reject them.
- **No output cap per request** (`LOCAL_ANALYZER_MAX_OUTPUT=0`, dfranzen: 12,288),
  and reasoning effort `xhigh`.

A smoke test (ft09, 3 actions, $0.03) and a one-game trial meant to stop after level 1
(`runs/gpt61sol-level1`) ran first. The trial won ft09 6/6 in 75 actions in 3.5 minutes,
for $0.19, before the stop could catch it.

## Transcripts

[gpt-6.1-sol Base Agent Transcripts](https://claude.ai/artifact/7jMLaPrRyRDfxdvZAxYHK6): one card
per analysis step, with the harness prompt, the reasoning summary, the python and its output,
the actions and the board after them. Rebuild it from the unpacked run:

```bash
uv run --no-sync python scripts/pack_run.py unpack runs/base-gpt61sol-dfranzen
uv run --no-sync python scripts/base_transcripts/build.py runs/base-gpt61sol-dfranzen <out.html> \
  --compare-json exp/base-gpt61sol-compare.json --title "gpt-6.1-sol Base Agent Transcripts"
```

## Results

| game | score | levels | actions per level (human) | output tokens (reasoning) | prompt tokens (cached) | max prompt | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft09 | **100.0** | 6/6 won | 4 7 14 16 21 15 (43 12 23 28 65 37) | 7.7K (3.4K) | 0.38M (90%) | 42K | $0.20 | 4 |
| lp85 | **100.0** | 8/8 won | 9 11 16 **25** 10 23 5 11 (17 38 31 16 41 60 26 159) | 13.9K (6.7K) | 1.85M (95%) | 91K | $0.53 | 8 |
| ls20 | **100.0** | 7/7 won | 16 79 43 49 52 86 93 (22 123 73 84 96 192 186) | 33.4K (19.6K) | 3.69M (95%) | 119K | $1.17 | 23 |
| sp80 | **100.0** | 6/6 won | 7 9 17 31 25 34 (39 58 25 148 96 152) | 23.0K (14.8K) | 1.29M (95%) | 72K | $0.52 | 13 |
| vc33 | **100.0** | 7/7 won | **11** 8 23 25 68 20 49 (7 18 44 61 131 34 152) | 21.2K (14.1K) | 1.38M (93%) | 96K | $0.57 | 11 |
| **total** | **mean 100.0** | 34/34 | 932 actions | 99K (58.6K) | 8.59M (95%) | | **$3.00** | 23 |

Numbers above the human baseline are in bold. Actions per level are `benchmark.json`'s.
Tokens are summed from the `usage` of the response records in the request logs; cost is
computed from them at $2 / $0.10 / $2.50 / $10 per million uncached input, cached input,
cache-write and output tokens (OpenAI returns no cost). No HTTP errors or retries.

- Every game was won, in 4 to 23 minutes. Only two levels took more actions than the
  human baseline: vc33's level 1 (11 against 7) and lp85's level 4 (25 against 16).
  Per-level scores up to 115 on the others keep both games at 100.
- The model reasons little, at effort `xhigh`: 58.6K reasoning tokens over the five
  games, and at most a few hundred per response. Most of its output is the python it
  runs, and one python call plays several actions: 180 requests for 932 actions.
- Prompt-cache hits were 90-95%, so input cost about $2.01 of the $3.00 (output $0.99).

## Comparison

| game | gpt-6.1-sol, dfranzen settings | base max, dfranzen settings | base flash, dfranzen settings (`20261004_135539`) | dfranzen notebook, 4 passes (mean) | v12 flash | v12 max |
| --- | --- | --- | --- | --- | --- | --- |
| ft09 | 100.0 | 100.0 | 100.0 | 86.9 | 100.0 | 100.0 |
| lp85 | 100.0 | 78.7 | 100.0 | 100.0 | 77.8 | 100.0 |
| ls20 | 100.0 | 7.8 | 31.0 | 26.9 | 10.7 | 10.8 |
| sp80 | 100.0 | 4.8 | 3.7 | 13.1 | 47.6 | 23.0 |
| vc33 | 100.0 | 100.0 | 100.0 | 80.4 | 75.0 | 35.7 |
| **mean** | **100.0** | **58.2** | **66.9** | **61.5** | **62.2** | **53.9** |
| cost | $3.00 | $29.12 | $1.86 | - | $9.90 | $112.79 |
| output tokens | 99K | 1.17M | 1.41M | - | 2.84M | 2.18M |
| actions | 932 | 2,461 | - | - | - | - |

## Caveats

- **One pass.** One pass's 5-game mean had an SD of 9.9 over the dfranzen notebook's
  four passes. A perfect score leaves no room to measure that here.
- **Public games.** These five have been public since the ARC-AGI-3 preview, with
  recordings and write-ups online. A model trained after that may have seen them,
  which would help it on exactly these games. The other 20 official games in
  `environment_files/` would test that.
- **Not the same request settings as the qwen runs:** no sampling settings, no output
  cap, effort `xhigh`, and the reasoning sent back. The run measures the model and
  that API path together.

## Conclusions

- With the base agent and dfranzen's settings, gpt-6.1-sol won all five games, 34/34
  levels, in 23 minutes and $3.00: mean 100.0 against 66.9 for flash and 58.2 for max.
  The three games the qwen models stalled on for hours (ls20, sp80, vc33) took 11 to
  23 minutes.
- It used 8% of max's output tokens and 38% of its actions.
- Next: the same run on the other official games, to separate the model from what it
  may have seen of these five, and more than one pass.
