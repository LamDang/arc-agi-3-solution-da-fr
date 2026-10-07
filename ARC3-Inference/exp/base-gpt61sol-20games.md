# The base agent with gpt-6.1-sol on the 20 other official games

## Question

gpt-6.1-sol won all five of the base agent's games ([base-gpt61sol-5games.md](base-gpt61sol-5games.md)),
but those five have been public since the ARC-AGI-3 preview and the model may have seen
them. Does it do as well on the 20 other official games, with the same harness and
settings?

## Setup

| | |
| --- | --- |
| run | `runs/base-gpt61sol-20games` (DVC) |
| code | `d59dbbb` (the overload retry below) |
| launched with | `scripts/dvc_eval.py --make CONFIG_PATH=configs/inference.openai.json --make MODEL=gpt-6.1-sol --make GAME=<the 20> --make CONCURRENT_JOBS=10 --env LOCAL_ANALYZER_MAX_OUTPUT=0 --env OPENAI_REASONING_EFFORT=xhigh --env ARC3_SEND_REASONING_DETAILS=1 --env ARC3_OPENAI_PRICING=2,0.1,2.5,10` |
| started | 2026-10-07 12:27 UTC, finished 16:41 |
| games, limits | ar25 bp35 cd82 cn04 dc22 g50t ka59 lf52 m0r0 r11l re86 s5i5 sb26 sc25 sk48 su15 tn36 tr87 tu93 wa30; 1 pass, 10 at a time, 500K output tokens and 240 minutes per game |

The settings are the 5-game run's: `params.yaml` (dfranzen's) through OpenAI's Responses
API, effort `xhigh`, no output cap per request, the encrypted reasoning sent back, no
game-code access. The only difference is 10 games at a time instead of 5, because the
machine has 4 CPUs.

**Overload retry.** The first launch met `server_is_overloaded` errors, which OpenAI sends
as an error event inside a 200 stream. The harness's HTTP retry never saw them, so each one
rolled the turn back, and ten in a row would have ended the game. That launch was stopped
after 3 minutes. `d59dbbb` retries these errors in place, and the run was relaunched. It
retried 38 overloads, with no failed turn.

## Results

| game | score | levels | actions per level (human) | output tokens (reasoning) | prompt tokens (cached) | cost | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar25 | **100.0** | 8/8 won | 17 13 41 22 28 54 37 47 (32 50 75 37 89 159 233 73) | 11.4K (5.4K) | 0.82M (94%) | $0.32 | 12 |
| bp35 | **100.0** | 9/9 won | 15 **68** **59** 23 31 59 53 108 89 (21 48 44 38 33 87 86 131 163) | 155.1K (116.5K) | 17.76M (95%) | $5.65 | 159 |
| cd82 | **100.0** | 6/6 won | 25 6 17 14 13 16 (55 8 41 21 23 23) | 8.7K (4.7K) | 0.94M (94%) | $0.32 | 24 |
| cn04 | **100.0** | 6/6 won | 14 30 24 34 48 43 (29 54 85 300 208 113) | 17.1K (6.9K) | 2.13M (96%) | $0.57 | 15 |
| dc22 | **100.0** | 6/6 won | 25 42 45 70 136 170 (59 102 67 98 324 578) | 53.4K (35.9K) | 12.25M (95%) | $3.10 | 116 |
| g50t | **100.0** | 7/7 won | 25 32 64 32 50 52 56 (78 175 179 230 96 54 67) | 25.8K (14.2K) | 4.40M (95%) | $1.18 | 63 |
| ka59 | **100.0** | 7/7 won | 25 39 38 44 20 52 107 (28 109 51 51 33 132 326) | 23.3K (17.6K) | 3.51M (95%) | $1.03 | 32 |
| lf52 | **100.0** | 10/10 won | 9 53 46 52 89 111 168 68 121 61 (32 81 60 71 205 148 244 109 164 225) | 137.0K (103.1K) | 23.86M (95%) | $6.43 | 200 |
| m0r0 | **100.0** | 6/6 won | 15 38 61 11 51 53 (30 111 203 26 500 237) | 17.7K (10.7K) | 1.61M (95%) | $0.53 | 17 |
| r11l | **100.0** | 6/6 won | 3 12 11 13 22 17 (22 33 51 26 52 49) | 21.9K (14.3K) | 1.72M (96%) | $0.56 | 15 |
| re86 | **100.0** | 8/8 won | 20 36 47 46 90 65 109 202 (26 42 86 108 189 139 424 241) | 35.0K (23.6K) | 3.52M (97%) | $0.98 | 101 |
| s5i5 | **100.0** | 8/8 won | 13 26 38 30 28 25 55 36 (20 89 106 54 162 38 86 83) | 37.1K (23.5K) | 3.47M (94%) | $1.20 | 72 |
| sb26 | **100.0** | 8/8 won | 9 15 15 15 17 19 17 17 (18 28 18 19 31 23 58 18) | 4.8K (1.6K) | 0.57M (92%) | $0.22 | 43 |
| sc25 | **100.0** | 6/6 won | 22 5 12 32 39 35 (36 6 32 83 143 50) | 7.8K (5.3K) | 1.17M (94%) | $0.37 | 25 |
| sk48 | 77.3 | 7/8, time limit | 20 34 62 30 **437** 125 53, then 39 on level 8 (61 177 101 103 230 181 125 92) | 278.6K (232.6K) | 20.36M (94%) | $7.72 | 254 |
| su15 | **100.0** | 9/9 won | 11 13 14 10 17 16 5 35 15 (22 42 26 115 36 31 8 40 41) | 53.1K (46.4K) | 3.20M (94%) | $1.34 | 71 |
| tn36 | **100.0** | 7/7 won | 10 11 12 14 19 32 47 (32 72 26 40 30 55 62) | 14.9K (8.6K) | 1.36M (95%) | $0.46 | 13 |
| tr87 | **100.0** | 6/6 won | 25 32 28 21 31 24 (54 58 40 45 71 146) | 17.7K (7.9K) | 1.44M (95%) | $0.49 | 65 |
| tu93 | **100.0** | 9/9 won | 18 10 24 18 29 28 14 21 30 (19 16 34 42 123 80 14 23 111) | 33.8K (18.2K) | 3.19M (97%) | $0.92 | 26 |
| wa30 | **100.0** | 9/9 won | 27 70 83 60 104 51 49 133 70 (71 119 183 98 368 68 79 442 415) | 124.8K (99.1K) | 19.36M (95%) | $5.29 | 163 |
| **total** | **mean 98.9** | 148/149 | 6,473 actions | 1.08M (0.80M) | 126.6M (95%) | **$38.67** | 4 h 14 |

Score 100 in bold; actions above the human baseline in bold. Actions per level are
`benchmark.json`'s; tokens are summed from the `usage` of the response records; cost is
computed at $2 / $0.10 / $2.50 / $10 per million uncached input, cached input, cache-write
and output tokens (OpenAI returns no cost).

- 19 of the 20 games were won. Only three levels took more actions than the human
  baseline: bp35's levels 2 and 3 (68 and 59 against 48 and 44), which still leave bp35 at
  100, and sk48's level 5.
- sk48 is the one loss. Level 5 took 437 actions against 230. The game
  reached level 8 and was stopped there by the 240-minute limit, at 279K output
  tokens, under the 500K cap. It scores 77.3.
- Four games (sk48, bp35, lf52, wa30) used 64% of the output tokens and $25 of the $38.67,
  and took 2.5 to 4 hours. Their output is 75-83% reasoning, against 32-68% for most of the
  quick wins.
- Input is most of the cost: 126.6M prompt tokens, 95% cached, about $28 of the $38.67.
  Long games re-send up to 120K tokens of history per request.

## All 25 official games

| | games won | levels | mean score | actions | output tokens | cost |
| --- | --- | --- | --- | --- | --- | --- |
| the five games of the qwen runs (`base-gpt61sol-dfranzen`) | 5/5 | 34/34 | 100.0 | 932 | 99K | $3.00 |
| the 20 other official games (`base-gpt61sol-20games`) | 19/20 | 148/149 | 98.9 | 6,473 | 1.08M | $38.67 |
| **all 25** | **24/25** | **182/183** | **99.1** | **7,405** | **1.18M** | **$41.67** |

The five public games were not easier for it: on the 20 others it scored 98.9 against
100.0, and the one loss came on a game outside the five. This pass does not show the
public games helping.

## Transcripts

[gpt-6.1-sol Base Agent Transcripts](https://claude.ai/artifact/7jMLaPrRyRDfxdvZAxYHK6)
has all 25 games, one card per analysis step. Rebuild it from the unpacked runs:

```bash
uv run --no-sync python scripts/pack_run.py unpack runs/base-gpt61sol-dfranzen
uv run --no-sync python scripts/pack_run.py unpack runs/base-gpt61sol-20games
uv run --no-sync python scripts/base_transcripts/build.py \
  "runs/base-gpt61sol-dfranzen=The five games of the earlier runs" \
  "runs/base-gpt61sol-20games=The 20 other official games" <out.html> \
  --compare-json exp/base-gpt61sol-compare.json --title "gpt-6.1-sol Base Agent Transcripts"
```

## Caveats

- One pass. A second pass would show how much of 99.1 holds.
- Reasoning summaries: with `auto`, only 20% of the responses that reasoned had one.
  `21dddfa` makes `detailed` the default for later runs (4 of 8 replayed requests got a
  summary, against 2 with `auto`). This run's token counts and cost do not depend on it.
- The request settings differ from the qwen runs' (no sampling settings, no output cap,
  effort `xhigh`, reasoning sent back); see [base-gpt61sol-5games.md](base-gpt61sol-5games.md#setup).

## Conclusions

- gpt-6.1-sol with the base agent and dfranzen's settings won 24 of the 25 official games,
  182 of 183 levels, mean 99.1, for $41.67 in about 4.5 hours of wall clock.
- It does as well on the 20 games outside the qwen runs' five as on those five, so the
  5-game result was not carried by games the model may have seen.
- The cost is concentrated: four long games took most of the tokens and money, and the one
  loss (sk48) ran out of time, not tokens.
