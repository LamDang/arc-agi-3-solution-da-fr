# Choosing a teacher: capability, price, speed

Follows [README.md](README.md), which lists the models whose API returns
their full reasoning. This page compares the full-reasoning candidates on
published benchmarks, price, measured speed and the estimated cost of
running them through the harness. Written 2026-10-07.

## Capability (published benchmarks)

Artificial Analysis (AA) indices are from OpenRouter's model catalogue
(`/api/v1/models`, `benchmarks.artificial_analysis`). MMMU-Pro (multimodal
academic reasoning) is from the [BenchLM leaderboard](https://benchlm.ai/benchmarks/mmmu-pro).
None of these models has a published ARC-AGI-3 score. "–" means no score
published.

| model | image input | AA intelligence | AA coding | AA agentic | MMMU-Pro |
| --- | --- | --- | --- | --- | --- |
| `qwen/qwen3.8-max-0902` | yes (+video) | 45.4 | 76.2 | **56.0** | 82.3 ("Qwen3.8 Max") |
| `moonshotai/kimi-k3` | yes (+video) | 43.6 | **76.2** | 50.0 | 81.6 |
| `z-ai/glm-5.3` | **no** | 44.8 | 74.8 | 53.1 | – |
| `qwen/qwen3.8-2.4t-a95b` | **no** | 39.9 | 71.9 | 50.1 | – |
| `deepseek/deepseek-v4-pro-0813` | **no** | 36.0 | 68.8 | 41.3 | – |
| `qwen/qwen3.8-27b` | yes | 33.7 | 68.1 | 45.8 | – |
| `xiaomi/mimo-v2.6-pro` | yes | **46.3** | – | – | – |
| `deepseek/deepseek-v4.1-flash` | yes | 39.5 | – | – | – |
| `minimax/minimax-m3` | yes | 29.2 | 58.6 | 29.5 | 78.1 |
| `thinkingmachines/inkling` | yes | 25.0 | 52.1 | 22.5 | 73.5 |
| *Gemini 3.1 Pro, for reference* | | | | | *83.9* |

- `qwen3.8-max-0902` is a dated snapshot of Qwen3.8 Max (2.4T-parameter MoE);
  `qwen3.8-max-prime` is the same model sold as a higher-throughput tier at
  double the price. The 82.3 MMMU-Pro was published for "Qwen3.8 Max".
- MiMo v2.6 Pro tops the open-weight intelligence rankings and scores 85.2%
  on vals.ai's Vibe Code Bench (12th of 109), but has no published vision
  score.

## Price and speed

Prices are from OpenRouter's endpoints API (`/api/v1/models/<id>/endpoints`),
in $ per million tokens. OpenRouter no longer reports throughput there (the
fields are null), so the speeds are output tokens per second measured in the
probe of [README.md](README.md): 1-3 non-streaming requests per model and
provider, queueing included. They are a rough guide only.

| model (provider) | input / cached / output | measured tok/s |
| --- | --- | --- |
| `qwen3.8-max-0902` (Alibaba, only provider) | 2.00 / 0.25 / 6.00 | 48-67 |
| `qwen3.8-max-prime` (Alibaba, only provider) | 4.00 / 0.50 / 12.00 | 114 |
| `kimi-k3` (Moonshot AI, Bedrock, BaseTen) | 3.00 / 0.30 / 15.00 | Moonshot 34, Fireworks 50, Bedrock 141, Parasail (fp4) 147 |
| `qwen3.8-2.4t-a95b` (Alibaba, Together) | 2.00 / 0.25 / 6.00 | Alibaba 40-49, Together 151-187, Modal 441 |
| `glm-5.3` (Z.AI, fp8) | 1.40 / 0.26 / 4.40 | Z.AI 86, Fireworks 129, Decart (fp4) 232 |
| `deepseek-v4-pro-0813` (Together) | 1.32 / 0.13 / 3.96 | 99-153 |
| `deepseek-v4-pro-0813` (DeepSeek) | 0.66 / 0.02 / 1.98 | blocked by the account's privacy setting |
| `minimax-m3` (Minimax) | 0.30 / 0.06 / 1.20 | 97-115 |
| `mimo-v2.6-pro` (Xiaomi) | 0.43 / 0.004 / 0.87 | 65 |
| `deepseek-v4.1-flash` (DeepSeek) | 0.15 / 0.003 / 0.60 | 113-227 |
| `openai/gpt-6.1-sol` (OpenAI standard) | 2.00 / 0.10 / 10.00 | – (summary only, see README) |
| `anthropic/claude-opus-5.5` | 4.00 / 0.20 / 20.00 | 40-62 (summary only) |

- Speed depends more on the provider than on the model. The fastest
  endpoints are often fp4 quantizations. Bedrock and Together were fast
  without being listed as fp4.
- A game of about 300K output tokens takes about 100 minutes of generation
  at 50 tok/s and about 35 at 150 tok/s. At Qwen3.8 Max 0902's speed, long
  games reach the 90-minute per-game limit of the default config.
- GPT-6.1 Sol has four price tiers on OpenRouter's OpenAI and Azure
  endpoints: flex 1/0.05/5, standard 2/0.10/10, Azure and Bedrock
  2.2/0.11/11, priority 4/0.20/20. Prompts over 272K tokens cost double;
  the harness never sends prompts that long.

## Cost per run

The token profile is the measured run `runs/base-max-dfranzen`
([exp/base-max-5games.md](../../exp/base-max-5games.md)): qwen3.8-max-0902 on
ft09, lp85, ls20, sp80 and vc33, one pass, with the dfranzen settings. It
spent 60.8M prompt tokens (93% served from cache) and 1.17M output tokens,
and cost $29.12. The table applies each model's prices to that profile, so
it assumes each teacher uses as many tokens as Qwen3.8 Max did.

| model | $/M input / cached / output | 5 games, 1 pass | 25 games, 1 pass |
| --- | --- | ---: | ---: |
| `qwen/qwen3.8-max-0902` | 2 / 0.25 / 6 | $30 (measured $29.12) | $148 |
| `openai/gpt-6.1-sol`, flex | 1 / 0.05 / 5 | $13 | $65 |
| `openai/gpt-6.1-sol`, standard | 2 / 0.10 / 10 | $26 | $129 |
| `openai/gpt-6.1-sol`, Azure or Bedrock | 2.2 / 0.11 / 11 | $28 | $142 |
| `openai/gpt-6.1-sol`, priority | 4 / 0.20 / 20 | $52 | $259 |
| `moonshotai/kimi-k3` | 3 / 0.30 / 15 | $47 | $236 |
| `qwen/qwen3.8-max-prime` | 4 / 0.50 / 12 | $59 | $297 |
| `z-ai/glm-5.3` | 1.4 / 0.26 / 4.4 | $26 | $129 |
| `qwen/qwen3.8-2.4t-a95b` | 2 / 0.25 / 6 | $30 | $148 |
| `deepseek/deepseek-v4-pro-0813` (Together) | 1.32 / 0.13 / 3.96 | $18 | $88 |
| `minimax/minimax-m3` | 0.3 / 0.06 / 1.2 | $6 | $30 |
| `xiaomi/mimo-v2.6-pro` | 0.43 / 0.004 / 0.87 | $3 | $15 |
| `deepseek/deepseek-v4.1-flash` | 0.15 / 0.003 / 0.6 | $2 | $8 |
| `anthropic/claude-opus-5.5` | 4 / 0.20 / 20 | $52 | $259 |

What moves the cost:

- **Stalled games, more than the price per token.** In the same run the
  three won games cost $1.16-4.27 (28-81 minutes); the two games stuck until
  the 240-minute limit cost $9.18 and $12.07.
- **The prompt cache hit rate.** Input is more than half of the bill for
  every model in the table. The first
  max run (`runs/base-max-default`, default settings, 38% cached) cost
  $59.68 for the same 5 games. Pin one provider with
  `OPENROUTER_PROVIDER_ORDER` so a game does not lose its cache to a
  provider switch.
- **Passes.** The OpenRouter config defaults to `n_passes: 5`, which
  multiplies every figure by 5. Passes exist for evaluation:
  `inference/tools/eval.py` averages a game's score over its passes and
  `make significance` uses their spread. For teacher data, one pass per
  game plus extra passes only on games the teacher wins sometimes is
  cheaper.
- **Reasoning length.** On the probe puzzle GPT-6.1 Sol used about 5x fewer
  reasoning tokens than Qwen3.8 Max (363 against about 1,800), one request
  only. If that holds in games, its cost lands below the table.

## Recommendation

- **Teacher with full reasoning:** `qwen/qwen3.8-max-0902`. It has the best
  agentic score of the full-reasoning models, ties Kimi K3 on coding, takes
  images, is in the student's model family, and costs about $150 for 25
  games at one pass. Its weak points are speed (about 50-65 tok/s) and a
  single provider.
- **Cheap pilot:** `xiaomi/mimo-v2.6-pro`, about $15 for 25 games.
- **Fast alternative:** `moonshotai/kimi-k3` on Amazon Bedrock: full
  reasoning, about 140 tok/s, but about $240 for 25 games.
- **GPT-6.1 Sol** (the current main solver) costs about $130 for 25 games at
  the standard tier, but returns only a reasoning summary. Its runs can only
  be distilled with the methods in
  [distillation-methods.md](distillation-methods.md) that do not need the
  teacher's thinking. Check OpenAI's terms on training with its outputs
  first.
