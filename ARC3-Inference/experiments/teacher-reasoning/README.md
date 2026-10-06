# Which models return their full reasoning (distillation teachers)

To distill a teacher's reasoning into the student, the API must return the
teacher's whole chain of thought, not a summary. Tested on 2026-10-06 through
OpenRouter with [`scripts/probe_reasoning.py`](../../scripts/probe_reasoning.py):
two requests per model with reasoning on at effort medium, a grid puzzle and
a turn that must call a `python` tool. For each request, the script compares
the reasoning tokens billed with the reasoning text returned. Per-request
results are in [`probe_20261006.jsonl`](probe_20261006.jsonl); the raw
responses are not kept. Total cost was under $1.

`ratio` is the returned reasoning's characters / 3.6 over the billed reasoning
tokens. It is about 0.6 to 1.2 when the full text comes back: Qwen, GLM, Kimi
and MiMo tokenizers sit near 0.6 to 0.7 on this digit-heavy puzzle, and
DeepSeek, MiniMax and Mistral near 1. It is well under 0.4 for a summary, and
0 when the reasoning is hidden or encrypted. The verdicts below also come from
reading the text: the full reasoning reads like working ("Let me draw the
grid…", "Row 0: . . . . . ."), while summaries read like status notes
("**Mapping Grid Constraints** I'm currently processing…").

## Full reasoning returned

| model | providers tested | ratio (puzzle) | notes |
| --- | --- | --- | --- |
| `moonshotai/kimi-k3` | Moonshot AI, **Amazon Bedrock**, Fireworks, Parasail, InferenceNet | 0.67-1.01 | Same on Bedrock. $15/M output. |
| `deepseek/deepseek-v4-pro-0813` | Together, Alibaba, Relace | 0.75-1.09 | The DeepSeek endpoint is blocked for this account (see below). |
| `deepseek/deepseek-v4.1-flash` | CoreWeave | 0.86 | Cheap: $1.2/M output. |
| `qwen/qwen3.8-2.4t-a95b` | Alibaba, Together, Modal | 0.60-1.06 | Same family as the student. |
| `qwen/qwen3.8-max-prime` | Alibaba (only provider) | 0.66 | |
| `qwen/qwen3.8-27b`, `qwen/qwen3.7-plus` | Darkbloom, Reka, Alibaba | 0.65 | |
| `z-ai/glm-5.3`, `z-ai/glm-5.3-prime` | Z.AI, Fireworks, Decart; Alibaba | 0.69-0.94 | `glm-5.3` returned no reasoning on the tool turn on all three providers: it called the tool without thinking. |
| `minimax/minimax-m3` | Minimax, CoreWeave | 0.80-1.06 | |
| `mistralai/mistral-large-4-0` | Mistral | 1.09 | |
| `xiaomi/mimo-v2.6-pro` | Xiaomi, GMICloud | 0.66-0.69 | |
| `nvidia/nemotron-3-ultra-550b-a55b` | BaseTen, DeepInfra | 0.72 | |
| `meituan/longcat-2.0` | AtlasCloud | 0.72 | |
| `tencent/hy4-preview` | SiliconFlow, Tencent | 0.88 | |
| `thinkingmachines/inkling` | Together, DeepInfra | 0.83-1.10 | Tool turn rate-limited (429), not tested. |
| `stepfun/step-3.7-flash` | Novita | — | Reasoning returned, but Novita bills it all as completion (0 reasoning tokens). |
| `sakana/fugu-max` | Sakana AI | 0.75 | Marked `reasoning.summary`, but the text reads as raw working. Unclear. |

All of these are open-weight models apart from `qwen3.8-max-prime`,
`glm-5.3-prime`, `mistral-large-4-0` and `fugu-max`.

## Summary, encrypted or nothing

| model | what comes back | ratio |
| --- | --- | --- |
| `anthropic/claude-opus-5.5`, `claude-fable-5.1` | Short summary (on Anthropic, Bedrock and Claude Platform on AWS alike) | 0.30-0.36 |
| `anthropic/claude-sonnet-5.5` | Did not think on either request | — |
| `openai/gpt-6.1-sol`, `gpt-6-luna` | `reasoning.summary` + `reasoning.encrypted` | 0.23-0.70 on small budgets, summary text |
| `openai/gpt-6-astra` | Encrypted only | 0 |
| `google/gemini-3.8-flash`, `~google/gemini-pro-latest` | Summary, labelled `reasoning.text`, plus encrypted on tool turns | 0.16-0.30 |
| `x-ai/grok-4.7` | Summary + encrypted | 0.10 |
| `bytedance-seed/seed-2-1-turbo` | Summary-style text ("I'm handling…") | 0.26 |

Anthropic's API documentation says the same for Claude: thinking is returned
as a summary (`display: "summarized"`) or omitted, and "the raw chain of
thought is never exposed on any model". So no closed frontier model on
OpenRouter or Bedrock can be a full-reasoning teacher. Their summaries could
still be used as rationales, but they are not what the model actually
generated.

## Not tested

- `meta/muse-spark-1.3`: OpenRouter requires an 18+ age confirmation on the
  account (https://openrouter.ai/settings/preferences).
- DeepSeek's own endpoint: OpenRouter filtered it out because of the account's
  privacy setting ("Paid model training violation"). DeepSeek may train on
  prompts, and the account disallows that. Other providers serve the same
  weights.
- Bedrock directly: the session's AWS key is the DVC IAM user
  (`arc-agi-3-dvc`), which has no `bedrock:*` permissions. Bedrock was tested
  only through OpenRouter's Amazon Bedrock provider (Kimi K3 and Claude).
- Azure: no credentials in the session.

## Caveats

- Many third-party providers serve fp4/fp8 quantizations (see each model's
  `/api/v1/models/<id>/endpoints`). For teacher data, pin the first-party or
  unquantized provider with `OPENROUTER_PROVIDER_ORDER` (the harness then
  disables fallbacks unless `OPENROUTER_ALLOW_FALLBACKS=1`).
- One small puzzle and one tool turn per model. Whether a model keeps
  reasoning on every turn of a long agentic game is untested. `glm-5.3`
  already skips it on simple tool turns.
- Check each provider's licence and terms for training on outputs before
  generating data. Open-weight licences differ, and closed providers generally
  forbid training competing models on their outputs.
