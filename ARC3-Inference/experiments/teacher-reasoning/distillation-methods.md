# Distilling a teacher whose reasoning is hidden

A literature review. Some teachers return their full reasoning
([README.md](README.md)); the strongest closed ones, including GPT-6.1 Sol,
return only a summary plus their tool calls. This page collects the methods
that distill a teacher without its reasoning, how they prompt and check the
generated reasoning, and what published comparisons found. Written
2026-10-07.

Each reference is marked with how its numbers were checked: **read** (the
paper itself, PDF or full HTML), **summary** (a web page summarizing the
paper; check before citing) or **unverified** (secondary source only).

## Methods

What is available without the teacher's thinking: its tool calls (Python
code and game actions), its reasoning summary, and the game's outcome. In
the harness's runs thinking is 80-90% of output tokens, so the summary and
the actions are most of what is left.

1. **Action-only imitation.** Fine-tune on the teacher's tool calls with
   empty thinking. Simplest. A thinking model trained this way tends to
   think less.
2. **Rationalization (STaR).** Give the student the context and the
   teacher's action (and summary) as a hint, have it write the reasoning
   leading to the action, keep traces that pass a check, and train on them
   with the hint removed. The reasoning is in the student's own style.
3. **Rejection sampling or expert iteration.** The student samples
   reasoning and actions without hints; keep samples whose action matches
   the teacher's or whose game run wins; fine-tune; repeat.
4. **RL with a likelihood reward (reasoning as a latent variable).** For
   state *c* and teacher action *a*, the student samples K traces z_k and
   each gets reward r_k = log p_student(a | c, z_k), optionally minus
   log p_student(a | c) with empty thinking. Optimize with GRPO, plus a
   supervised term on log p(a | c, z). This maximizes the likelihood of
   the teacher's actions with the reasoning summed out. A cheaper offline
   version: sample K traces, score them the same way, fine-tune on the best
   (or score-weighted), repeat.
5. **On-policy distillation with token probabilities (GKD, MiniLLM).** The
   student writes a trajectory and the teacher scores every token. Needs
   the teacher's probabilities on student-written text, so a self-hosted
   teacher with the same tokenizer; OpenRouter's chat API does not provide
   this.
6. **RL on the game score.** No teacher; the only way to exceed it. Usually
   started from one of the above.
7. **Context distillation.** Train with privileged information in the
   prompt (teacher actions, summary, game code), then train it away.

Where the teacher is queried also matters: labelling states the student
reaches (DAgger-style) instead of replaying teacher games. Methods 2-4 only
need the teacher's actions there, so a closed teacher works.

## Prompts used in the literature

**STaR** puts the correct answer into the question as a hint, with few-shot
examples in the same format, and drops the hint from the saved training
example, "as if the model had come up with the rationale without the hint":

```
Q: Where do you put your grapes just before checking out?
Answer Choices: (a) mouth (b) grocery cart (CORRECT) (c) super market ...
A: The answer should be the place where grocery items are placed before
checking out... Therefore, the answer is grocery cart (b).
```

**ActRe** asks for "Reason for the action:" given the observation and the
action taken, then places the reason before the action (ReAct order).

**WhiteCircle** (unverified) gives a separate model the preceding
conversation, the compacted thinking summary and the tool call that
followed, with the tool call as a constraint, and reconstructs the
reasoning chunk by chunk.

**Trace Inversion** uses no prompt: it trains an inverter model on
(input, summary, answer) to full-trace pairs from open reasoning models,
then applies it to a closed model's outputs.

**REER** searches instead of prompting: it edits the trace segment by
segment and keeps edits that lower the perplexity of the known good output.

Adapted to one harness turn (not tested):

```
[system] You are reconstructing the private reasoning of a game-playing agent.
[the agent's exact context for this turn: system prompt, history, current frame]

The agent's next call was:
<tool_call>{teacher tool call}</tool_call>
A short summary of its reasoning: <summary>{teacher summary}</summary>

Write the agent's step-by-step thinking before that call, in its usual
style. Use only facts visible in the context above: no future frames, and
never mention the summary or that the call was given. Explore and check
like a real solver: observe, form hypotheses, test them against the frame,
then decide. Your thinking must end in exactly that call.
```

The training example keeps only the context, the generated thinking and the
tool call.

## Checking generated reasoning

Cheapest first:

1. **Leak check.** Reject traces that mention "summary", "given", "hint",
   or facts not visible in the context, such as a later frame.
2. **Hint-removal test** (STaR, ActRe). From the context plus the trace,
   with no hint, does the student produce the teacher's call? Exact match
   for game moves; same output or same game action for Python. Graded
   version: log p(a | c, z) - log p(a | c), the likelihood gain used by
   REER and RLPR.
3. **Outcome filter** (ActRe). Keep only turns from runs that won or
   cleared levels.
4. **Faithfulness** (SCOTT). Change the trace's conclusion; the action
   should change too.
5. **Calibrate on Qwen3.8 Max.** Its real traces exist. Summarize them,
   reconstruct from the summaries, then compare the reconstructions with
   the real traces and compare students trained on each, before relying on
   reconstruction for GPT-6.1 Sol.
6. **Downstream evaluation.** Train students on answers only, summaries,
   reconstructed traces and (for Qwen) real traces; score on held-out
   games.

## Published comparisons: no reasoning vs generated reasoning

| paper | setting | no reasoning | generated reasoning | real reasoning | checked |
| --- | --- | --- | --- | --- | --- |
| Trace Inversion (Zhang, Morris, Shmatikov, 2026) | Qwen student; traces written by a trained inverter from answer + summary | MATH500 61.0 (answer) / 63.0 (+summary) | **71.8** | 79.8 | read |
| | | JEEBench 21.6 / 24.0 | **36.3** | 44.8 | |
| | | LiveCodeBench 25.4 / 25.6 | **30.9** | 33.2 | |
| STaR (Zelikman et al., NeurIPS 2022) | GPT-J 6B, CommonsenseQA; student-written traces | 60.0 (fine-tuned on answers) | **68.8**; **72.5** with rationalization | – (GPT-3 fine-tuned on answers: 73.0) | read |
| ReAct (Yao et al., ICLR 2023), agent | PaLM fine-tuned on HotpotQA with search; 3,000 successful runs from prompted PaLM-540B | actions only ≈24 (8B), ≈31.5 (62B) | thoughts + actions ≈25 (8B), ≈32.5 (62B) | – | read (bar chart) |
| SAND (Xia et al., EMNLP 2025), agent | ALFWorld + ScienceWorld; the base model writes deliberation over sampled, executed alternative actions | SFT on expert runs with short thoughts: Llama-3.1-8B 72.9, Qwen2.5-7B 69.4 | **88.9**, **84.6** after 3 rounds | – | read |
| Lu et al. 2026, "Hidden Thoughts Are Not Secret" | Qwen2.5-7B student | answer only: MATH500 25.5, AIME24 1.1 | summary: 69.3, 7.8 | 70.3, 14.4 | summary |
| WhiteCircle report | agent; reasoning rebuilt from summary + tool call | – | chunked: 40.4% vs 33.15% pass@1, 85% of the full-trace gain; one-pass: 25% | – | unverified |

What they show:

- **Reasoning tasks:** generated traces beat answers alone by 8-15 points
  and close 50-70% of the gap to real traces (Trace Inversion). A summary
  alone added little over the answer there (61.0 to 63.0), but much more in
  Lu et al. on easy problems; on hard problems summaries recover little
  (AIME24 7.8 against 14.4).
- **Agent tasks:** thinner and mixed evidence. In ReAct, thoughts added
  about 1 point over actions alone; training without actions did much
  worse. SAND's rich deliberation added 15-20 points over short thoughts,
  at 2-3x the output tokens.
- **Grounding matters.** SAND's ablation: when the base model invented
  alternative actions instead of sampling and executing them, the result
  could fall below plain SFT (Qwen2.5-7B, ALFWorld unseen: 62.7 against
  75.4). Generated reasoning not checked against the environment can hurt.
- **Online vs offline:** Xiong et al. 2025 ("A Minimalist Approach to LLM
  Reasoning") found rejection-sampling fine-tuning keeps up with GRPO early
  and falls behind later as its outputs lose diversity (not re-read here).
- **Full traces are the baseline to beat.** The DeepSeek-R1 report found
  distilling a strong teacher's full traces into Qwen-32B beat large-scale
  RL on that model (not re-read here).
- No published comparison covers ARC-style games.

## Likely ranking for this project

A judgement from the results above, not measured here:

1. Fine-tuning on Qwen3.8 Max 0902's full traces from won or level-clearing
   runs.
2. Likelihood-reward RL (method 4) for states with only a closed teacher's
   actions, or the offline score-and-select version first.
3. Rationalization with summary + tool call, chunked, with the checks above,
   for GPT-6.1 Sol runs.
4. Action-only imitation.
5. Then RL on the game score.

## References

Methods with generated reasoning:

- Zelikman et al., [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465), NeurIPS 2022. Read.
- Yang et al., [ReAct Meets ActRe](https://arxiv.org/html/2403.14589v2), 2024. Summary: ALFWorld 96% (1 round), 100% (4 rounds); WebShop 49%, 54.8% (human experts 59.6%).
- Zhang, Morris, Shmatikov, [How to Steal Reasoning Without Reasoning Traces (Trace Inversion)](https://arxiv.org/html/2603.07267v1), 2026. Read (full HTML).
- [Reverse-Engineered Reasoning for Open-Ended Generation (REER)](https://arxiv.org/pdf/2509.06160), ICLR 2026. Summary: an 8B model trained on 20K reverse-engineered traces matches GPT-4o and Claude 3.5 on HelloBench-HB-B.
- Wang et al., [SCOTT: Self-Consistent Chain-of-Thought Distillation](https://aclanthology.org/2023.acl-long.304/), ACL 2023. Summary.
- Xia et al., [SAND: Self-Taught Action Deliberation](https://aclanthology.org/2025.emnlp-main.152.pdf), EMNLP 2025. Read.
- Yao et al., [ReAct](https://arxiv.org/pdf/2210.03629), ICLR 2023. Read.
- Lu et al., [Hidden Thoughts Are Not Secret](https://arxiv.org/html/2606.00642v1), 2026. Summary. Its own method extracts hidden reasoning by prompt injection; cited only for the summary-vs-answer comparison.
- WhiteCircle reconstruction results, via [Hackread](https://hackread.com/hiding-chain-thought-doesnt-stop-distillation-attacks/). Unverified; no primary source found.

Likelihood-reward and latent-reasoning methods (cited from memory, not
re-read):

- Phan et al., Training Chain-of-Thought via Latent-Variable Inference (TRiCE), NeurIPS 2023.
- Chen et al., Language Models are Hidden Reasoners (LaTRO), 2024.
- Zhou et al., Reinforcing General Reasoning without Verifiers (VeriFree), 2025.
- Yu et al., RLPR: Extrapolating RLVR to General Domains without Verifiers, 2025.
- Zelikman et al., Quiet-STaR, 2024.
- Reinforcement Pre-Training (Microsoft), 2025.
- Cetin et al., Reinforcement Learning Teachers (Sakana), 2025.
- Snell et al., Learning by Distilling Context, 2022.

On-policy distillation and teacher queries:

- Agarwal et al., On-Policy Distillation of Language Models (GKD), ICLR 2024. From memory.
- Gu et al., MiniLLM, 2023. From memory.
- Thinking Machines, "On-Policy Distillation" (blog), 2025. From memory.
- Ye et al., [A Few Teacher Steps Go a Long Way](https://arxiv.org/pdf/2607.04574), 2026. Read (first pages): short teacher continuations at states the student reaches beat more full teacher demonstrations at matched budget, on HotpotQA, ALFWorld and Terminal-Bench-Dev.
- [QuestA: Expanding Reasoning Capacity via Question Augmentation](https://arxiv.org/abs/2507.13266), 2025: partial solutions as hints during RL. Summary: 1.5B model, AIME24 72.5%, AIME25 62.3%.

Training results cited for context (from memory, not re-read): Xiong et al.
2025, A Minimalist Approach to LLM Reasoning; DeepSeek-AI 2025,
DeepSeek-R1.

Terms: Trace Inversion and Lu et al. present reconstruction as an attack on
providers that hide reasoning. OpenAI's terms restrict using outputs to
develop competing models; check how they apply before training on
GPT-6.1 Sol outputs. Teachers with full reasoning and open licences avoid
the question.
