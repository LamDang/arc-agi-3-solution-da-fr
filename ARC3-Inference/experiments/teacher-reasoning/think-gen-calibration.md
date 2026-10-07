# Generated thinking against real thinking: calibration on qwen3.8-max

Follows [distillation-methods.md](distillation-methods.md) (method 2,
rationalization) and uses the pipeline in [think_gen/](../../think_gen/README.md).
Written 2026-10-07.

## Question

GPT-6.1 Sol hides its thinking. The plan is for qwen3.8-flash to write it
from what the request logs keep:

- the agent's context;
- the call that followed;
- for about a third of the requests, sol's reasoning summary.

Before generating it for sol, we need to know how close such thinking comes
to the real thing. qwen3.8-max returns its real thinking, so on a max run the
generated thinking can be compared with what the teacher actually thought.
This is check 5 of distillation-methods.md ("Calibrate on Qwen3.8 Max").

## Setup

| | |
| --- | --- |
| source run | `runs/base-max-dfranzen`: qwen3.8-max-0902, base agent, dfranzen settings |
| requests | the first 30 model responses of ft09, lp85 and vc33 (90 in all; levels 1-3) |
| generator | `qwen/qwen3.8-flash` on Alibaba through OpenRouter, its own reasoning off, temperature 0.7 |
| order | each game's requests run in order, with the thinking generated for earlier turns in the history |
| history placement | in the assistant message text, between `[thinking]` lines (`--history inline`; see "Not evaluated") |
| retries | up to 3 attempts while the leak filter or pasted-code check fails |

Max has no summaries, so the summary arms use synthetic ones. Flash writes
them from max's real thinking, in the style of four real sol summaries. They
came out at a median of 922 characters, about twice sol's length (see
[Sol's data](#what-the-sol-data-shows)).

| arm | prompt | summary | output |
| --- | --- | --- | --- |
| b1 summary | b1 | synthetic | `runs/think-calib-b1/sum` |
| b1 no summary | b1 | none ("keep it brief") | `runs/think-calib-b1/nosum` |
| b2 summary | b2 | the same synthetic ones | `runs/think-calib-b2/sum` |

- **b1** asks for the thinking in first person, as the solver would think it.
  The thinking should start from the newest input, use only what is
  visible, not mention the summary or the task, and end at the call
  without pasting the code.
- **b2** is b1 plus one instruction, added after reading b1's output: write
  the thinking as working, not as an explanation. That means listing and
  computing positions and values, saying "wait" and correcting mistakes,
  and deriving the numbers the code uses.

Both outputs are in DVC: `dvc pull runs/think-calib-b1.dvc runs/think-calib-b2.dvc`.
`runs/think-calib-b1/page.html` shows each request side by side: the newest
input, the call, the real thinking, the three generated versions and the
judge's verdicts.

## How the generated thinking is evaluated

Three measures. None needs a person to read the traces, but reading the page
is what found the style problem in b1.

1. **Automatic checks during generation** (`think_gen/checks.py`). A text
   fails if it:
   - mentions a summary, "the agent", reconstructing or a given output;
   - contains `[thinking]` or `<think>` tags;
   - contains more than half of the call's code lines (lines of 20+
     characters) verbatim.

   Failing texts are retried. The counts reported are records retried and
   records still failing after 3 attempts.
2. **Length.** Generated characters over real characters per request
   (median), and the correlation of the two lengths across requests.
3. **Judge** (`think_gen/judge.py`). qwen3.8-flash with its reasoning on,
   temperature 0.2 and JSON output. It sees the real thinking, the
   generated thinking and the output (the call), but not the game context.
   It answers five questions:
   1. list the real thinking's key points, at most 10, most important first:
      observations, hypotheses about the rules or the goal, plans,
      decisions;
   2. for each, does the generated thinking contain it, in any wording;
   3. list the generated thinking's claims that contradict the real
      thinking or the output, leaving out claims the real thinking simply
      doesn't mention;
   4. does the generated thinking lead to this exact output;
   5. does it leak that it was written afterwards?

   From the answers:
   - **coverage** = key points contained / key points listed, averaged over
     requests;
   - **contradictions** = claims listed under question 3, per request;
   - **leads to output** and **leak** = share of requests with a yes.

   An answer that can't be parsed is retried up to 3 times. Between 2 and 4
   requests per arm stayed unjudged and are left out.
4. **Paired comparison.** All arms use the same 90 requests, so arms are
   compared request by request: mean difference in coverage, its standard
   error, and how many requests each arm wins.

## Results

| arm | judged | coverage | contradictions per request | requests with one or more | leads to output | leak (judge) | length vs real (median) | length correlation | retried / still failing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| b1 summary | 89 | **0.70** | 1.90 | 75% | 82% | 1% | 0.50 | 0.27 | 11 / 0 |
| b2 summary | 86 | 0.68 | 2.13 | 83% | 84% | 2% | 0.49 | 0.16 | 11 / 2 |
| b1 no summary | 88 | 0.45 | 1.58 | 72% | 82% | 0% | 0.15 | 0.09 | 3 / 0 |

Paired, on the requests both arms have judged:

| comparison | n | coverage difference | requests won / tied / lost | contradictions difference |
| --- | ---: | --- | --- | ---: |
| summary − no summary (b1) | 87 | **+0.24** (s.e. 0.023) | 74 / 4 / 9 | +0.30 |
| b2 − b1 (with summary) | 85 | −0.01 (s.e. 0.025) | 31 / 9 / 45 | +0.27 |

- **The summary is worth a lot.** With a summary, the generated thinking
  contains 70% of the real thinking's key points; without one, 45%. It
  won on 74 of 87 requests. But the synthetic summaries are twice as long
  as sol's, so sol's real summaries will help less than this.
- **Even with a summary, most requests have a contradiction** (75%, about 2
  per request). In a sample of six requests, most were real misreadings
  of the board:
  - a cycle traced in the wrong direction;
  - three rows where the real thinking found four;
  - a button placed above a wall instead of below it;
  - more confidence than the real thinking, which kept several hypotheses
    open.

  A few were judge strictness, such as "dump all nodes" when the code skips
  three. The literature review warns about exactly this (SAND): reasoning
  that isn't grounded in the environment can make a student worse.
- **About one request in six (16-18%) doesn't lead to the output,
  according to the judge.** In the cases read, the thinking planned
  something slightly different from what the code does: another probe
  location, a different check, or a step the code doesn't perform.
- **The thinking is half as long as the real thinking (median 0.5), and its
  length barely follows the real length** (correlation 0.27). Without a
  summary it is 0.15 of the real length, as instructed. Coverage does not
  fall on long real thinking (0.68, 0.69 and 0.72 for under 2K, 2-6K and
  over 6K characters), but contradictions rise there (2.6 per request
  over 6K).
- **b2 didn't help.** Asking for working rather than explanation made the
  text read more like max's thinking, with derivations and corrections in
  it. But coverage didn't change (−0.01 ± 0.025), contradictions went up
  slightly (+0.27), and the lengths followed the real ones less. Written
  derivations add more places to get a coordinate wrong.
- **Leaks are rare.** The regex filter caught them on 11-12% of first
  attempts with a summary (mentions of "summary", "reconstruct", "the
  agent"), and retries fixed all but 2. The judge flagged 1-2% of final
  texts.

## What the sol data shows

From `runs/base-gpt61sol-dfranzen` (5 games, 180 responses). The request
logs' completion tokens add up to the run's total, 99,021, game by game.

| | requests | reasoning tokens, median | reasoning tokens, mean |
| --- | ---: | ---: | ---: |
| with a summary | 61 | 102 | 354 |
| without | 119 | 206 | 311 |

- Hidden reasoning is 58,635 tokens, **59% of sol's output**. The rest is
  tool-call code. The per-block `tokens` fields of the encrypted items add
  up to 58,611.
- **Requests without a summary are not shorter.** Whether sol returns a
  summary looks unrelated to how long it thought. The prompt's "keep it
  brief when there is no summary" rule has no basis in sol's data.
- **Sol's summaries have a fixed size.** One section of about 450
  characters, whatever the reasoning length: medians 440, 483, 483 and 449
  characters for under 100, 100-300, 300-1,000 and over 1,000 reasoning
  tokens. The correlation is 0.08. Under 100 tokens the summary is longer
  than the thinking it summarizes, so it comes from a separate writer and
  can contain words that were never in the thinking.
- Sol thinks briefly: about 330 tokens per request on average, against
  about 1,000-2,000 for max. The exact count is known for every sol
  request, so it could guide length where the summary can't.

## Limits of this evaluation

- **The judge is the generator's model** (flash), one sample per request,
  with no measure of agreement with people or with another judge. It
  doesn't see the game context, so it can't tell a true detail missing
  from the real thinking from an invented one. It was told to ignore such
  claims, so invented details that don't contradict the real thinking go
  uncounted.
- **The real thinking is not ground truth.** Max misreads boards too. A
  "contradiction" can be the generated thinking being right and max being
  wrong.
- **Early-game requests only** (the first 30 of each game, levels 1-3), and
  three of the five games. Longer contexts later in a game are untested.
- **The synthetic summaries are twice sol's length** and in a different
  shape (two sections instead of one), so the summary arms are
  optimistic for sol.
- **Coverage weights all key points equally**, and the judge decides what
  the key points are.
- **Nothing here measures the effect on a trained student.** Only a
  fine-tuning comparison (distillation-methods.md, check 6) can show that.
- **Cost:** generation $0.32 for the three arms, judging $1.03. The
  synthetic summaries' cost was not recorded.

## Changed since these runs, not evaluated

- **History placement.** Earlier generated thinking now goes in the
  history message's `reasoning` field by default (`--history native`). An
  early probe was misread as showing that OpenRouter drops it. A probe with
  a 1,000-token block shows flash does see it, and so do the
  prompt-token counts of `runs/base-max-dfranzen`. These runs used the
  older text form.
- **Synthetic summaries** now ask for sol's format: one section of about 75
  words, whatever the thinking length.

## Open choices for the next round

- **Length.** The "keep it brief" rule doesn't match sol's data. Either drop
  it, or give sol's reasoning-token count as a length guide on every
  request.
- **Grounding.** Most remaining errors are misread board details. Options:
  - let flash reason before writing (`--reasoning`);
  - give the judge the context, so it can check claims against the frames
    and tool outputs;
  - add the hint-removal test from distillation-methods.md: with the
    thinking and without the call, does flash produce the same game
    actions?
- **Rerun the calibration** with the shorter summaries and native history,
  including mid-game requests, before generating for sol.
