# Same-context Sol versus generated-thinking NLL comparison

**Use the generated-thinking pipeline as the reference and accept 256 experts.**
The fixed 30 requests have processor-identical contexts/images and identical
Python-code targets. All 120 evaluations were collected and verified, including
72,932 per-token losses. Scoring is teacher-forced, with full contexts and losses
on the final reply only. No training or gameplay ran.

The user explicitly requested thinking-to-thinking and code-to-code comparison,
so the reference and pruning gates use those category NLLs independently.
Each category pools token losses with the original population sampling weights.

| Panel | Experts | Thinking NLL | Python-code NLL | Overall primary (diagnostic) |
| --- | ---: | ---: | ---: | ---: |
| Original Sol | 512 | 1.990651 | 0.353068 | 0.858560 |
| Original Sol | 256 | 1.992926 | 0.349434 | 0.853351 |
| Generated thinking | 512 | **1.386608** | **0.315445** | 1.043429 |
| Generated thinking | 256 | **1.380128** | **0.321199** | 1.034266 |

At 512 experts, generated thinking reduces thinking NLL by about 30.3% and
code NLL by about 10.7%. It improves thinking on 29/30 requests and code on
21/30 requests. At 256 it beats the original panel's thinking on 30/30 and
code on 16/30 requests. See `paired-requests.csv` for every individual delta.

Within the reference panel, pruning 512 → 256 changes thinking by **-0.4673%**
and code by **+1.8241%**. Both are within the inclusive +5% limit. No intermediate
counts are needed or evaluated. This is a small fixed-panel NLL decision;
training feasibility and game scores remain unmeasured.

The aggregate primary NLL is higher on generated thinking because the target
mixture changed: 15,193 thinking tokens rather than 2,943, with the same 8,295
Python-code tokens. It is reported but is not the between-panel decision metric.
Thinking text is different, so its per-request mean losses compare different
strings/lengths. Python tokens are identical and receive exact aligned loss
diffs; changed reasoning still changes their conditioning context.

## Verification and reproducibility

- Reprocessed all 30 source/variant pairs: prompt token IDs and expanded image
  tensors match; only final thinking changes. Code-only tool contracts passed.
- Validated frozen manifests, maps, model/software/kernel/numerical settings,
  all token-loss checksums, lengths, positions and sums. Both GPU smoke deltas
  are zero under the unchanged 0.01-nat tolerance; longest-context checks passed.
- 52 NLL tests passed, including missing-result, changed-context/code/map and
  independent category-gate regression checks. The archive utility was also
  exercised on the complete original run.
- Generated-thinking results, exact scoring code, dependency locks, notebook,
  analysis code and all 16,590 aligned Python-token pairs are in
  [the private DVC archive](../../../../data/sol-nll-fold0-30-genthink-results-20261008/README.md).
  [The first-panel archive](../../../../data/sol-nll-fold0-30-results-20261008/README.md)
  preserves its own full token-loss arrays and runtime.

`summary.json` contains scalar aggregate/category scores and the final decision.
`decision.json` records the two independent +5% gates. `slices.json` contains
per-game and input-context-length comparisons. `context-verification.json`
records the repeated processor audit. Reproduce the comparison with
`nll/compare_panels.py --reference-metric categorywise`; see the result-dataset
README for complete fetch/extract commands. Exact runtime code stays separate
from analysis code so strict scoring resume identity is preserved.

Kaggle shutdown is recorded in `shutdown.json` after DVC remote round-trip
verification and the Git push complete. No authenticated server URL is recorded.
