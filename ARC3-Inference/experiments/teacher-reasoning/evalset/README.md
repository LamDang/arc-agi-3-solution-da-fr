# Evaluation set for generated thinking (qwen3.8-max, five games)

`max5.json` (written by `think_gen.evalset`) lists 100 requests of
`runs/base-max-dfranzen`, the qwen3.8-max run on ft09, lp85, ls20, sp80 and
vc33, whose real thinking is known.

How the requests were picked, per game:
- only requests with real reasoning and a tool call;
- sorted by reasoning length and cut into 5 bins of equal count;
- from each bin, 4 requests evenly spaced in game position;
- 2 of those 4 go to `dev` and 2 to `eval`, by a seeded shuffle (seed 0).

That gives 10 dev and 10 eval requests per game, 50 of each overall.

- **dev** is for error analysis and for building the judge.
- **eval** stays untouched until the judge is fixed. It then measures the
  judge, and after that the generator.

Generated thinking for the set is made with `think_gen.generate --manifest`.
Each request is generated on its own, with the teacher's real thinking in
the history, so errors from earlier generated turns don't carry over.
`think_gen.review` writes one readable bundle per request for the error
analysis.

[annotation-brief.md](annotation-brief.md) is the brief the dev
annotations were made with. Each game's 10 dev requests went to one
reviewer agent, which read the bundles and the full context text.
