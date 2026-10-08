# engine_re experiments

This folder documents the experiments of the engine reverse-engineering agent
([engine_re/README.md](../engine_re/README.md)): an agent rebuilds an
ARC-AGI-3 game engine from a recorded run, and the score is how many recorded
steps its engine reproduces. From v5 to v9 every experiment ran on one game,
lp85. The five-game runs (pilots, v2-v4) and the v5 trials are written up in
[engine_re/RESULTS.md](../engine_re/RESULTS.md); this folder adds one page per
later series and an index of every run that played lp85. v10, the
play-and-model agent, plays live games instead of a recording; its runs have
their own index ([Play runs](#play-runs)).

| page | runs | question |
| --- | --- | --- |
| [v6c-stepwise.md](v6c-stepwise.md) | v6, v6h, v6c (qwen and GLM) | Does the harness leading from one failing step to the next make the agent test and fix? |
| [v7-effort-low-medium.md](v7-effort-low-medium.md) | v7 low, v7 medium | Does a lower reasoning effort, with Qwen's recommended sampling, cut thinking? |
| [v8-objects-kernel-prompt.md](v8-objects-kernel-prompt.md) | v8 | Do frame pieces, a lenient edit tool and a parsimony prompt speed it up? |
| [v8-condense-per-turn.md](v8-condense-per-turn.md) | v8c, v6cc | Does condensing the conversation before every request save tokens? |
| [v9-bar-tolerance.md](v9-bar-tolerance.md) | v9, v9c, v9t | Does tolerating one HUD-bar pixel unblock level 2? With the threshold condenser, with a thinking budget? |
| [v10-play.md](v10-play.md) | v10 (sp80, ls20, ft09) | Can one agent play a live game while it fits `engine.py`, and plan its moves on that engine? |
| [v11-play.md](v11-play.md) | v11 (sp80, ls20) | Does stepping the replica directly, with support measured by the harness and the base prompt's guidance, take the play agent further? |
| [v12-forks.md](v12-forks.md) | v12 forks of v11: A sp80 t180, A ls20 t100, B ls20 t148 | On 100-turn forks of v11, does a context rebuilt at every request (A), or the v12 messages and harness (B), do as well as v11 over the same turns? |
| [v12-5games.md](v12-5games.md) | v12 flash and v12 max (ft09, lp85, ls20, sp80, vc33) | Does the full v12 harness hold up over whole games on the base agent's five games, and what does qwen3.8-max add? |
| [base-max-5games.md](base-max-5games.md) | base agent with qwen3.8-max-0902: dfranzen settings 58.2 ($29.12), default settings 45.1 ($59.68); flash baseline 66.9. Max is not worth it as a teacher | Does the larger model help the base agent, with and without dfranzen's settings? |
| [base-gpt61sol-5games.md](base-gpt61sol-5games.md) | base agent with gpt-6.1-sol (OpenAI Responses API, reasoning sent back, effort xhigh), dfranzen settings: 100.0, 34/34 levels, $3.00 | How far does gpt-6.1-sol get with the base agent's harness and settings? |
| [base-gpt61sol-20games.md](base-gpt61sol-20games.md) | base agent with gpt-6.1-sol on the 20 other official games: 19/20 won, 98.9 ($38.67); all 25 games 99.1, 182/183 levels, $41.67 | Does gpt-6.1-sol do as well on the games outside the qwen runs' five? |
| [python-rationale-replay.md](python-rationale-replay.md) | python tool with `description` and `reasoning` before `code` (ARC3_PYTHON_RATIONALE), 30 replayed gpt-6.1-sol requests: filled on 30/30 with strict mode, coherent | Can the model state what each call does and why, and keep its turns coherent? |
| [gpt61sol-compaction.md](gpt61sol-compaction.md) | gpt-6.1-sol after history compaction, 43 events in the 25-game runs: 29 no visible effect, 11 recover in Python, 2 re-probes (sk48, already stuck) | Does the model lose its rules when the oldest half of the history is dropped? |
| [stuck-detection.md](stuck-detection.md) | 17 hand-labelled stuck episodes in the gpt-6.1-sol, qwen3.8-max and qwen3.8-flash base-agent runs: board revisit over 60 actions OR 2 fails on a level OR 100K tokens on a level catches 14, with 1 false level of 250; three positions each visited a third time (2-pixel border cropped) instead catches all 17, earlier than the second game over, with 7 false levels | Can the game record tell when the agent is stuck, early and without false alarms? |
| [note-compaction.md](note-compaction.md) | 16 gpt-6.1-sol trigger points replayed with a note request (ARC3_NOTE_COMPACTION_TOKENS): 16/16 comment-only notes, tried probes kept; replay of sk48, lf52, bp35 at 120K: 8 notes, all correct and acted on, lf52 +4 and bp35 +3 levels after the first note against +1 and +2 logged; $6.47 | Before the cut, can the model write the note it needs to go on, and does play go better with it? |
| [gpt61sol-features-25games.md](gpt61sol-features-25games.md) | gpt-6.1-sol on all 25 games with reasoning fields, handover notes, no-budget-burn line and step-back hints, 300K output cap: 25/25 won, 183/183 levels, 100.0, $36.50 (baseline 99.1, $41.67); same actions, 1.27x output tokens, 39% fewer prompt tokens; budget burns replaced by UNDO rewinds; hints changed little | With the four new settings on, does the agent still win every game, and get through sk48-like stuck levels? |

RESULTS.md also cites two documents kept here: [analysis/](analysis/), one
page per game on how the v2 (and, for ft09 and sp80, v4) engines differ from
the real ones, and [real_engines_brief.md](real_engines_brief.md), the real
games' mechanics.

## Conventions

- **Code in git.** Each run names the commit of the harness it ran (the
  "code commit" column). To rerun one, check that commit out (for example
  `git worktree add ../engine-re-v9 34e7779`) and run the command its page
  gives, from `ARC3-Inference/`, with `OPENROUTER_API_KEY` set and the
  recorded run `runs/20261004_135539` pulled (see below). A commit marked
  `~` is inferred from the run's start time (`config.json` `started`) and the
  keys its `config.json` has; the others are the commit the run's worktree
  was at. `23e0b5f*` (v6cc) is only on the local branch
  `engine-re-v6c-condense`, not on the shared branch.
- **Run data in DVC.** Each run directory `runs/engine-re/<run>/` is
  archived with `dvc add` (one per run). Git keeps the pointer
  `runs/engine-re/<run>.dvc`; `.gitignore` keeps the run data itself out of
  git. The data is in the DVC remote `storage` (`s3://kaggle-arc-agi-3-dvc`).
- **Get one run** (the bucket allows anonymous reads), from `ARC3-Inference/`:

  ```bash
  dvc remote modify --local storage allow_anonymous_login true   # without AWS credentials
  env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY dvc pull runs/engine-re/qwen38flash-v9-lp85.dvc
  env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY dvc pull runs/20261004_135539.dvc   # the recording
  ```

- **Archive a new run**: `dvc add runs/engine-re/<run>` (writes and stages
  the pointer), `git commit`, then `dvc push runs/engine-re/<run>.dvc`. Play
  runs follow the same convention under `runs/engine-play/<run>`, with the
  pointer `runs/engine-play/<run>.dvc`.
- **What a run directory holds**: `config.json` (the `run_experiment`
  arguments), `run.log` (the console output, where it was kept),
  `summary.md` / `summary.json` (when the run finished), and one directory
  per game with `result.json` (status, turns, minutes, token usage and cost,
  best and final test, commits, `advances`), `transcript.jsonl` (everything
  the model was sent and answered), `tests.jsonl`, `final_test.txt`,
  `engine_best.py`, `workspace/engine.py` (the final engine),
  `engine_versions/`, `images/`, `trace/` and `visible_trace/`. The v7 runs
  also keep the transcript and tests of a discarded first start
  (`*.discarded-fresh-restart.jsonl`).
- `runs/engine-re/ports.dvc` is not an agent run: it holds the reference
  engines of the five games written in the fixed interface before v4 (see
  RESULTS.md, v4), from which `tests/fixtures/make_engine_re_level_starts.py`
  builds its fixture.
- **The v2 to v5 artifacts formerly in git.** RESULTS.md's run artifacts used
  to be committed under `engine_re/results/`; they are now archived in DVC like
  the runs, one pointer each in `runs/engine-re/`. Each holds copies of files
  of a full run directory in the index below (checked identical file by file;
  the transcripts are gzipped), or a page built from one; the full run
  directory is the one to work from.

  | DVC directory (`runs/engine-re/`) | contents | taken from |
  | --- | --- | --- |
  | `results-pilot-a-no-feedback` | `config.json`, each game's `result.json` | `qwen38flash-20261004` |
  | `results-pilot-b-nudges` | `config.json`, each game's `result.json` | `qwen38flash-run20261004_135539` |
  | `results-v2-main` | `config.json`, `summary.md`/`.json`, `evaluation_best.md`; per game `engine_best.py`, `engine_final.py` (= `workspace/engine.py`), `result.json`, `tests.jsonl`, `final_test.txt`, `evaluation_best.json`, `transcript.jsonl.gz`, and `notes.md` (= `workspace/notes.md`) where the agent kept one | `qwen38flash-v2-run20261004_135539` |
  | `results-v3-python-quota` | as v2 | `qwen38flash-v3quota-run20261004_135539` |
  | `results-v4-simple` | as v2 | `qwen38flash-v4-simple-run20261004_135539` |
  | `results-v5-trials` | `lp85-trial1` to `lp85-trial4`: `result.json`, `transcript.jsonl.gz` (trial 1 also `tests.jsonl`) | `qwen38flash-v5-lp85-50turns`, `qwen38flash-v5b-lp85-50turns`, `qwen38flash-v5c-lp85-50turns`, `qwen38flash-v5d-lp85-50turns` |
  | `results-v4-transcripts` | `v4-agent-transcripts.html`, the v4 ls20, lp85 and vc33 sessions turn by turn | built by `engine_re/tools/v4_transcripts/build.py` from the v4 run |
  | `results-ls20-trace-html` | `ls20_trace.html`, where the v2 ls20 engine diverges | built by `engine_re/tools/ls20_trace/build.py` from the v2 run |

## How to read the numbers

All numbers come from each run's `result.json` (lp85) and `config.json`.

- **best / final exact**: recorded steps of lp85 (120 in all) whose final
  frame and state the engine reproduces: `best.exact` for `engine_best.py`
  (the engine with the longest passing prefix), `final.exact` for the
  engine at the end of the run. "-" when the run did not record it.
- **end**: `turn limit`, `time limit`, `error (HTTP 429)` (the provider's
  rate limit, before unlimited retries), or `stopped`: the run was stopped
  by hand or interrupted and never finished, so `result.json` still says
  `running` and its numbers are those of its last turn.
- **prompt tokens**, **cached** (`cached_tokens` / `prompt_tokens`),
  **output tokens** (`completion_tokens`, reasoning included), **reasoning
  tokens** and **cost** (`cost_usd`, as OpenRouter billed it): `usage`.
- **commits**: `commit_engine` calls (`commit_calls`); "-" before v7, when
  the tool did not exist. In the stepwise harness a series page also gives
  the **advances**: the accepted step fixes, from the `advance` records of
  `transcript.jsonl` (`turn`, and the number of steps passing after it).
- **sampling**: temperature and top_p (0.7 and 0.95 by default); top_k,
  reasoning effort and thinking budget when set.
- **lp85's levels**, by levels completed before the step: level 0 is
  steps 0-8, level 1 steps 9-17, level 2 steps 18-36, level 3 steps 37-53,
  level 4 steps 54-65, level 5 steps 66-96, level 6 steps 97-105 and level 7
  steps 106-119. The last step of a level solves it and shows the next
  level's first frame, so an engine solves level n when its passing prefix
  goes past that step.
- For scale: the play agent that made the recording
  (`runs/20261004_135539`, qwen/qwen3.8-flash) solved all 8 levels of lp85
  in 89 requests, with 6,496,936 prompt tokens (6,074,624 cached),
  172,611 output tokens (141,254 reasoning), $0.242 and 44.6 minutes
  (`benchmark.json` and `lp85-305b61c3_p0_requests.jsonl`).

## Index of the lp85 runs

The five-game runs (pilot A to v4) list their lp85 game only; their run
directories hold all five games. Pilot A replayed another recording
(`runs/trace-qwen38flash-150k`, a shorter play run); every other run replays
`runs/20261004_135539`.

| experiment | harness | code commit | model | sampling | flags | end | turns | best / final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | run dir (`runs/engine-re/`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [pilot A](../engine_re/RESULTS.md#harness-iterations) | pilot A (prompt only) | - | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300` | stopped | 59 | - / - | 50.2 | $0.158 | 3.20M | 97% | 198,829 | 188,420 | - | `qwen38flash-20261004` |
| [pilot B](../engine_re/RESULTS.md#harness-iterations) | pilot B (+ test reminder) | - | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300` | stopped | 75 | - / - | 59.7 | $0.197 | 4.01M | 96% | 237,980 | 226,383 | - | `qwen38flash-run20261004_135539` |
| [v2](../engine_re/RESULTS.md#main-results-results-v2-main) | v2 (main) | `cd76c9d` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300` | passed | 151 | 120 / 120 | 78.0 | $0.469 | 15.08M | 95% | 279,880 | 250,359 | - | `qwen38flash-v2-run20261004_135539` |
| [v3](../engine_re/RESULTS.md#harness-iterations) | v3 (python quota) | `cd76c9d` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--python-quota 30` `--max-turns 300` | time limit | 246 | 0 / 0 | 116.7 | $0.908 | 30.56M | 95% | 436,095 | 359,360 | - | `qwen38flash-v3quota-run20261004_135539` |
| [v4](../engine_re/RESULTS.md#v4-the-make_levelstep-interface-results-v4-simple) | v4 (make_level/step) | `41df359` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300` | time limit | 194 | 0 / 0 | 115.0 | $0.818 | 22.96M | 93% | 470,502 | 402,185 | - | `qwen38flash-v4-simple-run20261004_135539` |
| [v5 trial 1](../engine_re/RESULTS.md#v5-trials-on-lp85-50-turns) | v5 | `fc0f08c` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 50` | stopped | 34 | 0 / - | 12.5 | $0.076 | 1.93M | 91% | 63,258 | 57,604 | - | `qwen38flash-v5-lp85-50turns` |
| [v5 trial 2](../engine_re/RESULTS.md#v5-trials-on-lp85-50-turns) | v5 | `7db527b` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 50` | stopped | 38 | - / - | 28.7 | $0.160 | 3.05M | 88% | 136,407 | 129,532 | - | `qwen38flash-v5b-lp85-50turns` |
| [v5 trial 3](../engine_re/RESULTS.md#v5-trials-on-lp85-50-turns) | v5 | `d10e41b` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 50` | stopped | 4 | - / - | 1.7 | $0.008 | 0.05M | 50% | 8,529 | 7,890 | - | `qwen38flash-v5c-lp85-50turns` |
| [v5 trial 4](../engine_re/RESULTS.md#v5-trials-on-lp85-50-turns) | v5 | `a655c2a` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 50` | turn limit | 50 | - / 0 | 29.2 | $0.154 | 4.01M | 94% | 120,086 | 108,812 | - | `qwen38flash-v5d-lp85-50turns` |
| v5e | v5 + harness opening | `e55d603`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 50` | turn limit | 50 | 1 / 1 | 31.1 | $0.181 | 3.65M | 90% | 153,619 | 143,872 | - | `qwen38flash-v5e-lp85-50turns` |
| GLM v5e | v5 + harness opening | `e55d603`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--max-turns 50` | turn limit | 50 | 1 / 1 | 13.4 | $0.105 | 2.03M | 89% | 43,681 | 34,965 | - | `glm53flash-v5e-lp85-50turns` |
| GLM v5e z-ai | v5 + harness opening | `9dc9afa`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--providers z-ai` `--max-turns 50` | stopped | 39 | 1 / - | 39.3 | $0.232 | 2.79M | 73% | 114,407 | 108,595 | - | `glm53flash-zai-v5e-lp85-50turns` |
| [v6](v6c-stepwise.md) | v6: a conversation per breaking step | `a46cab7`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--episode-turns 20 --attempts 2` | error (HTTP 429) | 55 | 9 / 9 | 38.9 | $0.120 | 1.89M | 82% | 94,779 | 84,007 | - | `qwen38flash-v6-lp85` |
| [GLM v6](v6c-stepwise.md) | v6: a conversation per breaking step | `a46cab7`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--providers z-ai` `--episode-turns 20 --attempts 2` | stopped | 29 | 8 / - | 50.4 | $0.173 | 1.47M | 68% | 144,930 | 137,254 | - | `glm53flash-zai-v6-lp85` |
| [v6h](v6c-stepwise.md) | v6h: per step, recording so far | `d81ae0d`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--episode-turns 20 --attempts 2` | error (HTTP 429) | 10 | 1 / 1 | 20.6 | $0.035 | 0.32M | 66% | 34,266 | 31,110 | - | `qwen38flash-v6h-lp85` |
| [GLM v6h](v6c-stepwise.md) | v6h: per step, recording so far | `d81ae0d`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--providers z-ai` `--episode-turns 20 --attempts 2` | stopped | 12 | 8 / - | 20.2 | $0.072 | 0.61M | 68% | 60,123 | 56,777 | - | `glm53flash-zai-v6h-lp85` |
| [v6c](v6c-stepwise.md) | v6c: one conversation | `2f3ee95`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | - | turn limit | 100 | 20 / 20 | 59.5 | $0.407 | 9.83M | 91% | 291,064 | 259,495 | - | `qwen38flash-v6c-lp85` |
| [v6c only-step](v6c-stepwise.md) | v6c: one conversation | `2f3ee95`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--only-step` | stopped | 17 | 1 / - | 9.1 | $0.036 | 0.56M | 84% | 31,485 | 28,630 | - | `qwen38flash-v6c-onlystep-lp85` |
| [GLM v6c](v6c-stepwise.md) | v6c: one conversation | `2f3ee95`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--providers z-ai` | turn limit | 100 | 17 / 17 | 83.6 | $0.615 | 9.53M | 81% | 231,168 | 202,250 | - | `glm53flash-zai-v6c-lp85` |
| [GLM v6c only-step](v6c-stepwise.md) | v6c: one conversation | `2f3ee95`~ | z-ai/glm-5.3-flash | T 0.7, top_p 0.95 | `--providers z-ai` `--only-step` | stopped | 0 | 1 / - | 0.0 | $0.000 | 0.00M | - | 0 | 0 | - | `glm53flash-zai-v6c-onlystep-lp85` |
| [v7 low](v7-effort-low-medium.md) | v7: commit_engine, Objects reference, resume | `1b1a304` | qwen/qwen3.8-flash | T 1.0, top_p 0.95, top_k 20, effort low | - | turn limit | 100 | 16 / 16 | 67.3 | $0.464 | 8.92M | 85% | 311,437 | 266,787 | 5 | `qwen38flash-v7-low-lp85` |
| [v7 medium](v7-effort-low-medium.md) | v7: commit_engine, Objects reference, resume | `1b1a304` | qwen/qwen3.8-flash | T 1.0, top_p 0.95, top_k 20, effort medium | - | turn limit | 100 | 16 / 16 | 49.5 | $0.349 | 8.21M | 90% | 218,413 | 184,945 | 5 | `qwen38flash-v7-medium-lp85` |
| [v8](v8-objects-kernel-prompt.md) | v8: pieces, lenient edits, parsimony prompt | `8dcb70c` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | - | turn limit | 100 | 20 / 1 | 71.3 | $0.517 | 10.10M | 85% | 336,912 | 297,568 | 5 | `qwen38flash-v8-lp85` |
| [v8c](v8-condense-per-turn.md) | v8 + per-turn condenser | `6c44be1` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--condense` | turn limit | 100 | 16 / 16 | 34.4 | $0.305 | 5.34M | 79% | 148,882 | 113,544 | 6 | `qwen38flash-v8c-lp85` |
| [v6cc](v8-condense-per-turn.md) | v6c + per-turn condenser | `23e0b5f`* | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--condense` | turn limit | 100 | 20 / 20 | 72.6 | $0.516 | 6.45M | 69% | 310,684 | 280,701 | - | `qwen38flash-v6cc-lp85` |
| [v9](v9-bar-tolerance.md) | v9: HUD-bar tolerance | `34e7779` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | - | turn limit | 100 | 65 / 65 | 49.3 | $0.359 | 9.70M | 92% | 205,589 | 165,193 | 13 | `qwen38flash-v9-lp85` |
| [v9c](v9-bar-tolerance.md) | v9 + threshold condenser | `5f683d5` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--condense` | turn limit | 100 | 65 / 0 | 57.7 | $0.364 | 9.43M | 92% | 245,628 | 196,015 | 12 | `qwen38flash-v9c-lp85` |
| [v9t](v9-bar-tolerance.md) | v9 + thinking budget | `5f683d5` | qwen/qwen3.8-flash | T 0.7, top_p 0.95, thinking 10,000 | - | turn limit | 100 | 54 / 24 | 45.2 | $0.373 | 10.03M | 91% | 192,058 | 122,827 | 12 | `qwen38flash-v9t-lp85` |

Headline: with the v9 HUD-bar tolerance the agent went from 20 to 65 of 120
lp85 steps (v6c, v8: 20; v9 and v9c: 65), at lower cost than v8.

## Play runs

Runs of the play-and-model agent (`engine_re.run_play`,
[PLAY_DESIGN.md](../engine_re/PLAY_DESIGN.md)). Each one plays the real game
and builds `engine.py` from what it plays. There is one row per game, and
the run directory holds every game. All numbers come from each game's
`result.json`; the score is TAAF's formula, the same as `make score_run`.

- **best / final exact** are `best.exact` and `final.exact`, followed by the
  number of steps played. `best` is recorded at the last test that improved
  it, so it can lag the game: ft09's `engine_best.py` replays all 78 steps.
- **commits** are `commit_engine` calls. The batches sent with
  `commit_moves` are in the page.

| experiment | harness | code commit | model | sampling | flags | end | score | levels | actions | turns | best / final exact | minutes | cost | prompt tokens | cached | output tokens | reasoning tokens | commits | run dir (`runs/engine-play/`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [v10 ft09](v10-play.md) | v10: play and model | `85e34e8` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300 --max-actions 500 --batch-size 10` | won | 100.0 | 6/6 | 77 | 156 | 74 / 78 of 78 | 70.0 | $0.605 | 17.42M | 92% | 289,302 | 229,091 | 11 | `qwen38flash-v10` |
| [v10 sp80](v10-play.md) | v10: play and model | `85e34e8` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300 --max-actions 500 --batch-size 10` | turn limit | 4.01 | 2/6 | 239 | 300 | 240 / 0 of 240 | 177.3 | $1.967 | 48.26M | 86% | 627,305 | 546,947 | 21 | `qwen38flash-v10` |
| [v10 ls20](v10-play.md) | v10: play and model | `85e34e8` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300 --max-actions 500 --batch-size 10` | turn limit | 3.57 | 1/7 | 239 | 300 | 230 / 240 of 240 | 182.5 | $1.904 | 48.00M | 87% | 675,320 | 575,226 | 18 | `qwen38flash-v10` |
| [v11 sp80](v11-play.md) | v11: replica stepped directly, support, base-prompt port | `5d2bda7` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300 --max-actions 500 --batch-size 10` | turn limit | 47.6 | 4/6 | 122 | 300 | 122 / 122 of 123 | 165.9 | $1.965 | 51.35M | 88% | 612,788 | 505,629 | 19 | `qwen38flash-v11-a` |
| [v11 ls20](v11-play.md) | v11: replica stepped directly, support, base-prompt port | `5d2bda7` | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 300 --max-actions 500 --batch-size 10` | turn limit | 10.7 | 2/7 | 231 | 300 | 232 / 232 of 232 | 135.5 | $1.937 | 52.64M | 88% | 468,632 | 376,296 | 18 | `qwen38flash-v11-a` |
| [v12 A sp80](v12-forks.md) | v12 A: v11 forked at turn 180, rebuilt context | `2518e9e`~* | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 280 --max-actions 500 --batch-size 10` | turn limit | 28.6 | 3/6 | 58 | 280 (181-280 forked) | 59 / 59 of 59 | 68.6 | $0.675 | 6.41M | 46% | 222,268 | 185,199 | 3 | `v12a-sp80-t180` |
| [v12 A ls20](v12-forks.md) | v12 A: v11 forked at turn 100, rebuilt context | `2518e9e`~* | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 200 --max-actions 500 --batch-size 10` | turn limit | 10.7 | 2/7 | 112 | 200 (101-200 forked) | 112 / 112 of 113 | 38.1 | $0.556 | 5.31M | 44% | 152,444 | 124,680 | 4 | `v12a-ls20-t100` |
| [v12 B ls20](v12-forks.md) | v12 B: v11 forked at turn 148, v12 messages and harness (v11's system prompt) | `ea685ec`~* | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 248 --max-actions 500 --batch-size 10` | turn limit | 10.7 | 2/7 | 161 | 248 (149-248 forked) | 161 / 6 of 162 | 44.7 | $0.816 | 21.89M | 86% | 133,456 | 101,539 | 5 | `v12b-ls20-t148` |
| [v12 B rerun ls20](v12-forks.md) | v12 B rerun: v11 forked at turn 148, v12 messages, harness and system prompt | `aa11286`~ | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--max-turns 248 --max-actions 500 --batch-size 10` | turn limit | 10.7 | 2/7 | 237 | 248 (149-248 forked) | 228 / 238 of 238 | 36.7 | $0.849 | 23.22M | 86% | 100,178 | 73,384 | 8 | `v12b2-ls20-t148` |
| [v12 flash ft09](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 6 --batch-size 10 --jobs 3` | won | 100.0 | 6/6 | 80 | 124 | 78 / 81 of 81 | 50.2 | $0.780 | 6.66M | 36% | 208,475 | 155,828 | 10 | `qwen38flash-v12` |
| [v12 flash lp85](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 6 --batch-size 10 --jobs 3` | turn limit | 77.8 | 7/8 | 86 | 300 | 86 / 86 of 87 | 176.1 | $2.351 | 21.78M | 43% | 694,972 | 602,500 | 22 | `qwen38flash-v12` |
| [v12 flash ls20](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 6 --batch-size 10 --jobs 3` | turn limit | 10.7 | 2/7 | 155 | 300 | 146 / 156 of 156 | 130.4 | $1.905 | 19.88M | 49% | 465,409 | 373,156 | 17 | `qwen38flash-v12` |
| [v12 flash sp80](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 6 --batch-size 10 --jobs 3` | turn limit | 47.6 | 4/6 | 127 | 300 | 127 / 127 of 128 | 204.7 | $2.228 | 21.24M | 46% | 753,445 | 666,077 | 19 | `qwen38flash-v12` |
| [v12 flash vc33](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-flash | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 6 --batch-size 10 --jobs 3` | turn limit | 75.0 | 6/7 | 185 | 300 | 185 / 185 of 186 | 200.9 | $2.635 | 25.02M | 43% | 714,003 | 603,139 | 22 | `qwen38flash-v12` |
| [v12 max ft09](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-max-0902 | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 30 --batch-size 10` | won | 100.0 | 6/6 | 78 | 84 | 77 / 79 of 79 | 53.6 | $9.409 | 5.63M | 28% | 152,348 | 118,223 | 12 | `qwen38max-v12` |
| [v12 max lp85](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-max-0902 | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 30 --batch-size 10` | won | 100.0 | 8/8 | 89 | 137 | 89 / 90 of 90 | 96.9 | $16.363 | 10.11M | 31% | 258,615 | 205,995 | 24 | `qwen38max-v12` |
| [v12 max ls20](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-max-0902 | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 30 --batch-size 10` | time limit | 10.8 | 3/7 | 329 | 258 | 329 / 329 of 330 | 243.4 | $28.809 | 21.24M | 46% | 585,702 | 520,718 | 22 | `qwen38max-v12` |
| [v12 max sp80](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-max-0902 | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 30 --batch-size 10` | time limit | 23.0 | 3/6 | 147 | 277 | 138 / 148 of 148 | 242.9 | $28.190 | 19.51M | 43% | 614,312 | 530,097 | 22 | `qwen38max-v12` |
| [v12 max vc33](v12-5games.md) | v12: full harness, five games | `6e4bded`, `fc81ce0` on resume | qwen/qwen3.8-max-0902 | T 0.7, top_p 0.95 | `--context rebuilt --max-turns 300 --max-actions 500 --max-cost 30 --batch-size 10` | cost limit | 35.7 | 4/7 | 224 | 257 | 224 / 224 of 225 | 207.4 | $30.023 | 20.93M | 42% | 564,837 | 490,143 | 26 | `qwen38max-v12` |

Headline: the loop won ft09 in 77 actions, against the base agent's 100 and
a human baseline of 208. It cost 2.9 times the base agent's output tokens
and 4.7 times its cost. It passed sp80's level 1 (2/6 against 1/6), but
stopped at 1/7 on ls20 (the base agent reached 5/7). The turn limit ended
both unfinished games.

v11 headline: on the same budget, sp80 went from 2/6 to 4/6 (score 4.01 to
47.6, 239 to 122 actions) and ls20 from 1/7 to 2/7 (3.57 to 10.7), every
solved level under the human baseline. The three levels v10 never solved
(sp80 2-3, ls20 1) all came from routes searched and verified on the
replica. ls20 stopped at level 2 on a win rule that left out colour; the
base agent still leads there (5/7).

v12 forks: each row is a v11 game forked at turn T and played 100 more turns.
Score, levels, actions and exact count the whole game at its end. Minutes,
cost, tokens and commits count the fork's own turns only, summed from the
transcript after its `fork` record, because `result.json`'s tokens include the
source's turns. The commits marked `~*` are inferred from the start time and
sit on unmerged worktree branches. B's final engine is an unfinished edit from
its last turn; the committed engine passes steps 0-160. Headline: the rebuilt
context (A) kept prompts at 83K and 68K at most, against v11's 273K and 228K
over the same turns. sp80 solved level 2 25 turns earlier, in 18 actions
against 21; ls20 solved level 1 as v11 did. Cost fell only 19% and 29%, as the
cached share dropped to 44-46%. B reached the patch that recolours ls20's
legend at turn 245 and step 161 (v11: 287 and 210); neither solved level 2 in
the window. B ran on v11's system prompt, so its v12 plan rules were not
tested. The B rerun (`aa11286`~, inferred from its start time, on this branch)
sent the v12 system prompt: it stated the colour rule at turn 181, covered the
patch at turn 236 (step 226, 148 level-2 actions) and ended with the colour lock
committed and 21 presses of a 31-press finish unsent.

v12 five games ([v12-5games.md](v12-5games.md)): the merged v12 harness on the base
agent's five games. Mean score: flash 62.2 ($9.90), max 53.9 ($112.79), base agent
66.9 ($2.10). Flash beat the base agent on sp80 (47.6 against 3.74) and lost on
ls20 (10.7 against 30.97), lp85 (7/8) and vc33 (6/7), on the 300-turn limit; max
won ft09 and lp85, and the $30 or 240-minute limit stopped its other three. Every
request stayed under 128K (max 97K). Flash ran three games at a time from 21:08 UTC
(HTTP 429s), and two flash games were resumed past Alibaba's input filter
(`fc81ce0`, the request asked again without its images).
