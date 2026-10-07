# gpt-6.1-sol on the 25 official games with notes, hints and stated reasoning (plan)

Status: **running** since 2026-10-07 20:58 UTC (`experiments/gpt61sol-features/run.sh`).
Smoke test: ar25 alone, note threshold 20000, won 8/8 in 7.8 minutes for $0.40; one note at step 12
(history 41K → 10K tokens), reasoning fields on all 27 calls; scoring and packing ran.

## Question

gpt-6.1-sol with the base agent and dfranzen's settings won 24 of the 25 official games,
182/183 levels, mean 99.1, for $41.67 ([base-gpt61sol-20games.md](base-gpt61sol-20games.md),
[base-gpt61sol-5games.md](base-gpt61sol-5games.md)). The one loss, sk48, spent 437 actions
and 100 minutes stuck on level 5, burned ~130 actions on purpose to force a reset, and ran out
of time on level 8. Four harness changes have been tested since, each in isolation and on
replayed requests or three games:

| change | setting | evidence so far |
| --- | --- | --- |
| stated reasoning in every tool call | `ARC3_PYTHON_RATIONALE=1` | 30 replayed requests: fields filled 30/30 (strict schema), coherent with the code, ~370-character reasoning ([python-rationale-replay.md](python-rationale-replay.md)) |
| handover note before the history cut | `ARC3_NOTE_COMPACTION_TOKENS=120000` | 3-game replay: 8 correct notes, no re-exploring after them, lf52/bp35 ahead per turn ([note-compaction.md](note-compaction.md)) |
| no deliberate budget burn | `ARC3_NO_BUDGET_BURN=1` | not run yet ([stuck-detection.md](stuck-detection.md#caveats)) |
| step-back note when positions repeat | `ARC3_REPEAT_HINT=1` | 9 replayed stuck moments: more stock-taking and new ideas, fewer burns; little change on gpt-6.1-sol ([stuck-detection.md](stuck-detection.md#in-the-harness-arc3_repeat_hint)) |

With all four on, over all 25 games: does the agent still win them, does it get through
sk48-like stuck levels faster, and what do the notes, hints and reasoning fields look like
over whole games?

## Setup

Everything is the baseline runs' settings (`params.yaml`, dfranzen's, through OpenAI's
Responses API, effort `xhigh`, no output cap per request, encrypted reasoning sent back, no
game-code access), plus the four changes. The only other difference: the reasoning summary is
`detailed` (the default since `21dddfa`; the baseline ran with `auto`).

| | baseline (`base-gpt61sol-dfranzen` + `base-gpt61sol-20games`) | this run |
| --- | --- | --- |
| run | two runs, 5 + 20 games | `runs/gpt61sol-features-25games`, one run |
| code | `6bd8eef` / `d59dbbb` | current `main` (`1e05a1d` or later) |
| games, passes | 25 official, 1 pass | same |
| at once | 5, then 10 | 10 |
| limits per game | 500K output tokens, 240 minutes | **300K** output tokens, 240 minutes |
| reruns | - | each game under 100 played again, up to twice; every attempt kept (`-retry1`, `-retry2`) |
| `OPENAI_REASONING_EFFORT` | `xhigh` | `xhigh` |
| `OPENAI_REASONING_SUMMARY` | `auto` | `detailed` |
| `LOCAL_ANALYZER_MAX_OUTPUT` | `0` (no cap) | `0` |
| `ARC3_SEND_REASONING_DETAILS` | `1` | `1` |
| `ARC3_PYTHON_RATIONALE` | off | **`1`**: `python(description, reasoning, code)`, strict schema, "detailed, step-by-step reasoning" wording |
| `ARC3_NOTE_COMPACTION_TOKENS` | `0` (off; the trimmer drops the oldest half at ~130K estimated, with no summary) | **`120000`** (fires at ~112-120K real tokens, a few turns before the trimmer) |
| `ARC3_NOTE_COMPACTION_KEEP_TURNS` | - | **`10`** (default; the 3-game replay used 20) |
| `ARC3_NO_BUDGET_BURN` | off | **`1`** (system-prompt line; active because `EXPOSE_RESET` is off) |
| `ARC3_REPEAT_HINT` | off | **`1`**, with the defaults: 3 positions, 3 visits each, 10 turns between messages |
| `LOCAL_ANALYZER_TOOL_TIMEOUT` | `30` | `30` |

What the model is told, in addition to the baseline prompt:

- System prompt, tool line: "call it with `description` (what the code does), `reasoning`
  (your detailed, step-by-step reasoning for the call: ...) and one ephemeral `code` string."
- System prompt: "Do not spend actions on purpose to run out the step budget or lose the attempt
  just to get the level reset, even when you think you need a fresh start to complete the level.
  Use those actions to explore new ideas or to test your understanding of the game's mechanics
  instead."
- Once the prompt reaches 120K (estimated): `NOTE_COMPACTION_PROMPT`
  (`inference/agent/prompts.py`), asking for a comments-only `python` note under RULES, TRIED,
  LEVEL, FUNCTIONS, PLAN, LESSONS; then the history is cut to the last 10 turns plus the note.
- When 3 positions of the level have each been reached 3 times: the step-back message
  (`_repeat_hint_text`, `inference/agent/tool_agent.py`) in the turn prompt, at most every 10
  turns.

## Steps

1. **Smoke test (~$1, ~15 minutes).** The note request and the strict rationale schema have
   not run together: under `strict: true` the note reply must also fill `description` and
   `reasoning`, and the note is read from `code`. Run ar25 alone with
   `ARC3_NOTE_COMPACTION_TOKENS=20000` and the other three settings, stop it after the second
   note, and check in the transcript and request log: the note is a `python` call with all
   three fields, its code is comments only, the history is cut after it, the next request is
   accepted, and the tool line and the budget-burn line are in the system prompt. Not archived.
   Plus `uv run --no-sync pytest tests/test_note_compaction.py -q`.
2. **The run.** `bash experiments/gpt61sol-features/run.sh`, launched detached, checked every
   30 minutes. It runs the command below (with `MAX_GENERATED_TOKENS_PER_GAME=300000` and
   `MAX_RUNTIME_MINUTES=240`), then reruns each game under 100 into
   `runs/gpt61sol-features-25games-retry1` and, if still under 100, `-retry2`. Long games are listed
   first so they start in the first wave of 10: sk48, lf52, bp35, wa30 and dc22 took 2-4 hours in
   the baseline.

   ```bash
   uv run --no-sync python scripts/dvc_eval.py --run-dir runs/gpt61sol-features-25games \
     --metrics runs/gpt61sol-features-25games.metrics.json \
     --make CONFIG_PATH=configs/inference.openai.json --make MODEL=gpt-6.1-sol \
     --make GAME=sk48,lf52,bp35,wa30,dc22,re86,s5i5,su15,g50t,tr87,ls20,sb26,ka59,tu93,sc25,cd82,m0r0,vc33,sp80,lp85,ar25,cn04,r11l,tn36,ft09 \
     --make CONCURRENT_JOBS=10 \
     --env LOCAL_ANALYZER_MAX_OUTPUT=0 --env OPENAI_REASONING_EFFORT=xhigh \
     --env OPENAI_REASONING_SUMMARY=detailed \
     --env ARC3_SEND_REASONING_DETAILS=1 --env ARC3_OPENAI_PRICING=2,0.1,2.5,10 \
     --env ARC3_PYTHON_RATIONALE=1 \
     --env ARC3_NOTE_COMPACTION_TOKENS=120000 --env ARC3_NOTE_COMPACTION_KEEP_TURNS=10 \
     --env ARC3_NO_BUDGET_BURN=1 \
     --env ARC3_REPEAT_HINT=1 --env ARC3_REPEAT_HINT_POSITIONS=3 \
     --env ARC3_REPEAT_HINT_VISITS=3 --env ARC3_REPEAT_HINT_COOLDOWN=10
   ```

   The defaults are passed explicitly so that `eval_settings.json` records them.
   `scripts/dvc_eval.py` scores and packs the run at the end.
3. **Archive.** `dvc add` each attempt's run directory, commit, `dvc push`.
4. **Analysis**, below, then this page becomes the write-up.

## Expected cost and time

| | baseline | expected |
| --- | --- | --- |
| cost | $41.67 ($3.00 + $38.67) | $45-55 for the first attempt, plus the reruns (a game stopped at 300K costs ~$7-8) |
| wall clock | 23 min + 4 h 14 | ~4.5 h for the first attempt, up to 4 h per rerun |

- Reasoning fields: ~100-150 output tokens per `python` call, on ~2,000 calls: ~0.25M output
  tokens, ~$2.5, plus the same tokens re-sent as cached input.
- Notes: ~50-60 (the replay found 52 triggers over the 25 games), $0.03-0.05 each, ~$2.5.
  With 10 kept turns the prompt drops to ~35-45K after a note instead of the trimmer's ~60K,
  which lowers the input cost of the long games.

## Analysis

Against the baseline, game by game:

- Score, levels, actions per level against the human baseline, output tokens (reasoning and
  fields), prompt tokens, cost, minutes; `make significance` between the two `score.json`
  files (one pass each, so only large differences will show).
- **sk48 level 5** and the other labelled stuck levels (dc22 L5, bp35 L8): actions and
  minutes on the level, and whether the agent burns the budget.
- **Budget burns**: runs of game-over-by-budget or repeated moves with no new board, counted
  with `scripts/stuck_detection/`; the baseline's one clear case is sk48 L5.
- **Repeat hint**: where it fired (`replay_hint.py` counts on the new run), and on how many
  normal levels; what the next turn did after each one.
- **Notes**: how many, where, tokens, and whether any turn after a note re-tests or re-probes
  what the note listed (the `compare_runs.py` turn-by-turn comparison for the games that had
  cuts).
- **Reasoning fields**: share of calls with both fields, median length, a sample read for
  coherence with the code, and whether the transcript page reads better than with summaries
  alone.
- Rebuild the transcript page with `scripts/base_transcripts/build.py` with this run next to
  the baseline.

## Caveats

- Four changes at once: the run says whether the bundle helps, not which change does. If it
  helps on stuck levels, single-setting replays of those games would separate them.
- One pass, against a baseline at 99.1. The score can only move on a game like sk48, and sk48
  alone swings widely between runs (the note-compaction replay won it in 285 actions, before
  any note). Efficiency (actions per level, tokens, minutes) is the more sensitive measure,
  and it is also noisy.
- The summary setting changes from `auto` to `detailed`. It changes which responses carry a
  summary, not the reasoning tokens or the cost.
- 10 games at a time on 4 CPUs: 30-second `python` calls time out under load, and 13 of 16
  replayed notes said a timeout had wiped the retained functions. Kept as in the baseline for
  comparability.
