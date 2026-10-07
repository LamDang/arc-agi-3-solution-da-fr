# v12: the play-and-model agent on the base agent's five games, flash and max

## Question

v11 ran only sp80 and ls20, and the v12 changes were tried only as 100-turn
forks ([v12-forks.md](v12-forks.md)). Does the full v12 harness, with its
messages, plan rules and rebuilt context, hold up over whole games on all five
games the base agent played (`runs/20261004_135539`: ft09, lp85, ls20, sp80,
vc33)? And how much does a larger model (qwen3.8-max-0902) add on the same
harness?

## Setup

| | v12 flash | v12 max | base agent |
| --- | --- | --- | --- |
| run | `runs/engine-play/qwen38flash-v12` | `runs/engine-play/qwen38max-v12` | `runs/20261004_135539` |
| model | qwen/qwen3.8-flash | qwen/qwen3.8-max-0902 | qwen/qwen3.8-flash |
| harness | `engine_re.play_agent` v12, `--context rebuilt` | same | the main text-only play harness |
| code | `6e4bded` (PR #8 merged), then `866f2d6` and `fc81ce0` on resume | `6e4bded`, resumed on `866f2d6` and `fc81ce0` | - |
| limits per game | 300 turns, 240 min, 500 actions, $6 | 300 turns, 240 min, 500 actions, $30 | 500K output tokens |
| sampling | T 0.7, top_p 0.95 | same | T 0.7, top_p 0.95, top_k 20 |
| context | rebuilt each request, counted exactly with the Qwen3 tokenizer, under 128K | same | its own drain at 59K |

Prices per million tokens on OpenRouter: flash $0.15 in, $0.47 out, $0.016
cached; max $2 in, $6 out, $0.25 cached. Max costs about 13 times as much per
token.

Both runs started at 20:00 UTC on 2026-10-06 with all ten games at once.

## Results

Score is TAAF's formula: each solved level scores (human actions / agent
actions)^2, capped at 1, and level n weighs n + 1.

| game | v12 flash | v12 max | base agent |
| --- | --- | --- | --- |
| ft09 | **100.0**, 6/6, 80 actions | **100.0**, 6/6, 78 actions | **100.0**, 6/6, 100 actions |
| lp85 | 77.8, 7/8, 86 actions | **100.0**, 8/8, 89 actions | **100.0**, 8/8, 119 actions |
| ls20 | 10.7, 2/7, 155 actions | 10.8, 3/7, 329 actions | **30.97**, 5/7, 866 actions |
| sp80 | **47.6**, 4/6, 127 actions | 23.0, 3/6, 147 actions | 3.74, 1/6, 111 actions |
| vc33 | 75.0, 6/7, 185 actions | 35.7, 4/7, 224 actions | **100.0**, 7/7, 331 actions |
| **mean** | **62.2** | **53.9** | **66.9** |
| cost | $9.90 | $112.79 | $2.10 |
| output tokens | 2.84M | 2.18M | 1.52M |
| prompt tokens (cached) | 94.6M (44%) | 77.4M (40%) | 55.5M |

The base agent still leads on the mean (66.9). v12 flash comes within 4.7
points of it at 4.7 times its cost, and beats it outright on sp80 (47.6
against 3.74). v12 max scores below v12 flash (53.9 against 62.2) at 11 times
flash's cost. Its two wins, ft09 and lp85, took fewer turns than flash's, but
on the three long games it ran into the $30 cap or the 240-minute limit
before turn 300.

### How each game ended

| game | v12 flash | v12 max |
| --- | --- | --- |
| ft09 | won at turn 124 (50 min, $0.78) | won at turn 84 (54 min, $9.41) |
| lp85 | turn limit (300), 7/8 | won at turn 137 (97 min, $16.36) |
| ls20 | turn limit (300), 2/7 | time limit (243 min, turn 258), 3/7 |
| sp80 | turn limit (300), 4/6 | time limit (243 min, turn 277), 3/6 |
| vc33 | turn limit (300), 6/7 | cost limit ($30.02, turn 257), 4/7 |

Flash used all 300 turns on four games, at 0.4 to 0.7 minutes a turn. Max
averaged about a minute a turn (it writes 20 to 40 output tokens a second;
its longest turns took 5 minutes) and about $0.11 a turn, so its clock or its
budget ran out first.

### Actions per level against the human baseline

Numbers above the human baseline are in bold. A number in brackets is the
actions spent on the level the game ended on, unsolved.

| game | level | human | v12 flash | v12 max | base agent |
| --- | --- | --- | --- | --- | --- |
| ft09 | 0-5 | 43 12 23 28 65 37 | 4 7 14 21 21 13 | 5 7 14 16 23 13 | 4 7 17 24 27 21 |
| lp85 | 0-7 | 17 38 31 16 41 60 26 159 | 9 9 16 13 9 20 7 (3) | 8 8 16 13 10 22 7 5 | 8 9 19 **17** 12 31 9 14 |
| ls20 | 0-6 | 22 123 73 84 96 192 186 | **25** 63 (67) | 21 **183** **125** | **24** 98 **151** **117** **129** (347) |
| sp80 | 0-5 | 39 58 25 148 96 152 | 16 40 11 35 (25) | 17 31 **37** (62) | **44** (67) |
| vc33 | 0-6 | 7 18 44 61 131 34 152 | **12** 7 24 61 59 20 (2) | **10** 7 39 33 (135) | 7 12 24 30 **135** **37** 86 |

v12 plays most of the levels it solves well under the human baseline, as v11
did. The base agent's advantage is in reach, not efficiency: on ls20 it
solved five levels in 866 actions, four of them over the human count, and
TAAF's formula still gives those partial credit. v12 stops at the turn limit
long before it could spend that many actions; ls20 flash used 155 of its 500.

### Against v11 (sp80 and ls20, flash)

| game | v11 | v12 flash |
| --- | --- | --- |
| sp80 | 47.6, 4/6, 122 actions, $1.97, prompt max 289K | 47.6, 4/6, 127 actions, $2.23, prompt max 97K |
| ls20 | 10.7, 2/7, 231 actions, $1.94, prompt max 297K | 10.7, 2/7, 155 actions, $1.91, prompt max 97K |

Same scores, with every request under 128K, the limit of the deployed vLLM
server. v11's requests reached 297K. Cost is about the same: v12 sends
less than half the prompt tokens (21M against 51M), but its cached share is
lower (46% against 88%), because rebuilding the context changes its start at
almost every request.

### Tokens

Output and prompt tokens per game. The base agent's figures are from
`runs/20261004_135539/analyses/by_level.csv`; v12's from each `result.json`.

| game | base agent | v12 flash | v12 max |
| --- | --- | --- | --- |
| ft09 | 101K out, 3.7M in (6/6) | 208K, 6.7M (6/6) | 152K, 5.6M (6/6) |
| lp85 | 173K, 6.5M (8/8) | 695K, 21.8M (7/8) | 259K, 10.1M (8/8) |
| ls20 | 502K, 24.2M (5/7) | 465K, 19.9M (2/7) | 586K, 21.2M (3/7) |
| sp80 | 501K, 13.0M (1/6) | 753K, 21.2M (4/6) | 614K, 19.5M (3/6) |
| vc33 | 241K, 8.1M (7/7) | 714K, 25.0M (6/7) | 565K, 20.9M (4/7) |
| **total** | **1.52M out (1.35M reasoning), 55.5M in, 729 calls** | **2.84M (2.40M), 94.6M, 1,324 turns** | **2.18M (1.87M), 77.4M, 1,013 turns** |

- v12 flash used 1.9 times the base agent's output tokens and 1.7 times its
  prompt tokens; v12 max 1.4 times both.
- On the three games the base agent won outright (ft09, lp85, vc33) it was 2
  to 4 times cheaper. v12 spends most of its turns in fit rounds (57-73% of
  flash's turns on four games), which buy action efficiency rather than
  levels.
- sp80 and ls20 are the only games where both sides spent about the same
  (500-750K output). There v12's tokens bought 3-4 sp80 levels against 1, and
  the base agent's bought 5 ls20 levels against 2-3.
- **At the base agent's budget of 500K output tokens per game**, v12 flash
  would have stopped at turn 244 on lp85 (6/8), 225 on sp80 (4/6) and 221 on
  vc33 (4/7), a mean of about 50 instead of 62.2. v12 max would have kept
  53.9, since it solved nothing after crossing 500K.

### Against the dfranzen notebook's four passes

One run per arm says little about a stochastic agent. Version 3 of the
dfranzen notebook ([kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution),
run 2026-10-03, read through the public Kaggle API, `kernels/output`) reran the
submission with 4 passes on all 25 demo games: 100 game runs, 33 won, mean
46.49. Its per-pass result lines for these five games:

| game | dfranzen passes | mean | won | v12 flash | v12 max |
| --- | --- | --- | --- | --- | --- |
| ft09 | 100, 100, 100, 47.6 | 86.9 | 3/4 | 100 (won) | 100 (won) |
| lp85 | 100, 100, 100, 100 | 100 | 4/4 | 77.8 | 100 (won) |
| ls20 | 37.9, 27.9, 25.4, 16.3 | 26.9 | 0/4 | 10.7 | 10.8 |
| sp80 | 28.6, 14.3, 4.8, 4.8 | 13.1 | 0/4 | 47.6 | 23.0 |
| vc33 | 100, 100, 100, 21.4 | 80.4 | 3/4 | 75.0 | 35.7 |
| **mean** | | **61.5** | **10/20** | **62.2** (1/5) | **53.9** (2/5) |

The notebook's passes ran a local quantized Qwen3.8-Flash-Next on Kaggle's
GPU, with a runtime budget per game (62K to 332K output tokens per game), not
qwen3.8-flash through OpenRouter. The base run of this page
(`runs/20261004_135539`, 66.9) falls inside their spread.

Is the difference significant? No:

- **Mean score.** Summing the per-game variances of the four passes, one
  pass's 5-game mean has a standard deviation of 9.9 points, most of it from
  ft09 and vc33, which a pass either wins or abandons. v12 flash is +0.8
  (0.1 standard deviations), v12 max -7.5 (0.7).
- **Wins.** 10 of 20 game passes against 1 of 5 for flash (Fisher exact
  p = 0.34) and 2 of 5 for max (p = 1.0).
- **Per game.** v12 flash's sp80 is above all five base results (the four
  passes and `runs/20261004_135539`), and its ls20 below all five. With one v12
  sample each, that happens by chance with probability 1/6 (p about 0.17).
  These are the most consistent signals, and still not conclusive.

Three more v12 flash runs on these games (about $10 each) would give four
runs per arm, enough to detect a difference of about 15 points in the mean;
sp80 and ls20 are where a real one would show first.

The base agent with qwen3.8-max-0902 is its own experiment: [base-max-5games.md](base-max-5games.md).

## What went wrong in the runs

- **Rate limit (flash only).** With ten games in flight, flash met 83 HTTP 429s
  in its first hour; max met none. The client retries them in place, so no
  game failed, but the waits added up. The flash run was stopped and resumed
  at 21:08 UTC with the new `--jobs 3` (`3f87932`), three games at a time;
  429s then fell to about one every five minutes.
- **Alibaba's input filter (flash only).** Two flash games died on HTTP 400
  `data_inspection_failed` ("Input text data may contain inappropriate
  content"), which the client treated as fatal: ft09 at turn 104 (4/6) and
  ls20 at turn 291. A retry did not help (20 refusals in a row). Probing the
  rebuilt ft09 request with 1-token calls showed that the filter judges the
  request as a whole. No message is refused on its own, and the request
  passes once its one image (the game frame) or its tool list is left out.
  Since `fc81ce0` the client asks once more as it is, then without the
  images, with a note in their place. Both games resumed past the refused
  request: ft09 went on to win 6/6 in 80 actions, and ls20 finished its
  last 9 turns. Alibaba is flash's only provider on OpenRouter, so the
  request cannot be routed elsewhere.
- **Restarts.** Background tasks here end after 2 hours, and the VM was
  replaced once (at turn 296 of flash vc33). Every game resumed from its
  transcript and trace: flash games resumed 2 or 3 times, max games 0 to 2.

## Conclusions

- The v12 harness holds up over whole games. It keeps every request under
  the 128K deployment limit with no loss against v11, and it scores 62.2
  across the five games against the base agent's 66.9 in one run and 61.5
  over the dfranzen notebook's four passes. The difference is not
  significant either way, and v12 spends 1.9 times the output tokens.
- It wins on sp80, the game the base agent cannot model (47.6 against 3.74),
  and loses on ls20 (10.7 against 30.97) and on the last levels of lp85 and
  vc33. In each of those, flash was still progressing when the 300-turn limit
  ended the game.
- A larger model does not pay off on this harness. Max solves lp85 that flash
  did not, and in fewer turns, but it is so slow that the time and cost limits
  cut its three long games short. On ls20 it spent 183 actions on level 1,
  against flash's 63.
- The turn limit binds flash on four games out of five. Raising it, or making
  fit rounds shorter ([v12-forks.md](v12-forks.md), anatomy of a fit round),
  is the direct lever on the remaining gap.
