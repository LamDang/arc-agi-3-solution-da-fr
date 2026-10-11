# Progressive thinking trajectories: NVIDIA DreamTeam games (22 won)

Same format as the parent [Sol25 trajectory dataset](../README.md), built from
gpt-6.1-sol's run on the 25 NVIDIA DreamTeam community games
([exp/gpt61sol-nvidia-25games.md](../../../ARC3-Inference/exp/gpt61sol-nvidia-25games.md)).
**58 trajectories, 22 games, 1,297 finalized responses: 1,187 supervised, 110
masked.** Read the parent README first; this one lists only what differs.

## Source and thinking generation

- **Source:** `ARC3-Inference/runs/gpt61sol-nvidia-25games-resume2`, the
  scored run directory, which holds the final request log of every game. Only
  the 22 games that ended `won` are used. fl5273, ma4173 and ss6041 were not
  won; they are left out.
- **Thinking:** the same pipeline and settings as the Sol25 run
  ([progressive-sol25.md](../../../ARC3-Inference/experiments/teacher-reasoning/progressive-sol25.md)),
  unchanged:
  - Qwen3.8-Flash draft with up to two refinements;
  - combined gpt-6.1-sol xhigh judges;
  - 15 evenly spaced audited turns per game;
  - 120K student-input limit for the per-request SFT export.
- **Run:** `ARC3-Inference/scripts/run_progressive_nvidia.py`, output
  `ARC3-Inference/runs/think-progressive-nvidia22` (DVC). It ran a two-turn
  pilot with a zero-call resume check, then all 22 games at 16 workers.
  - Estimated cost: **$69.32**, from returned usage. This is not an account
    statement.
  - Refinements: 370 turns needed none, 431 needed one, 496 needed two.
- **Final audits:** 327 sampled turns, all available. 191 were clean and 136
  were flagged. The Sol25 run had 207 clean of 373.

  | flag | turns |
  | --- | ---: |
  | fact grounding | 68 |
  | call equivalence | 56 |
  | coverage | 39 |
  | code consistency | 29 |

  As in Sol25, flags are kept for review and do not drop targets.

One code fix was needed:
- A text-only turn is keyed by a hash of its text. The harness removes blank
  lines when it keeps a text reply in history, so the reply and its later copy
  hashed differently, and the turn after a text reply failed.
- `think_gen.context.message_ref` now hashes the normalized text.
- Sol25 refs are unchanged: its single text reply has no blank lines.
- `scripts/migrate_progressive_nvidia_text_refs.py` updated the one finalized
  turn that was affected and recorded it in `text-ref-migration.json` in the
  run directory.

## Differences from the Sol25 trajectories

**Masked text-only turns.** 110 replies have no tool call. They are the
teacher giving up on a board it could not change, for example "RESET is not
available… please restart level 1".
- Cause: the NVIDIA adapter never reports GAME_OVER, and the harness hides
  RESET. See "Frozen games" in the experiment write-up.
- In all 9 games where they occur, the teacher later called the hidden RESET
  itself and went on to win.
- Handling: the replies stay in `messages` exactly as the later requests saw
  them, with their generated thinking. They are **not** loss targets:
  - their indices are in `masked_text_only_message_indices`;
  - each turn carries `"text_only": true`;
  - their tokens count as input.
- History copy: the harness drops blank lines from a text reply it keeps in
  history, and the trajectory carries that copy.
- Every `python` turn, including those that rediscover RESET, is supervised.

**Fewer retained turns at some compactions.** At 5 of the 33 compactions, ten
turns would not fit, so the harness kept fewer and said so in the notice:
- `retained_game_turns` records the number from the notice: 9 four times
  and 7 once.
- The other checks are those of the parent builder.

**Budget trims.** Three boundaries, all in fw4821, are the harness's budget
trimmer, not a compaction. Mid-turn, the request still exceeded the context
budget, so the oldest messages were dropped without a notice.
- The next trajectory is the system prompt, then an exact suffix of the
  previous trajectory, then new messages. The builder verifies this.
- Its `boundary_to_next` has `"kind": "context_trim"` with
  `dropped_messages`, `retained_messages` and `retained_assistant_messages`.
- Compaction boundaries have `"kind": "note_compaction"`.
- The Sol25 games never hit the trimmer.

**Over the 120K export limit.** Two hr2048 turns (#39, #40) exceeded the
120K student-input limit of the per-request export. Like the ten such turns
in Sol25, they fit the 130K trajectory limit and are supervised here.

## Statistics

| Measure | Min | P25 | Median | P75 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total tokens | 36,645 | 77,687 | 88,024 | 106,503 | 117,362 | 125,307 |
| Input tokens | 28,298 | 61,805 | 68,330 | 78,115 | 86,034 | 88,100 |
| Supervised output tokens | 698 | 15,537 | 21,746 | 28,563 | 36,851 | 40,667 |
| Responses per trajectory | 1 | 8 | 18 | 38 | 44 | 62 |
| Images | 12 | 28 | 35 | 49 | 57 | 66 |

- 22 initial trajectories and 36 after a boundary.
- Total: 5,216,796 sequence tokens, of which 1,280,722 are supervised output.
- Output/input ratio: 0.323 median and 0.325 token-weighted. Sol25 had 0.324
  and 0.325.
- Trajectories per game: 1 for 10 games and 2 for 6; 9 for mt4926, 8 for
  al7306 and 7 for fw4821.

`summary.json` has the counts and hashes. `index.json` has one row per
trajectory, plus `text_only_turns`. `ratios.csv` matches the parent's file.

## Reproduce

From the repository root, with the `ARC3-Inference` environment:

```sh
dvc pull ARC3-Inference/runs/gpt61sol-nvidia-25games-resume2.dvc \
  ARC3-Inference/runs/think-progressive-nvidia22.dvc
dvc repro data/progressive-sol25-trajectories/nvidia-25games/dvc.yaml
```

The pinned Qwen tokenizer and template come from the
`data/sft-gpt61sol-features-25games` fetch stage.

`build.py` imports the parent's `build.py` for:
- target token counting;
- the compaction check;
- the trajectory totals.

It adds the masking, the retained-turn count and the trim boundary. Its
outputs are pinned in `dvc.lock`.
