# Full 25-game generated-thinking dataset: verification and exploration

Analyzed 2026-10-09 from the finalized DVC checkout at [`runs/think-progressive-sol25`](../../runs/think-progressive-sol25.dvc). This is the complete generated-thinking run for all 25 Sol games. Its 25 SFT files contain **1,324 training samples** from **1,334 finalized turns**. Ten turns exceed the 120,000-token student input limit and have no training target, though their finalized thinking remains in later contexts.

## Verification

`dvc status ARC3-Inference/runs/think-progressive-sol25.dvc` reports the checkout up to date. The snapshot, 1,334 `final.json` checkpoints, 25 SFT files, and 1,324 distinct SFT IDs agree. The exact 10 omitted IDs match the snapshot; each omitted turn has 120,036–127,248 input tokens, and every exported sample is at or below 120,000. The 25 games each have a complete finalized chain.

For every SFT sample, the final loss-mask index points to the last assistant message, its generated thinking equals the normalized finalized checkpoint text, and its recorded input token count matches a fresh render and tokenization with the tokenizer and template whose SHA-256 hashes match the run manifest. The tool schema is code-only; 1,323 targets call `python(code)` and one final target is text-only. All 37,163 image occurrences have a 640×640 PNG header. The export is 388,695,311 bytes (370.7 MiB) across 25 JSONL files. The per-sample checks and counts are reproducible with [`analyze-progressive-sol25.py`](analyze-progressive-sol25.py).

## Token accounting

The **input** is the rendered prompt through the assistant generation prefix, including earlier generated thinking, tool outputs, and board images. The **output** is only the final assistant message trained by `loss_target_message_index`: generated `<think>` text, Python call or final text, and chat formatting. Counts use the pinned Qwen3.8-Flash-Next tokenizer and template; each 640×640 board image contributes 400 vision tokens (replacing one text placeholder). These are processor tokens, not provider usage. Repeated contexts and images are counted in each request where they occur.

| Per training sample | Total | Min | P25 | Median | P75 | P90 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Input tokens | 81,307,152 | 5,519 | 42,843 | 62,772 | 82,454 | 97,582 | 105,336 | 119,736 |
| Output tokens | 1,380,369 | 195 | 583 | 802 | 1,254 | 1,887 | 2,427 | 7,432 |
| Input + output | 82,687,521 | 5,885 | 43,812 | 63,634 | 83,435 | 98,872 | 106,943 | 121,022 |
| Generated thinking text alone | 949,190 | 116 | 407 | 567 | 859 | 1,317 | 1,672 | 4,784 |

Percentiles use linear interpolation and are rounded to the nearest token. Thinking text is tokenized separately; its count is diagnostic and is **not additive** with the complete output count because template boundaries affect tokenization. The input total is **58.9×** the output total. Across all 1,334 finalized turns, including the 10 excluded targets, recorded student inputs total **82,539,005 tokens**.

Generated thinking has a median **386 words** per eligible sample (P25 275, P75 594; range 94–3,573). Contexts contain **37,163 image occurrences**, or 14,865,200 vision tokens under the 400-token convention (18.3% of input tokens). The median sample has 28 images and 57 messages. The longest eligible input is `lf52-271a04aa_p0#36` (119,736 tokens); the longest output is `wa30-ee6fef47_p0#95` (7,432 tokens). The latter includes a long Python call; output length is not thinking length alone.

## Length distributions

### Input tokens

| Range | Samples | Share |
| --- | ---: | ---: |
| 0–19,999 | 136 | 10.3% |
| 20,000–39,999 | 163 | 12.3% |
| 40,000–59,999 | 318 | 24.0% |
| 60,000–79,999 | 338 | 25.5% |
| 80,000–99,999 | 258 | 19.5% |
| 100,000–120,000 | 111 | 8.4% |

### Output tokens

| Range | Samples | Share |
| --- | ---: | ---: |
| 0–499 | 177 | 13.4% |
| 500–999 | 648 | 48.9% |
| 1,000–1,999 | 387 | 29.2% |
| 2,000–3,999 | 102 | 7.7% |
| 4,000–7,999 | 10 | 0.8% |

### Generated thinking words

| Range | Samples | Share |
| --- | ---: | ---: |
| 0–249 | 222 | 16.8% |
| 250–499 | 661 | 49.9% |
| 500–999 | 326 | 24.6% |
| 1,000–1,999 | 108 | 8.2% |
| 2,000–3,999 | 7 | 0.5% |

## Coverage by game

The export covers all 25 games but is not balanced by game: eligible sample counts range from 14 to 147. The five longest games supply 578 of 1,324 samples (43.7%). A request-weighted result will therefore put more weight on long games than a game-weighted result.

| Game | Samples | Input tokens | Output tokens | Median input | Median output | Images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ar25-0c556536` | 25 | 1,132,542 | 18,556 | 44,612 | 649 | 409 |
| `bp35-0a0ad940` | 112 | 7,886,123 | 149,352 | 70,457 | 1,017 | 3,541 |
| `cd82-fb555c5d` | 39 | 2,143,678 | 27,523 | 56,282 | 580 | 1,231 |
| `cn04-2fe56bfb` | 32 | 1,565,369 | 25,124 | 49,291 | 619 | 700 |
| `dc22-fdcac232` | 90 | 6,097,069 | 93,282 | 68,634 | 867 | 2,791 |
| `ft09-0d8bbf25` | 15 | 403,880 | 10,965 | 26,218 | 637 | 148 |
| `g50t-5849a774` | 37 | 2,088,859 | 26,929 | 57,468 | 676 | 1,007 |
| `ka59-38d34dbb` | 54 | 3,285,345 | 49,666 | 62,956 | 759 | 1,624 |
| `lf52-271a04aa` | 121 | 8,533,411 | 133,246 | 71,295 | 886 | 4,207 |
| `lp85-305b61c3` | 31 | 1,610,098 | 28,650 | 52,474 | 883 | 552 |
| `ls20-9607627b` | 62 | 4,118,738 | 66,852 | 63,269 | 931 | 1,775 |
| `m0r0-492f87ba` | 31 | 1,684,169 | 24,992 | 55,091 | 715 | 746 |
| `r11l-495a7899` | 25 | 1,061,471 | 28,672 | 40,820 | 978 | 367 |
| `re86-8af5384d` | 64 | 4,357,513 | 52,216 | 70,275 | 626 | 2,068 |
| `s5i5-18d95033` | 40 | 2,682,733 | 50,178 | 76,804 | 1,089 | 838 |
| `sb26-7fbdac44` | 14 | 391,175 | 10,033 | 26,659 | 637 | 123 |
| `sc25-635fd71a` | 34 | 1,937,546 | 29,218 | 53,328 | 764 | 844 |
| `sk48-d8078629` | 147 | 9,534,593 | 169,556 | 65,211 | 758 | 4,591 |
| `sp80-589a99af` | 27 | 1,194,291 | 34,608 | 44,098 | 1,113 | 426 |
| `su15-1944f8ab` | 70 | 4,741,006 | 86,638 | 72,764 | 1,100 | 2,077 |
| `tn36-ef4dde99` | 30 | 1,440,119 | 27,183 | 47,564 | 795 | 508 |
| `tr87-cd924810` | 28 | 1,260,008 | 28,865 | 46,559 | 969 | 509 |
| `tu93-0768757b` | 63 | 3,947,034 | 58,710 | 62,853 | 758 | 1,932 |
| `vc33-5430563c` | 25 | 1,166,347 | 27,034 | 44,788 | 1,016 | 494 |
| `wa30-ee6fef47` | 108 | 7,044,035 | 122,321 | 66,363 | 815 | 3,655 |

Eligible sample counts by level are: 1: 172, 2: 115, 3: 128, 4: 132, 5: 228, 6: 172, 7: 147, 8: 115, 9: 106, 10: 9. The complete per-sample lengths and metadata are in [`progressive-sol25-lengths.csv`](progressive-sol25-lengths.csv).

## Generation and monitoring signals

Of the 1,334 finalized turns, 415 used the initial draft, 487 used one refinement, and 432 used two. The deterministic monitoring panel requested 374 final audits. One audit was unavailable; among the 373 available, **207 were clean and 166 flagged** by at least one judge criterion (44.5% flagged). The overlapping flag counts are coverage 42, code consistency 24, fact grounding 85, and functional call equivalence 88. Functional equivalence was unavailable for two monitored turns. The audit counts were recomputed from the checkpoints and match the snapshot.

These are the run's model-judge assessments on the monitoring subset, not independent ground truth. Flagged samples were retained in the SFT export, as designed; the ten training exclusions are solely for input length. Review these flags before treating the generated rationales as uniformly reliable.

## Reproduce

From the repository root, with the DVC checkout and pinned Qwen files present:

```bash
ARC3-Inference/.venv/bin/python \
  ARC3-Inference/experiments/teacher-reasoning/analyze-progressive-sol25.py \
  --csv ARC3-Inference/experiments/teacher-reasoning/progressive-sol25-lengths.csv \
  > /tmp/progressive-sol25-eda.json
```

The script checks checkpoint/export coverage, exclusion rules, manifest tokenizer hashes, loss masks, generated thinking, code-only targets, image dimensions, and every exported input token count. It retokenizes every output and prints machine-readable distributions and per-game/per-sample metrics.

## Trajectory grouping at note compactions

The 1,334 finalized responses form **58 contiguous context stretches**: 25 initial stretches plus 33 note-compaction boundaries. Every stretch contains at least one eligible training target. Stretch lengths are 1–43 finalized responses (median 20); eligible target counts per stretch are also 1–43 (median 20). Within a stretch, the earlier SFT message list is an exact prefix of the next one for all **1,266** adjacent eligible pairs checked. The other 33 adjacent pairs cross a compaction boundary.

| Game | Trajectories | Finalized responses per trajectory |
| --- | ---: | --- |
| `bp35` | 7 | 39, 19, 13, 13, 12, 14, 3 |
| `sk48` | 7 | 39, 20, 25, 21, 16, 16, 10 |
| `lf52` | 6 | 39, 20, 21, 20, 18, 5 |
| `wa30` | 5 | 38, 20, 21, 17, 12 |
| `dc22` | 4 | 39, 19, 18, 15 |
| `ls20`, `su15` | 3 each | 36, 19, 7; 42, 14, 14 |
| `ka59`, `re86`, `s5i5`, `sc25`, `tu93` | 2 each | 39, 15; 42, 23; 35, 8; 35, 1; 43, 20 |
| Other 13 games | 1 each | Entire game |

All 33 boundary prompts explicitly say **10 retained turns**. The harness counts these by ordinary game-turn opener, not by chat-message count. It can keep fewer if ten turns exceed its budget, but it did not do so in this release. The actual retained prefix has 10–24 prior assistant responses (median 11), because a game turn can contain multiple assistant/tool exchanges, plus the note request, assistant note call, and synthetic tool result. It may have 10–15 user messages when other notices are present. The first post-compaction SFT sample records the exact retained prefix; copy that rather than selecting the last ten messages yourself.

At a boundary, the old trajectory ends with the user compaction notice and assistant note call. The synthetic tool result first appears in the next request's context. The next trajectory begins with the system prompt, the actual retained ten-turn suffix, the same compaction notice/note/tool result, and then the next ordinary user message. Mask the overlapping retained assistant messages and repeated note call in the new trajectory; otherwise their targets receive loss twice. Under the old 120,000-input-token SFT filter, ten targets were excluded: six note-call targets and four in the run-up to compactions. Reconstruct those targets from the finalized checkpoints and source request logs rather than relying on SFT rows alone.

The original per-sample export repeats **81.31 million input tokens**. The sum of the 58 terminal prompt lengths is **5.58 million tokens**, before adding their final replies and tool results. That measures the scale of repeated context removed by this grouping; it is not a predicted training-speed ratio because long-sequence attention and packing costs differ. Six note-ending stretches exceed the 120,000-token *input* limit; their exact sizes including the assistant note output are below.

| Compaction note target | Input | Assistant output | Input + output |
| --- | ---: | ---: | ---: |
| `s5i5-18d95033_p0#34` | 126,409 | 2,760 | **129,169** |
| `sc25-635fd71a_p0#34` | 127,248 | 1,704 | 128,952 |
| `lf52-271a04aa_p0#38` | 124,388 | 2,511 | 126,899 |
| `dc22-fdcac232_p0#38` | 121,182 | 2,495 | 123,677 |
| `re86-8af5384d_p0#41` | 120,544 | 2,139 | 122,683 |
| `bp35-0a0ad940_p0#38` | 120,036 | 2,344 | 122,380 |

One more note-ending trajectory has an input below 120,000 but reaches **120,601** after its assistant output. Thus **7 of the 33** note-ending trajectories exceed a 120,000-token *total sequence* limit. The largest reaches **129,201 tokens** if its 32-token synthetic tool result is included as the closing message. A trainer capped at 120,000 total tokens needs an additional split or a larger window. Tokenize the final grouped sequences against the actual training limit before writing them.

Recompute the boundary count and prefix checks with [`analyze-progressive-sol25-trajectories.py`](analyze-progressive-sol25-trajectories.py). It reads the finalized checkpoints and SFT export and prints per-game trajectory lengths plus details of all 33 boundaries.

The [DVC trajectory export](../../../data/progressive-sol25-trajectories/README.md) implements a 130,000-token **input + output** cap and includes all 1,334 targets. It ends each note trajectory with the assistant answer; the note tool result appears in the next trajectory. The largest exported sequence is 129,169 tokens. Its README explains why the output/input ratio is lower after compaction: the retained context contributes masked input tokens, while those later trajectories contain fewer new supervised turns.
