# Trajectory preparation

This folder contains the shared data code used by the final architecture and
its benchmark preparation. Model execution lives in
[architecture/](../architecture/README.md).

The latest production loader uses the **58 compaction trajectories** in
`data/progressive-sol25-trajectories`, with explicit turn ownership. See
[production training](../architecture/TRAINING.md). The complete-game
reconstruction tools below support the earlier preparation/benchmark contract.

| File | Purpose |
| --- | --- |
| `dataset.py` | Stable hashes, processor loading, fold validation and request utilities |
| `trajectory_data.py` | Verified chronological reconstruction, immutable annotations and whole-trajectory update planning |
| `trajectory_encode.py` | Full input encoding and every-assistant target positions, including interleaved observations/images |
| `prepare_trajectories.py` | Atomic complete-game dataset preparation and source/fold checks |
| `tests/test_trajectory_data.py` | Reconstruction, supervision, vision identity and preparation checks |
| [TRAJECTORY_DATA.md](TRAJECTORY_DATA.md) | Data contract and preparation command |
| [HANDOFF_NEXT_SESSION.md](HANDOFF_NEXT_SESSION.md) | Current state and next-session constraints |

The complete-game preparation below contains one trajectory per game. Every assistant
turn is supervised; user/tool/image observations remain ignored-label context.
Missing history, missing generated thinking and overlength trajectories fail
preparation. Diagnostic benchmark prefixes do not change this production rule.

`architecture/prepare_benchmarks.py --encoder-root exp/sft-flash-next/train`
uses the encoder here. It prepares labeled PT inputs for the recorded capacity
experiments. The architecture runner accepts explicit labeled PT samples; a
production integration now has local mock/DVC tests; live GPU/data qualification
is still pending.

The previous training backends, qualification launchers, optimization trials,
trajectory training prototype and their tests are source-only history under
[experiments/legacy-training/](../experiments/legacy-training/README.md).
Their results and artifact pointers are retired. Current architecture run data,
reference gradients, benchmark inputs and verification tools are retained.

Restore the retained model-selection metadata using
[the evidence instructions](../EVIDENCE.md) when needed. Do not initialize
production training from a diagnostic or fold-0 overfit adapter.
