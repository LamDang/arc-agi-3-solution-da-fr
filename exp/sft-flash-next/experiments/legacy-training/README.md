# Retired training experiments — source only

This folder preserves earlier implementations for code review. It contains
source, tests and their configuration inputs. Historical results, tensor data,
logs, experiment reports, DVC pipeline state and artifact pointers are retired.

The supported implementation is [architecture/](../../architecture/README.md).
The shared trajectory encoder and preparation code remain in
[train/](../../train/README.md). Current architecture runs, numerical comparisons,
verification tools and benchmark inputs remain with the active implementation.

## Source groups

| Files | What they implemented |
| --- | --- |
| `backend.py`, `run.py`, `evaluate.py`, `checkpoints.py`, `launch_fit.sh` | Earlier custom quantized training backend, qualification and checkpoint workflow |
| `sparse_kernels.py`, `gdn_blocks.py`, `gdn_segments.py`, `offload.py`, `kernel_checks.py` | Custom sparse attention, segmented GDN, activation storage and kernel checks |
| `native_*`, `overfit_hf_reference.py`, `gradient_audit.py` | Original native HF reference, repeatability and gradient comparisons |
| `target_only_head.py`, `cce_*`, `liger_*`, `opt3_*`, `opt4_*`, `ple_preparation.py` | Earlier head, loss, mask, PLE and fusion candidates; accepted ports now live in architecture components |
| `bf16_*`, `trace_layer_dtypes.py` | Precision investigations and one-sample learning instrumentation |
| `kaggle_*` | Previous Jupyter dispatch and supervision; replaced by `architecture/jupyter.mjs` and its runtime |
| `trajectory_run.py`, `trajectory_training.py`, `trajectory_checkpoints.py`, `trajectory_head_loss.py`, `trajectory_initialize.py`, `trajectory_gradient_*` | Earlier production-training/resume prototype; not integrated with the final all-expert architecture |
| `prepare_*`, `build_*`, `capacity_probe.py`, `summarize_capacity.py`, `archive_capacity.py` | Previous preparation, packaging and capacity utilities |
| `configs/`, `tests/`, `reviews/`, `diagnostics/*/*.py` | Configuration inputs, source checks and small diagnostic programs belonging to these experiments |

## Status and dependencies

These are frozen research sources, not supported executable entry points.
Moving them does not qualify their algorithms or fix their old relative paths.
Several assume the former `train/` layout, model assets, or artifacts that were
intentionally removed. Use the preceding Git history if the original layout is
needed for source archaeology; this folder provides no historical data bundle.

Do not import this folder from the active model or add it to a training path.
The active arithmetic-port tests inspect selected source definitions with AST
locally; those checks do not execute the retired runners. The native indexer
source fixture used by that test is preserved under `architecture/tests/fixtures/`.

The trajectory prototype's whole-sample token normalization, resume and held-out
evaluation ideas remain available here as source. Their migration and production
qualification for the final architecture are future work. The current
architecture's training entry point starts fresh and is documented accordingly.
