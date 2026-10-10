# Flash-Next training

| Folder/file | Purpose |
| --- | --- |
| [architecture/](architecture/README.md) | Supported reference/optimized model, training, tests, Jupyter execution and benchmarks |
| [architecture/v0.md](architecture/v0.md) | Consolidated experiment and capacity results |
| [train/](train/README.md) | Shared trajectory reconstruction, encoding and preparation |
| [experiments/legacy-training/](experiments/legacy-training/README.md) | Retired experiments, source only |
| [EVIDENCE.md](EVIDENCE.md) | DVC storage and restoration of retained detailed evidence |
| [train/HANDOFF_NEXT_SESSION.md](train/HANDOFF_NEXT_SESSION.md) | Single current handoff |

The final stack uses target-only CCE exact, direct attention bias, and expert,
QSA, hyperconnection and PLE chunk replay. BF16 activation storage, GPU FP32
LoRA parameters/gradients, CPU frozen-expert prefetch and disk PLE are its base
settings. See the architecture documentation for exact contracts and limits.

Completed 96K/120K captures measure forward/backward capacity without optimizer
state or updates. Production learning, final-stack resume and held-out evaluation
remain unqualified. Source organization changes do not extend those claims.
