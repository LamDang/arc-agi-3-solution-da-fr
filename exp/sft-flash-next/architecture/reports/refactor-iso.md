# Clean architecture refactor isolation

Full16K replay `20261009215622723-239830ad`, execution commit `e727f4b`.
Optimized builder with every flag disabled; same nonzero diagnostic A/B fixture
and complete multimodal anchor as native v0. Process exited0; no timeout,
clipping, optimizer update or overfit.

| Check | Result |
| --- | --- |
| Loss | 0.6256952285766602, bitwise identical |
| Raw gradients | 744/744 bitwise identical, finite and nonzero |
| Initialization | 744/744 bitwise identical |
| Global gradient relative L2 | 0 |
| Gradient file SHA256 | `91465990e81dd57bb29368510354674415c1f1d8bbbaba5225b62fdf4a8cced2` |
| Sample | 16,249 tokens /651 targets /7 images |

| Phase | Seconds | GPU allocated GiB | GPU reserved GiB | Parent RSS GiB | Tree PSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| loading | 220.827 | 39.351 | 68.607 | 170.633 | 170.446 |
| forward | 172.837 | 77.802 | 78.064 | 162.665 | 162.590 |
| backward | 224.960 | 85.100 | 85.689 | 162.684 | 162.609 |
| gradient_export | 0.161 | 39.836 | 85.689 | 99.385 | 99.310 |

CUDA peaks are synchronized allocator counters. RAM peaks are sampled process
measurements; tree PSS includes DataLoader children. Timing is a single replay,
not a speed qualification or130K capacity result.

Independent verifier: `verify_iso.py`; source/input pins and every raw tensor
are checked without Torch using NumPy. Raw attempt and SHA256 manifest are in
local DVC. This replay verifies the legacy refactor path; it does not qualify
new precision changes, enabled kernels or the newly requested all-expert reference.

Initial attempt `20261009213548935-2c90d284` also matched exactly, then exited1
on a post-capture event logger error. Its raw evidence/failure are retained in
local DVC and its per-attempt report records process_success=false. Logger fix
is covered by a regression test and the clean-exit replay above.

A separate32-token all-component wiring capture exited0 and exported744 finite,
nonzero BF16 gradients. `components-smoke.json` explicitly records that it is
not numerical qualification. Historical Opt6 drift remains unresolved.

No Git or DVC push.
