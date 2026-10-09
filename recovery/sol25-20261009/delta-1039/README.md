# Sol25 incremental checkpoint at 1,039 turns

This branch extends the verified 1,004-turn backup at `9804ad9b87ab1e0f9da748e01fcb284725da8338`. It preserves the original backup unchanged. Generation stopped safely when the workspace proxy returned Envoy 503 before provider responses. The estimated successful-response cost is $71.89093128; no reservations remain. The pinned 939-turn partial DVC snapshot remains unpublished.

Restore the base backup using `recovery/sol25-20261009/HANDOFF.md`, including its Git branch/staging instructions. From a clone of this recovery branch, run `python recovery/sol25-20261009/delta-1039/restore-delta.py --output /path/to/active-release-checkout`. The script verifies every part, the full archive, and each overlay path before extraction. This overlay contains 310 new attempts, 150 completed stages, 35 finalized turns, and updated status files. The resulting run has 4,901 attempts, 4,492 stages, and 1,039 finalized turns.

Do not resume inference while provider probes return Envoy 503. A required `lf52` judge exhausted 12 attempts and needs a justified higher `--max-attempts` after connectivity returns. The $300 guard and all generation settings remain unchanged.
