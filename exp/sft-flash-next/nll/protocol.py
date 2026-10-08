"""Staged panel policy: full baseline, 256, then conditional intermediate scan."""
import math

BASE_COUNTS = (512, 256)
SCAN_COUNTS = (448, 384, 320)
POLICY = dict(version="512-then-256-relative-primary-nll-v1", baseline=512,
              first_candidate=256, threshold_relative=0.05,
              metric="equal-game mean of population-weighted request mean NLL",
              gate_requires_all_requests=30, scan_counts=list(SCAN_COUNTS),
              selection="smallest evaluated count within 5% of full baseline NLL")


def gate(full, candidate):
    if full is None or candidate is None:
        return dict(status="pending", relative_increase=None, threshold_relative=0.05)
    if not all(math.isfinite(x) and x >= 0 for x in (full, candidate)):
        raise ValueError("Gate requires finite nonnegative primary NLL")
    limit = full * 1.05
    # Only roundoff at the inclusive 5% boundary is tolerated.
    exceeds = candidate > limit and not math.isclose(candidate, limit, rel_tol=1e-12, abs_tol=0)
    return dict(status="scan_required" if exceeds else "stop_at_256",
                baseline_primary=full, candidate_primary=candidate,
                relative_increase=(candidate/full-1) if full else (0. if candidate == 0 else None),
                threshold_relative=0.05, threshold_primary=limit)


def choice(summaries):
    full = summaries["512"]["primary"]
    return min(int(c) for c, s in summaries.items()
               if gate(full, s["primary"])["status"] == "stop_at_256")


def budget(manifest):
    tokens = sum(r["total_tokens"] for r in manifest["samples"])
    targets = sum(r["target_tokens"] for r in manifest["samples"])
    return dict(protocol=POLICY, initial_jobs=60, maximum_jobs=150,
                initial_processed_tokens=tokens*2, conditional_scan_processed_tokens=tokens*3,
                maximum_processed_tokens=tokens*5, initial_scored_tokens=targets*2,
                maximum_scored_tokens=targets*5,
                scenarios={str(rate): {
                    name: dict(scoring_minutes=(minutes := tokens*models/rate/60),
                               cold_sessions=(sessions := math.ceil(minutes/90)),
                               total_minutes=[minutes+20*sessions, minutes+30*sessions])
                    for name, models in (("initial", 2), ("maximum", 5))}
                    for rate in (500, 850, 950)})


def execute_stages(samples, evaluate, checkpoint):
    """Checkpointed stage order; evaluate skips only checksum-verified results.

    A partial stage raises/pause upstream, so the gate cannot run on a subset.
    No intermediate forwards are admitted until all 60 base jobs are durable.
    """
    for count in BASE_COUNTS:
        for row in samples:
            evaluate(row, count)
            checkpoint()
    decision = checkpoint()
    if decision["gate"]["status"] == "scan_required":
        for count in SCAN_COUNTS:
            for row in samples:
                evaluate(row, count)
                checkpoint()
    elif decision["gate"]["status"] != "stop_at_256":
        raise ValueError("Complete base stages did not yield a gate decision")
    return checkpoint()
