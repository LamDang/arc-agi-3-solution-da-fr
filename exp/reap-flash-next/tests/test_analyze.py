"""analyze.py on synthetic statistics with a known answer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402

L, E, C = 3, 8, 3


def _stats(used: list[int], tokens: int = 100) -> dict:
    """Games route only to `used` experts, expert j with output norm j+1."""
    count = np.zeros((L, E, C))
    count[:, used, 1] = tokens
    gate = count * 0.25
    norm = count * (np.arange(E)[None, :, None] + 1)
    return {"count": count, "gate": gate, "norm": norm, "gate_norm": gate * (np.arange(E)[None, :, None] + 1),
            "prob": gate}


def _write(tmp_path, runs):
    for key, stats in runs.items():
        path = tmp_path / "stats" / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **stats)


def test_scores_keep_and_coverage():
    stats = analyze.aggregate({"a": _stats([0, 1, 5])}, ["a"], [0, 1, 2])
    scores = analyze.reap_scores(stats)
    assert np.allclose(scores[0], [0.25, 0.5, 0, 0, 0, 1.5, 0, 0])
    mask = analyze.keep_mask(scores, 2)
    assert mask[0].tolist() == [False, True, False, False, False, True, False, False]
    np.testing.assert_allclose(analyze.coverage(stats, mask), 2 / 3)


def test_criteria():
    stats = analyze.aggregate({"a": _stats([0, 1, 5]), "b": _stats([1], tokens=300)}, ["a", "b"], [0, 1, 2])
    # REAP ignores frequency (expert 5 has the largest outputs); gate and count favour expert 1
    assert analyze.keep_mask(analyze.expert_scores(stats, "reap"), 1)[0].nonzero()[0].tolist() == [5]
    for criterion in ("gate", "count", "prob"):
        assert analyze.keep_mask(analyze.expert_scores(stats, criterion), 1)[0].nonzero()[0].tolist() == [1]


def test_leave_one_game_out(tmp_path):
    # game b uses an expert (7) nobody else uses: keeping 3 experts chosen
    # without b misses it, chosen with b keeps it
    _write(tmp_path, {"k/aaaa_p0": _stats([0, 1, 2]), "k/aaaa_p1": _stats([0, 1, 2]), "k/bbbb_p0": _stats([1, 7])})
    runs = analyze.load(tmp_path)
    cv = analyze.cross_validate(runs, [3], [0, 1, 2], "reap")[3]
    by_game = {r["game"]: r for r in cv["games"]}
    assert by_game["bbbb"]["held_out_mean"] == 0.5  # expert 1 kept, expert 7 not
    assert by_game["bbbb"]["in_sample_mean"] == 1.0
    assert by_game["aaaa"]["held_out_mean"] < 1.0  # b alone ranks 7 first: it displaces one of a's experts


def test_position_bands():
    rng = np.random.default_rng(0)
    run = _stats([0, 1, 2, 3])
    # band 0 routes to experts 0-3, band 3 to experts 4-7
    for name in ("count", "gate", "gate_norm"):
        pos = np.zeros((L, E, C, 4))
        pos[:, :4, :, 0] = run[name][:, :4]
        pos[:, 4:, :, 3] = run[name][:, :4] + rng.random((L, 4, C))
        run[f"{name}_pos"] = pos
    runs = {"k/aaaa_p0": run}
    # the _pos arrays do not leak into the ordinary aggregate
    assert set(analyze.aggregate(runs, ["k/aaaa_p0"], [0, 1, 2])) == {"count", "gate", "norm", "gate_norm", "prob"}
    bands = analyze.position_bands(runs, ["k/aaaa_p0"], [0, 1, 2], [4])
    first, _, _, last = bands["keep"][4]
    assert first["overlap_with_first"] == 1.0 and first["coverage_own_choice"] == 1.0
    assert last["overlap_with_first"] == 0.0 and last["coverage_first_choice"] == 0.0
    assert last["coverage_own_choice"] == 1.0
