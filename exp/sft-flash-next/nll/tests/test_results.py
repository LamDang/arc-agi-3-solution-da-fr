
import numpy as np
import pytest

from common import COUNTS, digest, read_json, write_json
from report import generate, weighted_summary
from results import bind_run, load_result, paths, save_result
from run import fits_deadline, mirror_fresh, validate_bundle
from test_sampling import fixture
from prepare import sample_panel


def row():
    return dict(sample_id="game-v1-r2", target_tokens=3, positions=[40, 41, 42],
                target_sha256=digest([101, 102, 103]), target_ids=[101, 102, 103],
                labels=[0, 1, 4], weight=4, game="game-v1", target_text="abc",
                offsets=[(0, 1), (1, 2), (2, 3)], prompt_tokens=40, level=2)


def test_atomic_recovery_and_changed_identity(tmp_path, monkeypatch):
    import results
    r = row()
    identity = bind_run(tmp_path, {"manifest": "abc", "chunk": 8})
    writer = results.write_json
    def crash(*_):
        raise OSError("simulated failure after array write")
    monkeypatch.setattr(results, "write_json", crash)
    with pytest.raises(OSError):
        save_result(tmp_path, identity, 512, r, [1, 2, 3], r["positions"])
    assert load_result(tmp_path, identity, 512, r) is None
    monkeypatch.setattr(results, "write_json", writer)
    save_result(tmp_path, identity, 512, r, [1, 2, 3], r["positions"])
    assert load_result(tmp_path, identity, 512, r)[1].tolist() == [1, 2, 3]
    assert bind_run(tmp_path, {"manifest": "abc", "chunk": 8}) == identity
    with pytest.raises(ValueError, match="configuration"):
        bind_run(tmp_path, {"manifest": "abc", "chunk": 4})
    with pytest.raises(ValueError, match="identity"):
        load_result(tmp_path, "different", 512, r)


def test_corruption_wrong_positions_and_nonfinite_rejected(tmp_path):
    r = row()
    for bad in ([1, float("nan"), 3], [-1, 2, 3], [1, 2]):
        with pytest.raises(ValueError):
            save_result(tmp_path, "run", 512, r, bad, r["positions"])
    with pytest.raises(ValueError, match="positions"):
        save_result(tmp_path, "run", 512, r, [1, 2, 3], [39, 40, 41])
    save_result(tmp_path, "run", 512, r, [1, 2, 3], r["positions"])
    paths(tmp_path, 512, r["sample_id"])[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        load_result(tmp_path, "run", 512, r)


def test_weighted_request_and_token_metrics_are_distinct():
    a = row()
    b = dict(row(), sample_id="second", weight=1, target_tokens=1, labels=[1])
    s = weighted_summary([a, b], {a["sample_id"]: np.array([1., 2., 3.]), b["sample_id"]: np.array([10.])})
    assert s["primary"] == pytest.approx((4*2+10)/5)
    assert s["token_nll"] == pytest.approx((4*6+10)/(4*3+1))
    assert s["categories"]["assistant_text"]["nll"] is None
    assert sum(x["weighted_sum"] for x in s["categories"].values()) == 34


def synthetic_run(root):
    index, folds = fixture()
    samples, strata, valid, _ = sample_panel(index, folds)
    samples = [dict(row(), **{k: v for k, v in s.items() if k not in ("target_tokens", "weight")},
                    weight=s["weight"]) for s in samples]
    m = dict(samples=samples, counts=list(COUNTS), validation_games=valid, real_data=False,
             strata=strata, fixture=True)
    m["manifest_sha256"] = digest(m)
    write_json(root / "manifest.json", m)
    identity = bind_run(root, {"manifest_sha256": m["manifest_sha256"], "fixture": True})
    return m, identity


def test_paired_reporting_and_complete_30_request_resume(tmp_path):
    m, identity = synthetic_run(tmp_path)
    for r in m["samples"]:
        save_result(tmp_path, identity, 512, r, [1, 2, 3], r["positions"])
    first = generate(tmp_path)
    assert first["selected_experts"] is None and first["matched_requests"] == 0
    for count in COUNTS[1:]:
        for r in m["samples"]:
            penalty = .02 if count >= 320 else .2
            save_result(tmp_path, identity, count, r, np.array([1, 2, 3])+penalty, r["positions"])
    done = generate(tmp_path)
    assert done["matched_requests"] == 30 and done["selected_experts"] == 320
    assert done["status"] == "nll_candidate_selected"
    assert len(read_json(tmp_path / "comparison.json")["summaries"]) == 5
    assert (tmp_path / "comparison.html").exists()
    assert generate(tmp_path) == done  # idempotent after reconnect
    with pytest.raises(ValueError, match="real-data"):
        validate_bundle(tmp_path)  # synthetic artifacts cannot unlock a GPU launch


def test_quota_admission_accounts_for_tail_and_persistence_margin():
    assert fits_deadline(200, 0, 1000, 100)
    assert not fits_deadline(130, 0, 1000, 100)
    assert not fits_deadline(100, 0, 1000, 100)


def test_collector_ack_requires_matching_run_and_recent_verification(tmp_path):
    assert not mirror_fresh(tmp_path, "run", 300, now=1000)
    write_json(tmp_path / "mirror-ack.json", dict(identity="run", last_verified_unix=900))
    assert mirror_fresh(tmp_path, "run", 300, now=1000)
    assert not mirror_fresh(tmp_path, "other", 300, now=1000)
    assert not mirror_fresh(tmp_path, "run", 300, now=1200)
    assert not mirror_fresh(tmp_path, "run", 300, now=899)
