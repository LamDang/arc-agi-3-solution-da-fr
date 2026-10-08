import copy

import numpy as np
import pytest

from prepare import clean_maps, sample_panel
from common import write_json


def fixture():
    folds = {"folds": [{"fold": 0, "games": list("abcde"), "game_ids": [g+"-v1" for g in "abcde"]},
                       {"fold": 1, "games": ["train"], "game_ids": ["train-v1"]}]}
    index = [dict(game=g, request_index=i, line=100*j+i, context_tokens=(i+1)*1000,
                  output_tokens=5, level=i//3+1) for j, g in enumerate(folds["folds"][0]["game_ids"])
             for i in range(19)]
    index += [dict(game="train-v1", request_index=0, line=999, context_tokens=5)]
    return index, folds


def test_exactly_30_two_per_stratum_no_train_leakage():
    index, folds = fixture()
    panel, strata, valid, train = sample_panel(index, folds)
    assert len(panel) == len({r["sample_id"] for r in panel}) == 30
    assert train == ["train-v1"]
    assert all(r["game"] in valid for r in panel)
    for s in strata:
        rows = [r for r in panel if (r["game"], r["stratum"]) == (s["game"], s["stratum"])]
        assert len(rows) == 2
        assert sum(r["weight"] for r in rows) == s["population"]
    assert sample_panel(list(reversed(index)), folds)[0] == panel
    assert sample_panel(index, folds, seed=42)[0] != panel


def test_bad_or_small_population_fails_instead_of_resampling():
    index, folds = fixture()
    index[0]["game"] = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        sample_panel(index, folds)
    index, folds = fixture()
    with pytest.raises(ValueError, match="two rows"):
        sample_panel([r for r in index if r["request_index"] < 5], folds)
    index, folds = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        sample_panel(index+[copy.deepcopy(index[0])], folds)


def test_maps_exclude_entire_validation_fold_and_are_nested(tmp_path):
    _, folds = fixture()
    root = tmp_path / "stats" / "source"
    root.mkdir(parents=True)
    train = np.zeros((2, 16, 3))
    train[:, :, 0] = np.arange(16)
    np.savez(root / "train_p0.npz", gate_norm=train)
    write_json(root / "train_p0.json", dict(run="source/train_p0",
        categories=["context", "generated", "image"], samples=[dict(tokens=10)],
        model_dir="/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"))
    leaked = np.zeros_like(train)
    leaked[:, 0, :] = 1e10
    for g in "abcde":
        np.savez(root / f"{g}_p0.npz", gate_norm=leaked)
    maps = clean_maps(tmp_path, folds, counts=(16, 12, 8))
    assert maps["8"]["kept"]["0"] == list(range(8, 16))
    assert set(maps["8"]["kept"]["1"]) <= set(maps["12"]["kept"]["1"])
    assert len(maps["8"]["excluded_validation_runs"]) == 5
    assert list(maps["8"]["calibration_runs"]) == ["stats/source/train_p0.npz"]
    assert maps["8"]["calibration_provenance"]["source/train_p0"]["game"] == "train"
    write_json(root / "train_p0.json", dict(run="source/train_p0", categories=["image", "context", "generated"]))
    with pytest.raises(ValueError, match="provenance"):
        clean_maps(tmp_path, folds, counts=(16, 12, 8))
