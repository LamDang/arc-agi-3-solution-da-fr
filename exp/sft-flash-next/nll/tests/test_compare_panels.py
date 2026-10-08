import copy

import numpy as np
import pytest

from common import digest, read_json, write_json
from compare_panels import compare, loss_metrics, validate_pair
from results import bind_run, save_result
from test_results import synthetic_run


def panels(tmp_path):
    source, variant = tmp_path / "source", tmp_path / "variant"
    a, _ = synthetic_run(source)
    for row in a["samples"]:
        row["images"] = 0
    a.update(processor_files={}, maps={}, fold_sha256="fold", training_games=[])
    a["manifest_sha256"] = digest({k: v for k, v in a.items() if k != "manifest_sha256"})
    write_json(source / "manifest.json", a)
    b = copy.deepcopy(a)
    for row in b["samples"]:
        row.update(target_tokens=4, labels=[0, 0, 1, 4], target_ids=[101, 101, 102, 103],
                   positions=[40, 41, 42, 43], target_sha256=digest([101, 101, 102, 103]))
    b["variant"] = dict(source_manifest_sha256=a["manifest_sha256"],
                        tool_contract_audit=[dict(sample_id=r["sample_id"], prompts_images_code_unchanged=True)
                                             for r in a["samples"]])
    b["manifest_sha256"] = digest({k: v for k, v in b.items() if k != "manifest_sha256"})
    write_json(variant / "manifest.json", b)
    # Fixture run configs are intentionally independent of the real model.
    ia = read_json(source / "run.json")["identity"]
    ib = bind_run(variant, {"fixture": True, "variant": True})
    for count in (512, 256):
        for r in a["samples"]:
            save_result(source, ia, count, r, np.array([2., .4, .1]), r["positions"])
        for r in b["samples"]:
            save_result(variant, ib, count, r, np.array([1., 1., .3, .1]) * (1.07 if count == 256 else 1), r["positions"])
    return source, variant, a, b


def test_comparison_requires_complete_pairs_and_aligns_only_python_tokens(tmp_path):
    source, variant, _, _ = panels(tmp_path)
    out = tmp_path / "comparison"
    result = compare(source, variant, out)
    assert result["reference_panel"] == "genthink"
    assert result["pruning_gate"]["status"] == "scan_required"
    data = read_json(out / "comparison.json")
    assert len(data["requests"]) == 60
    row = data["requests"][0]
    assert row["source"]["thinking_tokens"] == 1
    assert row["genthink"]["thinking_tokens"] == 2
    with np.load(next((out / "code-token-diffs/512").glob("*.npz"))) as arr:
        assert arr["token_ids"].tolist() == [102]
        assert arr["delta"].tolist() == pytest.approx([-.1])
    missing = next((variant / "results/256").glob("*.json"))
    missing.unlink()
    with pytest.raises(ValueError, match="Incomplete panel"):
        compare(source, variant, tmp_path / "partial")
    assert not (tmp_path / "partial/decision.json").exists()


@pytest.mark.parametrize("change", ["weight", "code", "maps", "audit"])
def test_context_sampling_and_code_mismatch_rejected(tmp_path, change):
    _, _, a, b = panels(tmp_path)
    if change == "weight":
        b["samples"][0]["weight"] += 1
    elif change == "code":
        b["samples"][0]["target_ids"][2] = 999
    elif change == "maps":
        b["maps"] = {"256": "different"}
    else:
        b["variant"]["tool_contract_audit"][0]["prompts_images_code_unchanged"] = False
    b["manifest_sha256"] = digest({k: v for k, v in b.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError):
        validate_pair(a, b)


def test_category_means_do_not_mix_changed_thinking_lengths():
    metrics = loss_metrics({"labels": [0, 0, 1, 4]}, [2., 4., .3, .1])
    assert metrics["thinking_nll"] == 3.
    assert metrics["output_nll"] == pytest.approx(.2)
    assert metrics["mean_nll"] == pytest.approx(1.6)


def test_categorywise_reference_ignores_changed_mixture_and_gates_each_component(tmp_path):
    source, variant, _, b = panels(tmp_path)
    identity = read_json(variant / "run.json")["identity"]
    for row in b["samples"]:
        ids = [101]*20 + [102, 103]
        row.update(target_tokens=22, labels=[0]*20 + [1, 4], target_ids=ids,
                   positions=list(range(40, 62)), target_sha256=digest(ids))
        for count in (512, 256):
            # Thinking improves but the identical Python code worsens by 6%.
            values = [1.]*20 + [.3, .1] if count == 512 else [.99]*20 + [.318, .1]
            save_result(variant, identity, count, row, values, row["positions"])
    b["manifest_sha256"] = digest({k: v for k, v in b.items() if k != "manifest_sha256"})
    write_json(variant / "manifest.json", b)
    out = tmp_path / "categorywise"
    decision = compare(source, variant, out, reference_metric="categorywise")
    data = read_json(out / "comparison.json")
    assert data["summaries"]["genthink"]["512"]["primary"] > data["summaries"]["source"]["512"]["primary"]
    assert decision["reference_panel"] == "genthink"
    assert decision["pruning_gate"]["components"]["thinking"]["status"] == "stop_at_256"
    assert decision["pruning_gate"]["components"]["tool_code"]["status"] == "scan_required"
    assert decision["pruning_gate"]["status"] == "scan_required"
