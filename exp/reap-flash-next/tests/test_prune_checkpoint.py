"""A pruned checkpoint computes what the full model computes with the same
experts masked out of the router (the setting prune_eval.py measures)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import reap_model  # noqa: E402
import replay  # noqa: E402
import tiny  # noqa: E402

torch.set_grad_enabled(False)


def _hidden(model, ids):
    enc = {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids)}
    return replay.replay(model, None, enc, torch.zeros_like(ids), chunk_tokens=29, measure=False,
                         collect_hidden=True, device="cpu")["hidden"]


def test_pruned_checkpoint_matches_masked_router(tmp_path):
    ckpt = tmp_path / "model"
    tiny.write_checkpoint(tiny.reference_model(), ckpt)
    (ckpt / "tokenizer.json").write_text("{}")
    config = json.loads((ckpt / "config.json").read_text())["text_config"]
    L, E = config["num_hidden_layers"], config["num_experts"]
    rng = np.random.default_rng(0)
    for game in ("aaaa", "bbbb", "cccc"):
        stats = {f: rng.random((L, E, 3)) for f in ("count", "gate", "norm", "gate_norm", "prob")}
        path = tmp_path / "stats" / "stats" / "src" / f"{game}_p0.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **stats)

    out = tmp_path / "pruned"
    n = E // 2
    subprocess.run([sys.executable, str(HERE.parent / "prune_checkpoint.py"), "--model-dir", str(ckpt),
                    "--stats-dir", str(tmp_path / "stats"), "--keep", str(n), "--out", str(out),
                    "--exclude-games", "cccc"], check=True)
    keep = json.loads((out / "keep.json").read_text())
    assert keep["calibration_runs"] == ["src/aaaa_p0", "src/bbbb_p0"]
    runs = analyze.load(tmp_path / "stats")
    expected = analyze.keep_mask(analyze.expert_scores(
        analyze.aggregate(runs, ["src/aaaa_p0", "src/bbbb_p0"], [0, 1, 2]), "gate_norm"), n)
    mask = np.zeros((L, E), dtype=bool)
    for layer, ids in keep["kept"].items():
        mask[int(layer), ids] = True
    assert (mask == expected).all()
    assert json.loads((out / "config.json").read_text())["text_config"]["num_experts"] == n
    assert (out / "tokenizer.json").is_symlink() and (out / "model-ple-00001-of-00001.safetensors").is_symlink()
    assert not (out / "model-00001-of-00001.safetensors").is_symlink()

    full, _ = reap_model.load_model(ckpt, device="cpu", dtype=torch.float32, log=lambda *_: None)
    pruned, _ = reap_model.load_model(out, device="cpu", dtype=torch.float32, log=lambda *_: None)
    assert pruned.model.language_model.layers[0].mlp.gate.weight.shape[0] == n
    ids = torch.randint(0, 5000, (1, 70))
    reap_model.set_pruning(full, torch.from_numpy(mask))
    masked = _hidden(full, ids)
    reap_model.set_pruning(full, None)
    torch.testing.assert_close(_hidden(pruned, ids), masked, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(_hidden(full, ids), masked, atol=1e-3)


def test_keep_file_reproduces_a_pruning(tmp_path):
    ckpt = tmp_path / "model"
    tiny.write_checkpoint(tiny.reference_model(), ckpt)
    config = json.loads((ckpt / "config.json").read_text())["text_config"]
    L, E = config["num_hidden_layers"], config["num_experts"]
    rng = np.random.default_rng(1)
    kept = {str(layer): sorted(rng.choice(E, E // 2, replace=False).tolist()) for layer in range(L)}
    (tmp_path / "keep.json").write_text(json.dumps({"num_experts": E // 2, "source": "x", "kept": kept,
                                                    "criterion": "gate_norm"}))
    out = tmp_path / "pruned"
    subprocess.run([sys.executable, str(HERE.parent / "prune_checkpoint.py"), "--model-dir", str(ckpt),
                    "--keep-file", str(tmp_path / "keep.json"), "--keep", str(E // 2), "--out", str(out)], check=True)
    written = json.loads((out / "keep.json").read_text())
    assert written["kept"] == kept and written["criterion"] == "gate_norm"


def test_models_extract_matches_prune_checkpoint(tmp_path):
    """models/extract.py with a keep file writes what prune_checkpoint.py writes; --copy leaves no symlinks."""
    ckpt = tmp_path / "model"
    tiny.write_checkpoint(tiny.reference_model(), ckpt)
    (ckpt / "tokenizer.json").write_text("{}")
    config = json.loads((ckpt / "config.json").read_text())["text_config"]
    L, E = config["num_hidden_layers"], config["num_experts"]
    rng = np.random.default_rng(2)
    kept = {str(layer): sorted(rng.choice(E, E // 2, replace=False).tolist()) for layer in range(L)}
    (tmp_path / "keep.json").write_text(json.dumps({"num_experts": E // 2, "source": "x", "kept": kept}))
    subprocess.run([sys.executable, str(HERE.parent / "prune_checkpoint.py"), "--model-dir", str(ckpt),
                    "--keep-file", str(tmp_path / "keep.json"), "--keep", str(E // 2),
                    "--out", str(tmp_path / "ref")], check=True)
    extract = HERE.parents[2] / "models" / "extract.py"
    subprocess.run([sys.executable, str(extract), "--keep-file", str(tmp_path / "keep.json"), "--source", str(ckpt),
                    "--out", str(tmp_path / "copy"), "--copy"], check=True)
    ref, copy = tmp_path / "ref", tmp_path / "copy"
    assert sorted(p.name for p in ref.iterdir()) == sorted(p.name for p in copy.iterdir())
    assert not any(p.is_symlink() for p in copy.iterdir())
    for path in ref.iterdir():
        if path.name != "keep.json":
            assert path.read_bytes() == (copy / path.name).read_bytes(), path.name
    assert json.loads((copy / "keep.json").read_text())["kept"] == kept

    bad = {"num_experts": 2, "source": "x", "kept": {str(layer): [0, E] for layer in range(L)}}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    result = subprocess.run([sys.executable, str(extract), "--keep-file", str(tmp_path / "bad.json"),
                             "--source", str(ckpt), "--out", str(tmp_path / "bad")], capture_output=True, text=True)
    assert result.returncode != 0 and "does not fit" in result.stderr
