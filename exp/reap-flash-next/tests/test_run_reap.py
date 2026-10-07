"""run_reap.py end to end on CPU: the tiny model with the real processor and a
real log, truncated. Needs REAP_TEST_LOG and REAP_TEST_PROCESSOR (see
test_render.py)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tiny  # noqa: E402

LOG = os.environ.get("REAP_TEST_LOG")
PROCESSOR = os.environ.get("REAP_TEST_PROCESSOR")
pytestmark = pytest.mark.skipif(not (LOG and PROCESSOR and Path(LOG).exists() and Path(PROCESSOR).exists()),
                                reason="REAP_TEST_LOG / REAP_TEST_PROCESSOR not set")


def test_cli_saves_consistent_statistics(tmp_path):
    ckpt = tmp_path / "model"
    tiny.write_checkpoint(tiny.reference_model(), ckpt)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                 "processor_config.json"):
        shutil.copy(Path(PROCESSOR) / name, ckpt / name)
    logs = tmp_path / "logs"
    logs.mkdir()
    shutil.copy(LOG, logs / Path(LOG).name)
    out = tmp_path / "out"
    cmd = [sys.executable, str(HERE.parent / "run_reap.py"), "--model-dir", str(ckpt), "--traces", f"test={logs}",
           "--out", str(out), "--device", "cpu", "--max-tokens", "9000", "--chunk", "1024"]
    subprocess.run(cmd, check=True, timeout=1800)

    summary = json.loads((out / "summary.json").read_text())
    (stats,) = list((out / "stats" / "test").glob("*.npz"))
    meta = json.loads(stats.with_suffix(".json").read_text())
    data = np.load(stats)
    config = json.loads((ckpt / "config.json").read_text())["text_config"]
    tokens = sum(s["tokens"] for s in meta["samples"])
    assert tokens == summary["tokens"] and 0 < tokens <= 9000 * len(meta["samples"])
    per_layer = data["count"].sum(axis=(1, 2))
    np.testing.assert_array_equal(per_layer, tokens * config["num_experts_per_tok"])
    np.testing.assert_allclose(data["prob"].sum(axis=(1, 2)), tokens, rtol=1e-4)
    np.testing.assert_allclose(data["gate"].sum(axis=(1, 2)), tokens, rtol=1e-4)
    by_category = data["count"][0].sum(axis=0) / config["num_experts_per_tok"]
    expected = [sum(s["category_tokens"][c] for s in meta["samples"]) for c in meta["categories"]]
    np.testing.assert_array_equal(by_category, expected)
    assert expected[2] > 0 and expected[1] > 0  # images and generated tokens were present
    assert (data["gate_norm"] <= data["norm"] + 1e-6).all()
