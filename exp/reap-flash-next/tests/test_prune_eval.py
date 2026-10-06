"""prune_eval.py end to end on CPU with the tiny model: calibrate on one real
log, evaluate on another game. Needs REAP_TEST_LOG, REAP_TEST_PROCESSOR and
REAP_TEST_EVAL_LOG (a log of a different game)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tiny  # noqa: E402

LOG, PROCESSOR, EVAL_LOG = (os.environ.get(k) for k in ("REAP_TEST_LOG", "REAP_TEST_PROCESSOR", "REAP_TEST_EVAL_LOG"))
pytestmark = pytest.mark.skipif(not all(v and Path(v).exists() for v in (LOG, PROCESSOR, EVAL_LOG)),
                                reason="REAP_TEST_LOG / REAP_TEST_PROCESSOR / REAP_TEST_EVAL_LOG not set")


def test_prune_eval_cli(tmp_path):
    ckpt = tmp_path / "model"
    tiny.write_checkpoint(tiny.reference_model(), ckpt)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                 "processor_config.json"):
        shutil.copy(Path(PROCESSOR) / name, ckpt / name)
    calib, evals = tmp_path / "calib", tmp_path / "eval"
    calib.mkdir(), evals.mkdir()
    shutil.copy(LOG, calib / Path(LOG).name)
    shutil.copy(EVAL_LOG, evals / Path(EVAL_LOG).name)
    calib_game, eval_game = Path(LOG).name[:4], Path(EVAL_LOG).name[:4]
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(HERE.parent / "prune_eval.py"), "--model-dir", str(ckpt),
                    "--traces", f"a={calib}", "--traces", f"b={evals}", "--out", str(out), "--device", "cpu",
                    "--calib-games", calib_game, "--calib-passes", "0", "--calib-max-tokens", "7000",
                    "--eval-games", eval_game, "--eval-passes", "0", "--eval-max-tokens", "7000",
                    "--keep", "12,8,4", "--chunk", "2048"], check=True, timeout=1800)
    report = json.loads((out / "prune_eval.json").read_text())
    (sample,) = report["samples"]
    assert sample["calibrated_on"] == [f"a/{calib_game}_p0"]
    rows = {r["keep"]: r for r in sample["rows"]}
    assert set(rows) == {16, 12, 8, 4}
    assert rows[16]["agree_with_full"] == 1.0 and rows[16]["nll_increase"] == 0.0
    assert rows[4]["agree_with_full"] < 1.0
