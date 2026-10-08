import ast
import json
import zipfile

from build_setup import build, notebook


def test_notebook_has_no_automatic_gpu_launch_and_cells_compile():
    nb = notebook()
    assert nb["metadata"]["kaggle"]["isGpuEnabled"] is False
    assert nb["metadata"]["kaggle"]["accelerator"] == "none"
    text = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            text += source
    assert "START_GPU_RUN = False" in text
    assert "--preflight-only" in text
    assert "subprocess.Popen" in text


def test_code_only_package_cannot_claim_real_data_ready(tmp_path):
    archive = build(tmp_path / "build")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("setup-manifest.json"))
        assert manifest["has_prepared_real_data"] is False
        assert manifest["starts_gpu"] is False
        assert manifest["requests"] == 30 and manifest["jobs"] == 150
        import hashlib
        for name, checksum in manifest["sha256"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == checksum
        assert "reap/replay.py" in z.namelist()
        assert "nll/collect.py" in z.namelist()
