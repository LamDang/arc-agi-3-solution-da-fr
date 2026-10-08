import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import pytest

from collect import Jupyter, mirror_once
from common import read_json
from results import load_result, save_result
from test_results import synthetic_run


@pytest.fixture
def jupyter(tmp_path):
    """A real local HTTP contents server exercises GET/PUT and binary transport."""
    root = tmp_path / "server"
    root.mkdir()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def path_for(self):
            return root / unquote(urlsplit(self.path).path.split("/api/contents/", 1)[1])

        def reply(self, status, value):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            import base64
            if self.headers.get("Authorization") != "token local-test-only":
                return self.reply(403, {})
            path = self.path_for()
            if not path.exists():
                return self.reply(404, {})
            if path.is_dir():
                return self.reply(200, {"type": "directory", "content": [
                    dict(name=p.name, path=str(p.relative_to(root)), type="directory" if p.is_dir() else "file")
                    for p in path.iterdir()]})
            self.reply(200, {"type": "file", "format": "base64", "content": base64.b64encode(path.read_bytes()).decode()})

        def do_PUT(self):
            if self.headers.get("Authorization") != "token local-test-only":
                return self.reply(403, {})
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.path_for().write_text(value["content"])
            self.reply(201, {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Jupyter(f"http://127.0.0.1:{server.server_port}", "local-test-only")
    # This test server is on the same process host, not a remote destination.
    client.session.trust_env = False
    try:
        yield root, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_external_mirror_survives_remote_loss_and_is_idempotent(jupyter, tmp_path):
    root, client = jupyter
    remote = root / "sol-nll"
    manifest, identity = synthetic_run(remote)
    sample = manifest["samples"][0]
    save_result(remote, identity, 512, sample, [1, 2, 3], sample["positions"])
    local = tmp_path / "durable"
    assert mirror_once(client, "sol-nll", local) == 1
    assert mirror_once(client, "sol-nll", local) == 0
    assert read_json(remote / "mirror-ack.json")["identity"] == identity
    import shutil
    shutil.rmtree(remote)
    assert load_result(local, identity, 512, sample)[1].tolist() == [1, 2, 3]


def test_mirror_rejects_corrupt_remote_result_without_completion(jupyter, tmp_path):
    root, client = jupyter
    remote = root / "sol-nll"
    manifest, identity = synthetic_run(remote)
    sample = manifest["samples"][0]
    save_result(remote, identity, 512, sample, [1, 2, 3], sample["positions"])
    (remote / "results" / "512" / (sample["sample_id"]+".npz")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        mirror_once(client, "sol-nll", tmp_path / "durable")
    assert not (tmp_path / "durable/results/512" / (sample["sample_id"]+".json")).exists()


def test_remote_cleartext_and_url_tokens_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        Jupyter("http://remote.example", "x")
    with pytest.raises(ValueError, match="URL"):
        Jupyter("https://remote.example?token=x", "x")
