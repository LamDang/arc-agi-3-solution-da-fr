"""serve_bench.py against a fake OpenAI-compatible server with Prometheus metrics."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve_bench  # noqa: E402


def _write_log(path: Path, n: int):
    with open(path, "w") as f:
        for i in range(n):
            messages = [{"role": "system", "content": "s"}, {"role": "user", "content": f"turn {i}"}]
            f.write(json.dumps({"event": "request", "messages": messages, "tools": [], "tool_choice": "auto",
                                "chat_template_kwargs": {"preserve_thinking": True}, "action": i}) + "\n")
            f.write(json.dumps({"event": "response", "messages": messages,
                                "usage": {"prompt_tokens": 100 + i, "completion_tokens": 7 + i}}) + "\n")


def _server():
    state = {"generated": 0, "prompt": 0, "bodies": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["bodies"].append(body)
                state["generated"] += body["max_tokens"]
                state["prompt"] += 100
            out = json.dumps({"usage": {"completion_tokens": body["max_tokens"], "prompt_tokens": 100}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            with lock:
                text = (f'sglang:generation_tokens_total{{model="m"}} {state["generated"]}\n'
                        f'sglang:prompt_tokens_total{{model="m"}} {state["prompt"]}\n'
                        f'sglang:cached_tokens_total{{model="m"}} {state["prompt"] * 0.75}\n'
                        f'sglang:num_running_reqs{{model="m"}} 3\n# HELP x\n')
            out = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def test_sessions_and_measurement(tmp_path):
    _write_log(tmp_path / "ab12-0123abcd_p2_requests.jsonl", 5)
    _write_log(tmp_path / "cd34-0123abcd_p3_requests.jsonl", 4)
    _write_log(tmp_path / "ef56-0123abcd_p0_requests.jsonl", 4)  # pass not selected
    sessions = serve_bench.load_sessions(tmp_path, passes={2, 3})
    assert [s["name"] for s in sessions] == ["ab12_p2", "cd34_p3"]
    assert [t for _, t, _ in sessions[0]["requests"]] == [7, 8, 9, 10, 11]
    assert [p for _, _, p in sessions[0]["requests"]] == [100, 101, 102, 103, 104]
    longest = serve_bench.longest_prompts(sessions, 2)
    assert [p for _, p in longest] == [104, 103]
    body = json.loads(serve_bench.request_body(sessions[0]["requests"][0][0], 7, "flashnext"))
    assert body["max_tokens"] == 7 and body["ignore_eos"] and body["model"] == "flashnext"
    assert body["chat_template_kwargs"] == {"preserve_thinking": True} and "action" not in body

    server, state = _server()
    url = f"http://127.0.0.1:{server.server_port}"
    load = serve_bench.Load(url, "flashnext", sessions)
    result = serve_bench.measure(load, 2, warmup=0.3, seconds=1.0, log=lambda *_: None, sample_every=0.2)
    load.stop()
    server.shutdown()
    assert result["requests_finished"] > 0 and result["request_errors"] == 0
    assert result["generated_tok_s"] > 0 and abs(result["cache_hit"] - 0.75) < 1e-9
    assert result["num_running_reqs_mean"] == 3
    # every session advanced in its own order
    sent = [b["messages"][1]["content"] for b in state["bodies"]]
    assert len(sent) >= 4 and all(s.startswith("turn ") for s in sent)


def test_batch_test(tmp_path):
    _write_log(tmp_path / "ab12-0123abcd_p2_requests.jsonl", 3)
    _write_log(tmp_path / "cd34-0123abcd_p2_requests.jsonl", 3)
    sessions = serve_bench.load_sessions(tmp_path)
    server, state = _server()
    url = f"http://127.0.0.1:{server.server_port}"
    result = serve_bench.batch_test(url, "flashnext", serve_bench.longest_prompts(sessions, 2), 2, max_tokens=5,
                                    log=lambda *_: None, poll=0.05)
    server.shutdown()
    assert result["errors"] == 0 and result["streams"] == 2
    assert sorted(b["max_tokens"] for b in state["bodies"]) == [1, 1, 5, 5]  # prefill, then decode
    assert sorted(b["messages"][1]["content"] for b in state["bodies"]) == ["turn 2"] * 4
    assert result["decode_tok_s"] > 0 and abs(result["cache_hit"] - 0.75) < 1e-9
