"""Throughput of a running SGLang server on ARC-AGI-3 requests from the
harness's logs. Standard library only.

batch_test (what selects the serving config): n requests at once, each the
longest logged prompt of a different game run (about 110K tokens, close to
the harness's context limit), each generating a fixed number of tokens. It
answers whether n full-length requests fit (all n running, no retractions,
peak KV use) and the decode throughput with all n running.

Load/measure (below) replay whole games instead, for cache behaviour (not used
to pick the config):

    python serve_bench.py --url http://127.0.0.1:8001 --model flashnext --logs DIR \\
        [--passes 2,3] [--concurrency 10,16,20] [--warmup 180] [--measure 480]

Each logged game run is a session: its requests in order, each with the
logged messages (frames as images, tools, earlier replies) and max_tokens set
to the logged completion length with ignore_eos, so the server generates as
many tokens as the real run did. Sessions start at a random request, so
contexts are already long, as in steady state. `concurrency` workers take
the session at the head of a queue, send its next request and put it back at
the tail, like the harness rotating games through its active streams, so
prefixes of many games compete for the cache.

Rates come from the server's Prometheus counters (--enable-metrics) over the
measurement window: generated and prompt tokens, cached prompt tokens,
retracted requests; gauges (running and queued requests, KV usage,
speculative accept length) are sampled every 10 s.

One difference from the real harness: a session's next request carries the
logged reply, not the one just generated, so that reply is prefilled instead
of found in the cache. Prefill is somewhat overstated, equally for every
setting.
"""
from __future__ import annotations

import argparse
import json
import lzma
import random
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

LOG_RE = re.compile(r"^(?P<game>[a-z0-9]{4})-[0-9a-f]{8}_p(?P<pass_>\d+)_requests\.jsonl(\.xz)?$")
COUNTERS = ("sglang:generation_tokens_total", "sglang:prompt_tokens_total", "sglang:cached_tokens_total",
            "sglang:num_retracted_requests_total")
GAUGES = ("sglang:num_running_reqs", "sglang:num_queue_reqs", "sglang:token_usage", "sglang:spec_accept_length")
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}  # the harness's settings


def _open(path: Path):
    return lzma.open(path, "rt") if path.suffix == ".xz" else open(path)


def load_sessions(log_dir: Path, games=None, passes=None) -> list[dict]:
    """[{name, requests: [(request line, completion tokens, prompt tokens)]}] from *_pN_requests.jsonl logs."""
    sessions = []
    for path in sorted(Path(log_dir).iterdir()):
        m = LOG_RE.match(path.name)
        if not m or (games and m["game"] not in games) or (passes and int(m["pass_"]) not in passes):
            continue
        requests, pending = [], None
        with _open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") == "request":
                    pending = line
                elif row.get("event") == "response" and pending is not None:
                    usage = row.get("usage") or {}
                    tokens = int(usage.get("completion_tokens") or 0)
                    if tokens > 0:
                        requests.append((pending, tokens, int(usage.get("prompt_tokens") or 0)))
                    pending = None
        if requests:
            sessions.append({"name": f"{m['game']}_p{m['pass_']}", "requests": requests})
    return sessions


def request_body(line: str, max_tokens: int, model: str) -> bytes:
    row = json.loads(line)
    body = {k: row[k] for k in ("messages", "tools", "tool_choice", "chat_template_kwargs") if k in row}
    body.update(model=model, max_tokens=max_tokens, ignore_eos=True, stream=False, **SAMPLING)
    return json.dumps(body).encode()


def scrape(url: str) -> dict[str, float]:
    """Prometheus text -> {metric name: sum over label sets}."""
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as r:
        text = r.read().decode()
    values = defaultdict(float)
    for line in text.splitlines():
        if line.startswith("sglang:"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            try:
                values[name] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return dict(values)


class Load:
    """Workers replaying sessions round-robin; `target` of them are active."""

    def __init__(self, url: str, model: str, sessions: list[dict], seed: int = 0, timeout: float = 1800):
        rng = random.Random(seed)
        self.url, self.model, self.timeout = url, model, timeout
        self.queue = deque({**s, "next": rng.randrange(len(s["requests"]))} for s in sessions)
        self.lock = threading.Lock()
        self.target, self.stopping, self.workers = 0, False, []
        self.done: list[tuple[float, float, int, int]] = []  # (finished, seconds, completion, status)

    def _take(self):
        with self.lock:
            session = self.queue.popleft()
            line, tokens, _ = session["requests"][session["next"]]
            session["next"] = (session["next"] + 1) % len(session["requests"])
            return session, line, tokens

    def _worker(self, index: int):
        while not self.stopping:
            if index >= self.target:
                time.sleep(0.5)
                continue
            session, line, tokens = self._take()
            started, status, completion = time.time(), 0, 0
            try:
                req = urllib.request.Request(f"{self.url}/v1/chat/completions",
                                             data=request_body(line, tokens, self.model),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    completion = int(json.loads(r.read())["usage"]["completion_tokens"])
                    status = r.status
            except Exception as exc:  # noqa: BLE001 - counted, the session moves on
                status = getattr(exc, "code", -1)
            with self.lock:
                self.queue.append(session)
                self.done.append((time.time(), time.time() - started, completion, status))

    def set_concurrency(self, n: int):
        self.target = n
        while len(self.workers) < n:
            t = threading.Thread(target=self._worker, args=(len(self.workers),), daemon=True)
            self.workers.append(t)
            t.start()

    def stop(self):
        self.stopping = True
        self.target = 0


def measure(load: Load, concurrency: int, warmup: float, seconds: float, log=print, sample_every: float = 10) -> dict:
    load.set_concurrency(concurrency)
    log(f"[bench] concurrency {concurrency}: warm-up {warmup:.0f}s")
    time.sleep(warmup)
    start, t0 = scrape(load.url), time.time()
    samples = defaultdict(list)
    while time.time() - t0 < seconds:
        time.sleep(min(sample_every, max(0.0, seconds - (time.time() - t0))))
        now = scrape(load.url)
        for g in GAUGES:
            if g in now:
                samples[g].append(now[g])
    end, elapsed = scrape(load.url), time.time() - t0
    delta = {c: end.get(c, 0.0) - start.get(c, 0.0) for c in COUNTERS}
    with load.lock:
        window = [d for d in load.done if t0 <= d[0] <= t0 + elapsed]
    prompt, cached = delta["sglang:prompt_tokens_total"], delta["sglang:cached_tokens_total"]
    result = {
        "concurrency": concurrency,
        "seconds": elapsed,
        "generated_tok_s": delta["sglang:generation_tokens_total"] / elapsed,
        "prompt_tok_s": prompt / elapsed,
        "uncached_prompt_tok_s": (prompt - cached) / elapsed,
        "cache_hit": cached / prompt if prompt else None,
        "retracted_requests": delta["sglang:num_retracted_requests_total"],
        "requests_finished": len(window),
        "request_errors": sum(1 for d in window if d[3] != 200),
        "mean_request_seconds": sum(d[1] for d in window) / len(window) if window else None,
        **{g.split(":")[1] + "_mean": (sum(v) / len(v) if v else None) for g, v in samples.items()},
    }
    log("[bench] " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}))
    return result


def longest_prompts(sessions: list[dict], n: int) -> list[tuple[str, int]]:
    """The longest logged request of each session, longest first: (line, prompt tokens)."""
    best = [max(s["requests"], key=lambda r: r[2]) for s in sessions]
    best.sort(key=lambda r: -r[2])
    return [(line, prompt) for line, _, prompt in best[:n]]


def _send_all(url: str, model: str, lines: list[str], max_tokens: int, timeout: float) -> list[int]:
    statuses = []

    def send(line):
        status = -1
        try:
            req = urllib.request.Request(f"{url}/v1/chat/completions", data=request_body(line, max_tokens, model),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                json.loads(r.read())
                status = r.status
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "code", -1)
        statuses.append(status)

    threads = [threading.Thread(target=send, args=(line,), daemon=True) for line in lines]
    for t in threads:
        t.start()
    return threads, statuses


def batch_test(url: str, model: str, prompts: list[tuple[str, int]], n: int, max_tokens: int = 4096,
               log=print, poll: float = 2.0, timeout: float = 3600) -> dict:
    """n requests at once, each a long logged prompt of a different game.
    Phase 1 prefills them (1 output token) so they sit in the cache; phase 2
    sends them again with max_tokens each: they start decoding together from
    the cache, so decode throughput = generated tokens / phase-2 time. Phase 2
    finding its prompts in the cache (cache_hit ~1), with all n running and
    no retractions, means n full-length contexts fit."""
    assert len(prompts) >= n, f"only {len(prompts)} prompts for {n} streams"
    lines = [line for line, _ in prompts[:n]]
    t0 = time.time()
    threads, statuses = _send_all(url, model, lines, 1, timeout)
    for t in threads:
        t.join(timeout)
    prefill_seconds, prefill_errors = time.time() - t0, sum(1 for s in statuses if s != 200)

    before, t1 = scrape(url), time.time()
    threads, statuses = _send_all(url, model, lines, max_tokens, timeout)
    trace = []  # (running, token_usage, accept length)
    while any(t.is_alive() for t in threads) and time.time() - t1 < timeout:
        m = scrape(url)
        trace.append((m.get("sglang:num_running_reqs", 0.0), m.get("sglang:token_usage", 0.0),
                      m.get("sglang:spec_accept_length")))
        time.sleep(poll)
    for t in threads:
        t.join(timeout=5)
    seconds = time.time() - t1
    after = scrape(url)
    delta = {c: after.get(c, 0.0) - before.get(c, 0.0) for c in COUNTERS}
    prompt = delta["sglang:prompt_tokens_total"]
    accept = [a for _, _, a in trace if a]
    result = {
        "streams": n,
        "shortest_prompt": prompts[n - 1][1],
        "prefill_seconds": prefill_seconds,
        "decode_seconds": seconds,
        "decode_tok_s": delta["sglang:generation_tokens_total"] / seconds,
        "max_running": max((r for r, _, _ in trace), default=0),
        "peak_kv_usage": max((u for _, u, _ in trace), default=None),
        "cache_hit": delta["sglang:cached_tokens_total"] / prompt if prompt else None,
        "retracted_requests": delta["sglang:num_retracted_requests_total"],
        "spec_accept_length": sum(accept) / len(accept) if accept else None,
        "errors": prefill_errors + sum(1 for s in statuses if s != 200),
    }
    result["per_stream_tok_s"] = result["decode_tok_s"] / n
    log("[batch] " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="flashnext")
    parser.add_argument("--logs", required=True, help="directory of *_pN_requests.jsonl logs")
    parser.add_argument("--games", default="")
    parser.add_argument("--passes", default="2,3")
    parser.add_argument("--concurrency", default="10")
    parser.add_argument("--warmup", type=float, default=180)
    parser.add_argument("--measure", type=float, default=480)
    parser.add_argument("--out")
    args = parser.parse_args()
    games = set(filter(None, args.games.split(",")))
    passes = {int(p) for p in args.passes.split(",") if p}
    sessions = load_sessions(Path(args.logs), games, passes)
    print(f"[bench] {len(sessions)} sessions, {sum(len(s['requests']) for s in sessions)} requests", flush=True)
    load = Load(args.url, args.model, sessions)
    results = [measure(load, int(c), args.warmup, args.measure, log=lambda m: print(m, flush=True))
               for c in args.concurrency.split(",")]
    load.stop()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
