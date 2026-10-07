"""Stretch splitting, reply recovery and run-level attribution on synthetic logs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import traces  # noqa: E402

SYSTEM = {"role": "system", "content": "rules"}


def _turns(tag: str, n: int) -> list[list[dict]]:
    """Message lists of n consecutive requests of one game: each request adds
    the previous reply and a tool result."""
    history = [SYSTEM, {"role": "user", "content": f"{tag} start"}]
    requests = []
    for i in range(n):
        requests.append(list(history))
        history += [{"role": "assistant", "content": f"{tag} reply {i}"},
                    {"role": "tool", "content": f"{tag} result {i}"}]
    return requests


def _write(path: Path, requests: list[list[dict]], old_format: bool = True):
    with open(path, "w") as fh:
        for i, messages in enumerate(requests):
            fh.write(json.dumps({"event": "request", "messages": messages, "tools": []}) + "\n")
            response = {"event": "response", "usage": {"prompt_tokens": 100 + i, "completion_tokens": 7}}
            if old_format:
                response["messages"] = messages
            fh.write(json.dumps(response) + "\n")


def test_stretches_and_reply_recovery(tmp_path):
    requests = _turns("g", 5)
    trimmed = [SYSTEM] + requests[4][3:]  # history trim: drop the oldest turns
    trimmed_next = trimmed + [{"role": "assistant", "content": "g reply 4"}, {"role": "tool", "content": "g result 4"}]
    _write(tmp_path / "ab12-0123abcd_p0_requests.jsonl", requests + [trimmed_next])
    samples = traces.collect_samples(tmp_path, "test")
    assert [s.requests for s in samples] == [[0, 1, 2, 3, 4], [5]]
    first, last = samples
    messages, _, _ = traces.load_messages(first)
    # the last request of the stretch plus its reply, recovered across the trim
    assert messages == requests[4] + [{"role": "assistant", "content": "g reply 4"}]
    assert first.logged_prompt_tokens == 104
    assert (first.game, first.pass_) == ("ab12", 0)
    messages, _, _ = traces.load_messages(last)  # the run's final reply is not in the log
    assert messages == trimmed_next


def test_run_level_log_is_attributed_by_shared_messages(tmp_path):
    a, b = _turns("a", 4), _turns("b", 4)
    _write(tmp_path / "aaaa-00000000_p0_requests.jsonl", a[:2])
    _write(tmp_path / "bbbb-11111111_p1_requests.jsonl", b[:3])
    _write(tmp_path / "requests.jsonl", [a[2], a[3], b[3]])  # tails of both games, logged run-level
    samples = traces.collect_samples(tmp_path, "test")
    run_level = [s for s in samples if s.path.name == "requests.jsonl"]
    assert [(s.game, s.pass_, s.requests) for s in run_level] == [("aaaa", 0, [0, 1]), ("bbbb", 1, [2])]


def test_new_format_reply_is_used(tmp_path):
    requests = _turns("n", 2)
    path = tmp_path / "cccc-22222222_p0_requests.jsonl"
    with open(path, "w") as fh:
        for i, messages in enumerate(requests):
            fh.write(json.dumps({"event": "request", "messages": messages, "tools": []}) + "\n")
            fh.write(json.dumps({"event": "response", "reply": {"role": "assistant", "content": f"n reply {i}"},
                                 "usage": {"prompt_tokens": 10}}) + "\n")
    (sample,) = traces.collect_samples(tmp_path, "test")
    messages, _, _ = traces.load_messages(sample)
    assert messages[-1] == {"role": "assistant", "content": "n reply 1"}
