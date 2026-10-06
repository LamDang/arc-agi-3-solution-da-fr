"""Turn harness request logs into replay samples, one per history stretch.

A game run's log has one `request` line per model call, holding the whole
conversation sent. Between history trims each request extends the previous
one, so the last request of such a stretch (plus the reply it got) contains
every token of the stretch once. Replaying only those keeps the token count
at what the model actually processed, instead of re-reading the shared
prefix once per request.

Older logs repeat the request on `response` lines instead of the reply. The
reply of request i is then recovered from request i+1, where it is the
message after request i's last one. Only a game run's very last reply can be
missing.

Logs written while only one game was playing can land in a run-level
`requests.jsonl` (a harness naming bug, since fixed). Its stretches are
attributed to the game run whose messages they share.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import re
from dataclasses import dataclass, field
from pathlib import Path

GAME_LOG_RE = re.compile(r"^(?P<game>[a-z0-9]{4})-[0-9a-f]{8}_p(?P<pass_>\d+)_requests\.jsonl(\.xz)?$")
RUN_LOG_RE = re.compile(r"^requests\.jsonl(\.xz)?$")


def _open(path: Path):
    return lzma.open(path, "rt") if path.suffix == ".xz" else open(path)


def _hash(message: dict) -> str:
    return hashlib.md5(json.dumps(message, sort_keys=True).encode()).hexdigest()


@dataclass
class Request:
    line: int  # line number of the request in its file
    hashes: list[str]
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    reply: dict | None = None  # newer logs store the reply on the response line


def scan_log(path: Path) -> list[Request]:
    """First pass: message hashes and usage per request, without keeping messages."""
    requests: list[Request] = []
    with _open(path) as fh:
        for number, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event")
            if event == "request":
                requests.append(Request(line=number, hashes=[_hash(m) for m in row["messages"]]))
            elif event == "response" and requests:
                requests[-1].usage = row.get("usage") or {}
                requests[-1].finish_reason = row.get("finish_reason")
                if isinstance(row.get("reply"), dict):
                    requests[-1].reply = row["reply"]
    return requests


def split_stretches(requests: list[Request]) -> list[list[int]]:
    """Group requests into stretches: each request extends the stretch whose
    last request is a prefix of it. Matching against every open stretch, not
    just the previous request, keeps interleaved logs from duplicating tokens."""
    stretches: list[list[int]] = []
    for i, request in enumerate(requests):
        for stretch in reversed(stretches):
            last = requests[stretch[-1]].hashes
            if request.hashes[: len(last)] == last:
                stretch.append(i)
                break
        else:
            stretches.append([i])
    return stretches


def _reply_position(requests: list[Request], i: int) -> int | None:
    """Position of request i's reply inside request i+1, if it is there."""
    if i + 1 >= len(requests):
        return None
    last, following = requests[i].hashes[-1], requests[i + 1].hashes
    for position in range(len(following) - 2, -1, -1):
        if following[position] == last:
            return position + 1
    return None


@dataclass
class Sample:
    """One replay sequence: the last request of a stretch plus its reply."""
    source: str
    game: str
    pass_: int
    stretch: int
    requests: list[int]  # indexes of the stretch's requests in the log
    path: Path
    request_line: int
    reply_from: tuple[int, int] | None  # (line of the next request, message position)
    reply: dict | None
    logged_prompt_tokens: int
    logged_completion_tokens: int

    @property
    def run_key(self) -> str:
        return f"{self.source}/{self.game}_p{self.pass_}"


def samples_from_log(path: Path, source: str, game: str, pass_: int) -> tuple[list[Sample], list[Request]]:
    requests = scan_log(path)
    samples = []
    for s, stretch in enumerate(split_stretches(requests)):
        j = stretch[-1]
        reply_from = None
        reply = requests[j].reply
        if reply is None and (pos := _reply_position(requests, j)) is not None:
            reply_from = (requests[j + 1].line, pos)
        usage = requests[j].usage
        samples.append(Sample(
            source=source, game=game, pass_=pass_, stretch=s, requests=stretch, path=path,
            request_line=requests[j].line, reply_from=reply_from, reply=reply,
            logged_prompt_tokens=int(usage.get("prompt_tokens") or 0),
            logged_completion_tokens=int(usage.get("completion_tokens") or 0),
        ))
    return samples, requests


def collect_samples(log_dir: Path, source: str) -> list[Sample]:
    """All samples of one run directory: per-game logs, then the run-level log
    with each stretch attributed to the game run sharing most of its messages."""
    samples: list[Sample] = []
    run_hashes: dict[tuple[str, int], set[str]] = {}
    run_level: list[Path] = []
    for path in sorted(log_dir.iterdir()):
        if m := GAME_LOG_RE.match(path.name):
            game, pass_ = m["game"], int(m["pass_"])
            found, requests = samples_from_log(path, source, game, pass_)
            samples += found
            run_hashes[(game, pass_)] = {h for r in requests for h in r.hashes[1:]}  # skip the shared system prompt
        elif RUN_LOG_RE.match(path.name):
            run_level.append(path)
    for path in run_level:
        found, requests = samples_from_log(path, source, "unknown", -1)
        for sample in found:
            own = {h for i in sample.requests for h in requests[i].hashes[1:]}
            best = max(run_hashes, key=lambda key: len(own & run_hashes[key]), default=None)
            if best is not None and len(own & run_hashes[best]) >= 2:
                sample.game, sample.pass_ = best
        samples += found
    return samples


def load_messages(sample: Sample) -> tuple[list[dict], list[dict], dict]:
    """Second pass: the sample's messages (with the reply appended), tools and
    chat-template kwargs."""
    wanted = {sample.request_line}
    if sample.reply_from:
        wanted.add(sample.reply_from[0])
    rows = {}
    with _open(sample.path) as fh:
        for number, line in enumerate(fh):
            if number in wanted:
                rows[number] = json.loads(line)
                if len(rows) == len(wanted):
                    break
    request = rows[sample.request_line]
    messages = list(request["messages"])
    if sample.reply is not None:
        messages.append(sample.reply)
    elif sample.reply_from:
        line, position = sample.reply_from
        reply = rows[line]["messages"][position]
        if reply.get("role") == "assistant":
            messages.append(reply)
    return messages, request.get("tools") or [], request.get("chat_template_kwargs") or {}


def load_request(path: Path, line: int) -> dict:
    with _open(path) as fh:
        for number, text in enumerate(fh):
            if number == line:
                return json.loads(text)
    raise IndexError(line)
