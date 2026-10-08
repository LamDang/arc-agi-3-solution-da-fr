"""Read a base-agent run's request logs into one record per model response.

A request log (`<game>_p<pass>_requests.jsonl[.xz]`) has two lines per model
request: `request` (the full `messages` and `tools`) and `response` (the
`reply`, `usage`). A record pairs them and keeps what thinking generation
needs: the context, the call that followed, the teacher's reasoning summary
(OpenAI: `reasoning_details[].summary`) and, for teachers that return it, the
real reasoning (`reply.reasoning`).
"""
import json
import lzma
import re
from dataclasses import dataclass, field
from pathlib import Path

LOG_RE = re.compile(r"^(?P<game>.+)_p(?P<pass>\d+)_requests\.jsonl(\.xz)?$")


@dataclass
class Record:
    game: str                 # game run, e.g. "ft09-0d8bbf25_p0"
    index: int                # 0-based position among the game run's responses
    analysis_step: int
    request_in_turn: int
    messages: list            # the request's messages, as sent
    tools: list
    reply: dict               # assistant message: content, tool_calls, reasoning...
    reasoning_tokens: int     # hidden or visible reasoning tokens billed
    summary: str = ""         # teacher's reasoning summary, "" when none
    real_reasoning: str = ""  # teacher's full reasoning, when the API returned it
    usage: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.game}#{self.index}"

    @property
    def call_id(self) -> str | None:
        calls = self.reply.get("tool_calls") or []
        return calls[0].get("id") if calls else None


def open_log(path: Path):
    path = Path(path)
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def summary_text(reply: dict) -> str:
    """The reasoning summary of an OpenAI Responses reply, sections joined by
    blank lines. Summaries sit in `reasoning_details[].summary` as strings or
    `{"text": ...}` items; other teachers have none."""
    parts = []
    for d in reply.get("reasoning_details") or []:
        for s in d.get("summary") or []:
            t = s.get("text", "") if isinstance(s, dict) else str(s)
            if t.strip():
                parts.append(t.strip())
        if d.get("type") == "reasoning.summary" and isinstance(d.get("summary"), str):
            parts.append(d["summary"].strip())
    return "\n\n".join(parts)


def read_log(path: Path) -> list[Record]:
    m = LOG_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not a request log: {path}")
    game = f"{m['game']}_p{m['pass']}"
    records, pending = [], None
    with open_log(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("event") == "request":
                pending = d
            elif d.get("event") == "response" and pending is not None:
                if "reply" not in d:  # older logs repeat the request instead
                    raise ValueError(f"{path}: response lines carry no reply")
                usage = d.get("usage") or {}
                reply = d["reply"]
                # with hidden reasoning, `reasoning` holds the summary text
                hidden = any(x.get("type") == "reasoning.encrypted"
                             for x in reply.get("reasoning_details") or [])
                records.append(Record(
                    game=game,
                    index=len(records),
                    analysis_step=d.get("analysis_step", pending.get("analysis_step", 0)),
                    request_in_turn=d.get("request_index_within_turn", 1),
                    messages=pending["messages"],
                    tools=pending.get("tools") or [],
                    reply=reply,
                    reasoning_tokens=int((usage.get("completion_tokens_details") or {})
                                         .get("reasoning_tokens") or 0),
                    summary=summary_text(reply),
                    real_reasoning="" if hidden else (
                        reply.get("reasoning") or reply.get("reasoning_content") or ""),
                    usage=usage,
                ))
                pending = None
    return records


def request_logs(run_dir: Path, games: list[str] | None = None) -> list[Path]:
    """The run's per-game request logs, optionally only games whose id starts
    with one of `games`."""
    out = []
    for p in sorted(Path(run_dir).iterdir()):
        m = LOG_RE.match(p.name)
        if not m:
            continue
        if games and not any(m["game"].startswith(g) for g in games):
            continue
        # prefer the plain file when both exist
        if p.suffix == ".xz" and p.with_suffix("").exists():
            continue
        out.append(p)
    return out


def game_outcomes(run_dir: Path) -> dict[str, dict]:
    """Per game run: end state and levels completed, from benchmark.json."""
    path = Path(run_dir) / "benchmark.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out, seen = {}, {}
    for g in data.get("game_runs") or []:
        gid = g["game_id"]
        p = seen[gid] = seen.get(gid, -1) + 1  # passes are listed in order
        out[f"{gid}_p{p}"] = {
            "state": g.get("state"),
            "levels_completed": g.get("levels_completed"),
            "number_of_levels": g.get("number_of_levels"),
        }
    return out
