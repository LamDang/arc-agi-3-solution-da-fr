"""Break runs' model tokens down: thinking vs tool calls, and what the thinking is about.

    uv run --no-sync python scripts/token_breakdown.py runs/control runs/engine-code \\
        --out experiments/engine-code --label

Reads each game run's request log (needs ANALYZER_SAVE_REQUEST_LOGS=true). For
every model response it takes the provider's usage (prompt, cached, completion
and reasoning tokens, cost) and the response itself: the reasoning text and the
tool calls, which the next request carries as an assistant message. The few
responses no later request carries take their reasoning from the transcript.
Thinking tokens are `reasoning_tokens`; tool-call tokens are the rest of the
completion.

A tool call "reads code" when its code uses game_code_files, game_code() or
read_game_code(), the ARC3_GAME_CODE_DIR functions.

With --label, the reasoning is cut into excerpts of about 600 characters and
Claude Haiku 4.5 (through OpenRouter, OPENROUTER_API_KEY) labels each one:
mechanics, planning, tooling or other, and whether it discusses the game's
source code. Labels are cached in <out>/labels/<run>.jsonl, so a rerun only
labels what is new. An excerpt's tokens are its response's reasoning tokens,
shared by characters.

Writes to --out: responses.csv (one row per response), games.csv (per run and
game), levels.csv (per run, game and level, with actions and level scores),
summary.json, tokens.png (output tokens by part) and tokens_by_game.png.
--names sets the runs' names in the charts.

Input tokens spent on code text are estimated per request as its prompt tokens
times the share of its text characters that are code-reading tool results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

from inference.utils.run_artifacts import existing_log, open_log

CODE_READ = re.compile(r"\b(read_game_code|game_code|game_code_files)\b")
LEVEL = re.compile(r"Current state: step \d+, level (\d+)")
TOPICS = ("mechanics", "planning", "tooling", "other")
EXCERPT_CHARS = 600
LABEL_MODEL = "anthropic/claude-haiku-4.5"
LABEL_BATCH = 40
LABEL_PROMPT = """\
You label excerpts of an AI agent's private reasoning while it plays a grid puzzle \
video game (ARC-AGI-3). The agent sees the screen as a grid of colored cells and acts \
through Python code. Give each excerpt one topic:

- mechanics: working out how the game works: what the objects are, what actions do, \
the rules, win and lose conditions, interpreting what an action changed, forming or \
testing hypotheses about the rules. Reading or interpreting the game's source code to \
learn its rules, and matching the code to what is on screen, are mechanics too.
- planning: deciding what to do with rules it takes as known: choosing goals, moves, \
routes or click targets, computing, simulating or checking a sequence of actions, \
counting the steps or moves it needs, deciding what to try next or which probe to run.
- tooling: the agent's own Python and the harness: writing, fixing or debugging its \
snippets, printing or formatting output, tool errors and limits.
- other: anything else, such as restating instructions, bookkeeping of notes, filler.

Also set source_code to true when the excerpt discusses the game's source code (its \
classes, functions, variables, or sprite and level definitions as written in code), \
otherwise false.

Answer with JSON only: {"labels": [{"id": <id>, "topic": "<topic>", "source_code": \
<true|false>}, ...]}, one entry per excerpt, in the order given."""


@dataclass
class Response:
    game: str
    index: int
    analysis_step: int
    action: int
    level: int
    finish_reason: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost: float
    cost_input: float = 0.0
    cost_output: float = 0.0
    reasoning: str = ""
    matched: bool = False
    reasoning_from_transcript: bool = False
    call_ids: list[str] = field(default_factory=list)
    call_chars: int = 0
    code_read_calls: int = 0
    code_read_call_chars: int = 0
    tool_output_chars: int = 0
    code_output_chars: int = 0
    context_code_chars: int = 0
    context_text_chars: int = 0

    @property
    def tool_call_tokens(self) -> int:
        return self.completion_tokens - self.reasoning_tokens


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


def _reasoning(message: dict[str, Any]) -> str:
    return str(message.get("reasoning") or message.get("reasoning_content") or "")


def _message_key(message: dict[str, Any]) -> str:
    ids = [str(call.get("id")) for call in message.get("tool_calls") or []]
    if ids:
        return "ids:" + ",".join(ids)
    body = _reasoning(message) + "\x00" + _text(message.get("content"))
    return "sha:" + hashlib.sha1(body.encode("utf-8")).hexdigest()


def _records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with open_log(path) as lines:
            for line in lines:
                yield json.loads(line)


def load_responses(logs: list[Path], game: str) -> list[Response]:
    """The responses in a game run's logs, each with the assistant message it produced.

    Newer logs store the message on the response line (`reply`). In older ones
    it first appears in a later request: new messages there go to the most
    recent responses still waiting for one, so a message the harness never
    re-sent leaves only its own response without text.
    """
    responses: list[Response] = []
    waiting: list[Response] = []
    seen: set[str] = set()
    call_owner: dict[str, Response] = {}
    tool_seen: set[str] = set()
    code_call_ids: set[str] = set()
    level = 1
    request: dict[str, Any] = {}
    for record in _records(logs):
        if record.get("event") == "request":
            request = record
            messages = record.get("messages") or []
            new = []
            for message in messages:
                if message.get("role") != "assistant":
                    continue
                key = _message_key(message)
                if key not in seen:
                    seen.add(key)
                    new.append(message)
            owners = waiting[len(waiting) - len(new):] if new else []
            for response, message in zip(owners, new[-len(owners):] if owners else []):
                _attach(response, message, call_owner, code_call_ids)
            if new:
                waiting.clear()
            for message in messages:
                if message.get("role") != "tool":
                    continue
                call_id = str(message.get("tool_call_id", ""))
                if call_id in tool_seen or call_id not in call_owner:
                    continue
                tool_seen.add(call_id)
                size = len(_text(message.get("content")))
                call_owner[call_id].tool_output_chars += size
                if call_id in code_call_ids:
                    call_owner[call_id].code_output_chars += size
            for message in messages:
                if message.get("role") == "user":
                    found = LEVEL.findall(_text(message.get("content")))
                    if found:
                        level = int(found[-1])
            continue
        if record.get("event") != "response":
            continue
        usage = record.get("usage") or {}
        messages = request.get("messages") or []
        context_text = sum(len(_text(m.get("content"))) for m in messages)
        context_code = sum(
            len(_text(m.get("content")))
            for m in messages
            if m.get("role") == "tool" and str(m.get("tool_call_id", "")) in code_call_ids
        )
        response = Response(
            game=game,
            index=len(responses),
            analysis_step=int(record.get("analysis_step") or 0),
            action=int(record.get("action") or 0),
            level=level,
            finish_reason=str(record.get("finish_reason") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            ),
            cost=float(usage.get("cost") or 0.0),
            cost_input=float(
                (usage.get("cost_details") or {}).get("upstream_inference_prompt_cost") or 0.0
            ),
            cost_output=float(
                (usage.get("cost_details") or {}).get("upstream_inference_completions_cost") or 0.0
            ),
            context_code_chars=context_code,
            context_text_chars=context_text,
        )
        responses.append(response)
        reply = record.get("reply")
        if isinstance(reply, dict):
            seen.add(_message_key(reply))
            _attach(response, reply, call_owner, code_call_ids)
        else:
            waiting.append(response)
    return responses


def _attach(
    response: Response,
    message: dict[str, Any],
    call_owner: dict[str, Response],
    code_call_ids: set[str],
) -> None:
    response.matched = True
    response.reasoning = _reasoning(message)
    for call in message.get("tool_calls") or []:
        call_id = str(call.get("id"))
        arguments = str((call.get("function") or {}).get("arguments") or "")
        response.call_ids.append(call_id)
        response.call_chars += len(arguments)
        call_owner[call_id] = response
        try:
            code = str(json.loads(arguments).get("code", ""))
        except (ValueError, AttributeError):
            code = arguments
        if CODE_READ.search(code):
            code_call_ids.add(call_id)
            response.code_read_calls += 1
            response.code_read_call_chars += len(arguments)


def _continued_game(generic: Path, logs: dict[str, list[Path]]) -> str:
    """The game whose log the run-level requests.jsonl continues.

    Runs made before the log-naming fix moved a game's log there once it was
    the only game playing. Its first request repeats tool calls from that
    game's own log.
    """
    first = next(r for r in _records([generic]) if r.get("event") == "request")
    texts = {}
    for game, paths in logs.items():
        with open_log(paths[0]) as handle:
            texts[game] = handle.read()
    ids = [
        str(call.get("id"))
        for message in first.get("messages") or []
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]
    owners = [
        game
        for game, text in texts.items()
        if any(call_id in text for call_id in ids[:5])
    ]
    if len(owners) != 1:
        raise SystemExit(f"Cannot tell which game {generic} continues: {owners}")
    return owners[0]


def _benchmark(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "benchmark.json"
    if not path.exists():
        return {}
    runs = json.loads(path.read_text(encoding="utf-8"))["game_runs"]
    return {run["game_id"].split("-", 1)[0]: run for run in runs}


def _level_at(run: dict[str, Any], action: int) -> int:
    """The level being played when `action` is the number of the next action."""
    done, level = 0, 1
    for count in (run.get("actions_per_level") or [])[: int(run.get("levels_completed") or 0)]:
        done += count
        if done <= action - 1:
            level += 1
    return level


TRANSCRIPT_SECTION = re.compile(
    r"^\[(TOOL CALL|TOOL RESULT|ASSISTANT|ANALYZER STATUS|MODEL RESPONSE META|SYSTEM PROMPT|USER PROMPT)"
)


def _transcript_thinking(transcript: Path) -> list[str]:
    """The [THINKING] blocks of a transcript, one per model response."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in transcript.read_text(encoding="utf-8").split("\n"):
        if line == "[THINKING]":
            current = []
            blocks.append(current)
        elif current is not None and (
            TRANSCRIPT_SECTION.match(line) or line.startswith("--- analysis_step")
        ):
            current = None
        elif current is not None:
            current.append(line)
    return ["\n".join(block).strip() for block in blocks]


def _fill_from_transcript(responses: list[Response], transcript: Path) -> None:
    # A response whose message no later request carries (the game's last, or
    # one the harness dropped at a turn change) still has its thinking in the
    # transcript, which holds one block per response.
    if not transcript.exists():
        return
    blocks = _transcript_thinking(transcript)
    if len(blocks) != len(responses):
        print(f"{transcript}: {len(blocks)} thinking blocks for {len(responses)} responses; not used")
        return
    for response, block in zip(responses, blocks):
        if not response.matched and block:
            response.reasoning = block
            response.reasoning_from_transcript = True


def load_run(run_dir: Path) -> list[Response]:
    found = [*run_dir.glob("*_requests.jsonl"), *run_dir.glob("*_requests.jsonl.xz")]
    logs = {log.name.split("-", 1)[0]: [log] for log in sorted(found)}
    generic = existing_log(run_dir / "requests.jsonl")
    if generic is not None:
        logs[_continued_game(generic, logs)].append(generic)
    benchmark = _benchmark(run_dir)
    responses: list[Response] = []
    for game, paths in logs.items():
        game_responses = load_responses(paths, game)
        stem = paths[0].name.removesuffix(".xz").removesuffix("_requests.jsonl")
        _fill_from_transcript(game_responses, run_dir / "transcripts" / f"{stem}.txt")
        for response in game_responses:
            # benchmark.json places level changes exactly; the turn opener
            # only reports them at the next turn, but is current when
            # benchmark.json (saved every 10 minutes) is not
            if game in benchmark:
                response.level = max(response.level, _level_at(benchmark[game], response.action))
            responses.append(response)
    return responses


def excerpts(response: Response) -> list[str]:
    """The response's reasoning in pieces of about EXCERPT_CHARS, cut at sentences."""
    pieces = [p for p in re.split(r"(?<=[.!?:])\s+|\n+", response.reasoning) if p.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > 2 * EXCERPT_CHARS:
            chunks.append((current + " " + piece[:EXCERPT_CHARS]).strip())
            current, piece = "", piece[EXCERPT_CHARS:]
        if current and len(current) + len(piece) > EXCERPT_CHARS:
            chunks.append(current)
            current = ""
        current = (current + " " + piece).strip()
    if current:
        chunks.append(current)
    return chunks


def _label_batch(batch: list[tuple[str, str]], api_key: str) -> tuple[dict[str, dict], float]:
    body = "\n\n".join(
        f'<excerpt id="{index}">\n{text}\n</excerpt>' for index, (_, text) in enumerate(batch)
    )
    for _ in range(3):
        try:
            reply = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": LABEL_MODEL,
                    "temperature": 0,
                    "max_tokens": 4000,
                    "messages": [
                        {"role": "system", "content": LABEL_PROMPT},
                        {"role": "user", "content": body},
                    ],
                },
                timeout=180,
            )
            reply.raise_for_status()
            data = reply.json()
            text = data["choices"][0]["message"]["content"]
            # usually {"labels": [...]}, but sometimes the bare list
            start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
            parsed = json.loads(text[start : max(text.rfind("}"), text.rfind("]")) + 1])
            labels = parsed["labels"] if isinstance(parsed, dict) else parsed
        except (requests.RequestException, KeyError, ValueError, TypeError):
            continue
        out = {}
        for item in labels:
            index = int(item.get("id", -1))
            if 0 <= index < len(batch) and item.get("topic") in TOPICS:
                out[batch[index][0]] = {
                    "topic": item["topic"],
                    "source_code": bool(item.get("source_code")),
                }
        if len(out) == len(batch):
            return out, float((data.get("usage") or {}).get("cost") or 0.0)
    return {}, 0.0


def label(run: str, responses: list[Response], cache: Path, workers: int = 8) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            labels[item["id"]] = item
    todo = [
        (excerpt_id, text)
        for response in responses
        for number, text in enumerate(excerpts(response))
        if (excerpt_id := f"{response.game}:{response.index}:{number}") not in labels
    ]
    if not todo:
        return labels
    api_key = os.environ["OPENROUTER_API_KEY"]
    batches = [todo[start : start + LABEL_BATCH] for start in range(0, len(todo), LABEL_BATCH)]
    # about $0.33 per 1,000 excerpts with Haiku 4.5 at $1/$5 per M tokens
    print(
        f"{run}: labelling {len(todo)} excerpts in {len(batches)} requests, "
        f"about ${len(todo) * 0.00033:.2f}"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    pool = concurrent.futures.ThreadPoolExecutor(workers)
    try:
        with cache.open("a", encoding="utf-8") as sink:
            for done, (result, cost) in enumerate(
                pool.map(lambda batch: _label_batch(batch, api_key), batches), start=1
            ):
                spent += cost
                for excerpt_id, item in result.items():
                    labels[excerpt_id] = {"id": excerpt_id, **item}
                    sink.write(json.dumps(labels[excerpt_id]) + "\n")
                sink.flush()
                if done % 25 == 0 or done == len(batches):
                    print(f"{run}: {done}/{len(batches)} requests, ${spent:.2f}", flush=True)
    finally:
        # pool.map queues every batch up front; on an interrupt, drop the
        # ones not yet sent instead of paying for them
        pool.shutdown(wait=True, cancel_futures=True)
    return labels


def topic_tokens(response: Response, labels: dict[str, dict]) -> dict[str, float]:
    """The response's reasoning tokens by topic, plus `source_code` and `unlabelled`."""
    out: dict[str, float] = defaultdict(float)
    if not response.reasoning:
        out["unlabelled"] += response.reasoning_tokens
        return out
    chunks = excerpts(response)
    total = sum(len(chunk) for chunk in chunks) or 1
    for number, chunk in enumerate(chunks):
        share = response.reasoning_tokens * len(chunk) / total
        item = labels.get(f"{response.game}:{response.index}:{number}")
        if item is None:
            out["unlabelled"] += share
            continue
        out[item["topic"]] += share
        if item["source_code"]:
            out["source_code"] += share
    return out


def _sum(rows: list[Response], labels: dict[str, dict] | None) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for response in rows:
        total["responses"] += 1
        total["prompt_tokens"] += response.prompt_tokens
        total["cached_tokens"] += response.cached_tokens
        total["completion_tokens"] += response.completion_tokens
        total["thinking_tokens"] += response.reasoning_tokens
        total["tool_call_tokens"] += response.tool_call_tokens
        total["cost_usd"] += response.cost
        total["cost_input_usd"] += response.cost_input
        total["cost_output_usd"] += response.cost_output
        total["code_read_calls"] += response.code_read_calls
        total["code_output_chars"] += response.code_output_chars
        if response.call_chars:
            total["code_read_call_tokens"] += (
                response.tool_call_tokens * response.code_read_call_chars / response.call_chars
            )
        if response.context_text_chars:
            total["context_code_tokens_est"] += (
                response.prompt_tokens * response.context_code_chars / response.context_text_chars
            )
        if labels is not None:
            for topic, tokens in topic_tokens(response, labels).items():
                total[f"thinking_{topic}"] += tokens
    return dict(total)


def _game_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, float] = {}
    evaluation = run_dir / "evaluation.json"
    if evaluation.exists():
        for game in json.loads(evaluation.read_text(encoding="utf-8"))["games"]:
            scores[game["game_id"].split("-", 1)[0]] = game["score"]
    results: dict[str, dict[str, Any]] = {}
    for game, run in _benchmark(run_dir).items():
        results[game] = {
            "score": scores.get(game, run.get("final_score")),
            "state": run.get("state"),
            "levels_completed": int(run.get("levels_completed") or 0),
            "total_levels": run.get("number_of_levels"),
            "actions": sum(run.get("actions_per_level") or []),
            "benchmark_output_tokens": run.get("final_generated_tokens"),
            "minutes": round(float(run.get("final_wallclock_seconds") or 0) / 60, 1),
        }
    return results


def _level_results(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Actions per level and the official level score: min(115, (baseline / actions)^2 x 100)."""
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for game, run in _benchmark(run_dir).items():
        completed = int(run.get("levels_completed") or 0)
        base = run.get("base_actions_per_level") or []
        for number, actions in enumerate(run.get("actions_per_level") or [], start=1):
            solved = number <= completed
            results[(game, number)] = {
                "solved": solved,
                "actions": actions,
                "baseline_actions": base[number - 1] if number <= len(base) else None,
                "level_score": (
                    min(115.0, (base[number - 1] / actions) ** 2 * 100)
                    if solved and actions and number <= len(base)
                    else 0.0
                ),
            }
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# validated categorical slots 1-4 (light mode); baseline vs candidate uses a
# neutral reference gray and slot 7 so neither collides with the parts chart
PARTS = [
    ("Thinking: mechanics", ("thinking_mechanics",), "#2a78d6"),
    ("Thinking: planning", ("thinking_planning",), "#eb6834"),
    (
        "Thinking: tooling and other",
        ("thinking_tooling", "thinking_other", "thinking_unlabelled"),
        "#1baf7a",
    ),
    ("Tool calls (code written)", ("tool_call_tokens",), "#eda100"),
]
RUN_COLORS = ("#a3a29c", "#4a3aa7")


def _axis_style(axis: Any) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}K" if value else "0")


def _charts(out: Path, summary: dict[str, dict[str, Any]], names: dict[str, str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": TEXT_PRIMARY})
    runs = list(summary)

    # output tokens of each run, by part
    figure, axis = plt.subplots(figsize=(8, 1.4 + 0.55 * len(runs)), facecolor=SURFACE)
    _axis_style(axis)
    left = [0.0] * len(runs)
    for name, keys, color in PARTS:
        values = [sum(summary[r]["total"].get(k, 0.0) for k in keys) / 1e3 for r in runs]
        axis.barh(
            [names[r] for r in runs], values, left=left, height=0.5, color=color,
            edgecolor=SURFACE, linewidth=1.2, label=name,
        )
        left = [a + b for a, b in zip(left, values)]
    for row, total in enumerate(left):
        axis.text(total, row, f"  {total:,.0f}K", va="center", color=TEXT_PRIMARY, fontsize=9)
    axis.set_xlim(0, max(left) * 1.12)
    axis.invert_yaxis()
    axis.tick_params(axis="y", colors=TEXT_PRIMARY)
    axis.set_title("Output tokens by part", loc="left", fontsize=10, color=TEXT_PRIMARY)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, frameon=False,
        fontsize=8, labelcolor=TEXT_SECONDARY,
    )
    figure.tight_layout()
    figure.savefig(out / "tokens.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)

    # output tokens per game, one bar per run
    games = sorted({game for r in runs for game in summary[r]["by_game"]})
    figure, axis = plt.subplots(figsize=(8, 1.2 + 0.3 * len(games) * len(runs)), facecolor=SURFACE)
    _axis_style(axis)
    height = 0.8 / len(runs)
    largest = 0.0
    for index, (run, color) in enumerate(zip(runs, RUN_COLORS)):
        positions = [g + (index - (len(runs) - 1) / 2) * height for g in range(len(games))]
        values = [summary[run]["by_game"].get(game, {}).get("completion_tokens", 0) / 1e3 for game in games]
        largest = max(largest, *values)
        axis.barh(positions, values, height=height * 0.7, color=color, label=names[run])
        for position, value, game in zip(positions, values, games):
            score = summary[run]["games"].get(game, {}).get("score")
            note = f"  {value:,.0f}K" + (f"  (score {score:.0f})" if score is not None else "")
            axis.text(value, position, note, va="center", color=TEXT_SECONDARY, fontsize=8)
    axis.set_xlim(0, largest * 1.3)
    axis.set_yticks(range(len(games)), games)
    axis.tick_params(axis="y", colors=TEXT_PRIMARY)
    axis.invert_yaxis()
    axis.set_title("Output tokens per game", loc="left", fontsize=10, color=TEXT_PRIMARY)
    axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(out / "tokens_by_game.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", action="store_true", help="Label thinking topics.")
    parser.add_argument(
        "--names", help="Comma-separated display names for the runs in the charts."
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    response_rows, game_rows, level_rows = [], [], []
    summary: dict[str, dict[str, Any]] = {}
    for run_dir in args.runs:
        run = run_dir.name
        responses = load_run(run_dir)
        labels = (
            label(run, responses, args.out / "labels" / f"{run}.jsonl") if args.label else None
        )
        if labels is None and (args.out / "labels" / f"{run}.jsonl").exists():
            labels = label(run, [], args.out / "labels" / f"{run}.jsonl")
        results = _game_results(run_dir)
        by_game: dict[str, list[Response]] = defaultdict(list)
        by_level: dict[tuple[str, int], list[Response]] = defaultdict(list)
        for response in responses:
            by_game[response.game].append(response)
            by_level[(response.game, response.level)].append(response)
            row = {"run": run, **asdict(response), "tool_call_tokens": response.tool_call_tokens}
            row.pop("reasoning")
            row["call_ids"] = len(response.call_ids)
            if labels is not None:
                row.update(
                    {f"thinking_{k}": v for k, v in topic_tokens(response, labels).items()}
                )
            response_rows.append(row)
        for game, rows in sorted(by_game.items()):
            game_rows.append({"run": run, "game": game, **results.get(game, {}), **_sum(rows, labels)})
        level_results = _level_results(run_dir)
        for (game, level), rows in sorted(by_level.items()):
            level_rows.append({
                "run": run, "game": game, "level": level,
                **level_results.get((game, level), {}), **_sum(rows, labels),
            })
        matched = sum(r.matched or r.reasoning_from_transcript for r in responses)
        summary[run] = {
            "games": {game: results.get(game, {}) for game in sorted(by_game)},
            "by_game": {game: _sum(rows, labels) for game, rows in sorted(by_game.items())},
            "total": _sum(responses, labels),
            "responses_with_text": matched,
            "responses": len(responses),
        }

    _write_csv(args.out / "responses.csv", response_rows)
    _write_csv(args.out / "games.csv", game_rows)
    _write_csv(args.out / "levels.csv", level_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    names = dict(zip(summary, (args.names or "").split(","))) if args.names else {}
    _charts(args.out, summary, {run: names.get(run) or run for run in summary})
    for run, data in summary.items():
        total = data["total"]
        print(
            f"{run}: {data['responses']} responses ({data['responses_with_text']} with text), "
            f"output {total['completion_tokens']:,.0f} (thinking {total['thinking_tokens']:,.0f}), "
            f"input {total['prompt_tokens']:,.0f}, ${total['cost_usd']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
