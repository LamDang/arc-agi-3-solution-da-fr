"""Replay note-compaction requests on logged gpt-6.1-sol games.

    uv run --no-sync python experiments/note-compaction/replay_notes.py triggers
    uv run --no-sync python experiments/note-compaction/replay_notes.py replay v1 OUT_DIR
    uv run --no-sync python experiments/note-compaction/replay_notes.py dump OUT_DIR > notes.md

`triggers` lists, for each game whose prompt passed 100K tokens, the logged
compactions (prompt dropping by 25K or more) and where ARC3_NOTE_COMPACTION_TOKENS
(TRIG, default 110000) would fire: the first turn-start request at or above it
in each stretch between compactions. Only the first trigger of a game is the one
the new setting would see; later ones follow the logged run, which compacted
without a note.

`replay` sends, at the first trigger of each game plus the sk48 and bp35 points
of exp/gpt61sol-compaction.md, the logged history up to that turn's opener and
the note request instead of the opener, to gpt-6.1-sol (effort xhigh, the
logged tools, the encrypted reasoning sent back). `v0` is the bare request,
`v1` NOTE_COMPACTION_PROMPT. One JSON file per point. Needs OPENAI_API_KEY and
the runs pulled (dvc pull runs/base-gpt61sol-20games.dvc
runs/base-gpt61sol-dfranzen.dvc).
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import lzma
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.agent.prompts import NOTE_COMPACTION_PROMPT  # noqa: E402
from inference.utils.openai_compat import (  # noqa: E402
    assemble_streamed_responses,
    responses_payload_from_chat,
)

RUNS = ["runs/base-gpt61sol-20games", "runs/base-gpt61sol-dfranzen"]
TRIG = int(os.environ.get("TRIG", 110000))
KEEP = 20
EXTRA = {"sk48": (55, 72, 179), "bp35": (92,)}  # exp/gpt61sol-compaction.md cases

V0 = (
    "Context notice: this conversation has reached about 110K tokens and is about to be "
    "truncated. After your reply, only the system prompt, this message, your reply and your "
    "last 20 turns will be kept; every older turn will be deleted from your context.\n\n"
    "Before that happens, call the `python` tool once to write yourself a note with all the "
    "information you need to continue the game efficiently after the cut. Write the note as "
    "Python comments only (every line starts with `#`), so that running it does nothing and "
    "prints nothing: the note stays in your context as the code of this call. Do not call "
    "`action(...)` or run anything else in this call. After it, the game continues with the "
    "normal message for the current turn."
)


def requests_of(path: str):
    """(request, response) pairs of a request log."""
    pending = None
    with lzma.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if record["event"] == "request":
                pending = record
            elif record["event"] == "response" and pending is not None:
                yield pending, record
                pending = None


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def first_kept_step(history: list[dict]) -> int | None:
    starts = [i for i, m in enumerate(history) if m["role"] == "user" and not m.get("_arc3_control")]
    found = re.search(r"Current state: step (\d+)", text_of(history[starts[-KEEP]]))
    return int(found.group(1)) if found else None


def triggers() -> list[dict]:
    out = []
    for run in RUNS:
        for path in sorted(glob.glob(f"{run}/*_requests.jsonl.xz")):
            game = Path(path).name.split("_")[0]
            rows = [
                dict(step=q["analysis_step"], idx=q["request_index_within_turn"],
                     prompt=(r.get("usage") or {}).get("prompt_tokens") or 0)
                for q, r in requests_of(path)
            ]
            if max(row["prompt"] for row in rows) < 100000:
                continue
            cuts = [i for i in range(1, len(rows)) if rows[i]["prompt"] < rows[i - 1]["prompt"] - 25000]
            bounds = [0, *cuts, len(rows)]
            fire = []
            for a, b in zip(bounds, bounds[1:]):
                hit = next((i for i in range(a, b) if rows[i]["idx"] == 1 and rows[i]["prompt"] >= TRIG), None)
                if hit is not None:
                    fire.append(dict(i=hit, **rows[hit]))
            out.append(dict(game=game, path=path, compactions=[rows[i]["step"] for i in cuts], triggers=fire))
            print(f"{game:14} compactions at step {out[-1]['compactions']}; "
                  f"triggers at step {[(t['step'], t['prompt'] // 1000) for t in fire]}")
    return out


def replay_one(path: str, index: int, variant: str) -> dict:
    for n, (request, response) in enumerate(requests_of(path)):
        if n == index:
            break
    history = request["messages"][:-1]
    assert request["messages"][-1]["role"] == "user"
    step = first_kept_step(history[1:])
    text = V0 if variant == "v0" else NOTE_COMPACTION_PROMPT.format(
        threshold_k=TRIG // 1000, keep_turns=KEEP,
        kept_from=f" (from game step {step} on)" if step else "",
    )
    chat = {"model": "gpt-6.1-sol", "messages": [*history, {"role": "user", "content": text}],
            "tools": request["tools"], "tool_choice": "auto"}
    payload = responses_payload_from_chat(chat, reasoning_effort="xhigh")
    payload["stream"] = True
    for attempt in range(6):
        try:
            reply = requests.post(
                "https://api.openai.com/v1/responses", json=payload, stream=True, timeout=1800,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            )
            if reply.status_code != 200:
                raise RuntimeError(f"{reply.status_code} {reply.text[:300]}")
            result = assemble_streamed_responses(reply.iter_lines())
            if result.get("choices"):
                return dict(first_kept_step=step, prompt=text, result=result)
            raise RuntimeError(str(result.get("error")))
        except Exception as exc:  # retried: overloads come back inside the stream
            print("retry", path, index, exc, file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{path} request {index} failed")


def replay(variant: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for game in triggers():
        picks = [game["triggers"][0]] + [
            t for t in game["triggers"] if t["step"] in EXTRA.get(game["game"][:4], ())
        ]
        jobs += [dict(game=game["game"], path=game["path"], **t) for t in picks]

    def run(job: dict) -> None:
        dst = f"{out_dir}/{job['game'][:4]}_{job['step']}.json"
        if not os.path.exists(dst):
            json.dump({**job, **replay_one(job["path"], job["i"], variant)}, open(dst, "w"), indent=1)
            print(dst, flush=True)

    with cf.ThreadPoolExecutor(8) as pool:
        for future in cf.as_completed([pool.submit(run, job) for job in jobs]):
            future.result()


def dump(out_dir: str) -> None:
    for path in sorted(glob.glob(f"{out_dir}/*.json")):
        data = json.load(open(path))
        result = data["result"]
        message = result["choices"][0]["message"]
        usage = result["usage"]
        print(f"\n## {data['game']} step {data['step']} (prompt {usage['prompt_tokens']}, "
              f"output {usage['completion_tokens']}, turns kept from step {data['first_kept_step']})\n")
        if message.get("content"):
            print(message["content"])
        for call in message.get("tool_calls") or []:
            code = json.loads(call["function"]["arguments"]).get("code", "")
            print("```python\n" + code.rstrip() + "\n```")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "triggers":
        triggers()
    elif command == "replay":
        replay(sys.argv[2], sys.argv[3])
    elif command == "dump":
        dump(sys.argv[2])
