"""Resend logged turn requests with and without the ARC3_REPEAT_HINT message.

    uv run --no-sync python scripts/stuck_detection/replay_hint_requests.py \
        runs/base-gpt61sol-20games sk48 35,55 out.jsonl --samples 2

For each analysis step, the first request of that turn is read from the run's
request log and sent again through the harness's own request path
(`ToolAgent._chat_completion`), with the run's environment (eval_settings.json,
else params.yaml's eval.env): `--samples` times as logged (control) and
`--samples` times with the message inserted where the harness puts it, before
the "Only tool:" line of the turn prompt. The message's counts come from the
run's event log at that turn. Each reply is appended to the output file. Needs
OPENAI_API_KEY or OPENROUTER_API_KEY, and the run unpacked.
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, copy, json, os, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_env(run: Path) -> dict[str, str]:
    settings = run / "eval_settings.json"
    if settings.exists():
        data = json.loads(settings.read_text())
        return {k: str(v) for k, v in {**data.get("make", {}), **data.get("env", {})}.items()}
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["eval"]
    return {k: str(v) for k, v in {**params.get("make", {}), **params.get("env", {})}.items()}


def turn_request(run: Path, gid: str, step: int):
    from inference.utils.run_artifacts import open_log
    log = next(run.glob(f"{gid}*_requests.jsonl*"))
    for line in open_log(log):
        r = json.loads(line)
        if (r.get("event") == "request" and str(r.get("analysis_step")) == str(step)
                and str(r.get("request_index_within_turn")) == "1"):
            return r
    raise SystemExit(f"no request for step {step} in {log}")


def logged_reply(run: Path, gid: str, step: int):
    from inference.utils.run_artifacts import open_log
    log = next(run.glob(f"{gid}*_requests.jsonl*"))
    for line in open_log(log):
        r = json.loads(line)
        if (r.get("event") == "response" and str(r.get("analysis_step")) == str(step)
                and str(r.get("request_index_within_turn")) == "1"):
            return r.get("reply")
    return None


def repetition_at(run: Path, gid: str, action: int, visits: int, border: int = 2):
    """Positions of the current level seen `visits`+ times before `action` runs."""
    events = next((run / "artifacts").glob(f"{gid}*_events.jsonl"))
    boards = []
    for line in open(events):
        d = json.loads(line)
        if d["type"] in ("initial", "action"):
            b = tuple(tuple(row[border:-border]) for row in d["board"][border:-border])
            boards.append((d["level"], b))
    seen = boards[:max(action, 1)]
    level = seen[-1][0]
    counts: dict = {}
    for lv, b in seen:
        if lv == level:
            counts[b] = counts.get(b, 0) + 1
    return sum(n >= visits for n in counts.values()), counts[seen[-1][1]] >= visits, level


def with_hint(messages: list, lines: list[str]) -> list:
    out = copy.deepcopy(messages)
    content = out[-1]["content"]
    text = "\n".join(lines) + "\n"
    if isinstance(content, str):
        assert "Only tool:" in content
        out[-1]["content"] = content.replace("Only tool:", text + "Only tool:", 1)
        return out
    for part in content:
        if part.get("type") == "text" and "Only tool:" in part.get("text", ""):
            part["text"] = part["text"].replace("Only tool:", text + "Only tool:", 1)
            return out
    raise SystemExit("no 'Only tool:' line in the last user message")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("game"); ap.add_argument("steps"); ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    env = run_env(run)
    for k, v in env.items():
        if k.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "OPENAI_", "EXPOSE_")):
            os.environ[k] = v
    os.environ.setdefault("OPENAI_REASONING_SUMMARY", "detailed")
    config = json.loads((ROOT / env.get("CONFIG_PATH", "configs/inference.openrouter.json")).read_text())["shared"]
    model = env.get("MODEL") or json.loads((run / "run_config.json").read_text()).get("model")
    if not model:
        model = config["model_name"]
    provider = config.get("provider", "openrouter")
    key = os.environ["OPENAI_API_KEY"] if provider == "openai-responses" else os.environ["OPENROUTER_API_KEY"]

    sys.path.insert(0, str(ROOT))
    from inference.agent.tool_agent import ToolAgent, _repeat_hint_text   # after the env is set

    gid = next((run / "artifacts").glob(f"{args.game}*_events.jsonl")).name.split("_p0")[0]
    jobs = []
    for step in [int(s) for s in args.steps.split(",")]:
        req = turn_request(run, gid, step)
        repeated, current, level = repetition_at(run, gid, int(req["action"]), visits=3)
        hint = _repeat_hint_text(repeated, visits=3, current_repeated=current)
        for variant in ("control", "hint"):
            msgs = req["messages"] if variant == "control" else with_hint(req["messages"], hint)
            for sample in range(args.samples):
                jobs.append(dict(step=step, action=int(req["action"]), level=level, repeated=repeated,
                                 variant=variant, sample=sample, messages=msgs, tools=req["tools"]))

    def send(job):
        agent = ToolAgent(model=model, provider=provider, base_url=config["base_url"], api_key=key, timeout=900)
        t0 = time.time()
        try:
            result = agent._chat_completion(job["messages"], tools=job["tools"])
            return dict(message=result.message, usage=result.usage, seconds=round(time.time() - t0))
        except Exception as exc:   # keep the other samples
            return dict(error=str(exc)[:500], seconds=round(time.time() - t0))

    with cf.ThreadPoolExecutor(8) as pool, open(args.out, "a") as out:
        futures = {pool.submit(send, j): j for j in jobs}
        for fut in cf.as_completed(futures):
            j = futures[fut]
            rec = {k: v for k, v in j.items() if k not in ("messages", "tools")}
            rec.update(run=run.name, gid=gid, model=model, **fut.result())
            if j["variant"] == "control" and j["sample"] == 0:
                rec["logged_reply"] = logged_reply(run, gid, j["step"])
            out.write(json.dumps(rec, default=str) + "\n"); out.flush()
            print(run.name, gid, "step", j["step"], j["variant"], j["sample"], rec.get("error", "ok"), rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
