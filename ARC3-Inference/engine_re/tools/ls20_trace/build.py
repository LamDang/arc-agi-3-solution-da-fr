"""Build ls20_trace.html: why the qwen3.8-flash ls20 engine stops matching.

Replays the recorded run through the real engine (to classify each step from
its hidden state), replays the agent's best and final engines in the sandbox,
condenses the agent transcript, and embeds everything into template.html.

Input: the v2 main run's ls20 game, archived in DVC. From ARC3-Inference/:
  dvc pull runs/engine-re/qwen38flash-v2-run20261004_135539.dvc
It reads ls20/{trace,tests.jsonl,transcript.jsonl,result.json,engine_best.py}
and the final engine ls20/workspace/engine.py.

Output: runs/engine-re/results-ls20-trace-html/ls20_trace.html, a run artifact
tracked in DVC (runs/engine-re/results-ls20-trace-html.dvc), not in git. After
rebuilding it, run `dvc add runs/engine-re/results-ls20-trace-html` and
`dvc push` to archive the new version.

The archived page was built at commit 692c260. The current tester labels
failing steps differently (since v4 it compares only each action's final
frame, so it no longer reports "frame count"); to rebuild the archived page
exactly, run this script in a worktree of 692c260 (`git worktree add
../ls20-trace 692c260`, with the run directory pulled or linked there). The
frames, exact counts and first failures are the same either way.

Usage, from ARC3-Inference/: .venv/bin/python engine_re/tools/ls20_trace/build.py
"""
import base64, gzip, json, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from engine_re.trace import Trace, load_game_class, new_game, perform
from engine_re.tester import run_candidate, check_step

RUN = ROOT / "runs/engine-re/qwen38flash-v2-run20261004_135539/ls20"
OUT = ROOT / "runs/engine-re/results-ls20-trace-html/ls20_trace.html"
SCRATCH = Path(tempfile.mkdtemp(prefix="ls20_trace_"))

trace = Trace.load(RUN / "trace")
cls = load_game_class(ROOT / "environment_files/ls20/9607627b/ls20.py")
g = new_game(cls)
NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "CLICK", 7: "UNDO"}

def hidden(g):
    try:
        return dict(
            x=int(g.gudziatsk.x), y=int(g.gudziatsk.y),
            shape=int(g.fwckfzsyc), colour=int(g.hiaauhahz), rot=int(g.cklxociuu),
            budget=int(g._step_counter_ui.current_steps), budget_max=int(g._step_counter_ui.osgviligwp),
            lives=int(g.aqygnziho), level=int(g.level_index) if hasattr(g, "level_index") else None,
        )
    except Exception as e:  # before first reset
        return None

steps = []
real_frames = []
prev = None
for s in trace.steps:
    before = hidden(g)
    obs = perform(g, s.action)
    assert obs["frames"].shape == s.frames.shape and (obs["frames"] == s.frames).all(), s.index
    after = hidden(g)
    n = s.n_frames
    cat = "move"
    a = NAMES[s.action.id]
    lc_before = steps[-1]["lc"] if steps else 0
    if s.action.id == 0:
        cat = "reset"
    elif s.state == "GAME_OVER":
        cat = "gameover"
    elif s.levels_completed > lc_before:
        cat = "win"
    elif before and after and after["lives"] < before["lives"]:
        cat = "death"
    elif n >= 7:
        cat = "push"
    elif before and after and (after["shape"], after["colour"], after["rot"]) != (before["shape"], before["colour"], before["rot"]):
        cat = "hint" if n == 6 else "key"
    elif before and after and after["budget"] > before["budget"] - 0 and (after["x"], after["y"]) != (before["x"], before["y"]) and after["budget"] >= before["budget"]:
        cat = "refill"
    elif before and after and (after["x"], after["y"]) == (before["x"], before["y"]):
        cat = "goalbump" if n == 6 else "bump"
    keyd = None
    if before and after and cat in ("key", "hint", "push"):
        keyd = [k for k in ("shape", "colour", "rot") if after[k] != before[k]]
    steps.append(dict(i=s.index, a=a, n=n, st=s.state, lc=s.levels_completed, cat=cat, h=after, kd=keyd))
    real_frames.extend(s.frames)

print(Counter(x["cat"] for x in steps))

def pack(frames):
    arr = np.stack(frames).astype(np.uint8) if frames else np.zeros((0, 64, 64), np.uint8)
    flat = arr.reshape(len(arr), -1)
    packed = (flat[:, 0::2] << 4) | flat[:, 1::2]
    return base64.b64encode(gzip.compress(packed.tobytes(), 9)).decode(), len(arr)

def offsets(per_step):
    o = [0]
    for f in per_step:
        o.append(o[-1] + len(f))
    return o

real_b64, nreal = pack(real_frames)
real_off = offsets([s.frames for s in trace.steps])

actions = [s.action.to_json() for s in trace.steps]
engines = {}
for key, path in (("best", RUN / "engine_best.py"), ("final", RUN / "workspace/engine.py")):
    result, per_step = run_candidate(path, actions, scratch_root=SCRATCH)
    checks = []
    for i, s in enumerate(trace.steps):
        got = result["steps"][i] if i < len(result["steps"]) else None
        gf = per_step[i] if i < len(per_step) else None
        c = check_step(s, got, gf)
        checks.append(dict(ok=c.ok, final=c.final_ok, p=c.problems,
                           n=(len(gf) if gf is not None else None),
                           st=(got or {}).get("state"), lc=(got or {}).get("levels_completed")))
    b64, nf = pack([fr for st in per_step for fr in st])
    exact = sum(c["ok"] for c in checks)
    first_fail = next((i for i, c in enumerate(checks) if not c["ok"]), None)
    engines[key] = dict(checks=checks, frames=b64, nframes=nf, off=offsets(per_step),
                        error=result.get("error"), error_step=result.get("error_step"),
                        exact=exact, first_fail=first_fail, source=path.read_text())
    print(key, "exact", exact, "first fail", first_fail, "frames", nf, "err step", result.get("error_step"))

tests = [json.loads(l) for l in open(RUN / "tests.jsonl")]
tests_out = []
for t in tests:
    err = t.get("error")
    tests_out.append(dict(turn=t["turn"], exact=t["exact"], total=t["total"], first_fail=t.get("first_fail"),
                          auto=t.get("auto"), from_level=t.get("from_level"),
                          err=(err.strip().splitlines()[-1][:200] if err else None), err_step=t.get("error_step")))

# transcript
recs = [json.loads(l) for l in open(RUN / "transcript.jsonl")]
turns = {}
def T(n):
    return turns.setdefault(n, dict(turn=n, min=None, reasoning="", content=None, calls=[], outputs=[], auto=None, nudge=None, usage=None))
LIM = 8000
for r in recs:
    t = T(r["turn"])
    if "tool_calls" in r:
        t["min"] = r.get("elapsed_min")
        t["reasoning"] = r.get("reasoning") or ""
        t["content"] = r.get("content")
        u = r.get("usage") or {}
        t["usage"] = dict(p=u.get("prompt_tokens"), c=u.get("completion_tokens"),
                          r=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                          cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), cost=u.get("cost"))
        for c in r.get("tool_calls") or []:
            fn = c["function"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                args = {"_raw": fn["arguments"]}
            t["calls"].append(dict(name=fn["name"], args=args))
    elif "tool" in r:
        out = r.get("output") or ""
        trunc = len(out) > LIM
        t["outputs"].append(dict(tool=r["tool"], s=r.get("seconds"), out=out[:LIM] + (f"\n… [{len(out)-LIM:,} more characters not shown]" if trunc else "")))
    elif "auto_test" in r:
        t["auto"] = r["auto_test"]
        t["min"] = t["min"] or r.get("elapsed_min")
    elif "nudge" in r:
        t["nudge"] = r["nudge"]
        t["min"] = t["min"] or r.get("elapsed_min")
turns = [turns[k] for k in sorted(turns)]
print("turns", len(turns), "reasoning chars", sum(len(t["reasoning"]) for t in turns),
      "output chars", sum(len(o["out"]) for t in turns for o in t["outputs"]))

result = json.loads((RUN / "result.json").read_text())
level_starts = {}
for s in steps:
    level_starts.setdefault(s["lc"], s["i"])
data = dict(steps=steps, real=dict(frames=real_b64, nframes=nreal, off=real_off),
            engines=engines, tests=tests_out, turns=turns, result=result, level_starts=level_starts,
            notes=(RUN / "workspace/notes.md").read_text() if (RUN / "workspace/notes.md").exists() else "")
raw = json.dumps(data, separators=(",", ":"))
payload = base64.b64encode(gzip.compress(raw.encode(), 9)).decode()
html = (HERE / "template.html").read_text().replace("__PAYLOAD__", payload)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print("wrote", OUT, len(html), "bytes")
import shutil
shutil.rmtree(SCRATCH, ignore_errors=True)
