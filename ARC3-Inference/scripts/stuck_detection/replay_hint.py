"""When would ARC3_REPEAT_HINT have shown in the recorded runs? (turn = analysis step)

Run from the run.sh output directory, after run.sh."""
import json, os, pickle
from load import RUNS, RUNS_DIR
from evaluate import load_gt
games = pickle.load(open("games.pkl", "rb")); gt = load_gt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json"))
COOL = 10
tot = {"stuck_level": 0, "normal_level": 0}; per = []
for g in games:
    rd = RUNS[g["run"]]
    turns = []   # (analysis_step, action_num of the next action) in order, deduped by step
    for line in open(f"{RUNS_DIR}/{rd}/artifacts/{g['gid']}_p0_events.jsonl"):
        if '"type": "analysis"' not in line[:400] and '"type":"analysis"' not in line[:400]:
            d = json.loads(line)
            if d.get("type") != "analysis": continue
        else:
            d = json.loads(line)
        turns.append((d["analysis_step"], d["action_num"]))
    seen_steps = set(); T = []
    for s, a in turns:
        if s not in seen_steps: seen_steps.add(s); T.append((s, a))
    rows = g["rows"]
    frames = [(1, g["init"][2:-2, 2:-2].tobytes())] + [(r["level"], r["board"][2:-2, 2:-2].tobytes()) for r in rows]
    since = None; shown = []
    for s, a in T:
        # at the start of this turn the agent has seen the init + actions 1..a-1
        hist = frames[:max(a, 1)]
        level = hist[-1][0]
        if since is not None:
            since += 1
            if since < COOL: continue
        counts = {}
        for L, k in hist:
            if L == level: counts[k] = counts.get(k, 0) + 1
        if sum(n >= 3 for n in counts.values()) >= 3:
            since = 0; shown.append((s, a, level))
    key = (g["run"], g["game"])
    for s, a, L in shown:
        lab = gt.get(key, {}).get(L)
        ivs = lab["intervals"] if lab else []
        where = "stuck level" if ivs else ("normal (read)" if lab else "normal (unread)")
        inside = any(iv["start"] - 10 <= a <= iv["end"] for iv in ivs)
        tot["stuck_level" if ivs else "normal_level"] += 1
        per.append((key, s, a, L, where, inside))
    if shown:
        print(key, len(T), "turns;", "hint at (turn, action, level):", [(s, a, L) for s, a, L in shown])
print(tot)
print("normal-level hints:", [(k, s, a, L) for (k, s, a, L, w, i) in per if w != "stuck level"])
print("on stuck levels, inside or within 10 actions before a stuck stretch:", sum(i for *_, w, i in per if w == "stuck level"), "of", tot["stuck_level"])
