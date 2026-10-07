"""Build a per-action table for every game run of the five runs."""
import glob, hashlib, json, os, pickle, sys
import numpy as np
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs"))
RUNS = {
    "gpt20": "base-gpt61sol-20games",
    "gpt5": "base-gpt61sol-dfranzen",
    "maxdef": "base-max-default",
    "maxdf": "base-max-dfranzen",
    "flash": "20261004_135539",
}
MODEL = {"gpt20": "gpt", "gpt5": "gpt", "maxdef": "max", "maxdf": "max", "flash": "flash"}

def load():
    games = []
    for tag, rd in RUNS.items():
        bench = json.load(open(f"{RUNS_DIR}/{rd}/benchmark.json"))
        for g in bench["game_runs"]:
            gid = g["game_id"]
            ev = f"{RUNS_DIR}/{rd}/artifacts/{gid}_p0_events.jsonl"
            acts, init = [], None
            for line in open(ev):
                d = json.loads(line)
                if d["type"] == "initial":
                    init = np.array(d["board"], dtype=np.int8)
                elif d["type"] == "action":
                    acts.append(d)
            h = g["history"]
            assert len(h) == len(acts), (tag, gid, len(h), len(acts))
            rows = []
            for i, (a, hh) in enumerate(zip(acts, h)):
                rows.append(dict(
                    i=i, level=a["level"], name=a["action_name"], disp=a["action_display"],
                    data=hh["action"].get("data") or {}, changed=a.get("board_changed"),
                    game_over=a.get("game_over"), level_completed=a.get("level_completed"),
                    step=a.get("analysis_step"), t=hh["wallclock_seconds"], tok=hh["generated_tokens"],
                    state=a["state"], board=np.array(a["board"], dtype=np.int8),
                ))
            games.append(dict(run=tag, model=MODEL[tag], game=gid[:4], gid=gid, init=init, rows=rows,
                              base=g["base_actions_per_level"], apl=g.get("actions_per_level"),
                              nlev=g["number_of_levels"], state=g["state"], score=g.get("final_score")))
    return games

if __name__ == "__main__":
    games = load()
    pickle.dump(games, open("games.pkl", "wb"))
    for g in games:
        print(g["run"], g["game"], len(g["rows"]), g["state"], g["apl"])
