"""Count returns to the level's start board (life lost, death, reset), tolerant to counters."""
import pickle, numpy as np
from features import hud_mask
games = pickle.load(open("games.pkl", "rb"))
res = {}
for g in games:
    m = hud_mask(g); rows = g["rows"]
    diffs = [int((r["board"] != (rows[k-1]["board"] if k else g["init"]))[m].sum()) for k, r in enumerate(rows)]
    nz = [d for d in diffs if d > 0]; tau = int(min(8, 0.3 * np.median(nz))) if nz else 0
    start = g["init"][m]; out = []; was_at_start = True
    for k, r in enumerate(rows):
        L_before = rows[k-1]["level"] if k else 1
        v = r["board"][m]
        at = int((v != start).sum()) <= tau
        out.append(at and not was_at_start)      # an arrival back at the start board
        was_at_start = at
        if r["level"] != L_before: start = v; was_at_start = True
    res[(g["run"], g["game"])] = out
    print(g["run"], g["game"], sum(out))
pickle.dump(res, open("backstart.pkl", "wb"))
