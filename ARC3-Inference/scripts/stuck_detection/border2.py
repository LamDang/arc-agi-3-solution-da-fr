"""User's idea: drop the 2-pixel border, hash the board, flag a position seen a 3rd time on the level."""
import pickle, numpy as np
from features import hud_mask
games = pickle.load(open("games.pkl", "rb"))
res = {}
for g in games:
    rows = g["rows"]
    # tolerant variant needs the per-game tau used by near.py
    m = hud_mask(g)
    diffs = [int((r["board"] != (rows[k-1]["board"] if k else g["init"]))[m].sum()) for k, r in enumerate(rows)]
    nz = [d for d in diffs if d > 0]; tau = int(min(8, 0.3 * np.median(nz))) if nz else 0
    counts = {}; out = []; buf = []; bcnt = []
    start = g["init"][2:-2, 2:-2].tobytes(); counts[start] = 1
    buf = [g["init"][m].ravel()]; bcnt = [1]
    for k, r in enumerate(rows):
        L_before = rows[k-1]["level"] if k else 1
        key = r["board"][2:-2, 2:-2].tobytes()
        counts[key] = counts.get(key, 0) + 1
        # tolerant: nearest earlier board within tau
        v = r["board"][m].ravel(); P = np.stack(buf[-300:]); d = (P != v).sum(1)
        j = int(d.argmin())
        if d[j] <= tau:
            idx = len(buf) - len(P) + j; bcnt[idx] += 1; tc = bcnt[idx]
        else:
            buf.append(v); bcnt.append(1); tc = 1
        out.append((counts[key], tc))
        if r["level"] != L_before:
            counts = {key: 1}; buf = [v]; bcnt = [1]
    res[(g["run"], g["game"])] = out
pickle.dump(res, open("border2.pkl", "wb"))
print("ok")
