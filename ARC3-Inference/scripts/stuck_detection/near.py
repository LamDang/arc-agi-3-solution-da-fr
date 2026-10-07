"""Near-revisit: the board after an action is within tau pixels of a board seen earlier in the level."""
import pickle, numpy as np
from features import hud_mask

def near_revisit(g, lookback=300):
    m = hud_mask(g)
    rows = g["rows"]; prev = g["init"]
    diffs = [int((r["board"] != (rows[k-1]["board"] if k else g["init"]))[m].sum()) for k, r in enumerate(rows)]
    nz = [d for d in diffs if d > 0]
    tau = int(min(8, 0.3 * np.median(nz))) if nz else 0
    out = []; buf = [g["init"][m].ravel()]
    for k, r in enumerate(rows):
        L_before = rows[k-1]["level"] if k else 1
        v = r["board"][m].ravel()
        P = np.stack(buf[-lookback:])
        dist = (P != v).sum(1)
        out.append(int(dist.min()) <= tau)
        if r["level"] != L_before: buf = [v]   # new level: forget
        else: buf.append(v)
    return out, tau

if __name__ == "__main__":
    games = pickle.load(open("games.pkl", "rb"))
    res = {}
    for g in games:
        nr, tau = near_revisit(g)
        res[(g["run"], g["game"])] = nr
        print(g["run"], g["game"], "tau", tau, "near-revisit frac %.2f" % np.mean(nr))
    pickle.dump(res, open("near.pkl", "wb"))
