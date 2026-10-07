"""Per-action stuck signals, computed causally (only from the past) within each level."""
import pickle, zlib
import numpy as np

EDGE = 6        # HUD bands are looked for within this many pixels of the border
HUD_FREQ = 0.6  # a row/col that changes on >= this share of in-level steps is a counter

def hud_mask(g):
    rows = g["rows"]; prev = g["init"]
    R = np.zeros(64); C = np.zeros(64); n = 0
    for k, r in enumerate(rows):
        b = r["board"]
        if k and rows[k-1]["level"] == r["level"] and r["name"] != "RESET" and not r["game_over"]:
            d = b != prev
            if d.any():
                R += d.any(1); C += d.any(0); n += 1
        prev = b
    R /= max(n, 1); C /= max(n, 1)
    m = np.ones((64, 64), bool)
    edge = np.r_[0:EDGE, 64-EDGE:64]
    for i in edge:
        if R[i] >= HUD_FREQ: m[i, :] = False
        if C[i] >= HUD_FREQ: m[:, i] = False
    return m

def akey(r):
    d = r["data"]
    return r["name"] + (f"@{d.get('x')},{d.get('y')}" if r["name"] == "ACTION6" else "")

def features(g, W=30):
    m = hud_mask(g)
    rows = g["rows"]
    out = []
    lvl_start = 0; seen = {}; ngrams = set(); acts = []
    resets = gos = 0; tok0 = 0; t0 = 0.0; steps = set(); start_key = None
    done_levels = []   # actions spent on each finished level
    prev_t = 0.0
    for k, r in enumerate(rows):
        L_before = rows[k-1]["level"] if k else 1   # this action belongs to the level before it
        key = r["board"][m].tobytes()
        n_lvl = k - lvl_start + 1
        revisit = key in seen
        seen[key] = seen.get(key, 0) + 1
        a = akey(r); acts.append(a)
        gram = tuple(acts[-6:]) if len(acts) >= 6 else None
        gram_rep = gram is not None and gram in ngrams
        if gram is not None: ngrams.add(gram)
        is_reset = r["name"] == "RESET"; resets += is_reset; gos += bool(r["game_over"])
        steps.add(r["step"])
        out.append(dict(k=k, level=L_before, n_lvl=n_lvl, revisit=revisit, gram_rep=gram_rep,
                        nochange=not r["changed"], reset=is_reset, go=bool(r["game_over"]),
                        resets=resets, gos=gos, tok_lvl=sum(x["tok"] for x in rows[lvl_start:k+1]),
                        t_lvl=r["t"] - t0, turns_lvl=len(steps), solved=bool(r["level"] > L_before),
                        state_count=seen[key], dt=r["t"] - prev_t, tok=r["tok"], act=a,
                        prior_med=float(np.median(done_levels)) if done_levels else None))
        prev_t = r["t"]
        if r["level"] > L_before or (r["state"] == "WIN"):
            done_levels.append(n_lvl)
            lvl_start = k + 1; seen = {}; ngrams = set(); acts = []
            resets = gos = 0; t0 = r["t"]; steps = set()
    # sliding-window rates over the current level only
    for i, f in enumerate(out):
        lo = max(i - W + 1, i - f["n_lvl"] + 1)
        win = out[lo:i+1]
        n = len(win)
        f["w_n"] = n
        f["w_revisit"] = sum(x["revisit"] for x in win) / n
        f["w_gram"] = sum(x["gram_rep"] for x in win) / n
        f["w_nochange"] = sum(x["nochange"] for x in win) / n
        f["w_resets"] = sum(x["reset"] or x["go"] for x in win)
        s = "|".join(x["act"] for x in win).encode()
        f["w_compress"] = len(zlib.compress(s, 9)) / max(len(s), 1)
    return out

if __name__ == "__main__":
    games = pickle.load(open("games.pkl", "rb"))
    feats = {(g["run"], g["game"]): features(g) for g in games}
    pickle.dump(feats, open("feats.pkl", "wb"))
    F = feats[("gpt20", "sk48")]
    for f in F[140:600:15]:
        print(f["k"]+1, f["level"], f["n_lvl"], "rev%.2f gram%.2f noch%.2f" % (f["w_revisit"], f["w_gram"], f["w_nochange"]),
              "rst", f["resets"], f["gos"], "tok%dk" % (f["tok_lvl"]/1000), "min%.0f" % (f["t_lvl"]/60), "turns", f["turns_lvl"], f["act"])
