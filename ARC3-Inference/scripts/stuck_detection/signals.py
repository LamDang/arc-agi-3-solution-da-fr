"""Assemble per-action signal table (causal, within level) for every game run."""
import pickle, numpy as np

def rle(acts):
    out = []
    for a in acts:
        if not out or out[-1] != a: out.append(a)
    return out

def build():
    games = pickle.load(open("games.pkl", "rb"))
    feats = pickle.load(open("feats.pkl", "rb"))
    near = pickle.load(open("near.pkl", "rb"))
    back = pickle.load(open("backstart.pkl", "rb"))
    b2 = pickle.load(open("border2.pkl", "rb"))
    table = {}
    for g in games:
        key = (g["run"], g["game"]); F = feats[key]; NR = near[key]
        done = []   # (actions, tokens, turns) of finished levels
        lvl_rows = []
        for i, f in enumerate(F):
            f["near"] = NR[i]; f["back"] = back[key][i]; f["b2c"], f["tolc"] = b2[key][i]; f["t"] = g["rows"][i]["t"]
            if f["n_lvl"] == 1: lvl_rows = []
            lvl_rows.append(f)
            # windowed near-revisit over last 30 actions of this level
            win = lvl_rows[-30:]
            f["w_near"] = sum(x["near"] for x in win) / len(win)
            for W in (60, 100):
                ww = lvl_rows[-W:]
                f[f"w{W}_near"] = sum(x["near"] for x in ww) / len(ww)
            # RLE action 4-gram repetition within level (ignores straight runs like RIGHTx6)
            r = rle([x["act"] for x in lvl_rows])
            grams = [tuple(r[j:j+4]) for j in range(len(r) - 3)]
            last = grams[-30:]; earlier = set(grams[:-30])
            seen = set(earlier); rep = 0
            for gm in last:
                rep += gm in seen; seen.add(gm)
            f["w_rlegram"] = rep / len(last) if last else 0.0
            f["episodes"] = sum(x["reset"] or x["go"] for x in lvl_rows)  # double-counts GO+auto reset
            f["gos_lvl"] = sum(x["go"] for x in lvl_rows)
            f["vresets_lvl"] = sum(1 for j, x in enumerate(lvl_rows) if x["reset"] and not (j and lvl_rows[j-1]["go"]))
            f["fails_lvl"] = f["gos_lvl"] + f["vresets_lvl"]
            f["starts_lvl"] = sum(x["back"] for x in lvl_rows)
            # positions reaching their 3rd visit on this level (exact after 2-px border crop, and tolerant)
            f["b2_third"] = sum(x["b2c"] == 3 for x in lvl_rows)
            f["tol_third"] = sum(x["tolc"] == 3 for x in lvl_rows)
            f["b2_third_w30"] = sum(x["b2c"] >= 3 for x in lvl_rows[-30:])
            f["tol_third_w30"] = sum(x["tolc"] >= 3 for x in lvl_rows[-30:])
            if done:
                f["rel_n"] = f["n_lvl"] / np.median([d[0] for d in done])
                f["rel_tok"] = f["tok_lvl"] / max(np.median([d[1] for d in done]), 1)
                f["rel_turns"] = f["turns_lvl"] / max(np.median([d[2] for d in done]), 1)
            else:
                f["rel_n"] = f["rel_tok"] = f["rel_turns"] = None
            if f["solved"] or i == len(F) - 1:
                done.append((f["n_lvl"], f["tok_lvl"], f["turns_lvl"]))
        table[key] = F
    return games, table

if __name__ == "__main__":
    games, table = build()
    pickle.dump(table, open("table.pkl", "wb"))
    F = table[("gpt20", "sk48")]
    for f in F[145:590:20]:
        print(f["k"]+1, f["level"], f["n_lvl"], "near%.2f rle%.2f" % (f["w_near"], f["w_rlegram"]), "fails", f["fails_lvl"],
              "rel_n %.1f rel_tok %.1f rel_turns %.1f" % (f["rel_n"], f["rel_tok"], f["rel_turns"]))
