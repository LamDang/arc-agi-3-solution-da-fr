"""Score stuck detectors against hand-labelled stuck intervals.

An alarm is a maximal run of actions where the detector is on (within one level).
- hit: a labelled interval with an alarm action inside it; delay = first alarm action - interval start
- false alarm: an alarm run that touches no labelled interval (slack SLACK actions before the start)
"""
import json, os, pickle, sys
import numpy as np

SLACK = 10
RUN_TAG = {"base-gpt61sol-20games": "gpt20", "base-gpt61sol-dfranzen": "gpt5", "base-max-default": "maxdef",
           "base-max-dfranzen": "maxdf", "20261004_135539": "flash"}

def g(f, k, default=0):
    v = f.get(k); return default if v is None else v

DETECTORS = {
    # board-position repetition
    "near-revisit>=0.7 (30)":   lambda f: f["n_lvl"] >= 30 and f["w_near"] >= 0.7,
    "near-revisit>=0.5 (30)":   lambda f: f["n_lvl"] >= 30 and f["w_near"] >= 0.5,
    "near-revisit>=0.8 (60)":   lambda f: f["n_lvl"] >= 60 and f["w60_near"] >= 0.8,
    "near-revisit>=0.7 (100)":  lambda f: f["n_lvl"] >= 100 and f["w100_near"] >= 0.7,
    "near-revisit>=0.8 (100)":  lambda f: f["n_lvl"] >= 100 and f["w100_near"] >= 0.8,
    "border-2 hash: 1 position 3rd visit": lambda f: f["b2_third"] >= 1,
    "border-2 hash: 3 positions 3rd visit": lambda f: f["b2_third"] >= 3,
    "border-2 hash: 5 positions 3rd visit": lambda f: f["b2_third"] >= 5,
    "border-2 hash: >=10 of last 30 at 3+": lambda f: f["b2_third_w30"] >= 10,
    "tolerant: 1 position 3rd visit": lambda f: f["tol_third"] >= 1,
    "tolerant: 3 positions 3rd visit": lambda f: f["tol_third"] >= 3,
    "tolerant: 5 positions 3rd visit": lambda f: f["tol_third"] >= 5,
    "tolerant: >=10 of last 30 at 3+": lambda f: f["tol_third_w30"] >= 10,
    "exact-revisit>=0.5 (30)":  lambda f: f["n_lvl"] >= 30 and f["w_revisit"] >= 0.5,
    # action-sequence repetition
    "6gram-repeat>=0.6":        lambda f: f["n_lvl"] >= 30 and f["w_gram"] >= 0.6,
    "rle-4gram-repeat>=0.5":    lambda f: f["n_lvl"] >= 30 and f["w_rlegram"] >= 0.5,
    "no-effect>=0.4 (30)":      lambda f: f["n_lvl"] >= 30 and f["w_nochange"] >= 0.4,
    # resets / game overs
    "fails>=2 on level":        lambda f: f["fails_lvl"] >= 2,
    "fails>=3 on level":        lambda f: f["fails_lvl"] >= 3,
    "back-to-start>=3 on level": lambda f: f["starts_lvl"] >= 3,
    "back-to-start>=5 on level": lambda f: f["starts_lvl"] >= 5,
    # effort relative to the agent's own pace on earlier levels
    "actions>=3x own median":   lambda f: g(f, "rel_n") >= 3 and f["n_lvl"] >= 50,
    "actions>=4x own median":   lambda f: g(f, "rel_n") >= 4 and f["n_lvl"] >= 50,
    "turns>=4x own median":     lambda f: g(f, "rel_turns") >= 4 and f["turns_lvl"] >= 20,
    "tokens>=5x own median":    lambda f: g(f, "rel_tok") >= 5 and f["tok_lvl"] >= 40000,
    # absolute effort
    "actions>=150 on level":    lambda f: f["n_lvl"] >= 150,
    "turns>=40 on level":       lambda f: f["turns_lvl"] >= 40,
    "tokens>=100k on level":    lambda f: f["tok_lvl"] >= 100000,
    "minutes>=45 on level":     lambda f: f["t_lvl"] >= 45 * 60,
}

def combos():
    D = DETECTORS
    rev = lambda f: f["n_lvl"] >= 30 and f["w_near"] >= 0.6
    eff = lambda f: (g(f, "rel_n") >= 2.5 or (f["rel_n"] is None and f["n_lvl"] >= 100))
    effturn = lambda f: g(f, "rel_turns") >= 3 and f["turns_lvl"] >= 20
    D["COMBO A: near-revisit>=0.6 AND actions>=2.5x"] = lambda f: rev(f) and eff(f)
    D["COMBO B: A OR fails>=3 OR turns>=4x"] = lambda f: (rev(f) and eff(f)) or f["fails_lvl"] >= 3 or (g(f, "rel_turns") >= 4 and f["turns_lvl"] >= 20)
    D["COMBO C: (near>=0.6 OR fails>=2 OR rle>=0.5) AND (actions>=2.5x OR turns>=3x)"] = \
        lambda f: (rev(f) or f["fails_lvl"] >= 2 or (f["n_lvl"] >= 30 and f["w_rlegram"] >= 0.5)) and (eff(f) or effturn(f))
combos()

def load_gt(path):
    gt = json.load(open(path))
    out = {}
    for e in gt:
        key = (RUN_TAG.get(e["run"], e["run"]), e["gid"][:4])
        out.setdefault(key, {})[e["level"]] = e
    return out

def runs_of(mask):
    runs = []; s = None
    for i, v in enumerate(mask + [False]):
        if v and s is None: s = i
        if not v and s is not None: runs.append((s, i - 1)); s = None
    return runs

def evaluate(name, det, table, gt, verbose=False):
    hits, misses, delays, fas, fa_levels = [], [], [], [], set()
    on_actions = in_actions = 0
    for key, F in table.items():
        on = [bool(det(f)) for f in F]
        # split alarm runs at level boundaries
        runs = []
        for s, e in runs_of(on):
            cur = s
            for i in range(s, e + 1):
                if i > s and F[i]["level"] != F[i-1]["level"]:
                    runs.append((cur, i - 1)); cur = i
            runs.append((cur, e))
        ivs = []
        for L, e in gt.get(key, {}).items():
            merged = []
            for iv in sorted(e.get("intervals", []), key=lambda x: x["start"]):
                if merged and iv["start"] - merged[-1][1] <= 15:
                    merged[-1] = [merged[-1][0], max(merged[-1][1], iv["end"]), merged[-1][2] + "+" + iv.get("kind", "")]
                else:
                    merged.append([iv["start"], iv["end"], iv.get("kind", "")])
            for (a, b, kind) in merged:
                ivs.append((a - 1, b - 1, L, kind))   # to 0-based
        for (s, e, L, kind) in ivs:
            first = next((i for i in range(s, e + 1) if on[i]), None)
            # alarm already on when the interval starts (fired just before) also counts, delay 0
            if first is None:
                misses.append((key, L, s + 1, e + 1, kind))
            else:
                hits.append((key, L, s + 1, e + 1, kind, first - s))
                delays.append(((first - s), (F[first]["t"] - F[s]["t"]) / 60, (first - s) / max(e - s + 1, 1)))
        stuck_lbl = [False] * len(F)
        for (is_, ie, L, kind) in ivs:
            for i in range(max(0, is_ - SLACK), ie + 1): stuck_lbl[i] = True
        for (s, e) in runs:
            if not stuck_lbl[s]:
                fas.append((key, F[s]["level"], s + 1, e + 1)); fa_levels.add((key, F[s]["level"]))
        for i, v in enumerate(on):
            if v:
                on_actions += 1; in_actions += stuck_lbl[i]
    n_iv = len(hits) + len(misses)
    stuck_levels = {(k, L) for k, d in gt.items() for L, x in d.items() if x.get("intervals")}
    pure = sorted({(k, L) for (k, L, s, e) in fas if (k, L) not in stuck_levels})
    res = dict(name=name, recall=len(hits) / max(n_iv, 1), n_iv=n_iv, hits=len(hits),
               median_delay_actions=float(np.median([d[0] for d in delays])) if delays else None,
               median_delay_min=float(np.median([d[1] for d in delays])) if delays else None,
               median_delay_frac=float(np.median([d[2] for d in delays])) if delays else None,
               false_alarms=len(fas), fa_levels=len(fa_levels), pure_fa_levels=pure,
               precision_actions=in_actions / on_actions if on_actions else None,
               misses=misses, fas=fas, hit_list=hits)
    return res

def _final():
    D = DETECTORS
    loop60 = lambda f: f["n_lvl"] >= 60 and f["w60_near"] >= 0.8
    loop100 = lambda f: f["n_lvl"] >= 100 and f["w100_near"] >= 0.7
    fails = lambda f: f["fails_lvl"] >= 2
    tok = lambda f: f["tok_lvl"] >= 100000
    tokrel = lambda f: g(f, "rel_tok") >= 8 and f["tok_lvl"] >= 50000
    D["P1: loop60 OR fails>=2 OR tokens>=100k"] = lambda f: loop60(f) or fails(f) or tok(f)
    D["P2: loop100 OR fails>=2 OR tokens>=100k"] = lambda f: loop100(f) or fails(f) or tok(f)
    D["P3: loop60 OR fails>=2 OR tokens>=8x own (>=50k)"] = lambda f: loop60(f) or fails(f) or tokrel(f)
    D["P4: P1 OR turns>=4x own"] = lambda f: loop60(f) or fails(f) or tok(f) or (g(f, "rel_turns") >= 4 and f["turns_lvl"] >= 20)
    D["tokens>=8x own (>=50k)"] = tokrel
_final()
DETECTORS["P5: border-2 3 positions 3rd visit OR fails>=2 OR tokens>=100k"] = lambda f: f["b2_third"] >= 3 or f["fails_lvl"] >= 2 or f["tok_lvl"] >= 100000
DETECTORS["P6: border-2 5 positions 3rd visit OR fails>=2 OR tokens>=100k"] = lambda f: f["b2_third"] >= 5 or f["fails_lvl"] >= 2 or f["tok_lvl"] >= 100000

if __name__ == "__main__":
    table = pickle.load(open("table.pkl", "rb"))
    gt = load_gt(sys.argv[1])
    print("%-75s %6s %6s %7s %7s %6s %5s %6s" % ("detector", "recall", "delayA", "delayMin", "delay%", "FalseOn", "pureFAlvl", "prec"))
    allres = []
    for name, det in DETECTORS.items():
        r = evaluate(name, det, table, gt); allres.append(r)
        print("%-75s %3d/%-3d %6s %7s %7s %6d %5d %6s" % (name, r["hits"], r["n_iv"],
              r["median_delay_actions"], "%.0f" % r["median_delay_min"] if r["median_delay_min"] is not None else "-",
              "%.0f%%" % (100 * r["median_delay_frac"]) if r["median_delay_frac"] is not None else "-",
              r["false_alarms"], len(r["pure_fa_levels"]), "%.2f" % r["precision_actions"] if r["precision_actions"] else "-"))
    pickle.dump(allres, open("results.pkl", "wb"))
