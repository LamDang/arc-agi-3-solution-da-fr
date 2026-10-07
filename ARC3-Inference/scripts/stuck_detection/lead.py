"""Onset of each detector per stuck episode, against the episode start and the 2nd game over."""
import os, pickle
from evaluate import DETECTORS, load_gt
T = pickle.load(open("table.pkl", "rb")); gt = load_gt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json"))
dets = {"2nd GO": DETECTORS["fails>=2 on level"], "3pos": DETECTORS["border-2 hash: 3 positions 3rd visit"],
        "5pos": DETECTORS["border-2 hash: 5 positions 3rd visit"], "loop60": DETECTORS["near-revisit>=0.8 (60)"],
        "P1": DETECTORS["P1: loop60 OR fails>=2 OR tokens>=100k"], "P5": DETECTORS[[n for n in DETECTORS if n.startswith("P5")][0]]}
print("episode (stuck start-end)".ljust(40), "".join(k.rjust(16) for k in dets))
for key, levels in sorted(gt.items()):
    for L, e in sorted(levels.items()):
        if not e["intervals"]: continue
        F = T[key]; idx = [i for i, f in enumerate(F) if f["level"] == L]
        s = min(iv["start"] for iv in e["intervals"]); end = max(iv["end"] for iv in e["intervals"])
        t0 = F[s-1]["t"]
        cells = []
        for name, d in dets.items():
            on = next((i for i in idx if d(F[i])), None)
            cells.append("-" if on is None else f"{on+1-s:+d}a/{(F[on]['t']-t0)/60:+.0f}m")
        print(f"{key[0]} {key[1]} L{L} ({s}-{end})".ljust(40), "".join(c.rjust(16) for c in cells))
