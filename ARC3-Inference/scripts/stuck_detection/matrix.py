import os, pickle, sys
from evaluate import DETECTORS, evaluate, load_gt
table = pickle.load(open("table.pkl", "rb")); gt = load_gt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json"))
names = [n for n in DETECTORS if any(s in n for s in sys.argv[1:])] if len(sys.argv) > 1 else list(DETECTORS)
res = {n: evaluate(n, DETECTORS[n], table, gt) for n in names}
eps = sorted({(h[0], h[1], h[2], h[3], h[4]) for r in res.values() for h in r["hit_list"]} | {m for r in res.values() for m in r["misses"]})
print("episode".ljust(42), " | ".join(n[:14].ljust(14) for n in names))
for ep in eps:
    key, L, s, e, kind = ep
    F = table[key]; mins = (F[e-1]["t"] - F[s-1]["t"]) / 60
    cells = []
    for n in names:
        h = [x for x in res[n]["hit_list"] if x[:5] == ep]
        cells.append(("+%d" % h[0][5] if h else "MISS").ljust(14))
    print(f"{key[0]} {key[1]} L{L} {s}-{e} ({e-s+1}a,{mins:.0f}m) {kind[:12]}".ljust(42), " | ".join(cells))
print()
for n in names:
    print(n, "false onsets:", [(k[0], k[1], L, s, e) for (k, L, s, e) in res[n]["fas"]])
