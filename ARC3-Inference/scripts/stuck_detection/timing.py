import os, pickle, sys
from evaluate import DETECTORS, evaluate, load_gt
table = pickle.load(open("table.pkl", "rb")); gt = load_gt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json"))
for name in sys.argv[1:]:
    det = [n for n in DETECTORS if n.startswith(name)][0]
    r = evaluate(det, DETECTORS[det], table, gt)
    print("##", det)
    for h in sorted(r["hit_list"]) + [m + (None,) for m in r["misses"]]:
        key, L, s, e, kind, d = h
        F = table[key]
        if d is None:
            print(f"  MISS {key[0]} {key[1]} L{L} {s}-{e} ({e-s+1}a, {(F[e-1]['t']-F[s-1]['t'])/60:.0f}m) {kind}"); continue
        a = s - 1 + d
        # first alarm action anywhere in level before or at a
        first = a
        while first > 0 and F[first-1]["level"] == L and DETECTORS[det](F[first-1]): first -= 1
        tot = (F[e-1]["t"] - F[s-1]["t"]) / 60
        after = (F[e-1]["t"] - F[max(a, s-1)]["t"]) / 60
        lvl_start = next(i for i in range(len(F)) if F[i]["level"] == L)
        print(f"  {key[0]} {key[1]} L{L} stuck {s}-{e} ({e-s+1}a,{tot:.0f}m) {kind[:20]}: alarm on since action {first+1} "
              f"({(F[first]['t']-F[lvl_start]['t'])/60:.0f} min into level); stuck minutes left after alarm {min(after,tot):.0f}")
    print("  false onsets:", [(k[0], k[1], L, s, e) for (k, L, s, e) in r["fas"]])
    print("  pure FA levels:", r["pure_fa_levels"])
