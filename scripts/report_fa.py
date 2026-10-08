"""False-alarm EVENTS PER HOUR vs recall from the saved out-of-fold predictions (data/cache/oof_v3.npy).

Denominator = ALL RWF-2000 non-fight clips (5 s each), including clips with no usable person pair (they cannot alarm but are
still footage).  Alarm = EMA prob >= thr for >= hold s; re-armed once the EMA falls 0.10 below thr.
  python scripts/report_fa.py      # updates the fa_per_hour section of outputs/report/results.json
"""
import glob, json, os
import numpy as np

EMA = 0.35


def fa_curves():
    z = np.load("data/cache/all_windows_v3.npz"); X, cid, t = z["X"], z["cid"], z["t"]; O = np.load("data/cache/oof_v3.npy")
    inr = X[:, 0] <= 4.0; cl = []
    for f in sorted(glob.glob("data/cache/rwf_v2/rwf2000__*.npz")):
        _, sp, cls, nm = os.path.basename(f)[:-4].split("__"); cl.append(f"rwf:{sp}:{cls}:{nm}")
    lab = {c: int(":Fight:" in c) for c in cl}
    order = {c: np.flatnonzero((cid == c) & inr) for c in cl}

    def events(c, thr, hold):
        idx = order[c]
        if len(idx) == 0: return 0
        o = idx[np.argsort(t[idx])]; pr, tt = O[o], t[o]; e = 0.0; on = None; armed = True; n = 0
        for u in np.unique(tt):
            e = (1 - EMA) * e + EMA * pr[tt == u].max()
            if e >= thr and armed:
                on = u if on is None else on
                if u - on >= hold: n += 1; armed = False
            elif e < thr:
                on = None
                if e < thr - 0.10: armed = True
        return n

    pos = [c for c in cl if lab[c] == 1]; neg = [c for c in cl if lab[c] == 0]; hours = len(neg) * 5.0 / 3600
    curves = {}
    for hold in (0.0, 1.0, 2.0):
        curves[str(hold)] = [(float(thr), float(np.mean([events(c, thr, hold) > 0 for c in pos])), float(sum(events(c, thr, hold) for c in neg) / hours))
                             for thr in np.arange(0.50, 0.97, 0.02)]
    return dict(fa_per_hour=curves, nonfight_hours=hours, n_nonfight=len(neg))


if __name__ == "__main__":
    res = json.load(open("outputs/report/results.json")); res.update(fa_curves())
    json.dump(res, open("outputs/report/results.json", "w"))
    c = res["fa_per_hour"]; print(f"non-fight footage: {res['n_nonfight']} clips = {res['nonfight_hours']:.2f} h")
    for h, pts in c.items():
        print(f"hold {h}s:", "  ".join(f"thr{p[0]:.2f}->{p[1]:.0%}@{p[2]:.0f}/h" for p in pts[::3]))
