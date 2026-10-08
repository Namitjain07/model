"""Deployment-oriented operating curves from the saved out-of-fold predictions (data/cache/oof_v3.npy).

Besides clip recall / clip false-alarm rate, report FALSE-ALARM EVENTS PER HOUR on non-fight footage
(an always-on camera cares about this, not about the per-clip rate).  Alarm = EMA prob >= thr for >= hold s;
re-armed once EMA falls 0.10 below thr.  Non-fight footage = RWF-2000 NonFight (5 s clips) and Surveillance noFight (~2 s).
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_windows_all import describe, sources

z = np.load("data/cache/all_windows_v3.npz"); X, ds, cid, t = z["X"], z["ds"], z["cid"], z["t"]; P = np.load("data/cache/oof_v3.npy")
inr = X[:, 0] <= 4.0; EMA = 0.35; src = sources(); CL = {}
for d, pat in (("airt", "data/cache/airt_cam?_v2/airtlab__*.npz"), ("surv", "data/cache/surv_*_v2/survfight__*.npz"), ("rwf", "data/cache/rwf_v2/rwf2000__*.npz")):
    for p in glob.glob(pat):
        _, label, g, sp, c = describe(p, d, src); CL[c] = (d, label)
order = {c: np.flatnonzero((cid == c) & inr) for c in CL}
dur = {}
for c, idx in order.items():
    dur[c] = float(t[(cid == c)].max()) + 0.1 if (cid == c).any() else (5.0 if CL[c][0] == "rwf" else 2.0)


def events(c, thr, hold):
    idx = order[c]
    if len(idx) == 0: return 0
    o = idx[np.argsort(t[idx])]; pr, tt = P[o], t[o]; e = 0.0; on = None; armed = True; n = 0
    for u in np.unique(tt):
        e = (1 - EMA) * e + EMA * pr[tt == u].max()
        if e >= thr and armed:
            on = u if on is None else on
            if u - on >= hold: n += 1; armed = False
        elif e < thr:
            on = None
            if e < thr - 0.10: armed = True
    return n


def table(dset_filter, title):
    cs = [c for c in CL if dset_filter(CL[c][0])]
    neg = [c for c in cs if CL[c][1] == 0]; pos = [c for c in cs if CL[c][1] == 1]
    hours = sum(dur[c] for c in neg) / 3600
    print(f"\n{title}: {len(pos)} fight clips, {len(neg)} non-fight clips = {hours:.2f} h of non-fight footage")
    print("  thr  hold | fight-clip recall | non-fight clips alarmed | false-alarm events / hour")
    for thr in (0.75, 0.80, 0.85, 0.90, 0.95):
        for hold in (0.0, 1.0, 2.0):
            rec = np.mean([events(c, thr, hold) > 0 for c in pos]); fa = np.mean([events(c, thr, hold) > 0 for c in neg])
            ev = sum(events(c, thr, hold) for c in neg) / hours
            print(f"  {thr:.2f} {hold:.1f}s |      {rec:4.0%}        |        {fa:4.0%}             | {ev:7.1f}")

table(lambda d: d == "rwf", "RWF-2000 (all 2000 clips, out-of-fold)")
table(lambda d: d == "surv", "Surveillance-Fight (300 clips, out-of-fold)")
