"""Pick alarm threshold/hold from OUT-OF-FOLD predictions (grouped CV), then fit on all windows and save.

  python scripts/finalize_fight_model.py [--max-fa 0.15]
Saves models/fight_gb.joblib = dict(model, thr, hold, ema, features).  Estimates printed are out-of-fold,
single-room staged data: treat as optimistic for real CCTV.
"""
import argparse, os, sys
import joblib, numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fight_clf as T

ap = argparse.ArgumentParser(); ap.add_argument("--max-fa", type=float, default=0.15)
ap.add_argument("--dmax", type=float, default=6.0, help="only train/score pairs within this distance (torso lengths); smaller = fewer far-apart false alarms, lower recall")
a = ap.parse_args()
X, y, grp, cid, t = T.X, T.y, T.grp, T.cid, T.t
make = T.models()["gboost"]
inr = X[:, 0] <= a.dmax                                  # window is within reach
P = np.zeros(len(X))
for tr, te in GroupKFold(5).split(X, y, grp):
    trm = tr[inr[tr]]
    P[te] = make().fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]
P[~inr] = 0.0
clips = sorted(set(cid)); yt = np.array([int(c.startswith("violent")) for c in clips])
has = {c: bool((inr & (cid == c)).any()) for c in clips}  # clips with no in-reach window can never alarm

def table(hold):
    T.HOLD = hold; out = []
    for thr in np.arange(0.60, 0.97, 0.01):
        T.THR = float(thr)
        al = np.array([T.clip_decision(P[cid == c], t[cid == c])[1] if has[c] else False for c in clips])
        out.append((float(thr), al[yt == 1].mean(), al[yt == 0].mean()))
    return out

best = None
for hold in (0.0, 0.5, 1.0):
    tb = table(hold)
    print(f"\nhold={hold:.1f}s  thr: recall / false-alarm")
    print("  " + "  ".join(f"{thr:.2f}:{r:.0%}/{f:.0%}" for thr, r, f in tb[::4]))
    ok = [(thr, r, f) for thr, r, f in tb if f <= a.max_fa]
    if ok:
        thr, r, f = ok[0]
        if best is None or r > best[2]: best = (hold, thr, r, f)
hold, thr, r, f = best
print(f"\nCHOSEN (dmax={a.dmax}, max false alarms {a.max_fa:.0%}): hold={hold}s thr={thr:.2f}  -> out-of-fold recall {r:.0%}, false alarms {f:.0%}")
final = make().fit(X[inr], y[inr])
os.makedirs("models", exist_ok=True)
joblib.dump(dict(model=final, thr=thr, hold=hold, ema=T.EMA, dmax=a.dmax, features=list(T.names), oof_recall=r, oof_fa=f), f"models/fight_gb{T.TAG}.joblib")
print(f"saved models/fight_gb{T.TAG}.joblib")
