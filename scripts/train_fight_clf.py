"""Train + evaluate a window-level fight classifier with performance-grouped cross-validation.

Alarm logic mirrors the rule detector: EMA of window probability over time, alarm when EMA >= THR for HOLD s.
"""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAG = os.environ.get("CACHE_TAG", "_v2")
z = np.load(f"data/cache/fight_windows{TAG}.npz")
X, y, grp, cid, t, names = z["X"], z["y"], z["grp"], z["cid"], z["t"], list(z["features"])
EMA, THR, HOLD = 0.35, 0.5, 1.0

def clip_decision(prob, tt):
    """prob/tt for ONE clip's windows (all pairs, time-sorted) -> (peak_ema, alarm)."""
    o = np.argsort(tt); prob, tt = prob[o], tt[o]
    e, peak, on = 0.0, 0.0, None; alarm = False
    # collapse multiple pairs at the same instant to the max
    for u in np.unique(tt):
        e = (1 - EMA) * e + EMA * prob[tt == u].max(); peak = max(peak, e)
        if e >= THR:
            on = u if on is None else on
            alarm |= (u - on) >= HOLD
        else:
            on = None
    return peak, alarm

def models():
    w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)       # balance classes
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
        "gboost": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0),
    }

def cv(name, make):
    peaks, alarms, truth, ids = {}, {}, {}, {}
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = make().fit(X[tr], y[tr]); pr = m.predict_proba(X[te])[:, 1]
        for c in np.unique(cid[te]):
            sel = cid[te] == c
            peaks[c], alarms[c] = clip_decision(pr[sel], t[te][sel])
            truth[c] = int(c.startswith("violent"))
    cs = sorted(peaks)
    pk = np.array([peaks[c] for c in cs]); al = np.array([alarms[c] for c in cs]); yt = np.array([truth[c] for c in cs])
    print(f"\n[{name}] grouped 5-fold CV on {len(cs)} clips with >=1 window ({yt.sum()} violent / {(1-yt).sum()} non-violent)")
    print(f"  clip AUC (peak EMA prob)  : {roc_auc_score(yt, pk):.3f}")
    print(f"  alarm recall (violent)    : {al[yt==1].mean():.0%}   false-alarm rate (non-violent): {al[yt==0].mean():.0%}")
    return cs, pk, al, yt

if __name__ == "__main__":
    print(f"{len(X)} windows from {len(set(cid))} clips; features: {names}")
    for n, mk in models().items():
        cv(n, mk)
    m = models()["logreg"]().fit(X, y)
    coef = m[-1].coef_[0]
    print("\nlogreg standardized coefficients (+ => more fight-like):")
    for f, c in sorted(zip(names, coef), key=lambda fc: -abs(fc[1])): print(f"   {f:12s} {c:+.2f}")
