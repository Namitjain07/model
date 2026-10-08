"""Does gating to people within reach (pair distance <= Dmax torso lengths) remove far-apart false alarms?

Train AND score only windows with dist <= Dmax (grouped CV). Clips with no in-reach window can never alarm
(counted as misses for violent clips). Operating point chosen as the lowest threshold with <=15% false alarms.
"""
import os, sys
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fight_clf as T
from eval_fights import labels

X, y, grp, cid, t = T.X, T.y, T.grp, T.cid, T.t
clips = sorted(set(cid)); yt = np.array([int(c.startswith("violent")) for c in clips]); lab = labels()
print(f"{'Dmax':>5s} {'AUC':>6s} {'thr':>5s} {'recall':>7s} {'false':>6s}   hug  jump  highfive  greet | stab gunshot club punch slap")
for Dmax in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
    inr = X[:, 0] <= Dmax; P = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, grp):
        trm = tr[inr[tr]]; m = T.models()["gboost"]().fit(X[trm], y[trm]); P[te] = m.predict_proba(X[te])[:, 1]
    P[~inr] = 0.0
    T.HOLD = 0.5
    pk = np.array([T.clip_decision(P[cid == c], t[cid == c])[0] if (inr & (cid == c)).any() else 0.0 for c in clips])
    auc = roc_auc_score(yt, pk)
    best = None
    for thr in np.arange(0.5, 0.99, 0.01):
        T.THR = float(thr)
        al = np.array([T.clip_decision(P[cid == c], t[cid == c])[1] if (inr & (cid == c)).any() else False for c in clips])
        if al[yt == 0].mean() <= 0.15:
            best = (thr, al); break
    if best is None: print(f"{Dmax:5.1f} {auc:6.3f}  (no threshold reaches 15% FA)"); continue
    thr, al = best
    def rate(kind, act):
        s = [a for c, a in zip(clips, al) if c.startswith(kind) and act in lab[(c.split(':')[0], int(c.split(':')[2]))]]
        return f"{np.mean(s):4.0%}" if s else "  - "
    print(f"{Dmax:5.1f} {auc:6.3f} {thr:5.2f} {al[yt==1].mean():7.0%} {al[yt==0].mean():6.0%}   "
          f"{rate('non-violent','hug')} {rate('non-violent','jump')}  {rate('non-violent','highfive')}     {rate('non-violent','greet')} | "
          f"{rate('violent','stab')} {rate('violent','gunshot')}    {rate('violent','club')} {rate('violent','punch')}  {rate('violent','slap')}")
