"""Sanity checks + operating points for the learned fight classifier (all out-of-fold, grouped by performance)."""
import os, sys, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fight_clf import X, y, grp, cid, t, names, models, clip_decision
import train_fight_clf as T
from eval_fights import labels

cam = np.array([c.split(":")[1] for c in cid])
clips = sorted(set(cid)); yt = np.array([int(c.startswith("violent")) for c in clips])

def oof(make, mask_train=None, mask_test=None):
    P = np.full(len(X), np.nan)
    for tr, te in GroupKFold(5).split(X, y, grp):
        if mask_train is not None: tr = tr[mask_train[tr]]
        if mask_test is not None: te = te[mask_test[te]]
        P[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return P

def clip_peaks(P, thr=0.5):
    pk, al = {}, {}
    for c in clips:
        s = (cid == c) & np.isfinite(P)
        if s.any(): pk[c], al[c] = clip_decision(P[s], t[s]) if thr == T.THR else (None, None)
    return pk, al

def auc_of(pk):
    cs = [c for c in clips if c in pk]; return roc_auc_score([int(c.startswith("violent")) for c in cs], [pk[c] for c in cs])

if __name__ == "__main__":
    # 1) trivial-feature controls: if these alone separate the classes, the dataset has a shortcut
    dur = {c: t[cid == c].max() for c in clips}; nwin = {c: (cid == c).sum() for c in clips}
    meand = {c: X[cid == c, 0].mean() for c in clips}
    print("== shortcut controls (clip AUC, 0.5 = no information) ==")
    print(f"   clip duration       : {auc_of(dur):.3f}\n   number of windows   : {auc_of(nwin):.3f}\n   mean pair distance  : {auc_of(meand):.3f}")

    make = models()["gboost"]
    P = oof(make)
    pk, _ = clip_peaks(P)
    print(f"\n== gboost, grouped 5-fold OOF: clip AUC {auc_of(pk):.3f}")

    # 2) cross-camera + unseen performances: train on cam1 only, test on cam2 only (and vice versa)
    for a, b in (("cam1", "cam2"), ("cam2", "cam1")):
        Pc = oof(make, mask_train=(cam == a), mask_test=(cam == b)); pkc, _ = clip_peaks(Pc)
        print(f"   train {a} -> test {b} (new performances, new camera): clip AUC {auc_of(pkc):.3f}")

    # 3) operating points
    cs = sorted(pk); ytr = np.array([int(c.startswith("violent")) for c in cs]); pv = np.array([pk[c] for c in cs])
    print("\n== operating points (peak-EMA threshold -> recall on violent / false alarms on non-violent) ==")
    for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        al = pv >= thr
        print(f"   thr {thr:.2f}: recall {al[ytr==1].mean():4.0%}   false alarms {al[ytr==0].mean():4.0%}")
    ok = [thr for thr in np.arange(0.5, 0.99, 0.01) if (pv[ytr == 0] >= thr).mean() <= 0.15]
    thr = float(ok[0]) if ok else 0.9
    print(f"\n== at thr {thr:.2f} (lowest threshold with <=15% false alarms on this out-of-fold data) ==")
    lab = labels(); by_v, by_n = collections.defaultdict(list), collections.defaultdict(list)
    for c, p in zip(cs, pv):
        kind, _, num = c.split(":"); acts = lab[(kind, int(num))]
        for a in acts: (by_v if kind == "violent" else by_n)[a].append(p >= thr)
    print("   violent actions - recall   :", {a: f"{np.mean(v):.0%}(n={len(v)})" for a, v in sorted(by_v.items(), key=lambda kv: -len(kv[1]))})
    print("   non-violent - false alarms :", {a: f"{np.mean(v):.0%}(n={len(v)})" for a, v in sorted(by_n.items(), key=lambda kv: -len(kv[1]))})
    for cm in ("cam1", "cam2"):
        s = np.array([c.split(":")[1] == cm for c in cs])
        print(f"   {cm}: recall {(pv>=thr)[(ytr==1)&s].mean():.0%}, false alarms {(pv>=thr)[(ytr==0)&s].mean():.0%}")
