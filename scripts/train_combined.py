"""Train ONE fight classifier on AIRTLab (staged, close) + Surveillance Camera Fight Dataset (real, CCTV-style).

Out-of-fold (grouped by performance / source video) predictions give per-dataset recall & false-alarm tables;
the alarm threshold is the lowest one with <=15% false alarms on the *surveillance* clips (the realistic target).
  python scripts/train_combined.py [--balance]     # --balance: weight so each dataset counts equally
"""
import argparse, os, sys
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fight_clf as T          # clip_decision + EMA settings
from safety.pairfeat import FEATURES

ap = argparse.ArgumentParser(); ap.add_argument("--balance", action="store_true"); ap.add_argument("--max-fa", type=float, default=0.15)
ap.add_argument("--save", default="models/fight_gb_combined.joblib")
ap.add_argument("--dmax", type=float, default=4.0, help="only train/score pairs within this distance (torso lengths); 4.0 removed all normal-footage false alarms"); a = ap.parse_args()
TAG = T.TAG
A = np.load(f"data/cache/fight_windows{TAG}.npz"); S = np.load(f"data/cache/surv_windows{TAG}.npz")
X = np.vstack([A["X"], S["X"]]); y = np.concatenate([A["y"], S["y"]])
grp = np.concatenate([["A:" + g for g in A["grp"]], ["S:" + g for g in S["grp"]]])
cid = np.concatenate([["A:" + c for c in A["cid"]], ["S:" + c for c in S["cid"]]]); t = np.concatenate([A["t"], S["t"]])
isA = np.array([c.startswith("A:") for c in cid])
w = np.where(isA, 0.5 / isA.sum(), 0.5 / (~isA).sum()) * len(X) if a.balance else np.ones(len(X))
mk = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0)

inr = X[:, 0] <= a.dmax
P = np.zeros(len(X))
for tr, te in GroupKFold(5).split(X, y, grp):
    trm = tr[inr[tr]]
    P[te] = mk().fit(X[trm], y[trm], sample_weight=w[trm]).predict_proba(X[te])[:, 1]
P[~inr] = 0.0
clipsA = sorted({c for c in cid if c.startswith("A:")}); clipsS = sorted({c for c in cid if c.startswith("S:")})
# every clip must be listed even if it has no window (cannot alarm) -> add the clips with zero windows
import glob
def all_surv():
    return sorted({"S:" + os.path.basename(p)[:-4].split("__")[1] + ":" + os.path.basename(p)[:-4].split("__")[2] for g in (f"data/cache/surv_fight{TAG}/*.npz", f"data/cache/surv_nofight{TAG}/*.npz") for p in glob.glob(g)})
clipsS = all_surv()
lab = lambda c: int(c.split(":")[1] in ("violent", "fight"))

def decide(clips, thr, hold):
    T.THR, T.HOLD = thr, hold; pk, al = [], []
    for c in clips:
        s = cid == c
        k, v = T.clip_decision(P[s], t[s]) if s.any() else (0.0, False); pk.append(k); al.append(v)
    return np.array(pk), np.array(al)

yA = np.array([lab(c) for c in clipsA]); yS = np.array([lab(c) for c in clipsS])
print(f"dmax={a.dmax}; windows {len(X)} (AIRTLab {isA.sum()}, surveillance {(~isA).sum()}); clips AIRTLab {len(clipsA)}, surveillance {len(clipsS)}; balance={a.balance}")
for hold in (0.0, 0.5):
    pkA, _ = decide(clipsA, 0.5, hold); pkS, _ = decide(clipsS, 0.5, hold)
    print(f"\nhold={hold}s  clip AUC: AIRTLab {roc_auc_score(yA, pkA):.3f} | surveillance {roc_auc_score(yS, pkS):.3f}")
    print("  thr   | surveillance recall/FA | AIRTLab recall/FA")
    for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        _, aS = decide(clipsS, thr, hold); _, aA = decide(clipsA, thr, hold)
        print(f"  {thr:.2f}  |   {aS[yS==1].mean():4.0%} / {aS[yS==0].mean():4.0%}        |   {aA[yA==1].mean():4.0%} / {aA[yA==0].mean():4.0%}")
best = None
for hold in (0.0, 0.5):
    for thr in np.arange(0.5, 0.99, 0.01):
        _, aS = decide(clipsS, float(thr), hold)
        if aS[yS == 0].mean() <= a.max_fa:
            r = aS[yS == 1].mean()
            if best is None or r > best[2]: best = (hold, float(thr), r)
            break
hold, thr, r = best
_, aS = decide(clipsS, thr, hold); _, aA = decide(clipsA, thr, hold)
print(f"\nCHOSEN (<= {a.max_fa:.0%} false alarms on surveillance): hold={hold}s thr={thr:.2f}")
print(f"  surveillance OOF: recall {aS[yS==1].mean():.0%}, false alarms {aS[yS==0].mean():.0%} | AIRTLab OOF at same thr: recall {aA[yA==1].mean():.0%}, false alarms {aA[yA==0].mean():.0%}")
final = mk().fit(X[inr], y[inr], sample_weight=w[inr])
joblib.dump(dict(model=final, thr=thr, hold=hold, ema=T.EMA, dmax=a.dmax, features=list(FEATURES),
                 oof_surv_recall=float(aS[yS == 1].mean()), oof_surv_fa=float(aS[yS == 0].mean()),
                 oof_airt_recall=float(aA[yA == 1].mean()), oof_airt_fa=float(aA[yA == 0].mean())), a.save)
print("saved", a.save)
