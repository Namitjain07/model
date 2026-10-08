"""Train/evaluate the fight classifier on pose(14)+flow(6) window features from three datasets.

  E1  official RWF-2000 protocol: fit on train, test on val; threshold chosen ONLY from grouped CV inside train
  E2  leave-one-dataset-out transfer (AIRTLab staged / Surveillance-Fight 2 s clips / RWF-2000 5 s clips)
  E3  final model on everything; threshold from pooled grouped-CV out-of-fold clips of the realistic sets
Feature sets compared: pose14, flow6, pose14+flow6.   Clips with no usable person-pair window can never alarm.
  python scripts/train_all.py [--dmax 4.0] [--final]
"""
import argparse, glob, os, sys
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_windows_all import describe, sources

ap = argparse.ArgumentParser(); ap.add_argument("--dmax", type=float, default=4.0); ap.add_argument("--final", action="store_true"); ap.add_argument("--skip-e1", action="store_true"); ap.add_argument("--e2-sets", default="pose14+flow6")
ap.add_argument("--max-fa", type=float, default=0.15); ap.add_argument("--stride", type=int, default=3, help="train on every n-th window (they are highly correlated)"); ap.add_argument("--save", default="models/fight_gb_v3.joblib"); A = ap.parse_args()
EMA = 0.35
z = np.load("data/cache/all_windows_v3.npz"); X, y, ds, split, grp, cid, t = (z[k] for k in ("X", "y", "ds", "split", "grp", "cid", "t"))
names = list(z["features"]); P14 = list(range(14)); F6 = list(range(14, 20)); ALL = P14 + F6
inr = X[:, 0] <= A.dmax

# every clip (also those without windows) with label / dataset / split
src = sources(); CLIPS = {}
for d, pat in (("airt", "data/cache/airt_cam?_v2/airtlab__*.npz"), ("surv", "data/cache/surv_*_v2/survfight__*.npz"), ("rwf", "data/cache/rwf_v2/rwf2000__*.npz")):
    for p in glob.glob(pat):
        _, label, g, sp, c = describe(p, d, src); CLIPS[c] = (d, label, sp)
clips_of = lambda d, sp=None: sorted(c for c, (dd, _, s) in CLIPS.items() if dd == d and (sp is None or s == sp))
lab = lambda c: CLIPS[c][1]


def decide(P, clips, thr, hold=0.0):
    """P aligned with X rows -> per clip (peak EMA prob, alarm) using only in-reach windows."""
    pk, al = [], []
    for c in clips:
        s = (cid == c) & inr
        if not s.any(): pk.append(0.0); al.append(False); continue
        o = np.argsort(t[s]); pr, tt = P[s][o], t[s][o]; e = peak = 0.0; on = None; alarm = False
        for u in np.unique(tt):
            e = (1 - EMA) * e + EMA * pr[tt == u].max(); peak = max(peak, e)
            if e >= thr: on = u if on is None else on; alarm |= (u - on) >= hold
            else: on = None
        pk.append(peak); al.append(alarm)
    return np.array(pk), np.array(al)


def fit(idx, cols):
    idx = idx[inr[idx]][::A.stride]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced",
                                          random_state=0).fit(X[idx][:, cols], y[idx])


def predict(m, idx, cols):
    P = np.zeros(len(X)); P[idx] = m.predict_proba(X[idx][:, cols])[:, 1]; return P


def oof(idx, cols):
    P = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(idx, groups=grp[idx]):
        m = fit(idx[tr], cols); P[idx[te]] = m.predict_proba(X[idx[te]][:, cols])[:, 1]
    return P


def pick_thr(P, clips, max_fa):
    yt = np.array([lab(c) for c in clips])
    for thr in np.arange(0.4, 0.99, 0.01):
        _, al = decide(P, clips, float(thr))
        if al[yt == 0].mean() <= max_fa: return float(thr)
    return 0.95


def report(title, P, clips, thr):
    yt = np.array([lab(c) for c in clips]); pk, al = decide(P, clips, thr)
    acc = np.mean(al == yt)
    print(f"   {title:42s} AUC {roc_auc_score(yt, pk):.3f} | recall {al[yt==1].mean():4.0%} | false alarms {al[yt==0].mean():4.0%} | acc {acc:.3f}  (thr {thr:.2f}, n={len(clips)})")
    return roc_auc_score(yt, pk)


rows = lambda d, sp=None: np.flatnonzero((ds == d) & ((split == sp) if sp else True))
sets = {"pose14": P14, "flow6": F6, "pose14+flow6": ALL}
print(f"dmax={A.dmax} | windows {len(X)} | clips: airt {len(clips_of('airt'))}, surv {len(clips_of('surv'))}, rwf train {len(clips_of('rwf','train'))} val {len(clips_of('rwf','val'))}")

print("\nE1  OFFICIAL RWF-2000 PROTOCOL (fit on train, test on val; threshold from CV inside train only)")
tr_idx, va_clips = rows("rwf", "train"), clips_of("rwf", "val")
for nm, cols in ({} if A.skip_e1 else sets).items():
    Po = oof(tr_idx, cols); thr = pick_thr(Po, clips_of("rwf", "train"), A.max_fa)
    m = fit(tr_idx, cols); Pv = predict(m, rows("rwf", "val"), cols)
    report(f"RWF train->val [{nm}]", Pv, va_clips, thr)

print("\nE2  LEAVE-ONE-DATASET-OUT transfer (threshold = out-of-fold on the training datasets)")
for hold_out in ("airt", "surv", "rwf"):
    tr_rows = np.flatnonzero(ds != hold_out); te_clips = clips_of(hold_out) if hold_out != "rwf" else clips_of("rwf", "val")
    tr_clips = [c for c in CLIPS if CLIPS[c][0] != hold_out and not (CLIPS[c][0] == "rwf" and CLIPS[c][2] == "val")]
    for nm, cols in {k: v for k, v in sets.items() if k in A.e2_sets.split(",")}.items():
        Po = oof(tr_rows, cols); thr = pick_thr(Po, tr_clips, A.max_fa)
        m = fit(tr_rows, cols); Pt = predict(m, np.flatnonzero((ds == hold_out) & ((split == "val") if hold_out == "rwf" else True)), cols)
        report(f"hold out {hold_out} [{nm}]", Pt, te_clips, thr)

if A.final:
    print("\nE3  FINAL model on everything (pose14+flow6); threshold from pooled out-of-fold clips of the realistic sets")
    allrows = np.arange(len(X)); Po = oof(allrows, ALL)
    real = [c for c in CLIPS if CLIPS[c][0] in ("surv", "rwf")]
    thr = pick_thr(Po, real, A.max_fa)
    for d in ("airt", "surv", "rwf"): report(f"OOF {d}", Po, clips_of(d), thr)
    report("OOF surv+rwf (selection set)", Po, real, thr)
    np.save("data/cache/oof_v3.npy", Po)
    print("\n   operating curve (recall / false alarms, out-of-fold):")
    print("   thr   | surv          | rwf           | airt (staged hugs/high-fives/jumps)")
    for th in (0.5, 0.6, 0.64, 0.7, 0.75, 0.8, 0.85, 0.9):
        cells = []
        for d in ("surv", "rwf", "airt"):
            cl = clips_of(d); yt = np.array([lab(c) for c in cl]); _, al = decide(Po, cl, th)
            cells.append(f"{al[yt==1].mean():4.0%} / {al[yt==0].mean():4.0%}")
        print(f"   {th:.2f}  | {cells[0]:13s} | {cells[1]:13s} | {cells[2]}")
    m = fit(allrows, ALL)
    joblib.dump(dict(model=m, thr=thr, hold=0.0, ema=EMA, dmax=A.dmax, features=names, flow=True), A.save); print("saved", A.save)
