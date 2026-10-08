"""Re-run the pose14+flow6 fight-model evaluations WITHOUT the leakage found by scripts/audit_datasets.py.

 * training clips that are near-duplicates (dHash) of a held-out clip are removed from training (rwf-train: 61 clips)
 * results are reported on ALL held-out clips and on the CLEAN subset (held-out clips that have no near-duplicate anywhere in train)
 * AUC comes with a 95% bootstrap interval (clips resampled with replacement) -- the sets are small, so differences < ~0.04 are noise
 * the Surveillance-Fight transfer test trains on AIRTLab + RWF *train* only (RWF val is a different held-out set whose own duplicates of
   surv clips were not audited).
  python scripts/eval_clean.py            -> prints the table, writes outputs/report/clean_eval.json
"""
import glob, json, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_windows_all import describe, sources

DMAX, EMA, MAXFA = 4.0, 0.35, 0.15
z = np.load("data/cache/all_windows_v3.npz"); X, y, ds, split, grp, cid, t = (z[k] for k in ("X", "y", "ds", "split", "grp", "cid", "t"))
ALL = list(range(20)); inr = X[:, 0] <= DMAX
audit = json.load(open("outputs/report/audit.json")); norm = os.path.normpath
dup_train = {norm(p) for p in audit["near_duplicates"]["rwf_train"]}
aff = {norm(p) for v in audit["affected_heldout"].values() for p in v}

src = sources(); CLIPS = {}
for d, pat in (("airt", "data/cache/airt_cam?_v2/airtlab__*.npz"), ("surv", "data/cache/surv_*_v2/survfight__*.npz"), ("rwf", "data/cache/rwf_v2/rwf2000__*.npz")):
    for p in glob.glob(pat):
        vid, label, g, sp, c = describe(p, d, src); CLIPS[c] = dict(ds=d, label=label, split=sp, path=norm(vid), dup=norm(vid) in dup_train, aff=norm(vid) in aff)
print(f"clips {len(CLIPS)} | train near-duplicates dropped: {sum(v['dup'] for v in CLIPS.values())} | affected held-out clips: {sum(v['aff'] for v in CLIPS.values())}")
clips_of = lambda d, sp=None, clean=False: sorted(c for c, v in CLIPS.items() if v["ds"] == d and (sp is None or v["split"] == sp) and not (clean and v["aff"]))
lab = lambda c: CLIPS[c]["label"]
keep_row = np.array([not CLIPS[c]["dup"] for c in cid])


_rows_of = {}
for i, c in enumerate(cid):
    if inr[i]: _rows_of.setdefault(c, []).append(i)
_rows_of = {c: np.array(v) for c, v in _rows_of.items()}


def peaks(P, clips):
    """Per clip: peak of the EMA of the per-timestep max pair probability (only in-reach pairs); 0 if the clip has no usable pair."""
    pk = []
    for c in clips:
        r = _rows_of.get(c)
        if r is None: pk.append(0.0); continue
        o = np.argsort(t[r], kind="stable"); r = r[o]; tt = t[r]; pr = P[r]; e = peak = 0.0
        for u in np.unique(tt):
            e = (1 - EMA) * e + EMA * pr[tt == u].max(); peak = max(peak, e)
        pk.append(peak)
    return np.array(pk)


def decide(P, clips, thr):
    pk = peaks(P, clips); return pk, pk >= thr


def fit(idx):
    idx = idx[inr[idx]][::3]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[idx], y[idx])


def oof(idx):
    P = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(idx, groups=grp[idx]): P[idx[te]] = fit(idx[tr]).predict_proba(X[idx[te]])[:, 1]
    return P


def pick_thr(P, clips):
    yt = np.array([lab(c) for c in clips]); pk = peaks(P, clips)
    for thr in np.arange(0.4, 0.99, 0.01):
        if (pk[yt == 0] >= thr).mean() <= MAXFA: return float(thr)
    return 0.95


def boot_auc(yt, pk, n=1000, seed=0):
    r = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = r.integers(0, len(yt), len(yt))
        if len(set(yt[i])) == 2: v.append(roc_auc_score(yt[i], pk[i]))
    return np.percentile(v, [2.5, 97.5])


res = {}
def report(name, P, clips, thr):
    yt = np.array([lab(c) for c in clips]); pk, al = decide(P, clips, thr); lo, hi = boot_auc(yt, pk)
    r = dict(n=len(clips), auc=float(roc_auc_score(yt, pk)), auc_ci=[float(lo), float(hi)], recall=float(al[yt == 1].mean()), false_alarm=float(al[yt == 0].mean()), thr=thr)
    res[name] = r
    print(f"   {name:46s} n={r['n']:3d} AUC {r['auc']:.3f} [{lo:.3f},{hi:.3f}] | recall {r['recall']:4.0%} | false alarms {r['false_alarm']:4.0%} (thr {thr:.2f})")


rows = lambda d, sp=None: np.flatnonzero((ds == d) & ((split == sp) if sp else True) & keep_row)
print("\nE1  RWF-2000 official split: train (near-duplicates removed) -> val")
tr = rows("rwf", "train"); thr = pick_thr(oof(tr), [c for c in clips_of("rwf", "train") if not CLIPS[c]["dup"]]); m = fit(tr)
P = np.zeros(len(X)); vi = np.flatnonzero((ds == "rwf") & (split == "val")); P[vi] = m.predict_proba(X[vi])[:, 1]
report("RWF val, all", P, clips_of("rwf", "val"), thr); report("RWF val, clean subset", P, clips_of("rwf", "val", True), thr)

print("\nE2  transfer to Surveillance-Fight: train AIRTLab + RWF train (near-duplicates removed)")
tr = np.flatnonzero(((ds == "airt") | ((ds == "rwf") & (split == "train"))) & keep_row)
tcl = [c for c in CLIPS if (CLIPS[c]["ds"] == "airt" or (CLIPS[c]["ds"] == "rwf" and CLIPS[c]["split"] == "train")) and not CLIPS[c]["dup"]]
thr = pick_thr(oof(tr), tcl); m = fit(tr); P = np.zeros(len(X)); si = np.flatnonzero(ds == "surv"); P[si] = m.predict_proba(X[si])[:, 1]
report("Surveillance-Fight, all", P, clips_of("surv"), thr); report("Surveillance-Fight, clean subset", P, clips_of("surv", None, True), thr)

print("\nE2b transfer to AIRTLab (staged hugs/high-fives/jumps): train Surv + RWF train")
tr = np.flatnonzero((((ds == "surv")) | ((ds == "rwf") & (split == "train"))) & keep_row)
tcl = [c for c in CLIPS if (CLIPS[c]["ds"] == "surv" or (CLIPS[c]["ds"] == "rwf" and CLIPS[c]["split"] == "train")) and not CLIPS[c]["dup"]]
thr = pick_thr(oof(tr), tcl); m = fit(tr); P = np.zeros(len(X)); ai = np.flatnonzero(ds == "airt"); P[ai] = m.predict_proba(X[ai])[:, 1]
report("AIRTLab", P, clips_of("airt"), thr)
json.dump(res, open("outputs/report/clean_eval.json", "w"), indent=1); print("\nwrote outputs/report/clean_eval.json")
