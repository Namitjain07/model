"""Before/after numbers on the 400 held-out RWF-2000 validation clips -> outputs/report/results.json

 rules   = original hand-tuned fight score (pose only), default alarm logic
 prev    = previous learned model (pose only; trained on AIRTLab + Surveillance-Fight, NEVER on RWF-2000)
 new     = pose + optical-flow classifier trained on RWF-2000 TRAIN only (no validation clip seen), threshold 0.64
           (chosen from CV inside train, see README E1)
Also: false-alarm events/hour curves for the final model from out-of-fold predictions.
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety.fight_model import LearnedFightScorer
from safety.pipeline import SafetyPipeline
from safety.replay import run_cached

VAL = sorted(glob.glob("data/cache/rwf_v2/rwf2000__val__*.npz"))
key = lambda p: os.path.basename(p)[:-4].split("__", 3)[3]
truth = lambda p: int("__Fight__" in p)


def run_pipe(args):
    p, which = args
    if which == "rules":
        pipe = SafetyPipeline(None)
    else:
        sc = LearnedFightScorer("models/fight_gb_combined.joblib"); pipe = SafetyPipeline(None, fight=sc.params(), fight_scorer=sc)
    r = run_cached(pipe, p)
    return key(p), float(r["score"].max()) if len(r["score"]) else 0.0, any(a.kind == "FIGHT" for a in r["alerts"])


if __name__ == "__main__":
    FA_ONLY = "--fa-only" in sys.argv
    names = [key(p) for p in VAL]; y = np.array([truth(p) for p in VAL]); res = {"names": names, "truth": y.tolist(), "det": {}}
    with Pool(4) as pool:
        for which in ("rules", "prev"):
            out = dict((k, (s, a)) for k, s, a in pool.imap_unordered(run_pipe, [(p, which) for p in VAL], chunksize=8))
            res["det"][which] = dict(peak=[out[n][0] for n in names], alarm=[bool(out[n][1]) for n in names]); print(which, "done", flush=True)

    # new model: RWF train windows only
    z = np.load("data/cache/all_windows_v3.npz"); X, ds, split, cid, t, yy = z["X"], z["ds"], z["split"], z["cid"], z["t"], z["y"]
    inr = X[:, 0] <= 4.0; tr = np.flatnonzero((ds == "rwf") & (split == "train") & inr)[::3]
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[tr], yy[tr])
    THR, EMA = 0.64, 0.35; pk, al = [], []
    vidx = np.flatnonzero((ds == "rwf") & (split == "val") & inr)
    P = np.zeros(len(X)); P[vidx] = m.predict_proba(X[vidx])[:, 1]
    for p in VAL:
        cls = "Fight" if truth(p) else "NonFight"; name = os.path.basename(p)[:-4].split("__")[3]
        s = (cid == f"rwf:val:{cls}:{name}") & inr
        if not s.any(): pk.append(0.0); al.append(False); continue
        o = np.argsort(t[s]); pr, tt = P[s][o], t[s][o]; e = peak = 0.0
        for u in np.unique(tt): e = (1 - EMA) * e + EMA * pr[tt == u].max(); peak = max(peak, e)
        pk.append(float(peak)); al.append(bool(peak >= THR))
    res["det"]["new"] = dict(peak=pk, alarm=al, thr=THR)

    for k, d in res["det"].items():
        a = np.array(d["alarm"]); pkv = np.array(d["peak"])
        d.update(tp=int((a & (y == 1)).sum()), fn=int((~a & (y == 1)).sum()), fp=int((a & (y == 0)).sum()), tn=int((~a & (y == 0)).sum()),
                 auc=float(roc_auc_score(y, pkv)))
        fpr, tpr, _ = roc_curve(y, pkv); d["roc"] = [fpr.tolist(), tpr.tolist()]
        print(f"{k:6s} AUC {d['auc']:.3f} | TP {d['tp']} FN {d['fn']} FP {d['fp']} TN {d['tn']} | recall {d['tp']/200:.0%} FPR {d['fp']/200:.0%} acc {(d['tp']+d['tn'])/400:.3f}")

    # false-alarm events/hour from the saved out-of-fold predictions (final model protocol, RWF-2000 non-fight footage)
    from report_fa import fa_curves
    res.update(fa_curves())
    json.dump(res, open("outputs/report/results.json", "w")); print("saved results.json")
