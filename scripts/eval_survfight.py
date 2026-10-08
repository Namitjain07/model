"""Cross-dataset evaluation on the Surveillance Camera Fight Dataset (real YouTube surveillance fights).

 (a) rule-based score, (b) AIRTLab-trained classifier ZERO-SHOT (never saw these clips -> no leakage),
 (c) grouped CV inside this dataset (group = source YouTube video), (d) training on AIRTLab+this dataset.
Clip-level truth: fight/ vs noFight/.  Coverage = clips where >=2 people were tracked at all.
"""
import glob, os, re, sys
import joblib, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety.events import FightDetector
from safety.pairfeat import FEATURES, pair_features
from safety.pipeline import SafetyPipeline
from safety.replay import load_dets, run_cached
from safety.tracker import Tracker
import train_fight_clf as T

TAG = os.environ.get("CACHE_TAG", "_v2")
WIN = f"data/cache/surv_windows{TAG}.npz"
SURV_GLOBS = [f"data/cache/surv_fight{TAG}/survfight__*.npz", f"data/cache/surv_nofight{TAG}/survfight__*.npz"]


def surv_files():
    return sorted(sum((glob.glob(g) for g in SURV_GLOBS), []))


def sources():
    src, cur = {}, None
    for ln in open("data/survfight/videos.txt").read().splitlines():
        ln = ln.strip()
        if ln.startswith("http"): cur = ln
        elif re.match(r"(no)?fi\d+:", ln): src[ln.split(":")[0]] = cur
    return src


def build_windows():
    fd, src = FightDetector(), sources()
    X, y, grp, cid, tt = [], [], [], [], []
    files = surv_files()
    for p in files:
        _, kind, name = os.path.basename(p)[:-4].split("__")
        tr = Tracker(); frames, _ = load_dets(p)
        for t, dets in frames:
            tracks = [x for x in tr.update(dets, t) if max(x.box[2] - x.box[0], x.box[3] - x.box[1]) >= fd.p.min_size_px]
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    f = pair_features(fd, tracks[i], tracks[j], t)
                    if f is not None and f[0] < 6.0:
                        X.append(f); y.append(int(kind == "fight")); grp.append(src.get(name, name)); cid.append(f"{kind}:{name}"); tt.append(t)
    np.savez_compressed(WIN, X=np.array(X), y=np.array(y), grp=np.array(grp), cid=np.array(cid), t=np.array(tt))
    return files


def clip_scores(P, cid, t, clips, thr, hold):
    T.THR, T.HOLD = thr, hold
    pk, al = {}, {}
    for c in clips:
        s = cid == c
        pk[c], al[c] = T.clip_decision(P[s], t[s]) if s.any() else (0.0, False)
    return pk, al


def summarize(title, clips, pk, al):
    yt = np.array([int(c.startswith("fight")) for c in clips]); pv = np.array([pk[c] for c in clips]); a = np.array([al[c] for c in clips])
    print(f"  {title:44s} AUC {roc_auc_score(yt, pv):.3f} | alarm recall {a[yt==1].mean():4.0%} | false alarms {a[yt==0].mean():4.0%}")
    return pv, yt


if __name__ == "__main__":
    files = surv_files()
    if not os.path.exists(WIN) or "--rebuild" in sys.argv: build_windows()
    z = np.load(WIN); X, y, grp, cid, t = z["X"], z["y"], z["grp"], z["cid"], z["t"]
    allclips = sorted({os.path.basename(p)[:-4].split("__")[1] + ":" + os.path.basename(p)[:-4].split("__")[2] for p in files})
    yt_all = np.array([int(c.startswith("fight")) for c in allclips])
    print(f"{len(allclips)} clips ({yt_all.sum()} fight / {(1-yt_all).sum()} noFight), {len(X)} windows")

    # coverage: can the pipeline even see an interacting pair?
    have = {c: (cid == c).sum() for c in allclips}
    for k, n in (("fight", 1), ("noFight", 0)):
        cs = [c for c in allclips if c.startswith(k)]
        print(f"coverage {k:8s}: {np.mean([have[c] > 0 for c in cs]):.0%} of clips have >=1 usable person-pair window")

    print("\n(a) RULE-BASED score (original hand-tuned detector)")
    pk, al = {}, {}
    for p in files:
        _, kind, name = os.path.basename(p)[:-4].split("__"); c = f"{kind}:{name}"
        r = run_cached(SafetyPipeline(None), p); pk[c] = float(r["score"].max()) if len(r["score"]) else 0.0
        al[c] = any(a.kind == "FIGHT" for a in r["alerts"])
    summarize("rules", allclips, pk, al)

    meta = joblib.load(f"models/fight_gb{TAG}.joblib"); m = meta["model"]
    print(f"\n(b) AIRTLab-trained classifier, ZERO-SHOT (thr {meta['thr']:.2f}, hold {meta['hold']}s from AIRTLab out-of-fold)")
    P = m.predict_proba(X)[:, 1] if len(X) else np.array([])
    pk, al = clip_scores(P, cid, t, allclips, meta["thr"], meta["hold"])
    pv, yt = summarize("zero-shot @ AIRTLab threshold", allclips, pk, al)
    print("  threshold sweep (zero-shot):  " + "  ".join(
        f"{th:.2f}:{np.mean([clip_scores(P, cid, t, [c], th, meta['hold'])[1][c] for c in allclips if c.startswith('fight')]):.0%}/"
        f"{np.mean([clip_scores(P, cid, t, [c], th, meta['hold'])[1][c] for c in allclips if c.startswith('noFight')]):.0%}" for th in (0.5, 0.6, 0.7, 0.8, 0.9)) + "   (recall/false-alarm)")

    print("\n(c) in-dataset grouped CV (group = source YouTube video), gradient boosting")
    Po = np.zeros(len(X)); mk = T.models()["gboost"]
    for tr, te in GroupKFold(5).split(X, y, grp):
        Po[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pk, al = clip_scores(Po, cid, t, allclips, 0.5, 0.5); summarize("surv-only CV @ thr 0.50", allclips, pk, al)

    print("\n(d) train on AIRTLab + this dataset (grouped CV; AIRTLab groups = performance, here = source video)")
    A = np.load(f"data/cache/fight_windows{TAG}.npz"); XA, yA, gA = A["X"], A["y"], A["grp"]
    fa = [i for i in range(len(FEATURES))][:XA.shape[1]]
    assert XA.shape[1] == X.shape[1], "feature sets differ - rebuild windows"
    Pd = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, grp):
        mm = mk().fit(np.vstack([XA, X[tr]]), np.concatenate([yA, y[tr]])); Pd[te] = mm.predict_proba(X[te])[:, 1]
    pk, al = clip_scores(Pd, cid, t, allclips, meta["thr"], meta["hold"]); summarize("AIRTLab + surv(CV) @ AIRTLab thr", allclips, pk, al)
    pk, al = clip_scores(Pd, cid, t, allclips, 0.5, 0.5); summarize("AIRTLab + surv(CV) @ thr 0.50", allclips, pk, al)

    print("\n(e) reverse: train on this dataset only, test on AIRTLab (all clips; zero-shot the other way)")
    ms = mk().fit(X, y); PA = ms.predict_proba(XA)[:, 1]
    cA, tA, yAc = A["cid"], A["t"], None
    clipsA = sorted(set(cA)); ytA = np.array([int(c.startswith("violent")) for c in clipsA])
    T.THR, T.HOLD = 0.5, 0.5
    pkA = np.array([T.clip_decision(PA[cA == c], tA[cA == c])[0] for c in clipsA])
    print(f"  surv-trained -> AIRTLab: clip AUC {roc_auc_score(ytA, pkA):.3f}")
