"""Visual check on real surveillance clips with a HELD-OUT model (trained on everything except this clip's source video).
  python scripts/viz_survfight.py fight:fi022 noFight:nofi001 ...   -> outputs/airt/surv_heldout.jpg
"""
import os, sys, re
import cv2, joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_survfight import sources
from safety.backends import YoloPoseOnnx
from safety.fight_model import LearnedFightScorer
from safety.overlay import draw, banner
from safety.pipeline import SafetyPipeline

TAG = os.environ.get("CACHE_TAG", "_v2")
A = np.load(f"data/cache/fight_windows{TAG}.npz"); S = np.load(f"data/cache/surv_windows{TAG}.npz")
X = np.vstack([A["X"], S["X"]]); y = np.concatenate([A["y"], S["y"]]); grp = np.concatenate([["A:" + g for g in A["grp"]], ["S:" + g for g in S["grp"]]])
meta = joblib.load("models/fight_gb_combined.joblib"); src = sources(); be = YoloPoseOnnx(); rows = []
for spec in sys.argv[1:]:
    kind, name = spec.split(":"); g = "S:" + src.get(name, name)
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[grp != g], y[grp != g])
    sc = LearnedFightScorer.__new__(LearnedFightScorer); sc.model, sc.thr, sc.hold, sc.ema, sc.dmax = m, meta["thr"], meta["hold"], meta["ema"], 6.0
    pipe = SafetyPipeline(be, fight=sc.params(), fight_scorer=sc)
    cap = cv2.VideoCapture(f"data/survfight/{kind}/{name}.mp4"); fps = cap.get(5) or 25; every = max(1, round(fps / 10)); i = 0; imgs = []; first = None
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % every == 0:
            r = pipe.process(f, i / fps); o = draw(f, pipe, r, show_feats=False)
            if first is None and any(a.kind == "FIGHT" for a in r.alerts): first = len(imgs)
            best = max((st["score"] for st in pipe.fight.pairs.values()), default=0.0)
            imgs.append((o, best))
        i += 1
    n = len(imgs); pick = [max(0, first - 3), first, min(n - 1, first + 3)] if first is not None else [n // 5, n // 2, 4 * n // 5]
    tiles = []
    for k in pick:
        o, b = imgs[k]; o = cv2.resize(o, (420, int(420 * o.shape[0] / o.shape[1]))); cv2.putText(o, f"{name} score {b:.2f}", (6, 20), 0, 0.6, (255, 255, 255), 2); tiles.append(o)
    verdict = "ALARM" if first is not None else "no alarm"
    cv2.putText(tiles[0], f"truth={'FIGHT' if kind=='fight' else 'no fight'} -> {verdict}", (6, 42), 0, 0.6, (0, 0, 255) if verdict == "ALARM" else (0, 200, 0), 2)
    h = min(t.shape[0] for t in tiles); rows.append(np.hstack([t[:h] for t in tiles]))
    print(f"{spec:16s} truth={'FIGHT' if kind=='fight' else 'no fight':8s} -> {verdict}")
w = min(r.shape[1] for r in rows); os.makedirs("outputs/airt", exist_ok=True)
cv2.imwrite("outputs/airt/surv_heldout.jpg", np.vstack([r[:, :w] for r in rows])); print("saved outputs/airt/surv_heldout.jpg")
