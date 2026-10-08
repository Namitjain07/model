"""Visual check of the learned fight detector on REAL clips with a HELD-OUT model per clip.

For each clip the classifier is trained on every other performance (cam1+cam2 of this one excluded), then the
full pipeline runs on the real video and an annotated contact sheet is written to outputs/airt/.
  python scripts/viz_fight_clips.py violent:cam1:7 non-violent:cam1:1
"""
import os, sys
import joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_video import make_backend, run
from eval_fights import labels
from safety.fight_model import LearnedFightScorer
from safety.pipeline import SafetyPipeline

z = np.load("data/cache/fight_windows.npz"); X, y, grp = z["X"], z["y"], z["grp"]
meta = joblib.load("models/fight_gb.joblib"); lab = labels(); be = make_backend("yolo")
os.makedirs("outputs/airt", exist_ok=True)
for spec in sys.argv[1:]:
    kind, cam, num = spec.split(":"); g = f"{kind}:{num}"; tr = grp != g
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[tr], y[tr])
    sc = LearnedFightScorer.__new__(LearnedFightScorer); sc.model, sc.thr, sc.hold, sc.ema, sc.dmax = m, meta["thr"], meta["hold"], meta["ema"], meta.get("dmax", 6.0)
    pipe = SafetyPipeline(be, fight=sc.params(), fight_scorer=sc)
    s = run(f"data/airtlab/{kind}/{cam}/{num}.mp4", be, every=3, quiet=True, pipe=pipe, tiles=6, width=960,
            sheet=f"outputs/airt/heldout_{kind}_{cam}_{num}.jpg")
    fights = [(a["t"], a["detail"] if "detail" in a else None) for a in s["alerts"] if a["kind"] == "FIGHT"]
    print(f"{spec:22s} truth={'VIOLENT' if kind=='violent' else 'non-violent':11s} actions={lab[(kind,int(num))]} -> "
          f"{'FIGHT ALARM @' + str([a[0] for a in fights]) if fights else 'no alarm'}")
