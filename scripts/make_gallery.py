"""Outcome gallery on HELD-OUT RWF-2000 validation clips (random sample, live pipeline: YOLO + tracking + flow + classifier).

The classifier is trained on RWF-2000 TRAIN only. Clips are sampled at random (seed 7) and categorised by what the live pipeline
actually did, so nothing is hand-picked.   -> outputs/report/gallery_fights.png, gallery_normal.png
"""
import glob, json, os, pickle, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.backends import YoloPoseOnnx
from safety.fight_model import LearnedFightScorer
from safety.flowfeat import FLOW_FEATURES
from safety.overlay import draw
from safety.pairfeat import FEATURES
from safety.pipeline import SafetyPipeline

SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
THR = 0.64
z = np.load("data/cache/all_windows_v3.npz"); X, ds, split, yy = z["X"], z["ds"], z["split"], z["y"]
tr = np.flatnonzero((ds == "rwf") & (split == "train") & (X[:, 0] <= 4.0))[::3]
model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[tr], yy[tr])
sc = LearnedFightScorer.__new__(LearnedFightScorer)
sc.model, sc.thr, sc.hold, sc.ema, sc.dmax, sc.needs_flow, sc.names = model, THR, 0.0, 0.35, 4.0, True, FEATURES + FLOW_FEATURES
be = YoloPoseOnnx()


def run_clip(path):
    cap = cv2.VideoCapture(path); fps = cap.get(5) or 10.0; i = 0
    new = SafetyPipeline(be, fight=sc.params(), fight_scorer=sc, use_flow=True); old = SafetyPipeline(None)
    ts, sn, so, frames = [], [], [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        t = i / fps; dets = be(f); old.step(dets, t); new.flow.push(f, t); r = new.step(dets, t)
        sn.append(max((s["score"] for s in new.fight.pairs.values()), default=0.0)); so.append(max((s["score"] for s in old.fight.pairs.values()), default=0.0))
        ts.append(t); frames.append(draw(f, new, r, show_feats=False)); i += 1
    cap.release()
    fa = [a.t for a in new.log if a.kind == "FIGHT"]; oa = [a.t for a in old.log if a.kind == "FIGHT"]
    return dict(path=path, t=ts, new=sn, old=so, frames=frames, new_alarm=fa[0] if fa else None, old_alarm=oa[0] if oa else None)


rng = np.random.default_rng(7)
fights = sorted(glob.glob("data/rwf2000/val/Fight/*.avi")); normals = sorted(glob.glob("data/rwf2000/val/NonFight/*.avi"))
pool = [(p, 1) for p in rng.choice(fights, 24, replace=False)] + [(p, 0) for p in rng.choice(normals, 36, replace=False)]
runs = []
for k, (p, tr_) in enumerate(pool):
    r = run_clip(str(p)); r["truth"] = tr_; runs.append(r)
    if (k + 1) % 10 == 0: print(f"{k+1}/{len(pool)} clips", flush=True)
cat = lambda r: ("TRUE POSITIVE: real fight, caught" if r["new_alarm"] is not None else "MISS: real fight, no alarm") if r["truth"] else \
    ("FALSE ALARM: no fight, but alarmed" if r["new_alarm"] is not None else "TRUE NEGATIVE: no fight, no alarm")
live = {c: sum(cat(r) == c for r in runs) for c in sorted({cat(r) for r in runs})}
print("live outcomes on the random sample:", live)
pickle.dump([{k: v for k, v in r.items() if k != "frames"} | {"cat": cat(r)} for r in runs], open("outputs/report/gallery_sample.pkl", "wb"))


def figure(groups, fname, title):
    rows = [r for g in groups for r in [x for x in runs if cat(x) == g][:2]]
    fig = plt.figure(figsize=(14, 2.55 * len(rows) + 0.9), facecolor=SURF)
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1, 1, 1, 1.25], hspace=0.55, wspace=0.06)
    for i, r in enumerate(rows):
        n = len(r["frames"]); pk = int(np.argmax(r["new"])); idx = sorted({max(0, n // 5), pk, min(n - 1, 4 * n // 5)})
        while len(idx) < 3: idx.append(min(n - 1, idx[-1] + 5))
        for j, k in enumerate(idx[:3]):
            ax = fig.add_subplot(gs[i, j]); ax.imshow(cv2.cvtColor(r["frames"][k], cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_xlabel(f"t = {r['t'][k]:.1f} s" + ("  (peak score)" if k == pk else ""), fontsize=8.5, color=INK2, labelpad=2)
            if j == 0:
                bad = ("MISS" in cat(r)) or ("FALSE" in cat(r)); col = "#d03b3b" if bad else "#0a6f0a"
                ax.set_title(f"{cat(r)}     |  {os.path.basename(r['path'])[:-4]}   |  original rules: {'ALARM' if r['old_alarm'] is not None else 'no alarm'}",
                             loc="left", fontsize=10.5, color=col, fontweight="bold", pad=6)
        ax = fig.add_subplot(gs[i, 3]); ax.set_facecolor(SURF)
        ax.plot(r["t"], r["old"], color=ORANGE, lw=1.6); ax.plot(r["t"], r["new"], color=BLUE, lw=2.2)
        ax.axhline(THR, color=INK, lw=0.9); ax.text(r["t"][-1], THR + 0.025, f"alarm threshold {THR}", ha="right", fontsize=8, color=INK2)
        ax.set_ylim(0, 1.02); ax.set_xlim(0, r["t"][-1]); ax.grid(True, color=GRID, lw=0.7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(labelsize=8, colors=MUTED); ax.set_ylabel("fight score", fontsize=8.5, color=INK2)
        if i == 0:
            ax.text(0.02, 0.93, "new model", color=BLUE, fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")
            ax.text(0.02, 0.80, "original rules", color="#c24f20", fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, y=0.995)
    fig.savefig(fname, dpi=80, bbox_inches="tight", facecolor=SURF); plt.close(fig)


figure(["TRUE POSITIVE: real fight, caught", "MISS: real fight, no alarm"], "outputs/report/gallery_fights.png",
       "Real fights from held-out RWF-2000 clips: what the new model catches and misses (random sample, not hand-picked; boxes = tracked people, red = in an alarmed pair)")
figure(["FALSE ALARM: no fight, but alarmed", "TRUE NEGATIVE: no fight, no alarm"], "outputs/report/gallery_normal.png",
       "Normal clips from held-out RWF-2000: false alarms and correct silence (random sample, not hand-picked)")
print("gallery written")
