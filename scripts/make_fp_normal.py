"""Frames at which the final v3 model false-alarms on continuous NORMAL footage (no fights in any of these videos)."""
import os, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.backends import YoloPoseOnnx
from safety.overlay import draw
from safety.pipeline import learned_pipeline

be = YoloPoseOnnx(); shots = []
for v, lo, hi in (("vtest.avi", 0, 70), ("face-demographics-walking.mp4", 30, 55), ("people-detection.mp4", 20, 40), ("worker-zone-detection.mp4", 30, 55)):
    pipe = learned_pipeline(be, "models/fight_gb_v3.joblib")            # default threshold 0.64
    cap = cv2.VideoCapture(f"data/videos/{v}"); fps = cap.get(5); every = max(1, round(fps / 10)); i = 0; nal = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = i / fps
        if i % every == 0 and lo <= t <= hi:
            r = pipe.process(f, t)
            for a in r.alerts:
                if a.kind == "FIGHT":
                    nal += 1; d = a.detail
                    shots.append((v, t, cv2.cvtColor(draw(f, pipe, r, show_feats=False, k=1.6 if f.shape[1] > 1000 else 1.0), cv2.COLOR_BGR2RGB), d))
        i += 1
    cap.release(); print(v, "false alarms:", nal, flush=True)
sel = shots[:6]; cols = 3; rows = int(np.ceil(len(sel) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(14, 3.9 * rows + 0.8), facecolor="#fcfcfb")
for ax, (v, t, img, d) in zip(np.ravel(axs), sel):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f"{v}  at {t:.1f} s", loc="left", fontsize=10.5, color="#d03b3b", fontweight="bold")
    ax.set_xlabel(f"score {d.get('score', 0):.2f} | pair {d.get('dist', 0):.1f} body-lengths apart | motion energy {d.get('energy_max', 0):.1f} | strikes {d.get('strikes', 0):.0f}", fontsize=8.5, color="#52514e")
for ax in np.ravel(axs)[len(sel):]: ax.axis("off")
fig.suptitle("False alarms on normal footage (no fights in any of these videos): people walking and crossing paths, flagged at threshold 0.64", x=0.01, ha="left", fontsize=13, y=0.995)
fig.tight_layout(); fig.savefig("outputs/report/false_alarms_normal_footage.png", dpi=80, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", len(shots), "alarm frames, showing", len(sel))
