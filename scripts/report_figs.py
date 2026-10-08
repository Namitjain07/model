"""Figures 1-2 from outputs/report/results.json (reference palette: blue #2a78d6, orange #eb6834, aqua #1baf7a)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
res = json.load(open("outputs/report/results.json")); D = res["det"]; y = np.array(res["truth"])

# ---------------- Figure 1: ROC + operating points on 400 held-out RWF-2000 clips ----------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
ent = [("rules", "Original rules", ORANGE, "o", "(pose only)"), ("prev", "Previous learned model", AQUA, "s", "(pose only, never saw RWF)"),
       ("new", "New model", BLUE, "D", "(pose + motion, trained on RWF train)")]
for k, name, col, mk, sub in ent:
    f, t_ = D[k]["roc"]; ax.plot(f, t_, color=col, lw=2, solid_capstyle="round")
    r, fa = D[k]["tp"] / 200, D[k]["fp"] / 200
    ax.plot([fa], [r], marker=mk, ms=9, color=col, mec=SURF, mew=2, zorder=5)
ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
ax.text(0.97, 0.05, "coin flip", color=MUTED, fontsize=9, ha="right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_xlabel("false-alarm rate (non-fight clips flagged)"); ax.set_ylabel("recall (fight clips flagged)")
ax.set_title("ROC on 400 held-out RWF-2000 clips", loc="left", fontsize=12, color=INK, pad=12)
ax.text(0.50, 0.50, f"New model   AUC {D['new']['auc']:.2f}", color=BLUE, fontsize=10.5, fontweight="bold")
ax.text(0.50, 0.42, f"Previous   AUC {D['prev']['auc']:.2f}", color="#14805a", fontsize=10.5, fontweight="bold")
ax.text(0.50, 0.34, f"Original rules   AUC {D['rules']['auc']:.2f}", color="#c24f20", fontsize=10.5, fontweight="bold")
ax.text(0.50, 0.285, "rules rank fights fairly well, but their alarm\nthreshold almost never fires (marker near 0)", color=INK2, fontsize=8.5, va="top")
ax.text(0.03, 0.02, "marker = each detector's actual alarm setting", color=MUTED, fontsize=8.5)

rows = [("rules", "Original rules", ORANGE), ("prev", "Previous learned\n(never saw RWF)", AQUA), ("newzs", "New model\n(never saw RWF)", BLUE), ("new", "New model\n(trained on RWF train)", BLUE)]
yy = np.arange(len(rows))[::-1] * 1.0
for i, (k, name, col) in zip(yy, rows):
    d = D[k]; rec, fa = d["tp"] / 200, d["fp"] / 200
    bx.barh(i + 0.17, rec * 100, height=0.3, color=BLUE, zorder=3); bx.barh(i - 0.17, fa * 100, height=0.3, color=ORANGE, zorder=3)
    bx.text(rec * 100 + 1.5, i + 0.17, f"{rec:.0%}  recall  ({d['tp']} of 200 fights caught)", va="center", fontsize=9.5, color=INK)
    bx.text(fa * 100 + 1.5, i - 0.17, f"{fa:.0%}  false alarms  ({d['fp']} of 200 normal clips)", va="center", fontsize=9.5, color=INK)
bx.set_yticks(yy); bx.set_yticklabels([r[1] for r in rows], fontsize=10, color=INK); bx.set_xlim(0, 150); bx.set_xticks([0, 25, 50, 75, 100]); bx.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
bx.grid(axis="y", visible=False); bx.set_title("At each detector's own operating point", loc="left", fontsize=12, color=INK, pad=12)
bx.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)], labels=["recall (higher is better)", "false-alarm rate (lower is better)"],
          loc="upper right", frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Before vs after: same 400 held-out clips (200 fights, 200 non-fights; validation split, none used for training)", x=0.01, ha="left", fontsize=13.5, color=INK, y=1.0)
fig.text(0.01, -0.02, "Accuracy: rules 51.5%  |  previous 66.7%  |  new (never saw RWF) 76.5%  |  new (trained on RWF train) 82.3%.   Colours validated for colour-blind separation; all marks direct-labelled.", fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig("outputs/report/fig1_before_after.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: false alarms per hour vs recall (one trade-off curve) ----------------
fig, ax = plt.subplots(figsize=(9.8, 5.8)); C = res["fa_per_hour"]
p0 = C["0.0"]; ax.plot([p[2] for p in p0], [p[1] * 100 for p in p0], color=BLUE, lw=2, zorder=3)
for h, mk in (("1.0", "o"), ("2.0", "s")):
    ax.scatter([p[2] for p in C[h]], [p[1] * 100 for p in C[h]], s=26, facecolors="none", edgecolors=MUTED, linewidths=1.1, marker=mk, zorder=2)
for thr in (0.74, 0.80, 0.86, 0.92):
    q = min(p0, key=lambda p: abs(p[0] - thr)); ax.plot([q[2]], [q[1] * 100], "o", ms=9, color=BLUE, mec=SURF, mew=2, zorder=5)
    ax.annotate(f"threshold {q[0]:.2f}:  {q[1]:.0%} of fights caught,\n{q[2]:.0f} false alarms per hour", (q[2], q[1] * 100), xytext=(q[2] + 14, q[1] * 100 - 17),
                fontsize=9.5, color=INK, va="top", arrowprops=dict(arrowstyle="-", color=AXIS, lw=1, shrinkA=0, shrinkB=5))
ax.set_xlim(0, 185); ax.set_ylim(0, 100); ax.set_xlabel("false-alarm events per hour on non-fight footage"); ax.set_ylabel("fight clips detected (%)")
ax.axvspan(0, 5, color="#cde2fb", alpha=0.55, lw=0); ax.text(6.5, 95, "<= 5 per hour (shaded)", color=INK2, fontsize=9)
ax.set_title("What a camera would actually do: one trade-off curve", loc="left", fontsize=12.5, color=INK, pad=14)
ax.text(112, 14, "hollow markers = requiring 1 s (circles) or 2 s (squares) of sustained\nscore. They land on the same curve: persistence is no better\nthan raising the threshold.", fontsize=9, color=INK2)
fig.text(0.01, -0.03, f"Out-of-fold predictions on RWF-2000 (never trained on the clip being scored). {res['n_nonfight']:,} non-fight clips = {res['nonfight_hours']:.2f} h of busy real "
         "surveillance-style footage;\nquiet scenes would false-alarm less. Each point is one threshold.", fontsize=8.5, color=MUTED)
fig.tight_layout(); fig.savefig("outputs/report/fig2_false_alarms_per_hour.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("figures written")
