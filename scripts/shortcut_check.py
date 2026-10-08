"""Is the RWF-2000 result driven by metadata shortcuts?  Stratified AUC of the held-out model (trained on RWF train only).

Within a stratum (same resolution, or same fps) the metadata carries no label information, so the AUC there is the
shortcut-free estimate.  Compared with the overall AUC and with the metadata-only AUC.
"""
import json, os
import cv2, numpy as np
from sklearn.metrics import roc_auc_score

res = json.load(open("outputs/report/results.json")); y = np.array(res["truth"]); names = res["names"]
peak = np.array(res["det"]["new"]["peak"]); rules = np.array(res["det"]["rules"]["peak"])
meta = []
for n, t in zip(names, y):
    p = f"data/rwf2000/val/{'Fight' if t else 'NonFight'}/{n}.avi"; c = cv2.VideoCapture(p)
    meta.append((int(c.get(3)), int(c.get(4)), round(c.get(5)), int(c.get(7)))); c.release()
W = np.array([m[0] for m in meta]); H = np.array([m[1] for m in meta]); F = np.array([m[2] for m in meta]); N = np.array([m[3] for m in meta])
print(f"overall: model AUC {roc_auc_score(y, peak):.3f}  (400 val clips)")
def strata(keys, title, min_each=12):
    print(f"\nstratified by {title}:")
    tot_w = 0; acc = 0.0
    for k in sorted(set(keys), key=lambda k: -(keys == k).sum()):
        s = keys == k
        if (y[s] == 1).sum() >= min_each and (y[s] == 0).sum() >= min_each:
            a = roc_auc_score(y[s], peak[s]); w = (y[s] == 1).sum() * (y[s] == 0).sum(); tot_w += w; acc += a * w
            print(f"   {str(k):18s} n={s.sum():3d} (fight {int((y[s]==1).sum())}/non {int((y[s]==0).sum())})  model AUC {a:.3f}   rules AUC {roc_auc_score(y[s], rules[s]):.3f}")
        else:
            print(f"   {str(k):18s} n={s.sum():3d} (fight {int((y[s]==1).sum())}/non {int((y[s]==0).sum())})  too few of one class to evaluate")
    if tot_w: print(f"   -> pair-weighted AUC over evaluable strata: {acc/tot_w:.3f}")
strata(np.array([f"{w}x{h}" for w, h in zip(W, H)]), "resolution")
strata(F.astype(str), "fps")
print("\nlabel balance by resolution (fight share):", {f"{w}x{h}": f"{y[(W==w)&(H==h)].mean():.0%} of {((W==w)&(H==h)).sum()}" for w, h in sorted({(a, b) for a, b in zip(W, H)}, key=lambda x: -((W == x[0]) & (H == x[1])).sum())[:5]})
