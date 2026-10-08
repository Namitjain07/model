"""Data audit before training: (1) metadata shortcuts, (2) near-duplicate leakage between training sources and held-out sets.

(1) For each dataset: cross-validated AUC of a classifier that sees ONLY video metadata (width, height, aspect, fps, duration).
    ~0.5 = no shortcut; high = the label can be guessed without looking at people at all.
(2) dHash (64-bit) of 3 frames per clip; clips whose hash is within `--hamming` bits of any held-out clip are reported/dropped.
Writes outputs/report/audit.json (incl. the list of duplicate training clips to exclude).
"""
import argparse, glob, json, os, sys
import cv2, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ap = argparse.ArgumentParser(); ap.add_argument("--hamming", type=int, default=6); a = ap.parse_args()
SETS = {   # name -> (glob, label fn, role)
    "rwf_train": ("data/rwf2000/train/*/*.avi", lambda p: int("/Fight/" in p), "train"),
    "rwf_val": ("data/rwf2000/val/*/*.avi", lambda p: int("/Fight/" in p), "heldout"),
    "surv": ("data/survfight/*/*.mp4", lambda p: int("/fight/" in p), "heldout"),
    "airt": ("data/airtlab/*/cam1/*.mp4", lambda p: int("/violent/" in p), "heldout"),
    "rlvs": ("data/rlvs/raw/*/*", lambda p: int("/Violence/" in p), "train"),
}


def dhash(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return ((s[:, 1:] > s[:, :-1]).flatten())


def probe(p):
    c = cv2.VideoCapture(p); n = int(c.get(7)); fps = c.get(5) or 25.0; w, h = int(c.get(3)), int(c.get(4)); hs = []
    for q in (0.2, 0.5, 0.8):
        c.set(1, int(max(n - 1, 0) * q)); ok, f = c.read()
        if ok: hs.append(dhash(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
    c.release()
    return dict(w=w, h=h, ar=w / max(h, 1), fps=fps, dur=n / fps if fps else 0, hashes=hs)


meta = {}
for name, (pat, lab, role) in SETS.items():
    fs = sorted(glob.glob(pat)); meta[name] = [(f, lab(f), probe(f)) for f in fs]; print(name, len(fs), "clips probed", flush=True)

print("\n(1) METADATA-ONLY SHORTCUT (CV AUC of resolution/aspect/fps/duration -> label; 0.5 = none)")
out = {"shortcut_auc": {}}
for name, rows in meta.items():
    X = np.array([[r[2]["w"], r[2]["h"], r[2]["ar"], r[2]["fps"], min(r[2]["dur"], 60)] for r in rows]); y = np.array([r[1] for r in rows])
    if len(set(y)) < 2: continue
    p = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, max_iter=80, random_state=0), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p); out["shortcut_auc"][name] = float(auc)
    print(f"   {name:10s} n={len(y):5d}  metadata-only AUC {auc:.3f}" + ("   <-- shortcut!" if auc > 0.7 else ""))

print(f"\n(2) NEAR-DUPLICATES (dHash Hamming <= {a.hamming}) of held-out clips among TRAIN sources")
H = {k: [np.array(r[2]["hashes"]) for r in v] for k, v in meta.items()}
held = [(k, i, h) for k in ("rwf_val", "surv", "airt") for i, hs in enumerate(H[k]) for h in hs]
Hh = np.array([h for _, _, h in held], bool)
dups = {}
for k in ("rlvs", "rwf_train"):
    bad = []
    for i, hs in enumerate(H[k]):
        if not len(hs): continue
        d = (hs[:, None, :] != Hh[None, :, :]).sum(-1)            # (frames, held-out hashes)
        j = np.unravel_index(d.argmin(), d.shape)
        if d.min() <= a.hamming: bad.append((meta[k][i][0], held[j[1]][0], int(held[j[1]][1]), int(d.min())))
    dups[k] = bad; print(f"   {k:10s}: {len(bad)} of {len(H[k])} clips are near-duplicates of a held-out clip")
    for b in bad[:4]: print("      e.g.", os.path.basename(b[0]), "~", b[1], "clip#", b[2], "dist", b[3])
out["near_duplicates"] = {k: [b[0] for b in v] for k, v in dups.items()}
# which HELD-OUT clips are affected (so their results can be recomputed without them)
out["affected_heldout"] = {}
for k, v in dups.items():
    for tr_path, ho_name, ho_idx, dist in v:
        out["affected_heldout"].setdefault(ho_name, set()).add(meta[ho_name][ho_idx][0])
out["affected_heldout"] = {k: sorted(v) for k, v in out["affected_heldout"].items()}
out["pairs"] = [(k, b[0], b[1], meta[b[1]][b[2]][0], b[3]) for k, v in dups.items() for b in v]
import pickle; pickle.dump({k: [(f, l, {x: y for x, y in m.items() if x != "hashes"}) for f, l, m in v] for k, v in meta.items()}, open("outputs/report/audit_meta.pkl", "wb"))
print("held-out clips affected by a near-duplicate in a TRAIN source:", {k: len(v) for k, v in out["affected_heldout"].items()})
os.makedirs("outputs/report", exist_ok=True); json.dump(out, open("outputs/report/audit.json", "w"), indent=1)
print("\nwrote outputs/report/audit.json")
