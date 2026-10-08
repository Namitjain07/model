"""Honest evaluation of teacher/student predictions on the HELD-OUT sets.

  python gpu/evaluate.py --pred teacher=gpu_out/teacher_A/predictions.csv --pred student=gpu_out/student_A/predictions.csv --out gpu_out/metrics.json

 * The operating threshold is chosen on the DEV split only (the highest-recall point whose dev false-alarm rate is <= --max-fa),
   then applied unchanged to every held-out set -> no threshold tuning on the test data.
 * Held-out = RWF-2000 val (official) + Surveillance-Fight + AIRTLab (protocol A).  With protocol B (--final) those sets were trained on,
   so the script refuses to call them held-out.
 * Resolution-stratified AUC: metadata alone predicts the label in RLVS (AUC 0.95), so a model that only learned sharpness would score well
   overall but ~0.5 inside a resolution bucket.  Per-bucket AUC is the shortcut check.
 * False alarms per analysed hour: non-fight clips flagged / hours of analysed non-fight footage (clips are <=5 s analysis windows; this is
   a window-level rate, not an event rate on continuous video -- see README).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--pred", action="append", required=True, help="name=path/to/predictions.csv (repeatable)")
ap.add_argument("--out", default="gpu_out/metrics.json"); ap.add_argument("--clean", action="store_true", help="drop held-out clips flagged as near-duplicates of training sources (needs the manifest 'affected' column)"); ap.add_argument("--max-fa", type=float, nargs="+", default=[0.05, 0.02])
a = ap.parse_args(); args_clean = a.clean
WIN = 5.0


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(y) >= 10 and len(set(y)) == 2 else None


def pick_thr(dev, max_fa):
    neg, pos = dev[dev.label == 0].p_fight.values, dev[dev.label == 1].p_fight.values
    for thr in np.unique(np.r_[np.linspace(0.02, 0.98, 97), neg]):
        if (neg >= thr).mean() <= max_fa: return float(thr)
    return 0.99


def op(d, thr):
    pos, neg = d[d.label == 1], d[d.label == 0]
    fa = float((neg.p_fight >= thr).mean()) if len(neg) else None
    hours = float(np.minimum(neg.dur, WIN).sum() / 3600) if len(neg) else None
    return dict(recall=float((pos.p_fight >= thr).mean()) if len(pos) else None, false_alarm_rate=fa,
                false_alarms_per_analysed_hour=(float((neg.p_fight >= thr).sum() / hours) if hours else None), n_pos=len(pos), n_neg=len(neg))


res = {}
for spec in a.pred:
    name, path = spec.split("=", 1); df = pd.read_csv(path); dev = df[df.split == "dev"]
    held = df[df.split == "heldout"]
    if "affected" in df.columns and args_clean: held = held[held.affected == 0]
    r = dict(n_dev=len(dev), dev_auc=auc(dev.label.values, dev.p_fight.values), heldout={}, note="")
    if held.empty: r["note"] = "no held-out clips in predictions (protocol B / smoke run): only dev metrics are valid"
    for fa in a.max_fa:
        thr = pick_thr(dev, fa) if len(dev) else 0.5
        for ds, d in held.groupby("dataset"):
            h = r["heldout"].setdefault(ds, dict(n=len(d), auc=auc(d.label.values, d.p_fight.values), by_bucket={}, ops={}))
            h["ops"][f"dev_fa<={fa:.0%}"] = dict(thr=thr, **op(d, thr))
            for b, g in d.groupby("bucket"):
                h["by_bucket"][b] = dict(n=len(g), pos=int(g.label.sum()), auc=auc(g.label.values, g.p_fight.values))
        if len(held):
            r.setdefault("pooled", {})[f"dev_fa<={fa:.0%}"] = dict(thr=thr, **op(held, thr))
    res[name] = r

Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
for name, r in res.items():
    print(f"\n== {name}  (dev n={r['n_dev']}, dev AUC {r['dev_auc']})  {r['note']}")
    for ds, h in r["heldout"].items():
        print(f"  {ds:5s} n={h['n']:4d} AUC {h['auc'] if h['auc'] is None else round(h['auc'], 3)}")
        for k, o in h["ops"].items():
            f = lambda v, p=0: "n/a" if v is None else (f"{v:.0%}" if p == 0 else f"{v:.1f}")
            print(f"        {k:12s} thr {o['thr']:.2f} | recall {f(o['recall'])} | false alarms {f(o['false_alarm_rate'])} ({f(o['false_alarms_per_analysed_hour'], 1)}/analysed h)")
        print("        AUC inside resolution buckets:", {b: (None if v["auc"] is None else round(v["auc"], 3)) for b, v in h["by_bucket"].items()})
print("\nwrote", a.out)
