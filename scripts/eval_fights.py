"""Evaluate the fight detector on cached AIRTLab detections against real per-clip labels.

  python scripts/eval_fights.py            # baseline thresholds
Clip-level ground truth: violent/ vs non-violent/.  cam1 and cam2 are two views of the SAME performance.
"""
import glob, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.pipeline import SafetyPipeline
from safety.replay import run_cached
from safety.events import FightParams


def labels():
    lab = {}
    for kind, fn in (("violent", "violent-action-classes.csv"), ("non-violent", "nonviolent-action-classes.csv")):
        for ln in open(f"data/airtlab/{fn}").read().splitlines()[1:]:
            f, acts = ln.split(";")
            lab[(kind, int(f.replace(".mp4", "")))] = acts.strip().split(",")
    return lab


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def collect(params: FightParams | None = None, cams=("cam1", "cam2")):
    lab, rows = labels(), []
    for cam in cams:
        for p in sorted(glob.glob(f"data/cache/airt_{cam}/airtlab__*.npz")):
            kind, _, num = os.path.basename(p)[:-4].split("__")[1:]
            num = int(num)
            r = run_cached(SafetyPipeline(None, fight=params), p)
            fights = [a for a in r["alerts"] if a.kind == "FIGHT"]
            rows.append(dict(cam=cam, kind=kind, num=num, acts=lab[(kind, num)], peak=float(r["score"].max()),
                             peak_raw=float(r["raw"].max()), alarm=bool(fights), t_alarm=fights[0].t if fights else None,
                             max_people=r["max_people"], dur=float(r["t"][-1]) if len(r["t"]) else 0.0))
    return rows


def report(rows, title="baseline"):
    v = [r for r in rows if r["kind"] == "violent"]; n = [r for r in rows if r["kind"] == "non-violent"]
    print(f"\n=== {title}: {len(v)} violent, {len(n)} non-violent clips ===")
    print(f"alarm recall (violent clips flagged)      : {np.mean([r['alarm'] for r in v]):.0%}  ({sum(r['alarm'] for r in v)}/{len(v)})")
    print(f"false-alarm rate (non-violent flagged)    : {np.mean([r['alarm'] for r in n]):.0%}  ({sum(r['alarm'] for r in n)}/{len(n)})")
    print(f"peak-score AUC (violent vs non-violent)   : {auc([r['peak'] for r in v], [r['peak'] for r in n]):.3f}")
    print(f"median peak score  violent={np.median([r['peak'] for r in v]):.2f}  non-violent={np.median([r['peak'] for r in n]):.2f}")
    for cam in sorted({r['cam'] for r in rows}):
        vv = [r for r in v if r['cam'] == cam]; nn = [r for r in n if r['cam'] == cam]
        print(f"  {cam}: recall {np.mean([r['alarm'] for r in vv]):.0%}, false alarms {np.mean([r['alarm'] for r in nn]):.0%}, "
              f"AUC {auc([r['peak'] for r in vv], [r['peak'] for r in nn]):.3f}")
    print("per-action recall / mean peak (violent) :")
    by = collections.defaultdict(list)
    for r in v:
        for a in r["acts"]: by[a].append(r)
    for a, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"   {a:12s} n={len(rs):3d} alarm={np.mean([x['alarm'] for x in rs]):4.0%} peak={np.mean([x['peak'] for x in rs]):.2f}")
    print("per-action false alarms / mean peak (non-violent):")
    by = collections.defaultdict(list)
    for r in n:
        for a in r["acts"]: by[a].append(r)
    for a, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"   {a:12s} n={len(rs):3d} alarm={np.mean([x['alarm'] for x in rs]):4.0%} peak={np.mean([x['peak'] for x in rs]):.2f}")


if __name__ == "__main__":
    rows = collect()
    report(rows)
