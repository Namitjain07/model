"""Evaluate the fall detector on cached GMDCSA-24 detections using its per-clip time annotations.

Fall clips: did FALL/DOWN fire, and how late vs. the annotated start of the fall?
ADL clips (sitting, walking, sleeping ...): any FALL/DOWN is a false alarm (sleeping = deliberate lying down).
Clips are only 4-12 s, so the 10 s UNRESPONSIVE timer cannot be exercised here.
"""
import csv, glob, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.pipeline import SafetyPipeline
from safety.replay import run_cached
from safety.events import FallParams

SEG = re.compile(r"([A-Za-z][A-Za-z ]*?)\s*(?:\([^)]*\))?\s*\[\s*([\d.]+)\s*to\s*([\d.]+)\s*\]")


def annotations():
    ann = {}
    for s in (1, 2, 3, 4):
        for kind in ("Fall", "ADL"):
            with open(f"data/gmdcsa24/Subject {s}/{kind}.csv", newline="", encoding="utf-8-sig") as f:
                for row in list(csv.reader(f))[1:]:
                    if not row or not row[0].strip().endswith(".mp4"):
                        continue
                    segs = [(m.group(1).strip().lower(), float(m.group(2)), float(m.group(3))) for m in SEG.finditer(row[-1])]
                    ann[(s, kind, row[0].strip().replace(".mp4", ""))] = dict(segs=segs, desc=", ".join(row[4:-1]).strip())
    return ann


def collect(params: FallParams | None = None):
    ann, rows = annotations(), []
    for p in sorted(glob.glob("data/cache/gmdcsa/gmdcsa24__*.npz")):
        _, subj, kind, name = os.path.basename(p)[:-4].split("__")
        s = int(subj.split()[1]); a = ann.get((s, kind, name), dict(segs=[], desc=""))
        r = run_cached(SafetyPipeline(None, fall=params), p)
        al = {k: [x.t for x in r["alerts"] if x.kind == k] for k in ("FALL", "DOWN", "UNRESPONSIVE", "RECOVERED")}
        fall_start = min((st for lab, st, en in a["segs"] if lab.startswith("fall")), default=None)
        rows.append(dict(subj=s, kind=kind, name=name, segs=a["segs"], desc=a["desc"], fall_start=fall_start,
                         dur=float(r["t"][-1]) if len(r["t"]) else 0.0, max_people=r["max_people"], **al))
    return rows


def report(rows, title="baseline"):
    F = [r for r in rows if r["kind"] == "Fall"]; A = [r for r in rows if r["kind"] == "ADL"]
    print(f"\n=== {title}: {len(F)} fall clips, {len(A)} ADL clips ===")
    fa = lambda r: bool(r["FALL"]); da = lambda r: bool(r["FALL"] or r["DOWN"])
    print(f"fall clips  - FALL alarm: {np.mean([fa(r) for r in F]):.0%} ({sum(map(fa, F))}/{len(F)})   FALL-or-DOWN: {np.mean([da(r) for r in F]):.0%} ({sum(map(da, F))}/{len(F)})")
    print(f"ADL clips   - FALL false alarm: {np.mean([fa(r) for r in A]):.0%} ({sum(map(fa, A))}/{len(A)})   FALL-or-DOWN: {np.mean([da(r) for r in A]):.0%} ({sum(map(da, A))}/{len(A)})")
    lat = [min(r["FALL"] + r["DOWN"]) - r["fall_start"] for r in F if da(r) and r["fall_start"] is not None]
    if lat:
        print(f"alarm latency vs annotated fall start (s): median {np.median(lat):.1f}, p10 {np.percentile(lat,10):.1f}, p90 {np.percentile(lat,90):.1f}  (n={len(lat)})")
    print(f"clips with no person tracked: falls {sum(r['max_people']==0 for r in F)}, ADL {sum(r['max_people']==0 for r in A)}")
    for s in (1, 2, 3, 4):
        f = [r for r in F if r["subj"] == s]; a = [r for r in A if r["subj"] == s]
        if f and a:
            print(f"  subject {s}: fall recall(F|D) {np.mean([da(r) for r in f]):.0%} (n={len(f)}), ADL false (F|D) {np.mean([da(r) for r in a]):.0%} (n={len(a)})")
    print("ADL clips that raised FALL/DOWN (what activity was it?):")
    for r in A:
        if da(r):
            print(f"   S{r['subj']} {r['name']} {'FALL' if r['FALL'] else 'DOWN'}@{min(r['FALL']+r['DOWN']):.1f}s  segs={[(l,st) for l,st,_ in r['segs']]}")
    print("fall clips MISSED (no FALL/DOWN):")
    for r in F:
        if not da(r):
            print(f"   S{r['subj']} {r['name']} people={r['max_people']} {r['desc'][:90]}")


if __name__ == "__main__":
    report(collect())
