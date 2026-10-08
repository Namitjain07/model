"""Small threshold sweep for the fall detector on cached GMDCSA-24 detections, with a subject-wise split.

Select on subjects A, report on subjects B (and vice-versa) so we do not just memorise the 4 people.
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_falls import collect
from safety.events import FallParams

def stats(rows, subj):
    F = [r for r in rows if r["kind"] == "Fall" and r["subj"] in subj]; A = [r for r in rows if r["kind"] == "ADL" and r["subj"] in subj]
    fall_any = lambda r: bool(r["FALL"] or r["DOWN"])
    return dict(fall_rec=np.mean([bool(r["FALL"]) for r in F]), any_rec=np.mean([fall_any(r) for r in F]),
                fall_fa=np.mean([bool(r["FALL"]) for r in A]), any_fa=np.mean([fall_any(r) for r in A]))

if __name__ == "__main__":
    grid = list(itertools.product([1.5, 3.0, 5.0], [0.0, 0.25, 0.5], [45.0, 55.0], [0.4, 0.6]))
    res = []
    for ff, md, ax, hold in grid:
        rows = collect(FallParams(fall_fast_s=ff, min_drop=md, lying_axis=ax, lying_hold_s=hold))
        res.append(((ff, md, ax, hold), stats(rows, {1, 2, 3, 4}), stats(rows, {1, 2}), stats(rows, {3, 4})))
        s = res[-1][1]
        print(f"fast={ff:3.1f} drop={md:.2f} axis={ax:.0f} hold={hold:.1f} | FALL rec {s['fall_rec']:.0%} fa {s['fall_fa']:.0%} | any rec {s['any_rec']:.0%} fa {s['any_fa']:.0%}", flush=True)
