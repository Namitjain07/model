"""Pose (14) + optical-flow (6) window features for AIRTLab, Surveillance-Fight and RWF-2000 -> data/cache/all_windows_v3.npz

One row = one tracked pair at one time (1.5 s window).  Fields: X, y, ds, split, grp, cid, t.
  python scripts/build_windows_all.py [airt surv rwf]      (default: all; uses 4 processes)
"""
import glob, os, sys, re
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.events import FightDetector
from safety.flowfeat import FLOW_FEATURES, clip_flow, window_flow_features
from safety.pairfeat import FEATURES, pair_features
from safety.features import body_scale
from safety.replay import load_dets
from safety.tracker import Tracker

OUT = "data/cache/all_windows_v3.npz"


def sources():
    src, cur = {}, None
    for ln in open("data/survfight/videos.txt").read().splitlines():
        ln = ln.strip()
        if ln.startswith("http"): cur = ln
        elif re.match(r"(no)?fi\d+:", ln): src[ln.split(":")[0]] = cur
    return src


def describe(path, ds, src):
    """-> (video path, label, group, split, clip id)"""
    b = os.path.basename(path)[:-4]
    if ds == "airt":
        _, kind, cam, num = b.split("__")
        return f"data/airtlab/{kind}/{cam}/{num}.mp4", int(kind == "violent"), f"airt:{kind}:{num}", "all", f"airt:{kind}:{cam}:{num}"
    if ds == "surv":
        _, kind, name = b.split("__")
        return f"data/survfight/{kind}/{name}.mp4", int(kind == "fight"), "surv:" + src.get(name, name), "all", f"surv:{kind}:{name}"
    _, split, cls, name = b.split("__")
    return f"data/rwf2000/{split}/{cls}/{name}.avi", int(cls == "Fight"), "rwf:" + name.rsplit("_", 1)[0], split, f"rwf:{split}:{cls}:{name}"


def work(args):
    path, ds, src = args
    video, label, grp, split, cid = describe(path, ds, src)
    fd = FightDetector(); tr = Tracker()
    frames, meta = load_dets(path)
    if not frames: return []
    times = np.array([t for t, _ in frames], np.float32)
    flow, dt = clip_flow(video, len(frames))
    W, H = meta["size"]; rows = []
    for t, dets in frames:
        tracks = [x for x in tr.update(dets, t) if max(x.box[2] - x.box[0], x.box[3] - x.box[1]) >= fd.p.min_size_px]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                f = pair_features(fd, a, b, t)
                if f is None or f[0] >= 6.0: continue
                sc = 0.5 * (body_scale(a) + body_scale(b))
                ff = window_flow_features(flow, dt, times, t, fd.p.window_s, [a.box, b.box], (W, H), sc)
                rows.append((np.concatenate([f, ff]), label, ds, split, grp, cid, t))
    return rows


if __name__ == "__main__":
    want = sys.argv[1:] or ["airt", "surv", "rwf"]; src = sources(); jobs = []
    pats = {"airt": "data/cache/airt_cam?_v2/airtlab__*.npz", "surv": "data/cache/surv_*_v2/survfight__*.npz", "rwf": "data/cache/rwf_v2/rwf2000__*.npz"}
    for ds in want:
        fs = sorted(glob.glob(pats[ds])); print(ds, len(fs), "clips", flush=True); jobs += [(f, ds, src) for f in fs]
    rows = []
    with Pool(4) as p:
        for k, r in enumerate(p.imap_unordered(work, jobs, chunksize=4)):
            rows += r
            if (k + 1) % 250 == 0: print(f"{k+1}/{len(jobs)} clips, {len(rows)} windows", flush=True)
    X = np.array([r[0] for r in rows], np.float32)
    np.savez_compressed(OUT, X=X, y=np.array([r[1] for r in rows]), ds=np.array([r[2] for r in rows]), split=np.array([r[3] for r in rows]),
                        grp=np.array([r[4] for r in rows]), cid=np.array([r[5] for r in rows]), t=np.array([r[6] for r in rows], np.float32),
                        features=np.array(FEATURES + FLOW_FEATURES))
    print("windows", len(X), "features", X.shape[1], "| clips with >=1 window:", len({r[5] for r in rows}), "/", len(jobs))
