"""Extract pairwise window features from cached AIRTLab detections -> data/cache/fight_windows.npz.

Label = clip label (weak: not every window of a violent clip is violent). group = performance id, so the
two camera views of the same performance always land in the same CV fold (no leakage).
"""
import glob, os, sys
import numpy as np
TAG = os.environ.get("CACHE_TAG", "_v2")   # e.g. _v2 = relaxed-detector caches
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.events import FightDetector
from safety.pairfeat import FEATURES, pair_features
from safety.replay import load_dets
from safety.tracker import Tracker

fd = FightDetector()
X, y, grp, cam, tt, cid = [], [], [], [], [], []
files = sorted(glob.glob(f"data/cache/airt_cam?{TAG}/airtlab__*.npz"))
for k, p in enumerate(files):
    _, kind, c, num = os.path.basename(p)[:-4].split("__")
    frames, _ = load_dets(p)
    tr = Tracker(); label = int(kind == "violent")
    for t, dets in frames:
        tracks = [x for x in tr.update(dets, t) if max(x.box[2] - x.box[0], x.box[3] - x.box[1]) >= fd.p.min_size_px]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                f = pair_features(fd, tracks[i], tracks[j], t)
                if f is not None and f[0] < 6.0:                      # only people plausibly interacting
                    X.append(f); y.append(label); grp.append(f"{kind}:{num}"); cam.append(c); tt.append(t); cid.append(f"{kind}:{c}:{num}")
    if (k + 1) % 50 == 0:
        print(f"{k+1}/{len(files)} clips, {len(X)} windows", flush=True)
np.savez_compressed(f"data/cache/fight_windows{TAG}.npz", X=np.array(X), y=np.array(y), grp=np.array(grp), cam=np.array(cam),
                    t=np.array(tt), cid=np.array(cid), features=np.array(FEATURES))
print("windows:", len(X), "violent:", int(np.sum(y)), "clips with >=1 window:", len(set(cid)), "/", len(files))
