"""Window-level pairwise features for a *learned* fight classifier.

All distances/speeds are in torso lengths (camera-distance independent). One feature vector describes
the interaction between two tracked people over the last `window_s` seconds.
"""
from __future__ import annotations

import numpy as np

from .events import FightDetector
from .features import body_scale, box_iou, center, torso_angle
from .skeleton import CONF_MIN, L_SHO, L_WRI, R_SHO, R_WRI, mid

FEATURES = ["dist", "min_kp_dist", "strikes", "energy_max", "energy_min", "iou", "closing", "dist_std",
            "lean_rate", "vert_std", "body_speed", "hands_up", "feet_dy", "scale_ratio"]


def pair_features(det: FightDetector, a, b, t: float) -> np.ndarray | None:
    """Feature vector (len(FEATURES),) for tracks a,b at time t, or None if too little shared history."""
    p = det.p
    sa, sb = body_scale(a), body_scale(b)
    sm = (sa + sb) / 2
    da = {h[0]: h for h in a.window(t, p.window_s)}
    db = {h[0]: h for h in b.window(t, p.window_s)}
    ts = np.array(sorted(set(da) & set(db)))
    if len(ts) < 4 or ts[-1] - ts[0] < 0.3:
        return None
    ca = np.array([center(da[x][1], da[x][2]) for x in ts])
    cb = np.array([center(db[x][1], db[x][2]) for x in ts])
    dist = np.linalg.norm(ca - cb, axis=1) / sm
    span = ts[-1] - ts[0]
    closing = float((dist[-1] - dist[0]) / span)                       # <0: approaching

    mk, lean_a, lean_b, up = [], [], [], []
    for x in ts:
        ka, kb = da[x][1], db[x][1]
        va, vb = ka[:, 2] >= CONF_MIN, kb[:, 2] >= CONF_MIN
        if va.sum() and vb.sum():
            mk.append(np.linalg.norm(ka[va, None, :2] - kb[None, vb, :2], axis=2).min() / sm)
        for k, lst in ((ka, lean_a), (kb, lean_b)):
            ang = torso_angle(k)
            lst.append(np.nan if ang is None else ang)
        hu = 0.0
        for k in (ka, kb):
            s = mid(k, L_SHO, R_SHO)
            if s is not None and any(k[w, 2] >= CONF_MIN and k[w, 1] < s[1] for w in (L_WRI, R_WRI)):
                hu = 1.0
        up.append(hu)

    def rate(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            return 0.0
        tt, vv = ts[ok], v[ok]
        return float(np.mean(np.abs(np.diff(vv)) / np.maximum(np.diff(tt), 1e-3)))

    ea, eb = det._energy(a, t), det._energy(b, t)
    vert = max(ca[:, 1].std() / sa, cb[:, 1].std() / sb)
    body = float(np.median(np.concatenate([np.linalg.norm(np.diff(ca, axis=0), axis=1) / sa / np.diff(ts),
                                           np.linalg.norm(np.diff(cb, axis=0), axis=1) / sb / np.diff(ts)])))
    return np.array([
        dist[-1], min(mk) if mk else 9.0, det._strikes(a, b, t) + det._strikes(b, a, t), max(ea, eb), min(ea, eb),
        box_iou(a.box, b.box), closing, float(dist.std()), max(rate(lean_a), rate(lean_b)) / 100.0, float(vert),
        body, float(np.mean(up)),
        # depth cues: people who really interact stand at the same depth (similar apparent size, feet at
        # similar image height). 2D overlap of a foreground bystander with someone far behind them is not contact.
        abs(float(a.box[3] - b.box[3])) / sm, max(sa, sb) / min(sa, sb),
    ], np.float32)
