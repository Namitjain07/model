"""Scale-normalised kinematic features computed from tracked COCO-17 skeletons.

Everything is expressed in *torso lengths* (shoulder-mid -> hip-mid distance) so
thresholds do not depend on how far the person is from the camera.
"""
from __future__ import annotations

import math

import numpy as np

from .skeleton import CONF_MIN, L_ANK, L_HIP, L_KNE, L_SHO, R_ANK, R_HIP, R_KNE, R_SHO, mid

MIN_SCALE_PX = 8.0


def torso_len(kp: np.ndarray) -> float | None:
    s, h = mid(kp, L_SHO, R_SHO), mid(kp, L_HIP, R_HIP)
    if s is None or h is None:
        return None
    return float(np.linalg.norm(s - h))


def torso_angle(kp: np.ndarray) -> float | None:
    """Degrees between the hip->shoulder axis and image-up: 0 upright, ~90 lying, >120 inverted."""
    s, h = mid(kp, L_SHO, R_SHO), mid(kp, L_HIP, R_HIP)
    if s is None or h is None:
        return None
    v = s - h
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -v[1] / n))))


def body_axis_angle(kp: np.ndarray) -> float | None:
    """Degrees from vertical of the ankle(or knee)-mid -> shoulder-mid axis: the *whole-body* tilt.

    ~0 standing/sitting upright; stays small when only the upper body folds (bending over,
    legs vertical); ~90 when the whole body is horizontal. None if shoulders or legs unseen.
    """
    s = mid(kp, L_SHO, R_SHO)
    lo = mid(kp, L_ANK, R_ANK)
    if lo is None:
        lo = mid(kp, L_KNE, R_KNE)
    if s is None or lo is None:
        return None
    v = s - lo
    n = float(np.linalg.norm(v))
    if n < 1e-3:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -v[1] / n))))


def center(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Body centre: mid-hip when visible, else box centre."""
    h = mid(kp, L_HIP, R_HIP)
    return h if h is not None else (box[:2] + box[2:]) / 2.0


def box_aspect(box: np.ndarray) -> float:
    w, h = box[2] - box[0], box[3] - box[1]
    return float(w / max(h, 1.0))


def body_scale(track) -> float:
    """Robust body size in px: median torso length over recent history, else 0.33 * box height."""
    lens = [torso_len(h[1]) for h in list(track.hist)[-60:]]
    lens = [x for x in lens if x]
    if lens:
        return max(float(np.median(lens)), MIN_SCALE_PX)
    b = track.box
    return max(0.33 * float(max(b[3] - b[1], b[2] - b[0])), MIN_SCALE_PX)


def spread(track, t_now: float, seconds: float, scale: float, min_frames: int = 4) -> float | None:
    """Stillness measure: median per-joint positional std over the window, in torso lengths.

    ~0.00-0.05 for a motionless person (detector jitter only); >0.15 for someone moving.
    Median over joints so a couple of flickering/flipped keypoints do not hide stillness.
    """
    w = track.window(t_now, seconds)
    if len(w) < min_frames:
        return None
    K = np.stack([h[1] for h in w])                       # (n,17,3)
    vis = (K[:, :, 2] >= CONF_MIN).mean(axis=0) >= 0.6    # joints seen in most frames
    if vis.sum() < 4:
        return None
    stds = []
    for j in np.flatnonzero(vis):
        ok = K[:, j, 2] >= CONF_MIN
        p = K[ok, j, :2]
        stds.append(math.sqrt(float(p[:, 0].var() + p[:, 1].var())))
    return float(np.median(stds)) / scale


def rel_velocity(track, joints, scale: float):
    """Per-frame joint speed *relative to body centre* in torso-lengths/s.

    Returns (times, speeds) with speeds shape (n-1, len(joints)); NaN where a joint is unseen.
    Subtracting the body-centre motion keeps walking from looking like striking.
    """
    hist = list(track.hist)
    if len(hist) < 2:
        return np.empty(0), np.empty((0, len(joints)))
    ts, out = [], []
    for (t0, k0, b0), (t1, k1, b1) in zip(hist[:-1], hist[1:]):
        dt = t1 - t0
        if dt <= 1e-3:
            continue
        c0, c1 = center(k0, b0), center(k1, b1)
        row = []
        for j in joints:
            if k0[j, 2] < CONF_MIN or k1[j, 2] < CONF_MIN:
                row.append(np.nan)
                continue
            d = (k1[j, :2] - c1) - (k0[j, :2] - c0)
            row.append(float(np.linalg.norm(d)) / dt / scale)
        ts.append(t1)
        out.append(row)
    return np.array(ts), np.array(out, np.float32).reshape(len(ts), len(joints))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
