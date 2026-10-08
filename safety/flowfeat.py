"""Cheap optical-flow motion features (deployable: Farneback on a 128x72 greyscale frame, a few ms on CPU).

Pose fails in clinches / on the ground / in wide shots, but motion energy around the people does not.
Flow is computed between consecutive processed frames, stored as (n, H, W, 2) float16 in units of
*pixels of the 128-wide analysis frame*, plus per-frame dt, so features can be normalised by body size.
"""
from __future__ import annotations

import cv2
import numpy as np

AW, AH = 128, 72          # analysis frame
GW, GH = 64, 36           # stored flow grid
FLOW_FEATURES = ["flow_mean", "flow_p90", "flow_tstd", "flow_frac", "flow_ent", "flow_global"]


def small_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(bgr, (AW, AH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def flow_between(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    f = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    return cv2.resize(f, (GW, GH), interpolation=cv2.INTER_AREA).astype(np.float16)


def clip_flow(path: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Flow ending at each of the first `n` frames that scripts/cache_dets.py kept (same sampling rule:
    every = max(1, round(src_fps / 10))). Returns (flow[n,GH,GW,2] float16, dt[n] seconds)."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(src_fps / 10.0))
    flows = np.zeros((n, GH, GW, 2), np.float16)
    dts = np.full(n, 0.1, np.float32)
    prev, k, i = None, 0, 0
    while k < n:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            g = small_gray(fr)
            if prev is not None:
                flows[k] = flow_between(prev, g)
                dts[k] = every / src_fps
            prev, k = g, k + 1
        i += 1
    cap.release()
    return flows, dts


def window_flow_features(flow: np.ndarray, dt: np.ndarray, times: np.ndarray, t_now: float, window_s: float,
                         boxes: list, img_wh: tuple, scale_px: float) -> np.ndarray:
    """Features of the flow inside the union of `boxes` (pixel x1,y1,x2,y2 in the original frame) over the window.

    Local motion = flow minus the median flow *outside* the boxes (removes camera shake/pan). Units: torso lengths/s.
    """
    sel = np.flatnonzero((times <= t_now + 1e-6) & (times > t_now - window_s))
    sel = sel[sel > 0] if len(sel) > 1 else sel
    if len(sel) < 2:
        return np.zeros(len(FLOW_FEATURES), np.float32)
    W, H = img_wh
    mask = np.zeros((GH, GW), bool)
    for x1, y1, x2, y2 in boxes:
        a, b = int(max(0, x1) / W * GW), int(max(0, y1) / H * GH)
        c, d = int(np.ceil(min(W, x2) / W * GW)), int(np.ceil(min(H, y2) / H * GH))
        mask[b:max(d, b + 1), a:max(c, a + 1)] = True
    scale_a = max(scale_px * AW / W, 1.0)                 # torso length in analysis-frame pixels
    means, p90s, fracs, vx_all, vy_all, glob = [], [], [], [], [], []
    for j in sel:
        f = flow[j].astype(np.float32) / max(dt[j], 1e-3) / scale_a       # torso lengths / s
        out = ~mask
        cam = np.median(f[out], axis=0) if out.sum() > 20 else np.zeros(2, np.float32)
        loc = f - cam
        mag = np.linalg.norm(loc, axis=2)
        m = mag[mask]
        means.append(float(m.mean())); p90s.append(float(np.percentile(m, 90))); fracs.append(float((m > 1.0).mean()))
        vx_all.append(loc[..., 0][mask]); vy_all.append(loc[..., 1][mask]); glob.append(float(np.linalg.norm(cam)))
    vx, vy = np.concatenate(vx_all), np.concatenate(vy_all)
    ang = np.arctan2(vy, vx); w = np.hypot(vx, vy)
    hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi), weights=w)
    pr = hist / max(hist.sum(), 1e-6)
    ent = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum() / np.log(8))      # 0 = one direction, 1 = chaotic
    return np.array([np.mean(means), np.mean(p90s), np.std(means), np.mean(fracs), ent, np.mean(glob)], np.float32)


class FlowBuffer:
    """Live flow source for the pipeline: push each processed frame, read recent flow for window features."""

    def __init__(self, keep: int = 40):
        self.keep, self._g, self._t = keep, None, None
        self.flows, self.dts, self.times = [], [], []
        self.wh = None

    def push(self, bgr: np.ndarray, t: float):
        self.wh = (bgr.shape[1], bgr.shape[0])
        g = small_gray(bgr)
        f = flow_between(self._g, g) if self._g is not None else np.zeros((GH, GW, 2), np.float16)
        self.flows.append(f); self.dts.append(max(t - self._t, 1e-3) if self._t is not None else 0.1); self.times.append(t)
        self._g, self._t = g, t
        if len(self.flows) > self.keep:
            self.flows.pop(0); self.dts.pop(0); self.times.pop(0)

    def arrays(self):
        return np.stack(self.flows), np.asarray(self.dts, np.float32), np.asarray(self.times, np.float32)


class PrecomputedFlow:
    """Same interface as FlowBuffer for replaying cached clips (flow from clip_flow)."""

    def __init__(self, flow, dt, times, wh):
        self.flow, self.dt, self.times, self.wh = flow, dt, times, wh

    def arrays(self):
        return self.flow, self.dt, self.times
