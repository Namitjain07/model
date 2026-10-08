"""Synthetic COCO-17 skeleton sequences for *logic* tests of the detectors.

NOT a substitute for real footage: it only proves the state machines respond as designed.
Units: pixels, y down. `L` is the torso length in px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .skeleton import L_ANK, L_EAR, L_ELB, L_EYE, L_HIP, L_KNE, L_SHO, L_WRI, NOSE, R_ANK, R_EAR, R_ELB, R_EYE, R_HIP, R_KNE, R_SHO, R_WRI
from .tracker import Det

# Standing pose as offsets from mid-hip, in torso lengths (x right, y down).
_STAND = {
    NOSE: (0, -1.45), L_EYE: (-.05, -1.5), R_EYE: (.05, -1.5), L_EAR: (-.1, -1.45), R_EAR: (.1, -1.45),
    L_SHO: (-.2, -1), R_SHO: (.2, -1), L_ELB: (-.25, -.5), R_ELB: (.25, -.5), L_WRI: (-.25, 0), R_WRI: (.25, 0),
    L_HIP: (-.15, 0), R_HIP: (.15, 0), L_KNE: (-.15, 1), R_KNE: (.15, 1), L_ANK: (-.15, 2), R_ANK: (.15, 2),
}
_LOWER = [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
_UPPER = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI]


def _rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


@dataclass
class Person:
    L: float = 100.0
    pos: Callable[[float], tuple] = lambda t: (400.0, 300.0)       # mid-hip px
    tilt: Callable[[float], float] = lambda t: 0.0                 # whole-body rotation about hips (deg)
    bend: Callable[[float], float] = lambda t: 0.0                 # extra upper-body rotation (deg)
    walking: Callable[[float], bool] = lambda t: False
    punch: Callable[[float], tuple | None] = lambda t: None        # (arm_joint, progress 0..1, target xy)
    flail: Callable[[float], float] = lambda t: 0.0                # extra random limb motion amplitude (L)
    swing: Callable[[float], float] = lambda t: 0.0                # smooth 2 Hz arm swing amplitude (L), e.g. dancing
    jitter: float = 0.02
    crop: object = False          # bool or callable(t): webcam-style close-up, hips/legs not visible
    seed: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def skeleton(self, t: float) -> np.ndarray:
        L, c = self.L, np.array(self.pos(t), float)
        pts = {j: np.array(o, float) * L for j, o in _STAND.items()}
        if self.walking(t):
            ph = 2 * math.pi * 1.0 * t
            for j, amp in ((L_WRI, .25), (R_WRI, -.25), (L_ANK, -.5), (R_ANK, .5), (L_KNE, -.25), (R_KNE, .25)):
                pts[j][0] += amp * math.sin(ph) * L
        fl = self.flail(t)
        if fl:
            for j in (L_WRI, R_WRI, L_ELB, R_ELB, L_ANK, R_ANK):
                pts[j] = pts[j] + self._rng.normal(0, fl * L, 2)
        sw = self.swing(t)
        if sw:
            s = math.sin(2 * math.pi * 2.0 * t)
            pts[L_WRI] = pts[L_WRI] + np.array([sw * s, -0.5 * sw * (1 + s)]) * L
            pts[R_WRI] = pts[R_WRI] + np.array([-sw * s, -0.5 * sw * (1 - s)]) * L
            pts[L_ELB] = pts[L_ELB] + np.array([0.5 * sw * s, -0.25 * sw * (1 + s)]) * L
            pts[R_ELB] = pts[R_ELB] + np.array([-0.5 * sw * s, -0.25 * sw * (1 - s)]) * L
        pk = self.punch(t)
        if pk is not None:
            j, prog, tgt = pk
            rest = pts[j] + c
            pts[j] = (rest + (np.array(tgt, float) - rest) * prog) - c
        b = self.bend(t)
        if b:
            for j in _UPPER:
                pts[j] = _rot(pts[j], b)
        th = self.tilt(t)
        cropped = self.crop(t) if callable(self.crop) else self.crop
        kp = np.zeros((17, 3), np.float32)
        for j, v in pts.items():
            p = c + (_rot(v, th) if th else v)
            kp[j, :2] = p + self._rng.normal(0, self.jitter * L, 2)
            kp[j, 2] = 0.05 if (cropped and j in _LOWER) else 0.9
        return kp


def det_of(kp: np.ndarray) -> Det:
    xy = kp[kp[:, 2] >= 0.3, :2]            # box around what a detector could actually see
    box = np.array([xy[:, 0].min() - 8, xy[:, 1].min() - 8, xy[:, 0].max() + 8, xy[:, 1].max() + 8], np.float32)
    return Det(kp, box, 0.9)


def run(pipe, people: list[Person], seconds: float, fps: float = 10.0):
    """Feed simulated skeletons through `pipe.step`; return the list of all alerts."""
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        pipe.step([det_of(p.skeleton(t)) for p in people], t)
    return pipe.log


# ---- scenario helpers ----------------------------------------------------------------------
def ramp(t0, t1, v0, v1):
    return lambda t: v0 if t <= t0 else v1 if t >= t1 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def fall_person(t_fall=3.0, dur=0.6, L=100.0, cx=400.0, floor=500.0, recover_at=None, wiggle=0.0, seed=0):
    """Stands, falls over `dur` s, lies. Hips drop from floor-2L to ~floor-0.1L. Optional recovery / wiggle."""
    cy_up, cy_dn = floor - 2 * L, floor - 0.1 * L
    ang = ramp(t_fall, t_fall + dur, 0.0, 90.0)
    cyf = ramp(t_fall, t_fall + dur, cy_up, cy_dn)
    if recover_at is not None:
        ang = (lambda a, r: lambda t: a(t) if t < r else max(0.0, 90.0 * (1 - (t - r) / 1.5)))(ang, recover_at)
        cyf = (lambda a, r: lambda t: a(t) if t < r else cy_dn + (cy_up - cy_dn) * min(1.0, (t - r) / 1.5))(cyf, recover_at)
    base_pos = lambda t: (cx + (wiggle * L * math.sin(2 * math.pi * 0.8 * t) if t > t_fall + dur else 0.0), cyf(t))
    return Person(L=L, pos=base_pos, tilt=ang, flail=(lambda t: wiggle * 0.5 if t > t_fall + dur else 0.0), seed=seed)


def fighters(L=100.0, gap=1.5, punch_every=0.7, seed=1):
    """Two people facing each other trading punches at head height."""
    ya = 300.0
    xa, xb = 300.0, 300.0 + gap * L
    def mk(x_self, x_other, arm, offset, sd):
        def punch(t):
            ph = (t - offset) % punch_every
            if ph < 0.2:
                return (arm, ph / 0.2, (x_other, ya - 1.4 * L))
            if ph < 0.4:
                return (arm, 1 - (ph - 0.2) / 0.2, (x_other, ya - 1.4 * L))
            return None
        return Person(L=L, pos=lambda t: (x_self, ya), punch=punch, bend=lambda t: 10 * math.sin(2 * math.pi * 1.5 * t),
                      flail=lambda t: 0.08, seed=sd)
    return [mk(xa, xb, R_WRI, 0.0, seed), mk(xb, xa, L_WRI, punch_every / 2, seed + 1)]


def handshake(L=100.0, gap=1.1, seed=3):
    ya, xa, xb = 300.0, 300.0, 300.0 + gap * L
    def mk(xs, xo, arm, sd):
        return Person(L=L, pos=lambda t: (xs, ya),
                      punch=lambda t: (arm, min(1.0, t / 2.0) * 0.6, ((xs + xo) / 2, ya - 0.1 * L)), seed=sd)
    return [mk(xa, xb, R_WRI, seed), mk(xb, xa, L_WRI, seed + 1)]


def companions(L=100.0, gap=1.6, speed_px_s=140.0, seed=5):
    return [Person(L=L, pos=(lambda off: lambda t: (100 + speed_px_s * t, 300.0 + off))(o), walking=lambda t: True, seed=seed + k)
            for k, o in enumerate((0.0, 0.0))] and [
        Person(L=L, pos=lambda t: (100 + speed_px_s * t, 300.0), walking=lambda t: True, seed=seed),
        Person(L=L, pos=lambda t: (100 + speed_px_s * t + gap * L, 300.0), walking=lambda t: True, seed=seed + 1)]


def dancers(L=100.0, gap=1.6, amp=0.6, seed=7):
    """Two people dancing: smooth 2 Hz arm swings toward each other (hard negative for fights)."""
    def mk(x, sd):
        return Person(L=L, pos=lambda t: (x + 30 * math.sin(2 * math.pi * 1.0 * t), 300.0), swing=lambda t: amp,
                      bend=lambda t: 12 * math.sin(2 * math.pi * 2 * t), walking=lambda t: True, seed=sd)
    return [mk(300.0, seed), mk(300.0 + gap * L, seed + 1)]


def jittery_pair(L=100.0, gap=1.6, seed=7):
    """Two standing people whose keypoints jump around frame-to-frame (pure detector noise)."""
    return [Person(L=L, pos=(lambda x: lambda t: (x, 300.0))(x), flail=lambda t: 0.25, seed=sd)
            for x, sd in ((300.0, seed), (300.0 + gap * L, seed + 1))]
