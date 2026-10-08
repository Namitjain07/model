"""Small multi-person tracker (IoU + centre distance, Hungarian assignment).

Backend-agnostic: works for YOLO-pose or MediaPipe detections alike. Tracks are
kept alive through short occlusions so a person lying down behind furniture is
not instantly forgotten.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .skeleton import bbox_of


@dataclass
class Det:
    kpts: np.ndarray              # (17,3) px
    box: np.ndarray               # (4,) x1,y1,x2,y2 px
    score: float = 1.0


@dataclass
class Track:
    tid: int
    hist: deque = field(default_factory=lambda: deque(maxlen=300))  # (t, kpts, box)
    last_t: float = 0.0
    born_t: float = 0.0
    hits: int = 0
    missed: int = 0
    # per-detector scratch space (state machines hang their state here)
    state: dict = field(default_factory=dict)

    @property
    def kpts(self) -> np.ndarray:
        return self.hist[-1][1]

    @property
    def box(self) -> np.ndarray:
        return self.hist[-1][2]

    def window(self, t_now: float, seconds: float):
        return [h for h in self.hist if t_now - h[0] <= seconds]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    def __init__(self, max_lost_s: float = 2.0, min_hits: int = 3, gate: float = 1.0):
        self.max_lost_s, self.min_hits, self.gate = max_lost_s, min_hits, gate
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, dets: list[Det], t: float) -> list[Track]:
        """Associate detections with tracks; return tracks seen this frame (confirmed only)."""
        T, D = len(self.tracks), len(dets)
        matched_t, matched_d = set(), set()
        if T and D:
            cost = np.full((T, D), 1e3, np.float32)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    iou = _iou(tr.box, d.box)
                    h = max(tr.box[3] - tr.box[1], d.box[3] - d.box[1], 1.0)
                    ca = (tr.box[:2] + tr.box[2:]) / 2
                    cb = (d.box[:2] + d.box[2:]) / 2
                    dist = float(np.linalg.norm(ca - cb)) / h     # in body heights
                    # centre-distance fallback keeps a falling person (box reshapes fast) on one ID
                    if iou > 0.05 or dist < 0.6:
                        cost[i, j] = (1.0 - iou) + 0.5 * dist
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < self.gate + 0.5:
                    self._attach(self.tracks[i], dets[j], t)
                    matched_t.add(i); matched_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.missed += 1
        for j, d in enumerate(dets):
            if j not in matched_d:
                tr = Track(self._next, born_t=t)
                self._next += 1
                self._attach(tr, d, t)
                self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_lost_s]
        return [tr for tr in self.tracks if tr.last_t == t and tr.hits >= self.min_hits]

    @staticmethod
    def _attach(tr: Track, d: Det, t: float):
        tr.hist.append((t, d.kpts, d.box))
        tr.last_t = t
        tr.hits += 1
        tr.missed = 0
