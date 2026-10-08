"""Common COCO-17 skeleton so detectors are independent of the pose backend.

Keypoints are (17, 3) float arrays: x, y in *pixels*, confidence in [0, 1].
"""
from __future__ import annotations

import numpy as np

(NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
 L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK) = range(17)

HEAD = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
TORSO = [L_SHO, R_SHO, L_HIP, R_HIP]
STRIKERS = [L_WRI, R_WRI, L_ANK, R_ANK]          # limbs that deliver a blow
TARGETS = HEAD + TORSO                           # where a blow lands
LIMBS = [L_ELB, R_ELB, L_WRI, R_WRI, L_KNE, R_KNE, L_ANK, R_ANK]

COCO_EDGES = [
    (L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]

# MediaPipe 33-landmark index for each COCO-17 joint.
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CONF_MIN = 0.3


def mp33_to_coco17(lm: np.ndarray, width: int, height: int) -> np.ndarray:
    """(33,4) normalized MediaPipe landmarks -> (17,3) pixel COCO keypoints."""
    sel = lm[MP_TO_COCO]
    out = np.empty((17, 3), np.float32)
    out[:, 0] = sel[:, 0] * width
    out[:, 1] = sel[:, 1] * height
    out[:, 2] = sel[:, 3]
    return out


def visible(kp: np.ndarray, idx) -> np.ndarray:
    return kp[idx, 2] >= CONF_MIN


def mid(kp: np.ndarray, a: int, b: int):
    """Midpoint of two joints, or None if either is not visible."""
    if kp[a, 2] < CONF_MIN or kp[b, 2] < CONF_MIN:
        return None
    return (kp[a, :2] + kp[b, :2]) / 2.0


def bbox_of(kp: np.ndarray):
    """Tight box around visible keypoints (x1,y1,x2,y2), or None."""
    v = kp[:, 2] >= CONF_MIN
    if v.sum() < 3:
        return None
    p = kp[v, :2]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], np.float32)
