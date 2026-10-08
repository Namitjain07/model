"""OpenCV skeleton drawing used for visual verification of model output."""
from __future__ import annotations

import cv2
import numpy as np

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # shoulders + arms
    (11, 23), (12, 24), (23, 24),                              # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                    # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # feet
    (0, 11), (0, 12),                                          # head -> shoulders (visual aid only)
]


def draw_hands(img, lm: np.ndarray, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for hand in lm:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
        for a, b in HAND_EDGES:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def draw_poses(img, lm: np.ndarray, color=(255, 200, 0), vis_thresh=0.3):
    h, w = img.shape[:2]
    for pose in lm:
        pts = [(int(x * w), int(y * h)) for x, y, _, _ in pose]
        vis = pose[:, 3]
        for a, b in POSE_EDGES:
            if vis[a] > vis_thresh and vis[b] > vis_thresh:
                cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if vis[i] > vis_thresh:
                cv2.circle(img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def banner(img, text: str, color=(0, 0, 255), y=28):
    """Readable overlay text (dark outline + coloured fill)."""
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img
