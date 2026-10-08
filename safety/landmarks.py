"""Thin MediaPipe Tasks wrappers that return plain NumPy arrays.

Everything downstream (hand-signal, fall, aggression) consumes these arrays
only, so the classifiers never import MediaPipe and can be exported / ported
to the QCS6490 board independently of the landmark extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Hands:
    """lm: (N,21,3) normalized x,y,z. world: (N,21,3) metres. label: 'Left'/'Right' as seen in image."""
    lm: np.ndarray
    world: np.ndarray
    label: list[str]
    score: list[float]


@dataclass
class Poses:
    """lm: (N,33,4) normalized x,y,z,visibility. world: (N,33,3) metres, hip-centred."""
    lm: np.ndarray
    world: np.ndarray


def _mp_image(bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class _Base:
    def __init__(self, video: bool):
        self._video = video
        self._last_ts = -1

    def _ts(self, ts_ms: int | None) -> int:
        # MediaPipe VIDEO mode needs strictly increasing timestamps.
        ts = self._last_ts + 1 if ts_ms is None else max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def _run(self, det, bgr, ts_ms):
        img = _mp_image(bgr)
        return det.detect_for_video(img, self._ts(ts_ms)) if self._video else det.detect(img)

    def close(self):
        self._det.close()


class HandTracker(_Base):
    def __init__(self, num_hands: int = 2, video: bool = True, min_conf: float = 0.5):
        super().__init__(video)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._det = vision.HandLandmarker.create_from_options(opts)

    def process(self, bgr: np.ndarray, ts_ms: int | None = None) -> Hands:
        r = self._run(self._det, bgr, ts_ms)
        n = len(r.hand_landmarks)
        lm = np.array([[[p.x, p.y, p.z] for p in h] for h in r.hand_landmarks], np.float32).reshape(n, 21, 3)
        wd = np.array([[[p.x, p.y, p.z] for p in h] for h in r.hand_world_landmarks], np.float32).reshape(n, 21, 3)
        label = [h[0].category_name for h in r.handedness]
        score = [h[0].score for h in r.handedness]
        return Hands(lm, wd, label, score)


class PoseTracker(_Base):
    def __init__(self, num_poses: int = 1, variant: str = "lite", video: bool = True, min_conf: float = 0.5):
        super().__init__(video)
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / f"pose_landmarker_{variant}.task")),
            running_mode=mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._det = vision.PoseLandmarker.create_from_options(opts)

    def process(self, bgr: np.ndarray, ts_ms: int | None = None) -> Poses:
        r = self._run(self._det, bgr, ts_ms)
        n = len(r.pose_landmarks)
        lm = np.array([[[p.x, p.y, p.z, p.visibility] for p in a] for a in r.pose_landmarks], np.float32).reshape(n, 33, 4)
        wd = np.array([[[p.x, p.y, p.z] for p in a] for a in r.pose_world_landmarks], np.float32).reshape(n, 33, 3)
        return Poses(lm, wd)
