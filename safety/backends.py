"""Pose backends producing a common list[Det] (COCO-17 keypoints in pixels + person box).

YoloPoseOnnx   - Ultralytics YOLO11-pose exported to ONNX, run with onnxruntime (no torch needed;
                 the same file runs on the QCS6490 via onnxruntime's QNN/CPU providers).
MediaPipeBackend - MediaPipe Pose Landmarker (multi-person) mapped to COCO-17.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .skeleton import bbox_of, mp33_to_coco17
from .tracker import Det

MODELS = Path(__file__).resolve().parent.parent / "models"


class YoloPoseOnnx:
    def __init__(self, path: str | Path | None = None, conf: float = 0.15, iou: float = 0.7,
                 imgsz: int = 640, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path or MODELS / "yolo11n-pose.onnx"), so,
                                         providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape                  # [1,3,H,W] (static for this export)
        self.size = (int(shp[2]), int(shp[3])) if isinstance(shp[2], int) else (imgsz, imgsz)
        self.conf, self.iou = conf, iou

    def _letterbox(self, bgr):
        H, W = self.size
        h, w = bgr.shape[:2]
        r = min(H / h, W / w)
        nh, nw = round(h * r), round(w * r)
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (H - nh) // 2, (W - nw) // 2
        canvas = np.full((H, W, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        x = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(x), r, left, top

    def __call__(self, bgr: np.ndarray) -> list[Det]:
        x, r, left, top = self._letterbox(bgr)
        out = self.sess.run(None, {self.inp: x})[0][0].T          # (N, 56): cx,cy,w,h,conf,17*(x,y,c)
        out = out[out[:, 4] >= self.conf]
        if not len(out):
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), out[:, 4].tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(keep).reshape(-1):
            kp = out[i, 5:].reshape(17, 3).copy()
            kp[:, 0] = (kp[:, 0] - left) / r
            kp[:, 1] = (kp[:, 1] - top) / r
            box = np.array([(cx[i] - w[i] / 2 - left) / r, (cy[i] - h[i] / 2 - top) / r,
                            (cx[i] + w[i] / 2 - left) / r, (cy[i] + h[i] / 2 - top) / r], np.float32)
            dets.append(Det(kp.astype(np.float32), box, float(out[i, 4])))
        return dets


class MediaPipeBackend:
    def __init__(self, num_poses: int = 4, variant: str = "lite", min_conf: float = 0.5):
        from .landmarks import PoseTracker
        self.pt = PoseTracker(num_poses=num_poses, variant=variant, video=True, min_conf=min_conf)
        self._t = 0

    def __call__(self, bgr: np.ndarray, ts_ms: int | None = None) -> list[Det]:
        h, w = bgr.shape[:2]
        self._t = ts_ms if ts_ms is not None else self._t + 33
        poses = self.pt.process(bgr, self._t)
        dets = []
        for lm in poses.lm:
            kp = mp33_to_coco17(lm, w, h)
            box = bbox_of(kp)
            if box is None:
                continue
            pad = 0.08 * max(box[2] - box[0], box[3] - box[1])
            box = box + np.array([-pad, -pad, pad, pad], np.float32)
            dets.append(Det(kp, box, float(np.mean(kp[:, 2]))))
        return dets
