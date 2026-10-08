"""Top-down pose: person boxes (from any detector, e.g. YOLOX) -> MediaPipe pose landmarks on each person crop -> COCO-17.

Why: MediaPipe's built-in multi-person pose uses a detector tuned for one large, close person and misses most people in wide
surveillance shots. Given a box, its landmark network works well (and is tiny: ~3.4 M params, ~1.1 ms/person on a QCS6490 NPU
at w8a8 per Qualcomm AI Hub). The detector is swappable: YOLOX-M on a GPU box, Qualcomm AI Hub's Yolo-X on the board.
"""
from __future__ import annotations

import cv2
import numpy as np

from .landmarks import PoseTracker
from .skeleton import mp33_to_coco17
from .tracker import Det


class MediaPipeCropPose:
    def __init__(self, variant: str = "full", margin: float = 0.15, size: int = 256, min_conf: float = 0.3):
        self.pt = PoseTracker(num_poses=1, variant=variant, video=False, min_conf=min_conf)
        self.margin, self.size = margin, size

    def _crop(self, bgr: np.ndarray, box: np.ndarray):
        H, W = bgr.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        S = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * self.margin)
        S = max(S, 32.0)
        x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2)); s = int(round(S))
        pad = max(0, -x0, -y0, x0 + s - W, y0 + s - H)
        src = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114)) if pad else bgr
        crop = src[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
        interp = cv2.INTER_AREA if s > self.size else cv2.INTER_CUBIC
        return cv2.resize(crop, (self.size, self.size), interpolation=interp), (x0, y0, s)

    def __call__(self, bgr: np.ndarray, boxes: list[np.ndarray], scores: list[float] | None = None) -> list[Det]:
        out = []
        for k, box in enumerate(boxes):
            crop, (x0, y0, s) = self._crop(bgr, np.asarray(box, np.float32))
            r = self.pt.process(crop)
            if len(r.lm) == 0:
                continue
            kp = mp33_to_coco17(r.lm[0], s, s)                       # pixels inside the (square, s x s) crop
            kp[:, 0] += x0; kp[:, 1] += y0
            out.append(Det(kp.astype(np.float32), np.asarray(box, np.float32), float(scores[k]) if scores else 1.0))
        return out

    def close(self):
        self.pt.close()


# ---------------------------------------------------------------------------------------------------------------------
# RTMPose-m (Apache-2.0, 13.6 M params, 256x192 input): top-down pose that FOLLOWS the box it is given (no re-detection),
# so overlapping people in a fight keep their own keypoints.  ONNX from OpenMMLab (see scripts/download_models.sh).
# ---------------------------------------------------------------------------------------------------------------------
import glob
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)


class RTMPoseTopDown:
    W, H, PAD, SPLIT = 192, 256, 1.25, 2.0

    def __init__(self, path: str | None = None, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort
        path = path or glob.glob(str(MODELS / "rtmpose" / "m" / "**" / "end2end.onnx"), recursive=True)[0]
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def _prep(self, bgr, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = (box[2] - box[0]) * self.PAD, (box[3] - box[1]) * self.PAD
        ar = self.W / self.H
        if w > h * ar: h = w / ar
        else: w = h * ar
        src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
        dst = np.float32([[0, 0], [self.W, 0], [0, self.H]])
        M = cv2.getAffineTransform(src, dst)
        img = cv2.warpAffine(bgr, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        x = (img[:, :, ::-1].astype(np.float32) - _MEAN) / _STD
        return x.transpose(2, 0, 1), (cx, cy, w, h)

    def __call__(self, bgr: np.ndarray, boxes: list[np.ndarray], scores: list[float] | None = None) -> list[Det]:
        if not len(boxes):
            return []
        batch, geo = zip(*[self._prep(bgr, np.asarray(b, np.float32)) for b in boxes])
        sx, sy = self.sess.run(None, {self.inp: np.ascontiguousarray(np.stack(batch))})
        out = []
        for i, (cx, cy, w, h) in enumerate(geo):
            ix, iy = sx[i].argmax(-1), sy[i].argmax(-1)                      # (K,)
            conf = np.minimum(sx[i].max(-1), sy[i].max(-1))
            kx = ix / self.SPLIT / self.W * w + cx - w / 2
            ky = iy / self.SPLIT / self.H * h + cy - h / 2
            kp = np.stack([kx, ky, np.clip(conf, 0, 1)], 1).astype(np.float32)
            out.append(Det(kp, np.asarray(boxes[i], np.float32), float(scores[i]) if scores else 1.0))
        return out


# ---------------------------------------------------------------------------------------------------------------------
# YOLOX person detector (Apache-2.0).  Default model: YOLOX-M ONNX (25 M params).  Raw (undecoded) head -> decode + NMS here.
# ---------------------------------------------------------------------------------------------------------------------
class YoloxPerson:
    def __init__(self, path: str | None = None, conf: float = 0.25, nms: float = 0.45, size: int = 640,
                 providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path or MODELS / "yolox_m.onnx"), so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.nms, self.size = conf, nms, size
        grids, strides = [], []
        for st in (8, 16, 32):
            n = size // st
            gx, gy = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([gx, gy], -1).reshape(-1, 2)); strides.append(np.full((n * n, 1), st))
        self.grid, self.stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas.astype(np.float32).transpose(2, 0, 1)[None])          # BGR, 0-255, no normalisation
        o = self.sess.run(None, {self.inp: x})[0][0]                                            # (8400, 85)
        xy = (o[:, :2] + self.grid) * self.stride
        wh = np.exp(o[:, 2:4]) * self.stride
        score = o[:, 4] * o[:, 5]                                                               # objectness * P(person)
        keep = score >= self.conf
        if not keep.any():
            return [], []
        xy, wh, score = xy[keep], wh[keep], score[keep]
        boxes = np.concatenate([xy - wh / 2, wh], 1)                                            # x,y,w,h in letterbox px
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), self.conf, self.nms)
        res_b, res_s = [], []
        for i in np.array(idx).reshape(-1):
            x1, y1, bw, bh = boxes[i] / r
            res_b.append(np.array([max(0, x1), max(0, y1), min(w, x1 + bw), min(h, y1 + bh)], np.float32)); res_s.append(float(score[i]))
        return res_b, res_s
