"""Clip dataset over the preprocessed JPEG stacks.  Augmentations are shared by all frames of a clip.

Shortcut countermeasures (the audit found resolution/fps/duration alone predict the label, RLVS AUC 0.95):
  * random resolution degradation (down-up scaling 0.35-1.0x, blur, noise) so sharpness/native size is not a cue
  * bucket-balanced sampling weights (gpu/common.bucket_weights)
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gpu.common import CACHE

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
T = 16


def safe(cid: str) -> str:
    return cid.replace(":", "__").replace("/", "_")


def has_frames(cid: str) -> bool:
    return (CACHE / "frames" / (safe(cid) + ".pkl")).exists()


def _rrc(imgs, size=224):
    h, w = imgs[0].shape[:2]
    for _ in range(8):
        area = random.uniform(0.55, 1.0) * h * w; ar = random.uniform(3 / 4, 4 / 3)
        cw, ch = int(round((area * ar) ** 0.5)), int(round((area / ar) ** 0.5))
        if cw <= w and ch <= h:
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            return [cv2.resize(i[y:y + ch, x:x + cw], (size, size), interpolation=cv2.INTER_LINEAR) for i in imgs]
    return imgs


def augment(imgs):
    imgs = _rrc(imgs)
    if random.random() < 0.5: imgs = [i[:, ::-1] for i in imgs]
    b, c, s = (random.uniform(-0.25, 0.25) for _ in range(3))                         # brightness / contrast / saturation
    out = []
    for i in imgs:
        f = i.astype(np.float32); f = f * (1 + c) + 255 * b * 0.5
        g = f.mean(-1, keepdims=True); f = g + (f - g) * (1 + s)
        out.append(np.clip(f, 0, 255))
    imgs = out
    if random.random() < 0.5:                                                          # resolution degradation
        k = random.uniform(0.35, 1.0); d = max(32, int(224 * k))
        imgs = [cv2.resize(cv2.resize(i, (d, d), interpolation=cv2.INTER_AREA), (224, 224), interpolation=cv2.INTER_LINEAR) for i in imgs]
    if random.random() < 0.2:
        sg = random.uniform(0.5, 1.5); imgs = [cv2.GaussianBlur(i, (0, 0), sg) for i in imgs]
    if random.random() < 0.3:
        n = np.random.normal(0, random.uniform(2, 8), imgs[0].shape).astype(np.float32); imgs = [np.clip(i + n, 0, 255) for i in imgs]
    return imgs


class ClipDataset(Dataset):
    def __init__(self, df, train: bool, n_frames: int = 16):
        self.df, self.train, self.nf = df.reset_index(drop=True), train, n_frames

    def __len__(self):
        return len(self.df)

    def load(self, cid, phase):
        d = pickle.load(open(CACHE / "frames" / (safe(cid) + ".pkl"), "rb"))["frames"]
        idx = [phase * T + k for k in range(T)]
        if self.nf < T: idx = idx[::T // self.nf]
        return [cv2.imdecode(np.frombuffer(d[k], np.uint8), cv2.IMREAD_COLOR) for k in idx]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        imgs = self.load(r.clip_id, random.randint(0, 1) if self.train else 0)
        if self.train: imgs = augment(imgs)
        x = (np.stack([np.asarray(f, np.float32)[:, :, ::-1] for f in imgs]) / 255.0 - MEAN) / STD        # (T,H,W,3) RGB normalised
        return torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), int(r.label), i
