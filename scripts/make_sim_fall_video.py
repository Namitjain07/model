"""SIMULATED fall video on REAL pixels: a still photo of a standing person is rotated about the feet.

Real YOLO sees a real person image, but the *motion* is synthetic (rigid rotation of a photo), so this
tests the whole chain (detector -> tracker -> state machine) -- not real-world fall accuracy.

  python scripts/make_sim_fall_video.py outputs/sim_fall.mp4                 # fall, then lie still
  python scripts/make_sim_fall_video.py outputs/sim_recover.mp4 --recover-at 9
"""
import argparse, math
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--image", default="data/images/business-person.png")
ap.add_argument("--seconds", type=float, default=26); ap.add_argument("--fps", type=float, default=10)
ap.add_argument("--fall-at", type=float, default=4.0); ap.add_argument("--fall-dur", type=float, default=0.7)
ap.add_argument("--recover-at", type=float, default=None); ap.add_argument("--wiggle", type=float, default=0.0,
                help="keep moving while lying (degrees of swaying)")
a = ap.parse_args()

W, H = 960, 720
img = cv2.imread(a.image)
h0 = 640
img = cv2.resize(img, (int(img.shape[1] * h0 / img.shape[0]), h0))
bg = np.median(np.concatenate([img[0], img[:, 0], img[:, -1]]), axis=0).astype(np.uint8)
canvas0 = np.full((H, W, 3), bg, np.uint8)
x0, y0 = (W - img.shape[1]) // 2 - 150, H - 40 - h0
canvas0[y0:y0 + h0, x0:x0 + img.shape[1]] = img
pivot = (x0 + img.shape[1] / 2, y0 + h0 - 5)               # feet

def angle(t):
    if t < a.fall_at: return 0.0
    ang = 90.0 * min(1.0, (t - a.fall_at) / a.fall_dur)
    if a.recover_at is not None and t >= a.recover_at:
        ang = max(0.0, 90.0 * (1 - (t - a.recover_at) / 1.5))
    elif t > a.fall_at + a.fall_dur and a.wiggle:
        ang += a.wiggle * math.sin(2 * math.pi * 0.7 * t)
    return ang

vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
rng = np.random.default_rng(0)
for i in range(int(a.seconds * a.fps)):
    t = i / a.fps
    M = cv2.getRotationMatrix2D(pivot, -angle(t), 1.0)      # +angle = topple to the right
    f = cv2.warpAffine(canvas0, M, (W, H), borderValue=tuple(int(v) for v in bg))
    f = np.clip(f.astype(np.int16) + rng.integers(-2, 3, f.shape), 0, 255).astype(np.uint8)   # sensor noise
    vw.write(f)
vw.release(); print("wrote", a.out)
