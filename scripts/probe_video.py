"""Pose (+ optional hands) over a video: timing, detection rate, annotated contact sheet."""
import argparse, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.landmarks import PoseTracker, HandTracker
from safety.viz import draw_poses, draw_hands, banner

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--poses", type=int, default=3)
ap.add_argument("--variant", default="lite"); ap.add_argument("--max-frames", type=int, default=300)
ap.add_argument("--hands", action="store_true"); ap.add_argument("--tiles", type=int, default=6)
a = ap.parse_args()

pose = PoseTracker(num_poses=a.poses, variant=a.variant, video=True)
hand = HandTracker(num_hands=2, video=True) if a.hands else None
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), a.max_frames)
keep = set(np.linspace(0, total - 1, a.tiles).astype(int).tolist())
t_pose, t_hand, n_with, tiles, i = [], [], 0, [], 0
while i < total:
    ok, f = cap.read()
    if not ok: break
    ts = int(i * 1000 / fps)
    t0 = time.perf_counter(); p = pose.process(f, ts); t_pose.append(time.perf_counter() - t0)
    h = None
    if hand:
        t0 = time.perf_counter(); h = hand.process(f, ts); t_hand.append(time.perf_counter() - t0)
    n_with += len(p.lm) > 0
    if i in keep:
        o = draw_poses(f.copy(), p.lm)
        if h is not None: draw_hands(o, h.lm)
        banner(o, f"frame {i}  poses={len(p.lm)}" + (f" hands={len(h.lm)}" if h else ""))
        tiles.append(cv2.resize(o, (480, int(480 * f.shape[0] / f.shape[1]))))
    i += 1
tp = np.array(t_pose[5:]) * 1000  # skip warm-up
print(f"{os.path.basename(a.video)} [{a.variant}] frames={i} pose-detected={n_with/i:.0%}")
print(f"  pose  ms/frame: mean={tp.mean():.1f} p50={np.median(tp):.1f} p95={np.percentile(tp,95):.1f}  -> {1000/tp.mean():.1f} FPS on this CPU")
if t_hand:
    th = np.array(t_hand[5:]) * 1000
    print(f"  hands ms/frame: mean={th.mean():.1f} p95={np.percentile(th,95):.1f}")
rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles) - len(tiles) % 3, 3)]
os.makedirs("outputs/probe", exist_ok=True)
out = f"outputs/probe/sheet_{os.path.splitext(os.path.basename(a.video))[0]}.jpg"
cv2.imwrite(out, np.vstack(rows)); print("saved", out)
pose.close()
if hand: hand.close()
