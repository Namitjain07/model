"""Run hand + pose landmarking on still images, save overlays to outputs/probe/."""
import sys, glob, os
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.landmarks import HandTracker, PoseTracker
from safety.viz import draw_hands, draw_poses, banner

os.makedirs("outputs/probe", exist_ok=True)
hands = HandTracker(num_hands=4, video=False)
poses = PoseTracker(num_poses=4, video=False)
for path in sorted(glob.glob("data/images/*")):
    img = cv2.imread(path)
    if img is None:
        print("unreadable", path); continue
    h, p = hands.process(img), poses.process(img)
    out = draw_poses(draw_hands(img.copy(), h.lm), p.lm)
    banner(out, f"hands={len(h.lm)} poses={len(p.lm)}")
    name = os.path.basename(path).rsplit(".", 1)[0]
    cv2.imwrite(f"outputs/probe/{name}.jpg", out)
    print(f"{os.path.basename(path):28s} {img.shape[1]}x{img.shape[0]}  hands={len(h.lm)} {h.label}  poses={len(p.lm)}")
hands.close()
poses.close()
