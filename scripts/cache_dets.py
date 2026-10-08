"""Run YOLO11-pose once per clip and cache detections: data/cache/<name>/<clip>.npz  (resumable).

  python scripts/cache_dets.py airt_cam1 'data/airtlab/*/cam1/*.mp4' --threads 1
"""
import argparse, glob, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety.backends import YoloPoseOnnx
from safety.replay import save_dets

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("pattern"); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--target-fps", type=float, default=10.0)
ap.add_argument("--conf", type=float, default=0.15); ap.add_argument("--iou", type=float, default=0.7)
ap.add_argument("--shard", default="0/1", help="i/n: process every n-th clip starting at i (run n processes in parallel)")
a = ap.parse_args()
out_dir = f"data/cache/{a.name}"; os.makedirs(out_dir, exist_ok=True)
be = YoloPoseOnnx(threads=a.threads, conf=a.conf, iou=a.iou)
files = sorted(glob.glob(a.pattern), key=lambda p: (os.path.dirname(p), int(os.path.splitext(os.path.basename(p))[0]) if os.path.basename(p)[:-4].isdigit() else 0, p))
si, sn = (int(v) for v in a.shard.split("/"))
files = files[si::sn]
t0 = time.time()
for k, f in enumerate(files):
    key = f.replace("data/", "").replace("/", "__").rsplit(".", 1)[0]
    dst = f"{out_dir}/{key}.npz"
    if os.path.exists(dst):
        continue
    cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps / a.target_fps)); times, dets, i = [], [], 0
    size = (int(cap.get(3)), int(cap.get(4)))
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % every == 0:
            times.append(i / fps); dets.append(be(fr))
        i += 1
    cap.release()
    save_dets(dst, times, dets, fps / every, size)
    print(f"[{a.name}] {k+1}/{len(files)} {key} frames={len(times)} elapsed={time.time()-t0:.0f}s", flush=True)
print(f"[{a.name}] DONE {len(files)} clips in {time.time()-t0:.0f}s", flush=True)
