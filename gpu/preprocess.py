"""Standardise every clip: fixed 5 s window -> YOLOX-M person boxes -> person-centric square crop -> 224x224 JPEG frame stacks.

 * window   : whole clip if <= 7 s, else the central 5 s (RLVS has clips of several minutes)
 * frames   : 16 uniformly spaced frames per phase; 2 phases (offset by half a step) so training can jitter timing
 * crop     : union of YOLOX-M person boxes seen on 3 frames (+12% margin, min side 40% of the short image side, square);
              no person found -> whole frame letterboxed (grey padding, like the YOLO convention; replicated borders produced streak artefacts).  Wide CCTV shots become person-sized, which is what the video model needs.
 * output   : gpu_cache/frames/<clip>.pkl  (JPEG bytes, ~0.5 MB per clip)
  python gpu/preprocess.py [--workers 8] [--limit N]      (N>0: balanced smoke subset)
"""
import argparse, os, pickle, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import CACHE, ROOT, load_manifest

T, PHASES, WIN, SIZE, MAXSIDE = 16, 2, 5.0, 224, 960
FR = CACHE / "frames"
_det = None


def safe(cid: str) -> str:
    return cid.replace(":", "__").replace("/", "_")


def _init():
    global _det
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import onnxruntime as ort
    from safety.crop_pose import YoloxPerson
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    _det = YoloxPerson(providers=prov)


def square_crop(img, cx, cy, S):
    S = int(round(S)); x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2)); H, W = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + S - W, y0 + S - H)
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114)) if pad else img
    c = src[y0 + pad:y0 + pad + S, x0 + pad:x0 + pad + S]
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA if S > SIZE else cv2.INTER_CUBIC)


def process(row):
    out = FR / (safe(row["clip_id"]) + ".pkl")
    if out.exists():
        return row["clip_id"], "cached"
    cap = cv2.VideoCapture(str(ROOT / row["path"])); fps = cap.get(5) or 25.0; n = int(cap.get(7))
    if n <= 0:
        return row["clip_id"], "unreadable"
    dur = n / fps
    start, length = (0.0, min(dur, WIN)) if dur <= 1.4 * WIN else ((dur - WIN) / 2, WIN)
    s_idx, e_idx = int(start * fps), max(int(start * fps) + 2, min(n, int((start + length) * fps)))
    want = {}                                                    # source frame index -> list of slots
    for ph in range(PHASES):
        for k in range(T):
            want.setdefault(min(e_idx - 1, int(s_idx + (k + ph / PHASES) * (e_idx - s_idx) / T)), []).append((ph, k))
    if s_idx > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
    frames, i = {}, s_idx
    while i < e_idx:
        ok, f = cap.read()
        if not ok: break
        if i in want:
            h, w = f.shape[:2]; sc = min(1.0, MAXSIDE / max(h, w))
            frames[i] = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else f
        i += 1
    cap.release()
    if len(frames) < 4:
        return row["clip_id"], "too_few_frames"
    keys = sorted(frames); last = frames[keys[-1]]
    stack = {}
    for fi, slots in want.items():
        f = frames[fi] if fi in frames else frames[min(keys, key=lambda x: abs(x - fi))]
        for s in slots: stack[s] = f
    H, W = stack[(0, 0)].shape[:2]
    boxes = []
    for q in (0.15, 0.5, 0.85):
        b, s = _det(stack[(0, min(T - 1, int(q * T)))])
        boxes += [bb for bb, ss in zip(b, s) if (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 0.004 * H * W]
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:12]
    if boxes:
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes); x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        S = max(x2 - x1, y2 - y1) * 1.24; S = max(S, 0.40 * min(H, W)); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        S, cx, cy = max(H, W), W / 2, H / 2                       # whole frame, letterboxed (grey border)
    jp = [None] * (PHASES * T)
    for (ph, k), f in stack.items():
        ok, buf = cv2.imencode(".jpg", square_crop(f, cx, cy, S), [cv2.IMWRITE_JPEG_QUALITY, 92]); jp[ph * T + k] = buf.tobytes()
    pickle.dump(dict(frames=jp, crop=(float(cx), float(cy), float(S)), n_people=len(boxes), window=(start, length), hw=(H, W)), open(out, "wb"))
    return row["clip_id"], "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    FR.mkdir(parents=True, exist_ok=True); df = load_manifest()
    if a.limit:                                                    # balanced smoke subset: same number per (dataset, label)
        per = max(1, a.limit // (2 * df.dataset.nunique())); df = df.groupby(["dataset", "label"], group_keys=False).head(per)
    rows = df.to_dict("records"); stats = {}
    with Pool(a.workers, initializer=_init) as pool:
        for k, (cid, st) in enumerate(pool.imap_unordered(process, rows, chunksize=2)):
            stats[st] = stats.get(st, 0) + 1
            if (k + 1) % 100 == 0 or k + 1 == len(rows): print(f"{k+1}/{len(rows)} {stats}", flush=True)
    print("done", stats)
