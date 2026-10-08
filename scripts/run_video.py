"""Run the safety pipeline on a video: alert timeline, timing, annotated video + contact sheet.

  python scripts/run_video.py data/videos/people-detection.mp4 --every 1 --sheet
  python scripts/run_video.py clip.mp4 --backend mediapipe --out outputs/clip_annot.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety.overlay import banner, draw  # noqa: E402
from safety.pipeline import SafetyPipeline  # noqa: E402


def make_backend(name: str, **kw):
    if name == "yolo":
        from safety.backends import YoloPoseOnnx
        return YoloPoseOnnx(**kw)
    from safety.backends import MediaPipeBackend
    return MediaPipeBackend(num_poses=kw.get("num_poses", 4))


def run(video: str, backend, every: int = 1, max_seconds: float | None = None, out: str | None = None,
        sheet: str | None = None, tiles: int = 6, width: int = 960, pipe: SafetyPipeline | None = None,
        quiet: bool = False) -> dict:
    pipe = pipe or SafetyPipeline(backend)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total if max_seconds is None else min(total, int(max_seconds * fps))
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, width / W0)
    size = (int(W0 * scale), int(H0 * scale))
    writer = None
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, size)
    snaps, ms, n_det, max_conc, i, used = [], [], 0, 0, 0, 0
    alert_frames: list[tuple[int, str]] = []
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every:
            i += 1
            continue
        t = i / fps
        t0 = time.perf_counter()
        res = pipe.process(frame, t)
        ms.append((time.perf_counter() - t0) * 1000)
        used += 1
        n_det += len(res.tracks) > 0
        max_conc = max(max_conc, len(res.tracks))
        need_img = writer is not None or sheet
        if need_img:
            k = 1.0 / scale if scale < 1 else 1.0         # keep text legible after downscaling
            img = draw(frame, pipe, res, k=k)
            lines = [f"t={t:5.1f}s  tracks={len(res.tracks)}"]
            for a in pipe.log[-3:]:
                if t - a.t < 4:
                    lines.append(f"{a.kind} {a.ids} @{a.t:.1f}s")
            img = cv2.resize(banner(img, lines, k=k), size)
            if writer:
                writer.write(img)
            if res.alerts:
                alert_frames.append((len(snaps), res.alerts[0].kind))
            snaps.append((i, img))
        i += 1
    cap.release()
    if writer:
        writer.release()
    if sheet and snaps:
        pick = [k for k, _ in alert_frames][:tiles]
        if len(pick) < tiles:                       # fill with evenly spaced frames
            pick = sorted(set(pick) | set(np.linspace(0, len(snaps) - 1, tiles - len(pick)).astype(int).tolist()))
        rows, cur = [], []
        for k in pick[:tiles]:
            tile = cv2.resize(snaps[k][1], (480, int(480 * size[1] / size[0])))
            cv2.putText(tile, f"frame {snaps[k][0]}", (6, tile.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cur.append(tile)
            if len(cur) == 3:
                rows.append(np.hstack(cur)); cur = []
        if cur:
            cur += [np.zeros_like(cur[0])] * (3 - len(cur)); rows.append(np.hstack(cur))
        os.makedirs(os.path.dirname(sheet) or ".", exist_ok=True)
        cv2.imwrite(sheet, np.vstack(rows))
    arr = np.array(ms[3:]) if len(ms) > 3 else np.array(ms or [0.0])
    summary = dict(video=os.path.basename(video), frames=used, seconds=round(i / fps, 1),
                   frames_with_people=round(n_det / max(used, 1), 2), max_people=max_conc,
                   ms_mean=round(float(arr.mean()), 1), ms_p95=round(float(np.percentile(arr, 95)), 1),
                   alerts=[dict(kind=a.kind, ids=list(a.ids), t=round(a.t, 1), **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in a.detail.items()}) for a in pipe.log])
    if not quiet:
        print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--backend", default="yolo", choices=["yolo", "mediapipe"])
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame (emulates board FPS)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--out", help="annotated mp4 path")
    ap.add_argument("--sheet", action="store_true", help="save contact sheet to outputs/sheets/")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.video))[0]
    run(a.video, make_backend(a.backend), a.every, a.max_seconds, a.out,
        f"outputs/sheets/{name}_{a.backend}.jpg" if a.sheet else None)
