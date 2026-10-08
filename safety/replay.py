"""Cache per-frame pose detections to disk and replay them through the pipeline (fast tuning/eval)."""
from __future__ import annotations

import numpy as np

from .tracker import Det


def save_dets(path: str, times: list[float], dets_per_frame: list[list[Det]], fps: float, size: tuple):
    n = np.array([len(d) for d in dets_per_frame], np.int32)
    flat = [d for fr in dets_per_frame for d in fr]
    kp = np.stack([d.kpts for d in flat]) if flat else np.zeros((0, 17, 3), np.float32)
    bx = np.stack([d.box for d in flat]) if flat else np.zeros((0, 4), np.float32)
    sc = np.array([d.score for d in flat], np.float32)
    np.savez_compressed(path, t=np.array(times, np.float32), n=n, kpts=kp, boxes=bx, scores=sc,
                        fps=np.float32(fps), size=np.array(size, np.int32))


def load_dets(path: str):
    """-> (list[(t, list[Det])], meta). Frames with zero detections are kept (the tracker needs them)."""
    z = np.load(path)
    out, k = [], 0
    for t, n in zip(z["t"], z["n"]):
        out.append((float(t), [Det(z["kpts"][k + i], z["boxes"][k + i], float(z["scores"][k + i])) for i in range(int(n))]))
        k += int(n)
    return out, dict(fps=float(z["fps"]), size=tuple(int(v) for v in z["size"]))


def run_cached(pipe, path: str):
    """Replay cached detections through `pipe`; return per-frame traces + alerts."""
    frames, meta = load_dets(path)
    ts, score, raw, maxp = [], [], [], 0
    for t, dets in frames:
        pipe.step(dets, t)
        pairs = pipe.fight.pairs.values()
        ts.append(t)
        score.append(max((p["score"] for p in pairs), default=0.0))
        raw.append(max((p["raw"] for p in pairs), default=0.0))
        maxp = max(maxp, len(pipe.tracker.tracks))
    return dict(t=np.array(ts), score=np.array(score), raw=np.array(raw), alerts=list(pipe.log), meta=meta, max_people=maxp)
