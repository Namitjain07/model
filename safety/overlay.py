"""Annotated-frame rendering for visual verification of tracks, states and alerts."""
from __future__ import annotations

import cv2
import numpy as np

from .features import center
from .skeleton import COCO_EDGES, CONF_MIN

PHASE_COLOR = {"UPRIGHT": (0, 200, 0), "DOWN": (0, 165, 255), "UNRESPONSIVE": (0, 0, 255)}


def _txt(img, s, org, color, scale=0.5, k=1.0):
    """Outlined text. `k` pre-compensates for a later downscale so the final text stays legible."""
    sc, th = scale * k, max(1, round(k))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3 * th, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


def draw(img: np.ndarray, pipe, res, show_feats: bool = True, k: float = 1.0) -> np.ndarray:
    out = img.copy()
    fights = pipe.active_fights()
    in_fight = {i for k in fights for i in k}
    by_id = {tr.tid: tr for tr in res.tracks}
    for tr in res.tracks:
        st = pipe.person_state(tr)
        phase = st.get("phase", "UPRIGHT")
        col = (0, 0, 255) if tr.tid in in_fight else PHASE_COLOR.get(phase, (0, 200, 0))
        kp = tr.kpts
        for a, b in COCO_EDGES:
            if kp[a, 2] >= CONF_MIN and kp[b, 2] >= CONF_MIN:
                cv2.line(out, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), col, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = tr.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        tag = f"#{tr.tid} {phase}"
        if tr.tid in in_fight:
            tag += " FIGHT"
        if show_feats and st.get("angle") is not None:
            sp = st.get("spread")
            tag += f" a={st['angle']:.0f}" + (f" s={sp:.2f}" if sp is not None else "")
        _txt(out, tag, (x1, max(int(14 * k), y1 - 6)), col, k=k)
    for (i, j), st in pipe.fight.pairs.items():
        if i in by_id and j in by_id and (st["active"] or st["score"] > 0.15):
            a, b = center(by_id[i].kpts, by_id[i].box), center(by_id[j].kpts, by_id[j].box)
            col = (0, 0, 255) if st["active"] else (0, 200, 255)
            cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), col, 2, cv2.LINE_AA)
            m = ((a + b) / 2).astype(int)
            _txt(out, f"fight {st['score']:.2f}", (int(m[0]) - 30, int(m[1])), col, k=k)
    return out


def banner(img: np.ndarray, lines: list[str], color=(0, 0, 255), k: float = 1.0) -> np.ndarray:
    for n, s in enumerate(lines):
        _txt(img, s, (int(10 * k), int((24 + 22 * n) * k)), color, 0.6, k=k)
    return img
