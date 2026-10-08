"""Learned fight scorer: pairwise window features -> gradient-boosted classifier (trained on AIRTLab)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .events import FightParams
from .features import body_scale
from .flowfeat import FLOW_FEATURES, window_flow_features
from .pairfeat import FEATURES, pair_features

MODEL = Path(__file__).resolve().parent.parent / "models" / "fight_gb_v3.joblib"   # scripts/train_all.py --final


class LearnedFightScorer:
    def __init__(self, path: str | Path | None = None, thr: float | None = None, hold: float | None = None):
        d = joblib.load(path or MODEL)
        self.needs_flow = bool(d.get("flow", False))
        want = FEATURES + FLOW_FEATURES if self.needs_flow else FEATURES
        assert list(d["features"]) == want, "feature set changed; retrain (scripts/train_all.py --final)"
        self.names = want
        self.model, self.thr, self.hold, self.ema = d["model"], (d["thr"] if thr is None else thr), (d["hold"] if hold is None else hold), d["ema"]
        self.dmax = d.get("dmax", 6.0)

    def params(self) -> FightParams:
        return FightParams(ema=self.ema, on_thresh=self.thr, on_hold_s=self.hold,
                           off_thresh=max(0.3, self.thr - 0.25), off_hold_s=2.0, learned_dist=self.dmax)

    def __call__(self, det, a, b, t: float):
        f = pair_features(det, a, b, t)
        if f is None:
            return None
        if self.needs_flow:
            if det.flow is None:
                raise RuntimeError("this model needs optical-flow features: build the pipeline with use_flow=True (learned_pipeline does)")
            flow, dt, times = det.flow.arrays()
            ff = window_flow_features(flow, dt, times, t, det.p.window_s, [a.box, b.box], det.flow.wh, 0.5 * (body_scale(a) + body_scale(b)))
            f = np.concatenate([f, ff])
        return float(self.model.predict_proba(f[None])[0, 1]), {k: round(float(v), 2) for k, v in zip(self.names, f)}
