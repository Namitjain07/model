"""End-to-end: frame -> pose backend -> tracker -> fall/unresponsive + fight detectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import Alert, FallDetector, FallParams, FightDetector, FightParams
from .tracker import Det, Tracker


@dataclass
class FrameResult:
    t: float
    tracks: list
    alerts: list[Alert]


class SafetyPipeline:
    def __init__(self, backend, fall: FallParams | None = None, fight: FightParams | None = None,
                 max_lost_s: float = 2.0, fight_scorer=None, use_flow: bool = False):
        self.backend = backend
        self.tracker = Tracker(max_lost_s=max_lost_s)
        self.fall = FallDetector(fall)
        self.fight = FightDetector(fight, scorer=fight_scorer)
        self.flow = None
        if use_flow:
            from .flowfeat import FlowBuffer
            self.flow = self.fight.flow = FlowBuffer()
        self.log: list[Alert] = []

    def process(self, frame: np.ndarray, t: float) -> FrameResult:
        if self.flow is not None:
            self.flow.push(frame, t)
        return self.step(self.backend(frame), t)

    def step(self, dets: list[Det], t: float) -> FrameResult:
        """Detector half of the pipeline (also used directly by the skeleton simulator/tests)."""
        tracks = self.tracker.update(dets, t)
        alerts: list[Alert] = []
        for tr in tracks:
            alerts += self.fall.update(tr, t)
        alerts += self.fight.update(tracks, t)
        self.log += alerts
        return FrameResult(t, tracks, alerts)

    # --- state for overlays / monitoring -----------------------------------------------
    def person_state(self, tr) -> dict:
        return tr.state.get("fall", {})

    def active_fights(self) -> dict:
        return {k: v for k, v in self.fight.pairs.items() if v["active"]}


def learned_pipeline(backend, model_path: str | None = None, thr: float | None = None, hold: float | None = None, **kw) -> "SafetyPipeline":
    """Pipeline whose fight score comes from the trained classifier (models/fight_gb_v3.joblib by default).
    Models trained with motion features automatically turn on the optical-flow buffer."""
    from .fight_model import LearnedFightScorer
    sc = LearnedFightScorer(model_path, thr=thr, hold=hold)
    return SafetyPipeline(backend, fight=sc.params(), fight_scorer=sc, use_flow=sc.needs_flow, **kw)
