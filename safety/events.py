"""Event detectors on tracked skeletons: fall / unresponsive person, and fight.

Both are *rule-based with tunable parameters*. Fall defaults were tuned on the
real GMDCSA-24 clips (see scripts/tune_falls.py, subject-wise checked); fight scoring has a learned variant. Treat them as a
starting point; `scripts/eval_clips.py` reports false alarms on normal footage so
thresholds can be tuned, and the same features can feed a trained classifier.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import (body_axis_angle, body_scale, box_aspect, box_iou, center, rel_velocity, spread,
                       torso_angle)
from .skeleton import CONF_MIN, L_ANK, L_HIP, L_KNE, R_ANK, R_HIP, R_KNE, STRIKERS, TARGETS

LOWER = [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]


@dataclass
class Alert:
    kind: str                 # FALL | DOWN | UNRESPONSIVE | RECOVERED | FIGHT | FIGHT_END
    ids: tuple
    t: float
    detail: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# Fall / unconscious person
# ----------------------------------------------------------------------------------------
@dataclass
class FallParams:
    min_size_px: float = 60.0      # ignore people smaller than this (pose too noisy)
    lying_angle: float = 65.0      # torso this far from vertical => lying
    upright_angle: float = 40.0
    lying_axis: float = 45.0       # whole-body (ankle->shoulder) tilt required too, so bending != lying
    lying_hold_s: float = 0.4      # must stay lying this long to confirm a fall
    fall_fast_s: float = 3.0       # upright -> lying faster than this => FALL (else DOWN)
    min_drop: float = 0.0          # optional extra: hip must drop this many torso lengths for FALL
    still_window_s: float = 3.0
    still_spread: float = 0.06     # below this (torso lengths) = motionless
    unresponsive_s: float = 10.0   # lying AND motionless this long => UNRESPONSIVE alarm
    recover_hold_s: float = 1.0
    lost_evidence_s: float = 5.0   # DOWN with no measurable posture this long => reset
    move_spread: float = 0.12      # above this while UNRESPONSIVE => person is moving again


class FallDetector:
    UPRIGHT, DOWN, UNRESPONSIVE = "UPRIGHT", "DOWN", "UNRESPONSIVE"

    def __init__(self, params: FallParams | None = None):
        self.p = params or FallParams()

    def _st(self, tr):
        return tr.state.setdefault("fall", dict(
            phase=self.UPRIGHT, angles=deque(maxlen=3), last_upright_t=None, upright_hip_y=None,
            lying_since=None, still_since=None, upright_since=None, kind=None, angle=None,
            spread=None, lying=False, ok=False, evidence_t=None))

    def update(self, tr, t: float) -> list[Alert]:
        p, st, out = self.p, self._st(tr), []
        kp, box = tr.kpts, tr.box
        size = float(max(box[2] - box[0], box[3] - box[1]))
        st["ok"] = size >= p.min_size_px and (kp[:, 2] >= CONF_MIN).sum() >= 6
        if not st["ok"]:
            return out                                   # too small / too few joints: hold state
        scale = body_scale(tr)
        ang = torso_angle(kp)
        if ang is not None:
            st["angles"].append(ang)
            ang = float(np.median(st["angles"]))         # 3-frame median kills single-frame flips
        asp = box_aspect(box)
        axis = body_axis_angle(kp)
        n_low = int((kp[LOWER, 2] >= CONF_MIN).sum())   # hips/knees/ankles seen
        if ang is not None and axis is not None:       # torso AND legs horizontal => lying (not bending)
            lying = ang > p.lying_angle and axis > p.lying_axis
            upright = ang < p.upright_angle and axis < p.lying_axis
        elif ang is not None:                          # legs unseen: demand a clearly wide box too
            lying = ang > p.lying_angle and asp > 1.0
            upright = ang < p.upright_angle and asp < 1.0
        elif n_low >= 2:                               # lower body seen but torso joints unreliable
            lying, upright = asp > 1.4, asp < 0.8
        else:                                          # partial crop (e.g. head & shoulders close-up):
            lying, upright = False, False              # posture is unjudgeable -> never *start* an alarm
        if ang is not None or n_low >= 2:
            st["evidence_t"] = t
        sp = spread(tr, t, p.still_window_s, scale)
        st.update(angle=ang, spread=sp, lying=lying)

        if st["phase"] == self.UPRIGHT:
            if upright:
                st["last_upright_t"], st["lying_since"] = t, None
                st["upright_hip_y"] = float(center(kp, box)[1])
            elif lying:
                st["lying_since"] = st["lying_since"] or t
                if t - st["lying_since"] >= p.lying_hold_s:
                    lu = st["last_upright_t"]
                    drop = ((float(center(kp, box)[1]) - st["upright_hip_y"]) / scale
                            if st["upright_hip_y"] is not None else 0.0)
                    fast = lu is not None and (st["lying_since"] - lu) <= p.fall_fast_s and drop >= p.min_drop
                    st.update(phase=self.DOWN, kind="FALL" if fast else "DOWN",
                              still_since=None, upright_since=None)
                    out.append(Alert(st["kind"], (tr.tid,), t, dict(drop=round(drop, 2), angle=round(ang or -1, 1))))
            else:
                st["lying_since"] = None
        else:  # DOWN or UNRESPONSIVE
            if (st["phase"] == self.DOWN and st["evidence_t"] is not None
                    and t - st["evidence_t"] > p.lost_evidence_s):
                # we can no longer see enough of the body to justify a pre-alarm: reset (never latch).
                # UNRESPONSIVE is deliberately NOT reset here: a limp person partly hidden stays an alarm.
                st.update(phase=self.UPRIGHT, still_since=None, lying_since=None, last_upright_t=None)
                out.append(Alert("RECOVERED", (tr.tid,), t, dict(reason="lost_evidence")))
                return out
            st["upright_since"] = (st["upright_since"] or t) if upright else None
            if st["upright_since"] and t - st["upright_since"] >= p.recover_hold_s:
                st.update(phase=self.UPRIGHT, still_since=None, lying_since=None, last_upright_t=t)
                out.append(Alert("RECOVERED", (tr.tid,), t))
                return out
            if st["phase"] == self.DOWN:
                if sp is not None and sp < p.still_spread and lying:
                    st["still_since"] = st["still_since"] or t
                    if t - st["still_since"] >= p.unresponsive_s:
                        st["phase"] = self.UNRESPONSIVE
                        out.append(Alert("UNRESPONSIVE", (tr.tid,), t, dict(still_s=round(t - st["still_since"], 1))))
                else:
                    st["still_since"] = None
            else:  # UNRESPONSIVE -> only a clear movement clears it
                if sp is not None and sp > p.move_spread:
                    st.update(phase=self.DOWN, still_since=None)
                    out.append(Alert("RECOVERED", (tr.tid,), t, dict(reason="moving")))
        return out


# ----------------------------------------------------------------------------------------
# Fight
# ----------------------------------------------------------------------------------------
@dataclass
class FightParams:
    window_s: float = 1.5
    interact_dist: float = 4.0       # centre distance (torso lengths) beyond which pairs are ignored
    close_dist: float = 2.0          # fully "in reach" below this
    strike_speed: float = 2.5        # limb approach speed toward other's head/torso (torso lengths/s)
    strike_reach: float = 1.0        # ...while within this distance of a target joint (torso lengths)
    strike_rate_full: float = 2.0    # strike-frames per second that saturates the strike term
    energy_lo: float = 1.5           # pair limb-energy (torso lengths/s) where the energy term starts
    energy_hi: float = 5.0
    # Energy alone must never reach on_thresh (max w_energy + w_overlap < on_thresh): raw movement
    # (dancing, jitter, sport) is not a fight without directed strikes or close-contact grappling.
    w_strike: float = 0.50
    w_energy: float = 0.20
    w_overlap: float = 0.20
    ema: float = 0.35
    on_thresh: float = 0.45
    on_hold_s: float = 1.0
    off_thresh: float = 0.25
    off_hold_s: float = 2.0
    min_size_px: float = 60.0
    learned_dist: float = 6.0        # learned scorer: consider pairs closer than this (torso lengths)


class FightDetector:
    def __init__(self, params: FightParams | None = None, scorer=None):
        """scorer: optional callable(detector, track_a, track_b, t) -> (prob, feats) | None. When given, it
        replaces the hand-tuned rule score (which is ~chance on real footage, see README)."""
        self.p = params or FightParams()
        self.scorer = scorer
        self.flow = None                     # FlowBuffer / PrecomputedFlow, set by the pipeline when the model needs motion features
        self.pairs: dict[tuple, dict] = {}

    # --- per-pair features ---------------------------------------------------------------
    def _strikes(self, a, b, t: float) -> float:
        """Strike frames per second of `a` toward `b` within the window."""
        p = self.p
        sa, sb = body_scale(a), body_scale(b)
        wa = {h[0]: h for h in a.window(t, p.window_s)}
        wb = {h[0]: h for h in b.window(t, p.window_s)}
        times = sorted(set(wa) & set(wb))
        series = {j: ([], []) for j in STRIKERS}                       # per joint: (times, dist to nearest target)
        for ts in times:
            ka, kb = wa[ts][1], wb[ts][1]
            tg = [j for j in TARGETS if kb[j, 2] >= CONF_MIN]
            if not tg:
                continue
            for j in STRIKERS:
                if ka[j, 2] >= CONF_MIN:
                    series[j][0].append(ts)
                    series[j][1].append(float(np.min(np.linalg.norm(kb[tg, :2] - ka[j, :2], axis=1))) / sb)
        hits = set()
        for j, (ts, d) in series.items():
            if len(d) < 5:
                continue
            ts, d = np.asarray(ts), np.convolve(d, np.ones(3) / 3, mode="valid")   # 3-tap smoothing
            tm = ts[1:-1]
            for k in range(2, len(d)):
                # a strike = *sustained* approach: 3 consecutive smoothed samples all closing in.
                # (single-frame jumps are keypoint jitter, not punches)
                if d[k] < d[k - 1] < d[k - 2] and tm[k] > tm[k - 2]:
                    closing = (d[k - 2] - d[k]) / (tm[k] - tm[k - 2])      # torso lengths / s
                    if closing > p.strike_speed and d[k] < p.strike_reach:
                        hits.add(float(tm[k]))
        return len(hits) / max(p.window_s, 1e-3)

    def _energy(self, tr, t: float) -> float:
        """Limb energy: 80th-percentile relative limb speed over the window (torso lengths/s)."""
        _, v = rel_velocity(_Win(tr, t, self.p.window_s), STRIKERS, body_scale(tr))
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 80)) if v.size else 0.0

    # --- main ----------------------------------------------------------------------------
    def update(self, tracks: list, t: float) -> list[Alert]:
        p, out = self.p, []
        live = [tr for tr in tracks
                if max(tr.box[2] - tr.box[0], tr.box[3] - tr.box[1]) >= p.min_size_px]
        seen = set()
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                a, b = live[i], live[j]
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                seen.add(key)
                st = self.pairs.setdefault(key, dict(score=0.0, raw=0.0, on_since=None, off_since=None,
                                                     active=False, feats={}))
                sa, sb = body_scale(a), body_scale(b)
                dist = float(np.linalg.norm(center(a.kpts, a.box) - center(b.kpts, b.box))) / ((sa + sb) / 2)
                if self.scorer is not None:
                    got = self.scorer(self, a, b, t) if dist <= p.learned_dist else None
                    raw, feats = (got[0], got[1]) if got is not None else (0.0, dict(dist=round(dist, 2)))
                elif dist > p.interact_dist:
                    raw, feats = 0.0, dict(dist=round(dist, 2))
                else:
                    strikes = self._strikes(a, b, t) + self._strikes(b, a, t)
                    ea, eb = self._energy(a, t), self._energy(b, t)
                    energy = 0.5 * (max(ea, eb) + min(ea, eb))
                    iou = box_iou(a.box, b.box)
                    s_prox = float(np.clip((p.interact_dist - dist) / (p.interact_dist - p.close_dist), 0, 1))
                    s_strike = float(np.clip(strikes / p.strike_rate_full, 0, 1))
                    s_energy = float(np.clip((energy - p.energy_lo) / (p.energy_hi - p.energy_lo), 0, 1))
                    s_over = float(np.clip(iou / 0.3, 0, 1))
                    raw = s_prox * (p.w_strike * s_strike + p.w_energy * s_energy + p.w_overlap * s_over)
                    feats = dict(dist=round(dist, 2), strikes=round(strikes, 2), energy=round(energy, 2),
                                 iou=round(iou, 2))
                st["raw"] = raw
                st["score"] = (1 - p.ema) * st["score"] + p.ema * raw
                st["feats"] = feats
                if st["score"] >= p.on_thresh:
                    st["off_since"] = None
                    st["on_since"] = st["on_since"] or t
                    if not st["active"] and t - st["on_since"] >= p.on_hold_s:
                        st["active"] = True
                        out.append(Alert("FIGHT", key, t, dict(score=round(st["score"], 2), **feats)))
                elif st["score"] <= p.off_thresh:
                    st["on_since"] = None
                    if st["active"]:
                        st["off_since"] = st["off_since"] or t
                        if t - st["off_since"] >= p.off_hold_s:
                            st["active"] = False
                            out.append(Alert("FIGHT_END", key, t))
        for key in list(self.pairs):                       # decay pairs not currently visible
            if key not in seen:
                st = self.pairs[key]
                st["score"] *= 0.8
                if st["active"] and st["score"] < p.off_thresh:
                    st["active"] = False
                    out.append(Alert("FIGHT_END", key, t, dict(reason="pair lost")))
                if st["score"] < 0.01 and not st["active"]:
                    del self.pairs[key]
        return out


class _Win:
    """View of a track restricted to the last `seconds` (so rel_velocity sees only the window)."""
    def __init__(self, tr, t: float, seconds: float):
        self.hist = tr.window(t, seconds)
