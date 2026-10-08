"""Logic tests on simulated skeletons (see safety/sim.py: these check intent, not real-world accuracy)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.pipeline import SafetyPipeline
from safety.sim import (Person, companions, dancers, fall_person, fighters, handshake, jittery_pair, ramp, run)


def kinds(log):
    return [a.kind for a in log]


def pipe():
    return SafetyPipeline(backend=None)


def test_walking_person_no_alerts():
    log = run(pipe(), [Person(pos=lambda t: (100 + 140 * t, 300.0), walking=lambda t: True)], 20)
    assert kinds(log) == []


def test_standing_still_never_alerts():
    log = run(pipe(), [Person()], 30)
    assert kinds(log) == []


def test_hard_fall_then_motionless_raises_unresponsive():
    log = run(pipe(), [fall_person(t_fall=3.0, dur=0.6)], 22)
    k = kinds(log)
    assert "FALL" in k, k
    assert "UNRESPONSIVE" in k, k
    t_fall = next(a.t for a in log if a.kind == "FALL")
    t_unr = next(a.t for a in log if a.kind == "UNRESPONSIVE")
    assert 3.0 <= t_fall <= 5.0
    assert 9.5 <= t_unr - t_fall <= 14.5            # ~10 s of stillness after the fall


def test_fall_then_recovery_does_not_alarm():
    log = run(pipe(), [fall_person(t_fall=3.0, recover_at=8.0)], 16)
    k = kinds(log)
    assert "FALL" in k and "RECOVERED" in k, k
    assert "UNRESPONSIVE" not in k, k


def test_lying_but_moving_is_not_unresponsive():
    log = run(pipe(), [fall_person(t_fall=3.0, wiggle=0.5)], 25)
    k = kinds(log)
    assert any(x in k for x in ("FALL", "DOWN")), k
    assert "UNRESPONSIVE" not in k, k


def test_bending_over_is_not_a_fall():
    # upper body folds to ~85 deg for 3 s while legs stay vertical (picking something up)
    p = Person(bend=lambda t: ramp(3, 4, 0, 85)(t) if t < 7 else ramp(7, 8, 85, 0)(t))
    log = run(pipe(), [p], 14)
    assert kinds(log) == [], kinds(log)


def test_fight_is_detected():
    log = run(pipe(), fighters(), 10)
    assert "FIGHT" in kinds(log), [(a.kind, round(a.t, 1)) for a in log]
    assert next(a.t for a in log if a.kind == "FIGHT") <= 6.0


def test_handshake_is_not_a_fight():
    assert "FIGHT" not in kinds(run(pipe(), handshake(), 12))


def test_companions_walking_together_is_not_a_fight():
    assert "FIGHT" not in kinds(run(pipe(), companions(), 15))


def test_far_apart_people_never_fight():
    a = fighters(gap=1.5)
    a[1].pos = lambda t: (900.0, 300.0)             # same motion but 6 m away
    assert "FIGHT" not in kinds(run(pipe(), a, 10))


def test_keypoint_jitter_alone_is_not_a_fight():
    # energy without directed strikes must not alarm
    assert "FIGHT" not in kinds(run(pipe(), jittery_pair(), 12))


def test_gentle_dancing_is_not_a_fight():
    assert "FIGHT" not in kinds(run(pipe(), dancers(amp=0.3), 12))


@pytest.mark.xfail(strict=True, reason="KNOWN LIMITATION: vigorous dancing with arms swung at the partner is "
                   "kinematically indistinguishable from striking. Needs real negatives + a trained classifier.")
def test_vigorous_dancing_is_not_a_fight():
    assert "FIGHT" not in kinds(run(pipe(), dancers(amp=0.6), 12))


def test_head_and_shoulders_closeup_tilting_is_not_a_fall():
    # Regression: webcam close-up (no hips/legs visible); head tilt makes the box wide. Seen on real footage.
    p = Person(crop=True, bend=lambda t: ramp(3, 3.5, 0, 75)(t) if t < 8 else ramp(8, 8.5, 75, 0)(t))
    assert kinds(run(pipe(), [p], 40)) == []


def test_down_does_not_latch_when_posture_becomes_unmeasurable():
    # Regression: fall detected, person gets up, but is then only partly visible (no hips) -> the
    # DOWN state used to latch forever because "upright" could never be confirmed.
    p = fall_person(t_fall=2.0, recover_at=6.0)
    p.crop = lambda t: t >= 6.9                     # standing again but cropped to head & shoulders
    log = run(pipe(), [p], 25)
    k = kinds(log)
    assert "FALL" in k and "RECOVERED" in k, k
    assert "UNRESPONSIVE" not in k, k
