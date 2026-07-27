"""The modulation-depth fallback must be gated by the track's dynamic range.

Ungated, the fallback mislabelled sparse acoustic music: on "To Build a Home" it
put "intense" on the two QUIETEST non-ambient sections (-21 dB and -16 dB) while
the -11 dB string climax read "normal". The term exists to rescue burst trains,
where median energy sits in the gaps between hits -- but on sparse piano the gaps
are silence, and silence modulates the spectral centroid just as hard. Measured
correlation between modulation depth and loudness on that track: +0.05.

So the fallback only earns its keep where loudness CAN'T discriminate. These pin
the gate's direction on the two real tracks it has to separate.
"""
import numpy as np

from selfies_for_a_year.beats import _pace_depth_boost, _pace_dynamics_weight

# Per-beat loudness spreads measured 2026-07-24 (experiments/strike_burst.py's
# sibling analysis): Home swings 17 dB across sections, Push sits at -10..-11 dB.
HOME_IQR_DB = 14.8
PUSH_IQR_DB = 4.7


def _synth(iqr_db: float, n: int = 400) -> np.ndarray:
    """Per-beat loudness whose p75-p25 spread is exactly `iqr_db`."""
    # A uniform ramp has IQR = half its range, so span 2*iqr gives the target.
    return np.linspace(-20.0 - iqr_db, -20.0 + iqr_db, n)


def test_dynamic_track_kills_the_fallback():
    """A track with real dynamics is scored on them, not on texture.

    Asserts the boost the scorer actually applies, not the gate arithmetic --
    dropping the gate from the call site has to fail here.
    """
    assert _pace_dynamics_weight(_synth(HOME_IQR_DB)) == 1.0
    assert _pace_depth_boost(_synth(HOME_IQR_DB), 0.6) == 0.0


def test_compressed_track_keeps_the_fallback():
    """A uniformly loud track has no dynamics to read, so texture must decide.

    Push Upstairs is the approved 5/5 render and depends on this term; gating it
    to zero here would change that timeline.
    """
    w = _pace_dynamics_weight(_synth(PUSH_IQR_DB))
    assert 0.0 < w < 0.5, w
    # Must be the FULL ceiling, not a reduced one. Push sits on a tier boundary:
    # blending it down to 0.43 invented an "intense" section at 2:09 that ate 37
    # photos and cut 16s off the end. Anything less than untouched is a regression.
    assert _pace_depth_boost(_synth(PUSH_IQR_DB), 0.6) == 0.6


def test_gate_is_monotonic_and_bounded():
    weights = [_pace_dynamics_weight(_synth(iqr)) for iqr in (0.0, 3.0, 6.0, 9.0, 20.0)]
    assert weights == sorted(weights), weights
    assert weights[0] == 0.0 and weights[-1] == 1.0
    # Below the 3 dB deadband nothing is trusted: that is measurement noise, not
    # dynamics, and treating it as signal is what over-weights a flat track.
    assert _pace_dynamics_weight(_synth(2.0)) == 0.0
