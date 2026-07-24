"""Onset-anchored cutting for rubato/sparse sections.

Where the beat grid is fiction, cuts follow real note strikes. These cover the
pure scheduling pieces (no audio): grid-support scoring, span detection, the
photos-per-strike cadence, and splicing strike cuts over grid cuts.
"""
import numpy as np

from selfies_for_a_year.beats import (
    _PHOTOS_PER_STRIKE,
    _grid_support,
    _onset_anchor_cuts,
    _onset_anchor_spans,
    _splice_onset_anchor,
)


def test_grid_support_high_when_beats_on_strikes():
    beats = np.arange(0, 10, 0.5)
    strikes = beats + 0.02  # every beat has a strike within 20ms
    assert _grid_support(beats, strikes) == 1.0


def test_grid_support_low_when_grid_is_fiction():
    beats = np.arange(0, 20, 0.42)          # dense fictional grid
    strikes = np.array([0.2, 2.4, 4.6, 9.3, 11.4, 13.4])  # sparse rubato pulse
    assert _grid_support(beats, strikes) < 0.30


def test_onset_anchor_spans_flags_sparse_region():
    # 0-20s sparse (few strikes, dense grid -> low support); 20-40s dense/aligned.
    beats = np.arange(0, 40, 0.42)
    sparse = np.array([0.2, 2.4, 4.6, 9.3, 11.4, 13.4, 15.4, 17.6])
    dense = beats[beats >= 20] + 0.01  # a strike on every beat of the second half
    strikes = np.concatenate([sparse, dense])
    spans = _onset_anchor_spans(beats, strikes, 40.0)
    assert spans, "expected an onset-anchor span for the sparse first half"
    t0, t1 = spans[0]
    assert t0 < 5 and t1 <= 22, spans
    # the aligned second half must NOT be flagged
    assert not any(a >= 25 for a, b in spans)


def test_onset_anchor_spans_respects_min_span():
    # A single low window is shorter than the 8s minimum -> no span.
    beats = np.arange(0, 40, 0.42)
    strikes = np.arange(0, 40, 0.42) + 0.01
    # knock out support in just a 2s pocket by removing strikes there
    strikes = strikes[(strikes < 10) | (strikes > 12)]
    spans = _onset_anchor_spans(beats, strikes, 40.0)
    assert all(b - a >= 8.0 for a, b in spans)


def test_onset_anchor_spans_never_overlap():
    """Two sparse pockets close together must merge, not produce overlapping spans.

    Runs are padded by ±win_s/2, so nearby runs used to emit spans like
    (194,202) and (200,226). _splice_onset_anchor cuts strikes per span, so
    every strike in the overlap was emitted twice — coincident transition times
    that the duration floor turned into a 50ms flicker frame (seen on
    "To Build a Home" at 3:20).
    """
    beats = np.arange(0, 120.0, 0.42)
    # Strikes thinned to sit near the support threshold, so window support
    # wobbles across it the way real sparse passages do — that wobble is what
    # separates two low runs by one or two windows and makes their padded spans
    # overlap. Seed 0 produced (26,46) vs (44,52) before the fix.
    rng = np.random.default_rng(0)
    strikes = beats[rng.random(len(beats)) < 0.32] + 0.01
    spans = _onset_anchor_spans(beats, strikes, 120.0)
    assert len(spans) > 1, "construction should yield several spans"
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"overlapping spans {(a0, a1)} and {(b0, b1)}"


def test_splice_emits_each_strike_once_across_spans():
    """Even given adjacent spans, no strike may produce two cuts at the same time."""
    grid = [(t, "normal") for t in np.arange(0, 30, 1.0)]
    strikes = np.array([2.5, 7.5, 12.5, 17.5])
    spans = [(0.0, 10.0), (10.0, 20.0)]
    out = _splice_onset_anchor(grid, spans, strikes, lambda t: "intense")
    times = [t for t, _ in out]
    assert len(times) == len(set(times)), f"duplicate cut times: {times}"


def test_photos_per_strike_cadence():
    strikes = np.arange(0, 10, 1.0)  # a strike every second, 10 total
    span = (0.0, 10.0)
    for tier, step in _PHOTOS_PER_STRIKE.items():
        cuts = _onset_anchor_cuts(span, strikes, lambda t, _tier=tier: _tier)
        # consecutive cuts are `step` strikes (=`step` seconds here) apart
        ts = [t for t, _ in cuts]
        gaps = np.diff(ts)
        assert all(abs(g - step) < 1e-6 for g in gaps), (tier, ts)


def test_onset_anchor_cuts_linger_across_gaps():
    """A silent gap between strikes produces NO cut — we never invent a beat."""
    strikes = np.array([0.0, 1.0, 2.0, 8.0, 9.0])  # 2s..8s is a 6s gap
    cuts = _onset_anchor_cuts((0.0, 10.0), strikes, lambda t: "intense")  # every strike
    ts = [t for t, _ in cuts]
    assert ts == [0.0, 1.0, 2.0, 8.0, 9.0]  # cut on each strike, nothing in the gap


def test_splice_replaces_only_inside_spans():
    grid = [(t, "normal") for t in np.arange(0, 20, 1.0)]
    strikes = np.array([0.5, 5.5, 10.5])
    spans = [(0.0, 8.0)]
    out = _splice_onset_anchor(grid, spans, strikes, lambda t: "intense")
    # grid cuts in [0,8) gone; strike cuts (0.5, 5.5) present; grid cuts >=8 kept
    times = [t for t, _ in out]
    assert 0.5 in times and 5.5 in times
    assert not any(0 <= t < 8 and t not in (0.5, 5.5) for t in times)
    assert 10.0 in times and 15.0 in times  # untouched grid outside the span
