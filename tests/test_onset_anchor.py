"""Onset-anchored cutting for rubato/sparse sections.

Where the beat grid is fiction, cuts follow real note strikes. These cover the
pure scheduling pieces (no audio): grid-support scoring, span detection, the
per-strike burst shape, and splicing strike cuts over grid cuts.
"""
import numpy as np

from selfies_for_a_year.beats import (
    _BURST_FIRST_GAP_S,
    _BURST_FLOOR_S,
    _BURST_RATIO,
    _BURST_TABLE,
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


def test_burst_shape_is_identical_at_every_strike():
    """The point of the burst: the viewer can predict it, so it must not vary.

    Strikes far enough apart that nothing is damped -- every burst should then be
    the same geometric fan, anchored on its own strike.
    """
    strikes = np.array([0.0, 10.0, 20.0])
    cuts = _onset_anchor_cuts((0.0, 30.0), strikes, lambda t: "intense")
    ts = np.array([t for t, _ in cuts])
    _stride, depth = _BURST_TABLE["intense"]
    assert len(ts) == depth * len(strikes), ts
    shapes = [np.round(ts[i * depth : (i + 1) * depth] - s, 9)
              for i, s in enumerate(strikes)]
    assert all(np.array_equal(shapes[0], s) for s in shapes[1:]), shapes
    # ...and the shape is a decay, each gap _BURST_RATIO x the last.
    gaps = np.diff(shapes[0])
    ratios = gaps[1:] / gaps[:-1]
    assert np.allclose(ratios, _BURST_RATIO), ratios


def test_sparse_tiers_keep_their_pre_burst_cadence():
    """ambient and slow scored well on review, so the burst must not touch them.

    One photo every 2nd (ambient) / 4th (slow) strike, exactly as before bursts
    existed. Regressing these trades an approved behaviour for an unasked-for one.
    """
    strikes = np.arange(0, 20, 1.0)
    for tier, stride in (("ambient", 2), ("slow", 4)):
        assert _BURST_TABLE[tier] == (stride, 1), tier
        cuts = _onset_anchor_cuts((0.0, 20.0), strikes, lambda t, _t=tier: _t)
        ts = [t for t, _ in cuts]
        assert all(abs(g - stride) < 1e-6 for g in np.diff(ts)), (tier, ts)


def test_next_strike_damps_the_burst():
    """A re-strike cuts the previous burst short, like damping a piano key.

    This is what makes dense passages cut fast and sparse ones settle, without
    the shape itself changing.
    """
    lone = _onset_anchor_cuts((0.0, 30.0), np.array([0.0]), lambda t: "intense")
    crowded = _onset_anchor_cuts(
        (0.0, 30.0), np.array([0.0, 0.45, 20.0]), lambda t: "intense"
    )
    after_first = [t for t, _ in crowded if t < 0.45]
    assert len(after_first) < len(lone), (after_first, lone)
    assert after_first[0] == 0.0  # the strike itself still anchors a photo


def test_no_burst_photo_lands_under_the_legibility_floor():
    """A hold too brief to register is a wasted selfie -- drop it, don't show it.

    Strikes placed so the decay keeps colliding with the next trigger; every
    surviving hold must still clear the floor.
    """
    strikes = np.array([0.0, 0.42, 0.83, 1.25, 5.0, 5.3])
    cuts = _onset_anchor_cuts((0.0, 8.0), strikes, lambda t: "normal")
    ts = np.array([t for t, _ in cuts])
    holds = np.diff(np.append(ts, 8.0))
    assert holds.min() >= _BURST_FLOOR_S - 1e-9, sorted(holds)[:5]


def test_onset_anchor_cuts_linger_across_gaps():
    """A silent gap between strikes produces NO cut — we never invent a beat."""
    strikes = np.array([0.0, 1.0, 2.0, 8.0, 9.0])  # 2s..8s is a 6s gap
    cuts = _onset_anchor_cuts((0.0, 10.0), strikes, lambda t: "intense")
    ts = np.array([t for t, _ in cuts])
    # Every cut sits on a strike or inside that strike's burst; the 2s..8s gap
    # gets nothing beyond the tail of the burst that the strike at 2.0 fired.
    burst_end = 2.0 + sum(_BURST_FIRST_GAP_S * _BURST_RATIO**k
                          for k in range(_BURST_TABLE["intense"][1]))
    assert not ((ts > burst_end) & (ts < 8.0)).any(), ts


def test_splice_replaces_only_inside_spans():
    grid = [(t, "normal") for t in np.arange(0, 20, 1.0)]
    strikes = np.array([0.5, 5.5, 10.5])
    spans = [(0.0, 8.0)]
    out = _splice_onset_anchor(grid, spans, strikes, lambda t: "intense")
    # grid cuts in [0,8) gone; the strikes there anchor bursts; grid cuts >=8 kept
    times = [t for t, _ in out]
    assert 0.5 in times and 5.5 in times
    inside = [t for t in times if 0 <= t < 8]
    assert not any(abs(t - round(t)) < 1e-9 for t in inside), inside  # no grid cut left
    assert all(0.5 <= t < 0.5 + 2 or 5.5 <= t < 5.5 + 2 for t in inside), inside
    assert 10.0 in times and 15.0 in times  # untouched grid outside the span
