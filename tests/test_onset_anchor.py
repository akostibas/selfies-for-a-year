"""Onset-anchored cutting for rubato/sparse sections.

Where the beat grid is fiction, cuts follow real note strikes. These cover the
pure scheduling pieces (no audio): grid-support scoring, span detection, the
snapped/strike-driven schedulers, and splicing strike cuts over grid cuts.
"""
import numpy as np

from selfies_for_a_year.beats import (
    _MIN_HOLD_S,
    _SNAP_TOLERANCE_S,
    _STRIKE_STRIDE,
    _even_felt_gap,
    _felt_tier_gaps,
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


def test_even_felt_gap_breaks_ties_toward_the_faster_rung():
    """The bug behind "the long spacing gets boring" (#46).

    "To Build a Home" asks for a photo every 6 beats = 3 felt, exactly between
    the two legal rungs. Rounding up delivered every 8 — a rate the owner had
    already rejected as too slow — so ties now round down.
    """
    assert _even_felt_gap(3.0) == 2
    assert _even_felt_gap(5.0) == 4
    # ...without disturbing the unambiguous cases.
    assert _even_felt_gap(3.9) == 4
    assert _even_felt_gap(2.9) == 2
    assert _even_felt_gap(6.0) == 6
    assert _even_felt_gap(0.1) == 2  # never faster than every other felt beat


def test_felt_tier_gaps_keep_the_density_ordering():
    """intense < normal < slow must survive the even/bar snapping on any grid."""
    for sub in (1 / 16, 1 / 8, 1 / 6, 1 / 4, 1 / 2, 1.0):
        for slow_mult in (0.2, 0.33, 0.5, 1.0):
            g = _felt_tier_gaps(sub, slow_mult)
            assert g["intense"] < g["normal"] < g["slow"], (sub, slow_mult, g)
            assert g["ambient"] == g["slow"]
            assert g["normal"] % 2 == 0


def test_anchored_pace_is_the_tier_pace_not_the_note_count():
    """One rate per tier, both regimes (#46): `normal` cuts every g_normal felt
    beats inside an anchor span too, however many notes the pianist played."""
    felt = np.arange(0.0, 30.0, 0.5)
    gaps = {"intense": 1, "normal": 2, "slow": 4, "ambient": 4}
    sparse = np.array([0.0, 4.3, 11.7, 22.1])
    dense = np.arange(0.0, 30.0, 0.31)
    for strikes in (sparse, dense):
        cuts = _onset_anchor_cuts(
            (0.0, 30.0), strikes, lambda t: "normal",
            felt_beats=felt, felt_gaps=gaps,
        )
        ts = np.array([t for t, _ in cuts])
        # every 2 felt beats = 1.0s, give or take the snap
        assert abs(np.median(np.diff(ts)) - 1.0) <= _SNAP_TOLERANCE_S, ts


def test_ticks_land_on_real_notes_when_there_is_one_nearby():
    """The whole point of snapping: play near the tick and the cut is the note."""
    felt = np.arange(0.0, 20.0, 0.5)
    gaps = {"normal": 2}
    # A strike just inside tolerance of every scheduled tick (every 1.0s).
    strikes = np.arange(0.0, 20.0, 1.0) + 0.08
    cuts = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "normal",
        felt_beats=felt, felt_gaps=gaps,
    )
    ts = [t for t, _ in cuts]
    assert all(any(abs(t - s) < 1e-9 for s in strikes) for t in ts), ts


def test_ticks_out_of_reach_of_a_note_stay_on_the_beat():
    """Beyond tolerance the note is a separate event — don't drag the pace to it."""
    felt = np.arange(0.0, 20.0, 0.5)
    gaps = {"normal": 2}
    strikes = np.arange(0.0, 20.0, 1.0) + 0.30  # 300ms out, way past tolerance
    cuts = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "normal",
        felt_beats=felt, felt_gaps=gaps,
    )
    ts = [t for t, _ in cuts]
    assert all(any(abs(t - f) < 1e-9 for f in felt) for t in ts), ts


def test_lingering_tiers_stay_strike_driven():
    """ambient and slow reviewed at 5/5 with every cut on an attack; the
    re-pacing must not put them on the grid, even when a grid is available."""
    felt = np.arange(0.0, 20.0, 0.5)
    gaps = {"slow": 4, "ambient": 4, "normal": 2}
    strikes = np.array([0.13, 1.37, 2.61, 4.02, 5.44, 6.71, 8.09, 9.33])
    for tier in ("ambient", "slow"):
        cuts = _onset_anchor_cuts(
            (0.0, 20.0), strikes, lambda t, _t=tier: _t,
            felt_beats=felt, felt_gaps=gaps,
        )
        ts = [t for t, _ in cuts]
        assert ts == list(strikes[:: _STRIKE_STRIDE[tier]]), (tier, ts)


def test_anchored_intense_takes_every_note():
    """Inside an anchor span the beat grid is fiction, so "every raw beat" has no
    meaning; the faithful translation of every-beat intense is one cut per note.
    Unlike the felt-locked tiers, intense does NOT ride the grid here."""
    felt = np.arange(0.0, 20.0, 0.5)
    gaps = {"intense": 1, "normal": 2, "slow": 4, "ambient": 4}
    strikes = np.array([0.13, 1.37, 2.61, 4.02, 5.44, 6.71, 8.09, 9.33])
    cuts = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "intense",
        felt_beats=felt, felt_gaps=gaps,
    )
    ts = [t for t, _ in cuts]
    assert ts == list(strikes[:: _STRIKE_STRIDE["intense"]]), ts


def test_lingering_tiers_linger_across_gaps():
    """A silent gap produces NO cut — we never invent a beat between notes."""
    strikes = np.array([0.0, 1.0, 2.0, 8.0, 9.0])  # 2s..8s is a 6s gap
    cuts = _onset_anchor_cuts((0.0, 10.0), strikes, lambda t: "ambient")
    ts = np.array([t for t, _ in cuts])
    assert not ((ts > 2.0) & (ts < 8.0)).any(), ts


def test_no_cut_lands_under_the_legibility_floor():
    """A hold too brief to register is a wasted selfie -- drop it, don't show it.

    A tier change mid-span hands off between two schedulers, which is where a
    pair of near-coincident cuts would otherwise sneak in.
    """
    felt = np.arange(0.0, 8.0, 0.21)
    gaps = {"intense": 1, "normal": 2, "slow": 4, "ambient": 4}
    strikes = np.array([0.0, 0.42, 0.83, 1.25, 1.30, 5.0, 5.05, 5.3])
    cuts = _onset_anchor_cuts(
        (0.0, 8.0), strikes, lambda t: "intense" if t < 1.3 else "normal",
        felt_beats=felt, felt_gaps=gaps,
    )
    ts = np.array([t for t, _ in cuts])
    holds = np.diff(np.append(ts, 8.0))
    assert holds.min() >= _MIN_HOLD_S - 1e-9, sorted(holds)[:5]


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
