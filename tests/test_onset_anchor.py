"""Onset-anchored cutting for rubato/sparse sections.

Where the beat grid is fiction, AMBIENT cuts follow real note strikes while the
metronome tiers (intense/normal/slow) keep their grid pulse (issue #48). These
cover the pure scheduling pieces (no audio): grid-support scoring, span
detection, the strike-driven scheduler, and splicing over grid cuts.
"""
import numpy as np

from selfies_for_a_year.beats import (
    _MIN_HOLD_S,
    _NOTE_DRIVEN_TIERS,
    _STRIKE_STRIDE,
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
    rng = np.random.default_rng(0)
    strikes = beats[rng.random(len(beats)) < 0.32] + 0.01
    spans = _onset_anchor_spans(beats, strikes, 120.0)
    assert len(spans) > 1, "construction should yield several spans"
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"overlapping spans {(a0, a1)} and {(b0, b1)}"


def test_felt_tier_gaps_is_the_fixed_metronomic_ladder():
    """A fixed halving ladder in felt beats (issue #48): normal 1, slow 2, no
    per-song tuning. Intense is every raw beat and is not read from here."""
    g = _felt_tier_gaps()
    assert g == {"normal": 1, "slow": 2, "ambient": 2}
    assert g["normal"] < g["slow"]


def test_ambient_stays_strike_driven_even_with_a_grid():
    """Ambient reviewed at 5/5 following the actual notes; a grid covering the
    span must not pull it onto the metronome."""
    strikes = np.array([0.13, 1.37, 2.61, 4.02, 5.44, 6.71, 8.09, 9.33])
    grid = np.arange(0.0, 20.0, 1.0)  # a full metronome grid over the span
    cuts = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "ambient", grid_cuts=grid,
    )
    ts = [t for t, _ in cuts]
    assert ts == list(strikes[:: _STRIKE_STRIDE["ambient"]]), ts
    assert _NOTE_DRIVEN_TIERS == frozenset({"ambient"})


def test_metronome_tiers_emit_no_anchor_cuts_when_grid_covers_them():
    """intense/normal/slow keep their felt-grid pulse through a span, so where a
    retained grid cut covers the stretch the filler emits nothing."""
    strikes = np.array([0.13, 1.37, 2.61, 4.02, 5.44, 6.71, 8.09, 9.33])
    grid = np.arange(0.0, 20.0, 1.0)
    for tier in ("intense", "normal", "slow"):
        cuts = _onset_anchor_cuts(
            (0.0, 20.0), strikes, lambda t, _t=tier: _t, grid_cuts=grid,
        )
        assert cuts == [], (tier, cuts)


def test_uncovered_stretches_note_count_as_a_fallback():
    """No grid cut covering a stretch (past the last beat, or no felt grid at all)
    -> every tier follows the notes at its stride, so no silent gap opens up."""
    strikes = np.array([0.13, 1.37, 2.61, 4.02, 5.44, 6.71, 8.09, 9.33])
    slow = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "slow", grid_cuts=(),
    )
    assert [t for t, _ in slow] == list(strikes[:: _STRIKE_STRIDE["slow"]])
    normal = _onset_anchor_cuts(
        (0.0, 20.0), strikes, lambda t: "normal", grid_cuts=(),
    )
    assert [t for t, _ in normal] == list(strikes)  # stride 1 default


def test_ambient_lingers_across_gaps():
    """A silent gap produces NO cut — we never invent a beat between notes."""
    strikes = np.array([0.0, 1.0, 2.0, 8.0, 9.0])  # 2s..8s is a 6s gap
    cuts = _onset_anchor_cuts((0.0, 10.0), strikes, lambda t: "ambient")
    ts = np.array([t for t, _ in cuts])
    assert not ((ts > 2.0) & (ts < 8.0)).any(), ts


def test_no_cut_lands_under_the_legibility_floor():
    """A hold too brief to register is a wasted selfie -- drop it, don't show it.
    Near-coincident strikes must not produce a sub-floor flash."""
    strikes = np.array([0.0, 0.05, 1.0, 1.03, 5.0, 5.3])  # some pairs within a frame
    cuts = _onset_anchor_cuts(
        (0.0, 8.0), strikes, lambda t: "ambient", grid_cuts=(),
    )
    ts = np.array([t for t, _ in cuts])
    holds = np.diff(np.append(ts, 8.0))
    assert holds.min() >= _MIN_HOLD_S - 1e-9, sorted(holds)[:5]


def test_splice_retains_metronome_grid_cuts_in_spans():
    """The core of issue #48: a normal-tier grid cut inside an anchor span STAYS
    (metronome-locked), rather than being replaced by note-following cuts."""
    grid = [(t, "normal") for t in np.arange(0, 20, 1.0)]
    strikes = np.array([0.5, 3.7, 5.5])
    spans = [(0.0, 8.0)]
    out = _splice_onset_anchor(
        grid, spans, strikes, lambda t: "normal", has_grid=True,
    )
    times = [t for t, _ in out]
    # every original grid cut survives; no strike-driven cut was inserted
    assert times == [float(t) for t in np.arange(0, 20, 1.0)], times


def test_splice_replaces_ambient_grid_cuts_in_spans():
    """An ambient stretch inside a span DOES get note-following cuts, and its
    grid cuts are dropped — the metronome tiers around it are unaffected."""
    grid = [(t, "ambient") for t in np.arange(0, 20, 1.0)]
    strikes = np.array([0.5, 2.5, 5.5, 7.0])  # 4 in-span notes; ambient stride 2
    spans = [(0.0, 8.0)]
    out = _splice_onset_anchor(
        grid, spans, strikes, lambda t: "ambient", has_grid=True,
    )
    times = [t for t, _ in out]
    inside = [t for t in times if 0 <= t < 8]
    assert not any(abs(t - round(t)) < 1e-9 for t in inside), inside  # grid gone
    assert inside == [0.5, 5.5], inside                               # every 2nd note
    assert 10.0 in times and 15.0 in times                           # outside kept


def test_splice_without_grid_replaces_all_in_span():
    """No felt grid -> every in-span grid cut is replaced by note-driven cuts."""
    grid = [(t, "normal") for t in np.arange(0, 20, 1.0)]
    strikes = np.array([0.5, 5.5, 10.5])
    spans = [(0.0, 8.0)]
    out = _splice_onset_anchor(
        grid, spans, strikes, lambda t: "normal", has_grid=False,
    )
    times = [t for t, _ in out]
    inside = [t for t in times if 0 <= t < 8]
    assert not any(abs(t - round(t)) < 1e-9 for t in inside), inside
    assert 0.5 in times and 5.5 in times
    assert 10.0 in times and 15.0 in times  # untouched grid outside the span


def test_splice_emits_each_strike_once_across_spans():
    """Even given adjacent spans, no strike may produce two cuts at the same time."""
    grid = [(t, "ambient") for t in np.arange(0, 30, 1.0)]
    strikes = np.array([2.5, 7.5, 12.5, 17.5])
    spans = [(0.0, 10.0), (10.0, 20.0)]
    out = _splice_onset_anchor(
        grid, spans, strikes, lambda t: "ambient", has_grid=False,
    )
    times = [t for t, _ in out]
    assert len(times) == len(set(times)), f"duplicate cut times: {times}"
