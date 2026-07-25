"""Beat-lock guarantees for the occupancy pacer.

Two properties the owner cares about, both automatable:
  1. Every NON-intense cut lands on the FELT-downbeat parity (the "1 & 2"),
     never the in-between eighth-note — across tier boundaries. Intense is the
     deliberate exception: it cuts on every detected beat (see below).
  2. Within any one tier section, consecutive cuts are evenly spaced — a CONSTANT
     whole number of felt beats for the felt-locked tiers (no 1,2,1,2
     alternation), and every detected beat for intense.

Tested against _felt_locked_cut_indices (the pure scheduler behind occupancy
cuts), with synthetic per-beat tier masks so no audio is needed.
"""
import numpy as np
import pytest

from selfies_for_a_year.beats import _drop_opening_flash, _felt_locked_cut_indices


def _masks(n, regions):
    """regions: list of (start, end, tier). Returns intense/slow/ambient masks."""
    intense = np.zeros(n, bool)
    slow = np.zeros(n, bool)
    ambient = np.zeros(n, bool)
    for a, b, t in regions:
        {"intense": intense, "slow": slow, "ambient": ambient}.get(t, intense)[a:b] = (
            True if t in ("intense", "slow", "ambient") else False
        )
    return intense, slow, ambient


def _felt_index(k, parity):
    # index of beat k among the felt-parity beats (k has parity == parity)
    return (k - parity) // 2


@pytest.mark.parametrize("parity", [0, 1])
def test_all_cuts_on_felt_parity(parity):
    n = 240
    intense, slow, ambient = _masks(n, [(60, 120, "intense"), (160, 240, "slow")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, parity)
    assert idxs, "expected some cuts"
    # Intense cuts on every detected beat (both parities) by design; every other
    # tier must still land only on the felt parity, even across the boundaries.
    assert all(k % 2 == parity for k in idxs if not intense[k]), \
        "every non-intense cut must be on the felt parity"


@pytest.mark.parametrize("parity", [0, 1])
def test_constant_gap_within_each_tier(parity):
    n = 300
    # normal 0-90, intense 90-180, slow 180-300
    intense, slow, ambient = _masks(n, [(90, 180, "intense"), (180, 300, "slow")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, parity)

    def tier_of(k):
        if intense[k]:
            return "intense"
        if slow[k] or ambient[k]:
            return "slow"
        return "normal"

    # Group consecutive cuts that share a tier. Within a felt-locked run the
    # felt-beat gap must be a single constant value; an intense run cuts on every
    # detected beat, i.e. consecutive raw indices.
    ks = list(idxs)
    tiers = [tier_of(k) for k in ks]
    run_start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or tiers[i] != tiers[run_start]:
            run = ks[run_start:i]
            if tiers[run_start] == "intense":
                assert all(b - a == 1 for a, b in zip(run, run[1:])), (
                    f"intense must cut on every detected beat, got {run}"
                )
            else:
                fi = [_felt_index(k, parity) for k in run]
                gaps = {b - a for a, b in zip(fi, fi[1:])}
                assert len(gaps) <= 1, (
                    f"{tiers[run_start]} section has non-constant cut gaps {gaps}"
                )
            run_start = i


def test_denser_tier_has_smaller_gap():
    """intense should cut more often than normal, which cuts more often than slow."""
    n = 300
    intense, slow, ambient = _masks(n, [(90, 180, "intense"), (180, 300, "slow")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, 0)

    # Measure in RAW detected beats: intense cuts every beat, so its felt-index
    # gap would alternate 0,1 and mislead. In raw terms intense=1 < normal < slow.
    def gap_in(mask_lo, mask_hi):
        pts = [k for k in idxs if mask_lo <= k < mask_hi]
        d = np.diff(pts)
        return float(np.median(d)) if len(d) else None

    g_normal = gap_in(0, 90)
    g_intense = gap_in(90, 180)
    g_slow = gap_in(180, 300)
    assert g_intense < g_normal < g_slow, (g_intense, g_normal, g_slow)


def test_no_cuts_when_all_off_parity_region_is_short():
    """A tiny track still produces at least the opening cut."""
    n = 8
    intense = np.zeros(n, bool)
    slow = np.zeros(n, bool)
    ambient = np.zeros(n, bool)
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, 0)
    assert idxs[0] == 0


# --- salience quantization: gaps can't rotate the cut through the weak beats --- #


def _tier_gap(idxs, parity, lo, hi):
    pts = [_felt_index(k, parity) for k in idxs if lo <= k < hi]
    d = np.diff(pts)
    # gap is constant within a tier (proven elsewhere) -> the single value
    return int(d[0]) if len(d) else None


@pytest.mark.parametrize("parity", [0, 1])
def test_normal_cuts_every_felt_beat(parity):
    """Normal is a fixed 1 felt beat per photo (1:2 on the raw grid)."""
    n = 400
    intense, slow, ambient = _masks(n, [])  # all normal
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, parity)
    g = _tier_gap(idxs, parity, 0, n)
    assert g == 1, f"normal gap {g} must be 1 felt beat"


@pytest.mark.parametrize("parity", [0, 1])
def test_slow_cuts_every_second_felt_beat(parity):
    """Slow is a fixed 2 felt beats per photo (1:4 on the raw grid), landing on
    the same {1,3} strong subgroup of the bar every time."""
    n = 500
    intense, slow, ambient = _masks(n, [(0, n, "slow")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, parity)
    g = _tier_gap(idxs, parity, 0, n)
    assert g == 2, f"slow gap {g} must be 2 felt beats"


@pytest.mark.parametrize("parity", [0, 1])
def test_intense_cuts_on_every_detected_beat(parity):
    """The peak tier cuts on EVERY detected beat — parity AND off-parity — so it
    runs at double the every-felt-beat rate. This holds for every track (the old
    kick-detection gate that limited it to four-on-floor tracks is gone)."""
    n = 60
    intense, slow, ambient = _masks(n, [(20, 40, "intense")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, parity)
    intense_cuts = [k for k in idxs if 20 <= k < 40]
    # consecutive detected-beat indices -> gap of 1 in k, both parities present
    assert intense_cuts == list(range(20, 40)), intense_cuts


# --- opening flash suppression: photo 1 shouldn't flash out of the gate --- #


def test_opening_flash_dropped_when_lead_is_short():
    """A near-instant first cut (photo 1 flashes) is dropped so photo 1 rides."""
    # t=0 start, first beat lands at 0.09s, then a 6.9s ambient hold.
    tr = [(0.0, "ambient"), (0.09, "ambient"), (6.94, "ambient"), (13.8, "normal")]
    out = _drop_opening_flash(tr)
    assert out[0] == (0.0, "ambient")
    assert out[1] == (6.94, "ambient"), "the 0.09s flash cut should be gone"


def test_opening_flash_kept_when_lead_is_full():
    """A normal-length opening hold is left alone."""
    tr = [(0.0, "ambient"), (3.4, "normal"), (6.8, "normal"), (10.2, "normal")]
    out = _drop_opening_flash(tr)
    assert out == tr, "a full-length opening segment must not be coalesced"


def test_opening_flash_noop_when_first_cut_not_at_zero():
    """If the first transition isn't the t=0 start, don't touch it."""
    tr = [(0.5, "ambient"), (0.6, "ambient"), (6.9, "ambient")]
    out = _drop_opening_flash(tr)
    assert out == tr
