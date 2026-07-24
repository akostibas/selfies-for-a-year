"""Beat-lock guarantees for the occupancy pacer.

Two properties the owner cares about, both automatable:
  1. Every cut lands on the FELT-downbeat parity (the "1 & 2"), never the
     in-between eighth-note — across tier boundaries.
  2. Within any one tier section, consecutive cuts are a CONSTANT whole number
     of felt beats apart — no 1,2,1,2 alternation.

Tested against _felt_locked_cut_indices (the pure scheduler behind occupancy
cuts), with synthetic per-beat tier masks so no audio is needed.
"""
import numpy as np
import pytest

from selfies_for_a_year.beats import _felt_locked_cut_indices


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
@pytest.mark.parametrize("sub", [1 / 6, 1 / 8, 1 / 4])
def test_all_cuts_on_felt_parity(parity, sub):
    n = 240
    intense, slow, ambient = _masks(n, [(60, 120, "intense"), (160, 240, "slow")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, sub, 3.0, 0.33, parity)
    assert idxs, "expected some cuts"
    assert all(k % 2 == parity for k in idxs), "every cut must be on the felt parity"


@pytest.mark.parametrize("parity", [0, 1])
def test_constant_gap_within_each_tier(parity):
    n = 300
    # normal 0-90, intense 90-180, slow 180-300
    intense, slow, ambient = _masks(n, [(90, 180, "intense"), (180, 300, "slow")])
    sub = 1 / 6
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, sub, 3.0, 0.33, parity)

    def tier_of(k):
        if intense[k]:
            return "intense"
        if slow[k] or ambient[k]:
            return "slow"
        return "normal"

    # Group consecutive cuts that share a tier; within each such run the felt-beat
    # gap must be a single constant value.
    fi = [_felt_index(k, parity) for k in idxs]
    tiers = [tier_of(k) for k in idxs]
    run_start = 0
    for i in range(1, len(idxs) + 1):
        if i == len(idxs) or tiers[i] != tiers[run_start]:
            gaps = {fi[j] - fi[j - 1] for j in range(run_start + 1, i)}
            assert len(gaps) <= 1, (
                f"{tiers[run_start]} section has non-constant cut gaps {gaps}"
            )
            run_start = i


def test_denser_tier_has_smaller_gap():
    """intense should cut more often than normal, which cuts more often than slow."""
    n = 300
    intense, slow, ambient = _masks(n, [(90, 180, "intense"), (180, 300, "slow")])
    sub = 1 / 6
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, sub, 3.0, 0.33, 0)

    def gap_in(mask_lo, mask_hi):
        pts = [(_felt_index(k, 0)) for k in idxs if mask_lo <= k < mask_hi]
        d = np.diff(pts)
        return int(np.median(d)) if len(d) else None

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
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, 1 / 6, 3.0, 0.33, 0)
    assert idxs[0] == 0


# --- salience quantization: gaps can't rotate the cut through the weak beats --- #


def _tier_gap(idxs, parity, lo, hi):
    pts = [_felt_index(k, parity) for k in idxs if lo <= k < hi]
    d = np.diff(pts)
    # gap is constant within a tier (proven elsewhere) -> the single value
    return int(d[0]) if len(d) else None


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("sub", [1 / 4, 1 / 6, 1 / 8])
def test_normal_gap_is_even(parity, sub):
    """The normal-tier felt-beat gap must be even, so cuts stay on the {1,3}
    strong/medium subgroup of a 4-beat bar instead of walking onto 2 & 4."""
    n = 400
    intense, slow, ambient = _masks(n, [])  # all normal
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, sub, 3.0, 0.33, parity)
    g = _tier_gap(idxs, parity, 0, n)
    assert g is not None and g % 2 == 0, f"normal gap {g} must be even"


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("bar", [3, 4])
def test_slow_gap_is_bar_multiple(parity, bar):
    """The slow-tier felt-beat gap must be a whole number of bars, so every cut
    lands on the same bar position (one clean cut per N bars)."""
    n = 500
    intense, slow, ambient = _masks(n, [(0, n, "slow")])
    idxs = _felt_locked_cut_indices(
        n, intense, slow, ambient, 1 / 6, 3.0, 0.33, parity, bar_felt_beats=bar
    )
    g = _tier_gap(idxs, parity, 0, n)
    assert g is not None and g % bar == 0, f"slow gap {g} must be a multiple of the bar {bar}"


def test_intense_cuts_every_felt_beat():
    """Intense tier cuts on every felt beat — dense weak landings are the point."""
    n = 200
    intense, slow, ambient = _masks(n, [(0, n, "intense")])
    idxs = _felt_locked_cut_indices(n, intense, slow, ambient, 1 / 6, 3.0, 0.33, 0)
    g = _tier_gap(idxs, 0, 0, n)
    assert g == 1, f"intense gap {g} must be 1 (every felt beat)"
