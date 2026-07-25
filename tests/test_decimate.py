"""Choosing which photos survive when there are more photos than cuts.

The one property that matters: the video still travels the whole span. A
timelapse that stops in 2020 because the arithmetic ran out is worse than one
that skips months evenly (#44).
"""
from datetime import datetime, timedelta

from selfies_for_a_year.beats import _weighted_decimate_by_month


def _dates(spec):
    """spec: list of (year, month, count) -> a flat, date-ordered photo list."""
    return [datetime(y, m, 1) + timedelta(minutes=i)
            for y, m, n in spec for i in range(n)]


def test_more_months_than_cuts_still_reaches_the_end():
    """The #44 regression: 44 years of one-photo months, 200 cuts.

    Every month used to floor to 1, the over-allocation trim had nothing left to
    take, and the caller kept the first 200 in date order — so the render was
    1982 to 2000 and then stopped.
    """
    dates = _dates([(y, m, 1) for y in range(1982, 2026) for m in range(1, 13)])
    target = 200
    sel, log = _weighted_decimate_by_month(list(range(len(dates))), dates, target)
    assert len(sel) == target, len(sel)
    assert dates[sel[0]].year == 1982
    assert dates[sel[-1]].year == 2025
    # ...and the months it kept are spread, not clustered at either end.
    halfway = len([i for i in sel if dates[i].year < 2004])
    assert abs(halfway - target / 2) <= target * 0.1, halfway


def test_never_returns_more_than_target():
    """The caller head-slices the overflow, so an over-allocation is a silent
    truncation of the recent past — never let one out of here."""
    for target in (1, 5, 37, 199, 500):
        dates = _dates([(2020 + y, m, 3) for y in range(6) for m in range(1, 13)])
        sel, _ = _weighted_decimate_by_month(list(range(len(dates))), dates, target)
        assert len(sel) == min(target, len(dates)), (target, len(sel))


def test_busy_months_get_more_slots_when_there_is_room():
    """With slots to spare the allocation stays proportional — a month with ten
    times the photos should not read as one moment."""
    dates = _dates([(2024, 1, 100), (2024, 2, 10), (2024, 3, 10)])
    sel, log = _weighted_decimate_by_month(list(range(len(dates))), dates, 60)
    kept = {label: k for label, k, _ in log}
    assert kept["2024-01"] > kept["2024-02"] * 3, kept
    assert all(v >= 1 for v in kept.values()), kept


def test_selection_stays_in_date_order():
    dates = _dates([(2024, m, 5) for m in range(1, 13)])
    sel, _ = _weighted_decimate_by_month(list(range(len(dates))), dates, 20)
    assert sel == sorted(sel)
