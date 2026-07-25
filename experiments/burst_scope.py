"""See and score the per-strike burst against the music that triggered it.

"The normal section doesn't feel good" is not something you can tune from a
photos-per-second average -- the average is identical whether cuts land on the
notes or scatter between them. This shows the two things that decide it:

  * WHERE each cut sits relative to the note strikes (the burst's first photo is
    anchored on a strike; every decay step after it floats free, and if the
    floating ones outnumber the anchored ones the section stops reading as
    matched to the music)
  * WHETHER the shape is ever visible (a burst damped after one photo is not a
    decay the viewer can learn -- it is just a cut, and a section made only of
    damped bursts has a shape on paper and none on screen)

The plot, top to bottom:
  * onset-strength envelope -- where the music actually hits
  * strike ticks: tall blue = a trigger (fires a burst), short grey = skipped by
    the tier's stride
  * cut markers: green = on a strike, amber = a decay step, grey = grid cut
    outside any anchor span
  * anchor spans shaded, tier band underneath
  * hold durations as stems, with the legibility floor drawn

Everything is captured from the real build_timeline by wrapping
_onset_anchor_cuts -- nothing here re-derives strikes, spans or tiers, because a
reimplementation already lied to us once about which sections changed.

Run:
  uv run python experiments/burst_scope.py "<audio>" [start_s] [window_s]
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "src")
from selfies_for_a_year import beats as B  # noqa: E402

TIER_RGB = {"ambient": (150, 150, 150), "slow": (90, 200, 110),
            "normal": (240, 210, 70), "intense": (255, 95, 95)}
ON_STRIKE_EPS = 1e-6


def interval_burst(span, strikes, tier_at, ambient_default="slow"):
    """Candidate: scale the burst to the gap it has to fill, instead of fixed ms.

    The shipped burst uses absolute gaps (150/240/384ms), so it front-loads a
    flurry and then waits out however much of the note is left -- and the wait
    has no relationship to anything. Here the same geometric decay is stretched
    so its gaps SUM to the interval before the next trigger: the last and
    longest photo resolves exactly as the next note lands.

    Consequences worth judging on the plot:
      * the shape is self-similar -- every note gets the same proportional fan,
        so a slow passage decays slowly and a busy one decays fast, from one rule
      * long gaps stop being dead air; they get the same `depth` photos, just
        spread wide, which is what should fix the 24-34s hole
      * cuts sit at fixed PHASE positions within each note (0, ~19%, ~50%), so
        they stay tied to the music even when they aren't on the attack itself
    """
    t0, t1 = span
    sel = [float(s) for s in np.asarray(strikes) if t0 - 1e-6 <= s < t1 + 1e-6]
    out: list[tuple[float, str]] = []
    i = 0
    r = B._BURST_RATIO
    while i < len(sel):
        tier = tier_at(sel[i]) or ambient_default
        stride, depth = B._BURST_TABLE.get(tier, (2, 1))
        stride = max(1, stride)
        nxt = sel[i + stride] if i + stride < len(sel) else t1
        interval = nxt - sel[i]
        # gaps g0*r^k for k<depth summing to `interval`
        g0 = interval * (r - 1) / (r**depth - 1) if depth > 1 else interval
        c = sel[i]
        for k in range(depth):
            gap = g0 * r**k
            if gap < B._BURST_FLOOR_S - 1e-9:
                break  # too brief to register -- hold the previous photo instead
            out.append((c, tier))
            c += gap
        i += stride
    return out


def one_per_strike(span, strikes, tier_at, ambient_default="slow"):
    """Candidate: no burst at all in the busy tiers -- one photo per strike.

    The other half of what was asked for: "a well timed constant cut rate". Every
    cut is on an attack by construction, which is the one property the two
    sections rated 5/5 share (92% and 100% on-strike, against 36% in the section
    that was rejected). Pace still follows the playing -- a busy passage cuts
    fast -- but nothing floats between the notes.

    Cheaper than it sounds: normal currently spends ~2.8 photos per strike, so
    dropping to 1 costs less pace than the depth suggests, because those extra
    photos are crammed into the same interval rather than covering new ground.
    """
    t0, t1 = span
    sel = [float(s) for s in np.asarray(strikes) if t0 - 1e-6 <= s < t1 + 1e-6]
    out: list[tuple[float, str]] = []
    i = 0
    while i < len(sel):
        tier = tier_at(sel[i]) or ambient_default
        # ambient/slow keep their sparse stride; the busy tiers cut every strike
        stride = {"ambient": 2, "slow": 4}.get(tier, 1)
        out.append((sel[i], tier))
        i += stride
    return out


BEAT_TIMES: dict = {}  # filled before the run so snap_to_strike can see the grid
# Beats per photo by tier. The pace is stated ONCE, as a musical rate, and holds
# everywhere -- which is the actual bug the other candidates dance around:
# `normal` currently runs 7.8 beats/photo on the grid and 0.6 beats/photo inside
# anchor spans, a 13x swing within a single tier.
BEATS_PER_PHOTO = {"intense": 1.0, "normal": 2.0, "slow": 6.0, "ambient": 8.0}
SNAP_TOLERANCE_S = 0.12   # a tick this close to a real strike cuts on the strike


def snap_to_strike(span, strikes, tier_at, ambient_default="slow"):
    """Candidate: a constant musical rate, landing on real notes where they exist.

    Schedule cuts at the tier's beat rate, then pull each one onto the nearest
    strike within SNAP_TOLERANCE_S. Where the pianist played near the tick you
    cut on the note; where they didn't, you cut on the beat. Nothing is invented
    between the notes and nothing waits 8 beats for the next one -- which is the
    bind: long spacing is boring, but photos that aren't on anything don't read
    as connected.

    ambient and slow stay strike-driven (they are rated 5/5 and their whole
    character is lingering on a chord), so this only re-paces normal/intense.
    """
    t0, t1 = span
    sel = [float(s) for s in np.asarray(strikes) if t0 - 1e-6 <= s < t1 + 1e-6]
    if not sel:
        return []
    tier = tier_at(sel[0]) or ambient_default
    if tier in ("ambient", "slow"):
        return one_per_strike(span, strikes, tier_at, ambient_default)

    beats = BEAT_TIMES.get("t")
    if beats is None or not len(beats):
        return one_per_strike(span, strikes, tier_at, ambient_default)
    out: list[tuple[float, str]] = []
    used: set[float] = set()
    in_span = beats[(beats >= t0) & (beats < t1)]
    step = BEATS_PER_PHOTO.get(tier, 2.0)
    k = 0.0
    while int(k) < len(in_span):
        tick = float(in_span[int(k)])
        tier_here = tier_at(tick) or tier
        cand = [s for s in sel if abs(s - tick) <= SNAP_TOLERANCE_S and s not in used]
        t = min(cand, key=lambda s: abs(s - tick)) if cand else tick
        used.add(t)
        if not out or t - out[-1][0] >= B._BURST_FLOOR_S:
            out.append((t, tier_here))
        k += BEATS_PER_PHOTO.get(tier_here, step)
    return out


MODES = {"fixed": None, "interval": interval_burst, "onstrike": one_per_strike,
         "snap": snap_to_strike, "snaponly": snap_to_strike}

# The way out of the bind "long spacing is boring, but photos between the notes
# don't feel connected": cut on MORE OF THE REAL NOTES. Every cut stays on an
# attack (100% on-strike, like the sections rated 5/5) and the pace rises,
# because a solo piano plays far more notes than we currently detect -- most of
# the peaks in the onset envelope never get a strike. These loosen the two knobs
# that throw notes away: the prominence threshold (an ornament or a left-hand
# note scores below 30% of the local p90) and the coalescing window (anything
# inside 400ms of a louder neighbour is merged into it).
SENSITIVITY = {           # name: (prom_frac, coalesce_s)
    "dense": (0.15, 0.25),
    "denser": (0.08, 0.18),
}


def instrumented_timeline(path: Path, n_photos: int = 1158, mode: str = "fixed"):
    """Run the real pipeline, recording what each anchor span was handed.

    A `mode` naming a SENSITIVITY preset also re-tunes strike detection, then
    cuts one photo per strike -- the pace comes from finding more notes, not
    from adding photos between them.
    """
    calls: list[dict] = []
    orig = B._onset_anchor_cuts
    orig_prom, orig_coal = B._prominent_strikes, B._coalesce_strikes
    impl = MODES.get(mode) or (one_per_strike if mode in SENSITIVITY else orig)

    if mode in SENSITIVITY:
        prom_frac, coalesce_s = SENSITIVITY[mode]

        def _prom(y, sr, *, prom_frac=prom_frac, rolling_s=4.0):
            return orig_prom(y, sr, prom_frac=prom_frac, rolling_s=rolling_s)

        def _coal(times, heights, min_gap=coalesce_s):
            return orig_coal(times, heights, min_gap=coalesce_s)

        B._prominent_strikes, B._coalesce_strikes = _prom, _coal

    def spy(span, strikes, tier_at, ambient_default="slow"):
        out = impl(span, strikes, tier_at, ambient_default)
        calls.append({
            "span": span,
            "strikes": np.asarray(strikes, dtype=float),  # already coalesced
            "cuts": out,
        })
        return out

    # snap states the pace as a beat rate, so the GRID half has to be told the
    # same rate -- otherwise it stays at 7.8 beats/photo and the 13x swing
    # survives in the half of the track this function never touches.
    extra = {}
    if mode in ("snap", "snaponly"):
        BEAT_TIMES["t"] = np.asarray(B._detect_beats(path)[0], dtype=float)
    if mode == "snap":
        # NOTE: forcing beat_speed also switches onset-anchoring off entirely
        # (0 spans), so ambient loses its piano-attack cutting. "snaponly"
        # isolates the snapping without that side effect.
        extra["beat_speed"] = 1.0 / BEATS_PER_PHOTO["normal"]

    B._onset_anchor_cuts = spy
    try:
        tl = B.build_timeline(
            path,
            [datetime(2020, 1, 1) + timedelta(days=i * 2) for i in range(n_photos)],
            max_photos_per_second=4.0, min_photos_per_beat=1.0, vary_pace=True,
            intense_multiplier=3.0, slow_multiplier=0.33, **extra,
        )
    finally:
        B._onset_anchor_cuts = orig
        B._prominent_strikes, B._coalesce_strikes = orig_prom, orig_coal
    return tl, calls


def classify(tl, calls):
    """Label every cut: on-strike, decay step, or grid."""
    durs = np.array([s.duration for s in tl.segments])
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    tiers = [s.tier for s in tl.segments]

    all_strikes = np.concatenate([c["strikes"] for c in calls]) if calls else np.array([])
    anchored = {round(t, 6) for c in calls for t, _ in c["cuts"]}

    kinds = []
    for t in starts:
        if round(float(t), 6) not in anchored:
            kinds.append("grid")
        elif len(all_strikes) and np.min(np.abs(all_strikes - t)) <= ON_STRIKE_EPS:
            kinds.append("strike")
        else:
            kinds.append("decay")
    return starts, durs, tiers, kinds, all_strikes


def burst_depths(calls):
    """How many photos each trigger actually got before something damped it."""
    depths = {}
    for c in calls:
        cuts = c["cuts"]
        strikes = set(np.round(c["strikes"], 6))
        cur_tier, n = None, 0
        for t, tier in cuts:
            if round(t, 6) in strikes:          # a new trigger starts here
                if cur_tier is not None:
                    depths.setdefault(cur_tier, []).append(n)
                cur_tier, n = tier, 1
            else:
                n += 1
        if cur_tier is not None:
            depths.setdefault(cur_tier, []).append(n)
    return depths


def report(starts, durs, tiers, kinds, calls, total):
    span_of = lambda t: any(a <= t < b for c in calls for a, b in [c["span"]])  # noqa: E731
    print(f"\n{'tier':<9} {'regime':<9} {'cuts':>5} {'/sec':>6} {'median':>8} "
          f"{'hold spread':>12}   on-strike")
    print("-" * 68)
    rows = {}
    for t, d, tier, kind in zip(starts, durs, tiers, kinds):
        key = (tier, "anchored" if span_of(t) else "grid")
        rows.setdefault(key, {"holds": [], "kinds": []})
        rows[key]["holds"].append(d)
        rows[key]["kinds"].append(kind)
    for (tier, regime), v in sorted(rows.items()):
        h = np.array(v["holds"])
        n_on = sum(1 for k in v["kinds"] if k == "strike")
        secs = h.sum()
        # p90/p10 says "does the pace wander", which median alone hides
        spread = np.percentile(h, 90) / max(np.percentile(h, 10), 1e-9)
        print(f"{tier:<9} {regime:<9} {len(h):>5} {len(h) / max(secs, 1e-9):>6.2f} "
              f"{np.median(h) * 1000:>7.0f}ms {spread:>11.1f}x   "
              f"{100 * n_on / len(h):>3.0f}%")

    print(f"\nrealized burst depth (how far the decay got before being damped):")
    for tier, ds in sorted(burst_depths(calls).items()):
        ds = np.array(ds)
        hist = "  ".join(f"{k}:{int((ds == k).sum())}" for k in range(1, ds.max() + 1))
        stride, depth = B._BURST_TABLE.get(tier, (2, 1))
        print(f"  {tier:<9} table=(stride {stride}, depth {depth})  "
              f"mean {ds.mean():.2f}   {hist}")
        if depth > 1 and (ds == 1).mean() > 0.5:
            print(f"  {'':<9} ^ {100 * (ds == 1).mean():.0f}% damped to a single photo "
                  f"-- the decay shape never renders here")


def plot(path, tl, calls, starts, durs, tiers, kinds, t0, t1, out):
    import librosa
    y, sr = librosa.load(str(path), mono=True)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    et = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    m = (et >= t0) & (et <= t1)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(16, 6.5), height_ratios=[3, 1], sharex=True)

    for c in calls:  # anchor spans
        a, b = c["span"]
        if b > t0 and a < t1:
            ax.axvspan(max(a, t0), min(b, t1), color="#e8f0ff", zorder=0)

    ax.plot(et[m], env[m] / max(env[m].max(), 1e-9), color="#999", lw=0.8, zorder=1)

    trig = {round(t, 6) for c in calls for t, _ in c["cuts"]}
    for c in calls:
        for s in c["strikes"]:
            if t0 <= s <= t1:
                is_trig = round(float(s), 6) in trig
                ax.axvline(s, ymin=0, ymax=0.55 if is_trig else 0.22,
                           color="#2c6fd1" if is_trig else "#bbb",
                           lw=1.4 if is_trig else 0.8, zorder=2)

    colors = {"strike": "#1a9850", "decay": "#f0a500", "grid": "#888"}
    for t, d, kind in zip(starts, durs, kinds):
        if t0 <= t <= t1:
            ax.plot([t], [1.12], marker="v", ms=7, color=colors[kind], zorder=3)

    ax.set_ylim(0, 1.3)
    ax.set_ylabel("onset strength")
    ax.set_yticks([])
    ax.set_title(
        f"{path.stem[:40]}  {t0:.0f}-{t1:.0f}s   "
        f"blue tick = trigger, grey tick = skipped strike   "
        f"▼ green = cut on strike, amber = decay step, grey = grid cut",
        fontsize=9)

    floor_ms = B._BURST_FLOOR_S * 1000
    for t, d, kind in zip(starts, durs, kinds):
        if t0 <= t <= t1:
            ax2.plot([t, t], [0, d * 1000], color=colors[kind], lw=1.6)
    ax2.axhline(floor_ms, color="red", ls=":", lw=1,
                label=f"legibility floor {floor_ms:.0f}ms")
    ax2.set_yscale("log")
    ax2.set_ylabel("hold (ms)")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8, loc="upper right")

    for t, d, tier in zip(starts, durs, tiers):  # tier band
        if t + d > t0 and t < t1:
            ax2.add_patch(Rectangle(
                (max(t, t0), 0.0), min(t + d, t1) - max(t, t0), 0.05,
                transform=ax2.get_xaxis_transform(), clip_on=True,
                color=np.array(TIER_RGB[tier]) / 255.0, zorder=0))

    ax.set_xlim(t0, t1)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


def main(argv):
    path = Path(argv[0]).expanduser()
    t0 = float(argv[1]) if len(argv) > 1 else 21.0
    win = float(argv[2]) if len(argv) > 2 else 25.0
    mode = argv[3] if len(argv) > 3 else "fixed"
    tl, calls = instrumented_timeline(path, mode=mode)
    starts, durs, tiers, kinds, _ = classify(tl, calls)
    print(f"[{mode}] {path.stem}: {len(tl.segments)} photos / "
          f"{tl.total_duration:.0f}s = {len(tl.segments) / tl.total_duration:.2f}/s, "
          f"{len(calls)} anchor spans")
    report(starts, durs, tiers, kinds, calls, tl.total_duration)
    out = (f"/tmp/burst_scope_{path.stem[:12].replace(' ', '_')}"
           f"_{mode}_{t0:.0f}s.png")
    plot(path, tl, calls, starts, durs, tiers, kinds, t0, t0 + win, out)


if __name__ == "__main__":
    main(sys.argv[1:])
