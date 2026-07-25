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


def instrumented_timeline(path: Path, n_photos: int = 1158):
    """Run the real pipeline, recording what each anchor span was handed."""
    calls: list[dict] = []
    orig = B._onset_anchor_cuts

    def spy(span, strikes, tier_at, ambient_default="slow"):
        out = orig(span, strikes, tier_at, ambient_default)
        calls.append({
            "span": span,
            "strikes": np.asarray(strikes, dtype=float),  # already coalesced
            "cuts": out,
        })
        return out

    B._onset_anchor_cuts = spy
    try:
        tl = B.build_timeline(
            path,
            [datetime(2020, 1, 1) + timedelta(days=i * 2) for i in range(n_photos)],
            max_photos_per_second=4.0, min_photos_per_beat=1.0, vary_pace=True,
            intense_multiplier=3.0, slow_multiplier=0.33,
        )
    finally:
        B._onset_anchor_cuts = orig
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
    tl, calls = instrumented_timeline(path)
    starts, durs, tiers, kinds, _ = classify(tl, calls)
    print(f"{path.stem}: {len(tl.segments)} photos / {tl.total_duration:.0f}s = "
          f"{len(tl.segments) / tl.total_duration:.2f}/s, "
          f"{len(calls)} anchor spans")
    report(starts, durs, tiers, kinds, calls, tl.total_duration)
    out = f"/tmp/burst_scope_{path.stem[:12].replace(' ', '_')}_{t0:.0f}s.png"
    plot(path, tl, calls, starts, durs, tiers, kinds, t0, t0 + win, out)


if __name__ == "__main__":
    main(sys.argv[1:])
