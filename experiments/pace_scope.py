"""See and score where the cuts land against the music that they claim to follow.

"The normal section doesn't feel good" is not something you can tune from a
photos-per-second average -- the average is identical whether cuts land on the
notes or scatter between them. This shows the three things that decide it:

  * WHETHER a tier means one pace. `normal` once ran 7.8 beats/photo on the beat
    grid and 0.6 inside onset-anchored spans, and the same label felt dead in one
    section and frantic in the next (issue #46). The beats/photo column is the
    check: the two regimes of a tier should read the same number.
  * WHERE each cut sits relative to the note strikes. Inside an anchor span a
    scheduled tick either found a real note within the snap tolerance or it
    didn't; the green/amber split is that, and it is what "connected to the
    music" actually measures.
  * WHETHER the pace wanders. p90/p10 of the holds inside one (tier, regime)
    cell -- a number near 1 means the tier keeps its promise, a big one means
    the viewer can't learn the rhythm.

The plot, top to bottom:
  * onset-strength envelope -- where the music actually hits
  * strike ticks: tall blue = a note a cut landed on, short grey = a note nothing
    landed on
  * cut markers: green = on a strike, amber = a snapped tick that found no note
    (so it sits on the beat grid), grey = grid cut outside any anchor span
  * anchor spans shaded, tier band underneath
  * hold durations as stems, with the legibility floor drawn

Everything is captured from the real build_timeline by wrapping
_onset_anchor_cuts -- nothing here re-derives strikes, spans or tiers, because a
reimplementation already lied to us once about which sections changed.

Run:
  uv run python experiments/pace_scope.py "<audio>" [start_s] [window_s]
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

    def spy(span, strikes, tier_at, ambient_default="slow", **kw):
        out = orig(span, strikes, tier_at, ambient_default, **kw)
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
    """Label every cut: on a strike, on a snapped-but-empty beat, or grid."""
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
            kinds.append("beat")
    return starts, durs, tiers, kinds, all_strikes


def report(starts, durs, tiers, kinds, calls, bpm):
    span_of = lambda t: any(a <= t < b for c in calls for a, b in [c["span"]])  # noqa: E731
    print(f"\n{'tier':<9} {'regime':<9} {'cuts':>5} {'/sec':>6} {'median':>8} "
          f"{'hold spread':>12} {'beats/photo':>12}   on-strike")
    print("-" * 82)
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
              f"{np.median(h) * 1000:>7.0f}ms {spread:>11.1f}x "
              f"{np.median(h) * bpm / 60.0:>12.1f}   "
              f"{100 * n_on / len(h):>3.0f}%")

    # The headline check: a tier's two regimes should state the same rate.
    by_tier = {}
    for (tier, regime), v in rows.items():
        by_tier.setdefault(tier, {})[regime] = np.median(v["holds"]) * bpm / 60.0
    worst = [(t, r["grid"] / r["anchored"]) for t, r in by_tier.items()
             if r.get("grid") and r.get("anchored")]
    for tier, ratio in sorted(worst, key=lambda x: -max(x[1], 1 / x[1])):
        flag = "  <-- the two regimes disagree" if max(ratio, 1 / ratio) > 1.5 else ""
        print(f"\n{tier}: grid / anchored = {max(ratio, 1/ratio):.1f}x{flag}")


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

    landed = {round(t, 6) for c in calls for t, _ in c["cuts"]}
    for c in calls:
        for s in c["strikes"]:
            if t0 <= s <= t1:
                hit = round(float(s), 6) in landed
                ax.axvline(s, ymin=0, ymax=0.55 if hit else 0.22,
                           color="#2c6fd1" if hit else "#bbb",
                           lw=1.4 if hit else 0.8, zorder=2)

    colors = {"strike": "#1a9850", "beat": "#f0a500", "grid": "#888"}
    for t, d, kind in zip(starts, durs, kinds):
        if t0 <= t <= t1:
            ax.plot([t], [1.12], marker="v", ms=7, color=colors[kind], zorder=3)

    ax.set_ylim(0, 1.3)
    ax.set_ylabel("onset strength")
    ax.set_yticks([])
    ax.set_title(
        f"{path.stem[:40]}  {t0:.0f}-{t1:.0f}s   "
        f"blue tick = note a cut landed on, grey tick = note nothing landed on   "
        f"▼ green = cut on strike, amber = snapped tick with no note, grey = grid",
        fontsize=9)

    floor_ms = B._MIN_HOLD_S * 1000
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
    print(f"{path.stem}: {len(tl.segments)} photos / "
          f"{tl.total_duration:.0f}s = {len(tl.segments) / tl.total_duration:.2f}/s, "
          f"{len(calls)} anchor spans, {tl.bpm:.1f} BPM")
    report(starts, durs, tiers, kinds, calls, tl.bpm)
    out = f"/tmp/pace_scope_{path.stem[:12].replace(' ', '_')}_{t0:.0f}s.png"
    plot(path, tl, calls, starts, durs, tiers, kinds, t0, t0 + win, out)


if __name__ == "__main__":
    main(sys.argv[1:])
