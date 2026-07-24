"""See WHERE cuts land relative to the beats — without rendering video.

A compact timeline (default first 45s) showing:
  * every detected beat as a tick; the FELT-downbeat parity (the "1 & 2",
    stronger onset) drawn tall+blue, the in-between eighth-notes short+gray
  * each image CUT as a marker: green =landed on the felt beat, red =landed on
    the off-beat (the "between taps" problem)
  * the tier band underneath (ambient/slow/normal/intense)

Title reports the % of cuts on the felt beat — the number we're trying to drive
to 100%. Run:
  uv run python experiments/beat_cut_map.py "<audio>" [base_pace] [window_s]
    base_pace: current | occupancy   (default occupancy)
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "src")
from selfies_for_a_year.beats import build_timeline, _detect_beats

TIER_RGB = {"ambient": (150, 150, 150), "slow": (90, 200, 110),
            "normal": (240, 210, 70), "intense": (255, 95, 95)}


def main(argv):
    path = Path(argv[0]).expanduser()
    base_pace = argv[1] if len(argv) > 1 else "occupancy"
    win = float(argv[2]) if len(argv) > 2 else 45.0
    out = f"/tmp/beat_cut_map_{path.stem[:12].replace(' ','_')}_{base_pace}.png"

    bt, strengths, loud, la, bpm, dur = _detect_beats(path)
    bt = np.asarray(bt); strengths = np.asarray(strengths)
    # felt-downbeat parity = the stronger-onset every-other beat
    par = 0 if strengths[0::2].mean() >= strengths[1::2].mean() else 1

    n_photos = 183
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(n_photos)]
    tl = build_timeline(path, dates, min_photos_per_beat=0.25, max_photos_per_second=4.0,
                        vary_pace=True, pace_model="segment", base_pace=base_pace,
                        intense_multiplier=3, slow_multiplier=0.33)
    # cut times + tier per cut
    t = 0.0; cuts = []; ctiers = []
    for s in tl.segments:
        cuts.append(t); ctiers.append(s.tier); t += s.duration
    cuts = np.array(cuts)

    # classify each cut: nearest beat, and is that beat the felt parity?
    idx = np.searchsorted(bt, cuts); idx = np.clip(idx, 1, len(bt) - 1)
    near = np.where(np.abs(cuts - bt[idx - 1]) < np.abs(cuts - bt[idx]), idx - 1, idx)
    on_felt = (near % 2 == par)
    inwin = cuts < win
    pct = 100 * np.mean(on_felt[cuts < dur]) if len(cuts) else 0

    fig, ax = plt.subplots(figsize=(16, 3.2))
    # beats
    for i, b in enumerate(bt):
        if b > win:
            break
        if i % 2 == par:
            ax.vlines(b, 0.45, 1.0, color="#3a6ea5", lw=1.3)      # felt beat
        else:
            ax.vlines(b, 0.55, 0.8, color="0.7", lw=0.7)          # off-beat
    # cuts
    for c, t_, ok in zip(cuts[inwin], np.array(ctiers)[inwin], on_felt[inwin]):
        ax.plot([c], [1.15], marker="v", ms=9,
                color=("#2ca02c" if ok else "#d62728"))
        ax.vlines(c, 0.0, 1.1, color=("#2ca02c" if ok else "#d62728"),
                  lw=1.0, alpha=0.5)
    # tier band
    t = 0.0
    for s in tl.segments:
        a, b = t, t + s.duration; t = b
        if a > win:
            break
        ax.add_patch(Rectangle((a, -0.3), min(b, win) - a, 0.28,
                     color=tuple(c / 255 for c in TIER_RGB[s.tier])))

    ax.set_xlim(0, win); ax.set_ylim(-0.35, 1.3); ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{path.stem} [{base_pace}] — blue=felt beat (the 1&2), gray=off-beat eighth; "
                 f"▼green=cut ON felt, ▼red=cut OFF felt   |   {pct:.0f}% of cuts on the felt beat")
    plt.tight_layout(); plt.savefig(out, dpi=95)
    print("wrote", out, f"({pct:.0f}% on felt beat)")


if __name__ == "__main__":
    main(sys.argv[1:])
