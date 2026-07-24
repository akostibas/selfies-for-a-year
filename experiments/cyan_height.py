"""Turn "cyan height" (the upper envelope of the spectral centroid — how HIGH
the brightness reaches, moment to moment) into its own time-series graph, so we
can reason about it and set tier thresholds on it.

Panels (time-aligned):
  1. mel spectrogram + centroid (cyan)
  2. candidate HEIGHT signals: rolling-max envelope vs rolling-p90 envelope,
     normalized — plus the product-owner's hand-drawn 8-box target as shaded
     bands (red=intense, yellow=normal, blue=low, gray=ambient) so we can see
     which envelope best separates the tiers and where the thresholds sit.

Run:
  uv run python experiments/cyan_height.py "<audio>" [out.png]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d, percentile_filter, uniform_filter1d

HOP = 512

# product-owner's hand-drawn target for Push Upstairs (approx box boundaries, s)
TARGET = [
    (0, 28, "intense"), (28, 65, "normal"), (65, 155, "low"), (155, 172, "ambient"),
    (172, 202, "intense"), (202, 235, "low"), (235, 265, "normal"), (265, 274, "ambient"),
]
TIER_RGB = {"intense": (255, 95, 95), "normal": (240, 210, 70),
            "low": (90, 140, 230), "ambient": (150, 150, 150)}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    import librosa

    path = Path(argv[0]).expanduser()
    out = argv[1] if len(argv) > 1 else "/tmp/cyan_height.png"
    y, sr = librosa.load(str(path), mono=True)
    dur = len(y) / sr
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=HOP)

    w = int(2.0 * sr / HOP)
    env_max = maximum_filter1d(cent, w)                       # peak reach
    env_p90 = percentile_filter(cent, 90, size=w)             # robust reach
    env_p90s = uniform_filter1d(env_p90, w // 2)              # + light smoothing

    def norm(a):
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP, x_axis="time",
                             y_axis="mel", ax=ax1, cmap="magma")
    ax1.plot(ct, cent, color="cyan", lw=0.8, alpha=0.7)
    ax1.set_title(f"{path.stem} — 'cyan height' (centroid upper envelope) as a signal")

    # target bands behind the height signals
    for a, b, tier in TARGET:
        ax2.axvspan(a, b, color=tuple(c / 255 for c in TIER_RGB[tier]), alpha=0.30)
    ax2.plot(ct, norm(env_max), color="black", lw=0.7, alpha=0.4, label="rolling max (peak reach)")
    ax2.plot(ct, norm(env_p90s), color="black", lw=2.0, label="rolling p90 (robust reach) — candidate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("cyan height (norm)")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, dur)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.text(0.5, -0.28, "shaded = owner's target: red=intense yellow=normal blue=low gray=ambient",
             transform=ax2.transAxes, ha="center", fontsize=8, color="0.3")

    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
