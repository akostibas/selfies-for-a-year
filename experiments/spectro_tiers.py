"""Gut-check tool: overlay Recipe A's pacing tiers on the mel spectrogram.

Three stacked, time-aligned panels:
  1. mel spectrogram + spectral-centroid (brightness) line  — the ground truth
  2. the per-beat ENERGY signal Recipe A thresholds          — what it "sees"
  3. the resulting tier band (ambient/slow/normal/intense)   — what it decided

If tier 3 lines up with the structure visible in panel 1 (intense on the
bright/busy passages, ambient in the breakdowns), the pacing is sound — no
render needed. Colors match the video overlay.

Run:
  uv run python experiments/spectro_tiers.py "<audio>" [out.png]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pacing_recipes import HOP, _energy_signal, _recipe_c_boundaries, _recipe_c_features, load_features, recipe_c

# match _TIER_COLORS in cli.py (0-255 -> 0-1)
TIER_RGB = {
    "ambient": (150, 150, 150),
    "slow": (90, 200, 110),
    "normal": (240, 210, 70),
    "intense": (255, 95, 95),
}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0]).expanduser()
    out = argv[1] if len(argv) > 1 else "/tmp/spectro_tiers.png"

    import librosa

    f = load_features(path)
    intervals = recipe_c(f)
    energy = _energy_signal(f)
    # segmentation boundaries (beat indices -> times) to draw as vertical cuts
    hr, lr, mf, md = _recipe_c_features(f)
    seg_bounds = [float(f.beat_times[b]) for b in _recipe_c_boundaries(hr, lr, mf, md) if b < len(f.beat_times)]

    S = librosa.feature.melspectrogram(y=f.y, sr=f.sr, hop_length=HOP, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    cent = librosa.feature.spectral_centroid(y=f.y, sr=f.sr, hop_length=HOP)[0]
    dur = f.duration

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [4, 1.4, 0.8]},
    )
    librosa.display.specshow(S_db, sr=f.sr, hop_length=HOP, x_axis="time",
                             y_axis="mel", ax=ax1, cmap="magma")
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=f.sr, hop_length=HOP)
    ax1.plot(ct, cent, color="cyan", lw=1.0, alpha=0.8, label="spectral centroid")
    for sb in seg_bounds:
        ax1.axvline(sb, color="white", lw=0.8, ls="--", alpha=0.6)
    ax1.legend(loc="upper right")
    ax1.set_title(f"{path.stem} — Recipe C: segment cuts (white dashed) + tiers   ({dur:.0f}s, {f.bpm:.0f} BPM)")

    # panel 2: energy signal + tier center gridlines
    ax2.plot(f.beat_times, energy, color="white", lw=1.2)
    ax2.set_facecolor("0.15")
    ax2.set_ylabel("energy")
    ax2.set_ylim(-0.02, 1.02)
    for p in (7.5, 27.5, 62.5, 92.5):
        ax2.axhline(np.percentile(energy, p), color="0.5", lw=0.5, ls=":")

    # panel 3: tier band
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    for a, b, t in intervals:
        rgb = tuple(c / 255 for c in TIER_RGB[t])
        ax3.add_patch(Rectangle((a, 0), b - a, 1, color=rgb))
        if b - a > dur * 0.03:
            ax3.text((a + b) / 2, 0.5, t[:4], ha="center", va="center",
                     fontsize=7, color="black")
    ax3.set_xlim(0, dur)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("tier")

    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
