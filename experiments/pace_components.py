"""Show EVERY component the pacing algorithm measures, each as its own line
under the spectrogram — so you can compare each one directly against the audio
and the picture, the way you read the 'energy' panel.

Panels (all share the time axis; faint vertical lines = segment cuts):
  1. mel spectrogram + cyan centroid            — the audio, for reference
  2. BRIGHTNESS HEIGHT  — how HIGH the cyan reaches (p90 of the centroid)
  3. MOVEMENT           — how much the cyan JUMPS around (gated/churning);
                          rolling spread of the height over ~16 beats
  4. LOUDNESS           — how loud (RMS dB); faded when the track's dynamics
                          are too flat for loudness to matter (weight w shown)
  5. INTENSITY SCORE    — the single number the tiers threshold: the three
                          above combined; dotted lines = the tier cutoffs;
                          thick step = each section's p80 score
  6. TIER               — the decision (ambient/slow/normal/intense)

Every signal is normalized 0..1 so they're visually comparable. Run:
  uv run python experiments/pace_components.py "<audio>" [out.png]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "experiments")
from pacing_recipes import (
    HOP, TIERS, load_features, _robust_norm,
    _recipe_c_features, _recipe_c_energy, _recipe_c_boundaries, _mode_anchored_centers,
)

TIER_RGB = {"ambient": (150, 150, 150), "slow": (90, 200, 110),
            "normal": (240, 210, 70), "intense": (255, 95, 95)}
C_HEIGHT, C_MOVE, C_LOUD, C_SCORE = "#2ec4d6", "#b06cd0", "#e8a13a", "#ffffff"


def main(argv):
    import librosa
    path = Path(argv[0]).expanduser()
    out = argv[1] if len(argv) > 1 else "/tmp/pace_components.png"
    f = load_features(path)
    bt, dur = f.beat_times, f.duration

    hr, lr, mf, md = _recipe_c_features(f)
    height_n = _robust_norm(hr)
    move_n = _robust_norm(md)
    loud_n = _robust_norm(f.loudness_db)
    iqr_db = float(np.percentile(f.loudness_db, 75) - np.percentile(f.loudness_db, 25))
    w = float(np.clip((iqr_db - 3.0) / 6.0, 0.0, 1.0))
    energy = _recipe_c_energy(f, hr)
    combined = energy + 0.6 * move_n
    centers, _ = _mode_anchored_centers(combined)
    bounds = _recipe_c_boundaries(hr, lr, mf, md)
    seg_edges = list(zip(bounds, bounds[1:] + [len(bt)]))
    seg_t = [float(bt[b]) for b in bounds if b < len(bt)]

    # per-section p80 score as a step line + tier
    score_step = np.zeros(len(bt))
    tiers = []
    for s0, s1 in seg_edges:
        if s1 <= s0:
            continue
        sc = float(np.percentile(combined[s0:s1], 80))
        score_step[s0:s1] = sc
        tiers.append((float(bt[s0]), float(bt[s1]) if s1 < len(bt) else dur,
                      TIERS[int(np.argmin((sc - centers) ** 2))]))

    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=f.y, sr=f.sr, hop_length=HOP, n_mels=128), ref=np.max)
    cent = librosa.feature.spectral_centroid(y=f.y, sr=f.sr, hop_length=HOP)[0]
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=f.sr, hop_length=HOP)

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.2, 1.2, 1.2, 1.6, 0.6]})
    ax_spec, ax_h, ax_m, ax_l, ax_s, ax_t = axes

    librosa.display.specshow(S_db, sr=f.sr, hop_length=HOP, x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma")
    ax_spec.plot(ct, cent, color="cyan", lw=0.9, alpha=0.8, label="cyan = spectral centroid")
    ax_spec.legend(loc="upper right", fontsize=8)
    ax_spec.set_title(f"{path.stem} — every pacing component vs the audio   "
                      f"({dur:.0f}s, {f.bpm:.0f} BPM, beats end {float(bt[-1]):.0f}s)")

    def strip(ax, y, color, label, fade=1.0):
        ax.fill_between(bt, y, color=color, alpha=0.30 * fade)
        ax.plot(bt, y, color=color, lw=1.4, alpha=fade)
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 1]); ax.set_xlim(0, dur)
        ax.set_facecolor("0.12")
        ax.set_ylabel(label, fontsize=9)
        for sb in seg_t:
            ax.axvline(sb, color="white", lw=0.5, ls="--", alpha=0.35)

    strip(ax_h, height_n, C_HEIGHT, "BRIGHTNESS\nHEIGHT\n(how high)")
    strip(ax_m, move_n, C_MOVE, "MOVEMENT\n(how much it\njumps)")
    strip(ax_l, loud_n, C_LOUD, f"LOUDNESS\n(weight w={w:.2f})", fade=max(0.25, w))
    if w < 0.15:
        ax_l.text(dur * 0.5, 0.5, "loudness barely used on this track (dynamics too flat)",
                  ha="center", va="center", fontsize=9, color="0.7")

    # intensity score panel
    ax_s.plot(bt, combined, color="0.6", lw=0.8, alpha=0.7, label="per-beat combined")
    ax_s.plot(bt, score_step, color=C_SCORE, lw=2.0, label="per-section score (p80)")
    ax_s.set_facecolor("0.12"); ax_s.set_xlim(0, dur)
    ax_s.set_ylim(-0.03, max(1.03, float(combined.max()) + 0.03))
    ax_s.set_ylabel("INTENSITY\nSCORE", fontsize=9)
    for c, lbl in zip(centers, ["ambient", "slow", "normal", "intense"]):
        ax_s.axhline(c, color="0.45", lw=0.6, ls=":")
        ax_s.text(dur * 1.004, c, lbl, fontsize=7, va="center", color="0.55")
    for sb in seg_t:
        ax_s.axvline(sb, color="white", lw=0.5, ls="--", alpha=0.35)
    ax_s.legend(loc="upper right", fontsize=8)

    ax_t.set_ylim(0, 1); ax_t.set_yticks([]); ax_t.set_xlim(0, dur)
    for a, b, t in tiers:
        ax_t.add_patch(Rectangle((a, 0), b - a, 1, color=tuple(c / 255 for c in TIER_RGB[t])))
        if b - a > dur * 0.03:
            ax_t.text((a + b) / 2, 0.5, t[:4], ha="center", va="center", fontsize=7, color="black")
    ax_t.set_ylabel("TIER", fontsize=9); ax_t.set_xlabel("time (s)")

    plt.tight_layout(); plt.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
