"""Decompose the pacing decision: for each segment, show how much each term
contributed to the score that picked its tier. Answers "WHY is this section
intense?" by splitting `combined` into its additive parts:

    combined = height_n/(1+w)  +  w*loud_n/(1+w)  +  0.6*depth_n
               \___________/     \____________/     \_________/
                CYAN HEIGHT         LOUDNESS         BURSTINESS
                (the "star")     (variance-gated)   (modulation depth)

Prints a per-segment table and writes a stacked-contribution graph so you can
SEE which term carries each intense section.

Run: uv run python experiments/pace_debug.py "<audio>" [out.png]
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
    _recipe_c_features, _recipe_c_boundaries, _mode_anchored_centers,
)

TIER_RGB = {"ambient": (150, 150, 150), "slow": (90, 200, 110),
            "normal": (240, 210, 70), "intense": (255, 95, 95)}


def decompose(f):
    hr, lr, mf, md = _recipe_c_features(f)
    height_n = _robust_norm(hr)
    loud_n = _robust_norm(f.loudness_db)
    depth_n = _robust_norm(md)
    iqr_db = float(np.percentile(f.loudness_db, 75) - np.percentile(f.loudness_db, 25))
    w = float(np.clip((iqr_db - 3.0) / 6.0, 0.0, 1.0))
    # additive contributions to `combined` (they sum to it exactly)
    c_height = height_n / (1.0 + w)
    c_loud = w * loud_n / (1.0 + w)
    c_depth = 0.6 * depth_n
    combined = c_height + c_loud + c_depth
    bounds = _recipe_c_boundaries(hr, lr, mf, md)
    centers, _ = _mode_anchored_centers(combined)
    return dict(w=w, iqr_db=iqr_db, c_height=c_height, c_loud=c_loud,
                c_depth=c_depth, combined=combined, bounds=bounds, centers=centers)


def main(argv):
    import librosa
    path = Path(argv[0]).expanduser()
    out = argv[1] if len(argv) > 1 else "/tmp/pace_debug.png"
    f = load_features(path)
    d = decompose(f)
    bt, dur = f.beat_times, f.duration
    seg_edges = list(zip(d["bounds"], d["bounds"][1:] + [len(bt)]))

    print(f"\n{path.stem}   ({len(bt)} beats, {dur:.0f}s, {f.bpm:.0f} BPM)")
    print(f"loudness IQR = {d['iqr_db']:.1f} dB  ->  loudness weight w = {d['w']:.2f}"
          f"   (0=height only, 1=full loudness)\n")
    # First pass: each segment's top-20%-beat component means (what the p80 score
    # "sees"). Then baseline = the TYPICAL section (median across segments), so a
    # term is the "driver" only where it's elevated vs a normal section — the real
    # question for "why is THIS section more intense than its neighbours?"
    rows = []
    for s0, s1 in seg_edges:
        if s1 <= s0:
            continue
        seg_comb = d["combined"][s0:s1]
        top = seg_comb >= np.percentile(seg_comb, 80)
        rows.append(dict(
            s0=s0, s1=s1, t0=float(bt[s0]), t1=float(bt[s1]) if s1 < len(bt) else dur,
            score=float(np.percentile(seg_comb, 80)),
            h=float(d["c_height"][s0:s1][top].mean()),
            l=float(d["c_loud"][s0:s1][top].mean()),
            b=float(d["c_depth"][s0:s1][top].mean()),
        ))
    base_h = float(np.median([r["h"] for r in rows]))
    base_l = float(np.median([r["l"] for r in rows]))
    base_b = float(np.median([r["b"] for r in rows]))
    print(f"  typical section (median across segments): HEIGHT={base_h:.2f}  LOUD={base_l:.2f}  BURST={base_b:.2f}")
    print(f"  {'segment':>15s} | {'p80':>5s} | {'HEIGHT Δ':>14s} {'LOUD Δ':>12s} {'BURST Δ':>12s} | tier   | what lifts it")
    print("  " + "-" * 96)
    for r in rows:
        dh, dl, db = r["h"] - base_h, r["l"] - base_l, r["b"] - base_b
        driver = max({"HEIGHT": dh, "LOUD": dl, "BURST": db}.items(), key=lambda kv: kv[1])[0]
        tier = TIERS[int(np.argmin((r["score"] - d["centers"]) ** 2))]
        note = f"{driver} vs typical" if tier == "intense" else ""
        print(f"  {r['t0']:5.0f}-{r['t1']:<4.0f}s ({r['s1']-r['s0']:3d}b) | {r['score']:.2f}  |"
              f" {r['h']:.2f} ({dh:+.2f})  {r['l']:.2f} ({dl:+.2f})  {r['b']:.2f} ({db:+.2f}) |"
              f" {tier:7s}| {note}")
    print(f"\n  centers: ambient={d['centers'][0]:.2f} slow={d['centers'][1]:.2f} "
          f"normal={d['centers'][2]:.2f} intense={d['centers'][3]:.2f}")

    # ---- stacked contribution graph ----
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=f.y, sr=f.sr, hop_length=HOP, n_mels=128), ref=np.max)
    cent = librosa.feature.spectral_centroid(y=f.y, sr=f.sr, hop_length=HOP)[0]
    ct = librosa.frames_to_time(np.arange(len(cent)), sr=f.sr, hop_length=HOP)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 0.7]})
    librosa.display.specshow(S_db, sr=f.sr, hop_length=HOP, x_axis="time", y_axis="mel", ax=ax1, cmap="magma")
    ax1.plot(ct, cent, color="cyan", lw=0.9, alpha=0.8)
    for s0, _ in seg_edges:
        ax1.axvline(float(bt[s0]), color="white", lw=0.7, ls="--", alpha=0.5)
    ax1.set_title(f"{path.stem} — why each section gets its tier (stacked contributions)   w_loud={d['w']:.2f}")

    ax2.stackplot(bt, d["c_height"], d["c_loud"], d["c_depth"],
                  labels=["cyan HEIGHT", "LOUDNESS (gated)", "BURSTINESS (moddepth)"],
                  colors=["#2ec4d6", "#e8a13a", "#b06cd0"], alpha=0.85)
    for c, lbl in zip(d["centers"], ["amb", "slow", "norm", "int"]):
        ax2.axhline(c, color="0.3", lw=0.7, ls=":")
        ax2.text(dur * 1.005, c, lbl, fontsize=7, va="center", color="0.3")
    ax2.set_ylabel("contribution to score"); ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(0, dur)

    ax3.set_ylim(0, 1); ax3.set_yticks([]); ax3.set_xlim(0, dur)
    for s0, s1 in seg_edges:
        if s1 <= s0:
            continue
        score = float(np.percentile(d["combined"][s0:s1], 80))
        tier = TIERS[int(np.argmin((score - d["centers"]) ** 2))]
        a = float(bt[s0]); bb = float(bt[s1]) if s1 < len(bt) else dur
        ax3.add_patch(Rectangle((a, 0), bb - a, 1, color=tuple(c / 255 for c in TIER_RGB[tier])))
        if bb - a > dur * 0.03:
            ax3.text((a + bb) / 2, 0.5, tier[:4], ha="center", va="center", fontsize=7, color="black")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("tier")

    plt.tight_layout(); plt.savefig(out, dpi=90)
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
