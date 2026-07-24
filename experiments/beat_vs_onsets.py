"""Is the detected beat grid actually ON the music? Overlay the audio's onset
envelope (where the PIANO STRIKES / note attacks are) against librosa's detected
beats (what the metronome dot flashes on) and the scheduled CUTS. If the blue
beat ticks don't sit on the onset-envelope peaks, the metronome is off and every
cut inherits that error — no scheduling fix helps until detection is anchored.

Usage:
  uv run python experiments/beat_vs_onsets.py "<audio>" [window_s]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from selfies_for_a_year.beats import build_timeline, _detect_beats


def main(argv):
    import librosa
    path = Path(argv[0]).expanduser()
    win = float(argv[1]) if len(argv) > 1 else 15.0
    out = f"/tmp/beat_vs_onsets_{path.stem[:12].replace(' ', '_')}.png"

    y, sr = librosa.load(str(path), mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    env_t = librosa.times_like(onset_env, sr=sr)
    # peak-picked onsets = actual note attacks (piano strikes)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time",
                                         backtrack=True)

    bt, strengths, loud, la, bpm, dur = _detect_beats(path)
    bt = np.asarray(bt); strengths = np.asarray(strengths)
    par = 0 if strengths[0::2].mean() >= strengths[1::2].mean() else 1

    from datetime import datetime, timedelta
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(400)]
    tl = build_timeline(path, dates, min_photos_per_beat=0.25, max_photos_per_second=4.0,
                        vary_pace=True, pace_model="segment", base_pace="occupancy",
                        intense_multiplier=3, slow_multiplier=0.33)
    t = 0.0; cuts = []
    for s in tl.segments:
        cuts.append(t); t += s.duration
    cuts = np.array(cuts)

    m = env_t <= win
    fig, ax = plt.subplots(figsize=(17, 4))
    # onset envelope (piano strikes = peaks)
    ax.fill_between(env_t[m], 0, onset_env[m] / (onset_env[m].max() or 1),
                    color="0.75", label="onset strength (note attacks)")
    # peak-picked onsets = the actual strikes
    for o in onsets:
        if o <= win:
            ax.axvline(o, color="darkorange", lw=2.0, alpha=0.9)
    ax.plot([], [], color="darkorange", lw=2, label="detected note attacks (strikes)")
    # detected beats (metronome dot), felt parity tall
    for i, b in enumerate(bt):
        if b > win:
            break
        if i % 2 == par:
            ax.vlines(b, 0, 1.15, color="#1f4e9c", lw=1.6)
        else:
            ax.vlines(b, 0, 0.6, color="#8fb0e0", lw=0.9)
    ax.plot([], [], color="#1f4e9c", lw=1.6, label="detected FELT beat (metronome)")
    ax.plot([], [], color="#8fb0e0", lw=0.9, label="detected off-beat")
    # cuts
    for c in cuts[cuts <= win]:
        ax.axvline(c, color="green", lw=1.4, ls="--", alpha=0.8)
    ax.plot([], [], color="green", lw=1.4, ls="--", label="scheduled CUT")

    ax.set_xlim(0, win); ax.set_ylim(0, 1.25); ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{path.stem} — do the blue metronome beats sit on the orange strikes? "
                 f"(bpm~{60/np.median(np.diff(bt)):.0f}, felt~{60/np.median(np.diff(bt))/2:.0f})")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=95)
    print("wrote", out)
    # numeric: median distance from each detected FELT beat to nearest strike
    felt = bt[par::2]
    felt = felt[felt <= win]
    if len(onsets):
        d = [min(abs(f - onsets)) for f in felt]
        print(f"  felt beats in first {win:.0f}s: {len(felt)}; "
              f"median |felt-beat -> nearest strike| = {np.median(d)*1000:.0f} ms "
              f"(beat period {np.median(np.diff(bt))*2*1000:.0f} ms)")
    print(f"  first 3 cuts at: {[round(float(c),2) for c in cuts[:3]]}")
    print(f"  first 6 strikes at: {[round(float(o),2) for o in onsets[:6]]}")


if __name__ == "__main__":
    main(sys.argv[1:])
