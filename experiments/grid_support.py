"""Measure GRID SUPPORT: does librosa's beat grid actually land on the music?

waveform-eng's regime metric (Part 5): per region, the fraction of felt beats
that have a prominent onset strike within +/-70ms. Dense/tracked music scores
0.6-0.9; a rubato section where the grid is fiction scores near 0. We switch that
region to onset-anchored cutting when support < ~0.30.

Also prototypes the PROMINENCE filter for the strike picker: keep peak-picked
onsets whose envelope height exceeds a fraction of the local p90 (drop ornaments,
keep the chords). Prints how many strikes survive so we can tune it on TBAH to
keep ~8 chords.

Usage: uv run python experiments/grid_support.py            # all 4, summary
       uv run python experiments/grid_support.py "<audio>"  # one, with window plot
"""
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from selfies_for_a_year.beats import _detect_beats

# Test-track paths come from the environment (machine-specific, not committed);
# export TBAH/PUSH/RJD2/JWILL to point at your audio, e.g. via local.env or:
#   source /tmp/song_paths.sh
SONGS = {k: os.environ[k] for k in ("TBAH", "PUSH", "RJD2", "JWILL") if k in os.environ}

SUPPORT_WINDOW_MS = 70.0


def prominent_strikes(y, sr, prom_frac=0.30, rolling_s=4.0):
    """Peak-picked onset attacks kept when their envelope height exceeds
    prom_frac * local-p90(envelope). Height is read at the PEAK frame (not
    backtracked — backtrack moves to the preceding valley, whose height is
    meaningless for prominence). Returns (times, env, env_t, all_peak_times)."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr)
    env_t = librosa.times_like(env, sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=False)
    peak_t = librosa.frames_to_time(peaks, sr=sr)
    hop = 512
    w = max(1, int(rolling_s * sr / hop))
    kept = []
    for pf, pt in zip(peaks, peak_t):
        lo, hi = max(0, pf - w), min(len(env), pf + w)
        p90 = np.percentile(env[lo:hi], 90)
        if env[min(pf, len(env) - 1)] >= prom_frac * p90:
            kept.append(pt)
    return np.asarray(kept), env, env_t, peak_t


def grid_support(felt_beats, strikes, window_ms=SUPPORT_WINDOW_MS):
    """Fraction of felt beats with a strike within +/- window_ms."""
    if len(felt_beats) == 0 or len(strikes) == 0:
        return 0.0
    w = window_ms / 1000.0
    hits = sum(1 for b in felt_beats if np.min(np.abs(strikes - b)) <= w)
    return hits / len(felt_beats)


def analyze(name, path, plot=False):
    import librosa
    y, sr = librosa.load(str(path), mono=True)
    bt, strengths, loud, la, bpm, dur = _detect_beats(Path(path))
    bt = np.asarray(bt); strengths = np.asarray(strengths)
    par = 0 if strengths[0::2].mean() >= strengths[1::2].mean() else 1
    felt = bt[par::2]
    strikes, env, env_t, all_peaks = prominent_strikes(y, sr)

    overall = grid_support(felt, strikes)
    raw_support = grid_support(felt, all_peaks)  # no prominence filter
    # sliding 8s window support profile
    win, step = 8.0, 2.0
    prof = []
    t = 0.0
    while t < dur:
        fb = felt[(felt >= t) & (felt < t + win)]
        prof.append((t + win / 2, grid_support(fb, strikes)))
        t += step
    prof = np.array(prof)
    sparse_frac = float(np.mean(prof[:, 1] < 0.30)) if len(prof) else 0.0
    strike_rate = len(strikes) / dur
    print(f"{name:6s} bpm~{60/np.median(np.diff(bt)):.0f} felt~{60/np.median(np.diff(bt))/2:.0f}  "
          f"grid-support: prom={overall:.2f} raw={raw_support:.2f}  strikes={len(strikes)} "
          f"({strike_rate:.2f}/s, {len(all_peaks)} raw)  windows<0.30={sparse_frac*100:.0f}%")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = f"/tmp/grid_support_{Path(path).stem[:12].replace(' ','_')}.png"
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.plot(prof[:, 0], prof[:, 1], color="purple", lw=1.5)
        ax.axhline(0.30, color="red", ls="--", lw=1, label="onset-anchor threshold 0.30")
        ax.fill_between(prof[:, 0], 0, 0.30, color="orange", alpha=0.12)
        ax.set_ylim(0, 1); ax.set_xlim(0, dur)
        ax.set_xlabel("time (s)"); ax.set_ylabel("grid support (8s window)")
        ax.set_title(f"{name}: grid support over time — below red = onset-anchor regime")
        ax.legend(loc="upper right")
        plt.tight_layout(); plt.savefig(out, dpi=95); print("  wrote", out)
    return prof


if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser()
        analyze(p.stem[:6], p, plot=True)
    else:
        for n, p in SONGS.items():
            analyze(n, Path(p))
