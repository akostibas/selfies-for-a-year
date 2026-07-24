"""A/B harness for pacing-tier recipes.

Runs several pacing-tier "recipes" over the SAME audio and prints their tier
maps + sanity metrics side by side, with NO photo alignment and NO render — so
we can judge whether a recipe carves a song into sensible ambient/slow/normal/
intense stretches from text alone. This is the overfitting defense: a recipe is
only good if it looks reasonable across a genre-spread set, not on one song.

Recipes:
  current  — the shipping pipeline (onset-strength gate + quantile top-N
             sections). Faithful: driven through the real build_timeline.
  A        — simple relative (Gemini): per-beat energy = loudness(dB)+onset-rate,
             per-song percentile tiers, min-duration smoothing to kill chatter.
  B        — absolute-ish (research): loudness-normalized RMS-dB with FIXED-unit
             thresholds; tier boundaries snapped to Foote novelty segments.

RMS-in-dB stands in for short-term LUFS here (pyloudnorm won't build on 3.14);
after loudness-normalization it behaves the same for this comparison, and it
keeps the "fixed thresholds vs varying dynamic range" question fully in play.

Run:
  uv run python experiments/pacing_recipes.py "<audio.m4a>" ["<audio2>" ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HOP = 512  # librosa default hop for onset/rms frames

TIERS = ("ambient", "slow", "normal", "intense")
_TIER_ORD = {t: i for i, t in enumerate(TIERS)}
_TIER_GLYPH = {"ambient": "·", "slow": "░", "normal": "▓", "intense": "█"}


# --------------------------------------------------------------------------- #
# Shared audio front-end
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    path: Path
    y: np.ndarray
    sr: int
    duration: float
    bpm: float
    beat_times: np.ndarray       # (n_beats,)
    beat_frames: np.ndarray      # (n_beats,) onset/rms frame index per beat
    onset_strength_norm: np.ndarray  # per-beat, max-normalized (current's gate)
    loudness_db: np.ndarray      # per-beat mean RMS in dBFS
    onset_rate: np.ndarray       # per-beat onset density (onsets/sec, 1s window)
    brightness: np.ndarray       # per-beat mean spectral centroid (Hz) — filter sweeps
    flux: np.ndarray             # per-beat mean spectral flux — ACTIVITY/burstiness
    activity: np.ndarray         # per-beat spread (std) of centroid over ~2s
    height: np.ndarray           # per-beat "cyan height": p90 upper-envelope of the
                                 # centroid over ~2s = how HIGH the brightness reaches.
                                 # Cleanly orders intense>normal>low>ambient.


def _beat_agg(arr: np.ndarray, beat_frames: np.ndarray, agg=np.mean) -> np.ndarray:
    """Aggregate a per-frame feature over each beat's window [beat_i, beat_i+1).
    Window MEAN (not point-sample) so a gated/bursty beat reflects its bursts
    instead of whatever single frame the beat landed on (maybe a gap)."""
    n = len(beat_frames)
    out = np.zeros(n)
    for i in range(n):
        lo = int(beat_frames[i])
        hi = int(beat_frames[i + 1]) if i + 1 < n else len(arr)
        out[i] = agg(arr[lo:hi]) if hi > lo else arr[min(lo, len(arr) - 1)]
    return out


def load_features(path: Path) -> Features:
    import librosa

    y, sr = librosa.load(str(path), mono=True)
    duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP
    )
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])

    # Per-beat onset strength, max-normalized (matches _detect_beats).
    if len(beat_frames):
        strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
        if strengths.max() > 0:
            strengths = strengths / strengths.max()
    else:
        strengths = np.array([])

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]

    # Frame-level derived centroid signals over a ~2s window:
    #  - std  = "thickness of the cyan band" (activity/burstiness)
    #  - p90 envelope = "cyan height" (how HIGH brightness reaches) — the signal
    #    the product owner identified as cleanly ordering the tiers.
    from scipy.ndimage import percentile_filter, uniform_filter1d
    w = max(1, int(2.0 * sr / HOP))
    m = uniform_filter1d(centroid, w, mode="nearest")
    m2 = uniform_filter1d(centroid**2, w, mode="nearest")
    centroid_std = np.sqrt(np.maximum(m2 - m**2, 0.0))
    centroid_height = uniform_filter1d(percentile_filter(centroid, 90, size=w), w // 2, mode="nearest")

    if len(beat_frames):
        # Window-MEAN aggregates: a bursty/gated beat reads its activity, not a gap.
        loudness_db = _beat_agg(rms_db, beat_frames)
        brightness = _beat_agg(centroid, beat_frames)
        flux = _beat_agg(onset_env, beat_frames)  # spectral flux = activity/burstiness
        activity = _beat_agg(centroid_std, beat_frames)
        height = _beat_agg(centroid_height, beat_frames)
    else:
        loudness_db = brightness = flux = activity = height = np.array([])

    # Per-beat onset rate: onsets within a 1s window centered on each beat.
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time"
    )
    onset_times = np.asarray(onset_times)
    if len(beat_times) and len(onset_times):
        lo = np.searchsorted(onset_times, beat_times - 0.5)
        hi = np.searchsorted(onset_times, beat_times + 0.5)
        onset_rate = (hi - lo).astype(float)  # onsets per 1s window
    else:
        onset_rate = np.zeros(len(beat_times))

    return Features(
        path=path, y=y, sr=sr, duration=duration, bpm=bpm,
        beat_times=beat_times, beat_frames=beat_frames,
        onset_strength_norm=strengths, loudness_db=loudness_db, onset_rate=onset_rate,
        brightness=brightness, flux=flux, activity=activity, height=height,
    )


# --------------------------------------------------------------------------- #
# Metrics + rendering (one code path for all recipes -> apples to apples)
# --------------------------------------------------------------------------- #
@dataclass
class Summary:
    name: str
    intervals: list[tuple[float, float, str]]
    duration: float
    bpm: float

    @property
    def merged(self) -> list[tuple[float, float, str]]:
        """Collapse same-tier neighbors."""
        out: list[tuple[float, float, str]] = []
        for a, b, t in self.intervals:
            if out and out[-1][2] == t:
                out[-1] = (out[-1][0], b, t)
            else:
                out.append((a, b, t))
        return out

    def metrics(self) -> dict:
        m = self.merged
        runs = [b - a for a, b, _ in m]
        frac: dict[str, float] = {}
        for a, b, t in m:
            frac[t] = frac.get(t, 0.0) + (b - a) / self.duration if self.duration else 0.0
        # Implied peak cadence: intense photos/sec at the base subdivision.
        from selfies_for_a_year.beats import _pick_subdivision
        sub, _ = _pick_subdivision(self.bpm, 4.0, 1.0, None)
        bps = self.bpm / 60.0
        mult = {"intense": 2.0, "slow": 0.5, "normal": 1.0, "ambient": 0.0}
        peak_pps = max(
            (bps * sub * mult[t] for _, _, t in m if mult[t] > 0), default=0.0
        )
        return {
            "transitions": max(0, len(m) - 1),
            "mean_run": (sum(runs) / len(runs)) if runs else 0.0,
            "shortest_run": min(runs) if runs else 0.0,
            "frac": frac,
            "peak_pps": peak_pps,
        }

    def render(self) -> str:
        def fmt(t: float) -> str:
            mm, ss = divmod(int(t), 60)
            return f"{mm}:{ss:02d}"

        m = self.merged
        met = self.metrics()
        lines = [f"── {self.name} ──"]
        # 60-char strip of the whole song, one glyph per second-ish.
        WIDTH = 72
        strip = [" "] * WIDTH
        for a, b, t in m:
            i0 = int(a / self.duration * WIDTH) if self.duration else 0
            i1 = int(b / self.duration * WIDTH) if self.duration else 0
            for i in range(max(0, i0), min(WIDTH, max(i1, i0 + 1))):
                strip[i] = _TIER_GLYPH[t]
        lines.append("  [" + "".join(strip) + "]")
        for a, b, t in m:
            bar = _TIER_GLYPH[t] * max(1, round((b - a) / self.duration * 40))
            lines.append(f"  {fmt(a):>4}–{fmt(b):<4} {t:<8} {b - a:5.1f}s {bar}")
        frac = ", ".join(
            f"{k} {v * 100:.0f}%" for k, v in sorted(met["frac"].items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"  metrics: {met['transitions']} transitions, mean run "
            f"{met['mean_run']:.1f}s, shortest {met['shortest_run']:.1f}s, "
            f"peak ~{met['peak_pps']:.1f} ph/s"
        )
        lines.append(f"           time in tier: {frac}")
        return "\n".join(lines)


def energy_sparkline(f: Features, cols: int = 72) -> str:
    """Sparkline of the smoothed per-beat energy A sees — so we can eyeball
    whether tier assignments actually track the signal."""
    if len(f.beat_times) == 0:
        return ""
    energy = _energy_signal(f)
    blocks = "▁▂▃▄▅▆▇█"
    last_beat = f.beat_times[-1]
    # resample energy to `cols` by beat time. Past the last tracked beat there
    # is no data — carry the last value forward (mark with ':') rather than
    # drawing a false floor, so a beat-tracking dropout isn't read as silence.
    out = []
    for c in range(cols):
        t0 = c / cols * f.duration
        t1 = (c + 1) / cols * f.duration
        if t0 > last_beat:
            out.append(":")
            continue
        lo = int(np.searchsorted(f.beat_times, t0))
        hi = max(lo + 1, int(np.searchsorted(f.beat_times, t1)))
        v = float(np.mean(energy[lo:hi])) if hi > lo and lo < len(energy) else 0.0
        out.append(blocks[min(7, max(0, int(v * 8)))])
    # mark the intense cutoff (92.5 pct) as a reference
    p_int = np.percentile(energy, 92.5)
    return "  energy [" + "".join(out) + f"]  (intense cut @ {p_int:.2f})"


def beats_to_intervals(
    beat_times: np.ndarray, tiers: list[str], duration: float
) -> list[tuple[float, float, str]]:
    """Turn a per-beat tier list into (start, end, tier) runs covering [0, dur].

    Each beat owns [beat_time, next_beat_time). Pre-roll and tail inherit the
    nearest beat's tier."""
    if len(beat_times) == 0:
        return [(0.0, duration, "ambient")]
    out: list[tuple[float, float, str]] = []
    for i, t in enumerate(tiers):
        start = beat_times[i] if i > 0 else 0.0
        end = beat_times[i + 1] if i + 1 < len(beat_times) else duration
        out.append((float(start), float(end), t))
    # merge same-tier neighbors
    merged: list[tuple[float, float, str]] = []
    for a, b, t in out:
        if merged and merged[-1][2] == t:
            merged[-1] = (merged[-1][0], b, t)
        else:
            merged.append((a, b, t))
    return merged


# --------------------------------------------------------------------------- #
# Recipe: current (shipping pipeline, faithful via build_timeline)
# --------------------------------------------------------------------------- #
def recipe_current(f: Features) -> list[tuple[float, float, str]]:
    from selfies_for_a_year.beats import TrackProgression, build_timeline

    # Synthetic photo dates: enough that the video spans the WHOLE song (else
    # build_timeline ends the timeline when photos run out and truncates the
    # tier map). Tier labels are audio-only, so the exact count doesn't shift
    # them — we just need plenty.
    n = 4000
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    timeline = build_timeline(
        f.path, dates, vary_pace=True, tier_lead_seconds=0.0,
    )
    prog = TrackProgression.from_timeline(timeline)
    return [(s.start, s.end, s.tier) for s in prog.states]


# --------------------------------------------------------------------------- #
# Recipe A: simple relative (Gemini)
# --------------------------------------------------------------------------- #
def _robust_norm(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] using 5th/95th pct as the range (silence-robust)."""
    if len(x) == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _moving_median(x: np.ndarray, win: int) -> np.ndarray:
    """Odd-window moving median (edge-padded). Smooths the CONTINUOUS energy
    signal so tiers come out contiguous — the correct place to kill chatter,
    vs. smoothing the discrete labels (which cascades into one tier)."""
    if win <= 1 or len(x) == 0:
        return x
    if win % 2 == 0:
        win += 1
    r = win // 2
    pad = np.pad(x, r, mode="edge")
    return np.array([np.median(pad[i : i + win]) for i in range(len(x))])


def _viterbi_tiers(
    energy: np.ndarray, centers: np.ndarray, sigma: float, switch_penalty: float
) -> list[str]:
    """Assign each beat to a tier by Viterbi, trading emission fit against a
    per-switch penalty. High switch_penalty -> long contiguous runs, no chatter.
    ONE knob does what a pile of debounce rules can't: globally optimal, not greedy.

    emission cost = (energy - center)^2 / (2 sigma^2);  switch cost = penalty.
    """
    n = len(energy)
    k = len(centers)
    if n == 0:
        return []
    emit = (energy[:, None] - centers[None, :]) ** 2 / (2.0 * sigma**2)  # (n, k)
    cost = np.full((n, k), np.inf)
    back = np.zeros((n, k), dtype=int)
    cost[0] = emit[0]
    for t in range(1, n):
        for s in range(k):
            trans = cost[t - 1] + np.where(np.arange(k) == s, 0.0, switch_penalty)
            j = int(np.argmin(trans))
            cost[t, s] = trans[j] + emit[t, s]
            back[t, s] = j
    # backtrace
    path = [int(np.argmin(cost[-1]))]
    for t in range(n - 1, 0, -1):
        path.append(back[t, path[-1]])
    path.reverse()
    return [TIERS[s] for s in path]


def _merge_short_runs(tiers: list[str], min_beats: int) -> list[str]:
    """Merge any tier run shorter than min_beats into its longer neighbor.
    Applied to already-clean Viterbi output (few, long runs), so it's a stable
    single sweep — encodes 'a tier must last >= min_beats to register as a
    section' (a 2s pacing flip is a stutter, not a section)."""
    out = list(tiers)
    changed = True
    while changed:
        changed = False
        runs: list[list] = []  # [start, end, tier]
        i = 0
        while i < len(out):
            j = i
            while j + 1 < len(out) and out[j + 1] == out[i]:
                j += 1
            runs.append([i, j, out[i]])
            i = j + 1
        if len(runs) <= 1:
            break
        # shortest short run first, merge into longer neighbor
        shorts = [r for r in runs if (r[1] - r[0] + 1) < min_beats]
        if not shorts:
            break
        r = min(shorts, key=lambda r: r[1] - r[0])
        k = runs.index(r)
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        pick = (left if (not right or (left and (left[1] - left[0]) >= (right[1] - right[0])))
                else right)
        for idx in range(r[0], r[1] + 1):
            out[idx] = pick[2]
        changed = True
    return out


def _energy_signal(f: Features, smooth_beats: int = 6) -> np.ndarray:
    """Per-beat energy = cyan HEIGHT (p90 upper-envelope of the spectral centroid
    — how high the brightness reaches). This single signal empirically orders
    the tiers intense>normal>low>ambient far more cleanly than brightness (which
    inverts on gated passages), activity/spread, or loudness (flat on compressed
    tracks). Robust-normalized per song, then median-smoothed."""
    energy = _robust_norm(f.height)
    return _moving_median(energy, smooth_beats)


def recipe_a(
    f: Features, *, smooth_beats: int = 6, switch_penalty: float = 4.0,
    min_run_seconds: float = 4.0,
) -> list[tuple[float, float, str]]:
    if len(f.beat_times) == 0:
        return [(0.0, f.duration, "ambient")]
    energy = _energy_signal(f, smooth_beats=smooth_beats)
    # Tier centers anchored at the BASELINE (the mode — the energy the song sits
    # at most often = the groove/verse), NOT the median. The median is dragged
    # up by peaks, which pushes the baseline groove down into 'slow'; the mode
    # stays put. "normal" = baseline; "intense" a margin ABOVE; slow/ambient
    # below. Excursions are asymmetric (up-spread vs down-spread scaled
    # separately) so a track with big peaks but a flat floor (techno) gets lots
    # of intense and little slow, while a dynamic track gets both.
    lo, hi = float(np.percentile(energy, 5)), float(np.percentile(energy, 95))
    hist, edges = np.histogram(energy, bins=20, range=(energy.min(), energy.max() + 1e-9))
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    up = max(hi - mode, 0.05)
    dn = max(mode - lo, 0.05)
    centers = np.clip(
        np.array([mode - 0.9 * dn, mode - 0.45 * dn, mode, mode + 0.6 * up]),
        0.0, 1.0,
    )
    sigma = max(0.35 * (hi - lo), 0.05)
    tiers = _viterbi_tiers(energy, centers, sigma, switch_penalty)
    min_beats = max(1, round(min_run_seconds * f.bpm / 60.0))
    tiers = _merge_short_runs(tiers, min_beats)
    return beats_to_intervals(f.beat_times, tiers, f.duration)


# --------------------------------------------------------------------------- #
# Recipe B: absolute-ish RMS-dB + Foote novelty segments
# --------------------------------------------------------------------------- #
def _foote_novelty(ssm: np.ndarray, kernel_size: int) -> np.ndarray:
    """Foote checkerboard novelty along the diagonal of a self-similarity matrix."""
    L = kernel_size
    # Gaussian-tapered checkerboard kernel of size 2L x 2L.
    g = np.linspace(-1.0, 1.0, 2 * L)
    gauss = np.outer(np.exp(-4.0 * g**2), np.exp(-4.0 * g**2))
    sign = np.outer(np.sign(g), np.sign(g))
    kernel = gauss * sign
    n = ssm.shape[0]
    nov = np.zeros(n)
    for i in range(n):
        a = i - L
        b = i + L
        pa, pb = max(0, a), min(n, b)
        ka, kb = pa - a, 2 * L - (b - pb)
        patch = ssm[pa:pb, pa:pb]
        k = kernel[ka:kb, ka:kb]
        nov[i] = float((patch * k).sum())
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov = nov / nov.max()
    return nov


def _foote_boundaries(f: Features, kernel_beats: int = 16, thresh: float = 0.15) -> list[int]:
    """Beat indices where a new segment starts (Foote novelty peaks on beat-sync MFCC)."""
    import librosa
    from scipy.signal import find_peaks

    if len(f.beat_frames) < 4:
        return [0]
    mfcc = librosa.feature.mfcc(y=f.y, sr=f.sr, hop_length=HOP, n_mfcc=13)
    sync = librosa.util.sync(mfcc, f.beat_frames, aggregate=np.mean)  # (13, n_beats)
    x = sync.T  # (n_beats, 13)
    # cosine self-similarity
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norm, 1e-9)
    ssm = xn @ xn.T
    kb = min(kernel_beats, ssm.shape[0] // 2) or 1
    nov = _foote_novelty(ssm, kb)
    peaks, _ = find_peaks(nov, height=thresh, distance=kb)
    bounds = [0] + [int(p) for p in peaks]
    return sorted(set(bounds))


def recipe_b(
    f: Features, *, target_lufs: float = -18.0,
    bands: tuple[float, float, float] = (-30.0, -22.0, -14.0),  # amb|slow, slow|norm, norm|int
) -> list[tuple[float, float, str]]:
    if len(f.beat_times) == 0:
        return [(0.0, f.duration, "ambient")]
    # "loudness-normalize": shift so the track's overall RMS-dB sits at target.
    overall = float(np.median(f.loudness_db))
    loud_n = f.loudness_db + (target_lufs - overall)
    rate = _robust_norm(f.onset_rate)
    b_amb, b_slow, b_norm = bands

    bounds = _foote_boundaries(f)
    seg_edges = bounds + [len(f.beat_times)]
    tiers = ["normal"] * len(f.beat_times)
    for s0, s1 in zip(seg_edges[:-1], seg_edges[1:]):
        if s1 <= s0:
            continue
        seg_loud = float(np.median(loud_n[s0:s1]))
        seg_rate = float(np.median(rate[s0:s1]))
        if seg_loud < b_amb and seg_rate < 0.3:
            t = "ambient"
        elif seg_loud < b_slow:
            t = "slow"
        elif seg_loud < b_norm:
            t = "normal"
        else:
            t = "intense"
        for i in range(s0, s1):
            tiers[i] = t
    return beats_to_intervals(f.beat_times, tiers, f.duration)


# --------------------------------------------------------------------------- #
# Recipe C: segment-first (Foote on stacked beat-synced features) then score
# each SEGMENT by p80 cyan-height. Per waveform-eng's review.
# --------------------------------------------------------------------------- #
def _mode_anchored_centers(vals: np.ndarray) -> tuple[np.ndarray, float]:
    """Tier centers anchored at the BASELINE (mode) of the value distribution,
    with asymmetric up/down spread. Not clipped to [0,1]: `combined` (height +
    modulation-depth boost) ranges past 1.0, and clipping crushed the normal/
    intense centers together on tracks whose typical section already sits high
    (e.g. TBAH's rolling-p80 mode ~1.0), erasing the intense tier."""
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    hist, edges = np.histogram(vals, bins=20, range=(vals.min(), vals.max() + 1e-9))
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    up = max(hi - mode, 0.05)
    dn = max(mode - lo, 0.05)
    centers = np.maximum(
        np.array([mode - 0.9 * dn, mode - 0.45 * dn, mode, mode + 0.6 * up]), 0.0
    )
    return centers, max(0.35 * (hi - lo), 0.05)


def _recipe_c_features(f: Features):
    """UNSMOOTHED per-beat features for segmentation + a raw per-beat cyan-height.
    Smoothing is deliberately absent here: the Foote kernel keys on the burst/gap
    texture, which smoothing would erase (waveform-eng)."""
    import librosa

    bf = f.beat_frames
    cent = librosa.feature.spectral_centroid(y=f.y, sr=f.sr, hop_length=HOP)[0]
    logcent = np.log2(np.maximum(cent, 1e-6))  # mel-ish log-freq (owner drew on mel)
    rms = librosa.feature.rms(y=f.y, hop_length=HOP)[0]
    logrms = np.log(np.maximum(rms, 1e-6))
    mfcc = librosa.feature.mfcc(y=f.y, sr=f.sr, hop_length=HOP, n_mfcc=13)

    p90 = lambda a: float(np.percentile(a, 90)) if len(a) else 0.0
    height_raw = _beat_agg(logcent, bf, agg=p90)          # per-beat "cyan height"
    lrms_beat = _beat_agg(logrms, bf)
    # per-beat MFCC via the same window-mean aggregator (matches lengths exactly)
    mfcc_beat = np.column_stack([_beat_agg(mfcc[k], bf) for k in range(mfcc.shape[0])])

    # modulation depth: p90-p10 of raw height over ~16 beats (burst-train marker)
    n = len(height_raw)
    moddepth = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - 8), min(n, i + 9)
        w = height_raw[a:b]
        moddepth[i] = np.percentile(w, 90) - np.percentile(w, 10) if len(w) else 0.0
    return height_raw, lrms_beat, mfcc_beat, moddepth


def _rolling_p80(x: np.ndarray, win: int = 16) -> np.ndarray:
    """p80 of x in a centered ~win-beat window at every beat. Used to calibrate
    tier centers on the SAME statistic family sections are scored with (p80).
    Fitting centers on raw per-beat values instead is a statistic mismatch: p80
    is upward-biased vs per-beat, so on burst-texture tracks (many gap beats) the
    per-beat mode sits among the gaps and EVERY section's p80 rides above it —
    making slow/ambient unreachable for any section. Calibrating on rolling-p80
    makes the mode mean 'typical section-level score', so lulls can fall below
    it. Order statistic of actual values => magnitudes survive (unlike a rank
    transform, which manufactures range on flat tracks and flattens dynamic ones)."""
    n = len(x)
    if n == 0:
        return x
    r = win // 2
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - r), min(n, i + r + 1)
        out[i] = np.percentile(x[a:b], 80)
    return out


def _beat_smooth(x: np.ndarray, w: int = 8) -> np.ndarray:
    """Centered moving mean over ~w beats (1-D or per-column for 2-D)."""
    from scipy.ndimage import uniform_filter1d
    if x.ndim == 1:
        return uniform_filter1d(x, w, mode="nearest")
    return np.column_stack([uniform_filter1d(x[:, k], w, mode="nearest") for k in range(x.shape[1])])


def _recipe_c_boundaries(height_raw, lrms_beat, mfcc_beat, moddepth,
                         kernel_beats: int = 16, min_seg_beats: int = 16,
                         prominence: float = 0.12) -> list[int]:
    """Beat indices where segments start. The SSM compares TEXTURE SUMMARIES, not
    raw per-beat values (waveform-eng): the level channels (MFCC/log-RMS/height)
    alternate burst/gap inside a train and checker the SSM, so novelty phase-locks
    mid-train. An 8-beat texture-window (>= one burst cycle) flattens each channel
    to 'avg of burst+gap' across the train, so the block goes flat and novelty
    fires only at real edges. moddepth (already a 16b window stat) is smooth by
    construction. Kernel stays 16b so 20-30s sections aren't blurred."""
    from scipy.signal import find_peaks

    n = len(height_raw)
    if n < 2 * kernel_beats:
        return [0]
    level = _beat_smooth(np.column_stack([mfcc_beat, lrms_beat, height_raw]), 8)
    feat = np.column_stack([level, moddepth])
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)  # z-score each channel
    fn = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-9)
    ssm = fn @ fn.T
    kb = min(kernel_beats, n // 2) or 1
    nov = _foote_novelty(ssm, kb)
    peaks, _ = find_peaks(nov, prominence=prominence, distance=max(min_seg_beats, kb))
    return sorted(set([0] + [int(p) for p in peaks]))


def _recipe_c_energy(f: Features, height_raw: np.ndarray) -> np.ndarray:
    """Per-beat energy = cyan height with a VARIANCE-GATED loudness blend.
    Brightness alone under-ranks sustained full-band climaxes on dynamic tracks
    (less bright but loud); dB is useless only on brick-limited masters. So blend
    in normalized loudness with a weight w set by the song's own loudness spread
    (IQR of RMS-dB): flat/compressed masters -> w≈0 (pure height, nothing
    regresses); dynamic masters -> w->1 (loud climaxes lift the score). The
    /(1+w) keeps the blend in [0,1]. No cross-song absolutes."""
    height_n = _robust_norm(height_raw)
    loud_n = _robust_norm(f.loudness_db)
    iqr_db = float(np.percentile(f.loudness_db, 75) - np.percentile(f.loudness_db, 25))
    w = float(np.clip((iqr_db - 3.0) / 6.0, 0.0, 1.0))
    return (height_n + w * loud_n) / (1.0 + w)


def recipe_c(f: Features) -> list[tuple[float, float, str]]:
    if len(f.beat_times) < 8:
        return [(0.0, f.duration, "ambient")]
    height_raw, lrms_beat, mfcc_beat, moddepth = _recipe_c_features(f)
    energy = _recipe_c_energy(f, height_raw)         # per-beat, gated-loudness blend
    depth_n = _robust_norm(moddepth)                 # per-beat modulation depth [0,1]
    # Combined per-beat signal: energy + a modulation-depth boost that rescues
    # burst trains (median energy sits in the gaps; depth is high). Centers AND
    # segment scores both come from THIS signal, so they share a scale (the
    # baseline groove -> normal; only genuinely elevated segments -> intense).
    combined = energy + 0.6 * depth_n

    bounds = _recipe_c_boundaries(height_raw, lrms_beat, mfcc_beat, moddepth)
    seg_edges = list(zip(bounds, bounds[1:] + [len(f.beat_times)]))

    # Centers calibrated on the SECTION-scale statistic (rolling-p80 of combined
    # over 16 beats, n~600, time-weighted) so they share a scale with the p80
    # segment scores. Fitting on raw per-beat combined instead leaves slow/ambient
    # unreachable on burst-texture tracks (p80 rides above the per-beat mode).
    centers, _ = _mode_anchored_centers(_rolling_p80(combined, 16))
    tiers_by_beat = ["normal"] * len(f.beat_times)
    for s0, s1 in seg_edges:
        if s1 <= s0:
            continue
        # Score by p80, NOT median (waveform-eng): "how bright does this section
        # GET" rewards a segment that spikes even if calm on average, and the
        # percentile doubles as a duty-cycle floor (a burst train firing ~30% of
        # its beats reads intense; a lone spike in a quiet span stays quiet).
        score = float(np.percentile(combined[s0:s1], 80))
        tier = TIERS[int(np.argmin((score - centers) ** 2))]
        for i in range(s0, s1):
            tiers_by_beat[i] = tier
    return beats_to_intervals(f.beat_times, tiers_by_beat, f.duration)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for raw in argv:
        path = Path(raw).expanduser()
        if not path.is_file():
            print(f"!! not a file: {path}")
            continue
        print(f"\n########## {path.name}")
        f = load_features(path)
        print(
            f"  {f.duration:.1f}s, {f.bpm:.1f} BPM, {len(f.beat_times)} beats  "
            f"(loud {f.loudness_db.min():.0f}..{f.loudness_db.max():.0f} dBFS)"
        )
        print(energy_sparkline(f))
        for name, fn in (("A (per-beat)", recipe_a), ("C (segment+p80)", recipe_c)):
            try:
                intervals = fn(f)
            except Exception as e:  # noqa: BLE001 - harness: report and continue
                print(f"── {name} ── ERROR: {e!r}")
                continue
            print(Summary(name, intervals, f.duration, f.bpm).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
