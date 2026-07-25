"""Beat detection and timeline construction for music-synced timelapses.

Builds a list of (photo_index, duration_seconds) segments aligned to beats
detected in an audio file. Ambient (low-onset) regions fall back to a fixed
duration. See GitHub issue #26 for the design.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Segment:
    """One photo held for a specific duration."""
    photo_index: int
    duration: float  # seconds
    tier: str = "normal"  # "slow" | "normal" | "intense" | "ambient"


@dataclass
class BeatTimeline:
    segments: list[Segment]
    bpm: float
    subdivision: float  # photos per beat (e.g. 0.5, 1, 2, 4)
    beat_synced_regions: list[tuple[float, float]]  # (start, end) seconds
    ambient_regions: list[tuple[float, float]]
    total_duration: float
    photos_kept: int
    photos_dropped: int
    photos_held_short: int  # transitions filled by holding (fewer photos than transitions)
    selection_per_month: list[tuple[str, int, int]]  # (label, kept, available)
    audio_trimmed_seconds: float = 0.0  # how much trailing audio got cut to fit photos
    bounds_violation: str | None = None  # "ceiling" or "floor" if subdivision violated a bound
    intense_regions: list[tuple[float, float]] = field(default_factory=list)
    slow_regions: list[tuple[float, float]] = field(default_factory=list)
    intense_multiplier: float = 1.0
    slow_multiplier: float = 1.0
    beat_times: list[float] = field(default_factory=list)  # detected beat onsets, seconds
    # Spans where the grid was fiction and cuts followed real note strikes.
    onset_anchor_spans: list[tuple[float, float]] = field(default_factory=list)
    onset_strikes: list[float] = field(default_factory=list)  # the strike times used there

    def metronome_times(self) -> list[float]:
        """Times the metronome dot should flash: detected beats where the grid
        holds, but the real note STRIKES inside onset-anchor spans (so the dot
        marks what we actually cut on, not a fictional grid)."""
        if not self.onset_anchor_spans:
            return list(self.beat_times)

        def _in_span(t: float) -> bool:
            return any(a <= t < b for a, b in self.onset_anchor_spans)

        out = [t for t in self.beat_times if not _in_span(t)]
        out += [t for t in self.onset_strikes if _in_span(t)]
        return sorted(out)

    def _pacing_intervals(self) -> list[tuple[float, float, str]]:
        return _pacing_intervals_impl(
            self.total_duration,
            self.beat_synced_regions,
            self.ambient_regions,
            self.intense_regions,
            self.slow_regions,
            self.subdivision,
            self.intense_multiplier,
            self.slow_multiplier,
        )

    def summary(self) -> str:
        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            return f"{m}:{s:02d}"

        def fmt_regions(regions: list[tuple[float, float]], cap: int = 5) -> str:
            if not regions:
                return "(none)"
            shown = ", ".join(f"{fmt(a)}–{fmt(b)}" for a, b in regions[:cap])
            if len(regions) > cap:
                shown += f", … (+{len(regions) - cap} more)"
            return shown

        regions_str = fmt_regions(self.beat_synced_regions)
        amb_str = fmt_regions(self.ambient_regions)

        sub_label = _subdivision_label(self.subdivision)
        lines = [
            f"Audio: {self.bpm:.1f} BPM detected.",
            f"Subdivision: {sub_label} ({self.subdivision} photos/beat).",
            f"Beat-synced: {regions_str}.",
            f"Ambient/fallback: {amb_str}.",
            f"Timeline: {len(self.segments)} segment(s), total {self.total_duration:.1f}s.",
        ]
        if self.intense_regions:
            lines.append(
                f"Intense (×{self.intense_multiplier:g}): "
                f"{fmt_regions(self.intense_regions)}."
            )
        if self.slow_regions:
            lines.append(
                f"Slow (×{self.slow_multiplier:g}): "
                f"{fmt_regions(self.slow_regions)}."
            )

        # Unified pacing timeline so the user can scrub the video and know
        # what to expect when. Higher-priority overlay wins per region.
        timeline_intervals = self._pacing_intervals()
        if timeline_intervals:
            lines.append("Pacing timeline:")
            # Walk segments to compute per-interval counts and rates. A segment
            # belongs to the interval its start time falls into.
            seg_starts: list[float] = []
            t = 0.0
            for s in self.segments:
                seg_starts.append(t)
                t += s.duration
            for start, end, label in timeline_intervals:
                count = 0
                dur_sum = 0.0
                for i, st in enumerate(seg_starts):
                    if st >= end - 1e-6:
                        break
                    if st >= start - 1e-6:
                        count += 1
                        dur_sum += self.segments[i].duration
                rate = count / dur_sum if dur_sum > 0 else 0.0
                region_durs = [
                    self.segments[i].duration
                    for i, st in enumerate(seg_starts)
                    if start - 1e-6 <= st < end - 1e-6
                ]
                if region_durs:
                    max_d = max(region_durs)
                    min_rate = 1.0 / max_d if max_d > 0 else 0.0
                    extreme = (
                        f", slowest {min_rate:.2f}/s ({max_d:.1f}s/photo)"
                        if max_d > 2 * (sum(region_durs) / len(region_durs))
                        else ""
                    )
                else:
                    extreme = ""
                lines.append(
                    f"  • {fmt(start)}–{fmt(end)}  {label}  "
                    f"[{count} seg, {rate:.2f}/s{extreme}]"
                )
        if self.photos_dropped:
            lines.append(
                f"Photo selection: kept {self.photos_kept}, dropped "
                f"{self.photos_dropped} (weighted by month)."
            )
        if self.photos_held_short:
            lines.append(
                f"Note: {self.photos_held_short} trailing transition(s) re-used "
                f"the last photo (not enough photos for the beat grid)."
            )
        if self.audio_trimmed_seconds > 0:
            lines.append(
                f"Warning: too few photos to fill the song at the chosen "
                f"min-photos-per-beat floor; trimming {self.audio_trimmed_seconds:.1f}s "
                f"of trailing audio. Lower --min-photos-per-beat to stretch."
            )
        return "\n".join(lines)


def _pacing_intervals_impl(
    total_duration: float,
    beat_synced: list[tuple[float, float]],
    ambient: list[tuple[float, float]],
    intense: list[tuple[float, float]],
    slow: list[tuple[float, float]],
    base_subdivision: float,
    intense_multiplier: float,
    slow_multiplier: float,
) -> list[tuple[float, float, str]]:
    """Sample 0..total_duration at 0.05s and classify each tick by priority.

    Priority (highest wins): intense > slow > beat-synced (normal) > ambient.
    Emit contiguous runs as (start, end, label).
    """
    if total_duration <= 0:
        return []

    def in_any(t: float, regions: list[tuple[float, float]]) -> bool:
        for a, b in regions:
            if a - 1e-6 <= t <= b + 1e-6:
                return True
        return False

    def label_at(t: float) -> str:
        if in_any(t, intense):
            return f"intense (×{intense_multiplier:g})"
        if in_any(t, slow):
            return f"slow (×{slow_multiplier:g})"
        if in_any(t, beat_synced):
            return "normal"
        # "Uncovered" gaps (tiny slivers between region boundaries) collapse
        # into ambient — structurally the same: nothing notable happening.
        return "ambient (no clear beat)"

    step = 0.05
    n = int(total_duration / step) + 1
    out: list[tuple[float, float, str]] = []
    cur_start = 0.0
    cur_label = label_at(0.0)
    for i in range(1, n):
        t = i * step
        lbl = label_at(t)
        if lbl != cur_label:
            out.append((cur_start, t, cur_label))
            cur_start = t
            cur_label = lbl
    out.append((cur_start, total_duration, cur_label))
    # Filter out very short noise intervals (<0.5s)
    return [iv for iv in out if iv[1] - iv[0] >= 0.5]


def _subdivision_label(s: float) -> str:
    table = {4.0: "1/16 note", 2.0: "1/8 note", 1.0: "1/4 note (on the beat)",
             0.5: "1/2 note (every 2nd beat)", 0.25: "whole note (every 4th beat)"}
    return table.get(s, f"{s}x beat")


# --- Track progression model ----------------------------------------------
#
# A linear, time-indexed "sheet music" for a track: the sequence of pacing
# states (ambient/slow/normal/intense) the song passes through, plus a few
# sanity metrics that let an agent judge whether the pacing the tool landed on
# is reasonable *without rendering a video*. Warn-only: metrics never block a
# render, they just surface ⚠ flags in the text/JSON emit.

# Thresholds for the warn-only sanity flags. Declarative and tunable — a bad
# pacing param should make one of these fire in the text emit before you burn
# a render. See CLAUDE.md "favor false negatives": these lean toward flagging.
PEAK_PPS_WARN = 5.0        # photos/sec in the worst 1s window above this reads as frantic
SHORTEST_HOLD_WARN = 0.10  # a photo held under this (100ms) flickers rather than registers
MIN_STATE_RUN_WARN = 3.0   # mean tier-run shorter than this = choppy, indecisive tiering


def _label_to_tier(label: str) -> str:
    """Collapse a pacing-interval label ('intense (×2)', 'ambient (no clear
    beat)') to its base tier for the progression model."""
    if label.startswith("intense"):
        return "intense"
    if label.startswith("slow"):
        return "slow"
    if label.startswith("normal"):
        return "normal"
    return "ambient"


@dataclass
class ProgressionState:
    """One contiguous stretch of the song in a single pacing tier."""
    start: float
    end: float
    tier: str  # "slow" | "normal" | "intense" | "ambient"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackProgression:
    """Linear progression of pacing states over a track, with sanity metrics.

    Built from a :class:`BeatTimeline`. ``states`` is the "sheet music"; the
    metrics below summarize both the tier structure (how the song is carved up)
    and the photo cadence (how fast frames actually change), so pacing can be
    judged from text alone.
    """
    states: list[ProgressionState]
    total_duration: float
    bpm: float
    # tier-structure metrics
    transition_count: int          # number of tier changes (len(states) - 1)
    mean_state_run: float          # average state duration, seconds
    shortest_run: float            # shortest state duration, seconds
    time_fraction: dict[str, float]  # tier -> fraction of total_duration
    # photo-cadence metrics (the "2-3 per second is unreasonable" concern)
    photos_per_sec_mean: float
    photos_per_sec_peak: float     # max transitions in any 1s window
    shortest_hold: float           # shortest single-photo hold, seconds
    warnings: list[str]

    @classmethod
    def from_timeline(cls, timeline: BeatTimeline) -> TrackProgression:
        # States: reuse the same priority-ranked interval computation the
        # printed pacing timeline uses, collapsed to base tiers and merged
        # across any same-tier neighbors.
        intervals = timeline._pacing_intervals()
        states: list[ProgressionState] = []
        for start, end, label in intervals:
            tier = _label_to_tier(label)
            if states and states[-1].tier == tier:
                states[-1].end = end
            else:
                states.append(ProgressionState(start, end, tier))

        total = timeline.total_duration
        runs = [s.duration for s in states]
        transition_count = max(0, len(states) - 1)
        mean_state_run = (sum(runs) / len(runs)) if runs else 0.0
        shortest_run = min(runs) if runs else 0.0

        time_fraction: dict[str, float] = {}
        if total > 0:
            for s in states:
                time_fraction[s.tier] = time_fraction.get(s.tier, 0.0) + s.duration / total

        # Photo cadence from the actual segment durations.
        durations = [seg.duration for seg in timeline.segments]
        seg_starts: list[float] = []
        acc = 0.0
        for d in durations:
            seg_starts.append(acc)
            acc += d
        shortest_hold = min(durations) if durations else 0.0
        photos_per_sec_mean = (len(durations) / total) if total > 0 else 0.0
        # Peak: slide a 1s window across the segment-start times and take the
        # densest count. Starts are sorted, so a two-pointer sweep suffices.
        peak = 0
        j = 0
        for i, st in enumerate(seg_starts):
            while seg_starts[i] - seg_starts[j] > 1.0:
                j += 1
            peak = max(peak, i - j + 1)
        photos_per_sec_peak = float(peak)

        warnings: list[str] = []
        if photos_per_sec_peak > PEAK_PPS_WARN:
            warnings.append(
                f"peak cadence {photos_per_sec_peak:.0f}/s in the busiest 1s window "
                f"(> {PEAK_PPS_WARN:.0f}/s tends to read as frantic)"
            )
        if durations and shortest_hold < SHORTEST_HOLD_WARN:
            warnings.append(
                f"shortest photo hold {shortest_hold * 1000:.0f}ms "
                f"(< {SHORTEST_HOLD_WARN * 1000:.0f}ms flickers rather than registers)"
            )
        if len(states) > 1 and mean_state_run < MIN_STATE_RUN_WARN:
            warnings.append(
                f"{transition_count} tier changes, mean run {mean_state_run:.1f}s "
                f"(< {MIN_STATE_RUN_WARN:.0f}s = choppy, indecisive tiering)"
            )

        return cls(
            states=states,
            total_duration=total,
            bpm=timeline.bpm,
            transition_count=transition_count,
            mean_state_run=mean_state_run,
            shortest_run=shortest_run,
            time_fraction=time_fraction,
            photos_per_sec_mean=photos_per_sec_mean,
            photos_per_sec_peak=photos_per_sec_peak,
            shortest_hold=shortest_hold,
            warnings=warnings,
        )

    def to_dict(self) -> dict:
        return {
            "total_duration": round(self.total_duration, 3),
            "bpm": round(self.bpm, 2),
            "states": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "duration": round(s.duration, 3),
                    "tier": s.tier,
                }
                for s in self.states
            ],
            "metrics": {
                "transition_count": self.transition_count,
                "mean_state_run": round(self.mean_state_run, 3),
                "shortest_run": round(self.shortest_run, 3),
                "photos_per_sec_mean": round(self.photos_per_sec_mean, 3),
                "photos_per_sec_peak": round(self.photos_per_sec_peak, 3),
                "shortest_hold": round(self.shortest_hold, 3),
                "time_fraction": {k: round(v, 3) for k, v in self.time_fraction.items()},
            },
            "warnings": list(self.warnings),
        }

    def render_text(self) -> str:
        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            return f"{m}:{s:02d}"

        lines = [
            f"Track progression ({fmt(self.total_duration)}, {self.bpm:.1f} BPM):",
        ]
        for s in self.states:
            bar_frac = s.duration / self.total_duration if self.total_duration else 0.0
            bar = "█" * max(1, round(bar_frac * 40))
            lines.append(
                f"  {fmt(s.start):>4}–{fmt(s.end):<4} {s.tier:<8} "
                f"{s.duration:5.1f}s {bar}"
            )
        frac = ", ".join(
            f"{k} {v * 100:.0f}%"
            for k, v in sorted(self.time_fraction.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"  Metrics: {self.transition_count} tier changes, "
            f"mean run {self.mean_state_run:.1f}s, shortest {self.shortest_run:.1f}s"
        )
        lines.append(
            f"           cadence {self.photos_per_sec_mean:.2f}/s avg, "
            f"{self.photos_per_sec_peak:.0f}/s peak, "
            f"shortest hold {self.shortest_hold * 1000:.0f}ms"
        )
        lines.append(f"           time in tier: {frac}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        if not self.warnings:
            lines.append("  ✓ no pacing sanity flags")
        return "\n".join(lines)


def _detect_beats(
    audio_path: Path,
    *,
    tier_lead_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run librosa beat tracking.

    Returns (beat_times, beat_strengths, loudness_causal, loudness_anticausal,
    bpm, audio_duration).
    - beat_strengths: per-beat raw onset amplitude (for ambient classification
      and transient-snap refinement of section boundaries).
    - loudness_causal: ~4s trailing-window median of onset envelope, sampled
      at beat positions. Used to find when a loud section *begins* — rises
      only after the section is genuinely under way.
    - loudness_anticausal: ~4s forward-window median, sampled at beats. Used
      to find when a loud section *ends* — falls as soon as the section is
      about to be over (window starts filling with quiet).
    """
    import librosa  # lazy: heavy import
    from scipy.ndimage import median_filter

    y, sr = librosa.load(str(audio_path), mono=True)
    audio_duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_frames) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            float(np.atleast_1d(tempo)[0]),
            audio_duration,
        )

    # Per-beat instantaneous strength (for ambient classification).
    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    if strengths.max() > 0:
        strengths = strengths / strengths.max()

    # Smoothed loudness envelope for section-level accent detection.
    # CAUSAL median filter (~4s trailing window): at frame i, the smoothed
    # value reflects only past frames [i-w+1, i]. A symmetric filter leaks
    # ~half-window of future loudness backward in time, which would trip
    # the threshold up to 2s before the section actually begins — a full
    # measure early at typical dance tempos. Causal smoothing only rises
    # once the new section is genuinely under way.
    env_norm = librosa.util.normalize(onset_env)
    hop_length = 512  # librosa default
    window_seconds = 4.0
    window_frames = max(1, int(round(window_seconds * sr / hop_length)))
    if window_frames % 2 == 0:
        window_frames += 1
    # scipy median_filter's `origin` shifts the kernel: with size=w,
    # origin=+w//2 anchors the right edge of the window on the current
    # sample → fully causal (trailing window, [i-w+1, i]).
    # origin=-w//2 anchors the left edge → anticausal (leading window, [i, i+w-1]).
    env_causal = median_filter(env_norm, size=window_frames, origin=window_frames // 2, mode="nearest")
    env_anticausal = median_filter(env_norm, size=window_frames, origin=-(window_frames // 2), mode="nearest")
    if env_causal.max() > 0:
        env_causal = env_causal / env_causal.max()
    if env_anticausal.max() > 0:
        env_anticausal = env_anticausal / env_anticausal.max()

    # Optional look-ahead: shift sampling forward in time so the smoothed
    # value at a beat reflects upcoming loudness, not the surrounding span.
    lead_frames = int(round(tier_lead_seconds * sr / hop_length))
    sampled_idx = np.clip(beat_frames + lead_frames, 0, len(env_causal) - 1)
    loudness_causal = env_causal[sampled_idx]
    loudness_anticausal = env_anticausal[sampled_idx]

    return (
        beat_times,
        strengths,
        loudness_causal,
        loudness_anticausal,
        float(np.atleast_1d(tempo)[0]),
        audio_duration,
    )


# --------------------------------------------------------------------------- #
# Occupancy-driven base pace ("--base-pace occupancy")
#
# The base (normal-tier) photo density should match the SONG, not the photo
# count: a sparse ballad should linger, a dense track should drive. The owner's
# own signal — "how much black is in the spectrogram" — measures this directly
# (spectral occupancy). Two hard rules from the audio-engineer consult:
#  * Denominate OCTAVE-FREE. The detected-tempo scalar (e.g. a 143.55 BPM bin)
#    is a per-track coin flip on the octave, so pace never multiplies it. We use
#    median inter-beat-interval for the fine grid, then convert a wall-clock
#    target (photos/sec) to the nearest EVEN beats-per-photo. An even count on a
#    doubled grid is a whole count on the felt grid, so the schedule stays
#    musical under either octave hypothesis; odd counts read mechanical.
#  * Occupancy is the pipeline's only CROSS-song absolute, so it must be
#    level-normalized: perceptual-weight first (inaudible sub-bass shouldn't
#    fill cells), then threshold relative to the song's own p95 (mastering
#    loudness shouldn't bias it).
# Ladder + clamp are declarative so the mapping is visible, not hidden in code.
# --------------------------------------------------------------------------- #

# occupancy < edge -> target photos/sec at the normal tier. Coarse by design
# (a fitted curve on a handful of tracks is overfitting); bands are musical.
# occupancy < edge -> target photos/sec. Even-beat snapping is coarse (at ~143
# BPM most tracks land on 'every 6' = 0.40), so the bands mainly separate the
# EXTREMES: near-silent ambient lingers (every 8), wall-of-sound drives (every
# 4). Owner calibration: TBAH (occ 0.37) reads too slow at 0.30/every-8 and
# right at 0.40/every-6 — so the broad middle targets 0.40.
_OCCUPANCY_PACE_LADDER: tuple[tuple[float, float], ...] = (
    (0.25, 0.30),   # very sparse (near-silent / ambient) -> every 8, lingering
    (0.85, 0.40),   # normal range (ballad..dense) -> every 6  (owner's split)
    (2.00, 0.55),   # very dense (wall-of-sound) -> every 4, driving
)
_NORMAL_PPS_CLAMP = (0.15, 1.0)  # guard: don't strobe or freeze on outliers


def _spectral_occupancy(y: np.ndarray, sr: int, hop: int = 512) -> float:
    """Fraction of the (perceptually-weighted) spectrogram that is 'lit',
    thresholded relative to the song's own p95 level. High = dense/busy (little
    black) -> faster base; low = sparse (much black) -> slower base."""
    import librosa

    S = np.abs(librosa.stft(y, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    Sw = librosa.perceptual_weighting(S, freqs)  # dB, de-emphasizes inaudible LF
    p95 = float(np.percentile(Sw, 95))
    return float((Sw > (p95 - 30.0)).mean())


def _felt_tier_gaps() -> dict[str, int]:
    """The pace ladder, as a whole number of FELT beats per photo.

    A fixed metronomic subdivision that halves at each rung — the owner's model
    (2026-07-25): "Intense is 1:1 (photo:beat), Normal could be 1:2 and Slow 1:4."
    A felt beat is two raw detected beats, so on the raw grid this reads as
    intense every beat, normal every 2nd, slow every 4th:

      * intense -> every raw beat (1:1). NOT read from here; _felt_locked_cut_
        indices special-cases it, and it is the one tier that lands on both beat
        parities. The dict has no intense entry for that reason.
      * normal  -> 1 felt beat  (1:2 raw)
      * slow    -> 2 felt beats (1:4 raw)
      * ambient -> only used for the rare ambient beat that falls on the grid;
        in its own (sparse/rubato) spans ambient stays strike-driven, untouched.

    This replaced an adaptive ladder (spectral occupancy + even-gap rounding +
    bar-multiple snapping, issue #46). The owner wanted a pulse they could feel
    matched across tiers ("Normal and Slow don't seem to be metronome matched
    like Intense is"), not a per-song-tuned rate, so the --intense/--slow-
    multiplier knobs no longer affect the grid rate."""
    return {"normal": 1, "slow": 2, "ambient": 2}


def _felt_locked_cut_indices(
    n: int,
    intense_mask: np.ndarray,
    slow_mask: np.ndarray,
    ambient_mask: np.ndarray,
    parity: int,
) -> list[int]:
    """Beat indices to cut on, felt-locked to a fixed metronomic subdivision per
    tier (see _felt_tier_gaps). A "felt beat" is a beat on the felt-downbeat
    parity (`parity`); a raw detected beat is half of one.

      * intense -> every DETECTED beat (parity AND off-parity), so 1:1 on the raw
        grid. The peak tier's job is drive: at a climax the dense off-parity
        landings read as momentum, not as weak-beat wander (owner, 2026-07-25).
      * normal  -> every felt beat (gap 1), i.e. 1:2 on the raw grid.
      * slow / ambient -> every 2nd felt beat (gap 2), i.e. 1:4 on the raw grid.

    Every gap here is even in RAW-beat terms (intense=1 raw is the sole exception,
    and it deliberately covers both parities), so a non-intense cut never wanders
    onto the off-parity beat. Pure and deterministic; exercised by
    tests/test_felt_lock.py."""
    gaps = _felt_tier_gaps()
    g_normal, g_slow = gaps["normal"], gaps["slow"]

    felt_idx = -1
    last_emit: int | None = None
    idxs: list[int] = []
    for k in range(n):
        on_parity = (k % 2) == parity
        if on_parity:
            felt_idx += 1
        # Intense cuts on every detected beat — parity and off-parity alike — so
        # the peak tier runs at double the every-felt-beat rate (see docstring).
        if intense_mask[k]:
            idxs.append(k)
            if on_parity:
                last_emit = felt_idx
            continue
        if not on_parity:
            continue
        if slow_mask[k] or ambient_mask[k]:
            g = g_slow
        else:
            g = g_normal
        if last_emit is None or (felt_idx - last_emit) >= g:
            idxs.append(k)
            last_emit = felt_idx
    return idxs


def _drop_opening_flash(
    transitions: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """Suppress an "out of the gate" cut. The timeline always starts a segment at
    t=0, but the first real beat can land almost immediately (librosa's first
    detected beat is often <1s in), so photo 1 would only flash before the actual
    cadence begins. If that opening hold is much shorter than the next one (<half),
    drop the first cut so photo 1 rides to the following cut — the owner would
    rather the opening photo linger than see an instant flip. Pure; exercised by
    tests/test_felt_lock.py."""
    if len(transitions) >= 3 and transitions[0][0] < 1e-3:
        lead = transitions[1][0] - transitions[0][0]
        nxt = transitions[2][0] - transitions[1][0]
        if nxt > 0 and lead < 0.5 * nxt:
            return transitions[:1] + transitions[2:]
    return transitions


# --------------------------------------------------------------------------- #
# Onset-anchored cuts for rubato / sparse sections
#
# On a sparse rubato passage (a solo piano intro, a breakdown) librosa's beat
# grid is fiction: the tempo prior fills a flat onset autocorrelation, so the
# metronome lands hundreds of ms off the actual note attacks. Below ~a dozen
# events the EVENTS ARE THE PULSE — so in those spans we abandon the grid and
# cut on the real onset strikes instead. Regime is decided per region by GRID
# SUPPORT (fraction of felt beats with a prominent strike within 70ms); a span
# scoring below _ONSET_ANCHOR_THRESH for at least _ONSET_ANCHOR_MIN_SPAN seconds
# switches to onset-anchoring. Inside such a span, only AMBIENT stretches follow
# the notes (see _NOTE_DRIVEN_TIERS) — that cut-on-a-chord-and-hold character is
# the part the owner rated 5/5. Intense, normal and slow keep the felt-grid pulse
# even here: the owner wants them "metronome matched like Intense is"
# (2026-07-25), so the metronomic grid cuts are retained through the span rather
# than replaced by note-following ones.
# (Design: audio-engineer consult, agent-chat 'audio' Part 5; issues #46, #48.)
# --------------------------------------------------------------------------- #

_GRID_SUPPORT_WINDOW_S = 0.070   # a felt beat "supports" the grid if a strike is this close
_ONSET_ANCHOR_THRESH = 0.30      # grid support below this -> the grid is fiction here
_ONSET_ANCHOR_MIN_SPAN = 8.0     # min span seconds, so the regime can't flap
# A song whose OVERALL grid support clears this is fundamentally grid-locked
# (a driving four-on-floor track): local dips are breakdowns/filter sweeps that
# still feel beat-driven, so we never onset-anchor them. Only songs that are
# overall ambiguous/rubato (a ballad with a sparse intro) get per-section anchoring.
_ONSET_ANCHOR_SONG_GATE = 0.65
_STRIKE_COALESCE_S = 0.40        # merge strikes closer than this (keep the louder)

# A photo held under this reads as a flash, not an image you saw. We shipped a
# 50ms frame as a bug once; per the project's guiding star, a cut that can't
# clear the floor is DROPPED (the previous photo lingers) rather than spent on a
# frame nobody registers.
_MIN_HOLD_S = 0.100
# Tiers that follow the note strikes instead of the felt grid, as
# {tier: notes-per-photo}. slow/ambient LINGER — one photo every Nth prominent
# note, the cut-on-a-chord-and-hold character the owner rated 5/5. Used for
# ambient inside its spans (see _NOTE_DRIVEN_TIERS) and, when a song has no felt
# grid at all, as the fallback pace for every tier.
_STRIKE_STRIDE = {"slow": 4, "ambient": 2}
# Tiers that stay strike-driven even when a felt grid IS available. Only ambient:
# the owner wants intense/normal/slow metronome-locked to the grid everywhere,
# but the sparse ambient passages read best following the actual notes.
_NOTE_DRIVEN_TIERS = frozenset({"ambient"})


def _prominent_strikes(
    y: np.ndarray, sr: int, *, prom_frac: float = 0.30, rolling_s: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak-picked onset attacks (the note/chord strikes), returned as
    (times, heights). Height is read at the PEAK; a strike survives when it
    exceeds prom_frac of the local-p90 envelope (drops ornaments, keeps chords).

    Times are the ENVELOPE PEAK, not librosa's backtracked attack start. We used
    to backtrack, on the theory that the foot of the rise is where the note
    begins -- but that put cuts a median 46ms (p90 93ms) BEFORE the audible
    attack, and on review the picture visibly led the sound: "the image flipping
    is a bit early, making the effect not feel connected". Audio-visual
    tolerance is strongly asymmetric -- picture leading sound is objectionable
    from roughly 15ms, picture lagging is fine to about 45ms -- so landing on
    the peak, and erring late if at all, is the safe side to miss on.

    NOT coalesced: this is the raw strike set used to MEASURE grid support, and
    at a fast tempo (144 BPM = a beat every 0.42s) merging near-together strikes
    would erase real consecutive kicks and make a grid track look rubato.
    Coalescing a chord/grace pair happens only in the cut path (see
    _coalesce_strikes), inside sparse onset-anchor spans."""
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=False)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    hop = 512
    w = max(1, int(rolling_s * sr / hop))
    times, heights = [], []
    for pk in peaks:
        lo, hi = max(0, pk - w), min(len(env), pk + w)
        p90 = float(np.percentile(env[lo:hi], 90))
        h = float(env[min(pk, len(env) - 1)])
        if h >= prom_frac * p90:
            times.append(float(librosa.frames_to_time(pk, sr=sr)))
            heights.append(h)
    return np.asarray(times), np.asarray(heights)


def _coalesce_strikes(
    times: np.ndarray, heights: np.ndarray, min_gap: float = _STRIKE_COALESCE_S,
) -> np.ndarray:
    """Merge strikes closer than min_gap, keeping the louder of each pair, so an
    every-strike tier can't double-cut a fast chord/grace pair. Applied only
    inside onset-anchor spans, where strikes are already sparse."""
    t = np.asarray(times, dtype=float)
    if len(t) == 0:
        return t
    h = np.asarray(heights, dtype=float) if len(heights) else np.ones_like(t)
    order = np.argsort(t)
    t, h = t[order], h[order]
    keep_t, keep_h = [t[0]], [h[0]]
    for i in range(1, len(t)):
        if t[i] - keep_t[-1] < min_gap:
            if h[i] > keep_h[-1]:
                keep_t[-1], keep_h[-1] = t[i], h[i]
        else:
            keep_t.append(t[i])
            keep_h.append(h[i])
    return np.asarray(keep_t)


def _grid_support(felt_beats: np.ndarray, strikes: np.ndarray,
                  window_s: float = _GRID_SUPPORT_WINDOW_S) -> float:
    """Fraction of felt beats with a strike within +/- window_s. 1.0 = grid sits
    on the music, ~0 = the grid is fiction (rubato / sparse)."""
    if len(felt_beats) == 0 or len(strikes) == 0:
        return 0.0
    strikes = np.asarray(strikes)
    hits = sum(1 for b in felt_beats if np.min(np.abs(strikes - b)) <= window_s)
    return hits / len(felt_beats)


def _onset_anchor_spans(
    felt_beats: np.ndarray, strikes: np.ndarray, duration: float, *,
    thresh: float = _ONSET_ANCHOR_THRESH, min_span_s: float = _ONSET_ANCHOR_MIN_SPAN,
    win_s: float = 8.0, step_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Time spans where the beat grid is fiction and cuts should follow strikes.
    Slides a window computing grid support, keeps runs below `thresh`, and returns
    those lasting at least `min_span_s` (so the regime can't flap segment to
    segment). Pure; exercised by tests/test_onset_anchor.py."""
    felt_beats = np.asarray(felt_beats, dtype=float)
    if duration <= 0 or len(felt_beats) == 0:
        return []
    centers, low = [], []
    t = 0.0
    while t < duration:
        fb = felt_beats[(felt_beats >= t) & (felt_beats < t + win_s)]
        centers.append(t + win_s / 2.0)
        low.append(len(fb) > 0 and _grid_support(fb, strikes) < thresh)
        t += step_s
    # Merge consecutive low windows into spans (bridge the window overlap).
    spans: list[tuple[float, float]] = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            t0 = max(0.0, centers[i] - win_s / 2.0)
            t1 = min(duration, centers[j] + win_s / 2.0)
            if t1 - t0 >= min_span_s:
                spans.append((t0, t1))
            i = j + 1
        else:
            i += 1
    # Runs are padded by ±win_s/2, so two runs whose centers sit within a window
    # of each other produce spans that overlap or touch. Coalesce them: callers
    # splice strikes per span, and a strike inside two spans would be cut twice
    # (duplicate transition times -> a 50ms flicker frame).
    merged: list[tuple[float, float]] = []
    for t0, t1 in spans:
        if merged and t0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t1))
        else:
            merged.append((t0, t1))
    return merged


def _tier_stretches(
    span: tuple[float, float], sel: list[float],
    tier_at: callable, ambient_default: str,
) -> list[tuple[tuple[float, float], str]]:
    """Split a span into contiguous single-tier stretches, bounded by strikes.

    A span can straddle a tier change, and the two tiers are scheduled by
    different rules (grid-snapped vs strike-driven), so they have to be handed
    off cleanly rather than interleaved."""
    t0, t1 = span
    tiers = [tier_at(s) or ambient_default for s in sel]
    out: list[tuple[tuple[float, float], str]] = []
    i = 0
    while i < len(tiers):
        j = i
        while j + 1 < len(tiers) and tiers[j + 1] == tiers[i]:
            j += 1
        end = sel[j + 1] if j + 1 < len(sel) else t1
        out.append(((sel[i], end), tiers[i]))
        i = j + 1
    return out


def _strike_stride_cuts(
    stretch: tuple[float, float], sel: list[float], tier: str
) -> list[tuple[float, str]]:
    """One photo every Nth prominent note — no grid, so gaps between notes
    LINGER and every cut is on an attack. Used for the lingering tiers, and for
    everything when there is no felt grid to schedule against (stride 1 there:
    with no rate to hold to, the notes are the only pace available)."""
    a, b = stretch
    stride = max(1, _STRIKE_STRIDE.get(tier, 1))
    inside = [s for s in sel if a - 1e-9 <= s < b - 1e-9]
    return [(s, tier) for s in inside[::stride]]


def _onset_anchor_cuts(
    span: tuple[float, float], strikes: np.ndarray,
    tier_at: callable, ambient_default: str = "ambient", *,
    has_grid: bool = False,
) -> list[tuple[float, str]]:
    """Strike-driven cut times inside an onset-anchor span.

    Only the note-driven stretches produce cuts here: ambient (see
    _NOTE_DRIVEN_TIERS), and — when the song has no felt grid at all — every
    tier, since there is then no metronomic pulse to fall back on. Intense,
    normal and slow keep their felt-grid cuts through the span; the splice retains
    those rather than replacing them, so they stay metronome-locked. Pure;
    exercised by tests/test_onset_anchor.py."""
    t0, t1 = span
    sel = [float(s) for s in np.asarray(strikes) if t0 - 1e-6 <= s < t1 + 1e-6]
    if not sel:
        return []
    out: list[tuple[float, str]] = []
    for stretch, tier in _tier_stretches(span, sel, tier_at, ambient_default):
        if tier in _NOTE_DRIVEN_TIERS or not has_grid:
            out.extend(_strike_stride_cuts(stretch, sel, tier))
        # else: the metronomic grid cuts are kept by the splice; emit nothing.
    # The stretch hand-offs can put two cuts within a frame of each other; a hold
    # nobody registers is a wasted photo, so drop the later one. The span edge
    # counts too: grid cutting resumes at t1, so a cut just inside it would flash.
    # Drop it and let the last photo run into the boundary.
    out = [(t, k) for t, k in out if t <= t1 - _MIN_HOLD_S + 1e-9]
    out.sort(key=lambda tk: tk[0])
    deduped: list[tuple[float, str]] = []
    for t, tier in out:
        if deduped and t - deduped[-1][0] < _MIN_HOLD_S - 1e-9:
            continue
        deduped.append((t, tier))
    return deduped


def _splice_onset_anchor(
    transitions: list[tuple[float, str]],
    spans: list[tuple[float, float]],
    strikes: np.ndarray,
    tier_at: callable,
    heights: np.ndarray | None = None,
    has_grid: bool = False,
) -> list[tuple[float, str]]:
    """Inside onset-anchor spans, replace the grid transitions of the NOTE-DRIVEN
    tiers with strike-anchored cuts, but KEEP the grid transitions of the
    metronome-locked tiers (intense/normal/slow) so their pulse rides through the
    span. When the song has no felt grid at all (`has_grid` False), every tier is
    note-driven and all in-span grid transitions are replaced. Strikes are
    coalesced PER SPAN (a fast chord/grace pair collapses to its louder hit)
    before the pace mapping. Pure; exercised by tests."""
    if not spans:
        return transitions
    strikes = np.asarray(strikes, dtype=float)
    heights = np.asarray(heights, dtype=float) if heights is not None else np.ones_like(strikes)

    def _in_span(t: float) -> bool:
        return any(t0 <= t < t1 for t0, t1 in spans)

    def _replaced(t: float, kind: str) -> bool:
        # A grid cut is dropped only where a strike-driven cut will take its place:
        # note-driven tiers, or everything when there is no grid to hold onto.
        return _in_span(t) and (not has_grid or kind in _NOTE_DRIVEN_TIERS)

    kept = [(t, k) for (t, k) in transitions if not _replaced(t, k)]
    for span in spans:
        t0, t1 = span
        m = (strikes >= t0 - 1e-6) & (strikes < t1 + 1e-6)
        span_strikes = _coalesce_strikes(strikes[m], heights[m])
        kept.extend(_onset_anchor_cuts(
            span, span_strikes, tier_at, has_grid=has_grid,
        ))
    kept.sort(key=lambda tk: tk[0])
    return kept


def _occupancy_base_subdivision(
    y: np.ndarray, sr: int, beat_times: np.ndarray
) -> tuple[float | None, float, float]:
    """Return (subdivision photos-per-beat, occupancy, actual photos/sec) for the
    normal tier, octave-free. subdivision is None if the grid is unusable."""
    bt = np.asarray(beat_times, dtype=float)
    ibi = np.diff(bt)
    if len(ibi) == 0:
        return None, 0.0, 0.0
    bps = 1.0 / float(np.median(ibi))  # fine-grid beats/sec from IBIs, not the scalar
    occ = _spectral_occupancy(y, sr)
    target = next(t for edge, t in _OCCUPANCY_PACE_LADDER if occ < edge)
    beats_per_photo = max(2, int(round((bps / target) / 2.0) * 2))  # nearest EVEN
    actual_pps = float(np.clip(bps / beats_per_photo, *_NORMAL_PPS_CLAMP))
    return 1.0 / beats_per_photo, occ, actual_pps


def _pick_subdivision(
    bpm: float,
    max_pps: float,
    min_ppb: float,
    override: float | None,
) -> tuple[float, str | None]:
    """Pick the largest subdivision (photos per beat) within the bounds.

    Returns (subdivision, violation) where violation is None on success, or
    "ceiling"/"floor" if no subdivision in the ladder satisfies both bounds
    (in which case we pick the closest-to-bounds option).
    """
    if override is not None:
        return override, None
    bps = bpm / 60.0
    ladder = (4.0, 2.0, 1.0, 0.5, 0.25)
    # Largest subdivision satisfying BOTH bounds.
    for s in ladder:
        if s >= min_ppb and bps * s <= max_pps:
            return s, None
    # No subdivision satisfies both. Pick best effort:
    # - If even the floor exceeds the ceiling at this BPM (bps * min_ppb > max_pps),
    #   prefer to exceed the ceiling at min_ppb — slower-than-strobe is more
    #   recoverable than violating the user's pace floor.
    if bps * min_ppb > max_pps:
        return min_ppb, "ceiling"
    # Otherwise the floor is the issue (e.g., min_ppb=1.0 but no s>=1 fits ceiling).
    # Pick the largest sub that fits the ceiling, even if below floor.
    for s in ladder:
        if bps * s <= max_pps:
            return s, "floor"
    return ladder[-1], "ceiling"


def _classify_regions(
    beat_times: np.ndarray,
    strengths: np.ndarray,
    audio_duration: float,
    threshold: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split [0, audio_duration] into beat-synced and ambient regions.

    A beat is "ambient" if its onset strength is below threshold. Runs of
    ambient beats become ambient regions. Pre-roll before the first usable
    beat and trailing tail after the last are also ambient.
    """
    if len(beat_times) == 0:
        return [], [(0.0, audio_duration)]

    usable = strengths >= threshold

    # Denoise: short runs of low-strength beats embedded in otherwise-usable
    # stretches shouldn't fragment the timeline. A real ambient section is
    # sustained (intro, bridge, outro). Promote short ambient runs back to
    # usable so we keep transitioning through quieter individual beats.
    MIN_AMBIENT_RUN = 8
    n = len(usable)
    smoothed = usable.copy()
    i = 0
    while i < n:
        if not smoothed[i]:
            j = i
            while j + 1 < n and not smoothed[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len < MIN_AMBIENT_RUN and i > 0 and j < n - 1:
                # Internal short run — promote to usable
                smoothed[i : j + 1] = True
            i = j + 1
        else:
            i += 1
    usable = smoothed

    beat_synced: list[tuple[float, float]] = []
    ambient: list[tuple[float, float]] = []

    # Pre-roll
    first_usable_idx = int(np.argmax(usable)) if usable.any() else len(usable)
    if first_usable_idx == len(usable):
        # No usable beats at all
        return [], [(0.0, audio_duration)]
    if beat_times[first_usable_idx] > 0.05:
        ambient.append((0.0, float(beat_times[first_usable_idx])))

    # Walk through beats grouping contiguous usable / ambient runs.
    i = first_usable_idx
    n = len(beat_times)
    while i < n:
        if usable[i]:
            j = i
            while j + 1 < n and usable[j + 1]:
                j += 1
            beat_synced.append((float(beat_times[i]), float(beat_times[j])))
            i = j + 1
        else:
            j = i
            while j + 1 < n and not usable[j + 1]:
                j += 1
            ambient.append((float(beat_times[i]), float(beat_times[min(j + 1, n - 1)])))
            i = j + 1

    # Tail after last beat
    last_t = float(beat_times[-1])
    if audio_duration - last_t > 0.1:
        ambient.append((last_t, audio_duration))

    return beat_synced, ambient


def _transitions_in_region(
    beat_times: np.ndarray,
    strengths: np.ndarray,
    region: tuple[float, float],
    subdivision: float,
    threshold: float,
    intense_mask: np.ndarray | None = None,
    slow_mask: np.ndarray | None = None,
    intense_multiplier: float = 1.0,
    slow_multiplier: float = 1.0,
) -> list[float]:
    """Generate transition times within a beat-synced region.

    Each beat picks one of three multipliers based on per-beat masks:
    intense (loud sections), slow (quiet sections), or normal. The
    multiplier scales `subdivision` for that beat's gap. Interpolated
    frames are anchored between adjacent librosa beat timestamps so
    every transition stays on-grid.
    """
    start, end = region
    in_region = (beat_times >= start - 1e-6) & (beat_times <= end + 1e-6)
    region_beats = beat_times[in_region]
    if len(region_beats) == 0:
        return []
    nb = len(region_beats)
    region_intense = intense_mask[in_region] if intense_mask is not None else np.zeros(nb, dtype=bool)
    region_slow = slow_mask[in_region] if slow_mask is not None else np.zeros(nb, dtype=bool)

    def _local_sub(k: int) -> float:
        # Intense wins over slow if both somehow flagged.
        if region_intense[k]:
            return subdivision * intense_multiplier
        if region_slow[k]:
            return subdivision * slow_multiplier
        return subdivision

    times: list[float] = []
    phase = 0.0
    for k in range(nb):
        local = _local_sub(k)
        phase += local
        n_emits = int(phase + 1e-9)
        if n_emits > 0:
            gap = (
                region_beats[k + 1] - region_beats[k]
                if k + 1 < nb
                else (region_beats[k] - region_beats[k - 1] if k > 0 else 0.5)
            )
            for j in range(n_emits):
                times.append(float(region_beats[k] + gap * j / n_emits))
            phase -= n_emits
    return times


def _pick_top_sections(
    beat_times: np.ndarray,
    loudness: np.ndarray,
    *,
    direction: str,  # "intense" or "slow"
    top_n: int,
    min_duration: float,
    quantile: float,
    raw_strengths: np.ndarray | None = None,
    loudness_anticausal: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Pick top-N contiguous loud (or quiet) sections by total energy.

    1. Threshold the smoothed per-beat loudness at a quantile (top X% for
       intense, bottom X% for slow).
    2. Group contiguous beats above/below into candidate regions.
    3. Filter to regions sustained at least `min_duration` seconds.
    4. Rank by sum-of-deviation from the median (intensity-area).
    5. Keep the top N.

    Returns (regions, beat_mask) where beat_mask is a per-beat boolean
    aligned with beat_times marking beats inside the kept regions.
    """
    n = len(beat_times)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or top_n <= 0:
        return [], mask

    median = float(np.median(loudness))
    if direction == "intense":
        cutoff = float(np.quantile(loudness, 1.0 - quantile))
        candidate = loudness >= cutoff
        score_per_beat = np.maximum(loudness - median, 0.0)
    else:
        cutoff = float(np.quantile(loudness, quantile))
        candidate = loudness <= cutoff
        score_per_beat = np.maximum(median - loudness, 0.0)

    # Bridge short gaps (≤ MAX_GAP_BEATS) inside otherwise-contiguous runs
    # so a momentary dip mid-section doesn't fragment a real chorus/drop.
    MAX_GAP_BEATS = 6
    bridged = candidate.copy()
    i = 0
    while i < n:
        if not bridged[i]:
            j = i
            while j + 1 < n and not bridged[j + 1]:
                j += 1
            gap_len = j - i + 1
            if gap_len <= MAX_GAP_BEATS and i > 0 and j < n - 1:
                bridged[i : j + 1] = True
            i = j + 1
        else:
            i += 1
    candidate = bridged

    # Group contiguous candidate runs.
    candidates: list[tuple[int, int, float]] = []  # (start_idx, end_idx, score)
    i = 0
    while i < n:
        if candidate[i]:
            j = i
            while j + 1 < n and candidate[j + 1]:
                j += 1
            duration = float(beat_times[j] - beat_times[i])
            if duration >= min_duration:
                score = float(score_per_beat[i : j + 1].sum())
                candidates.append((i, j, score))
            i = j + 1
        else:
            i += 1

    # Rank by score, take top N. Greedy non-overlap is automatic since
    # candidates are disjoint by construction.
    candidates.sort(key=lambda c: c[2], reverse=True)
    picked = sorted(candidates[:top_n], key=lambda c: c[0])

    # Transient-snap (start): librosa's beat tracker has no concept of meter,
    # so beat_times[0] is not necessarily a downbeat. Refine each region's
    # start_idx by finding the beat within ±SNAP_RADIUS that has the highest
    # *unsmoothed* onset strength — the audible downbeat carries a much
    # stronger transient than the beats around it. Addresses Gemini's
    # "beat_times[0] is not a downbeat" objection.
    #
    # End refinement (anticausal): the causal smoothing used to find runs
    # lags ~w/2 beats past the true end (window stays full of past loud
    # samples after energy drops). An anticausal-smoothed envelope crosses
    # below cutoff ~w/2 beats *before* the true end (window starts filling
    # with future quiet). Averaging the two boundary positions lands close
    # to the audible end.
    SNAP_RADIUS = 4
    cutoff_anti = None
    if loudness_anticausal is not None:
        if direction == "intense":
            cutoff_anti = float(np.quantile(loudness_anticausal, 1.0 - quantile))
        else:
            cutoff_anti = float(np.quantile(loudness_anticausal, quantile))

    refined: list[tuple[int, int, float]] = []
    for start, end, score in picked:
        if raw_strengths is not None and direction == "intense":
            lo = max(0, start - SNAP_RADIUS)
            hi = min(n - 1, start + SNAP_RADIUS)
            window = raw_strengths[lo : hi + 1]
            if len(window) > 0:
                start = lo + int(np.argmax(window))
                end = max(end, start)
        if loudness_anticausal is not None and cutoff_anti is not None and direction == "intense":
            # Walk backward from causal-end to find the last anticausal-high
            # beat — that's the anticausal estimate of section end, biased
            # early by ~w/2. Average with the causal-end (biased late by
            # ~w/2) to estimate the true end.
            anti_end = start
            for i in range(end, start - 1, -1):
                if loudness_anticausal[i] >= cutoff_anti:
                    anti_end = i
                    break
            end = (end + anti_end) // 2
        if raw_strengths is not None and direction == "slow":
            # Gradient post-trim: inside a slow region, find the first beat
            # whose raw onset strength is significantly above a trailing-window
            # median (a drop, riser, or sudden re-entry of the kick). Trim the
            # region to end just before that beat — the smoothed loudness
            # filter has a ~4s lag and can include the drop in the "quiet"
            # classification, which felt wrong (slow pacing through the drop).
            # Baseline = median strength of the region's first K beats (when
            # it was genuinely quiet). Compare each subsequent beat against
            # that fixed floor, not a moving window. This catches risers
            # whose energy ramps gradually — a moving baseline rises with
            # the buildup and never crosses the spike threshold.
            BASELINE_K = 4
            SPIKE_RATIO = 1.15
            baseline_slice = raw_strengths[start : start + BASELINE_K]
            baseline = float(np.median(baseline_slice)) if len(baseline_slice) > 0 else 0.0
            if baseline > 0:
                # The min-duration filter applied during pick determines whether
                # a candidate qualifies as a "section." After that, trim is a
                # refinement: shortening to the audible boundary is desirable
                # even if it cuts the region below the original min-duration.
                # Floor at start+1 so we always keep a 2-beat region minimum.
                region_min_end = start + 1
                for i in range(start + BASELINE_K, end + 1):
                    if raw_strengths[i] > baseline * SPIKE_RATIO:
                        new_end = max(region_min_end, i - 1)
                        if new_end < end:
                            end = new_end
                        break
        refined.append((start, end, score))

    regions: list[tuple[float, float]] = []
    for start, end, _ in refined:
        regions.append((float(beat_times[start]), float(beat_times[end])))
        mask[start : end + 1] = True
    return regions, mask


def _ambient_transitions(
    region: tuple[float, float],
    fallback_duration: float,
    beat_times: np.ndarray | None = None,
) -> list[float]:
    """Place transitions inside an ambient region.

    If real beats exist inside the region (librosa just classified them
    low-confidence), snap transitions to those beats at the cadence-matched
    stride so photos still land on the kick. Otherwise distribute evenly
    across the region span.
    """
    start, end = region
    duration = end - start
    if duration <= 0 or fallback_duration <= 0:
        return []

    # Snap to beats inside the region when available — keeps photo cuts
    # aligned with the underlying tempo even where onset confidence dipped.
    if beat_times is not None and len(beat_times) > 0:
        inside = beat_times[(beat_times >= start - 1e-6) & (beat_times <= end + 1e-6)]
        if len(inside) > 0:
            # Determine stride: how many beats per emission.
            avg_beat = (
                float(np.mean(np.diff(inside))) if len(inside) > 1 else fallback_duration
            )
            stride = max(1, int(round(fallback_duration / avg_beat)))
            return [float(t) for t in inside[::stride]]

    n = max(1, int(round(duration / fallback_duration)))
    return [start + i * duration / n for i in range(n)]


def _weighted_decimate_by_month(
    indices: list[int],
    dates: list[datetime],
    target: int,
) -> tuple[list[int], list[tuple[str, int, int]]]:
    """Pick `target` indices, allocating slots per month proportional to count.

    Floors at 1 per non-empty month, unless there are more months than slots —
    then whole months are dropped, evenly across the span. Returns
    (selected_sorted, per_month_log).
    """
    if target >= len(indices):
        # Nothing to drop
        log = _summarize_per_month(indices, dates, indices)
        return list(indices), log

    by_month: dict[tuple[int, int], list[int]] = OrderedDict()
    for idx in indices:
        d = dates[idx]
        key = (d.year, d.month)
        by_month.setdefault(key, []).append(idx)

    if 0 < target < len(by_month):
        # More non-empty months than cuts in the whole video. One photo per
        # month is already too many, so something has to give — and what used to
        # give was the END of the timelapse: every month floored to 1, the
        # over-allocation trim bailed out (nothing left to trim), and the caller
        # kept the first `target` selections in date order. A 44-year span
        # rendered as 1982 through 2020 and simply stopped (#44).
        #
        # Drop whole MONTHS instead, sampled evenly across the span, so the
        # video still travels from the first photo to the last. Fewer months
        # shown, but the arc survives — which is the point of the thing.
        keys = list(by_month)
        chosen: set[int] = set()
        last = -1
        for v in np.linspace(0, len(keys) - 1, target):
            i = min(max(int(round(v)), last + 1), len(keys) - 1)
            chosen.add(i)
            last = i
        selected: list[int] = []
        log: list[tuple[str, int, int]] = []
        for i, key in enumerate(keys):
            bucket = by_month[key]
            # The middle photo of the month, not the first: months are ordered
            # within themselves, and the middle is the least likely to sit on a
            # boundary with the neighbouring month.
            picks = [bucket[len(bucket) // 2]] if i in chosen else []
            selected.extend(picks)
            log.append((f"{key[0]}-{key[1]:02d}", len(picks), len(bucket)))
        selected.sort()
        return selected, log

    total = sum(len(v) for v in by_month.values())
    raw = {k: len(v) / total * target for k, v in by_month.items()}
    floors = {k: max(1, int(raw[k])) for k in by_month}

    deficit = target - sum(floors.values())
    # Distribute remaining slots by largest fractional remainder
    remainders = sorted(
        ((raw[k] - int(raw[k]), k) for k in by_month), reverse=True
    )
    i = 0
    while deficit > 0 and i < len(remainders):
        floors[remainders[i][1]] += 1
        deficit -= 1
        i = (i + 1) % len(remainders)
        if i == 0 and deficit > 0:
            # Wrap-around: keep boosting the largest remainders
            pass
    # Trim if we over-allocated due to floors.
    while deficit < 0:
        biggest = max(floors, key=lambda k: floors[k])
        if floors[biggest] <= 1:
            break
        floors[biggest] -= 1
        deficit += 1

    selected: list[int] = []
    log: list[tuple[str, int, int]] = []
    for key, bucket in by_month.items():
        cnt = min(floors[key], len(bucket))
        if cnt >= len(bucket):
            picks = list(bucket)
        else:
            sample = np.linspace(0, len(bucket) - 1, cnt).round().astype(int)
            picks = [bucket[i] for i in sample]
        selected.extend(picks)
        log.append((f"{key[0]}-{key[1]:02d}", len(picks), len(bucket)))

    selected.sort()
    return selected, log


def _summarize_per_month(
    all_indices: list[int],
    dates: list[datetime],
    selected: list[int],
) -> list[tuple[str, int, int]]:
    sel_set = set(selected)
    by_month: dict[tuple[int, int], tuple[int, int]] = OrderedDict()
    for idx in all_indices:
        d = dates[idx]
        key = (d.year, d.month)
        kept, total = by_month.get(key, (0, 0))
        kept += 1 if idx in sel_set else 0
        total += 1
        by_month[key] = (kept, total)
    return [(f"{k[0]}-{k[1]:02d}", v[0], v[1]) for k, v in by_month.items()]


# --- Viterbi pacing model (experimental alternative to _classify_regions +
#     _pick_top_sections) ---------------------------------------------------
#
# Motivation (issue #43): the shipping tier map over-labels "ambient" because
# it gates on max-normalized ONSET STRENGTH (an event/attack detector), so a
# loud-but-steady passage reads as "no beat." This model instead builds a
# per-beat ENERGY signal from RMS loudness + onset rate, then labels the whole
# timeline with Viterbi (one switch-penalty knob) so tiers come out contiguous
# — no per-beat chatter, no per-genre parameters. Tiers stay per-song relative
# (a quiet song still gets its own loudest moment as "intense"), per intent #4.
_PACE_TIERS = ("ambient", "slow", "normal", "intense")


def _pace_robust_norm(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] using 5th/95th pct as the range (silence-robust)."""
    if len(x) == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _pace_moving_median(x: np.ndarray, win: int) -> np.ndarray:
    """Odd-window edge-padded moving median — smooth the CONTINUOUS energy so
    tiers come out contiguous (vs. smoothing labels, which cascades)."""
    if win <= 1 or len(x) == 0:
        return x
    if win % 2 == 0:
        win += 1
    r = win // 2
    pad = np.pad(x, r, mode="edge")
    return np.array([np.median(pad[i : i + win]) for i in range(len(x))])


# --------------------------------------------------------------------------- #
# Segment-first pacing model ("segment"): Foote novelty on stacked beat-synced
# features cuts the song into sections, then each SECTION is scored by its
# cyan-height (p90 spectral-centroid envelope) and labelled against mode-anchored
# tier centers. Boundaries land on beats by construction, so tier changes are
# crisp; scoring a whole section by its interior median avoids the per-beat
# chatter the height signal alone would produce. Developed in
# experiments/pacing_recipes.py (recipe_c); see docs there for the derivation.
# --------------------------------------------------------------------------- #
def _pace_beat_agg(arr: np.ndarray, beat_frames: np.ndarray, agg=np.mean) -> np.ndarray:
    """Aggregate a per-frame feature over each beat's window [beat_i, beat_i+1).
    Window stat (not point-sample) so a bursty beat reflects its bursts."""
    n = len(beat_frames)
    out = np.zeros(n)
    for i in range(n):
        lo = int(beat_frames[i])
        hi = int(beat_frames[i + 1]) if i + 1 < n else len(arr)
        out[i] = agg(arr[lo:hi]) if hi > lo else arr[min(lo, len(arr) - 1)]
    return out


def _pace_beat_smooth(x: np.ndarray, w: int = 8) -> np.ndarray:
    """Centered moving mean over ~w beats (1-D or per-column for 2-D)."""
    from scipy.ndimage import uniform_filter1d

    if x.ndim == 1:
        return uniform_filter1d(x, w, mode="nearest")
    return np.column_stack(
        [uniform_filter1d(x[:, k], w, mode="nearest") for k in range(x.shape[1])]
    )


def _pace_foote_novelty(ssm: np.ndarray, kernel_size: int) -> np.ndarray:
    """Foote checkerboard novelty along the diagonal of a self-similarity matrix."""
    L = kernel_size
    g = np.linspace(-1.0, 1.0, 2 * L)
    gauss = np.outer(np.exp(-4.0 * g**2), np.exp(-4.0 * g**2))
    sign = np.outer(np.sign(g), np.sign(g))
    kernel = gauss * sign
    n = ssm.shape[0]
    nov = np.zeros(n)
    for i in range(n):
        a, b = i - L, i + L
        pa, pb = max(0, a), min(n, b)
        ka, kb = pa - a, 2 * L - (b - pb)
        nov[i] = float((ssm[pa:pb, pa:pb] * kernel[ka:kb, ka:kb]).sum())
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov = nov / nov.max()
    return nov


def _pace_mode_anchored_centers(vals: np.ndarray) -> np.ndarray:
    """Tier centers anchored at the BASELINE (mode = the groove the song sits at
    most often), with asymmetric up/down spread. The median is dragged up by
    peaks and pushes the baseline into 'slow'; the mode stays put. Not clipped to
    [0,1]: `combined` ranges past 1.0, and clipping crushed normal/intense
    together on tracks whose typical section already sits high, erasing intense."""
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    hist, edges = np.histogram(vals, bins=20, range=(vals.min(), vals.max() + 1e-9))
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    up = max(hi - mode, 0.05)
    dn = max(mode - lo, 0.05)
    return np.maximum(
        np.array([mode - 0.9 * dn, mode - 0.45 * dn, mode, mode + 0.6 * up]), 0.0
    )


def _pace_rolling_p80(x: np.ndarray, win: int = 16) -> np.ndarray:
    """p80 of x in a centered ~win-beat window at every beat — used to calibrate
    tier centers on the SAME statistic sections are scored with (p80). Fitting
    centers on raw per-beat values is a statistic mismatch: p80 is upward-biased,
    so on burst-texture tracks the per-beat mode sits among gap beats and every
    section's p80 rides above it, making slow/ambient unreachable. See
    experiments/pacing_recipes.py _rolling_p80."""
    n = len(x)
    if n == 0:
        return x
    r = win // 2
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - r), min(n, i + r + 1)
        out[i] = np.percentile(x[a:b], 80)
    return out


def _pace_dynamics_weight(loud_beat: np.ndarray) -> float:
    """How far to trust this track's loudness as an intensity signal, in [0, 1].

    The interquartile spread of per-beat loudness: 1.0 for a track with real
    dynamics, ~0 for a compressed one where every section measures the same. Two
    terms key off it in OPPOSITE directions — loudness is weighted by it, and the
    modulation-depth fallback by (1 - it) — so exactly one of them decides any
    given track.
    """
    iqr_db = float(np.percentile(loud_beat, 75) - np.percentile(loud_beat, 25))
    return float(np.clip((iqr_db - 3.0) / 6.0, 0.0, 1.0))


def _pace_depth_boost(loud_beat: np.ndarray, ceiling: float) -> float:
    """Effective weight for the modulation-depth fallback on this track.

    Loudness and texture are ALTERNATIVE intensity reads, not complementary ones,
    so hold the fallback at full strength until loudness is clearly the better
    read, then ramp it off — rather than blending, which leaves a middling track
    scored well by neither. In loudness-spread terms: full fallback below 6 dB
    between the quartiles, none above 9 dB, linear across.

    Blending was tried first and rejected: it took Push Upstairs from 0.6 to 0.43,
    enough to tip a section already sitting on a tier boundary into a spurious
    "intense" at 2:09 that ate 37 photos and starved 16s off an approved render.
    """
    w = _pace_dynamics_weight(loud_beat)
    return ceiling * float(np.clip(2.0 * (1.0 - w), 0.0, 1.0))


def _pace_tiers_segment(
    audio_path: Path,
    beat_times: np.ndarray,
    bpm: float,
    *,
    kernel_beats: int = 16,
    min_seg_beats: int = 16,
    prominence: float = 0.12,
    depth_boost: float = 0.6,
) -> list[str]:
    """Per-beat pacing tiers via the segment-first cyan-height model (recipe C).

    1. Beat-synced texture features (log-centroid p90 = cyan height, log-RMS,
       MFCC, modulation depth) -> cosine SSM -> Foote novelty -> segment cuts.
    2. Per-beat energy = cyan height + variance-gated loudness booster, plus a
       modulation-depth boost that rescues burst trains (median energy sits in
       the gaps; depth is high there). Both extra terms are gated on the track's
       loudness spread, in opposite directions: a track with real dynamics is
       scored on them, a compressed one falls back to texture. `depth_boost` is
       the ceiling on that fallback, reached only when the track is fully flat.
    3. Mode-anchored tier centers over the per-beat combined signal; each SEGMENT
       labelled by the nearest center to its median.
    """
    import librosa

    if len(beat_times) < 8:
        return ["ambient"] * len(beat_times)

    y, sr = librosa.load(str(audio_path), mono=True)
    hop = 512
    bf = np.clip(np.asarray(librosa.time_to_frames(beat_times, sr=sr, hop_length=hop), dtype=int), 0, None)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    logcent = np.log2(np.maximum(cent, 1e-6))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    logrms = np.log(np.maximum(rms, 1e-6))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)

    def p90(a):
        return float(np.percentile(a, 90)) if len(a) else 0.0
    height_raw = _pace_beat_agg(logcent, bf, agg=p90)        # per-beat cyan height
    lrms_beat = _pace_beat_agg(logrms, bf)
    loud_beat = _pace_beat_agg(rms_db, bf)
    mfcc_beat = np.column_stack([_pace_beat_agg(mfcc[k], bf) for k in range(mfcc.shape[0])])

    # modulation depth: p90-p10 of raw height over ~16 beats (burst-train marker)
    n = len(height_raw)
    moddepth = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - 8), min(n, i + 9)
        wnd = height_raw[a:b]
        moddepth[i] = np.percentile(wnd, 90) - np.percentile(wnd, 10) if len(wnd) else 0.0

    # per-beat energy: cyan height + one-way, variance-gated loudness booster
    height_n = _pace_robust_norm(height_raw)
    loud_n = _pace_robust_norm(loud_beat)
    w_loud = _pace_dynamics_weight(loud_beat)
    energy = (height_n + w_loud * loud_n) / (1.0 + w_loud)
    # The depth boost is a CRUTCH for tracks with no dynamics to read, so gate it
    # by the same spread that gates loudness: where loudness is trustworthy
    # (w_loud -> 1) the crutch drops out. Ungated it mislabels sparse acoustic
    # music, because there the modulation is silence between notes, not a burst
    # train -- on "To Build a Home" it put "intense" on the two QUIETEST
    # non-ambient sections (-21 and -16 dB) while the -11 dB climax read "normal",
    # and measured correlation between moddepth and loudness there is +0.05.
    # Compressed tracks keep it: Push Upstairs is uniformly -10..-11 dB (IQR
    # 4.7 dB), so texture is the only signal and its tier map is unchanged.
    combined = energy + _pace_depth_boost(loud_beat, depth_boost) * _pace_robust_norm(moddepth)

    # segment boundaries: Foote novelty on TEXTURE-WINDOWED stacked features
    if n < 2 * kernel_beats:
        bounds = [0]
    else:
        level = _pace_beat_smooth(np.column_stack([mfcc_beat, lrms_beat, height_raw]), 8)
        feat = np.column_stack([level, moddepth])
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
        fn = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-9)
        ssm = fn @ fn.T
        kb = min(kernel_beats, n // 2) or 1
        nov = _pace_foote_novelty(ssm, kb)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(nov, prominence=prominence, distance=max(min_seg_beats, kb))
        bounds = sorted(set([0] + [int(p) for p in peaks]))

    seg_edges = list(zip(bounds, bounds[1:] + [n]))
    # Calibrate centers on the section-scale statistic (rolling-p80), matching the
    # p80 segment scores, so slow/ambient are reachable on burst-texture tracks.
    centers = _pace_mode_anchored_centers(_pace_rolling_p80(combined, 16))
    tiers = ["normal"] * n
    for s0, s1 in seg_edges:
        if s1 <= s0:
            continue
        # Score by p80 ("how bright does this section GET"), not median: rewards a
        # segment that spikes even if calm on average, and doubles as a duty-cycle
        # floor for burst trains. See experiments/pacing_recipes.py recipe_c.
        score = float(np.percentile(combined[s0:s1], 80))
        tier = _PACE_TIERS[int(np.argmin((score - centers) ** 2))]
        for i in range(s0, s1):
            tiers[i] = tier
    return tiers


def _pace_tiers_to_regions(
    tiers: list[str], beat_times: np.ndarray, audio_duration: float
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]],
    list[tuple[float, float]], list[tuple[float, float]],
    np.ndarray, np.ndarray,
]:
    """Convert per-beat tiers into the (intense, slow, ambient, beat/normal)
    regions + per-beat intense/slow masks the renderer consumes. Regions tile
    [0, audio_duration] with no gaps (each run spans to the next run's start)."""
    n = len(beat_times)
    intense: list[tuple[float, float]] = []
    slow: list[tuple[float, float]] = []
    ambient: list[tuple[float, float]] = []
    beat: list[tuple[float, float]] = []
    intense_mask = np.zeros(n, dtype=bool)
    slow_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tiers[j + 1] == tiers[i]:
            j += 1
        a = 0.0 if i == 0 else float(beat_times[i])
        b = float(beat_times[j + 1]) if j + 1 < n else audio_duration
        t = tiers[i]
        if t == "intense":
            intense.append((a, b))
            intense_mask[i : j + 1] = True
        elif t == "slow":
            slow.append((a, b))
            slow_mask[i : j + 1] = True
        elif t == "ambient":
            ambient.append((a, b))
        else:
            beat.append((a, b))
        i = j + 1
    return intense, slow, ambient, beat, intense_mask, slow_mask


def build_timeline(
    audio_path: Path,
    photo_dates: list[datetime],
    *,
    max_photos_per_second: float = 4.0,
    min_photos_per_beat: float = 1.0,
    beat_speed: float | None = None,
    beat_thresh: float = 0.30,
    fallback_duration_seconds: float = 0.150,
    vary_pace: bool = False,
    intense_multiplier: float = 2.0,
    slow_multiplier: float = 0.5,
    max_intense: int = 2,
    max_slow: int = 1,
    min_section_seconds: float = 5.0,
    section_quantile: float = 0.25,
    min_normal_bridge_beats: float = 8.0,
    snap_to_grid: bool = True,
    tier_lead_seconds: float = 0.0,
    pace_model: str = "segment",
    base_pace: str = "occupancy",
    onset_anchor: str = "auto",
) -> BeatTimeline:
    """Build a beat-aligned timeline matching photos to transition times.

    photo_dates: dates for the *kept* photos (post-alignment), in order.
    pace_model: "segment" (Foote-segmented cyan-height, per-section scored — the
      shipped model) drives the tiers; any other value falls back to the quantile
      top-N section picker.
    """
    n_photos = len(photo_dates)
    beat_times, strengths, loudness, loudness_anticausal, bpm, audio_duration = _detect_beats(
        audio_path, tier_lead_seconds=tier_lead_seconds
    )

    beat_regions, ambient_regions = _classify_regions(
        beat_times, strengths, audio_duration, beat_thresh
    )

    # Occupancy base pace: set the normal-tier subdivision from the song's own
    # spectral density (octave-free, from median-IBI), overriding the photo-count
    # driven subdivision. An explicit --beat-speed still wins over this.
    cut_felt_parity: int | None = None
    onset_strikes: np.ndarray = np.array([])
    onset_strike_heights: np.ndarray = np.array([])
    onset_anchor_spans_final: list[tuple[float, float]] = []
    if base_pace == "occupancy" and beat_speed is None and len(beat_times) > 1:
        import librosa

        y_occ, sr_occ = librosa.load(str(audio_path), mono=True)
        occ_sub, _occ, _pps = _occupancy_base_subdivision(y_occ, sr_occ, beat_times)
        if occ_sub is not None:
            beat_speed = occ_sub
            # Felt-downbeat parity: on a doubled grid the true pulse is every
            # OTHER beat; the stronger-onset parity (four-on-floor kick / ballad
            # chord) is the "1 & 2". Snapping cuts to it keeps every flip on the
            # felt pulse across tier boundaries, instead of odd tier spacings
            # (e.g. intense every 3 beats) drifting cuts onto the off-beat.
            s = np.asarray(strengths, dtype=float)
            if len(s) > 3:
                cut_felt_parity = 0 if s[0::2].mean() >= s[1::2].mean() else 1
        # Peak-picked note attacks, for onset-anchoring sparse/rubato spans where
        # the beat grid is fiction (see _prominent_strikes / _onset_anchor_spans).
        if onset_anchor != "never":
            onset_strikes, onset_strike_heights = _prominent_strikes(y_occ, sr_occ)

    # 3-tier pacing: pick the top-N most intense and top-N most quiet
    # *sustained* sections of the song. Rank-based (not threshold-based) so
    # we always get a small, consistent number of pace changes regardless of
    # how dynamic the track is.
    if vary_pace and len(loudness) > 0:
        intense_regions, intense_mask = _pick_top_sections(
            beat_times, loudness,
            direction="intense",
            top_n=max_intense,
            min_duration=min_section_seconds,
            quantile=section_quantile,
            raw_strengths=strengths,
            loudness_anticausal=loudness_anticausal,
        )
        slow_regions, slow_mask = _pick_top_sections(
            beat_times, loudness,
            direction="slow",
            top_n=max_slow,
            min_duration=min_section_seconds,
            quantile=section_quantile,
            raw_strengths=strengths,
            loudness_anticausal=loudness_anticausal,
        )
    else:
        intense_regions, intense_mask = [], np.zeros(len(beat_times), dtype=bool)
        slow_regions, slow_mask = [], np.zeros(len(beat_times), dtype=bool)
    eff_intense_mult = intense_multiplier if vary_pace else 1.0
    eff_slow_mult = slow_multiplier if vary_pace else 1.0

    # Snap multipliers to musical fractions on the 4/4 grid. A value like
    # 3 (triplets) or 0.33 (3-beat stride) creates a polyrhythm that
    # drifts off the downbeat. Snap to {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8,
    # 16} so cuts always land on a grid-aligned position. Opt out with
    # snap_to_grid=False.
    if snap_to_grid and vary_pace:
        grid = (1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16)

        def _snap(x: float) -> float:
            if x <= 0:
                return x
            return min(grid, key=lambda g: abs(g - x) / x)

        eff_intense_mult = _snap(eff_intense_mult)
        eff_slow_mult = _snap(eff_slow_mult)

    # Merge tiny "normal" bridges between overlay regions. A 3-second gap
    # between slow and intense at 143 BPM is only ~7 beats — too short to
    # register as its own tier, feels like a jump cut. Extend the *later*
    # region backward so the transition is direct (slow → intense).
    if vary_pace and bpm > 0:
        min_bridge_seconds = min_normal_bridge_beats * 60.0 / bpm

        def _merge_bridges(
            regions: list[tuple[float, float]],
            other: list[tuple[float, float]],
        ) -> list[tuple[float, float]]:
            # Combine all non-normal regions (self overlay, other overlay, and
            # ambient) so a short normal sliver between any two of them can be
            # absorbed. Ambient is read-only — we only extend overlay regions.
            combined = sorted(
                [(a, b, "self") for a, b in regions]
                + [(a, b, "other") for a, b in other]
                + [(a, b, "ambient") for a, b in ambient_regions]
            )
            new_self: list[tuple[float, float]] = []
            for i, (a, b, kind) in enumerate(combined):
                start = a
                # Extend start backward over any tiny normal gap from prior.
                if i > 0:
                    prev_end = combined[i - 1][1]
                    if 0 < a - prev_end < min_bridge_seconds:
                        start = prev_end
                if kind == "self":
                    new_self.append((start, b))
            return new_self

        intense_regions = _merge_bridges(intense_regions, slow_regions)
        slow_regions = _merge_bridges(slow_regions, intense_regions)
        # Refresh masks to match the extended regions.
        for a, b in intense_regions:
            intense_mask |= (beat_times >= a - 1e-6) & (beat_times <= b + 1e-6)
        for a, b in slow_regions:
            slow_mask |= (beat_times >= a - 1e-6) & (beat_times <= b + 1e-6)
        # An extended slow region might now overlap an intense one — intense wins.
        slow_mask &= ~intense_mask

    # Model-driven pacing ("segment"): replace the region set computed above with
    # per-beat tiers from the Foote-segmented cyan-height model (recipe C).
    # Overwrites rather than branches so all downstream transition/segment logic
    # is shared. The quantile _pick_top_sections result above stays as the
    # fallback when the model can't run (no beats / --no-vary-pace).
    if pace_model == "segment" and vary_pace and len(beat_times) > 0:
        tiers = _pace_tiers_segment(audio_path, beat_times, bpm)
        (intense_regions, slow_regions, ambient_regions, beat_regions,
         intense_mask, slow_mask) = _pace_tiers_to_regions(tiers, beat_times, audio_duration)
        eff_intense_mult = intense_multiplier
        eff_slow_mult = slow_multiplier
        if snap_to_grid:
            grid = (1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16)
            eff_intense_mult = min(grid, key=lambda g: abs(g - eff_intense_mult) / eff_intense_mult)
            eff_slow_mult = min(grid, key=lambda g: abs(g - eff_slow_mult) / eff_slow_mult)

    def _ambient_subregions(region: tuple[float, float]) -> list[tuple[tuple[float, float], float]]:
        """Slice an ambient region into sub-spans tagged with their tier
        multiplier, so a slow/intense overlay extends through ambient too."""
        start, end = region
        # Build a sorted list of overlay boundaries within (start, end).
        events = [start, end]
        for a, b in intense_regions:
            if b > start and a < end:
                events.append(max(start, a))
                events.append(min(end, b))
        for a, b in slow_regions:
            if b > start and a < end:
                events.append(max(start, a))
                events.append(min(end, b))
        events = sorted(set(events))
        out: list[tuple[tuple[float, float], float]] = []
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            if b - a < 1e-6:
                continue
            mid = (a + b) / 2
            mult = 1.0
            for ia, ib in intense_regions:
                if ia - 1e-6 <= mid <= ib + 1e-6:
                    mult = eff_intense_mult
                    break
            else:
                for sa, sb in slow_regions:
                    if sa - 1e-6 <= mid <= sb + 1e-6:
                        mult = eff_slow_mult
                        break
            out.append(((a, b), mult))
        return out

    def _kind_at(t: float) -> str:
        for a, b in beat_regions:
            if a - 1e-6 <= t <= b + 1e-6:
                return "beat"
        return "ambient"

    def _gen_transitions(sub: float) -> list[tuple[float, str]]:
        """Unified phase-accumulator pass over every beat in the song.

        Each beat contributes `sub × tier_multiplier` to the phase. When
        phase crosses an integer, that many transitions are emitted within
        this beat's gap to the next beat. This treats beat-synced and
        ambient regions uniformly so rate stays consistent across the
        beat-synced→ambient boundary inside an overlay region (the bug
        that caused intense rate to drop from 7.18/s to 2.39/s when
        crossing into an ambient sub-section).
        """
        out: list[tuple[float, str]] = []
        if len(beat_times) == 0:
            return out

        # Pre-compute a per-beat ambient flag: a beat is ambient if it sits
        # in an ambient region (no confident beat tracking). Ambient defaults
        # to the slow multiplier so atmospheric stretches breathe instead of
        # pacing like normal beat-synced sections.
        beat_is_ambient = np.zeros(len(beat_times), dtype=bool)
        for a, b in ambient_regions:
            beat_is_ambient |= (beat_times >= a - 1e-6) & (beat_times <= b + 1e-6)

        n = len(beat_times)

        if cut_felt_parity is not None:
            # Felt-lock: cuts on the felt-downbeat parity at a fixed metronomic
            # subdivision per tier (intense 1:1, normal 1:2, slow 1:4 on the raw
            # grid). See _felt_locked_cut_indices + tests/test_felt_lock.py.
            for k in _felt_locked_cut_indices(
                n, intense_mask, slow_mask, beat_is_ambient, cut_felt_parity,
            ):
                t = float(beat_times[k])
                out.append((t, _kind_at(t)))
            return out

        phase = 0.0
        for k in range(n):
            if intense_mask[k]:
                local = sub * eff_intense_mult
            elif slow_mask[k]:
                local = sub * eff_slow_mult
            elif beat_is_ambient[k]:
                local = sub * eff_slow_mult
            else:
                local = sub
            phase += local
            n_emits = int(phase + 1e-9)
            if n_emits > 0:
                gap = (
                    beat_times[k + 1] - beat_times[k]
                    if k + 1 < n
                    else (60.0 / bpm if bpm > 0 else 0.5)
                )
                for j in range(n_emits):
                    t = float(beat_times[k] + gap * j / n_emits)
                    out.append((t, _kind_at(t)))
                phase -= n_emits
        return out
        out.sort()
        # Dedup at kind boundaries only. Within a homogeneous run (a stretch
        # of beat-synced transitions, or a stretch of ambient transitions),
        # trust the generator. Across the boundary between beat-synced and
        # ambient, a small gap is the boundary artifact we want to absorb,
        # so drop the trailing one if it's <½ nominal cadence apart from the
        # last transition. Within a run we only kill exact numerical dups.
        nominal_gap = (60.0 / bpm) / sub if bpm > 0 and sub > 0 else 0.150
        boundary_min_gap = nominal_gap * 0.5
        deduped: list[tuple[float, str]] = []
        for t, kind in out:
            if deduped:
                prev_t, prev_kind = deduped[-1]
                gap = t - prev_t
                if gap < 0.020:
                    continue  # numerical duplicate
                if kind != prev_kind and gap < boundary_min_gap:
                    continue  # absorb boundary artifact
            deduped.append((t, kind))
        return deduped

    # Pick subdivision. If the user specified --beat-speed, honor it strictly.
    # Otherwise, start from the max-photos-per-second cap and back off (coarser
    # subdivisions) until n_transitions <= n_photos. This makes a short photo
    # set span the whole song instead of truncating to the first chunk.
    # Ambient regions adopt the beat cadence so an intro/outro doesn't suddenly
    # flip pace. Only fall back to a constant --duration when no BPM was
    # detected (silent / unmeasurable audio).
    def _effective_fallback(sub: float) -> float:
        if bpm <= 0 or sub <= 0:
            return fallback_duration_seconds
        return (60.0 / bpm) / sub

    bounds_violation: str | None = None
    if beat_speed is not None:
        subdivision = beat_speed
        fallback_duration_seconds = _effective_fallback(subdivision)
        transitions = _gen_transitions(subdivision)
    else:
        subdivision, bounds_violation = _pick_subdivision(
            bpm, max_photos_per_second, min_photos_per_beat, None
        )
        fallback_duration_seconds = _effective_fallback(subdivision)
        transitions = _gen_transitions(subdivision)
        ladder = [4.0, 2.0, 1.0, 0.5, 0.25]
        # Walk down from current subdivision until transitions <= photos OR
        # we hit the min-photos-per-beat floor. The floor prevents the video
        # from getting glacially slow on long songs with few photos — if the
        # song can't be filled at the floor, we'll truncate audio instead.
        while len(transitions) > n_photos and subdivision > ladder[-1]:
            try:
                idx = ladder.index(subdivision)
            except ValueError:
                break
            if idx + 1 >= len(ladder):
                break
            next_sub = ladder[idx + 1]
            if next_sub < min_photos_per_beat:
                break
            subdivision = next_sub
            fallback_duration_seconds = _effective_fallback(subdivision)
            transitions = _gen_transitions(subdivision)

    if not transitions:
        # Edge case: no transitions detected; fall back to one segment per photo
        per = audio_duration / max(1, n_photos)
        return BeatTimeline(
            segments=[Segment(i, per) for i in range(n_photos)],
            bpm=bpm,
            subdivision=subdivision,
            beat_synced_regions=[],
            ambient_regions=[(0.0, audio_duration)],
            total_duration=audio_duration,
            photos_kept=n_photos,
            photos_dropped=0,
            photos_held_short=0,
            selection_per_month=[],
            beat_times=[float(t) for t in beat_times],
        )

    # Ensure the timeline starts at audio time 0. If the first beat-emission
    # is well into the song (long intro), we'd otherwise hold one photo for
    # the entire pre-roll. Subdivide the pre-roll at the fallback cadence so
    # the intro paces like a normal ambient region. Keep-out zone: drop the
    # final inserted transition if it lands within 2× fallback of the first
    # real beat, so the first cut still aligns to the beat instead of
    # firing a jarring photo flip right before it.
    if transitions and transitions[0][0] > 1e-3:
        first_t = transitions[0][0]
        # Pass beat_times=None so we get even distribution at fallback cadence
        # instead of snapping to the single boundary beat that sits at first_t.
        # Scale by 1/slow_mult so pre-roll matches the slow rate that ambient
        # uses inside the beat-tracked range.
        ambient_fallback = fallback_duration_seconds / max(eff_slow_mult, 1e-3)
        pre_roll = _ambient_transitions(
            (0.0, first_t), ambient_fallback, beat_times=None
        )
        # Keep-out zone is one *normal* beat period — drops only transitions
        # close enough to the first real beat to feel like a double-hit.
        # Using ambient_fallback here would be too aggressive: ambient is
        # already half-rate, so a transition 0.85s before the beat is fine.
        keep_out = fallback_duration_seconds
        pre_roll = [t for t in pre_roll if first_t - t > keep_out]
        if not pre_roll or pre_roll[0] > 1e-3:
            pre_roll.insert(0, 0.0)
        for t in reversed(pre_roll):
            transitions.insert(0, (float(t), "ambient"))

    # Onset-anchor: in sparse/rubato spans the beat grid is fiction, so replace
    # its cuts with cuts on the real note strikes (the events ARE the pulse there).
    if onset_anchor != "never" and len(onset_strikes) > 0:
        felt_for_support = (
            beat_times[cut_felt_parity::2] if cut_felt_parity is not None else beat_times
        )
        if onset_anchor == "always":
            anchor_spans = [(0.0, audio_duration)]
        elif _grid_support(felt_for_support, onset_strikes) >= _ONSET_ANCHOR_SONG_GATE:
            # Overall grid-locked (e.g. four-on-floor): trust the grid everywhere,
            # even through breakdowns. Local dips aren't rubato, they're arrangement.
            anchor_spans = []
        else:
            anchor_spans = _onset_anchor_spans(
                felt_for_support, onset_strikes, audio_duration
            )

        def _tier_at_time(t: float) -> str:
            for a, b in intense_regions:
                if a - 1e-6 <= t <= b + 1e-6:
                    return "intense"
            for a, b in slow_regions:
                if a - 1e-6 <= t <= b + 1e-6:
                    return "slow"
            for a, b in ambient_regions:
                if a - 1e-6 <= t <= b + 1e-6:
                    return "ambient"
            return "normal"

        # With a felt grid, intense/normal/slow keep their metronomic grid cuts
        # through the span and only ambient follows the notes (issue #48). Without
        # the felt lock there is no grid to hold onto, so every tier note-counts.
        has_felt_grid = cut_felt_parity is not None
        transitions = _splice_onset_anchor(
            transitions, anchor_spans, onset_strikes, _tier_at_time,
            onset_strike_heights,
            has_grid=has_felt_grid,
        )
        onset_anchor_spans_final = anchor_spans
        # The splice may have removed the t=0 start (it sits inside the intro
        # span); restore it so the timeline still opens at audio time 0.
        if transitions and transitions[0][0] > 1e-3:
            transitions.insert(0, (0.0, transitions[0][1]))

    transitions = _drop_opening_flash(transitions)

    # Compute durations: each transition's duration is the gap to the next,
    # with the final segment running to audio_duration.
    times = [t for t, _ in transitions]
    kinds = [k for _, k in transitions]
    durations: list[float] = []
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else audio_duration
        durations.append(max(0.05, nxt - t))

    def _tier_at(t: float, kind: str) -> str:
        for a, b in intense_regions:
            if a - 1e-6 <= t <= b + 1e-6:
                return "intense"
        for a, b in slow_regions:
            if a - 1e-6 <= t <= b + 1e-6:
                return "slow"
        if kind == "ambient":
            return "ambient"
        return "normal"

    seg_tiers = [_tier_at(times[i], kinds[i]) for i in range(len(times))]

    n_transitions = len(durations)

    # Photo→segment assignment.
    held_short = 0
    audio_trimmed = 0.0
    if n_photos > n_transitions:
        # Decimate photos to fit beat grid (weighted by month)
        all_idx = list(range(n_photos))
        selected, per_month = _weighted_decimate_by_month(all_idx, photo_dates, n_transitions)
        photos_kept = len(selected)
        photos_dropped = n_photos - photos_kept
        segments = [
            Segment(selected[i], durations[i], seg_tiers[i])
            for i in range(min(photos_kept, n_transitions))
        ]
    elif n_photos < n_transitions:
        # Floor was hit: truncate timeline to fit photos.
        segments = [
            Segment(i, durations[i], seg_tiers[i]) for i in range(n_photos)
        ]
        per_month = _summarize_per_month(list(range(n_photos)), photo_dates, list(range(n_photos)))
        photos_kept = n_photos
        photos_dropped = 0
        held_short = 0
        audio_trimmed = audio_duration - sum(s.duration for s in segments)
    else:
        segments = [
            Segment(i, durations[i], seg_tiers[i]) for i in range(n_photos)
        ]
        per_month = _summarize_per_month(list(range(n_photos)), photo_dates, list(range(n_photos)))
        photos_kept = n_photos
        photos_dropped = 0

    total_duration = sum(s.duration for s in segments)

    return BeatTimeline(
        segments=segments,
        bpm=bpm,
        subdivision=subdivision,
        beat_synced_regions=beat_regions,
        ambient_regions=ambient_regions,
        total_duration=total_duration,
        photos_kept=photos_kept,
        photos_dropped=photos_dropped,
        photos_held_short=held_short,
        selection_per_month=per_month,
        audio_trimmed_seconds=audio_trimmed,
        bounds_violation=bounds_violation,
        intense_regions=intense_regions,
        slow_regions=slow_regions,
        intense_multiplier=eff_intense_mult,
        slow_multiplier=eff_slow_mult,
        beat_times=[float(t) for t in beat_times],
        onset_anchor_spans=onset_anchor_spans_final,
        onset_strikes=[float(t) for t in onset_strikes],
    )
