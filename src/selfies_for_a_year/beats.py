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
                    min_d = min(region_durs)
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
    def from_timeline(cls, timeline: "BeatTimeline") -> "TrackProgression":
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

    Floors at 1 per non-empty month. Returns (selected_sorted, per_month_log).
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


def _pace_viterbi(
    energy: np.ndarray, centers: np.ndarray, sigma: float, switch_penalty: float
) -> list[str]:
    """Label each beat by Viterbi: emission cost (energy-center)^2/(2 sigma^2)
    vs. a flat per-switch penalty. High penalty -> long contiguous runs."""
    n = len(energy)
    k = len(centers)
    if n == 0:
        return []
    emit = (energy[:, None] - centers[None, :]) ** 2 / (2.0 * sigma**2)
    cost = np.full((n, k), np.inf)
    back = np.zeros((n, k), dtype=int)
    cost[0] = emit[0]
    switch = np.full(k, switch_penalty)
    for t in range(1, n):
        for s in range(k):
            trans = cost[t - 1] + switch
            trans[s] -= switch_penalty  # staying in s is free
            j = int(np.argmin(trans))
            cost[t, s] = trans[j] + emit[t, s]
            back[t, s] = j
    path = [int(np.argmin(cost[-1]))]
    for t in range(n - 1, 0, -1):
        path.append(back[t, path[-1]])
    path.reverse()
    return [_PACE_TIERS[s] for s in path]


def _pace_merge_short_runs(tiers: list[str], min_beats: int) -> list[str]:
    """Merge any tier run shorter than min_beats into its longer neighbor.
    Applied to clean Viterbi output, so it's a stable sweep. Encodes 'a tier
    must last >= min_beats to register as a section' (a 2s flip is a stutter)."""
    out = list(tiers)
    while True:
        runs: list[list] = []
        i = 0
        while i < len(out):
            j = i
            while j + 1 < len(out) and out[j + 1] == out[i]:
                j += 1
            runs.append([i, j, out[i]])
            i = j + 1
        if len(runs) <= 1:
            break
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
    return out


def _pace_tiers_viterbi(
    audio_path: Path,
    beat_times: np.ndarray,
    bpm: float,
    *,
    w_loud: float = 0.7,
    smooth_beats: int = 6,
    switch_penalty: float = 4.0,
    min_run_seconds: float = 4.0,
) -> list[str]:
    """Per-beat pacing tiers via the energy+Viterbi model. Loads the audio to
    compute RMS loudness + onset rate at each beat."""
    import librosa

    if len(beat_times) == 0:
        return []
    y, sr = librosa.load(str(audio_path), mono=True)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
    beat_frames = np.clip(np.asarray(beat_frames, dtype=int), 0, None)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    loud = rms_db[np.clip(beat_frames, 0, len(rms_db) - 1)]

    onset_times = np.asarray(
        librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time")
    )
    if len(onset_times):
        lo = np.searchsorted(onset_times, beat_times - 0.5)
        hi = np.searchsorted(onset_times, beat_times + 0.5)
        rate = (hi - lo).astype(float)
    else:
        rate = np.zeros(len(beat_times))

    energy = w_loud * _pace_robust_norm(loud) + (1.0 - w_loud) * _pace_robust_norm(rate)
    energy = _pace_moving_median(energy, smooth_beats)
    # Fixed centers in the per-song normalized [0,1] space keep tiers from
    # collapsing when a long tail skews the distribution, while robust_norm
    # keeps them per-song relative.
    centers = np.array([0.12, 0.37, 0.62, 0.87])
    tiers = _pace_viterbi(energy, centers, 0.18, switch_penalty)
    min_beats = max(1, round(min_run_seconds * bpm / 60.0)) if bpm > 0 else 1
    return _pace_merge_short_runs(tiers, min_beats)


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
            intense.append((a, b)); intense_mask[i : j + 1] = True
        elif t == "slow":
            slow.append((a, b)); slow_mask[i : j + 1] = True
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
    pace_model: str = "current",
) -> BeatTimeline:
    """Build a beat-aligned timeline matching photos to transition times.

    photo_dates: dates for the *kept* photos (post-alignment), in order.
    pace_model: "current" (onset-strength gate + quantile top-N sections) or
      "viterbi" (RMS-loudness + onset-rate energy, Viterbi-labeled tiers).
    """
    n_photos = len(photo_dates)
    beat_times, strengths, loudness, loudness_anticausal, bpm, audio_duration = _detect_beats(
        audio_path, tier_lead_seconds=tier_lead_seconds
    )

    beat_regions, ambient_regions = _classify_regions(
        beat_times, strengths, audio_duration, beat_thresh
    )

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

    # Viterbi pacing model: replace the region set computed above with tiers
    # from the RMS-loudness + onset-rate energy signal. Overwrites rather than
    # branches so all downstream transition/segment logic is shared.
    if pace_model == "viterbi" and vary_pace and len(beat_times) > 0:
        tiers = _pace_tiers_viterbi(audio_path, beat_times, bpm)
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
    )
