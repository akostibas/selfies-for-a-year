"""Simulate "burst decay" cutting: every note strike fires the same photo burst.

The idea (product owner, 2026-07-24): inside onset-anchored spans, don't cut once
per N strikes -- take each strike as a trigger and fan photos out after it at
geometrically growing gaps (e.g. 100ms, 160ms, 256ms, ...). The SHAPE is constant
at every strike, so the viewer learns it and can predict it; only the trigger
times vary with the playing. It is an impulse response, which is what a struck
string actually does.

Retrigger semantics: a new strike damps the previous burst, exactly like damping
a piano key. That gives the shape its useful emergent property -- dense passages
truncate into fast cutting, sparse passages get the full decay and settle.

What this measures, because it decides feasibility:
  1. photo cost (we have ~1182 across all years; Home is 371s)
  2. realized photos/sec, vs the 0.42 that was judged too slow
  3. holds below the legibility floor -- a photo too brief to register is a
     wasted selfie, and the project's guiding star is to favor false negatives
  4. how often bursts truncate, i.e. whether the emergent property fires at all

Run:
  uv run python experiments/strike_burst.py "<audio>" [--floor-ms 100]
"""

import argparse
from pathlib import Path

import librosa
import numpy as np

from selfies_for_a_year import beats as B

# A photo held under this reads as a flash, not an image you saw. We shipped a
# 50ms frame as a BUG once (overlapping anchor spans, fixed 2026-07-24); the
# lesson is that sub-100ms holds burn a photo without showing it.
LEGIBILITY_FLOOR_MS = 100.0


def burst_gaps(first_ms: float, ratio: float, depth: int) -> list[float]:
    """The constant shape: `depth` photos, each gap `ratio` x the previous."""
    return [first_ms * ratio**k / 1000.0 for k in range(depth)]


def simulate(strikes: np.ndarray, spans, duration: float,
             first_ms: float, ratio: float, depth: int, floor_ms: float = 0.0):
    """Fire the burst at every strike inside a span; a later strike truncates.

    floor_ms > 0 applies the no-sliver rule: never emit a photo that the next
    event (the next burst step or the damping strike) would cut shorter than the
    floor. Without it, truncation leaves the burst's last photo with whatever
    sliver remains before the strike -- a flash, and a wasted selfie.
    """
    gaps = burst_gaps(first_ms, ratio, depth)
    cuts: list[float] = []
    in_span = [t for t in strikes if any(a <= t < b for a, b in spans)]
    if floor_ms:
        # Strikes closer than the floor can't each get a legible photo, so merge
        # them into one trigger rather than dropping a photo onto a sliver.
        merged = [in_span[0]] if in_span else []
        for t in in_span[1:]:
            if t - merged[-1] >= floor_ms / 1000.0:
                merged.append(t)
        in_span = merged
    for i, t in enumerate(in_span):
        nxt = in_span[i + 1] if i + 1 < len(in_span) else duration
        c = t
        for k, g in enumerate(gaps):
            if c >= nxt - 1e-9:      # damped by the next strike
                break
            # The photo ON the strike always fires -- it is the anchor, and the
            # coalescing above already guaranteed it room. Only decay steps are
            # subject to the floor.
            if k > 0 and floor_ms and min(c + g, nxt) - c < floor_ms / 1000.0:
                break                # would be a sliver -- hold the last photo instead
            cuts.append(c)
            c += g
    return sorted(cuts), len(in_span)


def report(name, cuts, n_strikes, duration, spans, floor_ms):
    holds = np.diff(np.array(cuts)) if len(cuts) > 1 else np.array([])
    span_s = sum(b - a for a, b in spans)
    short = int((holds < floor_ms / 1000.0).sum())
    # bursts that ran to completion vs damped early
    print(f"  {name:<22} photos {len(cuts):>5}  "
          f"{len(cuts) / max(span_s, 1e-9):>5.2f}/s in-span  "
          f"median hold {np.median(holds) * 1000 if len(holds) else 0:>6.0f}ms  "
          f"under floor {short:>4} ({100 * short / max(len(holds), 1):>4.1f}%)")
    return len(cuts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--floor-ms", type=float, default=LEGIBILITY_FLOOR_MS)
    ap.add_argument("--supply", type=int, default=1182, help="photos available")
    args = ap.parse_args()

    y, sr = librosa.load(args.audio, mono=True)
    duration = len(y) / sr
    hop = 512
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    strikes = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="time", backtrack=True
    )
    spans = B._onset_anchor_spans(beat_times, strikes, duration)
    span_s = sum(b - a for a, b in spans)
    in_span = [t for t in strikes if any(a <= t < b for a, b in spans)]

    print(f"{Path(args.audio).stem}")
    print(f"  {duration:.0f}s, {len(strikes)} strikes, {len(in_span)} inside "
          f"{len(spans)} anchor spans covering {span_s:.0f}s "
          f"({100 * span_s / duration:.0f}% of the song)")
    gapsz = np.diff(np.array(in_span))
    print(f"  gap between in-span strikes: median {np.median(gapsz) * 1000:.0f}ms, "
          f"p10 {np.percentile(gapsz, 10) * 1000:.0f}ms, "
          f"p90 {np.percentile(gapsz, 90) * 1000:.0f}ms")
    print(f"  today: 1 cut per 2 strikes in ambient -> ~{len(in_span) / 2:.0f} photos in-span")
    print(f"  photo supply: {args.supply}\n")

    print(f"  legibility floor {args.floor_ms:.0f}ms; "
          "'under floor' = photos too brief to register")
    for enforce in (False, True):
        print(f"\n  === no-sliver rule {'ON' if enforce else 'OFF'} ===")
        for depth in (3, 4, 5):
            print(f"  depth {depth}:")
            for first_ms, ratio in ((100, 1.6), (100, 2.0), (150, 1.6), (200, 1.5)):
                cuts, _ = simulate(strikes, spans, duration, first_ms, ratio, depth,
                                   floor_ms=args.floor_ms if enforce else 0.0)
                n = report(f"first {first_ms}ms x{ratio}", cuts, len(in_span),
                           duration, spans, args.floor_ms)
                if n > args.supply:
                    print(f"  {'':<24}^ EXCEEDS SUPPLY by {n - args.supply}")


if __name__ == "__main__":
    main()
