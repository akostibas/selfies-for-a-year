"""Measure the DOWNBEAT phase of a track — which beat of the bar is beat 1 —
without rendering, and without a heavy DBN dependency (madmom won't build on
py3.14/numpy2.4).

waveform-eng's ranked-vote design (agent-chat 'audio', Part 4). Downbeat is a
STRUCTURE question, not an accent question: the loudest beat is usually the
backbeat (snare on 2 & 4), so max-onset phase is a TRAP. Instead vote three
probes for the phase p in {0..P-1} that is beat 1 of the bar, where
P = 4 * tick_division detected beats per bar:

  1. CHROMA-FLUX phase (primary) — chords change AT the barline, so beat-to-beat
     chroma change spikes on the downbeat. Works even on an accent-free ballad.
  2. BASS-BAND onset phase — bass lands on 1 even in backbeat genres. The
     critical vote for a track whose snare dominates the broadband onset.
  3. BROADBAND onset phase (weak tiebreak only) — the backbeat trap; low weight.

Confidence = agreement of the per-16-bar-window winners. If the winner flips
window-to-window, confidence is LOW and a per-song config bit should win instead
of a silent guess. The phase is also cross-checked against the felt parity we
already ship: a valid downbeat must sit on the felt-beat parity.

Usage:
  uv run python experiments/downbeat_phase.py "<audio>" [tick_div]
    tick_div: 1 or 2 (default: cheap heuristic occ<0.5 & bpm>100 -> 2 else 1)
"""
import sys
from pathlib import Path
import numpy as np


def _bass_onset_env(y, sr, fmax=150.0):
    """Onset strength restricted to the sub-~150 Hz band (kick + bass)."""
    import librosa
    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    band = S[freqs <= fmax, :]
    return librosa.onset.onset_strength(S=librosa.power_to_db(band + 1e-10), sr=sr)


def _phase_profile(per_beat, P, parity):
    """Mean of per_beat grouped by index mod P, but only over beats that sit on
    the felt parity (p % 2 == parity when P is even and tick_div forces it).
    Returns the length-P profile (nan where a class is off-parity)."""
    prof = np.full(P, np.nan)
    for p in range(P):
        idx = np.arange(p, len(per_beat), P)
        if len(idx):
            prof[p] = float(np.nanmean(per_beat[idx]))
    return prof


def _winner_on_parity(prof, parity):
    """Argmax of prof, restricted to phases on the felt parity."""
    P = len(prof)
    cand = [p for p in range(P) if p % 2 == parity] if P % 2 == 0 else list(range(P))
    cand = [p for p in cand if not np.isnan(prof[p])]
    if not cand:
        cand = [p for p in range(P) if not np.isnan(prof[p])]
    return max(cand, key=lambda p: prof[p]) if cand else 0


def main(argv):
    import librosa
    path = Path(argv[0]).expanduser()
    y, sr = librosa.load(str(path), mono=True)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bpm, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(bpm)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # tick_division heuristic (same one that sets the metronome dot rate)
    from selfies_for_a_year.beats import _spectral_occupancy
    occ = _spectral_occupancy(y, sr)
    tick_div = int(argv[1]) if len(argv) > 1 else (2 if (occ < 0.5 and bpm > 100) else 1)
    P = 4 * tick_div

    # felt parity: stronger-onset every-other beat (what the scheduler already uses)
    strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
    strengths = strengths / (strengths.max() or 1.0)
    parity = 0 if strengths[0::2].mean() >= strengths[1::2].mean() else 1

    # --- probe 1: chroma flux per beat (change INTO this beat) ---
    # Full-mix chroma is smeared by broadband percussion, flattening the flux.
    # Separate the harmonic layer first and use energy-normalized CENS chroma so
    # the barline chord change actually shows up. (Set DB_RAW_CHROMA=1 to compare.)
    import os
    if os.environ.get("DB_RAW_CHROMA"):
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    else:
        y_harm = librosa.effects.harmonic(y, margin=4.0)
        chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
    chroma_beat = librosa.util.sync(chroma, beat_frames, aggregate=np.mean)  # 12 x n_beats
    cb = chroma_beat / (np.linalg.norm(chroma_beat, axis=0, keepdims=True) + 1e-9)
    cos = np.sum(cb[:, 1:] * cb[:, :-1], axis=0)
    chroma_flux = np.concatenate([[0.0], 1.0 - cos])  # len n_beats, flux at beat b

    # --- probe 2: bass-band onset per beat ---
    bass_env = _bass_onset_env(y, sr)
    bass_beat = bass_env[np.clip(beat_frames, 0, len(bass_env) - 1)]
    bass_beat = bass_beat / (bass_beat.max() or 1.0)

    # --- probe 3: broadband onset per beat (weak) ---
    broad_beat = strengths

    nb = len(beat_frames)
    probes = {"chroma_flux": chroma_flux[:nb], "bass_onset": bass_beat[:nb],
              "broadband": broad_beat[:nb]}
    weights = {"chroma_flux": 1.0, "bass_onset": 0.8, "broadband": 0.25}

    # overall winner per probe
    profs = {k: _phase_profile(v, P, parity) for k, v in probes.items()}
    wins = {k: _winner_on_parity(profs[k], parity) for k in probes}

    # weighted vote across probes (normalize each profile to [0,1] first)
    combined = np.zeros(P)
    for k, prof in profs.items():
        pr = np.nan_to_num(prof, nan=np.nanmin(prof))
        rng = pr.max() - pr.min()
        norm = (pr - pr.min()) / rng if rng > 0 else pr * 0
        combined += weights[k] * norm
    downbeat_phase = _winner_on_parity(combined, parity)

    # --- confidence: profile CONTRAST, not argmax stability ---
    # A flat profile's argmax is stably meaningless (Push reported "HIGH 75%"
    # off pure noise). Measure how much the winning phase stands out from the
    # rest: (max - median)/(max - min) of the combined vote profile. Below the
    # threshold there is no real peak -> emit NO seed and fall back to phase 0.
    valid = combined[~np.isnan(combined)]
    if len(valid) and (valid.max() - valid.min()) > 0:
        contrast = float((valid.max() - np.median(valid)) / (valid.max() - valid.min()))
    else:
        contrast = 0.0
    SEED_MIN_CONTRAST = 0.60
    confidence = "HIGH" if contrast >= 0.75 else ("MED" if contrast >= SEED_MIN_CONTRAST else "LOW")
    seed_phase = downbeat_phase if contrast >= SEED_MIN_CONTRAST else 0
    seeded = contrast >= SEED_MIN_CONTRAST

    print(f"\n=== {path.stem[:48]} ===")
    print(f"  bpm(median-IBI)~{60.0/np.median(np.diff(beat_times)):.1f}  occ={occ:.2f}  "
          f"tick_div={tick_div}  P={P} detected-beats/bar  felt_parity={parity}")
    print(f"  per-probe downbeat phase (mod {P}):")
    for k in probes:
        pr = profs[k]
        prof_s = " ".join(f"{'*' if p==wins[k] else ' '}{('%.2f'%pr[p]) if not np.isnan(pr[p]) else '  - '}"
                          for p in range(P))
        print(f"    {k:12s} -> phase {wins[k]}   [{prof_s}]")
    print(f"  VOTED downbeat phase = {downbeat_phase}  (mod {P})")
    print(f"  on felt parity? {'YES' if downbeat_phase % 2 == parity or P % 2 else 'NO — CONFLICT'}")
    print(f"  contrast={contrast:.2f} -> {confidence}   "
          f"SEED = {'phase %d' % seed_phase if seeded else 'none (default 0)'}")


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main(sys.argv[1:])
