# Render scorecard

A render can be "beatmatched well" and still feel wrong — because *matching the
beat*, *matching the song's pace*, and *matching the song's progression* are
three independent things. Judging them as one blurs which one broke. This
scorecard scores them separately, each with an **objective sub-metric** (computed
from the timeline, no eyeballing) and a **felt** rating (1–5, the human call the
numbers can't make).

Score every review render. A dimension that regresses tells you which knob to
turn without re-litigating the others.

---

## ① BEAT MATCH — *do cuts land on the beat?*

The mechanical layer. Purely objective; the ear only confirms.

| sub-metric | how it's measured | 5 (great) | 3 (ok) | 1 (bad) |
|---|---|---|---|---|
| mean \|cut → nearest beat\| | over all hard cuts | < 15 ms | < 40 ms | audibly off |
| on-beat rate | % cuts within ±30 ms | > 95 % | > 80 % | < 80 % |
| systematic drift | signed mean offset (early/late) | \|drift\| < 10 ms | < 25 ms | consistent lag/lead |

## ② SONG PACE MATCH — *is the base density right for the song's energy?*

The tempo layer — **the one Home failed.** A ballad should linger; a banger
should drive. Measured on the **normal tier** (the song's baseline groove),
because intense/slow are deliberate excursions from it.

| sub-metric | how it's measured | 5 (great) | 3 (ok) | 1 (bad) |
|---|---|---|---|---|
| felt tempo | detected BPM, ÷2 if double-time detected | — | — | — |
| base cadence | normal-tier photos/s | — | — | — |
| target band | derived from felt tempo (≈ 1 photo per 1–2 felt beats) | — | — | — |
| actual / target | ratio | 0.8–1.2× | 1.2–1.6× | > 2× (frantic) or < 0.5× (sleepy) |

**Calibration open question (under active design):** the target-band formula
has two inputs, both being worked out with the audio engineer:

1. **Felt tempo, not detected BPM.** Both Push and Home detect 143.6, but Home
   is a ~72 felt (double-time octave error). The metronome ticks the detected
   beat, so "1 photo per 4 ticks" means different things at 143.6 vs 72. The
   grid/pace must use the felt tempo.
2. **Spectral occupancy sets photos-per-beat** (the owner's "amount of black"
   idea). Fraction of mel cells above −50 dB, per song: Push 0.81, RJD2 0.70,
   JWilliams 0.41, TBAH 0.38 — a clean dense/sparse split that matches the eye.
   Sparse song (much black) → fewer photos per beat; dense → more.

Worked once the two are pinned: Home felt 72 + occ 0.38 → slow, lingering base;
Push felt ~126 + occ 0.81 → driving base. Owner's concrete anchor: **Home intro
= 1 photo per 4 felt ticks**.

## ③ SONG PROGRESSION MAPPING — *do the tiers follow the song's structure?*

The arrangement layer — **the one Push just nailed.** Does it breathe: intense
where the song lifts, slow/ambient in the lulls?

| sub-metric | how it's measured | 5 (great) | 3 (ok) | 1 (bad) |
|---|---|---|---|---|
| tier variety | distinct tiers used / entropy | 3–4 tiers, well spread | 2 tiers | 1 tier (flat) |
| structural alignment | tier changes near Foote novelty boundaries | most changes on boundaries | some | random |
| ground-truth hits | owner's known sections matched (e.g. Push "2 intense: intro + last⅓") | all | most | misses |

---

## The form (fill one per render)

```
━━━ RENDER SCORECARD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
render: ________________   song: ________________  date: ______

① BEAT MATCH
   mean |cut-beat|: ____ ms   on-beat(±30): ____ %   drift: ____ ms   [auto]
   objective: _/5     felt: _/5
   notes: _______________________________________________________

② SONG PACE MATCH
   detected BPM: ____   felt tempo: ____   base cadence: ____ ph/s   [auto]
   target band: ____–____   actual/target: ____×                    [auto]
   objective: _/5     felt: _/5
   notes: _______________________________________________________

③ SONG PROGRESSION MAPPING
   tiers: amb__ slo__ nor__ int__   variety: ____   gt-hits: __/__   [auto]
   objective: _/5     felt: _/5
   notes: _______________________________________________________

OVERALL  _/15        verdict:  ship │ iterate │ reject
weakest dimension → the knob to turn next: _______________________
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Worked example — the two current renders

```
② PACE   Home  detected 143.6 → felt 72 → target 0.30/s, actual 0.60/s = 2.0×  → 1/5  ✗ "lost magic"
         Push  detected 143.6 → felt ~126 → target 0.35/s, actual 0.37/s = 1.06× → 5/5  ✓
③ PROG   Push  4 tiers, intro+last-⅓ intense hit                                → 5/5  ✓ "great"
```

Home scores well on ③ (progression) but fails ② (pace) — which is why it can be
"correct" and still feel wrong. The fix is dimension-2-only: halve the base to
the felt tempo, leave the progression alone.

## Not yet built

The objective columns are computable from the timeline (`--analyze-only` already
emits cadences); a `score_render.py` that prints the filled form is the natural
next step. The **felt** ratings stay human — the scorecard structures the
judgment, it doesn't replace it.
