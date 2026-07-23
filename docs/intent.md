# selfies-for-a-year — Product Intent

This is the project's living north star: the utility selfies-for-a-year aims
to provide and its deliberate non-goals. It changes only when the *desire*
for the product changes — never because the implementation did. The test
this document must pass: an agent reading only this page should be able to
recognize when the tool is malfunctioning, and brainstorm improvements for
it.

## Purpose

1. selfies-for-a-year exists to evoke the evolution of one person through
   time — years of change compressed into minutes — built from their own
   materials: the daily selfies they took and the music they love, compiled
   into a keepsake they're proud to watch and share.

## Principles

2. The person is the still center that time visibly moves past — the face
   steady frame after frame while hair, light, seasons, and age flow — and
   every technique in the pipeline (alignment, cropping, beat-sync) exists
   to serve that effect.
3. The effect is a spell, and one bad frame breaks it where a gap never
   would — so when a photo is uncertain, drop it: fewer frames, every one
   right, beats complete coverage.
4. The result should feel like an authored portrait, not generated output —
   pacing that breathes with the music, cuts that land on beats — and when
   mechanical uniformity and feel conflict, favor feel.
5. The materials are the person's life as actually recorded — Apple Photos,
   ad-hoc folders, mixed formats and orientations — and the tool adapts to
   that mess rather than demanding a curated input.
6. The spell breaks at specific moments ("the April 2022 frame looks
   wrong"), so the path from a flagged moment back to its source photo must
   stay short, letting fixes target causes rather than one-off exclusions.
7. Getting the feel right takes many render–review–tweak cycles, so
   anything that makes a test render slower to produce or harder to judge
   slows the whole product down.

## Deliberate non-goals

8. **A general-purpose video editor** — one job: compressing a person's
   timeline into a watchable evolution, with no timelines, clips, titles,
   or effects beyond what serves that job.
9. **A photo manager or curator** — the tool reads the person's materials
   but never organizes, tags, edits, or writes to them, and frames it drops
   are dropped only from the render, never from the source.
10. **Manual per-photo tuning** — the pipeline should get frames right by
    rule, and recurring hand-exclusions signal a threshold or detector that
    needs improving, not a workflow to embrace.
