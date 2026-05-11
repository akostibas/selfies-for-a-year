# Face-Finding Approach

Living document. High-level overview of how we decide which photos to keep,
where the face is in each photo, and what we use to drive the alignment.

## Sources of face information

- **Apple Photos DB hint** — `ZDETECTEDFACE.ZCENTERX/ZCENTERY/ZSIZE` from the
  Photos SQLite store. Available only for photos imported via the Photos
  integration; gives a center point and face area as fractions of the image.
  Sometimes authoritative, sometimes wrong — especially for older scans where
  the user may have manually drawn a box around the whole person rather than
  just the face.
- **MediaPipe FaceLandmarker** — Our current primary detector. Produces 468
  facial landmarks per face when it succeeds; we use the eye corners for
  similarity-transform alignment. Strong on modern phone selfies, weak on
  older/lower-quality scans.
- **MediaPipe FaceDetector (BlazeFace short-range)** — Lighter face detector
  used as the *seed* of a two-stage flow: when FaceLandmarker fails on the
  full image, BlazeFace at conf 0.5 finds the face, we crop a tight ROI
  around it, and re-run FaceLandmarker on the crop with a permissive
  threshold (0.1). Hint agreement on the BlazeFace box is what justifies
  the relaxed landmarker confidence. When BlazeFace returns multiple boxes
  within the hint radius, we pick the one *closest to the hint*, not the
  highest-scoring one — the hint identifies the person, and a confident
  detection of the wrong face is worse than a less confident detection of
  the right one.

## Weighting inputs for keep/drop decisions

- **No face detected by any model** → drop the frame. A timelapse with fewer
  frames but every frame showing a real face is better than one peppered with
  misaligned shots of bodies or backgrounds.
- **One face detected, matches Apple's hint within tolerance** → keep, align
  using landmarks.
- **One face detected, does *not* match Apple's hint** → our recognition
  usually wins. Apple's hint may have been a manually drawn bounding box
  around a whole person.
- **Multiple faces detected** → the hint *identifies the person*, so the
  detection closest to the hint is "the target". We never substitute a
  different face that happens to be near the hint, because that picks the
  wrong sibling/friend/parent. The closest face is then yaw-gated like any
  other; if it fails yaw or sits outside `_FACE_MATCH_RADIUS` face-widths,
  we drop the frame rather than fall back to a different face.
- **Manually curated (selfie-dir) photos** → looser rules. User already vetted
  these, so center-crop fallback is acceptable when detection fails.

## Weighting inputs for alignment

- **Eye landmarks from FaceLandmarker** are the gold input — we compute a
  similarity transform (rotation + uniform scale + translation) that maps the
  detected eyes to canonical output positions. Landmarks come from either
  stage 1 (full-image landmarker at conf 0.5) or stage 2 (permissive
  landmarker on a BlazeFace-cropped ROI).
- **Center-crop** is the only non-landmark fallback, used exclusively for
  manually curated `selfie-dir` photos when detection fails (the user vetted
  those, so we don't silently drop them). Apple-Photos sourced frames with
  no usable face are dropped from the video instead of being hint-cropped —
  matching the "favor false negatives" principle.

## Quality gates

Independent of which stage produced the landmarks, we apply three gates:

- **Hint match radius (`_FACE_MATCH_RADIUS = 0.75`)** — detected face center
  must sit within this many face-widths of the Apple hint. Filters out
  multi-face misdetections (e.g. picking a sibling).
- **Yaw filter (`_MAX_YAW = 0.25`)** — projection of the nose offset onto
  the eye-line, normalized by inter-eye distance. 0 = forward, ±0.5 = true
  profile, |val| > 0.5 implies geometrically impossible / hallucinated
  landmarks. 0.25 ≈ ~30° head turn. Faces past this are skipped to keep
  video continuity from being broken by a sideways head among forward
  ones.
- **Key-points-in-bounds** — both eye centers and the nose tip must lie
  within the image rectangle. MediaPipe extrapolates landmarks for
  off-image face parts (returning all 468 even when half the face is
  outside the frame), so a face whose key alignment points sit at
  negative or out-of-range coordinates is cut off. We don't trust
  extrapolated pixels for alignment, so we drop the frame. This catches
  e.g. older scans where the camera caught only an ear, or where the
  face is half-cropped at the top.

## Quality signals (current + aspirational)

- **Face area fraction** — we can use Apple's `ZSIZE` and our own detector's
  bounding box to estimate how prominent the face is. Borderless phone
  selfies sit in a typical range; Polaroid-bordered scans read smaller
  because of the white margin.
- **Detector agreement** — when Apple's hint and our detector agree on
  location within tolerance, confidence is high. Disagreement is a red flag
  that warrants stricter decisions.
- **Landmark success** — if FaceLandmarker produces landmarks (not just a
  bounding box), we have enough information to align properly.

## Defaults and knobs

- Default stance: **favor false negatives**. Drop uncertain frames rather
  than include questionable ones. Thresholds can be relaxed over time once
  the baseline is clean.
- `_LM_CONF` (0.5) — stage-1 FaceLandmarker confidence on the full image.
- `_BF_CONF` (0.5) — stage-2 BlazeFace confidence; we want a confident seed.
- `_STAGE2_LM_CONF` (0.1) — permissive landmarker confidence on the crop.
- `_ROI_MULT` (2.0) — stage-2 crop side length as a multiple of the BF
  bbox's larger dimension. Puts the face at ~50% of the crop dim.
- `_FACE_MATCH_RADIUS` (0.75) — maximum distance (in face-widths) between
  Apple's hint and our detection.
- `_MAX_YAW` (0.25) — maximum |nose-offset projection| before a face is
  rejected as too turned (or hallucinated).
