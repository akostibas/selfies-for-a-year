# Model Evaluation

A running log of what we've tested for face detection and alignment. This
document tracks aggregated statistics — not per-photo details — so we can
compare detectors objectively as the approach evolves.

No photo filenames, dates, or paths appear here by design. Identifying data
lives outside the repo.

## Models under evaluation

| Model | File | Purpose |
|-------|------|---------|
| MediaPipe FaceLandmarker | `face_landmarker.task` | 468-point landmarks, current primary |
| MediaPipe FaceDetector (BlazeFace short-range) | `blaze_face_short_range.tflite` | Lightweight face detection, candidate fallback |
| Apple Photos DB | `ZDETECTEDFACE` table | Ground-truth hint from the Photos library |

## Data sources

- **Apple Photos library** — spans several decades, includes scanned
  analog photos, consumer digital cameras, and modern smartphone selfies.
  Substantial variation in resolution, color, and framing (Polaroid borders
  on some scans, cropped frames on others).
- **Curated selfie directory** — manually selected 2026 phone selfies,
  homogeneous in resolution and framing.

## Method

For each photo in a test set we record:
- Whether each detector found at least one face
- Whether the detected face (if any) matches the user's ground truth (visual check or agreement with Apple hint)
- Whether the detected face is close enough to the Apple hint to pass the match-radius check

## Results

### Pilot: a single challenging historical scan (1 photo)

A childhood scan where FaceLandmarker fails and Apple's hint is placed on the
subject's torso rather than the face. Not statistically meaningful on its own
— logged here as the seed example that motivated BlazeFace testing.

| Configuration | Detections | Correct | False positives | False negatives |
|---|---|---|---|---|
| FaceLandmarker, conf=0.50 | 0 | 0 | 0 | 1 |
| FaceLandmarker, conf=0.30 | 0 | 0 | 0 | 1 |
| FaceLandmarker, conf=0.10 | 0 | 0 | 0 | 1 |
| FaceLandmarker, conf=0.05 | 0 | 0 | 0 | 1 |
| FaceLandmarker, conf=0.01 | 0 | 0 | 0 | 1 |
| FaceLandmarker, conf=0.10, 2x upscale | 0 | 0 | 0 | 1 |
| BlazeFace short-range, conf=0.50 | 1 | 1 | 0 | 0 |
| BlazeFace short-range, conf=0.10 | 6 | 1 | 5 | 0 |

Takeaway: BlazeFace can find faces that FaceLandmarker's built-in detector
misses entirely. Low confidence thresholds on BlazeFace introduce many false
positives, so a post-filter (size range, hint agreement) is needed.

### Larger historical sweep

_Pending. Planned methodology: sample N photos per year across the full
library, compare detector outputs against manual ground truth, report
per-decade precision/recall._

### Apple hint coordinate convention

Sampled 9 photos across 1988→2026 and compared `ZDETECTEDFACE.ZCENTERY`
against the strongest BlazeFace detection's vertical center.

| Interpretation | Mean |center_y error| |
|---|---|
| Raw `ZCENTERY` (top-down, naive) | 0.267 |
| `1 - ZCENTERY` (CG-space flip) | 0.057 |

Conclusion: Apple stores Y in Core Graphics convention (origin bottom-left,
Y points up). X requires no flip. Fix applied in `photos.py:query_photos`.

This explained the long-standing "Apple hint lands on the torso" behavior on
the 1988 tee-ball pilot photo and likely degraded multi-face matching and
fallback hint crops across the entire library prior to the fix.

## Open questions

- At what confidence threshold does BlazeFace's precision drop below an
  acceptable level?
- Does running FaceLandmarker on a crop around a BlazeFace detection recover
  landmarks on photos where FaceLandmarker-on-full-image fails?
