# What We Tried

Approaches and design choices we moved on from, preserved here for context.

## Portrait aspect ratio (1080x1920) with small face

**What:** The initial output was 1080x1920 portrait with eyes positioned at 38%/62% of frame width (~260px inter-eye distance). The face occupied roughly 1/4 of the frame width.

**Problem:** iPhone front camera selfies are 4:3 (3024x4032 after EXIF orientation). Mapping them to 9:16 portrait with a small face required large affine transforms that frequently extended past the source image boundaries, producing black letterbox bars. When the subject wasn't centered in the original photo, the problem was severe.

**Resolution:** Switched to 1080x1080 square output with a much larger face (eyes at 30%/70% of frame width, ~432px inter-eye distance). Square crops more naturally from 4:3 sources, and the larger face means less extreme transforms.

## Reusing previous transform on detection failure

**What:** When mediapipe failed to detect a face in a frame, we reused the affine transform from the most recently successful frame.

**Problem:** If a frame had an off-center or unusually positioned face, its transform could be extreme. When the next frame's detection failed, it inherited that extreme transform, producing a badly misaligned frame. This cascaded — a single odd frame could throw off subsequent frames.

**Resolution:** Changed to always fall back to center-crop when detection fails. This produces a less-smooth transition for that single frame but prevents cascading misalignment. In practice, detection failure is rare (~2% of frames), so the occasional center-cropped frame is much less disruptive than a string of badly warped ones.

## Face too large at 30%/70% eye positions

**What:** After switching to square output, we set the eye positions at 30%/70% of frame width to fill ~3/4 of the frame, hoping to minimize out-of-bounds areas.

**Problem:** Way too close up — felt like a mugshot.

**Resolution:** Reverted to 38%/62% eye positions (24% inter-eye distance). The square frame + blur fill already solved the letterboxing problem, so we didn't need the face to be so large.

## Color normalization: full LAB histogram matching

**What:** Matched each frame's per-channel mean and standard deviation in LAB color space to a rolling average of its neighbors (window=11). Goal was to smooth brightness and color jitter between frames.

**Problem:** Matching both mean *and* std on all three channels flattened contrast and made well-lit photos look bland. The variance decreased but the images lost their character.

**Status:** Code is still in `color.py` but the approach was replaced.

## Color normalization: brightness-only with contrast compensation

**What:** Shifted only the L channel (brightness) toward a rolling average, then applied an inversely proportional contrast adjustment — darkened frames got a contrast boost, brightened frames got a reduction.

**Problem:** Still too finicky. The contrast compensation helped compared to the full histogram approach, but the results were inconsistent — some frames improved while others looked unnatural. Hard to find settings that work well across varied lighting conditions (indoor, outdoor, morning, evening).

**Status:** Code remains in `color.py`, available via `--normalize` flag (off by default). May revisit with a different approach (e.g., local tone mapping, or matching only extreme outliers rather than every frame).
