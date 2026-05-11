"""Face detection and alignment using mediapipe FaceLandmarker + BlazeFace.

Two-stage flow per frame:
  1. FaceLandmarker on the full image at production confidence.
  2. If that fails AND we have an Apple-Photos hint: BlazeFace seeds a tight
     crop around the face, then FaceLandmarker re-runs on the crop with a
     permissive threshold. Hint agreement on the BlazeFace box is what
     justifies the more permissive landmarker confidence.

Quality gates after either stage:
  - Hint match radius — detected face must sit within a small number of
    face-widths from the Photos DB hint.
  - Yaw filter — nose must sit roughly centered between the eyes; faces
    more than ~30° turned (or geometrically incoherent) are dropped.

When everything fails, behavior depends on the source:
  - Apple-Photos image: return None (caller drops the frame).
  - selfie-dir image: center-crop fallback, since the user manually vetted
    those and we don't want to silently drop them.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter

# Mediapipe FaceMesh landmark indices for key points.
_LEFT_EYE_INDICES = [33, 133]   # inner and outer corners of left eye
_RIGHT_EYE_INDICES = [362, 263] # inner and outer corners of right eye
_NOSE_TIP_INDEX = 1             # nose tip in the 468-point face mesh

# Face perimeter (jaw + chin + cheeks + brow), 36 indices forming a closed
# contour. From MediaPipe's FACEMESH_FACE_OVAL connections.
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

# Drop the frame if at least this many face-oval landmarks fall outside
# the image. The eyes/nose can sit just inside a frame edge while the
# rest of the face extends way off-image; that produces blurry-half-face
# alignments. Probed against 200 random kept photos: 94% had 0 oval
# landmarks off, max observed normal=11; the 2022-04-24 cut-off case
# had 21. Threshold 12 = one-step buffer over observed normals, well
# below the bad case.
_MAX_OVAL_OFF = 12

# Canonical face position as fractions of output dimensions.
# Face fills ~3/4 of frame width (eyes at 30% and 70% of width).
_EYE_Y_FRAC = 0.38        # eyes at 38% from top
_LEFT_EYE_X_FRAC = 0.38   # left eye at 38% from left
_RIGHT_EYE_X_FRAC = 0.62  # right eye at 62% from left

# If mediapipe's best-match face center is farther than this many face-widths
# from the Photos DB hint, treat it as a misdetection and reject.
_FACE_MATCH_RADIUS = 0.40

# Stage-1 / production FaceLandmarker confidence threshold.
_LM_CONF = 0.5

# Stage-2 BlazeFace confidence — we want a confident seed box.
_BF_CONF = 0.5

# Stage-2 FaceLandmarker confidence on the crop. Permissive because
# BlazeFace + hint agreement already filtered most false positives.
_STAGE2_LM_CONF = 0.1

# Stage-2 crop side length = _ROI_MULT * max(bbox_w, bbox_h), centered on
# the BlazeFace box. 2.0 makes the face occupy ~50% of each crop dim,
# which is what FaceLandmarker likes to see.
_ROI_MULT = 2.0

# Maximum allowed |yaw signal|. Yaw signal is the projection of the nose
# offset (from the inter-eye midpoint) onto the eye-line, normalized by
# inter-eye distance. 0 = forward face, ±0.5 = profile, |val| > 0.5 means
# landmarker hallucinated landmarks. 0.25 ≈ ~30° head turn.
_MAX_YAW = 0.25

# Model paths relative to the project root.
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_LANDMARKER_PATH = _MODELS_DIR / "face_landmarker.task"
_BLAZEFACE_PATH = _MODELS_DIR / "blaze_face_short_range.tflite"

# A face represented as (left_eye, right_eye, nose, oval) in pixel coords.
# `oval` is the 36 face-perimeter landmark positions in pixel coords; used
# by _face_oval_in_bounds to detect cut-off faces.
_Face = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    list[tuple[float, float]],
]


def _eye_center(
    landmarks: list[object],
    indices: list[int],
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """Average the landmark positions to get the eye center in pixel coords."""
    xs = [landmarks[i].x * img_w for i in indices]
    ys = [landmarks[i].y * img_h for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _key_points_in_bounds(face: _Face, img_w: int, img_h: int) -> bool:
    """True iff both eyes and the nose lie within the image rectangle.

    MediaPipe extrapolates landmarks for face parts that are off-image (a
    cut-off face still gets all 468 landmarks, just with some at negative
    or out-of-range coordinates). For alignment we depend on the eyes and
    nose being real pixels, so a face whose key points are off-image is
    untrustworthy and gets dropped.
    """
    for x, y in face[:3]:
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            return False
    return True


def _face_oval_in_bounds(face: _Face, img_w: int, img_h: int) -> bool:
    """True iff fewer than _MAX_OVAL_OFF face-perimeter landmarks are
    off-image. Catches partly-cut-off faces that still have eyes/nose
    inside the frame but extend past an edge."""
    off = sum(
        1 for x, y in face[3]
        if not (0 <= x <= img_w and 0 <= y <= img_h)
    )
    return off < _MAX_OVAL_OFF


def _yaw_signal(face: _Face) -> float:
    """Normalized nose offset along the eye-line.

    0  = nose centered between eyes (forward-facing)
    ±0.5 = nose under one eye (true profile / 90°)
    |val| > 0.5 = geometrically impossible → landmarker hallucinated
    """
    left, right, nose, _ = face
    mx, my = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    ex, ey = right[0] - left[0], right[1] - left[1]
    nx, ny = nose[0] - mx, nose[1] - my
    eye_dist_sq = ex * ex + ey * ey
    if eye_dist_sq == 0:
        return 0.0
    return (nx * ex + ny * ey) / eye_dist_sq


def _compute_similarity_transform(
    src_left: tuple[float, float],
    src_right: tuple[float, float],
    dst_left: tuple[float, float],
    dst_right: tuple[float, float],
) -> tuple[float, float, float, float, float, float]:
    """Compute a similarity transform (rotation + uniform scale + translation).

    Returns the 6 affine coefficients (a, b, c, d, e, f) for use with
    PIL's Image.transform() in AFFINE mode, which maps output pixels to
    input pixels: (x_in, y_in) = (a*x_out + b*y_out + c, d*x_out + e*y_out + f)
    """
    sx1, sy1 = src_left
    sx2, sy2 = src_right
    dx1, dy1 = dst_left
    dx2, dy2 = dst_right

    src_dx = sx2 - sx1
    src_dy = sy2 - sy1
    dst_dx = dx2 - dx1
    dst_dy = dy2 - dy1

    src_dist = math.hypot(src_dx, src_dy)
    dst_dist = math.hypot(dst_dx, dst_dy)

    scale = src_dist / dst_dist

    src_angle = math.atan2(src_dy, src_dx)
    dst_angle = math.atan2(dst_dy, dst_dx)
    angle = src_angle - dst_angle

    cos_a = scale * math.cos(angle)
    sin_a = scale * math.sin(angle)

    # PIL AFFINE maps OUT->IN: (x_in, y_in) = R*(x_out, y_out) + t
    # where R = [[cos_a, -sin_a], [sin_a, cos_a]] for rotation by `angle`.
    # tx, ty are derived to satisfy R*dst_left + t = src_left.
    tx = sx1 - cos_a * dx1 + sin_a * dy1
    ty = sy1 - sin_a * dx1 - cos_a * dy1

    return (cos_a, -sin_a, tx, sin_a, cos_a, ty)


def _apply_transform_with_blur_fill(
    img: Image.Image,
    transform: tuple[float, float, float, float, float, float],
    target_width: int,
    target_height: int,
) -> Image.Image:
    """Apply an affine transform, filling out-of-bounds areas with a blurred
    version of the image instead of black."""
    bg = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))

    aligned = img.transform(
        (target_width, target_height),
        Image.Transform.AFFINE,
        transform,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )

    white = Image.new("L", img.size, 255)
    mask = white.transform(
        (target_width, target_height),
        Image.Transform.AFFINE,
        transform,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    return Image.composite(aligned, bg, mask)


class FaceAligner:
    """Detects faces and aligns them to a canonical position.

    Usage:
        with FaceAligner(1080, 1080) as aligner:
            aligned = aligner.align(img, face_hint=hint)
            if aligned is None:
                # frame was skipped (no usable face)
                continue
    """

    def __init__(self, target_width: int, target_height: int) -> None:
        self.target_width = target_width
        self.target_height = target_height

        # Canonical eye positions in output pixel space
        self._dst_left = (
            _LEFT_EYE_X_FRAC * target_width,
            _EYE_Y_FRAC * target_height,
        )
        self._dst_right = (
            _RIGHT_EYE_X_FRAC * target_width,
            _EYE_Y_FRAC * target_height,
        )

        for path, hint in [
            (_LANDMARKER_PATH, "face_landmarker.task / face_landmarker"),
            (_BLAZEFACE_PATH, "blaze_face_short_range.tflite / face_detector/blaze_face_short_range"),
        ]:
            if not path.exists():
                raise SystemExit(
                    f"Required model not found at {path}.\n"
                    f"Download it from MediaPipe's {hint} URL."
                )

        # Stage 1: production-confidence landmarker
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_PATH)),
                num_faces=4,
                min_face_detection_confidence=_LM_CONF,
            )
        )
        # Stage 2b: permissive landmarker, used only on BlazeFace-cropped ROIs
        self._landmarker_permissive = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_PATH)),
                num_faces=4,
                min_face_detection_confidence=_STAGE2_LM_CONF,
            )
        )
        # Stage 2a: BlazeFace finds where to crop
        self._blazeface = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_BLAZEFACE_PATH)),
                min_detection_confidence=_BF_CONF,
            )
        )

        self.aligned_count = 0   # frames aligned via landmarks (stage 1 or 2)
        self.recovered_count = 0 # subset of aligned_count that needed stage 2
        self.fallback_count = 0  # selfie-dir frames using center-crop
        self.skipped_count = 0   # Apple-Photos frames dropped (no usable face)

    def __enter__(self) -> FaceAligner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()
        self._landmarker_permissive.close()
        self._blazeface.close()

    def _detect_faces(self, img: Image.Image, *, permissive: bool = False) -> list[_Face]:
        """Run FaceLandmarker, return _Face tuples in image-pixel coords."""
        landmarker = self._landmarker_permissive if permissive else self._landmarker
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
        result = landmarker.detect(mp_img)
        faces: list[_Face] = []
        for landmarks in result.face_landmarks:
            left = _eye_center(landmarks, _LEFT_EYE_INDICES, img.width, img.height)
            right = _eye_center(landmarks, _RIGHT_EYE_INDICES, img.width, img.height)
            nose = (
                landmarks[_NOSE_TIP_INDEX].x * img.width,
                landmarks[_NOSE_TIP_INDEX].y * img.height,
            )
            oval = [
                (landmarks[i].x * img.width, landmarks[i].y * img.height)
                for i in _FACE_OVAL_INDICES
            ]
            faces.append((left, right, nose, oval))
        return faces

    def _pick_face(
        self,
        faces: list[_Face],
        face_hint: tuple[float, float, float] | None,
        img_w: int,
        img_h: int,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Pick the face that matches the hint, then yaw-gate it.

        The hint identifies the target person, not just a region — so the
        face closest to the hint is "the person" even if it fails yaw. If
        the hint-matched face fails yaw (or is hallucinated), skip the
        frame rather than substituting a different person who happens to
        be in the same neighborhood.

        Returns (left_eye, right_eye) for the chosen face, or None if no
        candidate passes the quality gates.
        """
        if not faces:
            return None

        if face_hint is None:
            # Without a hint we can only safely use a single forward face.
            usable = [
                f for f in faces
                if abs(_yaw_signal(f)) <= _MAX_YAW
                and _key_points_in_bounds(f, img_w, img_h)
                and _face_oval_in_bounds(f, img_w, img_h)
            ]
            if len(usable) != 1:
                return None
            left, right, _, _ = usable[0]
            return (left, right)

        hint_cx, hint_cy, hint_size = face_hint
        hint_x = hint_cx * img_w
        hint_y = hint_cy * img_h
        hint_face_dim = math.sqrt(max(hint_size, 1e-6) * img_w * img_h)

        def distance(face: _Face) -> float:
            left, right, _, _ = face
            face_cx = (left[0] + right[0]) / 2
            face_cy = (left[1] + right[1]) / 2
            return math.hypot(face_cx - hint_x, face_cy - hint_y)

        # The hint says "the target person is here". The closest detection
        # is therefore the target — never substitute a farther face.
        closest = min(faces, key=distance)
        if distance(closest) > _FACE_MATCH_RADIUS * hint_face_dim:
            return None
        if abs(_yaw_signal(closest)) > _MAX_YAW:
            return None
        if not _key_points_in_bounds(closest, img_w, img_h):
            return None
        if not _face_oval_in_bounds(closest, img_w, img_h):
            return None
        left, right, _, _ = closest
        return (left, right)

    def _two_stage_eyes(
        self,
        img: Image.Image,
        face_hint: tuple[float, float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """BlazeFace finds the face, then FaceLandmarker re-runs on a tight crop.

        Returns mapped-to-full-image (left_eye, right_eye) or None if no
        BlazeFace candidate matches the hint, or if landmarker fails on
        the crop, or if the recovered face fails the yaw / hint filters.
        """
        hint_cx, hint_cy, hint_size = face_hint
        hint_x = hint_cx * img.width
        hint_y = hint_cy * img.height
        hint_face_dim = math.sqrt(max(hint_size, 1e-6) * img.width * img.height)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
        result = self._blazeface.detect(mp_img)
        # Keep only BF detections within hint radius — that's what licenses
        # the permissive stage-2 landmarker confidence.
        candidates = []
        for d in result.detections:
            b = d.bounding_box
            cx = b.origin_x + b.width / 2
            cy = b.origin_y + b.height / 2
            dist = math.hypot(cx - hint_x, cy - hint_y)
            if dist <= _FACE_MATCH_RADIUS * hint_face_dim:
                candidates.append((b, dist))
        if not candidates:
            return None

        # Hint identifies the person — pick the BF detection closest to it,
        # not the highest-scoring one (which can be a different face entirely).
        best_box, _ = min(candidates, key=lambda c: c[1])
        bcx = best_box.origin_x + best_box.width / 2
        bcy = best_box.origin_y + best_box.height / 2
        side = _ROI_MULT * max(best_box.width, best_box.height)
        rx0 = max(0, int(bcx - side / 2))
        ry0 = max(0, int(bcy - side / 2))
        rx1 = min(img.width, int(bcx + side / 2))
        ry1 = min(img.height, int(bcy + side / 2))
        crop = img.crop((rx0, ry0, rx1, ry1))

        crop_faces = self._detect_faces(crop, permissive=True)
        if not crop_faces:
            return None

        # Map landmarks back to full-image coordinates.
        full_faces: list[_Face] = []
        for left, right, nose, oval in crop_faces:
            full_faces.append((
                (left[0] + rx0, left[1] + ry0),
                (right[0] + rx0, right[1] + ry0),
                (nose[0] + rx0, nose[1] + ry0),
                [(x + rx0, y + ry0) for x, y in oval],
            ))

        eyes = self._pick_face(full_faces, face_hint, img.width, img.height)
        if eyes is None:
            return None

        # BlazeFace/landmarker agreement gate: the LM eye-midpoint must fall
        # inside the BlazeFace box. Stage-2 LM at _STAGE2_LM_CONF=0.1 will
        # happily hallucinate a face mesh on ear / profile regions within an
        # otherwise-correct crop — those hallucinations land just outside the
        # BF box. Requiring eye-mid ∈ BF box rejects them. Probe data:
        # bad cases sit at eye-mid offset ≈0.5×bf_dim (just outside);
        # see akostibas/selfies-for-a-year#21 for the full analysis and an
        # alternative (re-run BlazeFace on the LM-claimed face area to verify
        # it still detects a face) we can fall back to if this is too strict.
        left_eye, right_eye = eyes
        em_x = (left_eye[0] + right_eye[0]) / 2
        em_y = (left_eye[1] + right_eye[1]) / 2
        bx0, by0 = best_box.origin_x, best_box.origin_y
        bx1 = bx0 + best_box.width
        by1 = by0 + best_box.height
        if not (bx0 <= em_x <= bx1 and by0 <= em_y <= by1):
            return None
        return eyes

    def align(
        self,
        img: Image.Image,
        *,
        face_hint: tuple[float, float, float] | None = None,
    ) -> Image.Image | None:
        """Align a face in the image to the canonical position.

        Returns:
            The aligned image, or None if no usable face was found and the
            source has a hint (Apple Photos), in which case the caller should
            drop the frame. For hint-less sources (selfie-dir), falls back
            to a center crop and returns it.

        Args:
            img: The image to align.
            face_hint: Optional (center_x, center_y, size) from the Photos DB,
                where center is fractions of image dimensions and size is face
                area as a fraction of image area. Used to (a) pick the correct
                face when multiple are detected, (b) license the stage-2
                BlazeFace seed, and (c) reject misdetections.
        """
        # Stage 1: full-image landmarker at production confidence.
        faces = self._detect_faces(img)
        eyes = self._pick_face(faces, face_hint, img.width, img.height)

        # Stage 2: BlazeFace + permissive landmarker on a crop. Only safe
        # when we have a hint to validate the BlazeFace seed against.
        recovered_via_stage2 = False
        if eyes is None and face_hint is not None:
            eyes = self._two_stage_eyes(img, face_hint)
            recovered_via_stage2 = eyes is not None

        if eyes is not None:
            self.aligned_count += 1
            if recovered_via_stage2:
                self.recovered_count += 1
            src_left, src_right = eyes
            transform = _compute_similarity_transform(
                src_left, src_right, self._dst_left, self._dst_right
            )
            return _apply_transform_with_blur_fill(
                img, transform, self.target_width, self.target_height
            )

        # No usable face found.
        if face_hint is None:
            # selfie-dir: vetted by the user, keep with a center-crop fallback.
            self.fallback_count += 1
            return self._center_crop(img)
        # Apple Photos: drop the frame to favor false negatives.
        self.skipped_count += 1
        return None

    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Center-crop fallback matching images.load_and_prepare logic."""
        src_ratio = img.width / img.height
        tgt_ratio = self.target_width / self.target_height

        if src_ratio > tgt_ratio:
            new_height = self.target_height
            new_width = round(img.width * (self.target_height / img.height))
        else:
            new_width = self.target_width
            new_height = round(img.height * (self.target_width / img.width))

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = (new_width - self.target_width) // 2
        top = (new_height - self.target_height) // 2
        return img.crop((left, top, left + self.target_width, top + self.target_height))
