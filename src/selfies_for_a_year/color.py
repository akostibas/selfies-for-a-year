"""Color and exposure normalization across frames.

Smooths frame-to-frame brightness and color jitter by matching each frame's
histogram to a rolling average of its neighbors in LAB color space.
Operating in LAB lets us adjust luminance (L) independently of color (A, B),
preserving natural hues while smoothing exposure differences.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _rgb_to_lab(img: Image.Image) -> np.ndarray:
    """Convert an RGB PIL Image to LAB via numpy.

    Uses the sRGB -> XYZ -> LAB conversion. Returns float64 array with
    L in [0, 100], A and B in roughly [-128, 127].
    """
    arr = np.asarray(img, dtype=np.float64) / 255.0

    # Linearize sRGB
    mask = arr > 0.04045
    arr = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)

    # sRGB -> XYZ (D65 reference white)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = arr @ mat.T

    # Normalize by D65 white point
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 2] /= 1.08883

    # XYZ -> LAB
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    xyz_f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    lab = np.empty_like(xyz)
    lab[:, :, 0] = 116.0 * xyz_f[:, :, 1] - 16.0  # L
    lab[:, :, 1] = 500.0 * (xyz_f[:, :, 0] - xyz_f[:, :, 1])  # A
    lab[:, :, 2] = 200.0 * (xyz_f[:, :, 1] - xyz_f[:, :, 2])  # B

    return lab


def _lab_to_rgb(lab: np.ndarray) -> Image.Image:
    """Convert a LAB numpy array back to an RGB PIL Image."""
    # LAB -> XYZ
    fy = (lab[:, :, 0] + 16.0) / 116.0
    fx = lab[:, :, 1] / 500.0 + fy
    fz = fy - lab[:, :, 2] / 200.0

    epsilon = 0.008856
    kappa = 903.3

    x_mask = fx ** 3 > epsilon
    y_mask = lab[:, :, 0] > kappa * epsilon
    z_mask = fz ** 3 > epsilon

    xyz = np.empty_like(lab)
    xyz[:, :, 0] = np.where(x_mask, fx ** 3, (116.0 * fx - 16.0) / kappa)
    xyz[:, :, 1] = np.where(y_mask, fy ** 3, lab[:, :, 0] / kappa)
    xyz[:, :, 2] = np.where(z_mask, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # Denormalize by D65 white point
    xyz[:, :, 0] *= 0.95047
    xyz[:, :, 2] *= 1.08883

    # XYZ -> linear sRGB
    mat = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    rgb = xyz @ mat.T

    # Gamma encode
    rgb = np.clip(rgb, 0, None)
    mask = rgb > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055, 12.92 * rgb)
    rgb = np.clip(rgb, 0, 1)

    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _channel_stats(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std."""
    mean = np.array([lab[:, :, c].mean() for c in range(3)])
    std = np.array([lab[:, :, c].std() for c in range(3)])
    return mean, std


def normalize_colors(
    images: list[Image.Image],
    window: int = 11,
) -> list[Image.Image]:
    """Normalize brightness across a sequence of images.

    Shifts each frame's L channel (brightness) toward a rolling average of
    its neighbors, then applies a compensating contrast adjustment so that
    darkened frames get a contrast boost and brightened frames get a slight
    reduction. Color channels (A, B) are left untouched.

    Args:
        images: List of RGB PIL Images (all same size).
        window: Rolling window size. Larger = more smoothing.

    Returns:
        List of normalized RGB PIL Images.
    """
    if len(images) <= 1:
        return images

    # Pre-compute LAB arrays and L-channel stats
    labs = [_rgb_to_lab(img) for img in images]
    l_means = np.array([lab[:, :, 0].mean() for lab in labs])
    l_stds = np.array([lab[:, :, 0].std() for lab in labs])

    # Compute rolling average of L means
    half = window // 2
    n = len(images)
    target_l_means = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        target_l_means[i] = l_means[lo:hi].mean()

    # Apply brightness shift with compensating contrast adjustment
    result = []
    for i, lab in enumerate(labs):
        normalized = lab.copy()

        shift = target_l_means[i] - l_means[i]
        # Compensating contrast: if we brighten (shift > 0), slightly reduce
        # contrast, and vice versa. Scale is roughly proportional to the
        # brightness change relative to the original mean.
        if l_means[i] > 1e-6 and l_stds[i] > 1e-6:
            # Ratio of new mean to old mean — used to inversely scale contrast
            brightness_ratio = target_l_means[i] / l_means[i]
            contrast_factor = 1.0 / brightness_ratio  # inverse relationship

            # Apply: shift mean, then scale contrast around new mean
            new_mean = l_means[i] + shift
            normalized[:, :, 0] = (
                (normalized[:, :, 0] - l_means[i]) * contrast_factor + new_mean
            )

        result.append(_lab_to_rgb(normalized))

    return result
