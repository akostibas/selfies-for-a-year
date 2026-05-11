"""Image discovery, date extraction, and preparation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ExifTags, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support unavailable

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".heic"}

# Regex patterns for date extraction from filenames (tried in order)
_FILENAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ISO 8601: 2026-01-02T07:19:37-08:00
    (re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})"), "%Y-%m-%d %H:%M:%S"),
    # macOS screenshot style: 2024-01-02 at 10.28.21
    (re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{2}\.\d{2}\.\d{2})"), "%Y-%m-%d %H.%M.%S"),
    # Underscore style: 2023-01-05_1 (just the date part)
    (re.compile(r"(\d{4}-\d{2}-\d{2})"), "%Y-%m-%d"),
]


def discover_images(directory: Path) -> list[Path]:
    """Find all image files in a directory (non-recursive)."""
    return [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _exif_date(path: Path) -> datetime | None:
    """Extract DateTimeOriginal from EXIF data."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            # DateTimeOriginal tag
            for tag_id, tag_name in ExifTags.Base.__members__.items():
                if tag_name == "DateTimeOriginal":
                    val = exif.get(ExifTags.Base[tag_name])
                    if val:
                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                    break
            # Fall back to DateTime
            dt_val = exif.get(ExifTags.Base.DateTime)
            if dt_val:
                return datetime.strptime(dt_val, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def _filename_date(path: Path) -> datetime | None:
    """Parse a date from the filename using known patterns."""
    stem = path.stem
    for pattern, fmt in _FILENAME_PATTERNS:
        m = pattern.search(stem)
        if m:
            date_str = " ".join(m.groups())
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    return None


def _mtime_date(path: Path) -> datetime:
    """Fall back to file modification time."""
    return datetime.fromtimestamp(path.stat().st_mtime)


def sort_key(path: Path) -> datetime:
    """Get the best available date for sorting."""
    return _exif_date(path) or _filename_date(path) or _mtime_date(path)


def date_label(path: Path) -> str:
    """Get a 'YYYY - Month' label for a photo."""
    dt = sort_key(path)
    return dt.strftime("%Y - %B")


def sort_images(paths: list[Path]) -> list[Path]:
    """Sort images chronologically using the best available date."""
    return sorted(paths, key=sort_key)


def load_image(path: Path) -> Image.Image:
    """Load an image, normalize to upright via EXIF, and convert to RGB.

    `ImageOps.exif_transpose` handles all file types correctly:
    - HEIC: pillow-heif has already rotated pixels upright on decode and
      reset EXIF orientation to 1, so this is a no-op.
    - JPEG: reads raw-sensor EXIF orientation (1-8) and rotates to upright.

    Apple's DB `ZORIENTATION` is NOT a user-facing rotation override — it
    records the original file's raw EXIF orientation, which `exif_transpose`
    already consumes. Threading it through the pipeline caused a
    double-rotation regression (see git log for 9759cd5 -> this commit).
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def load_and_prepare(
    path: Path,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """Load an image, fix orientation, and center-crop to target dimensions."""
    img = load_image(path)

    # Scale so the image covers the target area, then center-crop
    src_ratio = img.width / img.height
    tgt_ratio = target_width / target_height

    if src_ratio > tgt_ratio:
        # Image is wider than target — scale by height, crop width
        new_height = target_height
        new_width = round(img.width * (target_height / img.height))
    else:
        # Image is taller than target — scale by width, crop height
        new_width = target_width
        new_height = round(img.height * (target_width / img.width))

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))

    return img
