"""Read-only access to the Apple Photos library database.

Queries the Photos.sqlite database to find photos of a specific person
using Apple's face recognition data. All database access is strictly
read-only.

macOS only — requires Full Disk Access or appropriate permissions to
read ~/Pictures/Photos Library.photoslibrary/.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Apple Core Data epoch: 2001-01-01 00:00:00 UTC
_CORE_DATA_EPOCH_OFFSET = 978307200

# Default Photos library location
_DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"

# Image UTIs we support (exclude videos)
_IMAGE_UTIS = ("public.jpeg", "public.heic", "public.png")


@dataclass
class PhotosPersonInfo:
    """A named person from the Apple Photos library."""

    person_id: int
    display_name: str
    full_name: str | None
    face_count: int


@dataclass
class PhotosImage:
    """A photo from the Apple Photos library with face metadata."""

    path: Path
    date: datetime
    face_quality: float
    face_size: float  # fraction of image area
    face_center_x: float  # face center x as fraction of image width
    face_center_y: float  # face center y as fraction of image height
    # Apple IQA signal: higher value = Apple thinks the photo's blur is
    # unintentional / a mistake. Validated in #19 as a useful blur gate
    # signal; None when Apple hasn't computed attributes for this asset.
    tastefully_blurred: float | None


def _connect(library: Path) -> sqlite3.Connection:
    """Open the Photos database in read-only mode."""
    db_path = library / "database" / "Photos.sqlite"
    if not db_path.exists():
        raise SystemExit(
            f"Photos database not found at {db_path}.\n"
            "Make sure Apple Photos is set up and the library exists."
        )
    # URI mode for read-only access
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _resolve_path(
    library: Path, directory: str, filename: str, *, has_adjustments: bool = False,
) -> Path:
    """Resolve a ZASSET directory + filename to an actual file path.

    When has_adjustments is True, checks for a rendered (edited) version first.
    The rendered version reflects user edits in Photos (rotation, crop, etc.).
    """
    if has_adjustments and len(directory) <= 1:
        # Rendered files follow pattern: renders/{dir}/{stem}_1_201_a.{ext}
        stem = Path(filename).stem
        ext = Path(filename).suffix
        rendered = library / "resources" / "renders" / directory / f"{stem}_1_201_a{ext}"
        if rendered.exists():
            return rendered

    # Single hex character = local photo under originals/
    if len(directory) <= 1:
        return library / "originals" / directory / filename
    # Full or relative path (cloud-shared photos)
    candidate = library / "scopes" / "cloudsharing" / "data" / directory / filename
    if candidate.exists():
        return candidate
    # Fall back to originals with directory as subfolder
    return library / "originals" / directory / filename


def _core_data_ts(dt: date) -> float:
    """Convert a Python date to a Core Data timestamp."""
    unix = datetime(dt.year, dt.month, dt.day).timestamp()
    return unix - _CORE_DATA_EPOCH_OFFSET


def list_people(
    library: Path = _DEFAULT_LIBRARY,
    min_faces: int = 1,
) -> list[PhotosPersonInfo]:
    """List named people in the Photos library, sorted by face count."""
    conn = _connect(library)
    try:
        rows = conn.execute(
            """
            SELECT Z_PK, ZDISPLAYNAME, ZFULLNAME, ZFACECOUNT
            FROM ZPERSON
            WHERE ZDISPLAYNAME IS NOT NULL AND ZDISPLAYNAME != ''
              AND ZFACECOUNT >= ?
            ORDER BY ZFACECOUNT DESC
            """,
            (min_faces,),
        ).fetchall()
    finally:
        conn.close()

    return [
        PhotosPersonInfo(
            person_id=row[0],
            display_name=row[1],
            full_name=row[2],
            face_count=row[3],
        )
        for row in rows
    ]


def find_person(
    name: str,
    library: Path = _DEFAULT_LIBRARY,
) -> PhotosPersonInfo | None:
    """Find a person by name (case-insensitive substring match).

    Returns the best match (shortest name that contains the query),
    or None if no match found.
    """
    conn = _connect(library)
    try:
        rows = conn.execute(
            """
            SELECT Z_PK, ZDISPLAYNAME, ZFULLNAME, ZFACECOUNT
            FROM ZPERSON
            WHERE ZDISPLAYNAME IS NOT NULL
              AND (LOWER(ZDISPLAYNAME) LIKE '%' || LOWER(?) || '%'
                   OR LOWER(ZFULLNAME) LIKE '%' || LOWER(?) || '%')
              AND ZFACECOUNT > 0
            ORDER BY
                CASE WHEN LOWER(ZDISPLAYNAME) = LOWER(?) THEN 0 ELSE 1 END,
                LENGTH(ZDISPLAYNAME)
            LIMIT 1
            """,
            (name, name, name),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    row = rows[0]
    return PhotosPersonInfo(
        person_id=row[0],
        display_name=row[1],
        full_name=row[2],
        face_count=row[3],
    )


def query_photos(
    person_id: int,
    *,
    year_start: date | None = None,
    year_end: date | None = None,
    library: Path = _DEFAULT_LIBRARY,
) -> list[PhotosImage]:
    """Find all photos of a person, optionally filtered by date range.

    Returns photos sorted chronologically.
    """
    conn = _connect(library)
    try:
        params: list[object] = [person_id]

        # Build placeholders for UTIs (these come next in the SQL)
        uti_placeholders = ",".join("?" for _ in _IMAGE_UTIS)
        params.extend(_IMAGE_UTIS)

        date_clause = ""
        if year_start is not None:
            date_clause += " AND a.ZDATECREATED >= ?"
            params.append(_core_data_ts(year_start))

        if year_end is not None:
            date_clause += " AND a.ZDATECREATED < ?"
            params.append(_core_data_ts(year_end))

        rows = conn.execute(
            f"""
            SELECT a.ZDIRECTORY, a.ZFILENAME,
                   a.ZDATECREATED,
                   df.ZQUALITY, df.ZSIZE,
                   a.ZADJUSTMENTSSTATE,
                   df.ZCENTERX, df.ZCENTERY,
                   caa.ZTASTEFULLYBLURREDSCORE
            FROM ZDETECTEDFACE df
            JOIN ZASSET a ON df.ZASSETFORFACE = a.Z_PK
            LEFT JOIN ZCOMPUTEDASSETATTRIBUTES caa ON caa.ZASSET = a.Z_PK
            WHERE df.ZPERSONFORFACE = ?
              AND a.ZTRASHEDSTATE = 0
              AND a.ZUNIFORMTYPEIDENTIFIER IN ({uti_placeholders})
              {date_clause}
            ORDER BY a.ZDATECREATED
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    results = []
    for (directory, filename, created_ts, quality, size,
         adj_state, cx, cy, tastefully_blurred) in rows:
        has_adjustments = adj_state == 2
        path = _resolve_path(library, directory, filename, has_adjustments=has_adjustments)
        if not path.exists():
            continue
        dt = datetime.fromtimestamp(created_ts + _CORE_DATA_EPOCH_OFFSET)
        results.append(PhotosImage(
            path=path,
            date=dt,
            face_quality=quality if quality is not None else 0.0,
            face_size=size if size is not None else 0.0,
            face_center_x=cx if cx is not None else 0.5,
            # Apple stores ZCENTERY in CG/Quartz convention (origin at
            # bottom-left, y points up). Flip to top-down image coords.
            face_center_y=(1.0 - cy) if cy is not None else 0.5,
            tastefully_blurred=tastefully_blurred,
        ))

    return results


def pick_best_photo(photos: list[PhotosImage]) -> PhotosImage:
    """Pick the best photo from a group (e.g., same day).

    Stub implementation: picks the highest face quality score.
    See issue #11 for planned improvements.
    """
    return max(photos, key=lambda p: p.face_quality)


def deduplicate_by_day(
    photos: list[PhotosImage],
) -> list[PhotosImage]:
    """Keep only the best photo per calendar day."""
    by_day: dict[date, list[PhotosImage]] = {}
    for photo in photos:
        day = photo.date.date()
        by_day.setdefault(day, []).append(photo)

    result = []
    for day in sorted(by_day):
        candidates = by_day[day]
        result.append(
            candidates[0] if len(candidates) == 1
            else pick_best_photo(candidates)
        )
    return result
