"""Materialize Apple Photos originals from iCloud to a local cache.

For users with "Optimize Mac Storage" enabled, full-resolution originals
live in iCloud and the on-disk library only holds lower-res derivatives.
This module:

  1. Requests TCC authorization to access Photos.
  2. Screens local derivatives with the face aligner to identify which
     photos are worth materializing (avoids downloading the whole library).
  3. Triggers per-asset iCloud downloads via PhotoKit, writing originals
     to /tmp where macOS auto-reaps them.

PhotoKit authorization is attributed to the parent terminal (Terminal.app,
iTerm, etc.) via TCC. First run shows a system prompt; subsequent runs
are silent. SSH / cron sessions cannot trigger the prompt — we detect
and report that case clearly.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("/tmp/selfies-for-a-year/icloud-cache")


@dataclass
class DownloadResult:
    uuid: str
    path: Path | None
    error: str | None


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def authorize() -> tuple[bool, str | None]:
    """Ensure the parent terminal has Photos access via TCC.

    Returns (granted, message). On first run macOS may show a system
    dialog attributed to the parent terminal; we block until it resolves.
    """
    import Photos  # type: ignore[import-not-found]

    status = Photos.PHPhotoLibrary.authorizationStatus()
    # 0=notDetermined 1=restricted 2=denied 3=authorized 4=limited
    if status == 3 or status == 4:
        return True, None

    if status == 2 or status == 1:
        return False, (
            "Photos access is denied or restricted for this terminal. "
            "Open System Settings → Privacy & Security → Photos, enable "
            "access for your terminal app, then re-run."
        )

    # notDetermined: request — this shows the system prompt
    event = threading.Event()
    result: list[int] = [status]

    def handler(new_status: int) -> None:
        result[0] = new_status
        event.set()

    Photos.PHPhotoLibrary.requestAuthorization_(handler)
    # 60s should be plenty for the user to click. On SSH/cron the prompt
    # never shows and we fall through.
    if not event.wait(timeout=60):
        return False, (
            "Timed out waiting for Photos authorization. If you're on SSH "
            "or a non-GUI session the prompt can't appear — run from a "
            "normal terminal window."
        )

    if result[0] == 3 or result[0] == 4:
        return True, None
    return False, "Photos authorization was not granted."


def _fetch_asset(uuid: str):
    """Resolve a bare UUID to a PHAsset via its localIdentifier."""
    import Photos  # type: ignore[import-not-found]

    local_id = f"{uuid}/L0/001"
    result = Photos.PHAsset.fetchAssetsWithLocalIdentifiers_options_([local_id], None)
    if result.count() == 0:
        return None
    return result.firstObject()


def _photo_resource(asset):
    """Pick the photo (not adjustment / alternate) resource."""
    import Photos  # type: ignore[import-not-found]

    resources = Photos.PHAssetResource.assetResourcesForAsset_(asset)
    for i in range(resources.count()):
        r = resources.objectAtIndex_(i)
        if r.type() == Photos.PHAssetResourceTypePhoto:
            return r
    return None


def download_one(uuid: str, dest_dir: Path | None = None, timeout: float = 120) -> DownloadResult:
    """Download a single asset's original to dest_dir/<UUID><ext>.

    Already-cached files are returned without re-downloading.
    """
    import Photos  # type: ignore[import-not-found]
    from Foundation import NSURL  # type: ignore[import-not-found]

    dest_dir = dest_dir or _ensure_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    asset = _fetch_asset(uuid)
    if asset is None:
        return DownloadResult(uuid, None, "asset not found in Photos")

    resource = _photo_resource(asset)
    if resource is None:
        return DownloadResult(uuid, None, "no photo resource (deleted / cloud-shared?)")

    ext = Path(resource.originalFilename()).suffix or ".jpg"
    out_path = dest_dir / f"{uuid}{ext}"
    if out_path.exists() and out_path.stat().st_size > 0:
        return DownloadResult(uuid, out_path, None)

    options = Photos.PHAssetResourceRequestOptions.alloc().init()
    options.setNetworkAccessAllowed_(True)

    event = threading.Event()
    err_holder: list = [None]

    def completion(error):
        err_holder[0] = error
        event.set()

    Photos.PHAssetResourceManager.defaultManager().writeDataForAssetResource_toFile_options_completionHandler_(
        resource,
        NSURL.fileURLWithPath_(str(out_path)),
        options,
        completion,
    )

    if not event.wait(timeout=timeout):
        return DownloadResult(uuid, None, f"timed out after {timeout:.0f}s")

    err = err_holder[0]
    if err is not None:
        return DownloadResult(uuid, None, str(err.localizedDescription()))

    if not out_path.exists() or out_path.stat().st_size == 0:
        return DownloadResult(uuid, None, "completion handler succeeded but file is missing/empty")

    return DownloadResult(uuid, out_path, None)


def download_many(
    uuids: list[str],
    dest_dir: Path | None = None,
    *,
    max_workers: int = 4,
    progress_cb=None,
) -> dict[str, DownloadResult]:
    """Download many assets in parallel. progress_cb(done, total, last_result)."""
    dest_dir = dest_dir or _ensure_cache_dir()
    out: dict[str, DownloadResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_one, u, dest_dir): u for u in uuids}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            out[res.uuid] = res
            if progress_cb is not None:
                progress_cb(i, len(uuids), res)
    return out


def screen_derivatives(
    photos: list,
    *,
    library: Path,
    width: int,
    height: int,
    min_face_fraction: float,
    progress_cb=None,
) -> list[str]:
    """Run the face aligner against each photo's local derivative.

    Returns UUIDs whose derivative passes screening (face detected,
    aligned, and survives min_face_fraction). These are the assets worth
    downloading originals for.
    """
    from PIL import Image  # local import to keep module load cheap

    from selfies_for_a_year.align import FaceAligner
    from selfies_for_a_year.photos import derivative_path

    kept: list[str] = []
    with FaceAligner(width, height) as aligner:
        for i, photo in enumerate(photos, 1):
            deriv = derivative_path(library, photo.uuid)
            if deriv is None or not deriv.exists():
                if progress_cb is not None:
                    progress_cb(i, len(photos), photo.uuid, False)
                continue
            if min_face_fraction > 0 and photo.face_size < min_face_fraction:
                if progress_cb is not None:
                    progress_cb(i, len(photos), photo.uuid, False)
                continue
            try:
                img = Image.open(deriv).convert("RGB")
            except Exception:
                if progress_cb is not None:
                    progress_cb(i, len(photos), photo.uuid, False)
                continue
            hint = (photo.face_center_x, photo.face_center_y, photo.face_size)
            out = aligner.align(img, face_hint=hint)
            ok = out is not None
            if ok:
                kept.append(photo.uuid)
            if progress_cb is not None:
                progress_cb(i, len(photos), photo.uuid, ok)
    return kept
