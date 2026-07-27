"""CLI entry point for selfies-for-a-year."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import click
import typer
from PIL import Image, ImageDraw, ImageFont
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from selfies_for_a_year.images import (
    discover_images,
    load_and_prepare,
    load_image,
    sort_key,
)
from selfies_for_a_year.video import (
    check_ffmpeg,
    compile_video,
    compile_video_variable,
    get_duration,
    mux_audio,
)

app = typer.Typer(help="Turn daily selfies into a timelapse video.")

FPS = 24


def _parse_duration(value: str) -> tuple[float, bool]:
    """Parse a duration string into (number, is_percent).

    Accepts: "150" or "150ms" (milliseconds), "40%" (percentage).
    """
    value = value.strip()
    if value.endswith("%"):
        try:
            return (float(value[:-1]), True)
        except ValueError:
            pass
    if value.endswith("ms"):
        try:
            return (float(value[:-2]), False)
        except ValueError:
            pass
    try:
        return (float(value), False)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid duration: {value!r}. Use a number (ms), '150ms', or '40%'."
        )


def _resolve_frames(value: str, reference_frames: int, label: str) -> int:
    """Parse a duration string and resolve to a frame count.

    For absolute (ms): converts ms to frames at FPS.
    For percentage: applies percentage to reference_frames.
    """
    num, is_pct = _parse_duration(value)
    if is_pct:
        if not 0 <= num <= 100:
            raise typer.BadParameter(f"--{label} percentage must be 0-100, got {num}%.")
        return round(reference_frames * num / 100)
    return max(0, round(num / 1000 * FPS))


_COLOR_NAMES: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def _parse_color(value: str) -> tuple[int, int, int]:
    """Parse a color name or hex string like '#ff0000' to an RGB tuple."""
    if value.lower() in _COLOR_NAMES:
        return _COLOR_NAMES[value.lower()]
    v = value.lstrip("#")
    if len(v) == 6:
        try:
            return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
        except ValueError:
            pass
    raise typer.BadParameter(f"Invalid color: {value!r}. Use 'black', 'white', or hex like '#ff0000'.")


def _parse_year(value: str) -> tuple[date, date]:
    """Parse a year or year range like '2026' or '2024-2026'.

    Returns (start_date, end_date) where end_date is exclusive (Jan 1 of the next year).
    """
    value = value.strip()
    if "-" in value and not value.startswith("-"):
        parts = value.split("-", 1)
        try:
            start_year = int(parts[0])
            end_year = int(parts[1])
        except ValueError:
            raise typer.BadParameter(f"Invalid year range: {value!r}. Use '2026' or '2024-2026'.")
        if end_year < start_year:
            raise typer.BadParameter(f"End year must be >= start year: {value!r}.")
        return date(start_year, 1, 1), date(end_year + 1, 1, 1)
    try:
        year = int(value)
    except ValueError:
        raise typer.BadParameter(f"Invalid year: {value!r}. Use '2026' or '2024-2026'.")
    return date(year, 1, 1), date(year + 1, 1, 1)


def _overlay_label(img: Image.Image, label: str) -> Image.Image:
    """Draw a date label at the bottom of the frame.

    "YYYY - Month" splits at the dash: year right-justified just left of
    center, month left-justified just right of center. This stabilizes
    the year's position (its right edge never moves) and the month's
    position (its left edge never moves), so the eye can latch onto each
    half independently as the timelapse races through dates. A single
    centered label whips around because year+month length varies (April
    → September is a 5-character swing).
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    font_size = img.width // 20
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)

    y = img.height - img.height // 20
    center = img.width // 2
    gap = font_size // 3  # half-gap between year right-edge and month left-edge

    parts = label.split(" - ", 1)
    if len(parts) == 2:
        year, month = parts
        year_x = center - gap
        month_x = center + gap
        # Year: anchor right-baseline. Month: anchor left-baseline.
        draw.text((year_x + 2, y + 2), year, fill=(0, 0, 0, 180), font=font, anchor="rs")
        draw.text((year_x, y), year, fill=(255, 255, 255), font=font, anchor="rs")
        draw.text((month_x + 2, y + 2), month, fill=(0, 0, 0, 180), font=font, anchor="ls")
        draw.text((month_x, y), month, fill=(255, 255, 255), font=font, anchor="ls")
    else:
        draw.text((center + 2, y + 2), label, fill=(0, 0, 0, 180), font=font, anchor="ms")
        draw.text((center, y), label, fill=(255, 255, 255), font=font, anchor="ms")

    return img


_TIER_COLORS = {
    # Energy ramp green→yellow→red for the active tiers; gray for ambient
    # (neutral, "nothing notable happening").
    "slow": (90, 200, 110),       # green
    "normal": (240, 210, 70),     # yellow
    "intense": (255, 95, 95),     # red
    "ambient": (150, 150, 150),   # gray
}


_DEBUG_FONT_PATHS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Courier.ttc",
)


def _debug_font(img: Image.Image) -> tuple[ImageFont.ImageFont, int]:
    """The monospace font + size used by the bottom-left debug block."""
    font_size = img.width // 44
    for path in _DEBUG_FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size), font_size
        except OSError:
            continue
    return ImageFont.load_default(size=font_size), font_size


def _debug_block_top(img: Image.Image, n_rows: int) -> tuple[int, int, int, int, int]:
    """Geometry of the debug block: (x0, y0, pad, line_h, font_size).

    Shared by _overlay_debug (which draws the block) and _overlay_metronome
    (which pins its dot onto the block's first row), so the two never drift.
    """
    _, font_size = _debug_font(img)
    pad = max(5, img.width // 110)
    line_h = font_size + pad // 2
    block_h = line_h * n_rows + pad
    date_top = img.height - img.height // 20 - img.width // 20
    x0 = pad
    y0 = date_top - block_h - pad
    return x0, y0, pad, line_h, font_size


def _debug_song_label(song: str, bpm: float | None) -> str:
    """The song/BPM row text — the metronome dot is pinned to the end of this."""
    label = song if len(song) <= 30 else song[:29] + "…"
    if bpm is not None and bpm > 0:
        label = f"{label}  {bpm:.1f} BPM"
    return label


def _overlay_debug(
    img: Image.Image,
    *,
    tier: str | None = None,
    duration: float | None = None,
    filename: str | None = None,
    song: str | None = None,
    bpm: float | None = None,
) -> Image.Image:
    """Stack debug lines bottom-left over a black background, monospace.

    Tier line uses a color per tier; filename stays white. The song/BPM row
    reserves trailing space for the metronome dot (drawn per-frame by
    _overlay_metronome), so this block and that dot form one HUD.
    """
    if tier is None and filename is None and song is None:
        return img
    img = img.copy()
    draw = ImageDraw.Draw(img)

    font, font_size = _debug_font(img)

    rows: list[tuple[str, tuple[int, int, int]]] = []
    if song is not None:
        # Trailing spaces reserve room for the metronome dot at the row's end.
        rows.append((_debug_song_label(song, bpm) + "  ", (200, 200, 255)))
    if tier is not None and duration is not None:
        rate = 1.0 / duration if duration > 0 else 0.0
        text = f"{tier.upper():<8}{rate:>5.2f}/s  {duration * 1000:>4.0f}ms"
        rows.append((text, _TIER_COLORS.get(tier, (255, 255, 255))))
    if filename is not None:
        if len(filename) > 40:
            filename = filename[:30] + "…" + filename[-8:]
        rows.append((filename, (255, 255, 255)))

    x0, y0, pad, line_h, _ = _debug_block_top(img, len(rows))
    block_h = line_h * len(rows) + pad
    widest = max(int(draw.textlength(t, font=font)) for t, _ in rows)
    block_w = widest + pad * 2
    draw.rectangle((x0, y0, x0 + block_w, y0 + block_h), fill=(0, 0, 0, 255))
    text_x = x0 + pad
    text_y = y0 + pad // 2
    for text, color in rows:
        draw.text((text_x, text_y), text, fill=color, font=font, anchor="lt")
        text_y += line_h
    return img


def _overlay_progression_bar(
    img: Image.Image,
    states: list[tuple[float, float, str]],
    t: float,
    total_duration: float,
) -> Image.Image:
    """Draw a horizontal track-progression bar with a playhead across the top.

    ``states`` is a list of (start, end, tier) runs — the song's "sheet music."
    Each run is a colored segment (reusing ``_TIER_COLORS``); a white playhead
    marks the current time ``t``. Gives an at-a-glance read of where we are in
    the track and what pacing state we're in, alongside the textual tier
    overlay. Drawn per-frame, so kept cheap (a handful of rectangles + a line).
    """
    if total_duration <= 0 or not states:
        return img
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    margin = img.width // 20
    x0 = margin
    x1 = img.width - margin
    span = x1 - x0
    bar_h = max(6, img.height // 90)
    y0 = max(6, img.height // 40)
    y1 = y0 + bar_h

    def x_at(sec: float) -> float:
        return x0 + span * max(0.0, min(1.0, sec / total_duration))

    # Track background so tiers read against any photo.
    draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(0, 0, 0, 140))
    for start, end, tier in states:
        sx = x_at(start)
        ex = x_at(end)
        color = _TIER_COLORS.get(tier, (200, 200, 200))
        draw.rectangle((sx, y0, ex, y1), fill=(*color, 255))

    # Playhead: a white line spilling a little above and below the bar.
    px = x_at(t)
    over = bar_h
    draw.line((px, y0 - over, px, y1 + over), fill=(255, 255, 255, 255), width=max(2, img.width // 640))

    # Current tier label just below the playhead, clamped into frame.
    cur = next((tr for a, b, tr in states if a - 1e-6 <= t <= b + 1e-6 for tr in [tr]), None)
    if cur is not None:
        font_size = img.width // 55
        font = None
        for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default(size=font_size)
        text = cur.upper()
        tw = draw.textlength(text, font=font)
        tx = min(max(px - tw / 2, x0), x1 - tw)
        ty = y1 + over + 2
        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0, 200), font=font, anchor="lt")
        draw.text((tx, ty), text, fill=_TIER_COLORS.get(cur, (255, 255, 255)), font=font, anchor="lt")

    return img


def _metronome_dot_anchor(
    img: Image.Image,
    *,
    song: str | None,
    bpm: float | None,
    debug_tier_overlay: bool,
    debug_filename_overlay: bool,
) -> tuple[tuple[float, float], ImageFont.ImageFont, str]:
    """Fixed screen position for the metronome dot, computed once per render.

    When the debug block has a song/BPM row, the dot rides at the end of that
    row (one HUD, one place to look). Otherwise it falls back to the
    bottom-right corner so --debug-metronome still works on its own.
    Returns ((x, y), font, anchor).
    """
    song_row = debug_tier_overlay and song is not None
    if song_row:
        font, font_size = _debug_font(img)
        n_rows = 2 + (1 if debug_filename_overlay else 0)  # song + tier (+ filename)
        x0, y0, pad, line_h, _ = _debug_block_top(img, n_rows)
        label = _debug_song_label(song, bpm)
        label_w = ImageDraw.Draw(img).textlength(label, font=font)
        x = x0 + pad + label_w + font_size * 0.5
        y = y0 + pad // 2 + font_size / 2  # vertical center of the first row
        return (x, y), font, "lm"
    # Fallback: standalone bottom-right.
    font, _ = _debug_font(img)
    return (img.width - img.width // 40, img.height - img.height // 20), font, "rs"


def _overlay_metronome(
    img: Image.Image,
    beat_times: list[float],
    t: float,
    *,
    bpm: float | None,
    anchor_xy: tuple[float, float],
    font: ImageFont.ImageFont,
    anchor: str,
) -> Image.Image:
    """Draw a metronome dot that flashes on each beat at a precomputed anchor.

    Filled ● in a short flash window right after the most recent beat, hollow ○
    the rest of the time — an at-a-glance tempo pulse. Because it blinks on the
    detected beats (not the cut grid), you can eyeball whether hard cuts land
    on the beat: in intense sections you'll see several cuts per flash, in
    slow/ambient sections one flash spans several held photos.
    """
    if not beat_times:
        return img
    # Most recent beat at or before t (beat_times is sorted ascending).
    import bisect

    i = bisect.bisect_right(beat_times, t) - 1
    since = (t - beat_times[i]) if i >= 0 else 1e9
    beat_period = (60.0 / bpm) if bpm and bpm > 0 else 0.4
    on = since < min(0.12, beat_period * 0.4)

    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    glyph = "●" if on else "○"
    color = (255, 95, 95, 255) if on else (150, 150, 150, 230)
    x, y = anchor_xy
    draw.text((x, y), glyph, fill=color, font=font, anchor=anchor)
    return img


def _crossfade(
    frames: Iterator[tuple[Image.Image, str]],
    hold_frames: int,
    fade_frames: int,
) -> Iterator[Image.Image]:
    """Add hold frames and crossfade transitions between labeled frames."""
    item = next(frames, None)
    if item is None:
        return
    prev_img, prev_label = item
    prev_labeled = _overlay_label(prev_img, prev_label)
    for _ in range(hold_frames):
        yield prev_labeled

    for current_img, current_label in frames:
        current_labeled = _overlay_label(current_img, current_label)
        for i in range(1, fade_frames + 1):
            alpha = i / (fade_frames + 1)
            yield Image.blend(prev_labeled, current_labeled, alpha)
        for _ in range(hold_frames):
            yield current_labeled
        prev_labeled = current_labeled


def _no_crossfade(
    frames: Iterator[tuple[Image.Image, str]],
) -> Iterator[Image.Image]:
    """Yield labeled frames without crossfade."""
    for img, label in frames:
        yield _overlay_label(img, label)


def _beat_frames(
    prepared: list[Image.Image],
    labels: list[str],
    paths: list[Path],
    segments: list,
    *,
    fps: int,
    crossfade: bool,
    debug_tier_overlay: bool,
    debug_filename_overlay: bool,
    song: str | None = None,
    bpm: float | None = None,
    progression_states: list[tuple[float, float, str]] | None = None,
    metronome_beats: list[float] | None = None,
    max_seconds: float | None = None,
) -> tuple[Iterator[Image.Image], int]:
    """Render beat-sync segments as constant-fps frames.

    With ``crossfade=True`` each photo peaks at its segment's start (the beat)
    and morphs into the next over the segment's duration. With
    ``crossfade=False`` segments are hard cuts — the current photo is held
    until the next beat, then swapped instantly. Both modes run at constant fps
    so the per-frame overlays (progression playhead, metronome) can animate;
    the hard-cut mode is what lets those debug HUDs work without a crossfade.
    """
    decorated: list[Image.Image] = []
    for seg in segments:
        img = prepared[seg.photo_index]
        framed = _overlay_label(img, labels[seg.photo_index])
        if debug_tier_overlay or debug_filename_overlay:
            framed = _overlay_debug(
                framed,
                tier=seg.tier if debug_tier_overlay else None,
                duration=seg.duration if debug_tier_overlay else None,
                filename=paths[seg.photo_index].name if debug_filename_overlay else None,
                song=song if debug_tier_overlay else None,
                bpm=bpm if debug_tier_overlay else None,
            )
        decorated.append(framed)

    total_duration = sum(seg.duration for seg in segments)
    n_frames = max(1, round(total_duration * fps))
    if max_seconds is not None:
        n_frames = min(n_frames, max(1, round(max_seconds * fps)))

    # The metronome dot rides at a fixed screen position (end of the debug
    # block's BPM row), so compute its anchor once rather than per frame.
    metro_anchor = metro_font = metro_align = None
    if metronome_beats:
        metro_anchor, metro_font, metro_align = _metronome_dot_anchor(
            decorated[0],
            song=song,
            bpm=bpm,
            debug_tier_overlay=debug_tier_overlay,
            debug_filename_overlay=debug_filename_overlay,
        )

    def _apply_frame_overlays(frame: Image.Image, t: float) -> Image.Image:
        if progression_states:
            frame = _overlay_progression_bar(frame, progression_states, t, total_duration)
        if metronome_beats:
            frame = _overlay_metronome(
                frame, metronome_beats, t,
                bpm=bpm, anchor_xy=metro_anchor, font=metro_font, anchor=metro_align,
            )
        return frame

    def gen() -> Iterator[Image.Image]:
        seg_i = 0
        seg_start = 0.0
        seg_end = segments[0].duration
        last_i = len(segments) - 1
        for f in range(n_frames):
            t = f / fps
            while seg_i < last_i and t >= seg_end:
                seg_start = seg_end
                seg_i += 1
                seg_end = seg_start + segments[seg_i].duration
            if not crossfade or seg_i >= last_i:
                frame = decorated[seg_i]
            else:
                alpha = (t - seg_start) / segments[seg_i].duration
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
                frame = Image.blend(decorated[seg_i], decorated[seg_i + 1], alpha)
            yield _apply_frame_overlays(frame, t)

    return gen(), n_frames


def _apply_bookend_fades(
    frames: Iterator[Image.Image],
    width: int,
    height: int,
    fade_in_frames: int,
    fade_out_frames: int,
    fade_in_color: tuple[int, int, int],
    fade_out_color: tuple[int, int, int],
    total: int,
) -> Iterator[Image.Image]:
    """Wrap a frame iterator with fade-in from and fade-out to a solid color."""
    solid_in = Image.new("RGB", (width, height), fade_in_color)
    solid_out = Image.new("RGB", (width, height), fade_out_color)

    fade_out_start = total - fade_out_frames
    for i, frame in enumerate(frames):
        if i < fade_in_frames:
            alpha = (i + 1) / (fade_in_frames + 1)
            yield Image.blend(solid_in, frame, alpha)
        elif i >= fade_out_start:
            remaining = total - i
            alpha = remaining / (fade_out_frames + 1)
            yield Image.blend(solid_out, frame, alpha)
        else:
            yield frame


def _label_from_date(dt: datetime) -> str:
    """Format a datetime as a 'YYYY - Month' label."""
    return dt.strftime("%Y - %B")


# (path, date, face_hint, apple_quality)
# face_hint is (center_x, center_y, size) as fractions, or None for selfie-dir photos.
# size is face area as a fraction of image area (from Apple Photos ZDETECTEDFACE.ZSIZE).
# apple_quality is (face_quality, tastefully_blurred) from the Apple Photos DB,
# or None for selfie-dir photos. Used by pre-alignment drop gates (see #19).
_FaceHint = tuple[float, float, float] | None
_AppleQuality = tuple[float, float | None] | None
_SourceItem = tuple[Path, datetime, _FaceHint, _AppleQuality]

# Manifest entry: (date, path, status, reason)
ManifestEntry = tuple[datetime, Path, str, str]


def _maybe_materialize_from_icloud(
    photos: list,
    *,
    width: int,
    height: int,
    min_face_fraction: float,
    threshold: float,
    force: bool,
    manifest: list,
) -> list:
    """If too few originals are on disk, offer to screen derivatives and
    download just the keepers from iCloud.

    Returns the (possibly path-rewritten) photo list. Any photo that
    remains unmaterialized after this call gets dropped with a manifest
    entry — the rest of the pipeline assumes paths point at real files.
    """
    on_disk = sum(1 for p in photos if p.is_materialized)
    total = len(photos)
    if total == 0:
        return photos
    rate = on_disk / total
    if rate >= threshold:
        # Drop any stragglers that aren't on disk and tell the user.
        kept = []
        for p in photos:
            if p.is_materialized:
                kept.append(p)
            else:
                manifest.append((
                    p.date, p.path, "dropped",
                    "original not on disk (iCloud); above materialize threshold so skipped silently",
                ))
        if len(kept) != total:
            typer.echo(
                f"Apple Photos: {on_disk}/{total} originals on disk "
                f"({100*rate:.0f}%) — dropping {total - on_disk} iCloud-only frames."
            )
        return kept

    typer.echo(
        f"Apple Photos: only {on_disk}/{total} originals on disk "
        f"({100*rate:.0f}%) — looks like Optimize Mac Storage is on."
    )

    if not force:
        try:
            ans = input(
                "Screen local thumbnails and download missing originals from iCloud "
                "to /tmp? [y/N] "
            ).strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            typer.echo("Aborted. Re-run with --force to skip the prompt.", err=True)
            raise typer.Exit(1)

    from selfies_for_a_year.icloud import (
        CACHE_DIR,
        authorize,
        download_many,
        screen_derivatives,
    )
    from selfies_for_a_year.photos import _DEFAULT_LIBRARY

    ok, msg = authorize()
    if not ok:
        typer.echo(f"Photos authorization failed: {msg}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Screening {total} derivatives...")
    last_pct = [-1]
    def screen_progress(i, n, _uuid, _ok):
        pct = (i * 100) // max(n, 1)
        if pct != last_pct[0] and pct % 10 == 0:
            last_pct[0] = pct
            typer.echo(f"  ...{pct}% ({i}/{n})")

    keeper_uuids = set(screen_derivatives(
        photos,
        library=_DEFAULT_LIBRARY,
        width=width,
        height=height,
        min_face_fraction=min_face_fraction,
        progress_cb=screen_progress,
    ))
    typer.echo(f"  {len(keeper_uuids)} candidates survived screening.")

    to_download = [p.uuid for p in photos if p.uuid in keeper_uuids and not p.is_materialized]
    if to_download:
        typer.echo(f"Downloading {len(to_download)} originals from iCloud to {CACHE_DIR}...")
        last_pct = [-1]
        def dl_progress(i, n, _res):
            pct = (i * 100) // max(n, 1)
            if pct != last_pct[0] and pct % 5 == 0:
                last_pct[0] = pct
                typer.echo(f"  ...{pct}% ({i}/{n})")
        results = download_many(to_download, CACHE_DIR, progress_cb=dl_progress)
        successes = sum(1 for r in results.values() if r.path is not None)
        typer.echo(f"  downloaded {successes}/{len(to_download)}.")
    else:
        results = {}

    # Rewrite paths for downloaded photos; drop everything else that's
    # still not on disk.
    kept = []
    for p in photos:
        if p.is_materialized:
            kept.append(p)
            continue
        r = results.get(p.uuid)
        if r is not None and r.path is not None:
            p.path = r.path
            p.is_materialized = True
            kept.append(p)
        else:
            reason = (
                "filtered out by derivative screening" if p.uuid not in keeper_uuids
                else (r.error if r is not None else "no download attempted")
            )
            manifest.append((
                p.date, p.path, "dropped",
                f"iCloud original not materialized: {reason}",
            ))
    return kept


def _collect_sources(
    input_dirs: list[Path],
    apple_photos_name: str | None,
    year_range: tuple[date, date] | None,
    show_label: bool,
    since: date | None = None,
    until: date | None = None,
    exclude: list[str] | None = None,
    *,
    width: int = 1080,
    height: int = 1080,
    min_face_fraction: float = 0.0,
    materialize_threshold: float = 0.50,
    force: bool = False,
    max_per_day: int = 1,
) -> tuple[list[Path], list[datetime], list[str], list[_FaceHint], list[_AppleQuality], list[ManifestEntry], dict[date, list[_SourceItem]]]:
    """Collect and merge photo paths from all sources.

    Returns (paths, dates, labels, face_hints, apple_quality, manifest,
    apple_fallbacks) sorted chronologically, one photo per day, with selfie-dir
    photos taking priority over Apple Photos. `apple_fallbacks` maps a day to its
    runner-up Apple candidates (best-first) for the align-aware fallback.

    Face hints: (center_x, center_y, size) for Apple Photos images (center as
    fractions of image dimensions, size as fraction of image area from the
    Photos DB face detection), None for selfie-dir images.

    Apple quality: (face_quality, tastefully_blurred) for Apple Photos images,
    None for selfie-dir images. Consumed by pre-alignment quality gates.

    Manifest: entries for photos dropped during collection (dedup, priority).
    """
    manifest: list[ManifestEntry] = []

    # Collect selfie-dir photos
    dir_items: list[_SourceItem] = []
    for input_dir in input_dirs:
        for p in discover_images(input_dir):
            dir_items.append((p, sort_key(p), None, None))
    dir_items.sort(key=lambda item: item[1])

    # Apply year filter to selfie-dir photos
    if year_range is not None and dir_items:
        year_start, year_end = year_range
        dir_items = [
            item for item in dir_items
            if year_start <= item[1].date() < year_end
        ]

    # Collect Apple Photos
    photos_items: list[_SourceItem] = []
    # Per-day runner-up candidates, best-quality-first, for the align-aware
    # fallback: when the chosen photo for a day fails face alignment, compile
    # retries these before giving the day up (see the fallback pass in `compile`).
    apple_fallbacks: dict[date, list[_SourceItem]] = {}
    if apple_photos_name is not None:
        from selfies_for_a_year.photos import deduplicate_by_day, find_person, query_photos

        person = find_person(apple_photos_name)
        if person is None:
            typer.echo(
                f"Error: no person named {apple_photos_name!r} found in Apple Photos.\n"
                "Use 'selfies-for-a-year list-people' to see available names.",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(
            f"Apple Photos: found {person.display_name!r} "
            f"({person.face_count} photos in library)"
        )

        all_photos = query_photos(
            person.person_id,
            year_start=year_range[0] if year_range else None,
            year_end=year_range[1] if year_range else None,
            include_unmaterialized=True,
        )

        # In one-per-day mode, record dedup losers and keep them (best-first) as
        # per-day fallbacks so a day whose winner fails alignment can be recovered
        # from a runner-up instead of vanishing (issue #47). With max_per_day > 1
        # the extra depth already provides same-day alternatives, so no separate
        # fallback map is needed.
        if max_per_day == 1:
            from selfies_for_a_year.photos import pick_best_photo
            by_day: dict[date, list] = {}
            for p in all_photos:
                by_day.setdefault(p.date.date(), []).append(p)
            for day in sorted(by_day):
                candidates = by_day[day]
                if len(candidates) > 1:
                    best = pick_best_photo(candidates)
                    losers = sorted(
                        (p for p in candidates if p is not best),
                        key=lambda p: p.face_quality, reverse=True,
                    )
                    apple_fallbacks[day] = [
                        (p.path, p.date,
                         (p.face_center_x, p.face_center_y, p.face_size),
                         (p.face_quality, p.tastefully_blurred))
                        for p in losers
                    ]
                    for p in losers:
                        manifest.append((
                            p.date, p.path, "dropped",
                            f"dedup: lost to {best.path.name} "
                            f"(quality {p.face_quality:.3f} vs {best.face_quality:.3f})",
                        ))

        photos = deduplicate_by_day(all_photos, max_per_day=max_per_day)

        # If too many originals live in iCloud, offer to screen derivatives
        # and materialize just the keepers to /tmp.
        photos = _maybe_materialize_from_icloud(
            photos,
            width=width,
            height=height,
            min_face_fraction=min_face_fraction,
            threshold=materialize_threshold,
            force=force,
            manifest=manifest,
        )
        photos_items = [
            (
                p.path,
                p.date,
                (p.face_center_x, p.face_center_y, p.face_size),
                (p.face_quality, p.tastefully_blurred),
            )
            for p in photos
        ]

    # Merge: selfie-dir takes priority per calendar day
    if dir_items and photos_items:
        dir_days: set[date] = {dt.date() for _, dt, _, _ in dir_items}

        # Record Apple Photos dropped due to selfie-dir priority
        for item in photos_items:
            if item[1].date() in dir_days:
                manifest.append((
                    item[1], item[0], "dropped",
                    "selfie-dir photo takes priority for this day",
                ))

        # Only keep Apple Photos for days not in selfie-dir
        photos_items = [
            item for item in photos_items
            if item[1].date() not in dir_days
        ]

        all_items = dir_items + photos_items
        all_items.sort(key=lambda item: item[1])

        source_counts = f"{len(dir_items)} from selfie dirs, {len(photos_items)} from Apple Photos"
        typer.echo(f"Merged: {len(all_items)} photos ({source_counts})")
    elif dir_items:
        all_items = dir_items
    elif photos_items:
        all_items = photos_items
    else:
        typer.echo("No images found from any source.", err=True)
        raise typer.Exit(1)

    if since is not None or until is not None:
        before = len(all_items)
        kept = []
        for item in all_items:
            d = item[1].date()
            if since is not None and d < since:
                manifest.append((item[1], item[0], "dropped", f"--since={since.isoformat()} filter"))
                continue
            if until is not None and d > until:
                manifest.append((item[1], item[0], "dropped", f"--until={until.isoformat()} filter"))
                continue
            kept.append(item)
        all_items = kept
        typer.echo(f"Date filter: {len(all_items)} kept of {before} (since={since}, until={until}).")

    if exclude:
        before = len(all_items)
        kept = []
        for item in all_items:
            name = item[0].name
            hit = next((pat for pat in exclude if pat in name), None)
            if hit is not None:
                manifest.append((item[1], item[0], "dropped", f"--exclude match: {hit!r}"))
                continue
            kept.append(item)
        all_items = kept
        typer.echo(f"Exclude filter: {len(all_items)} kept of {before} (patterns={exclude}).")

    paths = [item[0] for item in all_items]
    dates = [item[1] for item in all_items]
    labels = [_label_from_date(item[1]) if show_label else "" for item in all_items]
    face_hints = [item[2] for item in all_items]
    apple_quality = [item[3] for item in all_items]
    return paths, dates, labels, face_hints, apple_quality, manifest, apple_fallbacks


@app.command()
def compile(
    ctx: typer.Context,
    input_dir: Annotated[
        list[Path] | None,
        typer.Argument(help="Directories containing selfie images (one or more)."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Output .mp4 file path."),
    ] = Path("output.mp4"),
    apple_photos: Annotated[str | None, typer.Option(help="Pull photos of this person from Apple Photos.")] = None,
    year: Annotated[str | None, typer.Option(help="Filter all sources to year or range: '2026' or '2024-2026'.")] = None,
    since: Annotated[str | None, typer.Option(help="Drop photos dated before YYYY-MM-DD.")] = None,
    until: Annotated[str | None, typer.Option(help="Drop photos dated after YYYY-MM-DD.")] = None,
    exclude: Annotated[str | None, typer.Option(help="Comma-separated substrings (e.g. UUID prefixes) — any photo whose filename contains one is dropped.")] = None,
    width: Annotated[int, typer.Option(help="Output width in pixels.")] = 1080,
    height: Annotated[int, typer.Option(help="Output height in pixels.")] = 1080,
    align: Annotated[bool, typer.Option(help="Align faces across frames.")] = True,
    normalize: Annotated[bool, typer.Option(help="Normalize color/exposure across frames (experimental).")] = False,
    duration: Annotated[int, typer.Option(help="Time each photo is shown, in milliseconds.")] = 150,
    fade: Annotated[str, typer.Option(help="Crossfade between photos: '40%%' of photo duration, or '60ms' absolute.")] = "40%",
    label: Annotated[bool, typer.Option(help="Overlay date labels on frames.")] = True,
    fade_in: Annotated[str, typer.Option(help="Fade-in duration: milliseconds or '5%%' of video length (0 to disable).")] = "0",
    fade_out: Annotated[str, typer.Option(help="Fade-out duration: milliseconds or '5%%' of video length (0 to disable).")] = "0",
    fade_in_color: Annotated[str, typer.Option(help="Fade-in color: 'black', 'white', or hex like '#ff0000'.")] = "black",
    fade_out_color: Annotated[str, typer.Option(help="Fade-out color: 'black', 'white', or hex like '#ff0000'.")] = "black",
    music: Annotated[Path | None, typer.Option(help="Audio file to add as music track.")] = None,
    music_fade_out: Annotated[float, typer.Option(help="Audio fade-out duration in seconds.")] = 2.0,
    fit_to_music: Annotated[bool, typer.Option(help="Scale video duration to match music length (requires --music).")] = False,
    beat_sync: Annotated[bool, typer.Option(help="Sync frame transitions to detected beats in --music. Hard cuts only; --fade/--fade-in/--fade-out are ignored.")] = False,
    max_photos_per_second: Annotated[float, typer.Option(help="[--beat-sync] Visual ceiling. Auto-picks the largest subdivision (photos per beat) keeping the rate under this cap.")] = 4.0,
    min_photos_per_beat: Annotated[float, typer.Option(help="[--beat-sync] Pace floor. Auto-backoff (when too few photos for the song) won't drop below this. Hitting the floor trims trailing audio. Set <1.0 to stretch a small photo set across a long song.")] = 1.0,
    beat_speed: Annotated[float | None, typer.Option(help="[--beat-sync] Manual subdivision override (photos per beat). Bypasses both bounds above. e.g. 2.0=eighth notes, 0.5=every 2nd beat.")] = None,
    beat_thresh: Annotated[float, typer.Option(help="[--beat-sync] Onset-strength threshold (0–1) below which a beat is treated as ambient. Ambient sections fall back to --duration timing.")] = 0.30,
    force: Annotated[bool, typer.Option(help="[--beat-sync] Render best-effort even when --max-photos-per-second / --min-photos-per-beat can't be jointly satisfied. Loosened constraints are reported.")] = False,
    vary_pace: Annotated[bool, typer.Option(help="[--beat-sync] Three-tier pacing: pick a small number of the song's most intense and most quiet sustained sections; speed up the intense ones, slow down the quiet ones.")] = False,
    intense_multiplier: Annotated[float, typer.Option(help="[--vary-pace] Rate multiplier for the song's most intense sections. 2.0 = double the normal photo rate.")] = 2.0,
    slow_multiplier: Annotated[float, typer.Option(help="[--vary-pace] Rate multiplier for the song's quietest sections. 0.5 = half the normal photo rate.")] = 0.5,
    max_intense: Annotated[int, typer.Option(help="[--vary-pace] How many intense sections to pick (ranked by total energy). Keep small (1–2) for clear punctuation.")] = 2,
    max_slow: Annotated[int, typer.Option(help="[--vary-pace] How many quiet sections to pick. Typically 1 (intro/breakdown/outro).")] = 1,
    min_section_seconds: Annotated[float, typer.Option(help="[--vary-pace] Minimum sustained duration (seconds) for a section to qualify. Filters out blippy peaks.")] = 5.0,
    min_normal_bridge_beats: Annotated[float, typer.Option(help="[--vary-pace] Tiny 'normal' bridges between two overlay regions (slow→intense etc.) shorter than this many beats get absorbed into the next region so the transition is direct.")] = 8.0,
    snap_to_grid: Annotated[bool, typer.Option(help="[--vary-pace] Snap --intense/--slow-multiplier to the nearest power-of-2 musical fraction (1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16) so cuts stay on the 4/4 grid. Disable to allow triplets/polyrhythms.")] = True,
    tier_lead_seconds: Annotated[float, typer.Option(help="[--vary-pace] Shift tier-region detection earlier by N seconds for anticipation (intense kicks in just before the audible cue). 0 = no shift. Try 0.3–1.0s for a music-video feel.")] = 0.0,
    onset_anchor: Annotated[str, typer.Option(help="[--beat-sync] Where the beat grid is fiction (sparse/rubato passages), cut on the real note strikes instead: 'auto' (detect those spans by grid support), 'never' (always trust the grid), 'always' (treat the whole song as rubato). Metronome dot moves onto the strikes there.")] = "auto",
    preview_seconds: Annotated[float | None, typer.Option(help="Fast feel-check: render only the first N seconds. Keeps full-song beat/pace analysis but aligns just the photos needed for the clip, so a ~5min render becomes ~15s. For iteration, not final output.")] = None,
    beat_crossfade: Annotated[bool, typer.Option(help="[--beat-sync] Replace hard cuts with continuous crossfade: each photo peaks at its beat and morphs into the next over the segment.")] = False,
    debug: Annotated[bool, typer.Option(help="Overlay ALL review HUDs: pacing tier + song/BPM, source filename, the track-progression bar with playhead, and the metronome dot (flashes on each cut target — strikes in onset-anchor spans). On for iteration/review; leave off for a clean production render. Forces constant-fps rendering.")] = False,
    emit_progression: Annotated[bool, typer.Option(help="[--beat-sync] Print the track progression model (states + pacing sanity metrics) to stdout, then continue rendering.")] = False,
    emit_progression_json: Annotated[Path | None, typer.Option(help="[--beat-sync] Write the track progression model as JSON to this path.")] = None,
    analyze_only: Annotated[bool, typer.Option(help="[--beat-sync] Run beat/pacing analysis and emit the progression model, then exit WITHOUT rendering video. Fast iteration loop for pacing params.")] = False,
    min_face_fraction: Annotated[
        float,
        typer.Option(
            help=(
                "Skip Apple Photos frames where the face area (per Apple's hint) "
                "is below this fraction of image area. 0 to disable."
            ),
        ),
    ] = 0.030,
    min_face_quality: Annotated[
        float,
        typer.Option(
            help=(
                "Skip Apple Photos frames where Apple's face-quality score "
                "(ZDETECTEDFACE.ZQUALITY) is below this threshold. Best "
                "available blur signal; see #19. 0 to disable."
            ),
        ),
    ] = 0.18,
    max_tastefully_blurred: Annotated[
        float,
        typer.Option(
            help=(
                "Skip Apple Photos frames where Apple's 'tastefully blurred' "
                "score (ZTASTEFULLYBLURREDSCORE) is above this threshold — "
                "Apple's own signal for unintentional blur. 0 to disable."
            ),
        ),
    ] = 0.30,
    max_per_day: Annotated[
        int,
        typer.Option(
            help=(
                "Max Apple Photos frames to keep per calendar day. Default 1 (one "
                "selfie per day). Raise it for a subject whose photos cluster (many "
                "on some days, none for weeks) so a fixed-length song can be filled "
                "— at the cost of several same-day frames in a row. 0 = no cap."
            ),
        ),
    ] = 1,
    manifest: Annotated[
        Path | None,
        typer.Option(
            help="Write a TSV manifest of every photo considered: date, path, status, reason.",
        ),
    ] = None,
) -> None:
    """Compile a directory of selfie images into a timelapse MP4 video."""
    check_ffmpeg()

    # Normalize input_dirs
    input_dirs = input_dir or []

    # Validate source inputs
    if not input_dirs and apple_photos is None:
        typer.echo("Error: provide INPUT_DIR(s) and/or --apple-photos.", err=True)
        raise typer.Exit(1)

    for d in input_dirs:
        if not d.is_dir():
            typer.echo(f"Error: {d} is not a directory.", err=True)
            raise typer.Exit(1)

    if music is not None and not music.is_file():
        typer.echo(f"Error: {music} is not a file.", err=True)
        raise typer.Exit(1)

    if fit_to_music and music is None:
        typer.echo("Error: --fit-to-music requires --music.", err=True)
        raise typer.Exit(1)

    # Default to the settled beat-synced pacing pipeline. Providing --music alone
    # gives a beat-synced, pace-varied render (segment tiers + occupancy base
    # pace); opt out with --no-beat-sync (plain slideshow) or --no-vary-pace.
    _src = ctx.get_parameter_source
    _CMD = click.core.ParameterSource.COMMANDLINE
    if music is not None and _src("beat_sync") != _CMD:
        beat_sync = True
    if beat_sync and _src("vary_pace") != _CMD:
        vary_pace = True

    # One --debug flag drives every review HUD; the render helpers still take the
    # individual toggles internally.
    debug_tier_overlay = debug
    debug_filename_overlay = debug
    debug_progression_overlay = debug
    debug_metronome = debug

    if beat_sync and music is None:
        typer.echo("Error: --beat-sync requires --music.", err=True)
        raise typer.Exit(1)

    if beat_sync and fit_to_music:
        typer.echo("Error: --beat-sync and --fit-to-music are mutually exclusive.", err=True)
        raise typer.Exit(1)

    if (emit_progression or emit_progression_json or analyze_only) and not beat_sync:
        typer.echo(
            "Error: --emit-progression / --emit-progression-json / --analyze-only "
            "require --beat-sync (the progression model is built from the beat timeline).",
            err=True,
        )
        raise typer.Exit(1)

    if onset_anchor not in ("auto", "never", "always"):
        typer.echo(f"Error: --onset-anchor must be 'auto', 'never', or 'always', got '{onset_anchor}'.", err=True)
        raise typer.Exit(1)


    # --duration is meaningless under --fit-to-music (the per-photo time is
    # derived from audio length / photo count). Warn loudly if the user
    # explicitly set both, since the duration value will be silently ignored.
    if fit_to_music and ctx.get_parameter_source("duration") == click.core.ParameterSource.COMMANDLINE:
        typer.echo(
            f"Warning: --duration {duration} is ignored under --fit-to-music "
            f"(per-photo duration is computed from audio length).",
            err=True,
        )

    # Parse year filter
    year_range = _parse_year(year) if year is not None else None
    since_date = date.fromisoformat(since) if since is not None else None
    until_date = date.fromisoformat(until) if until is not None else None

    # Collect photos from all sources
    typer.echo("Scanning sources ...")
    exclude_patterns = [s.strip() for s in exclude.split(",") if s.strip()] if exclude else None
    paths, dates, labels, face_hints, apple_quality, manifest_entries, apple_fallbacks = _collect_sources(
        input_dirs, apple_photos, year_range, label, since=since_date, until=until_date,
        exclude=exclude_patterns,
        width=width, height=height,
        min_face_fraction=min_face_fraction, force=force,
        max_per_day=max_per_day,
    )

    # Read audio length now so we can fit-to-music after alignment knows
    # how many frames actually survived (skipped frames would otherwise
    # leave the video shorter than the music).
    audio_seconds: float | None = None
    if fit_to_music:
        assert music is not None
        audio_seconds = get_duration(music)
        typer.echo(
            f"Fitting to music ({audio_seconds:.1f}s); "
            f"per-photo duration will be sized after alignment."
        )
    else:
        typer.echo(f"Found {len(paths)} images. ({duration}ms/photo at {FPS}fps)")

    # Preview mode: only the first N seconds are encoded, so only the photos that
    # can appear in that window need aligning. Cap at the ceiling pace so we never
    # starve the clip; the occupancy pace is slower, leaving a safe margin.
    if preview_seconds is not None:
        k = max(2, int(preview_seconds * max_photos_per_second) + 2)
        if k < len(paths):
            paths = paths[:k]
            dates = dates[:k]
            labels = labels[:k]
            face_hints = face_hints[:k]
            apple_quality = apple_quality[:k]
            typer.echo(f"Preview: first {preview_seconds:g}s — aligning {len(paths)} photos only.")

    # --- Pass 1: Align/prepare all frames ---
    if align:
        from selfies_for_a_year.align import FaceAligner

        typer.echo("Face alignment enabled.")
        prepared: list[Image.Image] = []
        kept_labels: list[str] = []
        kept_dates: list[datetime] = []
        kept_paths: list[Path] = []

        # Pre-alignment gates, applied in this order. First-match wins.
        # Each gate has a name, the flag that controls it, and the count of
        # frames it dropped. Reported in this order so the user can see
        # where photos disappeared.
        gate_drops: dict[str, tuple[str, int]] = {
            "tiny_face": (f"face area < --min-face-fraction={min_face_fraction}", 0),
            "low_face_quality": (f"Apple face quality < --min-face-quality={min_face_quality}", 0),
            "tastefully_blurred": (f"Apple tastefully-blurred > --max-tastefully-blurred={max_tastefully_blurred}", 0),
            "no_usable_face": ("alignment found no usable face", 0),
        }

        def _bump(gate: str, delta: int = 1) -> None:
            desc, n = gate_drops[gate]
            gate_drops[gate] = (desc, n + delta)

        # Records which day/gate each Apple photo failed on, so the fallback pass
        # can retry that day's runner-ups and, on success, credit the recovery
        # back to the exact gate the winner tripped.
        failed_apple: dict[date, tuple[str, str]] = {}  # day -> (label, gate)

        with FaceAligner(width, height) as aligner:
            def _align_one(path, hint, apple):
                """Pre-gates + face alignment for one photo. Returns
                (aligned_img|None, gate_key|None, reason|None). Selfie-dir frames
                (no hint/apple) bypass the cheap Apple gates — the user vetted
                them. Pure w.r.t. accounting: the caller bumps counters/manifest."""
                if (
                    min_face_fraction > 0
                    and hint is not None
                    and hint[2] < min_face_fraction
                ):
                    return None, "tiny_face", (
                        f"face area {hint[2]:.4f} < min-face-fraction {min_face_fraction}"
                    )
                if apple is not None and min_face_quality > 0 and apple[0] < min_face_quality:
                    return None, "low_face_quality", (
                        f"face quality {apple[0]:.3f} < min-face-quality {min_face_quality}"
                    )
                if (
                    apple is not None
                    and apple[1] is not None
                    and max_tastefully_blurred > 0
                    and apple[1] > max_tastefully_blurred
                ):
                    return None, "tastefully_blurred", (
                        f"tastefully-blurred {apple[1]:.3f} > max {max_tastefully_blurred}"
                    )
                aligned = aligner.align(load_image(path), face_hint=hint)
                if aligned is None:
                    return None, "no_usable_face", "no usable face detected"
                return aligned, None, None

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Aligning frames", total=len(paths))
                for path, dt, lbl, hint, apple in zip(
                    paths, dates, labels, face_hints, apple_quality
                ):
                    aligned_img, gate, reason = _align_one(path, hint, apple)
                    if aligned_img is not None:
                        prepared.append(aligned_img)
                        kept_labels.append(lbl)
                        kept_dates.append(dt)
                        kept_paths.append(path)
                        manifest_entries.append((dt, path, "kept", ""))
                    else:
                        _bump(gate)
                        manifest_entries.append((dt, path, "dropped", reason))
                        # An Apple photo (has a hint) that failed can be recovered
                        # from a runner-up on the same day; remember the day + gate.
                        if hint is not None and dt.date() in apple_fallbacks:
                            failed_apple.setdefault(dt.date(), (lbl, gate))
                    progress.update(task, advance=1)

            # --- Fallback pass: recover dropped days from same-day runner-ups ---
            # A day whose best-quality photo failed alignment is not necessarily a
            # day with no usable face — the winner is picked on Apple quality
            # BEFORE mediapipe runs. Retry that day's other photos (best-first)
            # until one aligns, so a stray no-face winner doesn't cost the day
            # (issue #49). Extra alignment work is paid only for failed days.
            recovered_fallback = 0
            if failed_apple:
                for day, (lbl, gate) in failed_apple.items():
                    for fpath, fdt, fhint, fapple in apple_fallbacks.get(day, []):
                        if not Path(fpath).exists():
                            continue  # unmaterialized iCloud original; can't align
                        aligned_img, _g, _r = _align_one(fpath, fhint, fapple)
                        if aligned_img is not None:
                            prepared.append(aligned_img)
                            kept_labels.append(lbl)
                            kept_dates.append(fdt)
                            kept_paths.append(fpath)
                            manifest_entries.append((
                                fdt, fpath, "kept", "recovered via same-day fallback",
                            ))
                            _bump(gate, -1)  # the winner's drop is now made good
                            recovered_fallback += 1
                            break
            if recovered_fallback:
                # Recovered photos were appended out of order; restore date order.
                order = sorted(range(len(kept_dates)), key=lambda i: kept_dates[i])
                prepared = [prepared[i] for i in order]
                kept_labels = [kept_labels[i] for i in order]
                kept_dates = [kept_dates[i] for i in order]
                kept_paths = [kept_paths[i] for i in order]
                typer.echo(
                    f"Recovered {recovered_fallback} day(s) via same-day fallback "
                    "(winner failed alignment, a runner-up aligned)."
                )

            if aligner.recovered_count:
                typer.echo(
                    f"Recovered {aligner.recovered_count} frame(s) via two-stage detection."
                )
            if aligner.fallback_count:
                typer.echo(
                    f"Used center-crop fallback on {aligner.fallback_count} "
                    f"selfie-dir frame(s) where detection failed.",
                    err=True,
                )

            # Per-gate drop summary, in gate application order.
            total_dropped = sum(n for _, n in gate_drops.values())
            total_in = len(paths)
            if total_dropped:
                typer.echo(
                    f"\nDropped {total_dropped}/{total_in} frame(s) "
                    f"({total_dropped/total_in*100:.0f}%) by quality gates "
                    f"(in order):",
                    err=True,
                )
                for desc, n in gate_drops.values():
                    if n:
                        typer.echo(
                            f"  {n:>4d}  ({n/total_in*100:4.1f}%)  {desc}",
                            err=True,
                        )

            # Heads-up when the total drop rate is large enough to materially
            # shorten the video. Points at the main knobs the user can loosen.
            if total_in > 0 and total_dropped / total_in > 0.50:
                typer.echo(
                    "\nWarning: over half of source photos were dropped. "
                    "To loosen: lower --min-face-fraction / --min-face-quality, "
                    "raise --max-tastefully-blurred, or check for "
                    "misframed/wrong-person photos in the source.",
                    err=True,
                )

        labels = kept_labels
    else:
        prepared = []
        kept_dates = list(dates)
        kept_paths = list(paths)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing frames", total=len(paths))
            for path, dt in zip(paths, dates):
                prepared.append(load_and_prepare(path, width, height))
                manifest_entries.append((dt, path, "kept", ""))
                progress.update(task, advance=1)

    # --- Beat-sync path (mutually exclusive with --fit-to-music) ---
    if beat_sync:
        from selfies_for_a_year.beats import build_timeline

        if not prepared:
            typer.echo("Error: no frames survived alignment; cannot beat-sync.", err=True)
            raise typer.Exit(1)

        assert music is not None
        typer.echo(f"Detecting beats in {music.name} ...")
        timeline = build_timeline(
            music,
            kept_dates,
            max_photos_per_second=max_photos_per_second,
            min_photos_per_beat=min_photos_per_beat,
            beat_speed=beat_speed,
            beat_thresh=beat_thresh,
            fallback_duration_seconds=duration / 1000.0,
            vary_pace=vary_pace,
            intense_multiplier=intense_multiplier,
            slow_multiplier=slow_multiplier,
            max_intense=max_intense,
            max_slow=max_slow,
            min_section_seconds=min_section_seconds,
            min_normal_bridge_beats=min_normal_bridge_beats,
            snap_to_grid=snap_to_grid,
            tier_lead_seconds=tier_lead_seconds,
            pace_model="segment",
            base_pace="occupancy",
            onset_anchor=onset_anchor,
        )
        typer.echo(timeline.summary())

        # Track progression model: the linear pacing "sheet music" plus
        # warn-only sanity metrics. Cheap to build from the timeline; drives
        # both the --emit-progression* diagnostics and the overlay bar.
        from selfies_for_a_year.beats import TrackProgression

        progression = TrackProgression.from_timeline(timeline)
        if emit_progression or analyze_only:
            typer.echo("")
            typer.echo(progression.render_text())
        if emit_progression_json is not None:
            import json

            emit_progression_json.write_text(json.dumps(progression.to_dict(), indent=2))
            typer.echo(f"Wrote progression JSON to {emit_progression_json}")
        if analyze_only:
            typer.echo("\n(--analyze-only: skipping render)")
            return

        # States for the overlay bar: (start, end, tier) runs.
        progression_states = [(s.start, s.end, s.tier) for s in progression.states]

        # Validate bounds. Bounds are only enforced in auto mode; if the user
        # set --beat-speed they've explicitly opted out of auto-pick. Preview
        # mode deliberately uses only the first few photos, so the "too few to
        # fill the song" guard doesn't apply — skip it like --force.
        if beat_speed is None and not force and preview_seconds is None:
            problems: list[str] = []
            remedies: list[str] = []
            if timeline.bounds_violation == "ceiling":
                bps = timeline.bpm / 60.0
                rate = bps * timeline.subdivision
                problems.append(
                    f"At {timeline.bpm:.1f} BPM with --min-photos-per-beat="
                    f"{min_photos_per_beat}, the resulting rate ({rate:.1f}/sec) "
                    f"exceeds --max-photos-per-second={max_photos_per_second}."
                )
                remedies.append(
                    f"Raise --max-photos-per-second to ≥ {rate:.1f}, "
                    f"or lower --min-photos-per-beat below {min_photos_per_beat}."
                )
            if timeline.bounds_violation == "floor":
                problems.append(
                    f"No subdivision ≥ --min-photos-per-beat={min_photos_per_beat} "
                    f"fits under --max-photos-per-second={max_photos_per_second} "
                    f"at {timeline.bpm:.1f} BPM."
                )
                remedies.append(
                    "Lower --min-photos-per-beat, or raise --max-photos-per-second."
                )
            if timeline.audio_trimmed_seconds > 0:
                problems.append(
                    f"Too few photos ({len(kept_dates)}) to fill the song at "
                    f"--min-photos-per-beat={min_photos_per_beat}; "
                    f"{timeline.audio_trimmed_seconds:.1f}s of audio would be trimmed."
                )
                remedies.append(
                    "Lower --min-photos-per-beat (e.g. 0.5 or 0.25) to stretch "
                    "photos across the full song, or accept the trim with --force."
                )

            if problems:
                typer.echo("\nBeat-sync constraints can't be satisfied:", err=True)
                for p in problems:
                    typer.echo(f"  - {p}", err=True)
                typer.echo("\nFix one of:", err=True)
                for r in remedies:
                    typer.echo(f"  - {r}", err=True)
                typer.echo(
                    "\nOr re-run with --force to render best-effort. "
                    "Use --beat-speed FLOAT to set the subdivision manually.",
                    err=True,
                )
                raise typer.Exit(1)

        # The progression playhead and metronome need a per-frame clock, so
        # any render using them must go through the constant-fps generator —
        # even hard-cut mode, which otherwise holds one frame per segment.
        per_frame_overlays = debug_progression_overlay or debug_metronome
        # In onset-anchor spans the dot flashes on the real note strikes, not the
        # fictional grid — so the owner scores beat-match against what we cut on.
        metronome_beats = timeline.metronome_times() if debug_metronome else None

        if beat_crossfade or per_frame_overlays:
            frames_iter, n_out = _beat_frames(
                prepared,
                kept_labels,
                kept_paths,
                timeline.segments,
                fps=FPS,
                crossfade=beat_crossfade,
                debug_tier_overlay=debug_tier_overlay,
                debug_filename_overlay=debug_filename_overlay,
                song=music.stem,
                bpm=timeline.bpm,
                progression_states=progression_states if debug_progression_overlay else None,
                metronome_beats=metronome_beats,
                max_seconds=preview_seconds,
            )
            mode = "continuous crossfade" if beat_crossfade else "constant-fps hard cut"
            typer.echo(
                f"Encoding {len(timeline.segments)} segment(s) as {mode} "
                f"({n_out} frames at {FPS}fps, {n_out / FPS:.1f}s) ..."
            )
            compile_video(
                frames_iter,
                output,
                fps=FPS,
                width=width,
                height=height,
                total=n_out,
            )
            rendered_count = n_out
        else:
            # Production hard cuts: one held frame per segment, variable
            # duration. Efficient (no per-frame work) but no animated overlays.
            rendered: list[Image.Image] = []
            seg_durations: list[float] = []
            for seg in timeline.segments:
                img = prepared[seg.photo_index]
                framed = _overlay_label(img, kept_labels[seg.photo_index])
                if debug_tier_overlay or debug_filename_overlay:
                    framed = _overlay_debug(
                        framed,
                        tier=seg.tier if debug_tier_overlay else None,
                        duration=seg.duration if debug_tier_overlay else None,
                        filename=kept_paths[seg.photo_index].name if debug_filename_overlay else None,
                        song=music.stem if debug_tier_overlay else None,
                        bpm=timeline.bpm if debug_tier_overlay else None,
                    )
                rendered.append(framed)
                seg_durations.append(seg.duration)

            typer.echo(f"Encoding {len(rendered)} variable-duration segment(s) ...")
            compile_video_variable(
                rendered,
                seg_durations,
                output,
                width=width,
                height=height,
            )
            rendered_count = len(rendered)

        # Mux audio. mux_audio's loop-and-trim logic naturally handles
        # video shorter than audio (truncate via -shortest is implicit in
        # the existing filter chain via atrim to video_duration).
        import tempfile

        typer.echo(f"Adding audio from {music} ...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        output.rename(tmp_path)
        try:
            mux_audio(tmp_path, music, output, audio_fade_out=music_fade_out)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Manifest write (same as the constant path).
        if manifest is not None:
            manifest_entries.sort(key=lambda e: e[0])
            with open(manifest, "w") as f:
                f.write("date\tpath\tstatus\treason\n")
                for dt, path, status, reason in manifest_entries:
                    f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}\t{path}\t{status}\t{reason}\n")

        typer.echo(
            f"Done! {output} ({rendered_count} frames, {rendered_count / FPS:.1f}s)"
        )
        return

    # Now that we know how many frames actually survived alignment, size
    # per-photo duration to match the audio (if requested).
    if fit_to_music:
        assert audio_seconds is not None
        if not prepared:
            typer.echo("Error: no frames survived alignment; cannot fit to music.", err=True)
            raise typer.Exit(1)
        duration = round(audio_seconds / len(prepared) * 1000)
        typer.echo(
            f"Fitting {len(prepared)} surviving frame(s) to {audio_seconds:.1f}s of audio: "
            f"{duration}ms per photo."
        )
        if duration < 50:
            typer.echo("Warning: very short per-photo duration (<50ms). Consider fewer photos or longer audio.", err=True)
        elif duration > 2000:
            typer.echo("Warning: very long per-photo duration (>2s). Consider more photos or shorter audio.", err=True)

    # Convert duration/fade to frame counts (now that duration is final)
    total_frames_per_photo = max(1, round(duration / 1000 * FPS))
    fade_frames = _resolve_frames(fade, total_frames_per_photo, "fade")
    fade_frames = min(fade_frames, total_frames_per_photo)  # can't fade longer than hold
    hold_frames = total_frames_per_photo - fade_frames
    typer.echo(
        f"Encoding {len(prepared)} frame(s): "
        f"{duration}ms/photo ({hold_frames} hold + {fade_frames} fade frames at {FPS}fps)."
    )

    # --- Pass 2: Color normalization ---
    if normalize:
        from selfies_for_a_year.color import normalize_colors

        typer.echo("Normalizing color/exposure...")
        prepared = normalize_colors(prepared)

    # --- Pass 3: Labels, crossfade, and encoding ---
    labeled_frames: Iterator[tuple[Image.Image, str]] = iter(zip(prepared, labels))

    # `prepared` may be shorter than `paths` if frames were skipped during alignment.
    n_frames = len(prepared)
    if fade_frames > 0 or hold_frames > 1:
        frames = _crossfade(labeled_frames, hold_frames, fade_frames)
        output_frame_count = n_frames * hold_frames + max(0, n_frames - 1) * fade_frames
    else:
        frames = _no_crossfade(labeled_frames)
        output_frame_count = n_frames

    # --- Bookend fades (fade from/to solid color at start/end) ---
    fade_in_frame_count = _resolve_frames(fade_in, output_frame_count, "fade-in")
    fade_out_frame_count = _resolve_frames(fade_out, output_frame_count, "fade-out")

    if fade_in_frame_count > 0 or fade_out_frame_count > 0:
        in_rgb = _parse_color(fade_in_color)
        out_rgb = _parse_color(fade_out_color)
        frames = _apply_bookend_fades(
            frames, width, height,
            fade_in_frame_count, fade_out_frame_count,
            in_rgb, out_rgb,
            output_frame_count,
        )

    compile_video(
        frames,
        output,
        fps=FPS,
        width=width,
        height=height,
        total=output_frame_count,
    )

    video_duration = output_frame_count / FPS

    # --- Mux audio track ---
    if music is not None:
        import tempfile

        typer.echo(f"Adding audio from {music} ...")
        # Write video to a temp file, then mux audio into the final output
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        output.rename(tmp_path)
        try:
            mux_audio(tmp_path, music, output, audio_fade_out=music_fade_out)
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Write manifest ---
    if manifest is not None:
        manifest_entries.sort(key=lambda e: e[0])
        with open(manifest, "w") as f:
            f.write("date\tpath\tstatus\treason\n")
            for dt, path, status, reason in manifest_entries:
                f.write(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}\t{path}\t{status}\t{reason}\n")
        kept = sum(1 for _, _, s, _ in manifest_entries if s == "kept")
        dropped = sum(1 for _, _, s, _ in manifest_entries if s == "dropped")
        typer.echo(f"Manifest: {manifest} ({kept} kept, {dropped} dropped)")

    typer.echo(f"Done! {output} ({output_frame_count} frames, {video_duration:.1f}s at {FPS}fps)")


@app.command()
def list_people(
    min_faces: Annotated[int, typer.Option(help="Minimum face count to show.")] = 10,
) -> None:
    """List named people in Apple Photos, sorted by face count."""
    from selfies_for_a_year.photos import list_people as _list_people

    people = _list_people(min_faces=min_faces)
    if not people:
        typer.echo("No named people found in Apple Photos.")
        raise typer.Exit(0)

    typer.echo(f"{'Name':<30} {'Full Name':<35} {'Photos':>8}")
    typer.echo("-" * 75)
    for person in people:
        full = person.full_name or ""
        typer.echo(f"{person.display_name:<30} {full:<35} {person.face_count:>8}")
