"""CLI entry point for selfies-for-a-year."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import click
import typer
from PIL import Image, ImageDraw, ImageFont
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

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
    "slow": (90, 170, 255),       # blue
    "normal": (220, 220, 220),    # near-white
    "intense": (255, 95, 95),     # red
    "ambient": (255, 190, 80),    # amber
}


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

    Tier line uses a color per tier; filename stays white.
    """
    if tier is None and filename is None and song is None:
        return img
    img = img.copy()
    draw = ImageDraw.Draw(img)

    font_size = img.width // 44
    font = None
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Courier.ttc",
    ):
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size=font_size)

    rows: list[tuple[str, tuple[int, int, int]]] = []
    if song is not None:
        label = song if len(song) <= 30 else song[:29] + "…"
        if bpm is not None and bpm > 0:
            label = f"{label}  {bpm:.1f} BPM"
        rows.append((label, (200, 200, 255)))
    if tier is not None and duration is not None:
        rate = 1.0 / duration if duration > 0 else 0.0
        text = f"{tier.upper():<8}{rate:>5.2f}/s  {duration * 1000:>4.0f}ms"
        rows.append((text, _TIER_COLORS.get(tier, (255, 255, 255))))
    if filename is not None:
        if len(filename) > 40:
            filename = filename[:30] + "…" + filename[-8:]
        rows.append((filename, (255, 255, 255)))

    pad = max(5, img.width // 110)
    line_h = font_size + pad // 2
    block_h = line_h * len(rows) + pad
    widest = max(int(draw.textlength(t, font=font)) for t, _ in rows)
    block_w = widest + pad * 2
    date_top = img.height - img.height // 20 - img.width // 20
    x0 = pad
    y0 = date_top - block_h - pad
    draw.rectangle((x0, y0, x0 + block_w, y0 + block_h), fill=(0, 0, 0, 255))
    text_x = x0 + pad
    text_y = y0 + pad // 2
    for text, color in rows:
        draw.text((text_x, text_y), text, fill=color, font=font, anchor="lt")
        text_y += line_h
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


def _beat_crossfade_frames(
    prepared: list[Image.Image],
    labels: list[str],
    paths: list[Path],
    segments: list,
    *,
    fps: int,
    debug_tier_overlay: bool,
    debug_filename_overlay: bool,
    song: str | None = None,
    bpm: float | None = None,
) -> tuple[Iterator[Image.Image], int]:
    """Render beat-sync segments as a continuous crossfade.

    Each photo reaches peak opacity at its segment's start (the beat) and
    morphs into the next photo over the segment's duration. The final
    segment is held at full opacity for its duration.
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
            if seg_i >= last_i:
                yield decorated[last_i]
            else:
                alpha = (t - seg_start) / segments[seg_i].duration
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
                yield Image.blend(decorated[seg_i], decorated[seg_i + 1], alpha)

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


def _collect_sources(
    input_dirs: list[Path],
    apple_photos_name: str | None,
    year_range: tuple[date, date] | None,
    show_label: bool,
    since: date | None = None,
    until: date | None = None,
    exclude: list[str] | None = None,
) -> tuple[list[Path], list[datetime], list[str], list[_FaceHint], list[_AppleQuality], list[ManifestEntry]]:
    """Collect and merge photo paths from all sources.

    Returns (paths, labels, face_hints, apple_quality, manifest) sorted
    chronologically, one photo per day, with selfie-dir photos taking
    priority over Apple Photos.

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
    if apple_photos_name is not None:
        from selfies_for_a_year.photos import find_person, query_photos, deduplicate_by_day

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
        )

        # Record dedup losers before they're dropped
        from selfies_for_a_year.photos import pick_best_photo
        by_day: dict[date, list] = {}
        for p in all_photos:
            by_day.setdefault(p.date.date(), []).append(p)
        for day in sorted(by_day):
            candidates = by_day[day]
            if len(candidates) > 1:
                best = pick_best_photo(candidates)
                for p in candidates:
                    if p is not best:
                        manifest.append((
                            p.date, p.path, "dropped",
                            f"dedup: lost to {best.path.name} "
                            f"(quality {p.face_quality:.3f} vs {best.face_quality:.3f})",
                        ))

        photos = deduplicate_by_day(all_photos)
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
    return paths, dates, labels, face_hints, apple_quality, manifest


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
    beat_crossfade: Annotated[bool, typer.Option(help="[--beat-sync] Replace hard cuts with continuous crossfade: each photo peaks at its beat and morphs into the next over the segment.")] = False,
    debug_tier_overlay: Annotated[bool, typer.Option(help="[--vary-pace] Overlay the current pacing tier (slow/normal/intense/ambient) on each frame for visual debugging.")] = False,
    debug_filename_overlay: Annotated[bool, typer.Option(help="Overlay the source photo filename (truncated) on each frame for tracing back to originals.")] = False,
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

    if beat_sync and music is None:
        typer.echo("Error: --beat-sync requires --music.", err=True)
        raise typer.Exit(1)

    if beat_sync and fit_to_music:
        typer.echo("Error: --beat-sync and --fit-to-music are mutually exclusive.", err=True)
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
    paths, dates, labels, face_hints, apple_quality, manifest_entries = _collect_sources(
        input_dirs, apple_photos, year_range, label, since=since_date, until=until_date,
        exclude=exclude_patterns,
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

        def _bump(gate: str) -> None:
            desc, n = gate_drops[gate]
            gate_drops[gate] = (desc, n + 1)

        with FaceAligner(width, height) as aligner:
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
                    # Pre-filter Apple-sourced frames on cheap signals before
                    # touching the image. Selfie-dir frames (no hint/apple)
                    # bypass these gates since the user vetted them manually.
                    if (
                        min_face_fraction > 0
                        and hint is not None
                        and hint[2] < min_face_fraction
                    ):
                        _bump("tiny_face")
                        manifest_entries.append((
                            dt, path, "dropped",
                            f"face area {hint[2]:.4f} < min-face-fraction {min_face_fraction}",
                        ))
                        progress.update(task, advance=1)
                        continue
                    if (
                        apple is not None
                        and min_face_quality > 0
                        and apple[0] < min_face_quality
                    ):
                        _bump("low_face_quality")
                        manifest_entries.append((
                            dt, path, "dropped",
                            f"face quality {apple[0]:.3f} < min-face-quality {min_face_quality}",
                        ))
                        progress.update(task, advance=1)
                        continue
                    if (
                        apple is not None
                        and apple[1] is not None
                        and max_tastefully_blurred > 0
                        and apple[1] > max_tastefully_blurred
                    ):
                        _bump("tastefully_blurred")
                        manifest_entries.append((
                            dt, path, "dropped",
                            f"tastefully-blurred {apple[1]:.3f} > max {max_tastefully_blurred}",
                        ))
                        progress.update(task, advance=1)
                        continue
                    img = load_image(path)
                    aligned_img = aligner.align(img, face_hint=hint)
                    if aligned_img is not None:
                        prepared.append(aligned_img)
                        kept_labels.append(lbl)
                        kept_dates.append(dt)
                        kept_paths.append(path)
                        manifest_entries.append((dt, path, "kept", ""))
                    else:
                        _bump("no_usable_face")
                        manifest_entries.append((
                            dt, path, "dropped", "no usable face detected",
                        ))
                    progress.update(task, advance=1)

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
                    f"\nWarning: over half of source photos were dropped. "
                    f"To loosen: lower --min-face-fraction / --min-face-quality, "
                    f"raise --max-tastefully-blurred, or check for "
                    f"misframed/wrong-person photos in the source.",
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
        )
        typer.echo(timeline.summary())

        # Validate bounds. Bounds are only enforced in auto mode; if the user
        # set --beat-speed they've explicitly opted out of auto-pick.
        if beat_speed is None and not force:
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

        if beat_crossfade:
            frames_iter, n_out = _beat_crossfade_frames(
                prepared,
                kept_labels,
                kept_paths,
                timeline.segments,
                fps=FPS,
                debug_tier_overlay=debug_tier_overlay,
                debug_filename_overlay=debug_filename_overlay,
                song=music.stem,
                bpm=timeline.bpm,
            )
            typer.echo(
                f"Encoding {len(timeline.segments)} segment(s) as continuous crossfade "
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
            # Render each segment's photo with its label, no crossfade.
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
            f"Done! {output} ({rendered_count} frames, {timeline.total_duration:.1f}s)"
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
