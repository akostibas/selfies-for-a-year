# selfies-for-a-year

Turn years of daily selfies into a smooth timelapse video with automatic face alignment. Faces are detected and aligned so your eyes stay at the same position across every frame, producing a stable, professional-looking result.

## Prerequisites

- **macOS** (tested on macOS 15+)
- **[Homebrew](https://brew.sh)**

## Setup

Install system dependencies:

```bash
brew install uv ffmpeg
```

Clone the repo, install, and download the face detection model:

```bash
git clone https://github.com/akostibas/selfies-for-a-year.git
cd selfies-for-a-year
uv sync
curl -L -o models/face_landmarker.task --create-dirs \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

`uv sync` creates an isolated virtual environment and installs all Python dependencies. Your system Python is not modified.

## Usage

```bash
uv run selfies-for-a-year compile [INPUT_DIR ...] --output timelapse.mp4
```

Inputs can be one or more directories of images, an Apple Photos person
(`--apple-photos "Name"`), or both combined. Use `list-people` to see named
people in your Apple Photos library:

> **Apple Photos access:** `--apple-photos` reads the Photos library
> SQLite database directly. macOS will prompt for **Full Disk Access** the
> first time your terminal touches `~/Pictures/Photos Library.photoslibrary`.
> Grant it in System Settings → Privacy & Security → Full Disk Access.

```bash
uv run selfies-for-a-year list-people
```

### Examples

```bash
# Default: 1080x1080 square, 150ms per photo, face-aligned, date label on
uv run selfies-for-a-year compile ~/Photos/2026-selfies --output timelapse.mp4

# Pull from Apple Photos by named person, restrict to a year range
uv run selfies-for-a-year compile \
  --apple-photos "Alexi" --year 2020-2026 --output timelapse.mp4

# Combine on-disk dirs with an Apple Photos source
uv run selfies-for-a-year compile ~/selfies/2020 ~/selfies/2021 \
  --apple-photos "Alexi" --since 2020-01-01 --until 2026-12-31 \
  --output timelapse.mp4

# Disable face alignment (simple center-crop)
uv run selfies-for-a-year compile ~/Photos/2026-selfies --no-align \
  --output timelapse.mp4

# Slower pacing (300ms per photo instead of 150ms)
uv run selfies-for-a-year compile ~/Photos/2026 --duration 300 \
  --output timelapse.mp4

# Portrait output
uv run selfies-for-a-year compile ~/Photos/selfies \
  --width 1080 --height 1920 --output timelapse.mp4
```

### Core options

| Option                  | Default | Description                                          |
|-------------------------|---------|------------------------------------------------------|
| `--output`              | `output.mp4` | Output MP4 path.                                |
| `--apple-photos NAME`   |         | Pull photos of this named person from Apple Photos.  |
| `--year`                |         | Filter all sources to a year or range (`2026`, `2020-2026`). |
| `--since YYYY-MM-DD`    |         | Drop photos dated before this date.                  |
| `--until YYYY-MM-DD`    |         | Drop photos dated after this date.                   |
| `--exclude PATTERNS`    |         | Comma-separated substrings; any photo whose filename contains one is dropped. Useful with `--debug-filename-overlay` for picking weeders by UUID prefix. |
| `--width` / `--height`  | 1080    | Output dimensions in pixels.                         |
| `--duration`            | 150 ms  | Time each photo is shown.                            |
| `--fade`                | `40%`   | Crossfade between photos (`40%` of duration or `60ms`). |
| `--fade-in` / `--fade-out` | 0    | Open/close fade (`5%` of total, or absolute ms).     |
| `--fade-in-color` / `--fade-out-color` | `black` | `black`, `white`, or hex.            |
| `--align` / `--no-align`| on      | Face-align frames vs. simple center-crop.            |
| `--normalize`           | off     | Experimental color/exposure normalization.           |
| `--label` / `--no-label`| on      | Overlay date label on frames.                        |
| `--manifest PATH`       |         | Write a TSV manifest of every photo considered (kept/dropped + reason). |

### Quality gates (Apple Photos)

When sourcing from Apple Photos, frames are filtered using Apple's own
metadata before face-alignment runs:

| Option                       | Default | Description                                                |
|------------------------------|---------|------------------------------------------------------------|
| `--min-face-fraction`        | `0.03`  | Skip frames where Apple's face-area hint is below this fraction of the image. |
| `--min-face-quality`         | `0.18`  | Skip frames where Apple's `ZQUALITY` face-quality score is below this. |
| `--max-tastefully-blurred`   | `0.3`   | Skip frames Apple flagged as "tastefully blurred" above this score. |

## Music

Add a soundtrack with `--music`. Two timing modes:

```bash
# Constant pacing + audio (default mode when --music is set)
uv run selfies-for-a-year compile ~/Photos/2026 --output out.mp4 \
  --music song.mp3

# Stretch per-photo duration to fill the audio length exactly
uv run selfies-for-a-year compile ~/Photos/2026 --output out.mp4 \
  --music song.mp3 --fit-to-music

# Beat-sync: photos transition on detected beats
uv run selfies-for-a-year compile ~/Photos/2026 --output out.mp4 \
  --music song.mp3 --beat-sync
```

`--fit-to-music` and `--beat-sync` are mutually exclusive. Use
`--music-fade-out SECONDS` (default `2.0`) to fade the audio out at the end.

### `--beat-sync` flag interactions

Under `--beat-sync`, photos transition on detected beats. By default these
are hard cuts (`--fade`, `--fade-in`, `--fade-out` are ignored). Add
`--beat-crossfade` to replace the cuts with a continuous crossfade where
each photo reaches peak opacity exactly at its beat and morphs into the
next over the segment's duration — useful for slow-morph aesthetics on
subject matter that changes gradually (e.g. beard growth, weight change,
seasonal lighting). Tier multipliers from `--vary-pace` still apply, so
intense sections morph faster and slow sections linger.

Pace is bounded by two flags:

| Flag                       | Default | Role                                                  |
|----------------------------|---------|-------------------------------------------------------|
| `--max-photos-per-second`  | `4.0`   | Visual ceiling. Auto-picks the largest *subdivision* (photos per beat from {4, 2, 1, ½, ¼}) that stays under this rate. |
| `--min-photos-per-beat`    | `1.0`   | Pace floor. When photos < transitions, the picker backs off to coarser subdivisions but won't drop below this. |
| `--beat-speed`             | *auto*  | Manual subdivision override. Bypasses both bounds.    |
| `--beat-thresh`            | `0.30`  | Onset-strength threshold (0–1). Beats below this are "ambient" and fall back to `--duration` timing. |
| `--beat-crossfade`         | off     | Replace hard cuts with continuous crossfade (peak opacity on the beat). |
| `--force`                  | off     | Render best-effort even when bounds can't be satisfied. Without it, the command exits with what to relax. |

**What happens when photos and beats don't match:**

- **photos > transitions** — drop photos, weighted by month so coverage stays even across the year.
- **photos < transitions** — auto-backoff picks a coarser subdivision until photos can fill the song, capped by `--min-photos-per-beat`.
- **floor reached, photos still too few** — by default the command **exits** with the constraint that needs to be relaxed. Either lower `--min-photos-per-beat` or pass `--force` to render with audio trimmed.

**Quick-tune cheatsheet:**

- *Strobing too fast* → lower `--max-photos-per-second`.
- *Dragging too slow* → raise `--min-photos-per-beat`.
- *Force half-time / double-time* → set `--beat-speed 0.5` or `--beat-speed 2.0`.
- *Intro/outro getting weird timing* → raise `--beat-thresh`.

### `--vary-pace` (three-tier pacing)

Layer `--vary-pace` on top of `--beat-sync` to detect the song's most intense
and most quiet sustained sections and pace photos differently in each. The
result is one base rate for most of the song with a small number of clearly
faster ("intense") and slower ("slow") regions for punctuation.

| Flag                       | Default | Role                                                  |
|----------------------------|---------|-------------------------------------------------------|
| `--intense-multiplier`     | `2.0`   | Rate multiplier for intense sections (2.0 = double pace). |
| `--slow-multiplier`        | `0.5`   | Rate multiplier for quiet sections (0.5 = half pace). |
| `--max-intense`            | `2`     | How many intense sections to pick (ranked by energy). |
| `--max-slow`               | `1`     | How many quiet sections to pick (typically intro/breakdown). |
| `--min-section-seconds`    | `5.0`   | Minimum sustained duration to qualify as a section.   |
| `--min-normal-bridge-beats`| `8.0`   | Tiny "normal" bridges shorter than this get absorbed into the adjacent region. |
| `--snap-to-grid` / `--no-snap-to-grid` | on | Snap multipliers to power-of-2 musical fractions so cuts stay on the 4/4 grid. |
| `--tier-lead-seconds`      | `0.0`   | Shift tier-region detection earlier by N seconds (anticipation). Try 0.3–1.0s. |
| `--debug-tier-overlay`     | off     | Overlay the current pacing tier on each frame for diagnosing pacing. |

## How it works

### Image sorting

Images are sorted chronologically using the best available date:
1. **EXIF `DateTimeOriginal`** (most accurate)
2. **Filename parsing** (supports ISO 8601, `YYYY-MM-DD at HH.MM.SS`, and `YYYY-MM-DD` formats)
3. **File modification time** (last resort)

### Face alignment

When `--align` is enabled (the default), each frame is processed through mediapipe's FaceLandmarker to detect facial landmarks. The positions of both eyes are used to compute a similarity transform (rotation + scale + translation) that maps the face to a canonical position:

- **Eyes at 30%/70% of frame width** — face fills roughly 3/4 of the frame
- **Eyes at 40% from top** — leaves room for hair/forehead with chin visible below

This produces a stable output where the eyes stay at consistent pixel coordinates across all frames despite variations in camera angle, distance, and head tilt.

If the transform would extend beyond the source image boundaries, the out-of-bounds areas are filled with a heavily blurred version of the same image rather than black bars.

If face detection fails on a frame, it falls back to a simple center-crop.

### Video encoding

Frames are streamed directly to ffmpeg (via stdin) to produce an H.264 MP4 with `yuv420p` pixel format for broad compatibility.

## Development

```bash
uv run python -m selfies_for_a_year compile <input-dir> --output output.mp4
```

For debugging visuals, pass `--debug-tier-overlay --debug-filename-overlay`
to annotate every frame with the current pacing tier and the source filename.

See [docs/what-we-tried.md](docs/what-we-tried.md) for approaches we tried and
moved on from, and [docs/render-log.md](docs/render-log.md) for the
chronological log of experimental renders.

## License

[MIT](LICENSE).
