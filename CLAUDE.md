# selfies-for-a-year

## Workflow

- **Commit on each feature tweak the user likes.** Don't batch up changes — commit as soon as a feature or adjustment is confirmed working and approved.
- Test changes against the 2026 selfies directory before presenting to the user.
- Output test videos to `/tmp/` for review.
- **Always include both debug overlays on non-production renders.** Pass `--debug-tier-overlay --debug-filename-overlay` whenever you render for review or iteration; only omit them when the user explicitly asks for a clean (production) render. The overlays show the current pacing tier (slow/normal/intense/ambient + photos-per-sec) and the source filename, which is essential context for diagnosing pacing or source-photo issues without re-rendering.
- **Append a row to `docs/render-log.md` for every non-production render.** One row per render, pipe-separated table format (see header in the file for column definitions). Append immediately when the render finishes, and update the `feedback` column the moment the user reacts. The log is a grep target across hundreds of past iterations — keep each row terse.
- **Local paths:** See `local.env` (gitignored) for machine-specific test paths. Key vars:
  - `SELFIES_BASE` — parent dir with per-year folders (2020/, 2021/, …, 2026/)
  - `SELFIES_2026` — single year for quick tests
  - For full runs, pass all year dirs as input alongside `--apple-photos`

## Guiding star: favor false negatives

When choosing thresholds or behavior in the face-detection / alignment pipeline, **prefer dropping uncertain photos over including malaligned or low-quality ones.** A timelapse with fewer frames where every frame is right beats a longer video peppered with bad frames. When in doubt: be conservative, skip the frame, and let the user notice the gap rather than the bad frame.

## Project

- Python CLI tool using uv, targeting Python 3.14
- mediapipe FaceLandmarker for face detection
- ffmpeg via subprocess for video encoding
- Frames streamed to ffmpeg stdin (no temp files)

## Debugging video output

After generating a test video, visually verify it before presenting to the user. Extract small thumbnails of the region of interest to inspect without blowing tokens:

```bash
# Extract every Nth frame, crop to the area you care about, scale down
ffmpeg -i /tmp/test.mp4 -vf "select='not(mod(n\,30))',crop=iw:100:0:ih-20,scale=360:33" -vsync vfr /tmp/thumbs/frame_%03d.png -y

# Stack into a single strip for comparison
ffmpeg -i /tmp/thumbs/frame_%03d.png -vf "tile=1x13" /tmp/strip.png -y
```

Then read the strip image to check alignment, positioning, or other visual properties across frames.

When the user is actively debugging visuals (probe outputs, alignment frames, thumbnail strips), `open <path>` the resulting images automatically rather than just printing the path. Don't make the user copy-paste paths to view what you produced.

**But only open images the user is meant to look at.** If an image was generated purely for my own verification (sanity-checking a fix, confirming a pipeline still works on known-good inputs), don't `open` it — read it with the Read tool instead. Opening clutters the user's screen with things they didn't ask to see.

## Tracing flagged frames back to source photos

When the user reviews a rendered video and flags problem moments by date (e.g. "2022 April face is cut off"), the source-photo identification is a two-step workflow:

1. **List candidates.** For each flagged month, query Apple Photos, apply `--min-face-fraction`, run the aligner, and report the original filename of every photo that survived. If only one survives, that's the offender.
2. **Disambiguate.** If multiple kept photos for the month, build a contact sheet of the *aligned outputs* (what the user actually saw in the video) labeled with the date and a truncated filename under each tile. Open it and let the user point by saying "2022-04-24 — 94A3319F is the cutoff one." Truncating the filename to the first 28 chars of the UUID stem is enough to disambiguate while staying readable.

Patterns to copy from for both steps live in `/tmp/identify_flagged.py` and `/tmp/contact_sheets.py` after the latest run.

After identification, **group by issue type** (wrong person / cut off / too small / blurry / etc.) and file one GitHub issue per category, listing the photos as cases. Don't file one issue per photo — categories are how we'll fix them in batches. Include a hypothesis on the cause and 2–3 possible directions in each issue so the next pass starts with context.

## GitHub issues are the session-crossing memory

General rule (persist findings durably, cite inputs by stable path) lives in
`~/.claude/agent-development.md`. The concrete venue here is GitHub Issues, and
the project-specific deltas are:

- **Every class of mistake gets an issue.** When we discover a new failure mode
  (wrong person / cut off / upside-down / too small / blurry / etc.), either
  file a new issue for that class or append a case to the existing class issue.
  Don't fix silently.
- **Source photo paths look like** `~/Pictures/Photos Library.photoslibrary/originals/E/EA71...heic` —
  the Apple Photos originals tree, so the next session can re-open the exact file.
- **Never paste photos into GitHub issues.** This repo is public — issue
  comments are public too. Point at the local path instead. Text, numbers,
  and annotated diagnostic output (verdicts, distances, yaws, EXIF values)
  are fine to paste.
- When a fix is proposed, link the file:line it will touch so the next pass
  knows where to look.

## Open issues

The authoritative list is `gh issue list`. Don't maintain a duplicate here — it goes stale fast.
