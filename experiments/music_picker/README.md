# music_picker (experiment)

Exploring how to pick a slideshow soundtrack automatically from the user's
Music.app library, using play count, user rating, skip count, and recency
as signals for "songs they actually like that fit the slideshow's year span."

This is a sketch — not wired into the CLI yet.

## Workflow

Two steps: dump the library to TSV via AppleScript, then score it in Python.

### 1. Dump

Two dumpers — pick one:

- `dump_library.applescript` — bulk-pulls played count, skipped count, rating,
  rating kind, year, date added, last played, duration, genre, artist, name.
  One Apple Event per property column, so it's fast (~6s for 2600 tracks).
  Output: `data/songs_full.tsv`.
- `dump_library_with_paths.applescript` — same columns plus the per-track
  POSIX path to the audio file (skips streaming-only tracks). Slow (~2 min)
  because `location` doesn't bulk-fetch and we have to loop per-track.
  Output: `data/songs_loc.tsv`.

Run from the project root, e.g.:

```
osascript experiments/music_picker/dump_library.applescript > experiments/music_picker/data/songs_full.tsv
```

(`data/` is gitignored — create it locally.)

### 2. Score

- `score.py` reads `data/songs_full.tsv`. Iterates several scoring formulas
  (A–M); the keeper is **M**: `user_rating + 20*log(1+plays) - 10*skips`,
  filtered to user-rated tracks, 1–8 min duration, last-played within the
  slideshow year range, song year within the slideshow year range, deduped
  to max 2 tracks per artist.
- `score_local.py` reads `data/songs_loc.tsv`, applies the M formula to
  local-file tracks only, plus a `looks_mixed` heuristic (regexes for
  "(Continuous Mix)", "Live from", "DJ-Kicks", etc.) and a DRM filter
  (drops `.m4p` Apple Music streaming downloads).

### 3. Materialize

`materialize.py` closes the streaming-only gap: it dumps the full library
(including cloud-only tracks) and can trigger Music.app to download a
specific track by persistent ID, polling until the local path appears.

```
# Dump library to /tmp/music_library.tsv
python experiments/music_picker/materialize.py refresh

# Search by artist/title/album substring
python experiments/music_picker/materialize.py find "RJD2"

# Download a cloud-only track and print local path
python experiments/music_picker/materialize.py download <pid>
```

Built during a beard-morph session (2026-05-11) where we needed to reach
into 2366 cloud-only tracks. AppleScript's `download` verb on a track
takes ~5–10s to materialize a typical song.

## Caveat

Streaming-only Apple Music tracks have no on-disk audio file and can't be
used as soundtrack source until they're downloaded locally in Music.app.
`materialize.py` automates that.
