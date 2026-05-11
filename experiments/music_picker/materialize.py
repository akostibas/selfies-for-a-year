#!/usr/bin/env python3
"""Browse and materialize tracks from Apple Music.app via AppleScript.

Usage:
    python music_lib.py refresh                 # dump library TSV to /tmp/music_library.tsv
    python music_lib.py find <query> [--max S]  # search by artist/title/album (case-insensitive)
    python music_lib.py download <pid>          # download a cloud track, print local path
    python music_lib.py path <pid>              # print local path (no download attempt)

The TSV columns: pid  cloud  dur  artist  album  name  location

`location` is empty when the track is cloud-only. `download` triggers Music.app's
AppleScript download command and polls until `location` is populated (or times out).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TSV = Path("/tmp/music_library.tsv")

DUMP_JXA = r"""
function run() {
    var Music = Application("Music");
    Music.includeStandardAdditions = false;
    var tracks = Music.tracks();
    var out = ["pid\tcloud\tdur\tartist\talbum\tname\tlocation"];
    for (var i = 0; i < tracks.length; i++) {
        var t = tracks[i];
        try {
            var loc = "";
            try { var f = t.location(); if (f) loc = f.toString(); } catch (e) {}
            var pid = ""; try { pid = t.persistentID(); } catch (e) {}
            var cloud = ""; try { cloud = t.cloudStatus(); } catch (e) {}
            var dur = ""; try { dur = t.duration(); } catch (e) {}
            var artist = ""; try { artist = (t.artist() || "").replace(/\t/g, " "); } catch (e) {}
            var album = "";  try { album  = (t.album()  || "").replace(/\t/g, " "); } catch (e) {}
            var name = "";   try { name   = (t.name()   || "").replace(/\t/g, " "); } catch (e) {}
            out.push([pid, cloud, dur, artist, album, name, loc].join("\t"));
        } catch (e) {
            out.push(["ERR", e.toString(), "", "", "", "", ""].join("\t"));
        }
    }
    return out.join("\n");
}
"""

DOWNLOAD_JXA = r"""
function run(argv) {
    var pid = argv[0];
    var Music = Application("Music");
    var ts = Music.tracks.whose({persistentID: pid})();
    if (ts.length === 0) return "ERR not_found";
    try { Music.download(ts[0]); return "OK"; }
    catch (e) { return "ERR " + e.toString(); }
}
"""

PATH_JXA = r"""
function run(argv) {
    var pid = argv[0];
    var Music = Application("Music");
    var ts = Music.tracks.whose({persistentID: pid})();
    if (ts.length === 0) return "";
    try { var f = ts[0].location(); return f ? f.toString() : ""; }
    catch (e) { return ""; }
}
"""


def jxa(script: str, *args: str) -> str:
    r = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script, *args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"osascript failed: {r.stderr.strip()}")
    return r.stdout.rstrip("\n")


def cmd_refresh() -> None:
    out = jxa(DUMP_JXA)
    TSV.write_text(out + "\n")
    n = out.count("\n")  # header + rows - 1 newline
    print(f"Wrote {TSV} ({n} tracks)")


def _file_url_to_path(loc: str) -> str:
    if loc.startswith("file://"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(loc).path)
    return loc


def cmd_find(query: str, max_results: int) -> None:
    if not TSV.exists():
        cmd_refresh()
    q = query.lower()
    rows = []
    for line in TSV.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        pid, cloud, dur, artist, album, name, loc = parts[:7]
        hay = f"{artist} {album} {name}".lower()
        if q in hay:
            rows.append((float(dur or 0), pid, cloud, artist, album, name, loc))
    rows.sort()
    for dur, pid, cloud, artist, album, name, loc in rows[:max_results]:
        tag = "LOCAL" if loc else "CLOUD"
        print(f"{pid}  {dur:5.0f}s  {tag:5}  {artist[:25]:<25}  {name[:35]:<35}  {album[:30]}")
    if len(rows) > max_results:
        print(f"... ({len(rows) - max_results} more; pass --max)")


def cmd_download(pid: str, timeout: float = 60.0) -> None:
    result = jxa(DOWNLOAD_JXA, pid)
    if result.startswith("ERR"):
        raise SystemExit(result)
    deadline = time.time() + timeout
    while time.time() < deadline:
        loc = jxa(PATH_JXA, pid)
        if loc:
            print(_file_url_to_path(loc))
            return
        time.sleep(1.0)
    raise SystemExit(f"timeout: track {pid} did not materialize within {timeout}s")


def cmd_path(pid: str) -> None:
    loc = jxa(PATH_JXA, pid)
    if not loc:
        raise SystemExit(f"no local file for {pid} (cloud-only — use `download`)")
    print(_file_url_to_path(loc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh")
    sf = sub.add_parser("find"); sf.add_argument("query"); sf.add_argument("--max", type=int, default=30)
    sd = sub.add_parser("download"); sd.add_argument("pid"); sd.add_argument("--timeout", type=float, default=60.0)
    sp = sub.add_parser("path"); sp.add_argument("pid")
    args = p.parse_args()
    if args.cmd == "refresh": cmd_refresh()
    elif args.cmd == "find": cmd_find(args.query, args.max)
    elif args.cmd == "download": cmd_download(args.pid, args.timeout)
    elif args.cmd == "path": cmd_path(args.pid)


if __name__ == "__main__":
    main()
