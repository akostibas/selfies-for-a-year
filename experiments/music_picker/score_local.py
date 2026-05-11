#!/usr/bin/env python3
"""Score local-file tracks; flag mix/live/DRM."""
import csv, math, re
from datetime import datetime

ROWS = []
# expects dump_library_with_paths.applescript output here
with open("experiments/music_picker/data/songs_loc.tsv") as f:
    for parts in csv.reader(f, delimiter="\t"):
        if len(parts) < 10:
            continue
        pc, sc, rt, rk, yr, dp, du, ar, nm, path = parts
        try:
            dur = float(du)
        except ValueError:
            dur = 0
        def parse_dt(s):
            if not s or s == "missing value":
                return None
            s = re.sub(r"^[A-Z][a-z]+, ", "", s)
            try:
                return datetime.strptime(s, "%B %d, %Y at %I:%M:%S %p")
            except ValueError:
                return None
        ROWS.append({
            "pc": int(pc), "sc": int(sc), "rt": int(rt), "rk": rk,
            "yr": yr, "ar": ar, "nm": nm, "dur": dur, "path": path,
            "last": parse_dt(dp),
            "user_rating": int(rt) if rk == "user" else 0,
        })

print(f"loaded {len(ROWS)} local tracks; {sum(1 for r in ROWS if r['user_rating']>0)} user-rated")

MIX_PATTERNS = [
    r"\(Continuous Mix\)", r"\(Mix\)", r"\(DJ Mix\)", r"Live from",
    r"Live at", r"Live Set", r"DJ-Kicks", r"Essential Mix",
    r"Boiler Room", r"LateNightTales", r"Continuous DJ",
]
def looks_mixed(r):
    blob = r["nm"] + " | " + r["path"]
    for p in MIX_PATTERNS:
        if re.search(p, blob, re.I):
            return True
    # heuristic: > 8 min on a "Compilations" path is usually a mix
    if r["dur"] > 480 and "/Compilations/" in r["path"]:
        return True
    return False

def is_drm(r):
    return r["path"].endswith(".m4p")

def base_score(r):
    return r["user_rating"] + 20 * math.log1p(r["pc"]) - 10 * r["sc"]

local_playable = [r for r in ROWS if not is_drm(r)]
local_clean = [r for r in local_playable if not looks_mixed(r) and 60 <= r["dur"] <= 480]
print(f"  drop DRM: {len(ROWS)-len(local_playable)}; drop mix/long: {len(local_playable)-len(local_clean)}")

ranked = sorted(local_clean, key=base_score, reverse=True)
print("\n=== Top 20: local, playable, non-mix, 1-8 min ===")
for i, r in enumerate(ranked[:20], 1):
    last = r["last"].strftime("%Y-%m") if r["last"] else "    -- "
    print(f"{i:2}. {base_score(r):6.1f}  pc={r['pc']:3} rt={r['user_rating']:3} dur={r['dur']/60:4.1f}m  last={last}  {r['ar'][:24]:24} — {r['nm'][:40]}")
    print(f"      → {r['path']}")
