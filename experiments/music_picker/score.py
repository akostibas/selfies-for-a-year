#!/usr/bin/env python3
"""Iteration 2: add duration filter + recency boost."""
import csv, math, re
from datetime import datetime

ROWS = []
# expects dump_library.applescript output here
with open("experiments/music_picker/data/songs_full.tsv") as f:
    for parts in csv.reader(f, delimiter="\t"):
        if len(parts) < 11:
            continue
        pc, sc, rt, rk, yr, da, dp, du, ge, ar, nm = parts
        try:
            dur = float(du)
        except ValueError:
            dur = 0
        # Parse "Tuesday, May 28, 2019 at 8:54:37 PM" → datetime
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
            "yr": yr, "ar": ar, "nm": nm, "dur": dur, "ge": ge,
            "added": parse_dt(da), "last": parse_dt(dp),
            "user_rating": int(rt) if rk == "user" else 0,
        })

NOW = datetime(2026, 5, 10)
print(f"loaded {len(ROWS)} rows; {sum(1 for r in ROWS if r['user_rating']>0)} user-rated\n")


def recency_boost(r):
    """0..1 — 1 if played within last 6 months, decays over 5 years to 0."""
    if not r["last"]:
        return 0
    days = (NOW - r["last"]).days
    if days < 180:
        return 1.0
    if days > 365 * 5:
        return 0.0
    return max(0.0, 1.0 - (days - 180) / (365 * 5 - 180))


def base_score(r):
    """Best so far: rating + 20*log(plays) - skip penalty."""
    return r["user_rating"] + 20 * math.log1p(r["pc"]) - 10 * r["sc"]


def fmt_row(i, r, score):
    rec = recency_boost(r)
    last = r["last"].strftime("%Y-%m") if r["last"] else "    -- "
    return (f"{i:2}. {score:6.1f} pc={r['pc']:3} rt={r['user_rating']:3} "
            f"dur={r['dur']/60:4.1f}m rec={rec:.2f} last={last}  "
            f"{r['ar'][:22]:22} — {r['nm'][:40]}")


def show(name, scorefn, filterfn=lambda r: True, cap=2):
    pool = [r for r in ROWS if filterfn(r)]
    ranked = sorted(pool, key=scorefn, reverse=True)
    out = []
    counts = {}
    for r in ranked:
        if counts.get(r["ar"], 0) >= cap:
            continue
        out.append(r)
        counts[r["ar"]] = counts.get(r["ar"], 0) + 1
        if len(out) >= 20:
            break
    print(f"=== {name} ===")
    for i, r in enumerate(out, 1):
        print(fmt_row(i, r, scorefn(r)))
    print()


# H: Add duration filter (60s..480s = 1..8 min — typical song)
show("H: G + dur 1-8min",
     base_score,
     filterfn=lambda r: 60 <= r["dur"] <= 480)

# I: G + recency multiplier
show("I: base * (0.4 + 0.6*recency), dur 1-8min",
     lambda r: base_score(r) * (0.4 + 0.6 * recency_boost(r)),
     filterfn=lambda r: 60 <= r["dur"] <= 480)

# J: Same but require last-played within 5 years
show("J: same, require last-played",
     lambda r: base_score(r) * (0.4 + 0.6 * recency_boost(r)),
     filterfn=lambda r: 60 <= r["dur"] <= 480 and r["last"] is not None
                        and (NOW - r["last"]).days < 365*5)

# M: slideshow-bounded — both song year AND last_played must fall in [start,end]
SLIDESHOW = (1981, 2026)

def in_slideshow_year(r):
    try:
        y = int(r["yr"])
    except (ValueError, TypeError):
        return True  # unknown year — keep
    return SLIDESHOW[0] <= y <= SLIDESHOW[1]

def in_slideshow_last_played(r):
    if r["last"] is None:
        return False
    return SLIDESHOW[0] <= r["last"].year <= SLIDESHOW[1]

# Inside the slideshow span, plays are equally valid no matter how long ago.
# So we drop the recency multiplier entirely.
show("M: slideshow-bounded, no recency penalty",
     base_score,
     filterfn=lambda r: 60 <= r["dur"] <= 480
                        and in_slideshow_year(r)
                        and in_slideshow_last_played(r))


# L: J + bound song year to photo range (1981-2026)
def year_in_range(r, lo=1981, hi=2026):
    try:
        y = int(r["yr"])
    except (ValueError, TypeError):
        return False  # unknown year — drop
    return lo <= y <= hi

show("L: J + year in [1981,2026]",
     lambda r: base_score(r) * (0.4 + 0.6 * recency_boost(r)),
     filterfn=lambda r: 60 <= r["dur"] <= 480 and r["last"] is not None
                        and (NOW - r["last"]).days < 365*5
                        and year_in_range(r))

# What got dropped?
dropped = [r for r in ROWS
           if 60 <= r["dur"] <= 480 and r["last"] is not None
           and (NOW - r["last"]).days < 365*5
           and not year_in_range(r)]
print(f"=== dropped by year filter ({len(dropped)} tracks) ===")
for r in sorted(dropped, key=lambda r: base_score(r) * (0.4 + 0.6 * recency_boost(r)), reverse=True)[:15]:
    score = base_score(r) * (0.4 + 0.6 * recency_boost(r))
    print(fmt_row(0, r, score) + f"  yr={r['yr']!r}")
print()

# K: Even harsher recency — multiply, no floor
show("K: base * recency (recency=0 → drop)",
     lambda r: base_score(r) * recency_boost(r),
     filterfn=lambda r: 60 <= r["dur"] <= 480)
