"""GMDCSA-24 fall-detection dataset (4 subjects, home setups, 720p laptop camera; MIT license).
Alam et al., "GMDCSA24: A Dataset for Human Fall Detection in Videos", Data in Brief (2024),
doi:10.5281/zenodo.12921216.  Per-clip labels incl. time ranges are in the CSVs.  Output: data/gmdcsa24/ (git-ignored).
"""
import csv, io, os, subprocess, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor

BASE = "https://raw.githubusercontent.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/master"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gmdcsa24")

def curl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    r = subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-delay", "2", "-m", "180", "-o", dest + ".part", url])
    if r.returncode == 0:
        os.replace(dest + ".part", dest); return True
    return False

jobs = []
for s in (1, 2, 3, 4):
    for kind in ("Fall", "ADL"):
        rel = f"Subject {s}/{kind}.csv"
        dest = os.path.join(OUT, f"Subject {s}", f"{kind}.csv")
        assert curl(f"{BASE}/{urllib.parse.quote(rel)}", dest), rel
        for row in csv.DictReader(open(dest, newline="", encoding="utf-8-sig")):
            name = (row.get("File Name") or "").strip()
            if name.endswith(".mp4"):
                r = f"Subject {s}/{kind}/{name}"
                jobs.append((f"{BASE}/{urllib.parse.quote(r)}", os.path.join(OUT, f"Subject {s}", kind, name)))
with ThreadPoolExecutor(6) as ex:
    ok = list(ex.map(lambda j: curl(*j), jobs))
bad = [j[1] for j, k in zip(jobs, ok) if not k]
print(f"{sum(ok)}/{len(jobs)} clips ok", "| FAILED:" if bad else "", *bad[:20])
