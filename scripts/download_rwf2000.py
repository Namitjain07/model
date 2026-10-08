"""RWF-2000 (Cheng, Cai, Li, Li, ICPR 2020): 2,000 real surveillance-style 5-s clips, official split
(train 800 Fight/800 NonFight, val 200/200).  Mirror: huggingface.co/datasets/A1mal/RWF-2000-Dataset (MIT card).
Clip names are <youtube_id>_<slice>.avi -> use the id as the CV group.  Files go to data/rwf2000/ (git-ignored)."""
import json, os, subprocess, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "A1mal/RWF-2000-Dataset"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rwf2000")

def api(path=""):
    u = f"https://huggingface.co/api/datasets/{REPO}/tree/main" + ("/" + urllib.parse.quote(path) if path else "")
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))

def walk(path=""):
    for it in api(path):
        if it["type"] == "directory": yield from walk(it["path"])
        elif it["path"].lower().endswith(".avi"): yield it["path"]

def fetch(p):
    rel = p.replace("data/RWF-2000 Sliced/", "")                       # train/Fight/<name>.avi
    dst = os.path.join(OUT, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000: return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/" + urllib.parse.quote(p)
    r = subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-delay", "2", "-m", "120", "-o", dst + ".part", url])
    if r.returncode == 0: os.replace(dst + ".part", dst); return True
    return False

paths = list(walk()); print(len(paths), "clips listed")
with ThreadPoolExecutor(8) as ex: ok = list(ex.map(fetch, paths))
bad = [p for p, k in zip(paths, ok) if not k]
print(f"{sum(ok)}/{len(paths)} ok", "| FAILED:" if bad else "", *bad[:10])
