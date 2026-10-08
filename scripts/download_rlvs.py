"""Real Life Violence Situations (RLVS): 1,000 Violence + 1,000 NonViolence YouTube clips (HF mirror 34data/real-life-violence).
No licence is stated on the mirror -> research use only, provenance unverified.  Extracted to data/rlvs/raw/{Violence,NonViolence}/.
AUDIT NOTE: metadata alone (resolution/fps/duration) predicts the label with AUC 0.95, and 11 clips are near-duplicates of
held-out clips (scripts/audit_datasets.py) -> train with bucket-balanced sampling and drop the flagged duplicates.
"""
import os, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rlvs"); os.makedirs(OUT, exist_ok=True)
URL = "https://huggingface.co/datasets/34data/real-life-violence/resolve/main/real-life-violence_{:04d}.zip"

def get(i):
    dst = os.path.join(OUT, f"part{i}.zip")
    if not os.path.exists(dst) or os.path.getsize(dst) < 1e6:
        subprocess.run(["curl", "-fsSL", "--retry", "4", "-m", "1200", "-o", dst + ".part", URL.format(i)], check=True); os.replace(dst + ".part", dst)
    return dst

if os.path.isdir(os.path.join(OUT, "raw", "Violence")) and len(os.listdir(os.path.join(OUT, "raw", "Violence"))) >= 900:
    print("RLVS already extracted")
else:
    with ThreadPoolExecutor(4) as ex: parts = list(ex.map(get, (1, 2, 3, 4)))
    for z in parts: zipfile.ZipFile(z).extractall(os.path.join(OUT, "raw")); os.remove(z)
print({k: len(os.listdir(os.path.join(OUT, "raw", k))) for k in ("Violence", "NonViolence")})
