"""Shared manifest + sampling logic for the GPU pipeline.

Manifest = one row per clip: dataset, split, label, group, native resolution bucket.
 splits   : train | dev | heldout
   rwf    : train/val from the official RWF-2000 split; val is HELD OUT (report only); 10% of train videos -> dev (early stopping)
   rlvs   : train (minus near-duplicates of held-out clips, see audit.json); 10% -> dev
   surv   : Surveillance-Fight clips  -> heldout in protocol A, train in final protocol B
   airt   : AIRTLab clips             -> heldout in protocol A, train in final protocol B
 affected : 1 = held-out clip that has a near-duplicate in some training source (gpu/audit.json).  The duplicates are dropped from training
            here, so GPU results are clean for every clip; the flag lets older models (trained before the audit) be compared on the clean subset.
 bucket   : native-resolution class.  Metadata alone predicts the label in these datasets (RLVS AUC 0.95), so training uses
            bucket-balanced sampling weights: within every bucket fight and non-fight carry equal total weight.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, OUT = ROOT / "data", ROOT / "gpu_cache", ROOT / "gpu_out"
MANIFEST = CACHE / "manifest.csv"


def _bucket(w: int, h: int) -> str:
    s = min(w, h)
    return "xs" if s <= 260 else "s" if s <= 380 else "m" if s <= 500 else "l"


def _hash10(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10


def _probe(p: str):
    c = cv2.VideoCapture(p); n = int(c.get(7)); fps = c.get(5) or 25.0; w, h = int(c.get(3)), int(c.get(4)); c.release()
    return w, h, fps, (n / fps if fps else 0.0)


def build_manifest(final: bool = False, exclude_duplicates: bool = True) -> pd.DataFrame:
    dup, aff = set(), set()
    audit = Path(__file__).with_name("audit.json")
    if audit.exists():
        a = json.load(open(audit)); aff = {os.path.normpath(p) for v in a.get("affected_heldout", {}).values() for p in v}
        if exclude_duplicates: dup = {os.path.normpath(p) for v in a.get("near_duplicates", {}).values() for p in v}
    rows = []
    def add(ds, split, label, group, path):
        rel = os.path.normpath(os.path.relpath(path, ROOT))
        if rel in dup: return
        w, h, fps, dur = _probe(str(path))
        if w == 0: return
        rows.append(dict(clip_id=f"{ds}:{Path(path).parent.name}:{Path(path).stem}" if ds != "airt" else f"{ds}:{Path(path).parents[1].name}:{Path(path).parent.name}:{Path(path).stem}",
                         dataset=ds, split=split, label=label, group=group, path=str(path.relative_to(ROOT)), w=w, h=h, fps=fps, dur=dur, bucket=_bucket(w, h), affected=int(rel in aff)))
    for sp in ("train", "val"):
        for p in sorted((DATA / "rwf2000" / sp).glob("*/*.avi")):
            g = "rwf:" + p.stem.rsplit("_", 1)[0]
            split = "heldout" if sp == "val" else ("dev" if _hash10(g) == 0 else "train")
            add("rwf", split, int(p.parent.name == "Fight"), g, p)
    for p in sorted((DATA / "rlvs" / "raw").glob("*/*")):
        g = "rlvs:" + p.stem; add("rlvs", "dev" if _hash10(g) == 0 else "train", int(p.parent.name == "Violence"), g, p)
    src, cur = {}, None
    vt = DATA / "survfight" / "videos.txt"
    if vt.exists():
        import re
        for ln in vt.read_text().splitlines():
            ln = ln.strip()
            if ln.startswith("http"): cur = ln
            elif re.match(r"(no)?fi\d+:", ln): src[ln.split(":")[0]] = cur
    for p in sorted((DATA / "survfight").glob("*/*.mp4")):
        add("surv", "train" if final else "heldout", int(p.parent.name == "fight"), "surv:" + str(src.get(p.stem, p.stem)), p)
    for p in sorted((DATA / "airtlab").glob("*/cam*/*.mp4")):
        kind = p.parents[1].name
        add("airt", "train" if final else "heldout", int(kind == "violent"), f"airt:{kind}:{p.stem}", p)
    df = pd.DataFrame(rows)
    CACHE.mkdir(exist_ok=True, parents=True); df.to_csv(MANIFEST, index=False)
    return df


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def bucket_weights(df: pd.DataFrame, cap: float = 4.0) -> np.ndarray:
    """Sampling weights that make the label uninformative given the resolution bucket (and dataset).

    For every (dataset, bucket) cell, fight and non-fight get equal total weight; cells are weighted by their size so the overall
    mix is unchanged. Individual weights are capped at `cap` x the median so a handful of rare clips cannot dominate.
    """
    w = np.ones(len(df), np.float64)
    for _, g in df.groupby(["dataset", "bucket"]):
        n1, n0 = int((g.label == 1).sum()), int((g.label == 0).sum())
        if n1 == 0 or n0 == 0: continue
        tot = n1 + n0
        w[g.index[g.label == 1]] = tot / (2 * n1); w[g.index[g.label == 0]] = tot / (2 * n0)
    return np.minimum(w, cap * np.median(w))


def summarize(df: pd.DataFrame) -> str:
    t = df.groupby(["dataset", "split", "label"]).size().unstack(fill_value=0)
    return t.to_string()


if __name__ == "__main__":
    import sys
    d = build_manifest(final="--final" in sys.argv)
    print(summarize(d)); print("buckets:", d.groupby("bucket").label.agg(["mean", "size"]).round(2).to_dict("index"))
