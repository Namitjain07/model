"""Fine-tune a Kinetics-pretrained VideoMAE-base on the preprocessed person-centric clips (the 'teacher').

  python gpu/train_videomae.py --out gpu_out/teacher_A            # protocol A: surv/airt/rwf-val stay held out
  python gpu/train_videomae.py --out gpu_out/teacher_B --final    # protocol B: also trains on surv/airt (no clean test left)
Selection uses the DEV split carved out of the training sources, never the held-out sets.
Writes best.pt, hf/ (fp16 safetensors), predictions.csv (logits + p_fight for every preprocessed clip).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import ROOT, bucket_weights, load_manifest
from gpu.dataset import ClipDataset, has_frames

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="gpu_out/teacher"); ap.add_argument("--model", default="MCG-NJU/videomae-base-finetuned-kinetics")
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--ld", type=float, default=0.75); ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev_t = "cuda" if torch.cuda.is_available() else "cpu"
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

df = load_manifest(); df = df[df.clip_id.map(has_frames)].reset_index(drop=True)
tr, dv = df[df.split == "train"].reset_index(drop=True), df[df.split == "dev"].reset_index(drop=True)
if a.smoke: tr, dv, a.epochs, a.bs, a.accum, a.workers = tr.head(8), dv.head(6), 1, 2, 1, 0
print(f"train {len(tr)} dev {len(dv)} | device {dev_t} | buckets (label mean):", tr.groupby('bucket').label.mean().round(2).to_dict(), flush=True)

from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained(a.model, num_labels=2, ignore_mismatched_sizes=True,
                                                       id2label={0: "normal", 1: "violence"}, label2id={"normal": 0, "violence": 1}).to(dev_t)
model.gradient_checkpointing_enable()
NL = model.config.num_hidden_layers


def lid(n):
    if "embeddings" in n: return 0
    if ".encoder.layer." in n: return int(n.split(".encoder.layer.")[1].split(".")[0]) + 1
    return NL + 1
groups = {}
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    l = lid(n); head = n.startswith("classifier")
    scale = (a.ld ** (NL + 1 - l)) * (10.0 if head else 1.0); nd = p.ndim == 1 or n.endswith(".bias")
    groups.setdefault((scale, nd), []).append(p)
opt = torch.optim.AdamW([dict(params=ps, lr=a.lr * s, weight_decay=0.0 if nd else a.wd, base=a.lr * s) for (s, nd), ps in groups.items()], betas=(0.9, 0.999))

w = bucket_weights(tr)
sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
dl = DataLoader(ClipDataset(tr, True), batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=len(tr) >= a.bs, pin_memory=dev_t == "cuda", persistent_workers=a.workers > 0)
steps = max(1, math.ceil(len(dl) / a.accum) * a.epochs); warm = max(1, steps // a.epochs)
amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev_t == "cuda" else dict(device_type="cpu", enabled=False)


@torch.no_grad()
def predict(d):
    model.eval(); L = []
    for x, y, i in DataLoader(ClipDataset(d, False), batch_size=max(a.bs, 4), num_workers=a.workers):
        with torch.autocast(**amp): L.append(model(pixel_values=x.to(dev_t)).logits.float().cpu())
    return torch.cat(L).numpy() if L else np.zeros((0, 2))


def dev_auc():
    if len(dv) < 4 or dv.label.nunique() < 2: return float("nan")
    lg = predict(dv); return roc_auc_score(dv.label, lg[:, 1] - lg[:, 0])


best, step, t0 = -1, 0, time.time()
for ep in range(a.epochs):
    model.train(); run = 0.0
    for b, (x, y, _) in enumerate(dl):
        with torch.autocast(**amp): lg = model(pixel_values=x.to(dev_t)).logits
        loss = F.cross_entropy(lg.float(), y.to(dev_t), label_smoothing=0.1) / a.accum; loss.backward(); run += loss.item() * a.accum
        if (b + 1) % a.accum == 0 or b + 1 == len(dl):
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups: g["lr"] = g["base"] * f
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if a.smoke and b >= 1: break
    auc = dev_auc(); print(f"epoch {ep+1}/{a.epochs} loss {run/max(1,b+1):.4f} dev AUC {auc:.4f} | {time.time()-t0:.0f}s", flush=True)
    sc = auc if not math.isnan(auc) else ep
    if sc > best: best = sc; torch.save(model.state_dict(), out / "best.pt")
model.load_state_dict(torch.load(out / "best.pt", map_location=dev_t))
lg = predict(df)
res = df[["clip_id", "dataset", "split", "label", "group", "bucket", "w", "h", "dur", "affected"]].copy(); res["logit0"], res["logit1"] = lg[:, 0], lg[:, 1]
res["p_fight"] = 1 / (1 + np.exp(-(lg[:, 1] - lg[:, 0]))); res.to_csv(out / "predictions.csv", index=False)
model.half().save_pretrained(out / "hf", safe_serialization=True)
json.dump(dict(best_dev_auc=float(best), args=vars(a), n_train=len(tr), n_dev=len(dv)), open(out / "train_summary.json", "w"), indent=1)
print("saved", out)
