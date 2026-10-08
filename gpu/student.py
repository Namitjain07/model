"""Distil the VideoMAE teacher into a board-sized student: TSM-MobileNetV3-Large, 8 frames, ~2 GFLOPs/clip.

  python gpu/student.py --teacher gpu_out/teacher_A --out gpu_out/student_A
Online knowledge distillation: the SAME augmented clip goes through the frozen teacher (16 frames) and the student (every 2nd frame);
loss = 0.5 * CE(hard label) + 0.5 * T^2 * KL(teacher || student) with T = 2.  Selection by DEV AUC, never the held-out sets.

TSM (Lin et al., ICCV 2019): the first 1/8 of the channels shift one frame forward in time and the next 1/8 one frame back inside every
residual block.  Zero extra FLOPs/params, and it uses only Slice/Concat/Conv, which export cleanly to ONNX and run on the Hexagon NPU.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TemporalShift(nn.Module):
    def __init__(self, n_seg: int, fold_div: int = 8):
        super().__init__(); self.n_seg, self.fold_div = n_seg, fold_div

    def forward(self, x):                                         # x: (B*T, C, H, W)
        nt, c, h, w = x.shape; x = x.view(-1, self.n_seg, c, h, w); f = c // self.fold_div
        fwd = torch.cat([torch.zeros_like(x[:, :1, :f]), x[:, :-1, :f]], 1)               # frame t receives t-1
        bwd = torch.cat([x[:, 1:, f:2 * f], torch.zeros_like(x[:, :1, f:2 * f])], 1)      # frame t receives t+1
        return torch.cat([fwd, bwd, x[:, :, 2 * f:]], 2).view(nt, c, h, w)


class TSMMobileNetV3(nn.Module):
    def __init__(self, n_seg: int = 8, pretrained: bool = True, drop: float = 0.3):
        super().__init__()
        import torchvision
        w = None
        if pretrained:
            try: w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            except Exception: w = None
        try: net = torchvision.models.mobilenet_v3_large(weights=w)
        except Exception as e:                                    # offline smoke test
            print("WARNING: ImageNet weights unavailable (", type(e).__name__, ") -> random init", flush=True); net = torchvision.models.mobilenet_v3_large(weights=None)
        self.n_seg, self.features = n_seg, net.features
        for blk in self.features:                                 # residual blocks only (shift inside the residual branch)
            if getattr(blk, "use_res_connect", False): blk.block = nn.Sequential(TemporalShift(n_seg), blk.block)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(drop), nn.Linear(256, 2))

    def forward(self, x):                                         # (B, T, 3, H, W) -> (B, 2)
        b, t = x.shape[:2]
        f = self.pool(self.features(x.flatten(0, 1))).flatten(1)  # (B*T, 960)
        return self.head(f.view(b, t, -1).mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="gpu_out/teacher_A"); ap.add_argument("--out", default="gpu_out/student_A")
    ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.02); ap.add_argument("--temp", type=float, default=2.0); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-teacher", action="store_true", help="plain CE (ablation)")
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from gpu.common import ROOT, bucket_weights, load_manifest
    from gpu.dataset import ClipDataset, has_frames
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(); df = df[df.clip_id.map(has_frames)].reset_index(drop=True)
    tr, dv = df[df.split == "train"].reset_index(drop=True), df[df.split == "dev"].reset_index(drop=True)
    if a.smoke: tr, dv, a.epochs, a.bs, a.workers = tr.head(8), dv.head(6), 1, 4, 0

    teacher = None
    if not a.no_teacher:
        from transformers import VideoMAEForVideoClassification
        tdir = ROOT / a.teacher
        teacher = VideoMAEForVideoClassification.from_pretrained(tdir / "hf", torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    model = TSMMobileNetV3(8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sampler = WeightedRandomSampler(bucket_weights(tr), num_samples=len(tr), replacement=True)
    dl = DataLoader(ClipDataset(tr, True), batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=len(tr) >= a.bs, pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    steps = max(1, len(dl) * a.epochs); warm = max(1, len(dl)); step = 0
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    @torch.no_grad()
    def predict(d):
        model.eval(); L = []
        for x, y, i in DataLoader(ClipDataset(d, False), batch_size=32, num_workers=a.workers):
            with torch.autocast(**amp): L.append(model(x[:, ::2].to(dev)).float().cpu())
        return torch.cat(L).numpy() if L else np.zeros((0, 2))

    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for b, (x, y, _) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(**amp): s = model(x[:, ::2]).float()
            loss = F.cross_entropy(s, y, label_smoothing=0.05)
            if teacher is not None:
                with torch.no_grad(): tl = teacher(pixel_values=x.to(next(teacher.parameters()).dtype)).logits.float()
                kd = F.kl_div(F.log_softmax(s / a.temp, -1), F.softmax(tl / a.temp, -1), reduction="batchmean") * a.temp ** 2
                loss = (1 - a.alpha) * loss + a.alpha * kd
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups: g["lr"] = a.lr * f
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1; run += loss.item()
            if a.smoke and b >= 1: break
        lg = predict(dv); auc = roc_auc_score(dv.label, lg[:, 1] - lg[:, 0]) if len(dv) > 3 and dv.label.nunique() > 1 else float("nan")
        print(f"epoch {ep+1}/{a.epochs} loss {run/max(1,b+1):.4f} dev AUC {auc:.4f} | {time.time()-t0:.0f}s", flush=True)
        sc = auc if not math.isnan(auc) else ep
        if sc > best: best = sc; torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)); lg = predict(df)
    res = df[["clip_id", "dataset", "split", "label", "group", "bucket", "w", "h", "dur", "affected"]].copy(); res["logit0"], res["logit1"] = lg[:, 0], lg[:, 1]
    res["p_fight"] = 1 / (1 + np.exp(-(lg[:, 1] - lg[:, 0]))); res.to_csv(out / "predictions.csv", index=False)
    json.dump(dict(best_dev_auc=float(best), args=vars(a), params=sum(p.numel() for p in model.parameters())), open(out / "train_summary.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
