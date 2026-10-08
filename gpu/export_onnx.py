"""Export the teacher (VideoMAE, 16x224x224) and student (TSM-MobileNetV3, 8x224x224) to ONNX and check ORT-vs-PyTorch parity.

  python gpu/export_onnx.py --teacher gpu_out/teacher_A --student gpu_out/student_A --out gpu_out/onnx
Next step for the QCS6490 board (needs a free Qualcomm AI Hub account):
  import qai_hub as hub;  dev = [d for d in hub.get_devices() if "RB3" in d.name or "6490" in d.name.lower()][0]
  hub.submit_compile_job(model="student.onnx", device=dev, input_specs={"video": ((1, 8, 3, 224, 224), "float32")}, options="--quantize_full_type w8a8 ...")
Quantisation needs a calibration set (a few hundred preprocessed clips from gpu_cache/frames).  Always re-measure accuracy after quantising.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu.common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=None); ap.add_argument("--student", default=None); ap.add_argument("--out", default="gpu_out/onnx"); ap.add_argument("--opset", type=int, default=18)
a = ap.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
import onnxruntime as ort


def export(m, x, path, opset):
    """Single-file ONNX on any torch >= 2.0: try the new exporter without external data, then the legacy one."""
    kw = dict(input_names=["video"], output_names=["logits"], opset_version=opset)
    for extra in (dict(dynamo=True, external_data=False), dict(dynamo=False), {}):
        try:
            for f in (path, Path(str(path) + ".data")): f.unlink(missing_ok=True)
            torch.onnx.export(m, (x,), str(path), **kw, **extra); return
        except Exception as e:
            print(f"  export with {extra or 'defaults'} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
    raise RuntimeError("all ONNX export routes failed")


def parity(name, m, x, path):
    m.eval()
    with torch.no_grad(): ref = m(x).float().numpy()
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]); got = s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]
    err = float(np.abs(ref - got).max()); print(f"{name}: {path.name} {path.stat().st_size/1e6:.1f} MB | max |torch - onnxruntime| = {err:.2e}", flush=True)
    assert err < 5e-2, f"{name} ONNX parity failed ({err})"


if a.student:
    from gpu.student import TSMMobileNetV3
    m = TSMMobileNetV3(8, pretrained=False); m.load_state_dict(torch.load(ROOT / a.student / "best.pt", map_location="cpu")); m.eval()
    x = torch.randn(1, 8, 3, 224, 224); p = out / "student_tsm_mnv3_8f.onnx"
    export(m, x, p, a.opset); parity("student", m, x, p)

if a.teacher:
    from transformers import VideoMAEForVideoClassification

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(pixel_values=x).logits
    m = Wrap(VideoMAEForVideoClassification.from_pretrained(ROOT / a.teacher / "hf", torch_dtype=torch.float32)).eval()
    x = torch.randn(1, 16, 3, 224, 224); p = out / "teacher_videomae_16f.onnx"
    export(m, x, p, a.opset); parity("teacher", m, x, p)
print("done")
