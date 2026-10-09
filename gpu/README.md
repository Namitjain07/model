# GPU run: fine-tune a video model on person-centric clips, distil it for the QCS6490

Why: the CPU-side pose + optical-flow model tops out around **RWF-2000 val AUC 0.87** and needs tens of false alarms per hour for useful
recall. A Kinetics-pretrained video transformer, fine-tuned on person-centric crops, is the standard way past that. It is far too heavy to run
always-on on the board, so it is used as (a) a **teacher** and (b) a **second-stage verifier** that only looks at windows the cheap pose + flow stage
flagged, and its knowledge is **distilled into a small TSM-MobileNetV3 student** that can run on the NPU.

```
frame ─► YOLOX-M person boxes ─► light pose (RTMPose/MediaPipe) + flow GBM   (always on, cheap, high recall)
                                        │ candidate window
                                        ▼
                          student TSM-MobileNetV3, 8 frames   (this run)   ─► optional teacher VideoMAE on the host / sparse windows
```

## What the run does
| stage | script | what |
|---|---|---|
| manifest | `gpu/common.py` | one row per clip (dataset / split / label / source-video group / resolution bucket). Drops the 72 near-duplicate training clips found by the audit, flags the 52 affected held-out clips. |
| preprocess | `gpu/preprocess.py` | 5 s window → 16 frames ×2 phases → **YOLOX-M** person boxes → person-centric square crop → 224×224 JPEG. Wide CCTV shots become person-sized. |
| teacher | `gpu/train_videomae.py` | `MCG-NJU/videomae-base-finetuned-kinetics`, new 2-class head, layer-wise LR decay, bf16, label smoothing, bucket-balanced sampling, early stopping on **dev** AUC. |
| student | `gpu/student.py` | TSM-MobileNetV3-Large (ImageNet init), online distillation from the teacher (T=2, α=0.5). Also an ablation without the teacher. |
| evaluate | `gpu/evaluate.py` | threshold picked on **dev only** at ≤5 % / ≤2 % false-alarm rate, then applied unchanged to the held-out sets; AUC inside resolution buckets (shortcut check); false alarms per analysed hour. |
| export | `gpu/export_onnx.py` | single-file ONNX for teacher (16×224²) and student (8×224²) + ONNX-Runtime parity check. |
| push | `gpu/push_results.sh` | commits metrics / predictions / ONNX to `Namitjain07/model` under `results/<run>/` (git-lfs for files > 50 MB; the teacher is ~0.5 GB, so mind the 1 GB free LFS quota — set `NO_TEACHER=1` to skip it). |

### Data (all downloaded by `run_all.sh`)
RWF-2000 (2,000 clips, MIT-labelled mirror), **RLVS (2,000 clips, no licence stated → research only)**, Surveillance-Fight (300 clips, MIT), AIRTLab (350 staged clips, research/education).
Held-out in protocol A (never used for training, thresholds or early stopping): RWF-2000 **val**, Surveillance-Fight, AIRTLab.
Groups = source video, so clips cut from one video never straddle train/dev.

### Known dataset problems the pipeline already accounts for
* **Metadata shortcut.** Resolution / fps / duration alone predict the label (RLVS AUC 0.95, RWF val 0.79, Surv-Fight 0.73). Countermeasures: bucket-balanced sampling (fight and non-fight carry equal weight inside every resolution bucket), random down-up resolution degradation, blur and noise augmentation, and the per-bucket AUC in `evaluate.py`. If bucket AUC ≈ 0.5 while overall AUC is high, the model learned the shortcut — don't trust it.
* **Near-duplicates across datasets** (YouTube re-uploads: 61 RWF-train + 11 RLVS clips duplicate held-out clips). Removed from training.
* **Staged vs real violence.** AIRTLab is acted in a room; Surveillance-Fight / RWF are real CCTV/phone footage. Numbers are reported per dataset, never pooled into one headline.

## Renting the machine (vast.ai)
* 1 × GPU with **≥ 24 GB** (RTX 4090 / A5000 / L4-24G / 3090), **≥ 8 vCPU**, 32 GB RAM, **80 GB disk**, image `pytorch/pytorch:2.x-cuda12.x-cudnn…` (or any "PyTorch" template), on-demand (not interruptible) is safer.
* Rough estimate (not measured): preprocessing 20–60 min (CPU-bound, decoding 1080p videos is the slow part), teacher 6 epochs ≈ 0.5–1.5 h on a 4090 (a 3090 is roughly 1.5× slower; it supports bf16, so nothing changes), student ≈ 20–40 min ×2, evaluation + export minutes → **about 2–4 h ≈ $1–3**, plus download time.
* Stop the instance when `ALL DONE` prints and results are pushed.

```bash
git clone https://github.com/Namitjain07/model.git && cd model          # this repo contains the code
export GITHUB_TOKEN=<your fine-grained PAT, Contents: read/write on Namitjain07/model>   # stays on the box, never paste it in chat
PUSH=1 bash gpu/run_all.sh            # protocol A (trustworthy numbers) + push results
# optional afterwards: the deployable model trained on everything (no clean test left, evaluation numbers are NOT valid for it)
bash gpu/run_all.sh --final
```
Stages can be run separately: `STAGES="preprocess train" bash gpu/run_all.sh`. A re-run skips already-preprocessed clips.

## Reading the results honestly
* Only `metrics_A*.json` (protocol A) are test numbers. `metrics_A_clean.json` additionally drops the 52 held-out clips that had near-duplicates.
* "False alarms per analysed hour" is a **window-level** rate: non-fight 5 s windows flagged ÷ hours of those windows. It is not an event rate on continuous video. The CPU pipeline's continuous-footage numbers (8 false alarms in 8.4 min of normal footage at the old threshold) are measured separately.
* Distillation: the student sees the teacher on the *same augmented clip*, so soft targets are not simply memorised labels; the no-teacher ablation tells you if it helped.

## Licences — read before any commercial use
* **VideoMAE weights `MCG-NJU/videomae-base-finetuned-kinetics` are CC-BY-NC-4.0 (non-commercial)**. The teacher and any model distilled from it inherit that doubt. `student_A_nokd` (ImageNet MobileNetV3, BSD-style weights, trained on labels only) is the licence-clean fallback.
* YOLOX-M (Apache-2.0), RTMPose (Apache-2.0), MediaPipe (Apache-2.0) are permissive. YOLO11-pose (Ultralytics) is **AGPL-3.0**.
* Dataset terms differ (RLVS: none stated; AIRTLab: research/education only; RWF-2000: research). Check them before shipping anything trained on them.

## Qualcomm QCS6490 notes
* AI Hub reference latencies on this SoC: MediaPipe pose 1.5 ms + 1.1 ms/person, YOLOX (S) 12 ms, **Video-MAE (87.7 M) ≈ 1.2 s/clip** → verifier only. The TSM-MobileNetV3 student is ≈ 2 GFLOPs/clip (8 frames) — expected tens of ms on the NPU, **not yet measured**: compile `student_tsm_mnv3_8f.onnx` with `qai_hub.submit_compile_job` + `submit_profile_job` on the QCS6490 device, quantise with a calibration set drawn from `gpu_cache/frames`, and re-check accuracy after quantisation.
