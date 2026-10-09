# HANDOFF — read this first (context for a fresh Claude Code session)

Written at the end of a long cloud-sandbox session (4 CPU cores, no GPU, HTTPS-only network). The user is moving the work to their own PC
(where Claude can SSH to a rented GPU). Everything below is what was decided, built, measured and left open. Numbers are from real runs;
anything marked **NOT MEASURED / NOT RUN** is exactly that.

## 1. Mission and user preferences
* Build a **women-safety video-analytics model** that detects **fights/assaults**, **falls**, and **unconscious (unresponsive) persons**, plus (later) a
  **"Signal for Help" hand-gesture** stage (stage 1 — *not started*).
* Deployment target: **Qualcomm QCS6490 board** (Hexagon NPU, aarch64). Development/testing happens elsewhere with real downloaded datasets.
* The user wants **visual verification** of conclusions, **honest false-positive reporting**, and **fewer false alarms**.
* Latest request: *"Find more datasets, do fine tuning and data preprocessing and make a more accurate model, reduce false alarms."*
  Follow-ups: use **YOLOX-M** wherever a YOLO-style detector is needed; use **MediaPipe or other light-weight models for pose** (reference the
  Qualcomm AI Hub for QCS6490); results of the GPU run are handed back through the GitHub repo **Namitjain07/model**.
* The user rented a **vast.ai RTX 3090 (24 GB)** (ask them for the current `ssh -p <port> root@<ip>` command; the instance is ephemeral). They said
  *"Ask me if u want a gpu"* — GPU decisions go through the user. They also said: "I only rented this GPU" when authorising SSH to it.
* Standing rules from the session: never ask the user to paste API keys/tokens in chat (Roboflow key → gitignored `.env` / `ROBOFLOW_API_KEY`; GitHub
  token → env var on the GPU box only); **do not open PRs unless asked**; do not put model identifiers in commits/artifacts; be explicit about
  unverified claims; confirm before outward-facing or hard-to-reverse actions.

## 2. Repositories
* **`Namitjain07/model` (branch `main`) = source of truth**; the cloud session pushed all code there. It also holds this file, the report figures
  (`docs/report/`) and the small trained fight model (`models/fight_gb_v3.joblib`, 196 KB).
* `jnamitindia-india/vv` (branch `claude/pc-specs-gpu-stk3a5`) had the same history but **pushes return HTTP 403** (Claude GitHub App not installed
  there). Commits exist only in the dead sandbox → ignore that repo unless the user installs the app.
* Not in git (git-ignored, regenerate with scripts): `data/` (≈4.5 GB of datasets), `models/*.onnx|*.task|*.pt`, `outputs/`, `gpu_cache/`, `gpu_out/`.

## 3. What exists (all under the repo root)
```
safety/            pipeline: backends.py (YoloPoseOnnx conf .15/NMS .7, MediaPipeBackend) → tracker.py → events.py (FallDetector, FightDetector)
                   pairfeat.py (14 pair features) flowfeat.py (6 optical-flow features, FlowBuffer) fight_model.py (LearnedFightScorer)
                   crop_pose.py (YoloxPerson, RTMPoseTopDown, MediaPipeCropPose)   replay.py   overlay.py   sim.py
scripts/           download_*.{sh,py}  cache_dets.py  build_windows_all.py  train_all.py  eval_*.py  tune_falls.py  oof_curves.py  report_*.py
                   audit_datasets.py  shortcut_check.py  eval_clean.py  download_models.sh (YOLOX-M, RTMPose-m)
gpu/               the vast.ai pipeline (section 7)       tests/test_events.py (synthetic-skeleton logic tests)
docs/report/       figures + results.json/audit.json/clean_eval.json    README.md (project README with all result tables)
```
Python env used: venv with onnxruntime, opencv-python-headless, mediapipe, scikit-learn, numpy, pandas, CPU torch + transformers + torchvision (for the GPU
pipeline smoke tests). MediaPipe needs `libegl1 libgles2 libgl1` on Linux. Only 4 CPU cores were available: keep `OMP_NUM_THREADS` low when running
several jobs.

## 4. Current detectors and measured results
**Pose:** YOLO11n-pose ONNX (AGPL-3.0!) by default; validated alternative = YOLOX-M person boxes + RTMPose-m (both Apache-2.0), keypoints agree with YOLO11 within
0.10 torso lengths (median, 100 frames). **MediaPipe Pose is single-person → unsuitable for two close people** (fine for the future hand-signal stage; MediaPipe
Hand Landmarker is already downloaded by `download_assets.sh`). The downstream effect of swapping the pose branch was **NOT tested** on RWF.

**Fall / unresponsive (`FallDetector`, GMDCSA-24: 79 fall + 81 ADL clips):** FALL-or-DOWN recall **80 %**, strict FALL 54 %, median alarm 1.5 s after the annotated
start. **~21 % of ADL clips raise DOWN, nearly all people sleeping in bed** (needs scene context such as a bed zone). The 10 s UNRESPONSIVE timer cannot be exercised
by the clips (4–12 s). Head-and-shoulders close-ups are deliberately not judged (gated on visible lower body).

**Fight (v3 = 14 pose + 6 optical-flow window features → HistGradientBoosting; EMA 0.35; pair distance gate 4.0 torso lengths; default alarm threshold 0.75):**
| test | AUC | recall / false alarms |
|---|---|---|
| RWF-2000 val (official split; trained on RWF train; threshold from CV in train) | 0.871 (clean re-run 0.874; 0.869 on the 385 clips without duplicates) | 79–80 % / 14–16 % |
| Surveillance-Fight, transfer (train AIRTLab + RWF train, duplicates removed) | 0.843 (clean subset 0.847) | 65–68 % / 11–12 % |
| AIRTLab, transfer (train Surv + RWF) | 0.755 | 97 % / **82 %** — real-CCTV-trained models flag staged hugs/high-fives/jumps |
Bootstrap 95 % CIs are about ±0.05 → differences < 0.04 AUC are noise. Before/after table vs the old rule-based detector is in `README.md`.

**False alarms per hour — the honest deployment number** (out-of-fold, RWF non-fight footage, 1,000 clips × 5 s = 1.39 h): 64 % recall ⇒ ~80 false-alarm events/h;
57 % ⇒ 56/h; 44 % ⇒ 31/h; 23 % ⇒ 8/h. Persistence (1–2 s hold) lands on the same curve as a higher threshold. **Conclusion: a triage signal for a human, not an
unattended alarm.** On 8.4 min of easy normal footage the default config raised 0 alerts (a floor check only; an earlier 0.64 threshold raised 8 → default is 0.75).
Latency 40–60 ms/frame on 4 x86 vCPUs; **NOT measured on the board.**

## 5. Datasets (all public; licences matter)
| dataset | size | notes |
|---|---|---|
| AIRTLab | 350 staged 1080p clips, 2 cameras | hard negatives (hugs, high-fives, jumps); research/education only |
| GMDCSA-24 | 79 fall / 81 ADL | MIT; falls |
| RWF-2000 | 2,000 real 5 s clips, official train/val | HF mirror `A1mal/RWF-2000-Dataset`; research |
| Surveillance Camera Fight | 150 + 150 two-second clips, 44 source videos | MIT; `videos.txt` gives the group per source video |
| RLVS | 1,000 + 1,000 YouTube clips, many 1080p, some minutes long | HF mirror `34data/real-life-violence`; **no licence stated → research only**; `scripts/download_rlvs.py` |
Still gated: NTU CCTV-Fights (request form). Roboflow Universe works via the connector (fork → `versions_generate` raw → `versions_export` → signed URL) but the
workspace has a **10,000-image hard limit**; one fork (`kick-and-punch-object-detection-d62av`, 2,047 images, MMA/UFC broadcast frames = wrong domain) is still in the user's
workspace — **ask the user whether to delete it**.

### Audit findings (scripts/audit_datasets.py, shortcut_check.py, eval_clean.py)
* **Metadata shortcut:** resolution/aspect/fps/duration alone predict the label — RLVS AUC **0.95**, RWF val 0.79, Surveillance-Fight 0.73, RWF train 0.71, AIRTLab 0.40.
  Pose+flow RWF-val AUC stratified by resolution bucket stays ≈ 0.855 (mostly not a shortcut). Countermeasures built into `gpu/`: bucket-balanced sampling, resolution-degradation
  augmentation, per-bucket AUC at evaluation.
* **Near-duplicates** (dHash ≤ 6): 61 RWF-train + 11 RLVS clips duplicate held-out clips (15 RWF-val + 37 Surveillance-Fight affected). Removed from training in `gpu/`.
  Re-running with them removed barely changed the numbers (table above). `gpu/audit.json` lists them.

## 6. Architecture decided for the board
Stage 0 YOLOX-M person boxes → Stage 1 always-on light pose + flow GBM (cheap, high recall) → Stage 2 video verifier on flagged windows (cuts false alarms).
Qualcomm AI Hub, QCS6490 (w8a8/w8a16 NPU): MediaPipe pose 1.5 ms + 1.1 ms/person, Posenet-Mobilenet 2.4 ms, HRNetPose 3.4 ms/crop (MIT), YOLOv11-Pose 21 ms (AGPL),
Yolo-X (AI Hub = S size, 9 M params) 12 ms, **Video-MAE 87.7 M ≈ 1.2 s/clip** → verifier only. Plan: distil into **TSM-MobileNetV3-Large (8 frames, ≈2 GFLOPs)**;
its on-board latency is **NOT measured** (compile + profile through `qai_hub` on the QCS6490 device, calibrate quantisation on `gpu_cache/frames`, re-check accuracy after INT8).
Caveat: YOLOX-M is detection-only (no keypoints) — pose needs a second model.

## 7. The GPU pipeline (`gpu/`, see `gpu/README.md`) — status: smoke-tested, NOT trained
`bash gpu/run_all.sh` = setup → download datasets+models → manifest → preprocess → VideoMAE teacher → student (+ no-teacher ablation) → evaluate → ONNX export; `PUSH=1` pushes
`results/<run>/` to `Namitjain07/model` (needs `GITHUB_TOKEN` on the box; git-lfs for files > 50 MB; `NO_TEACHER=1` skips the 0.5 GB teacher). `--final` adds protocol B.
* `gpu/common.py` manifest: RWF val + Surveillance-Fight + AIRTLab are **held out** (protocol A); dev = 10 % of train groups (md5 hash) → early stopping and thresholds; excludes
  audit duplicates; `bucket_weights` equalises fight/non-fight inside each (dataset, resolution bucket). 4,578 clips.
* `gpu/preprocess.py`: 5 s window → 16 frames ×2 phases → YOLOX-M person boxes (3 frames) → union crop (+12 %, min side 40 % of short side, square, **grey** letterbox) → 224² JPEG pickles.
* `gpu/train_videomae.py`: `MCG-NJU/videomae-base-finetuned-kinetics` + 2-class head, layer-wise LR decay 0.75, bf16, grad checkpointing, bs 8×accum 2, 6 epochs.
  `TEACHER_ARGS="--bs 4 --accum 4"` if out of memory.
* `gpu/student.py`: TSM-MobileNetV3, online KD from the teacher on the same augmented clip (T=2, α=0.5).  `gpu/evaluate.py`: threshold from RWF dev only (≈85 negatives → coarse),
  per-dataset AUC/recall/false alarms, `--clean` drops the 52 affected clips, resolution-stratified AUC, false alarms per *analysed* hour (window-level, not event-level).
  `gpu/export_onnx.py`: single-file ONNX + ORT parity check (verified ≈1e-7 on random weights).
* Smoke test (CPU, 40 clips, 1 epoch) proved the code runs end to end; **no accuracy claim exists for any GPU-trained model yet.** Time/cost estimates (2–4 h, $1–3 on a 3090) are guesses.
* Licence: **VideoMAE weights are CC-BY-NC-4.0 (non-commercial)** → teacher and possibly the distilled student are non-commercial; `student_A_nokd` is the licence-clean fallback.

## 8. What to do next (priority order)
1. **Get the GPU run done.** With SSH from the PC: `git clone https://github.com/Namitjain07/model.git && cd model`, set `GITHUB_TOKEN` on the box (user supplies it; never in chat/logs), run inside `tmux`:
   `PUSH=1 bash gpu/run_all.sh 2>&1 | tee run.log`. Expect to debug a few first-contact problems (preprocessing speed with 1080p RLVS videos, HF download rate limits, OOM → `TEACHER_ARGS`).
   Then pull `results/<run>/metrics_A.json` (+ `_clean`) and judge honestly: per-bucket AUC ≈ 0.5 with high overall AUC = shortcut.
2. **Combine and calibrate:** fuse pose+flow GBM (stage 1) with the video model (stage 2); choose the operating point on **false-alarm events per hour** with normal footage (not clip-level rates);
   evaluate on the held-out sets; update README/figures (`scripts/report_*.py`, `make_gallery.py`) and keep the honest limits.
3. **CPU-side accuracy:** cache YOLO detections/flow for RLVS (needs start-offset/`--max-seconds` plumbing for long videos in `cache_dets.py`, `clip_flow`, `build_windows_all.py`) and retrain the GBM with RLVS (duplicates
   excluded, bucket-balanced) — uncertain gain because RLVS is shortcut-prone; consider swapping the pose branch to YOLOX-M + RTMPose-m (needs a downstream RWF re-evaluation).
4. **Fall false alarms:** sleeping ≈ false DOWN → add a bed/lying-zone or context cue; test with continuous footage.
5. **Stage 1 hand-signal ("Signal for Help") detector** with MediaPipe Hand Landmarker (task #3, not started) + visual verification.
6. **QCS6490 deployment prep** (task #6): export, AI Hub compile/profile, INT8 calibration, latency budget, aarch64 packaging.
7. Ask the user about deleting the unused Roboflow fork.

## 9. Lessons / bugs already fixed (don't repeat)
* `pkill -f` matched the own shell → kill by explicit PID. Foreground `sleep` is blocked in this harness → use until-loops / background jobs.
* Rule-based fight detector ≈ chance on real data (AUC 0.55) → learned model; depth-cue and knocked-down features gave **no** gain (dropped).
* Clip-level false-alarm rates mislead → use events/hour with all non-fight clips as the denominator (an earlier denominator bug overstated rates ~1.7×).
* Optical-flow clip fps bug: the detection cache stores the *effective* fps → read source fps from the video.
* Head-and-shoulders false FALLs, bending ≠ lying (need torso AND whole-body axis horizontal), jitter counted as punches (3-frame sustained approach) — all fixed with regression tests.
* numpy truthiness bug in `preprocess.py` (`a or b` on arrays) and ONNX opset-conversion failure (use opset 18 + `external_data=False`) — fixed.
* Replicated-border padding produced streak artefacts in crops → grey padding.
* Roboflow: `projects_fork` fails when the workspace would exceed 10,000 images.
* Proxy-only network in the cloud sandbox: no raw SSH, no ssh client — which is why the PC session is needed to drive the GPU box.

## 10. Prompt to start the new session
> Read HANDOFF.md and README.md in this repo (Namitjain07/model). Continue the women-safety fight/fall/unconscious-detection project from section 8. First check that the environment works
> (`pip install -r requirements.txt`, `python -m pytest tests`), then SSH into my rented RTX 3090 (I'll give you the command; the GitHub token goes only in an env var on that box) and run
> `PUSH=1 bash gpu/run_all.sh` in tmux, fixing issues as they appear. Be honest about numbers: report held-out results only, per-bucket AUC, and false alarms per hour.
