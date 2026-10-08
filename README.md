# vv — women-safety detection from pose (YOLO11-pose / MediaPipe)

Detects, from camera video, (1) a **fall / unresponsive (unconscious) person** and (2) a **fight**,
using pose keypoints + tracking + small detectors. Targeted at a Qualcomm QCS6490 board.

```
frame -> pose backend -> COCO-17 keypoints -> tracker (IDs) -> FallDetector (per person)
                                                             -> FightDetector (per pair; learned scorer) -> alerts
```
* `safety/backends.py` – `YoloPoseOnnx` (YOLO11n-pose via onnxruntime, no torch) and `MediaPipeBackend`.
* `safety/skeleton.py` / `tracker.py` / `features.py` – COCO-17 format, tracker, scale-normalised kinematics (torso lengths).
* `safety/events.py` – `FallDetector` (UPRIGHT → DOWN/FALL → UNRESPONSIVE; "lying" needs torso *and* whole-body axis
  horizontal so bending ≠ lying) and `FightDetector` (rule score, or a learned scorer).
* `safety/pairfeat.py`, `fight_model.py` – 14 pairwise window features → gradient-boosted fight classifier.
* `safety/replay.py` – cache YOLO detections per clip and replay through the pipeline (tuning in seconds).
* `safety/sim.py` – synthetic skeletons for logic tests (`tests/`); proves intent, **not** real-world accuracy.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
./scripts/download_assets.sh          # models (yolo11n-pose.onnx, MediaPipe), sample images/videos
sudo apt-get install -y libegl1 libgles2 libgl1    # MediaPipe >=1.x native deps (also on the board image)
```

## Real datasets (downloaded from public GitHub; git-ignored, do not redistribute)
| dataset | what | script |
|---|---|---|
| **AIRTLab** (Bianculli et al., Data in Brief 2020, doi:10.1016/j.dib.2020.106587) | 350 staged 1080p clips, 2 cameras (close selfie cam + 170° fisheye). Violent: kick/punch/push/slap/choke/stab/club/gunshot. **Hard negatives:** hug/high-five/handshake/jump/greet/gestures. Research/education use. | `scripts/download_airtlab.sh` |
| **GMDCSA-24** (Alam et al., 2024, MIT) | 79 fall + 81 daily-activity clips at home (incl. sleeping, exercising), time-range labels | `scripts/download_gmdcsa24.py` |
| **RWF-2000** (Cheng et al., ICPR 2020; HF mirror `A1mal/RWF-2000-Dataset`, MIT card) | 2,000 real surveillance-style 5-s clips (1,000 fight / 1,000 non-fight), official train/val split (0 source-video overlap), 640×360 @ 10 fps | `scripts/download_rwf2000.py` |
| **Surveillance Camera Fight Dataset** (Akti, Tataroglu, Ekenel, IPTA 2019, MIT) | 150 **real** fight + 150 non-fight 2-s clips cut from 44 YouTube surveillance-style videos (cafés, streets, buses, prison); 320×240–1080p, 10–30 fps; `videos.txt` maps clips to source video | `scripts/download_survfight.sh` |

Network access changed during the project: those hosts were initially blocked and later opened (Hugging Face, Zenodo, Mendeley, Drive, ...).
Still gated: NTU CCTV-Fights (request form). Roboflow Universe works through the connector (fork → version → export → download) but the
workspace has a **10,000-image hard limit**, and the one dataset tried (kick/punch/grappling, 2,047 images) turned out to be MMA/UFC broadcast frames —
the wrong domain for CCTV, so it was not used.

## Run
```bash
python scripts/cache_dets.py airt_cam1 'data/airtlab/*/cam1/*.mp4'      # YOLO once per clip (also airt_cam2, gmdcsa)
python scripts/eval_fights.py          # RULE-BASED fight score on AIRTLab (baseline)
./scripts/download_survfight.sh && python scripts/cache_dets.py surv_fight_v2 'data/survfight/fight/*.mp4' --conf 0.15 --iou 0.7   # (+ noFight, AIRTLab *_v2)
python scripts/eval_survfight.py       # cross-dataset test on real surveillance fights
python scripts/train_combined.py       # (earlier pose-only default: AIRTLab + surveillance)
python scripts/download_rwf2000.py     # 2,000 real clips; then cache_dets.py rwf_v2 'data/rwf2000/*/*/*.avi' --shard i/4 (x4)
python scripts/build_windows_all.py && python scripts/train_all.py --final   # v3 pose+flow model -> models/fight_gb_v3.joblib
python scripts/oof_curves.py           # false-alarm events per hour on non-fight footage
python scripts/build_fight_windows.py && python scripts/train_fight_clf.py      # learned fight model, grouped CV
python scripts/eval_fight_clf.py       # shortcut controls, cross-camera test, operating points
python scripts/finalize_fight_model.py # pick threshold from out-of-fold data, save models/fight_gb.joblib
python scripts/eval_falls.py ; python scripts/tune_falls.py                 # falls on GMDCSA-24
python scripts/viz_fight_clips.py violent:cam1:7 non-violent:cam1:1         # held-out model + annotated sheets
python scripts/run_video.py clip.mp4 --sheet --out outputs/annot.mp4
python -m pytest tests
```
Always **look at** `outputs/**/*.jpg` — colour = state (green UPRIGHT, orange DOWN, red UNRESPONSIVE/FIGHT).

## Current best fight model (v3: pose + optical flow, trained on three datasets) — read the deployment note

`safety/flowfeat.py` adds six cheap motion features (Farneback flow on a 128×72 frame, camera motion subtracted, normalised by body size;
~3 ms/frame). Motion alone beats pose alone on the official benchmark, and the two together are best
(`scripts/build_windows_all.py` → `scripts/train_all.py --final`; `scripts/oof_curves.py`):

**Official RWF-2000 protocol** (fit on train, test on the 400 val clips, threshold chosen only from CV inside train):

| features | AUC | accuracy | recall / false alarms |
|---|---|---|---|
| pose (14) | 0.851 | 78.5% | 73% / 16% |
| motion (6) | 0.858 | 82.0% | 80% / 16% |
| **pose + motion (default)** | **0.871** | **82.3%** | 80% / 16% |

**Leave-one-dataset-out** (the model never saw the test dataset): Surveillance-Fight AUC 0.847 (68% / 12%); RWF-2000 val AUC 0.853
(59% / 6%); AIRTLab AUC 0.747 — trained only on real CCTV it flags 82% of AIRTLab's hugs/high-fives/jumps, because real non-fight CCTV
is mostly walking. Training on all three fixes most of that but not all (OOF AUC 0.81 on AIRTLab).

**Before vs after on 400 held-out RWF-2000 clips** (validation split; 200 fights / 200 non-fights; `scripts/report_data.py`, figures:
`scripts/report_figs.py`, `make_gallery.py`, `make_fp_normal.py` → `outputs/report/*.png`):

| detector | AUC | fights caught | normal clips falsely flagged | accuracy |
|---|---|---|---|---|
| original hand-tuned rules (pose) | 0.80 | 4% (8/200) | 1% (2/200) | 51.5% |
| previous learned model (pose; never saw RWF) | 0.78 | 41% (82/200) | 8% (15/200) | 66.7% |
| new model, never saw RWF (pose + motion; trained on AIRTLab + Surveillance-Fight) | 0.85 | 59% (118/200) | 6% (12/200) | 76.5% |
| **new model, trained on RWF train (pose + motion)** | **0.87** | **80% (161/200)** | **16% (32/200)** | **82.3%** |

A random 60-clip sample through the full live pipeline (YOLO + tracking + flow + classifier) caught 21/24 fights and false-alarmed on
4/36 normal clips. Misses were fights at the frame edge / tiny people (no usable pair) and a pair that peaked just under the threshold; false
alarms were crowded scenes (bus door, busy street) where many close moving pairs look like fighting.

**Deployment note — clip-level false-alarm rates are misleading.** Out-of-fold, on RWF-2000's non-fight footage (1,000 clips × 5 s = 1.39 h):

| threshold / hold | fight-clip recall | false-alarm events per hour |
|---|---|---|
| 0.74 / 0 s | 64% | 80 |
| 0.80 / 0 s | 57% | 56 |
| 0.86 / 0 s | 44% | 31 |
| 0.92 / 0 s | 23% | 8 |
| 0.86 / 2 s | 14% | 4 |

Requiring 1–2 s of sustained score lands on the same curve as raising the threshold — persistence buys nothing extra.
Live pipeline (YOLO + on-the-fly flow, ~50–58 ms/frame on CPU) on the 8.4 min of normal-activity clips: **8 false alarms at threshold 0.64
(ordinary pedestrians walking near each other, scores 0.64–0.70), 1 at 0.75** — so the shipped default is **0.75**
(`models/fight_gb_v3.joblib`; the clip-level 0.64 is kept in the file as `thr_clip_level`). This is a good *ranking/triage* signal (flag
segments for review, trigger recording) but **not yet an unattended alarm**: ~5 false alarms/hour costs most of the recall. Next steps
that should move this: a learned video model (fine-tune on RWF-2000 — where a GPU helps) and threshold calibration on the real camera.
`learned_pipeline(backend, thr=..., hold=...)` sets the operating point at run time.

## Earlier results (pose-only model; kept for history)
*Fight (AIRTLab, 350 clips; performance-grouped 5-fold CV so cam1/cam2 of one performance never straddle train/test):*

| detector | clip AUC | recall / false alarms |
|---|---|---|
| hand-tuned rule score (original) | **0.55 (chance)** | 34% / 23% |
| learned classifier (12 features) | 0.90 | 81% / 15% at chosen operating point |

Controls: clip length alone AUC 0.53; mean pair distance alone 0.79 (staging confound); motion features *without* any
proximity feature still 0.88; train cam1 → test cam2 (new performances, new camera) 0.89. Weakest: slaps (44–53% recall);
jumping and hugging are the hardest negatives. Two depth-cue features (apparent-size ratio, feet-height gap) were added for
elevated cameras; they made **no measurable AUC difference** (0.896 vs 0.902) — kept as principled, not as proven gain.

**Far-apart false alarms (found by visual inspection of held-out clips).** The learned scorer sometimes fires on two people
~2.5 m apart with zero strikes/contact when one moves energetically (jumping/gesturing) — it partly learned the staging
shortcut "high motion at moderate distance ⇒ violent". Restricting to pairs within reach trades recall for fewer of these
(out-of-fold, ≤15% false alarms; `scripts/eval_distance_gate.py`, `finalize_fight_model.py --dmax`):

| max pair distance (torso lengths) | 2.5 | 3.0 | 3.5 | 4.0 | 5.0 | 6.0 (default) |
|---|---|---|---|---|---|---|
| recall | 40% | 51% | 59% | 65% | 74% | 81% |
| hug false alarms | 3% | 9% | 6% | 12% | 19% | 16% |

Choose the gate for your deployment (a shop-counter camera ≠ an open plaza); the default is the ungated, higher-recall one.

**Real surveillance fights (Surveillance Camera Fight Dataset, 300 clips) — the realistic estimate.** The AIRTLab figures above
are for staged footage; real CCTV-style fights are much harder. All numbers clip-level, out-of-fold, grouped by source video
(`scripts/eval_survfight.py`, `scripts/train_combined.py`):

| model | real-surveillance clip AUC | recall / false alarms |
|---|---|---|
| hand-tuned rules (original) | 0.78 (ranks fights, but its alarm almost never fires: 5% recall) | 5% / 0% |
| AIRTLab-trained, zero-shot on these clips | 0.75 | 47% / 16% at its own threshold |
| trained on these clips only (grouped CV) | 0.80 | 73% / 28% at thr 0.5 |
| **combined AIRTLab + surveillance, pairs within 4 torso lengths (default model)** | **0.78** | **54% / 13%** (thr 0.77) |

At that same threshold it gets 85% recall but **28% false alarms on the staged AIRTLab clips** — a reminder of how different
the two domains are. The default only scores pairs within 4 torso lengths (`train_combined.py --dmax`). Ungated it scored slightly
higher out-of-fold (59% / 15%) but raised **3 false FIGHT alarms in 8.4 min of normal footage** (a busy pedestrian clip: people
walking 4.5–5 torso lengths apart, no contact); the 4.0 gate gave 0, a 3.0 gate gave 1. Visual inspection of held-out clips: caught a bar brawl and a café fight; missed a street
fight ending on the ground and a wide-shot prison brawl (fighters small/on the floor → no usable pair); false-alarmed a stairwell
clip with one person crouching below another.

Why it is hard: in real fights bodies merge in clinches (one box for two people, garbage skeleton), victims on the ground are
detected only at low confidence, and many shots are wide. Relaxing the YOLO detector (confidence 0.35→0.15, NMS IoU 0.5→0.7, now the
default) raised the share of fight clips with a usable two-person view from 77% to 93% and every metric above; it did not hurt
AIRTLab. Tried and dropped (no measurable gain): depth-cue features, and "knocked-down-next-to-standing-person" features.

*Falls (GMDCSA-24, 79 fall / 81 ADL clips):* FALL-or-DOWN recall 80% (subjects 1+2: 90%, 3+4: 68%), median alarm 1.5 s after
the annotated fall start; strict `FALL` 54%. **21% of ADL clips raise DOWN — almost all are people sleeping in bed** (deliberate
lying down; needs scene context such as a bed zone). Misses are mostly body-out-of-frame, falls onto beds, or toward the camera.
The 10 s `UNRESPONSIVE` timer can't be exercised: clips are 4–12 s.

*Normal-activity false alarms:* 0 alerts on 8.4 min of walking/standing footage with the default configuration (relaxed YOLO +
learned fight model, 4.0 gate; `python scripts/eval_clips.py yolo learned`). Speed 40–60 ms/frame on a 4-vCPU x86 CPU; **not measured on the board**.
That footage is easy (no fights, little close interaction), so 0 is a floor check, not a false-alarm rate for crowded scenes.

## Dataset audit: shortcuts and leakage (`scripts/audit_datasets.py`, `scripts/eval_clean.py`)

Before spending GPU time I checked whether the datasets can be "solved" without looking at people:

| dataset | clips | AUC of resolution / aspect / fps / duration alone → label (0.5 = no shortcut) |
|---|---|---|
| RLVS | 2,000 | **0.95** |
| RWF-2000 val | 400 | **0.79** |
| Surveillance-Fight | 300 | **0.73** |
| RWF-2000 train | 1,600 | **0.71** |
| AIRTLab | 175 | 0.40 |

Fights and non-fights come from different sources with different cameras, so a model can score well on sharpness alone. Countermeasures in
the GPU pipeline: bucket-balanced sampling, random resolution degradation, per-resolution-bucket AUC at evaluation. For the pose + flow model
(`scripts/shortcut_check.py`, stratified by resolution bucket) the RWF-val AUC stays ≈ 0.855, i.e. most of its skill is not the shortcut.

Near-duplicates (perceptual hash, Hamming ≤ 6): **61 RWF-train and 11 RLVS clips duplicate held-out clips** (YouTube re-uploads), affecting
15 RWF-val and 37 Surveillance-Fight clips. Re-running the pose + flow evaluations with those training clips removed:

| test | all held-out clips | clean subset (no duplicate anywhere in train) |
|---|---|---|
| RWF-2000 val (train on RWF train) | AUC 0.874 [0.841, 0.909], 79% recall / 14% false alarms | AUC 0.869 [0.832, 0.903], 79% / 15% (n=385) |
| Surveillance-Fight (train AIRTLab + RWF train) | AUC 0.843 [0.795, 0.889], 65% / 11% | AUC 0.847 [0.796, 0.892], 68% / 12% (n=263) |
| AIRTLab (train Surveillance-Fight + RWF train) | AUC 0.755 [0.699, 0.810], 97% recall / 82% false alarms | — |

(95% bootstrap intervals over clips. Threshold chosen from grouped CV on the training data only.) **The earlier headline numbers were not
materially inflated by the leakage** — differences are well inside the intervals — but they come with the caveat that the held-out sets are
small: differences below ~0.04 AUC are noise. The AIRTLab row shows the same domain gap as before: a model trained only on real CCTV flags
hugs, high-fives and jumps.

## GPU fine-tuning pipeline (`gpu/`, see `gpu/README.md`)

The next accuracy step is a Kinetics-pretrained video transformer (VideoMAE-base) fine-tuned on person-centric 224×224 crops
(YOLOX-M boxes, as requested) and distilled into a TSM-MobileNetV3 student for the QCS6490, with RLVS added as training data (duplicates
removed, bucket-balanced). It needs a GPU, so it is packaged as one script (`bash gpu/run_all.sh`) for a rented machine; it was smoke-tested
end-to-end on CPU (manifest → preprocessing → teacher → student → evaluation → ONNX export with ONNX-Runtime parity) but **has not been
trained at scale — no accuracy claim is made for it yet**. The VideoMAE weights are CC-BY-NC-4.0 (non-commercial).

## Known limitations — read before relying on this
* AIRTLab and the fall datasets are **staged by actors in a few rooms**; RWF-2000 and Surveillance-Fight are real CCTV/phone footage but
  are YouTube compilations (selection bias, resolution shortcuts — see the audit above). No footage of real assaults on women in the
  target deployment setting has been used, so expect a domain gap.
* Fight classifier: every window inherits its clip label (weak labels); thresholds picked on the same out-of-fold data used to
  report them. Hugging/jumping/dancing still cause false alarms; bystander overlap is partly handled by depth-cue features.
* Small people (< ~60 px tall) are missed by yolo11n @ 640 px; heavily occluded pairs merge into one box.
* Sleeping/lying on a bed looks like a fall without scene context. Head-and-shoulders close-ups are deliberately unjudged.
* An UNRESPONSIVE alarm is not re-raised if the person is occluded for longer than the tracker memory (2 s).
* A rule/learned "fight" flag is a screening signal, not evidence — keep a human in the loop.
