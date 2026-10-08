#!/usr/bin/env bash
# Fetch MediaPipe model bundles and a few real test images.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models data/images
M=https://storage.googleapis.com/mediapipe-models
A=https://storage.googleapis.com/mediapipe-assets
fetch() { [ -s "$2" ] || curl -fsSL --retry 3 -o "$2" "$1"; }

fetch $M/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task models/pose_landmarker_lite.task
fetch $M/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task models/pose_landmarker_full.task
fetch $M/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task models/hand_landmarker.task

for f in pose.jpg woman_hands.jpg thumb_up.jpg victory.jpg fist.jpg hand-woman-man.jpg man-woman-okay.jpg male_full_height_hands.jpg left_hands.jpg right_hands.jpg business-person.png; do
  fetch "$A/$f" "data/images/$f"
done
# Real people-in-motion clips (normal activity: walking/standing). Used for
# false-alarm testing and pipeline/latency checks. No fall/assault footage here.
mkdir -p data/videos
G=https://raw.githubusercontent.com
for f in people-detection one-by-one-person-detection face-demographics-walking head-pose-face-detection-female worker-zone-detection; do
  fetch "$G/intel-iot-devkit/sample-videos/master/$f.mp4" "data/videos/$f.mp4"
done
fetch "$G/opencv/opencv/4.x/samples/data/vtest.avi" data/videos/vtest.avi
echo "done"
