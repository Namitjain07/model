#!/usr/bin/env bash
# Person detector + top-down pose weights (all permissive licences; no keypoint model needs AGPL code).
#   YOLOX-M  (Megvii, Apache-2.0)          COCO person boxes, used for person-centric cropping and as the top-down detector
#   RTMPose-m (OpenMMLab, Apache-2.0)      body7 256x192 SimCC keypoints (COCO-17), ONNX
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/rtmpose
[ -s models/yolox_m.onnx ] || { curl -fsSL --retry 4 -o models/yolox_m.onnx.part https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx && mv models/yolox_m.onnx.part models/yolox_m.onnx; }
if [ -z "$(find models/rtmpose/m -name end2end.onnx 2>/dev/null)" ]; then
  curl -fsSL --retry 4 -o models/rtmpose/rtmpose-m.zip https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
  mkdir -p models/rtmpose/m && unzip -qo models/rtmpose/rtmpose-m.zip -d models/rtmpose/m
fi
ls -la models/yolox_m.onnx; find models/rtmpose -name end2end.onnx
