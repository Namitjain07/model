#!/usr/bin/env bash
# Publish the run to github.com/Namitjain07/model under results/<RUN>/ using YOUR credentials (never put a token in chat).
#   export GITHUB_TOKEN=<fine-grained PAT: Contents read/write on Namitjain07/model>   (or have `gh auth login` / an ssh key set up)
#   bash gpu/push_results.sh [run-name]
# Small files (metrics, predictions, logs, student ONNX, manifest) go in plain git; anything >50 MB (teacher weights/ONNX) goes through git-lfs.
set -euo pipefail
cd "$(dirname "$0")/.."; R="$(pwd)"
RUN="${1:-run-$(date +%Y%m%d-%H%M)}"; REPO=Namitjain07/model
URL="https://github.com/$REPO.git"; [ -n "${GITHUB_TOKEN:-}" ] && URL="https://x-access-token:${GITHUB_TOKEN}@github.com/$REPO.git"
W=$(mktemp -d); git clone -q --depth 1 "$URL" "$W/model"; cd "$W/model"
D="results/$RUN"; mkdir -p "$D"
for f in metrics_A.json metrics_A_clean.json logs onnx_A/student_tsm_mnv3_8f.onnx student_A/train_summary.json student_A/predictions.csv student_A_nokd/predictions.csv student_A_nokd/train_summary.json teacher_A/train_summary.json teacher_A/predictions.csv; do
  [ -e "$R/gpu_out/$f" ] && { mkdir -p "$D/$(dirname "$f")"; cp -r "$R/gpu_out/$f" "$D/$f"; }; done
cp "$R/gpu_cache/manifest.csv" "$D/manifest.csv" 2>/dev/null || true
for f in student_A/best.pt $([ "${NO_TEACHER:-0}" = 1 ] || echo onnx_A/teacher_videomae_16f.onnx teacher_A/hf/model.safetensors teacher_A/hf/config.json); do
  [ -e "$R/gpu_out/$f" ] && { mkdir -p "$D/$(dirname "$f")"; cp "$R/gpu_out/$f" "$D/$f"; }; done
if [ -d "$R/gpu_out/onnx_B" ]; then mkdir -p "$D/final_B"; cp "$R"/gpu_out/onnx_B/*.onnx "$R"/gpu_out/student_B/train_summary.json "$D/final_B/" 2>/dev/null || true; fi
if find "$D" -size +50M | grep -q .; then
  command -v git-lfs >/dev/null || { apt-get update -qq && apt-get install -y -qq git-lfs; }
  git lfs install --local >/dev/null; find "$D" -size +50M | while read -r f; do git lfs track "$f" >/dev/null; done
fi
git add -A; git -c user.name="${GIT_AUTHOR_NAME:-gpu-run}" -c user.email="${GIT_AUTHOR_EMAIL:-gpu-run@users.noreply.github.com}" commit -qm "GPU run $RUN: VideoMAE teacher + TSM-MobileNetV3 student, metrics, ONNX"
git push origin HEAD
echo "pushed -> https://github.com/$REPO/tree/HEAD/$D"
