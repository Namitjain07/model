#!/usr/bin/env bash
# One command for the rented GPU box (vast.ai: 1x 24 GB GPU, >=8 vCPU, 32 GB RAM, 80 GB disk, PyTorch 2 / CUDA 12 image).
#   bash gpu/run_all.sh            # protocol A: RWF val + Surveillance-Fight + AIRTLab stay HELD OUT  (the trustworthy numbers)
#   bash gpu/run_all.sh --final    # additionally protocol B: retrain on EVERYTHING for the deployable model (no clean test set left)
#   STAGES="train student evaluate" bash gpu/run_all.sh     # run only some stages (download manifest preprocess train student evaluate export push)
set -euo pipefail
cd "$(dirname "$0")/.."
FINAL=0; [ "${1:-}" = "--final" ] && FINAL=1
STAGES="${STAGES:-setup download manifest preprocess train student evaluate export}"
[ "${PUSH:-0}" = 1 ] && STAGES="$STAGES push"
W=$(nproc); mkdir -p gpu_out/logs
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo -e "\n=== $(date +%H:%M:%S) $* ==="; }

has setup && { log setup; pip install -q -r gpu/requirements.txt; nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU VISIBLE"; }
has download && { log download
  bash scripts/download_models.sh
  python scripts/download_rwf2000.py
  python scripts/download_rlvs.py
  bash scripts/download_survfight.sh
  bash scripts/download_airtlab.sh; }
has manifest && { log manifest; python gpu/common.py | tee gpu_out/logs/manifest.txt; }
has preprocess && { log preprocess; python gpu/preprocess.py --workers "$W" 2>&1 | tee gpu_out/logs/preprocess.log; }
has train && { log "teacher (protocol A)"; python gpu/train_videomae.py --out gpu_out/teacher_A --workers "$W" 2>&1 | tee gpu_out/logs/teacher_A.log; }
has student && { log "student (protocol A)"; python gpu/student.py --teacher gpu_out/teacher_A --out gpu_out/student_A --workers "$W" 2>&1 | tee gpu_out/logs/student_A.log; }
has student && { log "student ablation: same network, plain CE, no teacher (licence-clean baseline + does distillation help?)"; python gpu/student.py --no-teacher --out gpu_out/student_A_nokd --workers "$W" 2>&1 | tee gpu_out/logs/student_A_nokd.log; }
has evaluate && { log evaluate; python gpu/evaluate.py --pred teacher=gpu_out/teacher_A/predictions.csv --pred student=gpu_out/student_A/predictions.csv --pred student_nokd=gpu_out/student_A_nokd/predictions.csv --out gpu_out/metrics_A.json | tee gpu_out/logs/evaluate_A.txt
  python gpu/evaluate.py --clean --pred teacher=gpu_out/teacher_A/predictions.csv --pred student=gpu_out/student_A/predictions.csv --pred student_nokd=gpu_out/student_A_nokd/predictions.csv --out gpu_out/metrics_A_clean.json > gpu_out/logs/evaluate_A_clean.txt; }
has export && { log export; python gpu/export_onnx.py --teacher gpu_out/teacher_A --student gpu_out/student_A --out gpu_out/onnx_A 2>&1 | tee gpu_out/logs/export_A.log; }

if [ "$FINAL" = 1 ]; then
  log "protocol B: manifest + preprocess surv/airt as training data, retrain on everything"
  python - <<'PY'
import sys; sys.path.insert(0, ".")
from gpu.common import build_manifest, summarize
print(summarize(build_manifest(final=True)))
PY
  python gpu/preprocess.py --workers "$W" 2>&1 | tee gpu_out/logs/preprocess_B.log     # cached clips are skipped
  python gpu/train_videomae.py --out gpu_out/teacher_B --final --workers "$W" 2>&1 | tee gpu_out/logs/teacher_B.log
  python gpu/student.py --teacher gpu_out/teacher_B --out gpu_out/student_B --workers "$W" 2>&1 | tee gpu_out/logs/student_B.log
  python gpu/export_onnx.py --teacher gpu_out/teacher_B --student gpu_out/student_B --out gpu_out/onnx_B 2>&1 | tee gpu_out/logs/export_B.log
fi
has push && { log push; bash gpu/push_results.sh; }
log "ALL DONE -> gpu_out/  (metrics_A.json = the numbers to trust)"
