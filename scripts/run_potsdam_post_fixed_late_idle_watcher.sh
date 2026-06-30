#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"

cd "$PROJECT_DIR"
EVAL_CSV=outputs/ablation_potsdam_fixed_late_robust_mobilenetv3/eval_protocol_last_val/eval_protocol_summary.csv

echo "START potsdam_post_fixed_late_idle_watcher $(date -Is)"

for _ in $(seq 1 240); do
  if [[ -f "$EVAL_CSV" ]]; then
    break
  fi
  sleep 60
done

if [[ ! -f "$EVAL_CSV" ]]; then
  echo "MISSING fixed_late_eval_after_wait $(date -Is)"
  exit 2
fi

pick_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{
      gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
      if (($2 + 0) < 1000 && ($3 + 0) < 10) { print $1; exit }
    }'
}

wait_idle_gpu() {
  local gpu=""
  for _ in $(seq 1 240); do
    gpu="$(pick_idle_gpu || true)"
    if [[ -n "$gpu" ]]; then
      echo "$gpu"
      return 0
    fi
    sleep 60
  done
  return 1
}

run_on_idle_gpu() {
  local label="$1"
  shift
  local gpu
  gpu="$(wait_idle_gpu)"
  echo "RUN $label gpu=$gpu $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
  echo "DONE $label exit=$? $(date -Is)"
}

if [[ ! -f outputs/strengthening/severity/potsdam_fixed_late_robust/severity_curve_summary.csv ]]; then
  run_on_idle_gpu severity "$PY" scripts/eval_degradation_severity_curve.py \
    --config configs/potsdam_fixed_late_fusion_mobilenetv3_robust_resume.yaml \
    --checkpoint outputs/ablation_potsdam_fixed_late_robust_mobilenetv3/last_model.pt \
    --split val_split \
    --output-dir outputs/strengthening/severity/potsdam_fixed_late_robust \
    --trials 3 \
    --include-combined
else
  echo "SKIP severity_exists"
fi

if [[ ! -f outputs/strengthening/qualitative_compare/potsdam_combined/comparative_qualitative_index.csv ]]; then
  run_on_idle_gpu qualitative "$PY" scripts/export_comparative_qualitative.py \
    --base-config configs/potsdam_rgb_dsm_mobilenetv3_robust_finetune.yaml \
    --model 'FixedLateClean|configs/potsdam_fixed_late_fusion_mobilenetv3.yaml|outputs/baseline_potsdam_fixed_late_fusion_mobilenetv3/best_model.pt' \
    --model 'FixedLateRobust|configs/potsdam_fixed_late_fusion_mobilenetv3_robust_resume.yaml|outputs/ablation_potsdam_fixed_late_robust_mobilenetv3/last_model.pt' \
    --model 'QALFClean|configs/potsdam_rgb_dsm_mobilenetv3_finetune_aug.yaml|outputs/qalf_potsdam_rgb_dsm_mobilenetv3_finetune_aug/best_model.pt' \
    --model 'QALFRobust|configs/potsdam_rgb_dsm_mobilenetv3_robust_finetune.yaml|outputs/qalf_potsdam_rgb_dsm_mobilenetv3_robust_finetune/last_model.pt' \
    --split val_split \
    --scenario combined \
    --sample-ids val_7-8_0_1536 \
    --output-dir outputs/strengthening/qualitative_compare/potsdam_combined \
    --cell-size 160
else
  echo "SKIP qualitative_exists"
fi

echo "END potsdam_post_fixed_late_idle_watcher $(date -Is)"
