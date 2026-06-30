#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
LOG_DIR="$PROJECT_DIR/outputs/strengthening/misalignment"
LOG_FILE="$LOG_DIR/misalignment_idle_watcher.log"
IDLE_MAX_UTIL="${IDLE_MAX_UTIL:-5}"
IDLE_MIN_FREE_MB="${IDLE_MIN_FREE_MB:-12000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
  while IFS=',' read -r idx mem total util; do
    idx="$(echo "$idx" | xargs)"
    mem="$(echo "$mem" | xargs)"
    total="$(echo "$total" | xargs)"
    util="$(echo "$util" | xargs)"
    free=$((total - mem))
    if [[ "$free" -ge "$IDLE_MIN_FREE_MB" && "$util" -le "$IDLE_MAX_UTIL" ]]; then
      echo "$idx"
      return 0
    fi
  done
  return 1
}

run_job() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local out_dir="$4"
  local summary="$out_dir/misalignment_summary.csv"

  if [[ -f "$summary" ]]; then
    echo "$(date -Is) SKIP $name summary_exists" | tee -a "$LOG_FILE"
    return 0
  fi

  local gpu=""
  until gpu="$(idle_gpu)"; do
    echo "$(date -Is) WAIT idle_gpu for $name" | tee -a "$LOG_FILE"
    sleep 300
  done

  mkdir -p "$out_dir"
  echo "$(date -Is) START $name gpu=$gpu" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_misalignment_stress.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split val_split \
    --output-dir "$out_dir" \
    --shifts 0,2,4,8,16,32 \
    --directions right,down,left,up \
    --batch-size "$BATCH_SIZE" \
    --no-progress >> "$LOG_FILE" 2>&1
  echo "$(date -Is) DONE $name" | tee -a "$LOG_FILE"
}

run_job \
  "vaihingen_qalf_robust" \
  "configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml" \
  "outputs/qalf_vaihingen_irrg_dsm_mobilenetv3_robust_finetune/last_model.pt" \
  "outputs/strengthening/misalignment/vaihingen_qalf_robust"

run_job \
  "vaihingen_fixed_late_robust" \
  "configs/vaihingen_fixed_late_fusion_mobilenetv3_robust_finetune.yaml" \
  "outputs/ablation_vaihingen_fixed_late_robust_mobilenetv3/last_model.pt" \
  "outputs/strengthening/misalignment/vaihingen_fixed_late_robust"

run_job \
  "potsdam_qalf_robust" \
  "configs/potsdam_rgb_dsm_mobilenetv3_robust_finetune.yaml" \
  "outputs/qalf_potsdam_rgb_dsm_mobilenetv3_robust_finetune/last_model.pt" \
  "outputs/strengthening/misalignment/potsdam_qalf_robust"

run_job \
  "potsdam_fixed_late_robust" \
  "configs/potsdam_fixed_late_fusion_mobilenetv3_robust_finetune.yaml" \
  "outputs/ablation_potsdam_fixed_late_robust_mobilenetv3/last_model.pt" \
  "outputs/strengthening/misalignment/potsdam_fixed_late_robust"

echo "$(date -Is) DONE all_misalignment_jobs" | tee -a "$LOG_FILE"
