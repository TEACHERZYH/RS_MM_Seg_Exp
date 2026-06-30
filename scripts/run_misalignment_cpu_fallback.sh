#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
LOG_DIR="$PROJECT_DIR/outputs/strengthening/misalignment_cpu"
LOG_FILE="$LOG_DIR/misalignment_cpu_fallback.log"
THREADS="${THREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

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

  mkdir -p "$out_dir"
  echo "$(date -Is) START_CPU $name threads=$THREADS batch=$BATCH_SIZE workers=$NUM_WORKERS" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    nice -n 10 "$PY" scripts/eval_misalignment_stress.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split val_split \
      --output-dir "$out_dir" \
      --shifts 0,2,4,8,16,32 \
      --directions right,down,left,up \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --no-progress >> "$LOG_FILE" 2>&1
  echo "$(date -Is) DONE_CPU $name" | tee -a "$LOG_FILE"
}

run_job \
  "vaihingen_qalf_robust" \
  "configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml" \
  "outputs/qalf_vaihingen_irrg_dsm_mobilenetv3_robust_finetune/last_model.pt" \
  "outputs/strengthening/misalignment_cpu/vaihingen_qalf_robust"

run_job \
  "vaihingen_fixed_late_robust" \
  "configs/vaihingen_fixed_late_fusion_mobilenetv3_robust_finetune.yaml" \
  "outputs/ablation_vaihingen_fixed_late_robust_mobilenetv3/last_model.pt" \
  "outputs/strengthening/misalignment_cpu/vaihingen_fixed_late_robust"

run_job \
  "potsdam_qalf_robust" \
  "configs/potsdam_rgb_dsm_mobilenetv3_robust_finetune.yaml" \
  "outputs/qalf_potsdam_rgb_dsm_mobilenetv3_robust_finetune/last_model.pt" \
  "outputs/strengthening/misalignment_cpu/potsdam_qalf_robust"

run_job \
  "potsdam_fixed_late_robust" \
  "configs/potsdam_fixed_late_fusion_mobilenetv3_robust_finetune.yaml" \
  "outputs/ablation_potsdam_fixed_late_robust_mobilenetv3/last_model.pt" \
  "outputs/strengthening/misalignment_cpu/potsdam_fixed_late_robust"

echo "$(date -Is) DONE_CPU all_misalignment_jobs" | tee -a "$LOG_FILE"
