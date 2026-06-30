# RS_MM_Seg_Exp

Engineering repository for QALF, a quality-aware lightweight fusion framework for robust multimodal remote sensing semantic segmentation under missing and degraded modality conditions.

The code supports the ISPRS Vaihingen and Potsdam optical/DSM segmentation setting used in the accompanying manuscript. It includes model definitions, data preparation utilities, robust training/evaluation protocols, ablation configurations, efficiency measurement, qualitative export, quality/gate diagnostics, degradation severity curves, and DSM misregistration stress testing.

## Scope

- Modalities: RGB/IRRG optical imagery plus DSM.
- Datasets: ISPRS Vaihingen and ISPRS Potsdam after local dataset preparation.
- Main model: `QALFNet` with modality-specific lightweight encoders, a modality-quality estimator, global quality priors, and local dynamic fusion gates.
- Robustness protocols: full input, missing auxiliary DSM, degraded input, combined missing-DSM plus degraded input, missing-primary stress testing, and DSM misregistration stress testing.
- This repository does not redistribute ISPRS data, prepared tiles, model checkpoints, trained weights, or large output folders.

## Directory Layout

```text
RS_MM_Seg_Exp/
  configs/                 Experiment configuration files
  scripts/                 Data preparation, evaluation, diagnostics, and export scripts
  src/
    datasets/              ISPRS dataset loader and modality-state handling
    models/                QALF and baseline fusion models
    engine.py              Training and evaluation loops
    losses.py              Segmentation losses
    utils.py               Runtime helpers
  train.py                 Training entry point
  evaluate.py              Single-checkpoint evaluation entry point
  requirements.txt         Python dependencies
```

Ignored local-only directories include `data/`, `data_raw/`, `outputs/`, Python caches, logs, checkpoints, and trained weights.

## Environment

Install dependencies in an isolated Python environment:

```bash
pip install -r requirements.txt
```

The verified remote experiments used a PyTorch environment with CUDA-capable GPUs. CPU execution is supported for small checks and selected fallback evaluations, but full model training should be run on GPU.

## Dataset Preparation

Obtain the ISPRS Vaihingen and Potsdam datasets from the official ISPRS benchmark source according to its access policy. Place archives under `data_raw/`, then prepare local tiles:

```bash
python scripts/prepare_isprs.py --dataset potsdam --raw-zip data_raw/Potsdam.zip --output-root data/Potsdam_prepared
python scripts/prepare_isprs.py --dataset vaihingen --raw-zip data_raw/Vaihingen.zip --output-root data/Vaihingen_prepared
```

For password-protected shared archives, `scripts/download_isprs_all.ps1` expects the password in an environment variable and does not contain credentials:

```powershell
$env:ISPRS_SHARE_PASSWORD = "<your password>"
$env:PYTHON = "python"
$env:PROJECT_DIR = "path/to/RS_MM_Seg_Exp"
powershell -File scripts/download_isprs_all.ps1
```

## Training Examples

Clean QALF runs:

```bash
python train.py --config configs/vaihingen_irrg_dsm_mobilenetv3_clean_aug.yaml
python train.py --config configs/potsdam_rgb_dsm_mobilenetv3_finetune_aug.yaml
```

Robust QALF fine-tuning:

```bash
python train.py --config configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml
python train.py --config configs/potsdam_rgb_dsm_mobilenetv3_robust_finetune.yaml
```

Robust fixed-late fusion controls:

```bash
python train.py --config configs/vaihingen_fixed_late_fusion_mobilenetv3_robust_finetune.yaml
python train.py --config configs/potsdam_fixed_late_fusion_mobilenetv3_robust_finetune.yaml
```

## Evaluation Examples

Single-checkpoint evaluation:

```bash
python evaluate.py --config configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml --checkpoint outputs/qalf_vaihingen_irrg_dsm_mobilenetv3_robust_finetune/last_model.pt
```

Four-protocol evaluation:

```bash
python scripts/run_eval_protocol.py \
  --config configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml \
  --checkpoint outputs/qalf_vaihingen_irrg_dsm_mobilenetv3_robust_finetune/last_model.pt \
  --split val_split \
  --output-dir outputs/eval_protocol_vaihingen_qalf_robust
```

Efficiency measurement:

```bash
python scripts/measure_efficiency.py --config configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml
```

DSM misregistration stress:

```bash
python scripts/eval_misalignment_stress.py \
  --config configs/vaihingen_irrg_dsm_mobilenetv3_robust_finetune.yaml \
  --checkpoint outputs/qalf_vaihingen_irrg_dsm_mobilenetv3_robust_finetune/last_model.pt \
  --split val_split \
  --output-dir outputs/strengthening/misalignment/vaihingen_qalf_robust
```

Remote watcher scripts such as `run_misalignment_idle_watcher.sh` use configurable `PROJECT_DIR` and `PY` environment variables:

```bash
PROJECT_DIR=/path/to/RS_MM_Seg_Exp PY=/path/to/python bash scripts/run_misalignment_idle_watcher.sh
```

## Reproducibility Notes

- The manuscript distinguishes validation results, clean tile-level hold-out results, and ISPRS official test-server results. The released code does not claim official test-server performance.
- Missing primary optical inputs are implemented only as a stress check and remain outside the main robustness claim.
- Teacher-student distillation is available as an optional ablation but is not the main contribution.
- The main reported claim is bounded: robust training is the dominant empirical driver, while QALF provides reliability-aware fusion, availability-aware DSM suppression, and diagnostic fusion behavior.
