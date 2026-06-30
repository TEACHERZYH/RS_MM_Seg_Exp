from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.models.qalf_net import QALFNet
from src.utils import (
    confusion_matrix_from_predictions,
    load_checkpoint,
    load_config,
    overall_accuracy,
    per_class_iou,
    resolve_device,
)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class EvalCase:
    scenario: str
    severity: int
    corruption: str
    trial: int
    missing_aux: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate degradation severity curves")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val_split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--severities", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--corruptions", type=str, default="noise,blur,mask,lowres")
    parser.add_argument("--include-combined", action="store_true")
    return parser.parse_args()


def build_dataset(config: dict, split_key: str) -> ISPRSMultimodalDataset:
    ds = config["dataset"]
    split_file = str(Path(ds["split_dir"]) / ds[split_key])
    return ISPRSMultimodalDataset(
        root_dir=ds["root_dir"],
        split_file=split_file,
        image_dir=ds["image_dir"],
        aux_dir=ds["aux_dir"],
        mask_dir=ds["mask_dir"],
        image_suffix=ds["image_suffix"],
        aux_suffix=ds["aux_suffix"],
        mask_suffix=ds["mask_suffix"],
        input_size=ds["input_size"],
        missing_prob=0.0,
        degradation_prob=0.0,
        normalize_aux=ds["normalize_aux"],
        training=False,
        enable_missing=False,
        enable_degradation=False,
    )


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=image.device, dtype=image.dtype)
    std = STD.to(device=image.device, dtype=image.dtype)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def normalize(image: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=image.device, dtype=image.dtype)
    std = STD.to(device=image.device, dtype=image.dtype)
    return (torch.clamp(image, 0.0, 1.0) - mean) / std


def apply_corruption(
    image: torch.Tensor,
    aux: torch.Tensor,
    severity: int,
    corruption: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if severity <= 0 or corruption == "none":
        return image, aux

    torch.manual_seed(seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(seed)

    if corruption == "noise":
        image01 = denormalize(image)
        sigma = 0.025 * severity
        image01 = image01 + torch.randn_like(image01) * sigma
        return normalize(image01), aux

    if corruption == "blur":
        kernel = 2 * severity + 1
        image_blur = F.avg_pool2d(image, kernel_size=kernel, stride=1, padding=kernel // 2)
        aux_blur = F.avg_pool2d(aux, kernel_size=kernel, stride=1, padding=kernel // 2)
        return image_blur, aux_blur

    if corruption == "lowres":
        _, _, h, w = image.shape
        factor = [1, 2, 3, 4, 6, 8][min(severity, 5)]
        small_h = max(h // factor, 4)
        small_w = max(w // factor, 4)
        image_low = F.interpolate(image, size=(small_h, small_w), mode="bilinear", align_corners=False)
        image_low = F.interpolate(image_low, size=(h, w), mode="bilinear", align_corners=False)
        return image_low, aux

    if corruption == "mask":
        rng = np.random.default_rng(seed)
        image01 = denormalize(image)
        aux_masked = aux.clone()
        _, _, h, w = image.shape
        frac = min(0.08 + 0.055 * severity, 0.40)
        box_h = max(int(h * frac), 1)
        box_w = max(int(w * frac), 1)
        for idx in range(image.shape[0]):
            y0 = int(rng.integers(0, max(h - box_h + 1, 1)))
            x0 = int(rng.integers(0, max(w - box_w + 1, 1)))
            image01[idx, :, y0 : y0 + box_h, x0 : x0 + box_w] = 0.0
            aux_masked[idx, :, y0 : y0 + box_h, x0 : x0 + box_w] = 0.0
        return normalize(image01), aux_masked

    raise ValueError(f"Unsupported corruption: {corruption}")


@torch.no_grad()
def evaluate_case(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    case: EvalCase,
) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(loader, desc=f"{case.scenario}-s{case.severity}-{case.corruption}", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        if case.missing_aux:
            aux = torch.zeros_like(aux)
            aux_available = torch.zeros_like(aux_available)

        image, aux = apply_corruption(
            image,
            aux,
            severity=case.severity,
            corruption=case.corruption,
            seed=10_000 + case.severity * 100 + case.trial,
        )

        out = model(image, aux, aux_available, main_available)
        pred = out["logits"].argmax(dim=1).cpu().numpy()
        tgt = mask.cpu().numpy()
        confusion += confusion_matrix_from_predictions(pred, tgt, num_classes)

    ious = per_class_iou(confusion)
    return {
        "scenario": case.scenario,
        "severity": case.severity,
        "corruption": case.corruption,
        "trial": case.trial,
        "miou": float(np.mean(ious)),
        "oa": float(overall_accuracy(confusion)),
        "per_class_iou": ious.tolist(),
    }


def build_cases(args: argparse.Namespace) -> list[EvalCase]:
    severities = [int(item) for item in args.severities.split(",") if item.strip()]
    corruptions = [item.strip() for item in args.corruptions.split(",") if item.strip()]
    cases: list[EvalCase] = []
    for severity in severities:
        if severity == 0:
            cases.append(EvalCase("degraded", 0, "none", 0, False))
            if args.include_combined:
                cases.append(EvalCase("missing_aux_and_degraded", 0, "none", 0, True))
            continue
        for corruption in corruptions:
            for trial in range(args.trials):
                cases.append(EvalCase("degraded", severity, corruption, trial, False))
                if args.include_combined:
                    cases.append(EvalCase("missing_aux_and_degraded", severity, corruption, trial, True))
    return cases


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "severity_curve_long.csv"
    fieldnames = ["scenario", "severity", "corruption", "trial", "miou", "oa"]
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["severity"]), []).append(row)

    summary_rows: list[dict[str, str]] = []
    for (scenario, severity), group in sorted(groups.items()):
        miou = np.array([row["miou"] for row in group], dtype=np.float64)
        oa = np.array([row["oa"] for row in group], dtype=np.float64)
        summary_rows.append(
            {
                "scenario": scenario,
                "severity": str(severity),
                "miou_mean": f"{miou.mean():.6f}",
                "miou_std": f"{miou.std():.6f}",
                "oa_mean": f"{oa.mean():.6f}",
                "oa_std": f"{oa.std():.6f}",
                "n": str(len(group)),
            }
        )

    summary_csv = output_dir / "severity_curve_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "severity", "miou_mean", "miou_std", "oa_mean", "oa_std", "n"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "severity_curve_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {json_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    num_classes = int(config["dataset"]["num_classes"])

    model = QALFNet(**config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["eval"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    rows = [evaluate_case(model, loader, num_classes, device, case) for case in build_cases(args)]
    write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
