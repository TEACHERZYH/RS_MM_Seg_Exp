from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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


@dataclass(frozen=True)
class Scenario:
    name: str
    missing_prob: float
    degradation_prob: float
    enable_missing: bool
    enable_degradation: bool
    missing_target: str = "aux"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_split")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--include-missing-primary", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def eval_once(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)
        mask = batch["mask"].to(device)

        out = model(image, aux, aux_available, main_available)
        pred = out["logits"].argmax(dim=1).cpu().numpy()
        tgt = mask.cpu().numpy()
        confusion += confusion_matrix_from_predictions(pred, tgt, num_classes)

    ious = per_class_iou(confusion)
    miou = float(np.mean(ious))
    oa = overall_accuracy(confusion)
    return {
        "miou": miou,
        "oa": float(oa),
        "per_class_iou": ious.tolist(),
        "confusion": confusion.tolist(),
    }


def build_dataset(config: dict, split_key: str, scenario: Scenario) -> ISPRSMultimodalDataset:
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
        missing_prob=scenario.missing_prob,
        degradation_prob=scenario.degradation_prob,
        normalize_aux=ds["normalize_aux"],
        training=False,
        enable_missing=scenario.enable_missing,
        enable_degradation=scenario.enable_degradation,
        missing_target=scenario.missing_target,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    num_classes = int(config["dataset"]["num_classes"])

    model = QALFNet(**config["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scenarios = [
        Scenario("full", 0.0, 0.0, False, False),
        Scenario("missing_aux", 1.0, 0.0, True, False, "aux"),
        Scenario("degraded", 0.0, 1.0, False, True),
        Scenario("missing_aux_and_degraded", 1.0, 1.0, True, True, "aux"),
    ]
    if args.include_missing_primary:
        scenarios.append(Scenario("missing_primary", 1.0, 0.0, True, False, "main"))

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    csv_rows: list[dict[str, str]] = []

    for scenario in scenarios:
        trial_metrics: list[dict] = []
        for trial in range(args.trials):
            np.random.seed(1234 + trial)
            torch.manual_seed(1234 + trial)
            dataset = build_dataset(config, args.split, scenario)
            loader = DataLoader(
                dataset,
                batch_size=int(config["eval"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["eval"]["num_workers"]),
                pin_memory=device.type == "cuda",
            )
            metrics = eval_once(model, loader, num_classes, device)
            metrics["trial"] = trial
            trial_metrics.append(metrics)

        miou_mean = float(np.mean([m["miou"] for m in trial_metrics]))
        miou_std = float(np.std([m["miou"] for m in trial_metrics]))
        oa_mean = float(np.mean([m["oa"] for m in trial_metrics]))
        oa_std = float(np.std([m["oa"] for m in trial_metrics]))

        results[scenario.name] = {
            "mean": {"miou": miou_mean, "oa": oa_mean},
            "std": {"miou": miou_std, "oa": oa_std},
            "trials": trial_metrics,
        }
        csv_rows.append(
            {
                "scenario": scenario.name,
                "miou_mean": f"{miou_mean:.6f}",
                "miou_std": f"{miou_std:.6f}",
                "oa_mean": f"{oa_mean:.6f}",
                "oa_std": f"{oa_std:.6f}",
            }
        )

    json_path = output_dir / "eval_protocol_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "eval_protocol_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scenario", "miou_mean", "miou_std", "oa_mean", "oa_std"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    for row in csv_rows:
        print(row)


if __name__ == "__main__":
    main()
