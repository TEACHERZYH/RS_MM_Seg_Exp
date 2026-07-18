from __future__ import annotations

import argparse
import csv
import hashlib
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
    resolve_device,
    seed_data_worker,
)


CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter_background",
]


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
    parser.add_argument("--corruption-manifest", type=str, default="")
    parser.add_argument("--single-clean-trial", action="store_true")
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

    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.int64)
    false_positive = confusion.sum(axis=0).astype(np.float64) - true_positive
    false_negative = confusion.sum(axis=1).astype(np.float64) - true_positive
    union = true_positive + false_positive + false_negative
    f1_denom = 2.0 * true_positive + false_positive + false_negative
    ious: list[float | None] = []
    f1: list[float | None] = []
    for class_index in range(num_classes):
        if support[class_index] == 0:
            ious.append(None)
            f1.append(None)
        else:
            ious.append(float(true_positive[class_index] / max(union[class_index], 1.0)))
            f1.append(float(2.0 * true_positive[class_index] / max(f1_denom[class_index], 1.0)))
    miou = float(np.mean([value for value in ious if value is not None])) if all(value is not None for value in ious) else None
    oa = overall_accuracy(confusion)
    return {
        "miou_6class": miou,
        "oa": float(oa),
        "per_class_iou": ious,
        "per_class_f1": f1,
        "class_support": support.tolist(),
        "confusion": confusion.tolist(),
    }


def build_dataset(
    config: dict,
    split_key: str,
    scenario: Scenario,
    trial: int,
    corruption_manifest: str | None,
    return_pre_availability_inputs: bool = False,
) -> ISPRSMultimodalDataset:
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
        degradation_noise_std=ds.get("degradation_noise_std", 12.0),
        degradation_blur_kernel=ds.get("degradation_blur_kernel", 5),
        degradation_lowres_scale=ds.get("degradation_lowres_scale", 4),
        degradation_mask_min_fraction=ds.get("degradation_mask_min_fraction", 0.125),
        degradation_mask_max_fraction=ds.get("degradation_mask_max_fraction", 1.0 / 3.0),
        degradation_mask_position=ds.get("degradation_mask_position", "legacy_half"),
        normalize_aux=ds["normalize_aux"],
        training=False,
        enable_missing=scenario.enable_missing,
        enable_degradation=scenario.enable_degradation,
        missing_target=scenario.missing_target,
        corruption_seed=1234 + trial,
        corruption_manifest=corruption_manifest,
        corruption_scenario=scenario.name if corruption_manifest else None,
        corruption_trial=trial if corruption_manifest else None,
        return_pre_availability_inputs=return_pre_availability_inputs,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    num_classes = int(config["dataset"]["num_classes"])
    if num_classes != 6:
        raise RuntimeError(f"QALF M2 evaluation requires exactly six classes, found {num_classes}")
    corruption_manifest = Path(args.corruption_manifest) if args.corruption_manifest else None
    if corruption_manifest is not None and not corruption_manifest.exists():
        raise FileNotFoundError(corruption_manifest)

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
    classwise_rows: list[dict[str, str]] = []
    confusion_payload: dict[str, dict] = {}

    for scenario in scenarios:
        trial_metrics: list[dict] = []
        trial_ids = (
            (0,)
            if args.single_clean_trial and scenario.name in {"full", "missing_aux", "missing_primary"}
            else range(args.trials)
        )
        for trial in trial_ids:
            np.random.seed(1234 + trial)
            torch.manual_seed(1234 + trial)
            dataset = build_dataset(
                config,
                args.split,
                scenario,
                trial,
                str(corruption_manifest) if corruption_manifest is not None else None,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(config["eval"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["eval"]["num_workers"]),
                pin_memory=device.type == "cuda",
                worker_init_fn=seed_data_worker,
                generator=torch.Generator().manual_seed(2234 + trial),
            )
            metrics = eval_once(model, loader, num_classes, device)
            metrics["trial"] = trial
            trial_metrics.append(metrics)

        miou_values = [m["miou_6class"] for m in trial_metrics]
        miou_estimable = all(value is not None for value in miou_values)
        miou_mean = float(np.mean(miou_values)) if miou_estimable else None
        miou_std = float(np.std(miou_values)) if miou_estimable else None
        oa_mean = float(np.mean([m["oa"] for m in trial_metrics]))
        oa_std = float(np.std([m["oa"] for m in trial_metrics]))

        results[scenario.name] = {
            "mean": {"miou_6class": miou_mean, "oa": oa_mean},
            "std": {"miou_6class": miou_std, "oa": oa_std},
            "trials": trial_metrics,
        }
        confusion_payload[scenario.name] = {
            "schema": "qalf-m2-split-confusion-v1",
            "trials": [
                {
                    "trial": metric["trial"],
                    "matrix": metric["confusion"],
                    "class_support": metric["class_support"],
                }
                for metric in trial_metrics
            ],
        }
        csv_rows.append(
            {
                "scenario": scenario.name,
                "miou_6class": f"{miou_mean:.6f}" if miou_mean is not None else "not_estimable",
                "miou_6class_std": f"{miou_std:.6f}" if miou_std is not None else "not_estimable",
                "oa": f"{oa_mean:.6f}",
                "oa_std": f"{oa_std:.6f}",
            }
        )
        names = CLASS_NAMES if num_classes == len(CLASS_NAMES) else [f"class_{idx}" for idx in range(num_classes)]
        for class_idx, class_name in enumerate(names):
            supports = [int(metric["class_support"][class_idx]) for metric in trial_metrics]
            if len(set(supports)) != 1:
                raise RuntimeError(f"Split support drift across corruption trials: {scenario.name} class={class_idx}")
            support = supports[0]
            iou_values = [metric["per_class_iou"][class_idx] for metric in trial_metrics]
            f1_values = [metric["per_class_f1"][class_idx] for metric in trial_metrics]
            if support == 0:
                iou_mean = iou_std = f1_mean = f1_std = "not_estimable"
            else:
                if any(value is None for value in iou_values + f1_values):
                    raise RuntimeError(f"Supported class has non-estimable metric: {scenario.name} class={class_idx}")
                iou_mean = f"{float(np.mean(iou_values)):.6f}"
                iou_std = f"{float(np.std(iou_values)):.6f}"
                f1_mean = f"{float(np.mean(f1_values)):.6f}"
                f1_std = f"{float(np.std(f1_values)):.6f}"
            classwise_rows.append(
                {
                    "scenario": scenario.name,
                    "class_index": str(class_idx),
                    "class_name": class_name,
                    "support": str(support),
                    "iou_6class": iou_mean,
                    "iou_std": iou_std,
                    "f1_6class": f1_mean,
                    "f1_std": f1_std,
                    "trials": str(len(trial_metrics)),
                }
            )

    json_path = output_dir / "eval_protocol_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    confusion_path = output_dir / "eval_protocol_confusion.json"
    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "qalf-m2-eval-confusion-index-v1",
                "num_classes": num_classes,
                "class_names": CLASS_NAMES,
                "scenarios": confusion_payload,
            },
            f,
            indent=2,
            allow_nan=False,
        )

    csv_path = output_dir / "eval_protocol_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "miou_6class",
                "miou_6class_std",
                "oa",
                "oa_std",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    classwise_csv_path = output_dir / "eval_protocol_classwise.csv"
    with open(classwise_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "class_index",
                "class_name",
                "support",
                "iou_6class",
                "iou_std",
                "f1_6class",
                "f1_std",
                "trials",
            ],
        )
        writer.writeheader()
        writer.writerows(classwise_rows)

    metadata_path = output_dir / "eval_protocol_metadata.json"
    metadata = {
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "split": args.split,
        "trials": args.trials,
        "single_clean_trial": bool(args.single_clean_trial),
        "scenarios": [scenario.name for scenario in scenarios],
        "results_json_sha256": sha256_file(json_path),
        "summary_csv_sha256": sha256_file(csv_path),
        "classwise_csv_sha256": sha256_file(classwise_csv_path),
        "confusion_json_sha256": sha256_file(confusion_path),
        "corruption_manifest": str(corruption_manifest) if corruption_manifest is not None else None,
        "corruption_manifest_sha256": (
            sha256_file(corruption_manifest) if corruption_manifest is not None else None
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {confusion_path}")
    print(f"Saved: {classwise_csv_path}")
    print(f"Saved: {metadata_path}")
    for row in csv_rows:
        print(row)


if __name__ == "__main__":
    main()
