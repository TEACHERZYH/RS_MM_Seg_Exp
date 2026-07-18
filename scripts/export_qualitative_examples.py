from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.models.qalf_net import QALFNet
from src.utils import (
    confusion_matrix_from_predictions,
    load_checkpoint,
    load_config,
    per_class_iou,
    resolve_device,
)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_PALETTE = np.array(
    [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
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
    parser = argparse.ArgumentParser(description="Export qualitative segmentation examples")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-ids", type=str, default="")
    parser.add_argument("--selection", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-missing-primary", action="store_true")
    return parser.parse_args()


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


def choose_indices(dataset: ISPRSMultimodalDataset, args: argparse.Namespace) -> list[int]:
    if args.sample_ids.strip():
        wanted = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
        id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset.sample_ids)}
        missing = [sample_id for sample_id in wanted if sample_id not in id_to_index]
        if missing:
            raise ValueError(f"Sample ids not found in split: {missing}")
        return [id_to_index[sample_id] for sample_id in wanted]

    count = min(args.num_samples, len(dataset))
    if args.selection == "first":
        return list(range(count))

    rng = np.random.default_rng(args.seed)
    return sorted(rng.choice(len(dataset), size=count, replace=False).tolist())


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr * STD + MEAN) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def aux_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()[0]
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(CLASS_PALETTE))
    rgb[valid] = CLASS_PALETTE[labels[valid]]
    rgb[~valid] = np.array([40, 40, 40], dtype=np.uint8)
    return rgb


def error_to_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = (target >= 0) & (target < len(CLASS_PALETTE))
    error = (pred != target) & valid
    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[valid & ~error] = np.array([235, 235, 235], dtype=np.uint8)
    rgb[error] = np.array([220, 30, 30], dtype=np.uint8)
    rgb[~valid] = np.array([50, 50, 50], dtype=np.uint8)
    return rgb


def add_title(panel: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(panel)
    title_h = 28
    canvas = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(20, 20, 20))
    return canvas


def make_montage(panels: list[tuple[str, np.ndarray]], width: int = 256) -> Image.Image:
    titled: list[Image.Image] = []
    for title, panel in panels:
        image = add_title(panel, title)
        ratio = width / image.width
        height = int(round(image.height * ratio))
        titled.append(image.resize((width, height), Image.Resampling.BILINEAR))

    montage = Image.new("RGB", (width * len(titled), max(img.height for img in titled)), (255, 255, 255))
    x = 0
    for image in titled:
        montage.paste(image, (x, 0))
        x += width
    return montage


def sample_miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    confusion = confusion_matrix_from_predictions(pred[None, ...], target[None, ...], num_classes)
    return float(np.mean(per_class_iou(confusion)))


@torch.no_grad()
def export_samples(
    model: torch.nn.Module,
    config: dict,
    args: argparse.Namespace,
    scenario: Scenario,
    scenario_idx: int,
    output_root: Path,
) -> list[dict[str, str]]:
    num_classes = int(config["dataset"]["num_classes"])
    device = next(model.parameters()).device
    dataset = build_dataset(config, args.split, scenario)
    indices = choose_indices(dataset, args)
    rows: list[dict[str, str]] = []

    for rank, index in enumerate(indices):
        np.random.seed(args.seed + scenario_idx * 1000 + rank)
        torch.manual_seed(args.seed + scenario_idx * 1000 + rank)
        batch = dataset[index]
        image = batch["image"].unsqueeze(0).to(device)
        aux = batch["aux"].unsqueeze(0).to(device)
        aux_available = batch["aux_available"].view(1).to(device)
        main_available = batch["main_available"].view(1).to(device)

        out = model(image, aux, aux_available, main_available)
        pred = out["logits"].argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        target = batch["mask"].cpu().numpy().astype(np.int64)
        miou = sample_miou(pred, target, num_classes)

        sample_id = str(batch["sample_id"])
        sample_dir = output_root / scenario.name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        panels = {
            "input_optical": denormalize_image(batch["image"]),
            "input_dsm": aux_to_rgb(batch["aux"]),
            "ground_truth": labels_to_rgb(target),
            "prediction": labels_to_rgb(pred),
            "error": error_to_rgb(pred, target),
        }
        for name, panel in panels.items():
            Image.fromarray(panel).save(sample_dir / f"{name}.png")

        montage = make_montage(
            [
                ("Optical input", panels["input_optical"]),
                ("DSM input", panels["input_dsm"]),
                ("Ground truth", panels["ground_truth"]),
                ("Prediction", panels["prediction"]),
                ("Error map", panels["error"]),
            ]
        )
        montage_path = sample_dir / "montage.png"
        montage.save(montage_path)

        quality_main = out.get("quality_main", [])
        quality_aux = out.get("quality_aux", [])
        q_main_mean = float(torch.stack([q[0].detach().cpu() for q in quality_main]).mean()) if quality_main else 0.0
        q_aux_mean = float(torch.stack([q[0].detach().cpu() for q in quality_aux]).mean()) if quality_aux else 0.0

        rows.append(
            {
                "dataset": str(config["dataset"]["name"]),
                "scenario": scenario.name,
                "sample_id": sample_id,
                "sample_miou": f"{miou:.6f}",
                "main_available": f"{float(batch['main_available']):.1f}",
                "aux_available": f"{float(batch['aux_available']):.1f}",
                "mean_quality_main": f"{q_main_mean:.6f}",
                "mean_quality_aux": f"{q_aux_mean:.6f}",
                "montage": str(montage_path),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])

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

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for scenario_idx, scenario in enumerate(scenarios):
        all_rows.extend(export_samples(model, config, args, scenario, scenario_idx, output_root))

    csv_path = output_root / "figure_candidate_index.csv"
    fieldnames = [
        "dataset",
        "scenario",
        "sample_id",
        "sample_miou",
        "main_available",
        "aux_available",
        "mean_quality_main",
        "mean_quality_aux",
        "montage",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} qualitative rows to {csv_path}")


if __name__ == "__main__":
    main()
