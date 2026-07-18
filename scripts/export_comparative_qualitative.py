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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config, resolve_device


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: Path
    checkpoint: Path


@dataclass(frozen=True)
class Scenario:
    name: str
    missing_prob: float
    degradation_prob: float
    enable_missing: bool
    enable_degradation: bool
    missing_target: str = "aux"


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Model spec must be name|config|checkpoint")
    return ModelSpec(parts[0], Path(parts[1]), Path(parts[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export matched qualitative comparisons")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--model", type=parse_model_spec, action="append", required=True)
    parser.add_argument("--split", type=str, default="val_split")
    parser.add_argument("--scenario", choices=["full", "missing_aux", "degraded", "combined"], default="combined")
    parser.add_argument("--sample-ids", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cell-size", type=int, default=180)
    parser.add_argument("--selection-manifest", type=str, default="")
    parser.add_argument("--require-selection-manifest", action="store_true")
    return parser.parse_args()


def scenario_from_name(name: str) -> Scenario:
    if name == "full":
        return Scenario("full", 0.0, 0.0, False, False)
    if name == "missing_aux":
        return Scenario("missing_aux", 1.0, 0.0, True, False, "aux")
    if name == "degraded":
        return Scenario("degraded", 0.0, 1.0, False, True)
    return Scenario("missing_aux_and_degraded", 1.0, 1.0, True, True, "aux")


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


def load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    config = load_config(spec.config)
    model = QALFNet(**config["model"]).to(device)
    checkpoint = load_checkpoint(spec.checkpoint, map_location=device.type)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * STD + MEAN, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def aux_to_rgb(aux: torch.Tensor) -> np.ndarray:
    arr = aux.detach().cpu().numpy()[0]
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(CLASS_PALETTE) - 1)
    return CLASS_PALETTE[mask]


def error_to_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.full((*target.shape, 3), 245, dtype=np.uint8)
    out[pred != target] = np.array([220, 45, 45], dtype=np.uint8)
    out[pred == target] = np.array([220, 220, 220], dtype=np.uint8)
    return out


def labeled_cell(arr: np.ndarray, label: str, size: int) -> Image.Image:
    img = Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size + 26), "white")
    canvas.paste(img, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(20, 20, 20))
    return canvas


@torch.no_grad()
def predict(model: torch.nn.Module, sample: dict, device: torch.device) -> np.ndarray:
    image = sample["image"].unsqueeze(0).to(device)
    aux = sample["aux"].unsqueeze(0).to(device)
    aux_available = sample["aux_available"].view(1).to(device)
    main_available = sample.get("main_available", torch.ones_like(sample["aux_available"])).view(1).to(device)
    out = model(image, aux, aux_available, main_available)
    return out["logits"].argmax(dim=1)[0].detach().cpu().numpy()


def make_montage(
    sample: dict,
    predictions: list[tuple[str, np.ndarray]],
    output_path: Path,
    cell_size: int,
) -> None:
    target = sample["mask"].detach().cpu().numpy()
    cells = [
        labeled_cell(tensor_to_rgb(sample["image"]), "Optical input", cell_size),
        labeled_cell(aux_to_rgb(sample["aux"]), "DSM input", cell_size),
        labeled_cell(mask_to_rgb(target), "Ground truth", cell_size),
    ]
    for name, pred in predictions:
        cells.append(labeled_cell(mask_to_rgb(pred), f"{name} pred", cell_size))
        cells.append(labeled_cell(error_to_rgb(pred, target), f"{name} error", cell_size))

    gap = 8
    width = sum(cell.width for cell in cells) + gap * (len(cells) - 1)
    height = max(cell.height for cell in cells)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for cell in cells:
        canvas.paste(cell, (x, 0))
        x += cell.width + gap
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.base_config)
    device = resolve_device(base_config["train"]["device"])
    scenario = scenario_from_name(args.scenario)
    dataset = build_dataset(base_config, args.split, scenario)
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset.sample_ids)}
    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    selection_manifest_sha256 = ""
    if args.require_selection_manifest and not args.selection_manifest:
        raise RuntimeError("Formal qualitative export requires --selection-manifest")
    if args.selection_manifest:
        selection_path = Path(args.selection_manifest)
        root_name = str(base_config["dataset"]["root_dir"]).lower()
        dataset_key = "vaihingen" if "vaihingen" in root_name else "potsdam" if "potsdam" in root_name else ""
        if not dataset_key:
            raise RuntimeError("Cannot map base config to qualitative selection dataset")
        if selection_path.suffix.lower() == ".csv":
            with selection_path.open(newline="", encoding="utf-8-sig") as handle:
                selected_rows = [row for row in csv.DictReader(handle) if row.get("dataset") == dataset_key]
            selected_rows.sort(key=lambda row: int(row["rank"]))
            frozen_ids = [row["sample_id"] for row in selected_rows]
        else:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            frozen_ids = [item["sample_id"] for item in selection["datasets"][dataset_key]["selected"]]
        if sample_ids != frozen_ids:
            raise RuntimeError(f"Qualitative sample IDs differ from frozen selection: {dataset_key}")
        selection_manifest_sha256 = sha256_file(selection_path)
    missing = [sample_id for sample_id in sample_ids if sample_id not in id_to_index]
    if missing:
        raise ValueError(f"Sample ids not found in split: {missing}")

    models = [(spec.name, load_model(spec, device)) for spec in args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for sample_pos, sample_id in enumerate(sample_ids):
        np.random.seed(args.seed + sample_pos)
        torch.manual_seed(args.seed + sample_pos)
        sample = dataset[id_to_index[sample_id]]
        predictions = [(name, predict(model, sample, device)) for name, model in models]
        out_path = output_dir / f"{sample_id}_{scenario.name}_comparison.png"
        make_montage(sample, predictions, out_path, args.cell_size)
        target = sample["mask"].detach().cpu().numpy()
        for name, pred in predictions:
            rows.append(
                {
                    "sample_id": sample_id,
                    "scenario": scenario.name,
                    "model": name,
                    "pixel_accuracy": f"{float((pred == target).mean()):.6f}",
                    "montage": str(out_path),
                }
            )

    csv_path = output_dir / "comparative_qualitative_index.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "scenario", "model", "pixel_accuracy", "montage"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": "qalf-comparative-qualitative-v2",
        "base_config": args.base_config,
        "base_config_sha256": sha256_file(Path(args.base_config)),
        "split": args.split,
        "scenario": scenario.name,
        "sample_ids": sample_ids,
        "seed": args.seed,
        "cell_size": args.cell_size,
        "index_csv_sha256": sha256_file(csv_path),
        "selection_manifest": args.selection_manifest,
        "selection_manifest_sha256": selection_manifest_sha256,
        "models": [
            {
                "name": spec.name,
                "config": str(spec.config),
                "config_sha256": sha256_file(spec.config),
                "checkpoint": str(spec.checkpoint),
                "checkpoint_sha256": sha256_file(spec.checkpoint),
            }
            for spec in args.model
        ],
        "montages": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(output_dir.glob("*_comparison.png"))
        ],
    }
    (output_dir / "comparative_qualitative_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
