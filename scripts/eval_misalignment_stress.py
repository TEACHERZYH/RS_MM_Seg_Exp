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
    per_class_iou,
    resolve_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ShiftCase:
    shift_px: int
    direction: str
    dy: int
    dx: int
    missing_aux: bool

    @property
    def scenario(self) -> str:
        return "missing_aux" if self.missing_aux else "dsm_shift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optical-DSM misalignment stress")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val_split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shifts", type=str, default="0,2,4,8,16,32")
    parser.add_argument("--directions", type=str, default="right,down,left,up")
    parser.add_argument("--include-missing-aux", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional smoke-test limit; 0 uses the full split.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_split_file(config: dict, split_key: str) -> str:
    ds = config["dataset"]
    aliases = {
        "train": "train_split",
        "val": "val_split",
        "test": "test_split",
    }
    key = aliases.get(split_key, split_key)
    if key not in ds:
        raise KeyError(f"Unknown split key '{split_key}'. Available dataset keys: {sorted(ds)}")
    return str(Path(ds["split_dir"]) / ds[key])


def build_dataset(config: dict, split_key: str) -> ISPRSMultimodalDataset:
    ds = config["dataset"]
    return ISPRSMultimodalDataset(
        root_dir=ds["root_dir"],
        split_file=resolve_split_file(config, split_key),
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


def zero_padded_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return x

    shifted = torch.zeros_like(x)
    _, _, h, w = x.shape

    src_y0 = max(-dy, 0)
    src_y1 = min(h - dy, h)
    dst_y0 = max(dy, 0)
    dst_y1 = min(h + dy, h)

    src_x0 = max(-dx, 0)
    src_x1 = min(w - dx, w)
    dst_x0 = max(dx, 0)
    dst_x1 = min(w + dx, w)

    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return shifted

    shifted[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
    return shifted


def direction_to_offset(direction: str, shift_px: int) -> tuple[int, int]:
    if shift_px == 0:
        return 0, 0

    offsets = {
        "right": (0, shift_px),
        "down": (shift_px, 0),
        "left": (0, -shift_px),
        "up": (-shift_px, 0),
        "down_right": (shift_px, shift_px),
        "down_left": (shift_px, -shift_px),
        "up_right": (-shift_px, shift_px),
        "up_left": (-shift_px, -shift_px),
    }
    if direction not in offsets:
        raise ValueError(f"Unsupported direction: {direction}")
    return offsets[direction]


def build_cases(args: argparse.Namespace) -> list[ShiftCase]:
    shifts = [int(item) for item in args.shifts.split(",") if item.strip()]
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    cases: list[ShiftCase] = []
    for shift_px in shifts:
        if shift_px == 0:
            cases.append(ShiftCase(0, "none", 0, 0, False))
            continue
        for direction in directions:
            dy, dx = direction_to_offset(direction, shift_px)
            cases.append(ShiftCase(shift_px, direction, dy, dx, False))
    if args.include_missing_aux:
        cases.append(ShiftCase(-1, "missing_aux", 0, 0, True))
    return cases


@torch.no_grad()
def evaluate_case(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    case: ShiftCase,
    no_progress: bool,
) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    desc = f"{case.scenario}-{case.shift_px}-{case.direction}"
    for batch in tqdm(loader, desc=desc, leave=False, disable=no_progress):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        if case.missing_aux:
            aux = torch.zeros_like(aux)
            aux_available = torch.zeros_like(aux_available)
        else:
            aux = zero_padded_shift(aux, case.dy, case.dx)

        out = model(image, aux, aux_available, main_available)
        pred = out["logits"].argmax(dim=1).cpu().numpy()
        tgt = mask.cpu().numpy()
        confusion += confusion_matrix_from_predictions(pred, tgt, num_classes)

    ious = per_class_iou(confusion)
    return {
        "scenario": case.scenario,
        "shift_px": case.shift_px,
        "direction": case.direction,
        "dy": case.dy,
        "dx": case.dx,
        "miou": float(np.mean(ious)),
        "oa": float(overall_accuracy(confusion)),
        "per_class_iou": ious.tolist(),
        "confusion": confusion.tolist(),
    }


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "misalignment_long.csv"
    long_fields = ["scenario", "shift_px", "direction", "dy", "dx", "miou", "oa"]
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in long_fields})

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], int(row["shift_px"])), []).append(row)

    summary_rows: list[dict[str, str]] = []
    for (scenario, shift_px), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        miou = np.array([row["miou"] for row in group], dtype=np.float64)
        oa = np.array([row["oa"] for row in group], dtype=np.float64)
        summary_rows.append(
            {
                "scenario": scenario,
                "shift_px": str(shift_px),
                "miou_mean": f"{miou.mean():.6f}",
                "miou_std": f"{miou.std():.6f}",
                "oa_mean": f"{oa.mean():.6f}",
                "oa_std": f"{oa.std():.6f}",
                "n": str(len(group)),
            }
        )

    summary_csv = output_dir / "misalignment_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "shift_px", "miou_mean", "miou_std", "oa_mean", "oa_std", "n"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "misalignment_results.json"
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
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = build_dataset(config, args.split)
    if args.max_samples > 0:
        dataset.sample_ids = dataset.sample_ids[: args.max_samples]
    batch_size = args.batch_size if args.batch_size > 0 else int(config["eval"]["batch_size"])
    num_workers = args.num_workers if args.num_workers >= 0 else int(config["eval"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    torch.manual_seed(int(config["experiment"].get("seed", 1234)))
    np.random.seed(int(config["experiment"].get("seed", 1234)))

    rows = [
        evaluate_case(model, loader, num_classes, device, case, args.no_progress)
        for case in build_cases(args)
    ]
    output_dir = Path(args.output_dir)
    write_outputs(rows, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "misalignment_metadata.json").write_text(
        json.dumps(
            {
                "schema": "qalf-misalignment-metadata-v1",
                "arguments": vars(args),
                "config_sha256": sha256_file(Path(args.config)),
                "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
                "rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
