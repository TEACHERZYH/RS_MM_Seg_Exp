from __future__ import annotations

import argparse
import csv
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
from src.utils import load_checkpoint, load_config, resolve_device


@dataclass(frozen=True)
class Scenario:
    name: str
    missing_prob: float
    degradation_prob: float
    enable_missing: bool
    enable_degradation: bool
    missing_target: str = "aux"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export quality and gate diagnostics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val_split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--trials", type=int, default=3)
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


@torch.no_grad()
def collect_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    scenario: Scenario,
    trial: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for batch in tqdm(loader, desc=f"{scenario.name}-{trial}", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)
        out = model(image, aux, aux_available, main_available)

        quality_main = out.get("quality_main", [])
        quality_aux = out.get("quality_aux", [])
        gate_maps = out.get("gate_maps", [])
        batch_size = image.shape[0]

        for scale_idx in range(len(gate_maps)):
            q_main = quality_main[scale_idx].detach().flatten()
            q_aux = quality_aux[scale_idx].detach().flatten()
            gates = gate_maps[scale_idx].detach()
            gate_main = gates[:, 0].mean(dim=(1, 2)).flatten()
            gate_aux = gates[:, 1].mean(dim=(1, 2)).flatten()

            for item_idx in range(batch_size):
                rows.append(
                    {
                        "scenario": scenario.name,
                        "trial": trial,
                        "sample_id": str(batch["sample_id"][item_idx]),
                        "scale": scale_idx,
                        "main_available": float(main_available[item_idx].detach().cpu()),
                        "aux_available": float(aux_available[item_idx].detach().cpu()),
                        "quality_main": float(q_main[item_idx].cpu()),
                        "quality_aux": float(q_aux[item_idx].cpu()),
                        "gate_main": float(gate_main[item_idx].cpu()),
                        "gate_aux": float(gate_aux[item_idx].cpu()),
                    }
                )
    return rows


def scenarios(include_missing_primary: bool) -> list[Scenario]:
    values = [
        Scenario("full", 0.0, 0.0, False, False),
        Scenario("missing_aux", 1.0, 0.0, True, False, "aux"),
        Scenario("degraded", 0.0, 1.0, False, True),
        Scenario("missing_aux_and_degraded", 1.0, 1.0, True, True, "aux"),
    ]
    if include_missing_primary:
        values.append(Scenario("missing_primary", 1.0, 0.0, True, False, "main"))
    return values


def write_csv(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "quality_gate_diagnostics_long.csv"
    fieldnames = [
        "scenario",
        "trial",
        "sample_id",
        "scale",
        "main_available",
        "aux_available",
        "quality_main",
        "quality_aux",
        "gate_main",
        "gate_aux",
    ]
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scenario"], int(row["scale"])), []).append(row)

    summary_rows: list[dict[str, str]] = []
    for (scenario, scale), group in sorted(groups.items()):
        summary_rows.append(
            {
                "scenario": scenario,
                "scale": str(scale),
                "n": str(len(group)),
                "quality_main_mean": f"{np.mean([r['quality_main'] for r in group]):.6f}",
                "quality_main_std": f"{np.std([r['quality_main'] for r in group]):.6f}",
                "quality_aux_mean": f"{np.mean([r['quality_aux'] for r in group]):.6f}",
                "quality_aux_std": f"{np.std([r['quality_aux'] for r in group]):.6f}",
                "gate_main_mean": f"{np.mean([r['gate_main'] for r in group]):.6f}",
                "gate_main_std": f"{np.std([r['gate_main'] for r in group]):.6f}",
                "gate_aux_mean": f"{np.mean([r['gate_aux'] for r in group]):.6f}",
                "gate_aux_std": f"{np.std([r['gate_aux'] for r in group]):.6f}",
            }
        )

    summary_csv = output_dir / "quality_gate_diagnostics_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "scale",
                "n",
                "quality_main_mean",
                "quality_main_std",
                "quality_aux_mean",
                "quality_aux_std",
                "gate_main_mean",
                "gate_main_std",
                "gate_aux_mean",
                "gate_aux_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])
    model = QALFNet(**config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_rows: list[dict] = []
    for scenario in scenarios(args.include_missing_primary):
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
            all_rows.extend(collect_rows(model, loader, device, scenario, trial))

    write_csv(all_rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
