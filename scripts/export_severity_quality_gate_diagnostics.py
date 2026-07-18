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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config, resolve_device
from minimal_claim_severity import (
    EvalCase as M2EvalCase,
    apply_m2_case,
    assert_m2_case_contract,
    build_m2_cases,
)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class EvalCase:
    scenario: str
    severity: int
    corruption: str
    trial: int
    missing_aux: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export severity-resolved QALF quality/gate diagnostics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val_split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--severities", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--corruptions", type=str, default="noise,blur,mask,lowres")
    parser.add_argument("--include-combined", action="store_true")
    parser.add_argument("--protocol", choices=("legacy", "m2"), default="legacy")
    parser.add_argument("--dataset-key", choices=("vaihingen", "potsdam"), default="")
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


def apply_case_inputs(
    image: torch.Tensor,
    aux: torch.Tensor,
    sample_ids: list[str],
    case: EvalCase,
    protocol: str,
    dataset_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if protocol == "m2":
        return apply_m2_case(
            image,
            aux,
            sample_ids,
            dataset_key,
            M2EvalCase(case.scenario, case.severity, case.corruption, case.trial, case.missing_aux),
        )
    return apply_corruption(
        image,
        aux,
        severity=case.severity,
        corruption=case.corruption,
        seed=20_000 + case.severity * 100 + case.trial,
    )


def build_cases(args: argparse.Namespace) -> list[EvalCase]:
    if args.protocol == "m2":
        if not args.dataset_key or not args.include_combined:
            raise RuntimeError("M2 severity diagnostics require --dataset-key and --include-combined")
        cases = [EvalCase(case.scenario, case.severity, case.corruption, case.trial, case.missing_aux) for case in build_m2_cases()]
        assert_m2_case_contract([M2EvalCase(case.scenario, case.severity, case.corruption, case.trial, case.missing_aux) for case in cases])
        return cases
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


@torch.no_grad()
def collect_case_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    case: EvalCase,
    protocol: str = "legacy",
    dataset_key: str = "",
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for batch in tqdm(loader, desc=f"{case.scenario}-s{case.severity}-{case.corruption}", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        if case.missing_aux:
            aux = torch.zeros_like(aux)
            aux_available = torch.zeros_like(aux_available)

        image, aux = apply_case_inputs(
            image, aux, [str(item) for item in batch["sample_id"]], case, protocol, dataset_key
        )

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
            gate_main_max_abs = gates[:, 0].abs().amax(dim=(1, 2)).flatten()
            gate_aux_max_abs = gates[:, 1].abs().amax(dim=(1, 2)).flatten()
            for item_idx in range(batch_size):
                rows.append(
                    {
                        "scenario": case.scenario,
                        "severity": case.severity,
                        "corruption": case.corruption,
                        "trial": case.trial,
                        "sample_id": str(batch["sample_id"][item_idx]),
                        "scale": scale_idx,
                        "main_available": float(main_available[item_idx].detach().cpu()),
                        "aux_available": float(aux_available[item_idx].detach().cpu()),
                        "quality_main": float(q_main[item_idx].cpu()),
                        "quality_aux": float(q_aux[item_idx].cpu()),
                        "gate_main": float(gate_main[item_idx].cpu()),
                        "gate_aux": float(gate_aux[item_idx].cpu()),
                        "gate_main_max_abs": float(gate_main_max_abs[item_idx].cpu()),
                        "gate_aux_max_abs": float(gate_aux_max_abs[item_idx].cpu()),
                    }
                )
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "severity_quality_gate_long.csv"
    fieldnames = [
        "scenario",
        "severity",
        "corruption",
        "trial",
        "sample_id",
        "scale",
        "main_available",
        "aux_available",
        "quality_main",
        "quality_aux",
        "gate_main",
        "gate_aux",
        "gate_main_max_abs",
        "gate_aux_max_abs",
    ]
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario"]), int(row["severity"]), int(row["scale"])), []).append(row)

    summary_rows: list[dict[str, str]] = []
    for (scenario, severity, scale), group in sorted(groups.items()):
        summary_rows.append(
            {
                "scenario": scenario,
                "severity": str(severity),
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
                "gate_main_max_abs": f"{np.max([r['gate_main_max_abs'] for r in group]):.9f}",
                "gate_aux_max_abs": f"{np.max([r['gate_aux_max_abs'] for r in group]):.9f}",
            }
        )

    summary_csv = output_dir / "severity_quality_gate_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "severity",
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
                "gate_main_max_abs",
                "gate_aux_max_abs",
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

    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["eval"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    all_rows: list[dict] = []
    for case in build_cases(args):
        all_rows.extend(collect_case_rows(model, loader, device, case, args.protocol, args.dataset_key))
    output_dir = Path(args.output_dir)
    write_outputs(all_rows, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "severity_quality_gate_metadata.json").write_text(
        json.dumps(
            {
                "schema": "qalf-severity-quality-gate-metadata-v1",
                "arguments": vars(args),
                "config_sha256": sha256_file(Path(args.config)),
                "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
                "rows": len(all_rows),
                "protocol": args.protocol,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
