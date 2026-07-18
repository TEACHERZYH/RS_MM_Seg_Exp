from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
METRIC_NAMES = [
    "quality_main",
    "quality_aux",
    "beta_main",
    "beta_aux",
    "gate_main",
    "gate_aux",
]


@dataclass(frozen=True)
class EvalCase:
    scenario: str
    target: str
    corruption: str
    severity: int
    trial: int
    missing_aux: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export modality-specific QALF quality, raw local gate, and final fusion-weight diagnostics"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val_split")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--severities", default="0,1,2,3,4,5")
    parser.add_argument("--corruptions", default="noise,blur,mask,lowres")
    parser.add_argument("--targets", default="main,aux")
    parser.add_argument("--include-missing-aux", action="store_true")
    parser.add_argument("--seed-base", type=int, default=31_000)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
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


def seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def apply_mask(tensor: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    result = tensor.clone()
    rng = np.random.default_rng(seed)
    _, _, height, width = result.shape
    box_h = max(int(height * fraction), 1)
    box_w = max(int(width * fraction), 1)
    for item_idx in range(result.shape[0]):
        y0 = int(rng.integers(0, max(height - box_h + 1, 1)))
        x0 = int(rng.integers(0, max(width - box_w + 1, 1)))
        result[item_idx, :, y0 : y0 + box_h, x0 : x0 + box_w] = 0.0
    return result


def down_up(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    _, _, height, width = tensor.shape
    small_h = max(height // factor, 4)
    small_w = max(width // factor, 4)
    reduced = F.interpolate(tensor, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(reduced, size=(height, width), mode="bilinear", align_corners=False)


def apply_targeted_corruption(
    image: torch.Tensor,
    aux: torch.Tensor,
    target: str,
    severity: int,
    corruption: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if severity <= 0 or corruption == "none":
        return image, aux
    if target not in {"main", "aux"}:
        raise ValueError(f"Unsupported target modality: {target}")

    seed_torch(seed, image.device)
    sigma = 0.025 * severity
    kernel = 2 * severity + 1
    factor = [1, 2, 3, 4, 6, 8][min(severity, 5)]
    fraction = min(0.08 + 0.055 * severity, 0.40)

    if target == "main":
        image01 = denormalize(image)
        if corruption == "noise":
            image01 = torch.clamp(image01 + torch.randn_like(image01) * sigma, 0.0, 1.0)
        elif corruption == "blur":
            image01 = F.avg_pool2d(image01, kernel_size=kernel, stride=1, padding=kernel // 2)
        elif corruption == "lowres":
            image01 = down_up(image01, factor)
        elif corruption == "mask":
            image01 = apply_mask(image01, fraction, seed)
        else:
            raise ValueError(f"Unsupported corruption: {corruption}")
        return normalize(image01), aux

    aux01 = torch.clamp(aux, 0.0, 1.0)
    if corruption == "noise":
        aux01 = torch.clamp(aux01 + torch.randn_like(aux01) * sigma, 0.0, 1.0)
    elif corruption == "blur":
        aux01 = F.avg_pool2d(aux01, kernel_size=kernel, stride=1, padding=kernel // 2)
    elif corruption == "lowres":
        aux01 = down_up(aux01, factor)
    elif corruption == "mask":
        aux01 = apply_mask(aux01, fraction, seed)
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")
    return image, aux01


@torch.no_grad()
def forward_with_raw_gates(
    model: QALFNet,
    image: torch.Tensor,
    aux: torch.Tensor,
    aux_available: torch.Tensor,
    main_available: torch.Tensor,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    if model.fusion_mode != "dynamic_gated":
        raise ValueError("Raw beta diagnostics require fusion_mode=dynamic_gated")

    main_feats = model.main_encoder(image)
    aux_feats = model.aux_encoder(aux)
    fused_feats: list[torch.Tensor] = []
    quality_main_list: list[torch.Tensor] = []
    quality_aux_list: list[torch.Tensor] = []
    beta_maps: list[torch.Tensor] = []
    gate_maps: list[torch.Tensor] = []

    for idx, (main_feat, aux_feat) in enumerate(zip(main_feats, aux_feats)):
        q_main = model.quality_estimators_main[idx](main_feat, main_available)
        q_aux = model.quality_estimators_aux[idx](aux_feat, aux_available)
        fusion = model.fusions[idx]
        main_proj = fusion.main_proj(main_feat)
        aux_proj = fusion.aux_proj(aux_feat)
        gate_logits = fusion.gate_conv(torch.cat([main_proj, aux_proj], dim=1))
        beta = torch.softmax(gate_logits, dim=1)

        weighted_main = beta[:, 0:1] * q_main.view(-1, 1, 1, 1)
        weighted_aux = beta[:, 1:2] * q_aux.view(-1, 1, 1, 1)
        normalizer = weighted_main + weighted_aux + 1e-6
        gate_main = weighted_main / normalizer
        gate_aux = weighted_aux / normalizer
        gates = torch.cat([gate_main, gate_aux], dim=1)
        fused = gate_main * main_proj + gate_aux * aux_proj

        fused_feats.append(model.dropout(fused))
        quality_main_list.append(q_main)
        quality_aux_list.append(q_aux)
        beta_maps.append(beta)
        gate_maps.append(gates)

    logits = model.decoder(fused_feats)
    logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return {
        "logits": logits,
        "fused_features": fused_feats,
        "quality_main": quality_main_list,
        "quality_aux": quality_aux_list,
        "beta_maps": beta_maps,
        "gate_maps": gate_maps,
    }


def build_cases(args: argparse.Namespace) -> list[EvalCase]:
    severities = [int(item.strip()) for item in args.severities.split(",") if item.strip()]
    corruptions = [item.strip() for item in args.corruptions.split(",") if item.strip()]
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if any(target not in {"main", "aux"} for target in targets):
        raise ValueError(f"Targets must be main and/or aux, got {targets}")

    cases: list[EvalCase] = []
    for target in targets:
        for corruption in corruptions:
            for severity in severities:
                trials = [0] if severity == 0 else range(args.trials)
                for trial in trials:
                    cases.append(EvalCase("targeted_corruption", target, corruption, severity, trial))
    if args.include_missing_aux:
        cases.append(EvalCase("missing_aux", "aux", "missing", 0, 0, True))
    return cases


def case_seed(args: argparse.Namespace, case: EvalCase, batch_idx: int) -> int:
    target_offset = 0 if case.target == "main" else 1_000_000
    corruption_order = {"none": 0, "noise": 1, "blur": 2, "mask": 3, "lowres": 4, "missing": 5}
    return (
        args.seed_base
        + target_offset
        + corruption_order.get(case.corruption, 9) * 100_000
        + case.severity * 10_000
        + case.trial * 1_000
        + batch_idx
    )


@torch.no_grad()
def collect_case(
    model: QALFNet,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    case: EvalCase,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    label = f"{case.target}-{case.corruption}-s{case.severity}-t{case.trial}"
    for batch_idx, batch in enumerate(tqdm(loader, desc=label, leave=False)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        if case.missing_aux:
            aux = torch.zeros_like(aux)
            aux_available = torch.zeros_like(aux_available)
        else:
            image, aux = apply_targeted_corruption(
                image,
                aux,
                target=case.target,
                severity=case.severity,
                corruption=case.corruption,
                seed=case_seed(args, case, batch_idx),
            )

        out = forward_with_raw_gates(model, image, aux, aux_available, main_available)
        prediction = out["logits"].argmax(dim=1).cpu().numpy()
        target_mask = mask.cpu().numpy()
        confusion += confusion_matrix_from_predictions(prediction, target_mask, num_classes)

        for scale_idx, gates in enumerate(out["gate_maps"]):
            q_main = out["quality_main"][scale_idx].detach().flatten()
            q_aux = out["quality_aux"][scale_idx].detach().flatten()
            beta = out["beta_maps"][scale_idx].detach()
            final_gates = gates.detach()
            beta_main = beta[:, 0].mean(dim=(1, 2))
            beta_aux = beta[:, 1].mean(dim=(1, 2))
            gate_main = final_gates[:, 0].mean(dim=(1, 2))
            gate_aux = final_gates[:, 1].mean(dim=(1, 2))
            for item_idx in range(image.shape[0]):
                rows.append(
                    {
                        "scenario": case.scenario,
                        "target": case.target,
                        "corruption": case.corruption,
                        "severity": case.severity,
                        "trial": case.trial,
                        "sample_id": str(batch["sample_id"][item_idx]),
                        "scale": scale_idx,
                        "main_available": float(main_available[item_idx].detach().cpu()),
                        "aux_available": float(aux_available[item_idx].detach().cpu()),
                        "quality_main": float(q_main[item_idx].cpu()),
                        "quality_aux": float(q_aux[item_idx].cpu()),
                        "beta_main": float(beta_main[item_idx].cpu()),
                        "beta_aux": float(beta_aux[item_idx].cpu()),
                        "gate_main": float(gate_main[item_idx].cpu()),
                        "gate_aux": float(gate_aux[item_idx].cpu()),
                    }
                )

    ious = per_class_iou(confusion)
    metrics = {
        "scenario": case.scenario,
        "target": case.target,
        "corruption": case.corruption,
        "severity": case.severity,
        "trial": case.trial,
        "miou": float(np.mean(ious)),
        "oa": float(overall_accuracy(confusion)),
        "per_class_iou": json.dumps(ious.tolist(), separators=(",", ":")),
        "evaluated_pixels": int(confusion.sum()),
    }
    return rows, metrics


def mean_std(values: list[float]) -> tuple[str, str]:
    array = np.asarray(values, dtype=np.float64)
    return f"{array.mean():.6f}", f"{array.std():.6f}"


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 3:
        return float("nan")
    x_rank = rankdata(x_array[valid])
    y_rank = rankdata(y_array[valid])
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_diagnostics(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["scenario"],
            row["target"],
            row["corruption"],
            int(row["severity"]),
            int(row["scale"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict] = []
    for key, group in sorted(groups.items()):
        scenario, target, corruption, severity, scale = key
        item = {
            "scenario": scenario,
            "target": target,
            "corruption": corruption,
            "severity": severity,
            "scale": scale,
            "n": len(group),
        }
        for metric in METRIC_NAMES:
            mean, std = mean_std([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    return summary


def trial_diagnostic_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["scenario"],
            row["target"],
            row["corruption"],
            int(row["severity"]),
            int(row["trial"]),
            int(row["scale"]),
        )
        groups.setdefault(key, []).append(row)

    output: list[dict] = []
    for key, group in sorted(groups.items()):
        scenario, target, corruption, severity, trial, scale = key
        item = {
            "scenario": scenario,
            "target": target,
            "corruption": corruption,
            "severity": severity,
            "trial": trial,
            "scale": scale,
        }
        for metric in METRIC_NAMES:
            item[metric] = float(np.mean([float(row[metric]) for row in group]))
        output.append(item)
    return output


def build_correlations(diagnostic_rows: list[dict], metric_rows: list[dict]) -> list[dict]:
    trial_rows = trial_diagnostic_means(diagnostic_rows)
    metric_lookup = {
        (
            row["scenario"],
            row["target"],
            row["corruption"],
            int(row["severity"]),
            int(row["trial"]),
        ): float(row["miou"])
        for row in metric_rows
    }
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in trial_rows:
        if row["scenario"] != "targeted_corruption":
            continue
        lookup_key = (
            row["scenario"],
            row["target"],
            row["corruption"],
            int(row["severity"]),
            int(row["trial"]),
        )
        row = dict(row)
        row["miou"] = metric_lookup.get(lookup_key, float("nan"))
        groups.setdefault((row["target"], row["corruption"], int(row["scale"])), []).append(row)

    correlations: list[dict] = []
    for (target, corruption, scale), group in sorted(groups.items()):
        severities = [float(row["severity"]) for row in group]
        mious = [float(row["miou"]) for row in group]
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group]
            correlations.append(
                {
                    "target": target,
                    "corruption": corruption,
                    "scale": scale,
                    "metric": metric,
                    "n_points": len(group),
                    "spearman_severity": f"{spearman(severities, values):.6f}",
                    "spearman_miou": f"{spearman(values, mious):.6f}",
                }
            )
    return correlations


def write_outputs(
    diagnostic_rows: list[dict],
    metric_rows: list[dict],
    output_dir: Path,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_fields = [
        "scenario",
        "target",
        "corruption",
        "severity",
        "trial",
        "sample_id",
        "scale",
        "main_available",
        "aux_available",
        *METRIC_NAMES,
    ]
    summary_rows = summarize_diagnostics(diagnostic_rows)
    summary_fields = [
        "scenario",
        "target",
        "corruption",
        "severity",
        "scale",
        "n",
        *[f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")],
    ]
    metric_fields = [
        "scenario",
        "target",
        "corruption",
        "severity",
        "trial",
        "miou",
        "oa",
        "per_class_iou",
        "evaluated_pixels",
    ]
    correlation_rows = build_correlations(diagnostic_rows, metric_rows)
    correlation_fields = [
        "target",
        "corruption",
        "scale",
        "metric",
        "n_points",
        "spearman_severity",
        "spearman_miou",
    ]

    paths = {
        "long": output_dir / "modality_specific_diagnostics_long.csv",
        "summary": output_dir / "modality_specific_diagnostics_summary.csv",
        "case_metrics": output_dir / "modality_specific_case_metrics.csv",
        "correlations": output_dir / "modality_specific_correlations.csv",
        "metadata": output_dir / "modality_specific_diagnostics_metadata.json",
    }
    write_csv(paths["long"], diagnostic_rows, long_fields)
    write_csv(paths["summary"], summary_rows, summary_fields)
    write_csv(paths["case_metrics"], metric_rows, metric_fields)
    write_csv(paths["correlations"], correlation_rows, correlation_fields)
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"cases={len(metric_rows)} diagnostic_rows={len(diagnostic_rows)} "
        f"summary_rows={len(summary_rows)} correlation_rows={len(correlation_rows)}"
    )
    for path in paths.values():
        print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device) if args.device else resolve_device(config["train"]["device"])
    num_classes = int(config["dataset"]["num_classes"])

    model = QALFNet(**config["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if model.fusion_mode != "dynamic_gated":
        raise ValueError(f"Expected dynamic_gated QALF checkpoint, got {model.fusion_mode}")

    dataset = build_dataset(config, args.split)
    num_workers = int(config["eval"]["num_workers"]) if args.num_workers < 0 else args.num_workers
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    cases = build_cases(args)
    diagnostic_rows: list[dict] = []
    metric_rows: list[dict] = []
    for case in cases:
        case_rows, case_metrics = collect_case(model, loader, num_classes, device, case, args)
        diagnostic_rows.extend(case_rows)
        metric_rows.append(case_metrics)

    metadata = {
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "split": args.split,
        "device": str(device),
        "torch": torch.__version__,
        "cases": [asdict(case) for case in cases],
        "seed_base": args.seed_base,
        "max_batches": args.max_batches,
        "diagnostic_semantics": {
            "quality": "availability-masked global coefficient q",
            "beta": "raw local softmax response before multiplication by q",
            "gate": "final normalized fusion weight g used in the weighted sum",
        },
    }
    write_outputs(diagnostic_rows, metric_rows, Path(args.output_dir), metadata)


if __name__ == "__main__":
    main()
