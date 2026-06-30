from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_checkpoint(config: dict, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    output_dir = Path(config["experiment"]["output_dir"])
    candidate = output_dir / "best_model.pt"
    return candidate if candidate.exists() else None


def load_model(config_path: str, checkpoint: str | None, device: torch.device) -> tuple[QALFNet, dict, Path | None]:
    config = load_config(config_path)
    model = QALFNet(**config["model"]).to(device)
    ckpt_path = resolve_checkpoint(config, checkpoint)
    if ckpt_path is not None:
        ckpt = load_checkpoint(str(ckpt_path), map_location=device.type)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt_path


def estimate_flops(model: QALFNet, image: torch.Tensor, aux: torch.Tensor, aux_available: torch.Tensor) -> int | None:
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None
    try:
        activities = [ProfilerActivity.CPU]
        if image.is_cuda:
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                model(image, aux, aux_available)
        flops = sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages())
        return int(flops) if flops > 0 else None
    except Exception:
        return None


def measure_latency(
    model: QALFNet,
    image: torch.Tensor,
    aux: torch.Tensor,
    aux_available: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(image, aux, aux_available)
        if image.is_cuda:
            torch.cuda.synchronize()
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image, aux, aux_available)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(float(start.elapsed_time(end)))
        else:
            for _ in range(iterations):
                start = time.perf_counter()
                model(image, aux, aux_available)
                times_ms.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.mean(times_ms)
    p95_ms = sorted(times_ms)[max(0, int(len(times_ms) * 0.95) - 1)]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, p95_ms, fps


def parse_item(item: str) -> tuple[str, str, str | None]:
    parts = item.split("|")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2] or None
    raise ValueError(f"Invalid --item format: {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", required=True, help="label|config|checkpoint(optional)")
    parser.add_argument("--output-dir", default="outputs/efficiency")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else None,
    }

    for item in args.item:
        label, config_path, checkpoint = parse_item(item)
        model, config, ckpt_path = load_model(config_path, checkpoint, device)
        input_size = int(config["dataset"]["input_size"])
        image = torch.randn(args.batch_size, 3, input_size, input_size, device=device)
        aux = torch.randn(args.batch_size, 1, input_size, input_size, device=device)
        aux_available = torch.ones(args.batch_size, device=device)
        total_params, trainable_params = count_params(model)
        flops = estimate_flops(model, image, aux, aux_available)
        mean_ms, p95_ms, fps = measure_latency(model, image, aux, aux_available, args.warmup, args.iterations)
        rows.append(
            {
                "label": label,
                "config": config_path,
                "checkpoint": str(ckpt_path) if ckpt_path is not None else "",
                "fusion_mode": config["model"].get("fusion_mode", "dynamic_gated"),
                "total_params_m": f"{total_params / 1e6:.4f}",
                "trainable_params_m": f"{trainable_params / 1e6:.4f}",
                "flops_g": "" if flops is None else f"{flops / 1e9:.4f}",
                "latency_mean_ms": f"{mean_ms:.4f}",
                "latency_p95_ms": f"{p95_ms:.4f}",
                "fps": f"{fps:.4f}",
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "efficiency_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "efficiency_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
